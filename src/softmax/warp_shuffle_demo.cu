#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// 演示 warp shuffle 归约过程
__global__ void warp_shuffle_reduction_demo(float *input, float *output) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // 每个线程获取自己的值
    float my_value = input[tid];
    
    // Warp-level 最大值归约
    float max_val = my_value;
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, max_val, offset);
        max_val = fmaxf(max_val, other_val);
    }
    
    // 广播结果
    float global_max = __shfl_sync(0xffffffff, max_val, 0);
    
    // Warp-level 求和归约
    float sum_val = my_value;
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, sum_val, offset);
        sum_val += other_val;
    }
    float global_sum = __shfl_sync(0xffffffff, sum_val, 0);
    
    // 只有 warp 内的第一个线程保存结果
    if (lane_id == 0) {
        output[warp_id * 2] = global_max;      // 最大值
        output[warp_id * 2 + 1] = global_sum;  // 总和
    }
}

// 演示不同归约策略的性能
__global__ void performance_comparison(float *input, float *output, int method) {
    int tid = threadIdx.x;
    int block_id = blockIdx.x;
    
    float my_value = input[tid];
    
    if (method == 0) {
        // 方法 0: 使用共享内存归约
        extern __shared__ float sdata[];
        sdata[tid] = my_value;
        __syncthreads();
        
        // 归约
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            output[block_id] = sdata[0];
        }
    } else {
        // 方法 1: 使用 warp shuffle 归约
        float result = my_value;
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(0xffffffff, result, offset);
            result = fmaxf(result, other_val);
        }
        
        if (tid == 0) {
            output[block_id] = result;
        }
    }
}

int main() {
    printf("=== Warp Shuffle 演示程序 ===\n\n");
    
    const int num_elements = 64;  // 2个 warp
    const int num_bytes = num_elements * sizeof(float);
    
    // 分配内存
    float *h_input = (float*)malloc(num_bytes);
    float *h_output = (float*)malloc(2 * sizeof(float)); // 2个 warp 的结果
    float *d_input, *d_output;
    
    CUDA_CHECK(cudaMalloc(&d_input, num_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, 2 * sizeof(float)));
    
    // 初始化输入数据
    for (int i = 0; i < num_elements; i++) {
        h_input[i] = (float)(i % 10) + 1.0f;  // 1,2,3,...,10,1,2,3,...
    }
    
    printf("输入数据 (前16个元素): ");
    for (int i = 0; i < 16; i++) {
        printf("%.1f ", h_input[i]);
    }
    printf("\n\n");
    
    // 复制到 GPU
    CUDA_CHECK(cudaMemcpy(d_input, h_input, num_bytes, cudaMemcpyHostToDevice));
    
    // 运行 warp shuffle 演示
    warp_shuffle_reduction_demo<<<1, num_elements>>>(d_input, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 复制结果
    CUDA_CHECK(cudaMemcpy(h_output, d_output, 2 * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("Warp Shuffle 归约结果:\n");
    printf("Warp 0 - 最大值: %.1f, 总和: %.1f\n", h_output[0], h_output[1]);
    printf("Warp 1 - 最大值: %.1f, 总和: %.1f\n", h_output[2], h_output[3]);
    
    // 手动计算验证
    float expected_max_0 = 0.0f, expected_sum_0 = 0.0f;
    float expected_max_1 = 0.0f, expected_sum_1 = 0.0f;
    
    for (int i = 0; i < 32; i++) {
        expected_max_0 = fmaxf(expected_max_0, h_input[i]);
        expected_sum_0 += h_input[i];
    }
    for (int i = 32; i < 64; i++) {
        expected_max_1 = fmaxf(expected_max_1, h_input[i]);
        expected_sum_1 += h_input[i];
    }
    
    printf("\n验证结果:\n");
    printf("Warp 0 - 期望最大值: %.1f, 期望总和: %.1f\n", expected_max_0, expected_sum_0);
    printf("Warp 1 - 期望最大值: %.1f, 期望总和: %.1f\n", expected_max_1, expected_sum_1);
    
    printf("\n结果验证: %s\n", 
           (fabsf(h_output[0] - expected_max_0) < 1e-5f && 
            fabsf(h_output[1] - expected_sum_0) < 1e-5f &&
            fabsf(h_output[2] - expected_max_1) < 1e-5f && 
            fabsf(h_output[3] - expected_sum_1) < 1e-5f) ? "✅ 正确" : "❌ 错误");
    
    // 性能比较
    printf("\n=== 性能比较 ===\n");
    
    const int num_blocks = 1000;
    const int block_size = 32;  // 一个 warp
    const int iterations = 1000;
    
    float *d_perf_output;
    CUDA_CHECK(cudaMalloc(&d_perf_output, num_blocks * sizeof(float)));
    
    // 测试共享内存版本
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // 预热
    performance_comparison<<<num_blocks, block_size, block_size * sizeof(float)>>>(d_input, d_perf_output, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        performance_comparison<<<num_blocks, block_size, block_size * sizeof(float)>>>(d_input, d_perf_output, 0);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float shared_mem_time;
    CUDA_CHECK(cudaEventElapsedTime(&shared_mem_time, start, stop));
    shared_mem_time /= iterations;
    
    // 测试 warp shuffle 版本
    // 预热
    performance_comparison<<<num_blocks, block_size, 0>>>(d_input, d_perf_output, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        performance_comparison<<<num_blocks, block_size, 0>>>(d_input, d_perf_output, 1);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float warp_shuffle_time;
    CUDA_CHECK(cudaEventElapsedTime(&warp_shuffle_time, start, stop));
    warp_shuffle_time /= iterations;
    
    printf("共享内存版本: %.4f ms\n", shared_mem_time);
    printf("Warp Shuffle版本: %.4f ms\n", warp_shuffle_time);
    printf("加速比: %.2fx\n", shared_mem_time / warp_shuffle_time);
    
    // 清理
    free(h_input);
    free(h_output);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_perf_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return 0;
}

