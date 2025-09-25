#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cooperative_groups.h>
#include "../utils/perf.h"

using namespace cooperative_groups;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Optimized softmax using shared memory and cooperative groups
// Each block handles one row, threads cooperate within the block
__global__ void softmax_optimized(float *input, float *output, int batch_size, int dim) {
    auto block = this_thread_block();
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    if (batch_idx >= batch_size) return;
    
    float *row_input = input + batch_idx * dim;
    float *row_output = output + batch_idx * dim;
    
    // Shared memory for reductions
    extern __shared__ float shared_mem[];
    float *shared_data = shared_mem;
    
    // Phase 1: Find maximum value using parallel reduction
    float thread_max = -FLT_MAX;
    for (int i = tid; i < dim; i += block_size) {
        thread_max = fmaxf(thread_max, row_input[i]);
    }
    shared_data[tid] = thread_max;
    block.sync();
    
    // Parallel reduction to find global maximum
    for (int stride = block_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + stride]);
        }
        block.sync();
    }
    float max_val = shared_data[0];
    
    // Phase 2: Compute sum of exponentials using parallel reduction
    float thread_sum = 0.0f;
    for (int i = tid; i < dim; i += block_size) {
        thread_sum += expf(row_input[i] - max_val);
    }
    shared_data[tid] = thread_sum;
    block.sync();
    
    // Parallel reduction to find global sum
    for (int stride = block_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        block.sync();
    }
    float sum_exp = shared_data[0];
    
    // Phase 3: Compute final softmax values
    for (int i = tid; i < dim; i += block_size) {
        row_output[i] = expf(row_input[i] - max_val) / sum_exp;
    }
}

// Warp-level primitives version (for modern GPUs) - Single warp only
__global__ void softmax_warp_primitives(float *input, float *output, int batch_size, int dim) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int warp_size = 32;
    
    if (batch_idx >= batch_size) return;
    
    float *row_input = input + batch_idx * dim;
    float *row_output = output + batch_idx * dim;
    
    // Phase 1: Find maximum using warp shuffle
    float thread_max = -FLT_MAX;
    for (int i = tid; i < dim; i += blockDim.x) {
        thread_max = fmaxf(thread_max, row_input[i]);
    }
    
    // Warp-level reduction for max
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }
    
    // Broadcast max to all threads in the warp
    float max_val = __shfl_sync(0xffffffff, thread_max, 0);
    
    // Phase 2: Compute sum using warp shuffle
    float thread_sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        thread_sum += expf(row_input[i] - max_val);
    }
    
    // Warp-level reduction for sum
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    // Broadcast sum to all threads in the warp
    float sum_exp = __shfl_sync(0xffffffff, thread_sum, 0);
    
    // Phase 3: Compute softmax
    for (int i = tid; i < dim; i += blockDim.x) {
        row_output[i] = expf(row_input[i] - max_val) / sum_exp;
    }
}

// Multi-warp warp-level primitives version (supports block sizes > 32)
__global__ void softmax_warp_primitives_multi(float *input, float *output, int batch_size, int dim) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_size = 32;
    int num_warps = (blockDim.x + warp_size - 1) / warp_size;
    
    if (batch_idx >= batch_size) return;
    
    float *row_input = input + batch_idx * dim;
    float *row_output = output + batch_idx * dim;
    
    // Phase 1: Find maximum using warp shuffle
    float thread_max = -FLT_MAX;
    for (int i = tid; i < dim; i += blockDim.x) {
        thread_max = fmaxf(thread_max, row_input[i]);
    }
    
    // Warp-level reduction for max (each warp independently)
    unsigned mask = __ballot_sync(0xffffffff, tid < blockDim.x);
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        thread_max = fmaxf(thread_max, __shfl_down_sync(mask, thread_max, offset));
    }
    
    // Broadcast max within each warp
    float warp_max = __shfl_sync(mask, thread_max, 0);
    
    // If multiple warps, need shared memory for inter-warp reduction
    extern __shared__ float shared_data[];
    if (num_warps > 1) {
        // Store each warp's max in shared memory
        if (lane_id == 0) {
            shared_data[warp_id] = warp_max;
        }
        __syncthreads();
        
        // First warp reduces across all warp results
        if (warp_id == 0) {
            float global_max = -FLT_MAX;
            if (lane_id < num_warps) {
                global_max = shared_data[lane_id];
            }
            
            // Reduce across warp results
            for (int offset = 16; offset > 0; offset /= 2) {
                global_max = fmaxf(global_max, __shfl_down_sync(0xffffffff, global_max, offset));
            }
            
            // Broadcast final result
            global_max = __shfl_sync(0xffffffff, global_max, 0);
            
            // Store in shared memory for all threads
            if (lane_id == 0) {
                shared_data[0] = global_max;
            }
        }
        __syncthreads();
        
        warp_max = shared_data[0];
    }
    
    // Phase 2: Compute sum using warp shuffle
    float thread_sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        thread_sum += expf(row_input[i] - warp_max);
    }
    
    // Warp-level reduction for sum
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }
    
    // Broadcast sum within each warp
    float warp_sum = __shfl_sync(mask, thread_sum, 0);
    
    // If multiple warps, reduce sum across warps
    if (num_warps > 1) {
        if (lane_id == 0) {
            shared_data[warp_id] = warp_sum;
        }
        __syncthreads();
        
        if (warp_id == 0) {
            float global_sum = 0.0f;
            if (lane_id < num_warps) {
                global_sum = shared_data[lane_id];
            }
            
            // Reduce across warp results
            for (int offset = 16; offset > 0; offset /= 2) {
                global_sum += __shfl_down_sync(0xffffffff, global_sum, offset);
            }
            
            global_sum = __shfl_sync(0xffffffff, global_sum, 0);
            
            if (lane_id == 0) {
                shared_data[0] = global_sum;
            }
        }
        __syncthreads();
        
        warp_sum = shared_data[0];
    }
    
    // Phase 3: Compute softmax
    for (int i = tid; i < dim; i += blockDim.x) {
        row_output[i] = expf(row_input[i] - warp_max) / warp_sum;
    }
}

// Single-pass fused softmax kernel (most efficient)
__global__ void softmax_fused(float *input, float *output, int batch_size, int dim) {
    auto block = this_thread_block();
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    float *row_input = input + batch_idx * dim;
    float *row_output = output + batch_idx * dim;
    
    extern __shared__ float shared_data[];
    
    // Online algorithm for numerical stable softmax
    float max_val = -FLT_MAX;
    float sum_exp = 0.0f;
    
    // First pass: compute max and sum in a numerically stable way
    for (int i = tid; i < dim; i += blockDim.x) {
        float val = row_input[i];
        float old_max = max_val;
        max_val = fmaxf(max_val, val);
        sum_exp = sum_exp * expf(old_max - max_val) + expf(val - max_val);
    }
    
    // Store thread-local values
    shared_data[tid] = max_val;
    shared_data[tid + blockDim.x] = sum_exp;
    block.sync();
    
    // Reduce max values
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            float other_max = shared_data[tid + stride];
            float other_sum = shared_data[tid + stride + blockDim.x];
            float new_max = fmaxf(max_val, other_max);
            sum_exp = sum_exp * expf(max_val - new_max) + 
                     other_sum * expf(other_max - new_max);
            max_val = new_max;
            shared_data[tid] = max_val;
            shared_data[tid + blockDim.x] = sum_exp;
        }
        block.sync();
    }
    
    float global_max = shared_data[0];
    float global_sum = shared_data[blockDim.x];
    
    // Second pass: compute final softmax values
    for (int i = tid; i < dim; i += blockDim.x) {
        row_output[i] = expf(row_input[i] - global_max) / global_sum;
    }
}

// CPU reference implementation for softmax
void softmax_cpu(const float* input, float* output, int batch_size, int dim) {
    for (int batch = 0; batch < batch_size; batch++) {
        const float* row_input = input + batch * dim;
        float* row_output = output + batch * dim;
        
        // Find maximum for numerical stability
        float max_val = -FLT_MAX;
        for (int i = 0; i < dim; i++) {
            max_val = fmaxf(max_val, row_input[i]);
        }
        
        // Compute sum of exponentials
        float sum_exp = 0.0f;
        for (int i = 0; i < dim; i++) {
            row_output[i] = expf(row_input[i] - max_val);
            sum_exp += row_output[i];
        }
        
        // Normalize
        for (int i = 0; i < dim; i++) {
            row_output[i] /= sum_exp;
        }
    }
}

int main() {
    printf("=== CUDA Softmax Optimized Implementation with Performance Analysis ===\n\n");
    
    // Get GPU information using the new utility function
    GPUInfo gpu_info = get_gpu_info();
    printf("🖥️  GPU Information:\n");
    printf("  Device: %s\n", gpu_info.name);
    printf("  Compute Capability: %d.%d\n", gpu_info.compute_capability_major, gpu_info.compute_capability_minor);
    printf("  SMs: %d\n", gpu_info.multiprocessor_count);
    printf("  Memory: %.2f GB\n", gpu_info.total_memory / (1024.0f * 1024.0f * 1024.0f));
    printf("  Peak Memory Bandwidth: %.2f GB/s\n", gpu_info.peak_memory_bandwidth_gb_s);
    printf("  Peak Compute Throughput: %.2f TFLOPS\n", gpu_info.peak_compute_throughput_tflops);
    printf("  Warp Size: %d\n", gpu_info.warp_size);
    printf("\n");
    
    // Main configuration test
    const int batch_size = 49152;
    const int dim = 1024;
    const size_t total_elements = batch_size * dim;
    const size_t bytes = total_elements * sizeof(float);
    
    printf("Configuration:\n");
    printf("  Batch size: %d\n", batch_size);
    printf("  Dimension: %d\n", dim);
    printf("  Total elements: %zu\n", total_elements);
    printf("  Memory size: %.2f MB\n\n", bytes / (1024.0f * 1024.0f));
    
    // Allocate host memory
    float *h_input = (float*)malloc(bytes);
    float *h_output_gpu = (float*)malloc(bytes);
    float *h_output_cpu = (float*)malloc(bytes);
    
    // Initialize with random data
    for (size_t i = 0; i < total_elements; i++) {
        h_input[i] = (float)(rand() % 200) / 10.0f - 10.0f;
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    
    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    
    // CPU reference using the new utility function
    auto cpu_start = std::chrono::high_resolution_clock::now();
    softmax_cpu(h_input, h_output_cpu, batch_size, dim);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    float cpu_time = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
    
    // Test different optimized implementations
    printf("Testing different optimized implementations:\n");
    
    // 1. Shared memory optimized version
    int threads_per_block = (256 < dim) ? 256 : dim;
    if (threads_per_block > 1024) threads_per_block = 1024;
    threads_per_block = 256;
    int blocks_per_grid = batch_size;
    size_t shared_mem = threads_per_block * sizeof(float);
    
    printf("Kernel configuration:\n");
    printf("  Threads per block: %d\n", threads_per_block);
    printf("  Blocks per grid: %d\n", blocks_per_grid);
    printf("  Total threads: %d\n", threads_per_block * blocks_per_grid);
    printf("  Shared memory: %zu bytes\n\n", shared_mem);
    
    // Measure GPU execution time using the new utility function
    float gpu_time = measure_kernel_ms(softmax_optimized, d_input, d_output, 
                                     blocks_per_grid, threads_per_block, shared_mem,
                                     batch_size, dim);
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output_gpu, d_output, bytes, cudaMemcpyDeviceToHost));
    
    // Verify results using the new utility function
    if (verify_results(h_output_gpu, h_output_cpu, total_elements)) {
        printf("Verification:\n");
        printf("  ✅ Results match! GPU implementation is correct.\n\n");
    } else {
        printf("Verification:\n");
        printf("  ❌ Results don't match! GPU implementation has errors.\n\n");
        return 1;
    }
    
    // Calculate and display performance metrics using the new utility function
    PerformanceMetrics metrics = calculate_metrics("Softmax Optimized", gpu_time, cpu_time, total_elements, dim,
                                                  threads_per_block, blocks_per_grid, gpu_info);
    print_performance_analysis(metrics, gpu_info);
    
    // Test other optimized kernels for comparison
    printf("\n=== Kernel Comparison ===\n");
    
    // 2. Warp primitives version (single warp)
    if (dim <= 1024) {
        int warp_threads = (32 < dim) ? 32 : dim;
        
        // Measure GPU execution time using the new utility function
        float warp_time = measure_kernel_ms(softmax_warp_primitives, d_input, d_output, 
                                          blocks_per_grid, warp_threads, 0,
                                          batch_size, dim);
        PerformanceMetrics warp_metrics = calculate_metrics("Warp Primitives (Single)", warp_time, cpu_time, total_elements, dim,
                                                          warp_threads, blocks_per_grid, gpu_info);
        print_performance_analysis(warp_metrics, gpu_info);
        
        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(h_output_gpu, d_output, bytes, cudaMemcpyDeviceToHost));
        
        // Verify results
        if (verify_results(h_output_gpu, h_output_cpu, total_elements)) {
            printf("Warp Primitives (Single): ✅ Correct, Time: %.4f ms\n", warp_time);
        } else {
            printf("Warp Primitives (Single): ❌ Incorrect results\n");
        }
    }
    
    // 3. Multi-warp warp primitives version
    if (dim > 32) {
        int multi_warp_threads = 256;  // Test with 8 warps
        size_t multi_warp_shared_mem = ((multi_warp_threads + 31) / 32) * sizeof(float);
        
        // Measure GPU execution time using the new utility function
        float multi_warp_time = measure_kernel_ms(softmax_warp_primitives_multi, d_input, d_output, 
                                                blocks_per_grid, multi_warp_threads, multi_warp_shared_mem,
                                                batch_size, dim);
        PerformanceMetrics multi_warp_metrics = calculate_metrics("Warp Primitives (Multi)", multi_warp_time, cpu_time, total_elements, dim,
                                                                multi_warp_threads, blocks_per_grid, gpu_info);
        print_performance_analysis(multi_warp_metrics, gpu_info);
        
        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(h_output_gpu, d_output, bytes, cudaMemcpyDeviceToHost));
        
        // Verify results
        if (verify_results(h_output_gpu, h_output_cpu, total_elements)) {
            printf("Warp Primitives (Multi): ✅ Correct, Time: %.4f ms\n", multi_warp_time);
        } else {
            printf("Warp Primitives (Multi): ❌ Incorrect results\n");
        }
    }
    
    // // 3. Fused version
    // size_t fused_shared_mem = threads_per_block * 2 * sizeof(float); // max + sum
    // float fused_time = measure_kernel_ms(softmax_fused, d_input, d_output, 
    //                                    blocks_per_grid, threads_per_block, fused_shared_mem,
    //                                    batch_size, dim);
    // metrics = calculate_metrics("Fused Softmax", fused_time, cpu_time, total_elements, dim,
    //                                                 threads_per_block, blocks_per_grid, gpu_info);
    // print_performance_analysis(metrics, gpu_info);                                           
    
    // // Copy result back to host
    // CUDA_CHECK(cudaMemcpy(h_output_gpu, d_output, bytes, cudaMemcpyDeviceToHost));
    
    // // Verify results
    // if (verify_results(h_output_gpu, h_output_cpu, total_elements)) {
    //     printf("Fused: ✅ Correct, Time: %.4f ms\n", fused_time);
    // } else {
    //     printf("Fused: ❌ Incorrect results\n");
    // }
    
    // Run benchmark configurations using the new utility function
    // benchmark_configurations(softmax_optimized, gpu_info);
    
    // Cleanup
    free(h_input);
    free(h_output_gpu);
    free(h_output_cpu);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    printf("\n=== Performance Analysis Complete ===\n");
    return 0;
}