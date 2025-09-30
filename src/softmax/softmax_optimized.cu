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

// Data structure for storing test results
struct TestResult {
    int dim;
    float gpu_time_ms;
    float cpu_time_ms;
    float speedup;
    float memory_bandwidth_gb_s;
    float memory_utilization_percent;
    float compute_utilization_percent;
    float sfu_utilization_percent;
    float exp_operations_per_second;
    size_t memory_size_mb;
    const char* kernel_name;
};

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
        thread_sum += __expf(row_input[i] - warp_max);
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

// Single-pass Online softmax kernel (most efficient)
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

// Print performance summary table
void print_performance_summary(const std::vector<TestResult>& results, const GPUInfo& gpu_info) {
    printf("\n========================================================================================================================\n");
    printf("PERFORMANCE SUMMARY TABLE\n");
    printf("========================================================================================================================\n");
    printf("GPU: %s | Peak Bandwidth: %.1f GB/s | Peak Compute: %.1f TFLOPS\n", 
           gpu_info.name, gpu_info.peak_memory_bandwidth_gb_s, gpu_info.peak_compute_throughput_tflops);
    printf("========================================================================================================================\n");
    
    // Header
    printf("%-35s %-8s %-12s %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n",
           "Kernel", "Dim", "GPU(ms)", "CPU(ms)", "Speedup", "MemBW(GB/s)", "MemUtil(%)", "CompUtil(%)", "SFUUtil(%)", "MemSize(MB)");
    printf("------------------------------------------------------------------------------------------------------------------------------------\n");
    
    // Data rows
    for (const auto& result : results) {
        printf("%-35s %-8d %-12.4f %-12.4f %-12.1f %-12.1f %-12.1f %-12.1f %-12.1f %-12zu\n",
               result.kernel_name, result.dim, result.gpu_time_ms, result.cpu_time_ms, 
               result.speedup, result.memory_bandwidth_gb_s, result.memory_utilization_percent,
               result.compute_utilization_percent, result.sfu_utilization_percent, result.memory_size_mb);
    }
    
    printf("========================================================================================================================\n");
    
    // Analysis
    printf("\n📊 PERFORMANCE ANALYSIS:\n");
    printf("1. Memory Bandwidth Utilization:\n");
    for (const auto& result : results) {
        printf("   %s (dim=%d): %.1f%% - %s\n", result.kernel_name, result.dim, result.memory_utilization_percent,
               result.memory_utilization_percent > 50 ? "🟢 Good" : 
               result.memory_utilization_percent > 25 ? "🟡 Moderate" : "🔴 Needs optimization");
    }
    
    printf("\n2. Speedup Comparison:\n");
    for (const auto& result : results) {
        printf("   %s (dim=%d): %.1fx - %s\n", result.kernel_name, result.dim, result.speedup,
               result.speedup > 1000 ? "🟢 Excellent" : 
               result.speedup > 100 ? "🟡 Good" : "🔴 Needs improvement");
    }
    
    printf("\n3. Memory Efficiency:\n");
    for (const auto& result : results) {
        printf("   %s (dim=%d): %zu MB - %s\n", result.kernel_name, result.dim, result.memory_size_mb,
               result.memory_size_mb < 1000 ? "🟢 Efficient" : 
               result.memory_size_mb < 5000 ? "🟡 Moderate" : "🔴 High memory usage");
    }
    
    printf("\n4. SFU Utilization Analysis:\n");
    for (const auto& result : results) {
        printf("   %s (dim=%d): %.1f%% - %s\n", result.kernel_name, result.dim, result.sfu_utilization_percent,
               result.sfu_utilization_percent > 20 ? "🟢 High SFU usage" : 
               result.sfu_utilization_percent > 10 ? "🟡 Moderate SFU usage" : "🔴 Low SFU usage");
    }
    
    printf("\n5. Kernel Performance Ranking (by speedup):\n");
    std::vector<TestResult> sorted_results = results;
    std::sort(sorted_results.begin(), sorted_results.end(), 
              [](const TestResult& a, const TestResult& b) { return a.speedup > b.speedup; });
    
    for (size_t i = 0; i < sorted_results.size(); i++) {
        printf("   %zu. %s (dim=%d): %.1fx\n", i + 1, sorted_results[i].kernel_name, 
               sorted_results[i].dim, sorted_results[i].speedup);
    }
}

// Function to run a single test case
void run_test_case(int test_idx, int batch_size, int dim, const GPUInfo& gpu_info, std::vector<TestResult>* results = nullptr) {
    const size_t total_elements = batch_size * dim;
    const size_t bytes = total_elements * sizeof(float);
    
    printf("=== Test Case %d: Dim=%d ===\n", test_idx + 1, dim);
    printf("Configuration:\n");
    printf("  Batch size: %d\n", batch_size);
    printf("  Dimension: %d\n", dim);
    printf("  Total elements: %zu\n", total_elements);
    printf("  Memory size: %.2f MB\n", bytes / (1024.0f * 1024.0f));
    
    // Check memory requirements
    size_t required_memory_gb = bytes / (1024.0f * 1024.0f * 1024.0f);
    size_t available_memory_gb = gpu_info.total_memory / (1024.0f * 1024.0f * 1024.0f);
    
    if (required_memory_gb > available_memory_gb * 0.8) {  // Use 80% of available memory as safety margin
        printf("  ⚠️  Skipping test case - memory requirement (%.2f GB) exceeds safe limit (%.2f GB)\n\n", 
               required_memory_gb, available_memory_gb * 0.8);
        return;
    }
    
    printf("  Memory usage: %.2f GB / %.2f GB (%.1f%%)\n\n", 
           required_memory_gb, available_memory_gb, (required_memory_gb / available_memory_gb) * 100.0f);
    
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
    
    // CPU reference
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
        // Cleanup and return
        free(h_input);
        free(h_output_gpu);
        free(h_output_cpu);
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
        return;
    }
    
    // Calculate and display performance metrics using the new utility function
    PerformanceMetrics metrics = calculate_metrics("Shared memory only", gpu_time, cpu_time, total_elements, dim,
                                                  threads_per_block, blocks_per_grid, gpu_info);
    print_performance_analysis(metrics, gpu_info);
    
    // Store results for summary
    if (results) {
        TestResult result;
        result.dim = dim;
        result.gpu_time_ms = gpu_time;
        result.cpu_time_ms = cpu_time;
        result.speedup = metrics.speedup;
        result.memory_bandwidth_gb_s = metrics.memory_bandwidth_gb_s;
        result.memory_utilization_percent = metrics.memory_utilization_percent;
        result.compute_utilization_percent = metrics.compute_utilization_percent;
        result.sfu_utilization_percent = calculate_sfu_utilization(batch_size, dim, gpu_time, gpu_info);
        result.exp_operations_per_second = (batch_size * dim * 2) / (gpu_time / 1000.0f);
        result.memory_size_mb = bytes / (1024 * 1024);
        result.kernel_name = "Shared memory only";
        results->push_back(result);
    }
    
    // Test other optimized kernels for comparison
    printf("\n=== Kernel Comparison ===\n");
    
    // 2. Warp primitives version (single warp)
    if (dim <= 1024) {
        int warp_threads = (32 < dim) ? 32 : dim;
        
        // Measure GPU execution time using the new utility function
        float warp_time = measure_kernel_ms(softmax_warp_primitives, d_input, d_output, 
                                          blocks_per_grid, warp_threads, 0,
                                          batch_size, dim);
        PerformanceMetrics warp_metrics = calculate_metrics("Warp Primitives only", warp_time, cpu_time, total_elements, dim,
                                                          warp_threads, blocks_per_grid, gpu_info);
        print_performance_analysis(warp_metrics, gpu_info);
        
        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(h_output_gpu, d_output, bytes, cudaMemcpyDeviceToHost));
        
        // Verify results
        if (verify_results(h_output_gpu, h_output_cpu, total_elements)) {
            printf("Warp Primitives only: ✅ Correct, Time: %.4f ms\n", warp_time);
        } else {
            printf("Warp Primitives only: ❌ Incorrect results\n");
        }
        
        // Store results for summary
        if (results) {
            TestResult result;
            result.dim = dim;
            result.gpu_time_ms = warp_time;
            result.cpu_time_ms = cpu_time;
            result.speedup = warp_metrics.speedup;
            result.memory_bandwidth_gb_s = warp_metrics.memory_bandwidth_gb_s;
            result.memory_utilization_percent = warp_metrics.memory_utilization_percent;
            result.compute_utilization_percent = warp_metrics.compute_utilization_percent;
            result.sfu_utilization_percent = calculate_sfu_utilization(batch_size, dim, warp_time, gpu_info);
            result.exp_operations_per_second = (batch_size * dim * 2) / (warp_time / 1000.0f);
            result.memory_size_mb = bytes / (1024 * 1024);
            result.kernel_name = "Warp Primitives only";
            results->push_back(result);
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
        PerformanceMetrics multi_warp_metrics = calculate_metrics("Warp Primitives + Shard memory", multi_warp_time, cpu_time, total_elements, dim,
                                                                multi_warp_threads, blocks_per_grid, gpu_info);
        print_performance_analysis(multi_warp_metrics, gpu_info);
        
        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(h_output_gpu, d_output, bytes, cudaMemcpyDeviceToHost));
        
        // Verify results
        if (verify_results(h_output_gpu, h_output_cpu, total_elements)) {
            printf("Warp Primitives + Shard memory: ✅ Correct, Time: %.4f ms\n", multi_warp_time);
        } else {
            printf("Warp Primitives + Shard memory: ❌ Incorrect results\n");
        }
        
        // Store results for summary
        if (results) {
            TestResult result;
            result.dim = dim;
            result.gpu_time_ms = multi_warp_time;
            result.cpu_time_ms = cpu_time;
            result.speedup = multi_warp_metrics.speedup;
            result.memory_bandwidth_gb_s = multi_warp_metrics.memory_bandwidth_gb_s;
            result.memory_utilization_percent = multi_warp_metrics.memory_utilization_percent;
            result.compute_utilization_percent = multi_warp_metrics.compute_utilization_percent;
            result.sfu_utilization_percent = calculate_sfu_utilization(batch_size, dim, multi_warp_time, gpu_info);
            result.exp_operations_per_second = (batch_size * dim * 2) / (multi_warp_time / 1000.0f);
            result.memory_size_mb = bytes / (1024 * 1024);
            result.kernel_name = "Warp Primitives + Shard memory";
            results->push_back(result);
        }
    }
    
    // 4. Fused version
    size_t fused_shared_mem = threads_per_block * 2 * sizeof(float); // max + sum
    float fused_time = measure_kernel_ms(softmax_fused, d_input, d_output, 
                                       blocks_per_grid, threads_per_block, fused_shared_mem,
                                       batch_size, dim);
    metrics = calculate_metrics("Online softmax", fused_time, cpu_time, total_elements, dim,
                                                 threads_per_block, blocks_per_grid, gpu_info);
    print_performance_analysis(metrics, gpu_info);                                           
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output_gpu, d_output, bytes, cudaMemcpyDeviceToHost));
    
    // Verify results
    if (verify_results(h_output_gpu, h_output_cpu, total_elements)) {
        printf("Fused: ✅ Correct, Time: %.4f ms\n", fused_time);
    } else {
        printf("Fused: ❌ Incorrect results\n");
    }
    
    // Store results for summary
    if (results) {
        TestResult result;
        result.dim = dim;
        result.gpu_time_ms = fused_time;
        result.cpu_time_ms = cpu_time;
        result.speedup = metrics.speedup;
        result.memory_bandwidth_gb_s = metrics.memory_bandwidth_gb_s;
        result.memory_utilization_percent = metrics.memory_utilization_percent;
        result.compute_utilization_percent = metrics.compute_utilization_percent;
        result.sfu_utilization_percent = calculate_sfu_utilization(batch_size, dim, fused_time, gpu_info);
        result.exp_operations_per_second = (batch_size * dim * 2) / (fused_time / 1000.0f);
        result.memory_size_mb = bytes / (1024 * 1024);
        result.kernel_name = "Online softmax";
        results->push_back(result);
    }
    
    // Cleanup
    free(h_input);
    free(h_output_gpu);
    free(h_output_cpu);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    printf("\n=== Test Case %d Complete ===\n\n", test_idx + 1);
}

int main() {
    printf("=== CUDA Shared memory only Implementation with Performance Analysis ===\n\n");
    
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
    
    // Test configurations: batch_size=49152, dim=[128, 1024, 16384]
    // Note: dim=65536 would require ~12GB memory, which exceeds GPU capacity
    const int batch_size = 49152;
    const int test_dims[] = {128, 1024, 16384};
    const int num_tests = sizeof(test_dims) / sizeof(test_dims[0]);
    
    // Performance data storage
    std::vector<TestResult> results;
    
    printf("=== Running %d Test Cases ===\n", num_tests);
    printf("Batch size: %d (fixed)\n", batch_size);
    printf("Dimensions: [128, 1024, 16384]\n");
    printf("Kernels: [Shared memory only, Warp Primitives only, Warp Primitives + Shard memory, Online softmax]\n\n");
    
    for (int test_idx = 0; test_idx < num_tests; test_idx++) {
        run_test_case(test_idx, batch_size, test_dims[test_idx], gpu_info, &results);
    }
    
    // Print performance summary table
    print_performance_summary(results, gpu_info);
    
    printf("\n=== Educational Notes ===\n");
    printf("🎯 Softmax Formula: softmax(x_i) = exp(x_i - max(x)) / Σ(exp(x_j - max(x)))\n");
    printf("🔑 Key Points:\n");
    printf("  • Shared Memory: Uses block-level parallel reduction for better performance\n");
    printf("  • Warp Primitives: Hardware-level instructions for efficient communication\n");
    printf("  • Fused Kernels: Single-pass algorithms that reduce memory access\n");
    printf("  • Numerical Stability: Subtract max value before exp() to prevent overflow\n");
    printf("  • Output values sum to 1.0 for each batch element\n");
    printf("  • Commonly used in neural networks for classification\n");
    
    printf("\n📊 Performance Insights:\n");
    printf("  • Different kernels excel at different data sizes\n");
    printf("  • Memory bandwidth utilization is key to performance\n");
    printf("  • Fused kernels often provide the best overall performance\n");
    printf("  • Choose kernel based on your specific data size and requirements\n");
    
    printf("\n=== All Test Cases Complete ===\n");
    return 0;
}