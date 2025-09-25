#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cstring>
#include "../utils/perf.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// CUDA kernel for basic softmax implementation
// Each thread handles one row (batch element)
__global__ void softmax_basic(float *input, float *output, int batch_size, int dim) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    float *row_input = input + batch_idx * dim;
    float *row_output = output + batch_idx * dim;
    
    // Step 1: Find maximum value for numerical stability
    float max_val = row_input[0];
    for (int i = 1; i < dim; i++) {
        max_val = fmaxf(max_val, row_input[i]);
    }
    
    // Step 2: Compute sum of exponentials
    float sum_exp = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum_exp += expf(row_input[i] - max_val);
    }
    
    // Step 3: Compute softmax values
    for (int i = 0; i < dim; i++) {
        row_output[i] = expf(row_input[i] - max_val) / sum_exp;
    }
}

// CPU reference implementation
void softmax_cpu(float *input, float *output, int batch_size, int dim) {
    for (int b = 0; b < batch_size; b++) {
        float *row_input = input + b * dim;
        float *row_output = output + b * dim;
        
        // Find maximum
        float max_val = row_input[0];
        for (int i = 1; i < dim; i++) {
            max_val = fmaxf(max_val, row_input[i]);
        }
        
        // Compute sum of exponentials
        float sum_exp = 0.0f;
        for (int i = 0; i < dim; i++) {
            sum_exp += expf(row_input[i] - max_val);
        }
        
        // Compute softmax
        for (int i = 0; i < dim; i++) {
            row_output[i] = expf(row_input[i] - max_val) / sum_exp;
        }
    }
}

// Verification function
bool verify_results(float *gpu_result, float *cpu_result, int total_elements, float tolerance = 1e-5) {
    for (int i = 0; i < total_elements; i++) {
        if (fabsf(gpu_result[i] - cpu_result[i]) > tolerance) {
            printf("Mismatch at element %d: GPU=%.6f, CPU=%.6f, diff=%.6f\n", 
                   i, gpu_result[i], cpu_result[i], fabsf(gpu_result[i] - cpu_result[i]));
            return false;
        }
    }
    return true;
}

int main() {
    printf("=== CUDA Softmax Basic Implementation with Performance Analysis ===\n\n");
    
    // Get GPU information
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
    
    // Configuration
    const int batch_size = 49152;
    const int dim = 1024;
    const int total_size = batch_size * dim;
    const size_t bytes = total_size * sizeof(float);
    
    printf("Configuration:\n");
    printf("  Batch size: %d\n", batch_size);
    printf("  Dimension: %d\n", dim);
    printf("  Total elements: %d\n", total_size);
    printf("  Memory size: %.2f MB\n\n", bytes / (1024.0f * 1024.0f));
    
    // Allocate host memory
    float *h_input = (float*)malloc(bytes);
    float *h_output_gpu = (float*)malloc(bytes);
    float *h_output_cpu = (float*)malloc(bytes);
    
    // Initialize input data with random values
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < dim; i++) {
            h_input[b * dim + i] = (float)(rand() % 100) / 10.0f - 5.0f; // Range: -5.0 to 5.0
        }
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    
    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    
    // CPU reference timing
    auto cpu_start = std::chrono::high_resolution_clock::now();
    softmax_cpu(h_input, h_output_cpu, batch_size, dim);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
    float cpu_time = cpu_duration.count() / 1000.0f;
    
    // GPU kernel configuration
    int threads_per_block = std::min(256, batch_size);
    int blocks_per_grid = (batch_size + threads_per_block - 1) / threads_per_block;
    
    printf("Kernel configuration:\n");
    printf("  Threads per block: %d\n", threads_per_block);
    printf("  Blocks per grid: %d\n", blocks_per_grid);
    printf("  Total threads: %d\n", threads_per_block * blocks_per_grid);
    printf("\n");
    
    // GPU timing using the unified measure_kernel_ms function
    float gpu_time = measure_kernel_ms(softmax_basic, d_input, d_output, 
                                     blocks_per_grid, threads_per_block, 0,
                                     batch_size, dim);
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output_gpu, d_output, bytes, cudaMemcpyDeviceToHost));
    
    // Verify results
    if (verify_results(h_output_gpu, h_output_cpu, total_size)) {
        printf("Verification:\n");
        printf("  ✅ Results match! GPU implementation is correct.\n\n");
    } else {
        printf("Verification:\n");
        printf("  ❌ Results don't match! GPU implementation has errors.\n\n");
        return 1;
    }
    
    // Calculate and display performance metrics using the unified utility function
    PerformanceMetrics metrics = calculate_metrics("Basic Softmax", gpu_time, cpu_time, 
                                                 total_size, dim, 
                                                 threads_per_block, blocks_per_grid, gpu_info);
    print_performance_analysis(metrics, gpu_info);
    
    // Run configuration benchmarking using the unified utility function
    benchmark_configurations(softmax_basic, gpu_info);
    
    // Educational explanation
    printf("\n=== Educational Notes ===\n");
    printf("🎯 Softmax Formula: softmax(x_i) = exp(x_i - max(x)) / Σ(exp(x_j - max(x)))\n");
    printf("🔑 Key Points:\n");
    printf("  • Numerical Stability: Subtract max value before exp() to prevent overflow\n");
    printf("  • Each thread handles one batch element (row-wise parallelization)\n");
    printf("  • Three passes: find max, compute sum of exp, normalize\n");
    printf("  • Output values sum to 1.0 for each batch element\n");
    printf("  • Commonly used in neural networks for classification\n\n");
    
    printf("📊 Performance Insights:\n");
    printf("  • Memory bandwidth is often the limiting factor, not computation\n");
    printf("  • Larger batch sizes improve GPU utilization\n");
    printf("  • Thread block size affects occupancy and performance\n");
    printf("  • Memory access patterns determine bandwidth efficiency\n");
    printf("  • exp() function is computationally expensive but necessary\n\n");
    
    printf("🚀 Next Steps:\n");
    printf("  • Try optimized version with shared memory (run11)\n");
    printf("  • Test with transformer-like shapes (make test)\n");
    printf("  • Learn about kernel fusion for better performance\n");
    printf("  • Explore mixed precision (FP16) implementations\n");
    
    // Cleanup
    free(h_input);
    free(h_output_gpu);
    free(h_output_cpu);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    printf("\n=== Program Complete ===\n");
    return 0;
}
