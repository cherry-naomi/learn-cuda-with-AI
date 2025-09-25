#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cstring>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// GPU information structure
struct GPUInfo {
    int device_id;
    char name[256];
    size_t total_memory;
    int multiprocessor_count;
    int max_threads_per_block;
    int max_threads_per_multiprocessor;
    int warp_size;
    int memory_bus_width;
    int memory_clock_rate;
    int compute_capability_major;
    int compute_capability_minor;
    float peak_memory_bandwidth_gb_s;
    float peak_compute_throughput_tflops;
};

// Performance analysis structure
struct PerformanceMetrics {
    char kernel_name[128];
    float gpu_time_ms;
    float cpu_time_ms;
    float speedup;
    size_t total_elements;
    float memory_bandwidth_gb_s; 
    float memory_utilization_percent;
    float compute_utilization_percent;
    int threads_per_block;
    int blocks_per_grid;
};

// Get detailed GPU information
GPUInfo get_gpu_info() {
    GPUInfo info = {};
    
    CUDA_CHECK(cudaGetDevice(&info.device_id));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, info.device_id));
    
    // Copy device properties to our structure
    strncpy(info.name, prop.name, sizeof(info.name) - 1);
    info.name[sizeof(info.name) - 1] = '\0';
    info.total_memory = prop.totalGlobalMem;
    info.multiprocessor_count = prop.multiProcessorCount;
    info.max_threads_per_block = prop.maxThreadsPerBlock;
    info.max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;
    info.warp_size = prop.warpSize;
    info.memory_bus_width = prop.memoryBusWidth;
    info.memory_clock_rate = prop.memoryClockRate;
    info.compute_capability_major = prop.major;
    info.compute_capability_minor = prop.minor;
    
    // Calculate peak memory bandwidth
    info.peak_memory_bandwidth_gb_s = (info.memory_clock_rate * 1e3f * 2 * info.memory_bus_width / 8.0f) / 1e9f;
    
    // Calculate peak compute throughput (TFLOPS)
    // Formula: SMs × CUDA_cores_per_SM × Clock_frequency × FLOPS_per_cycle / 1e12
    // A100 has 64 CUDA cores per SM, each can perform 2 FLOPS per cycle (FMA operations)
    float clock_frequency_ghz = prop.clockRate / 1e6f; // Convert kHz to GHz
    
    // CUDA cores per SM for different architectures
    int cuda_cores_per_sm;
    if (prop.major == 8) {  // Ampere (A100, RTX 30xx, RTX 40xx)
        cuda_cores_per_sm = 64;
    } else if (prop.major == 7) {  // Turing/Volta
        cuda_cores_per_sm = 64;
    } else if (prop.major == 6) {  // Pascal
        cuda_cores_per_sm = 128;
    } else {  // Default fallback
        cuda_cores_per_sm = 64;
    }
    
    double tflops = ((double)info.multiprocessor_count * cuda_cores_per_sm * 
                    clock_frequency_ghz * 1e-3);
    
    info.peak_compute_throughput_tflops = (float)tflops;
    
    return info;
}

// Calculate performance metrics
PerformanceMetrics calculate_metrics(const char* kernel_name, float gpu_time_ms, float cpu_time_ms, 
                                   size_t total_elements, int dim,
                                   int threads_per_block, int blocks_per_grid,
                                   const GPUInfo& gpu_info) {
    PerformanceMetrics metrics = {};
    strncpy(metrics.kernel_name, kernel_name, sizeof(metrics.kernel_name) - 1);
    metrics.kernel_name[sizeof(metrics.kernel_name) - 1] = '\0';
    metrics.gpu_time_ms = gpu_time_ms;
    metrics.cpu_time_ms = cpu_time_ms;
    metrics.speedup = cpu_time_ms / gpu_time_ms;
    metrics.total_elements = total_elements;
    metrics.threads_per_block = threads_per_block;
    metrics.blocks_per_grid = blocks_per_grid;
    
    // Memory performance
    size_t bytes = total_elements * sizeof(float);
    metrics.memory_bandwidth_gb_s = (2.0f * bytes) / (gpu_time_ms * 1e-3f) / 1e9f; // Read + Write
    metrics.memory_utilization_percent = (metrics.memory_bandwidth_gb_s / gpu_info.peak_memory_bandwidth_gb_s) * 100.0f;
    
    // Compute utilization (rough estimation)
    // Each thread does: 3*dim comparisons + 2*dim exp() + 2*dim divisions
    float operations_per_element = 5.0f; // Approximate operations
    float total_operations = metrics.total_elements * operations_per_element;
    float operations_per_second = total_operations / (gpu_time_ms * 1e-3f);
    metrics.compute_utilization_percent = (operations_per_second / (gpu_info.peak_compute_throughput_tflops * 1e12f)) * 100.0f;
    
    return metrics;
}

// Print detailed performance analysis
void print_performance_analysis(const PerformanceMetrics& metrics, const GPUInfo& gpu_info) {
    printf("\n=== Detailed Performance Analysis ===\n");
    printf("🚀 Kernel: %s\n", metrics.kernel_name);
    printf("📊 Execution Time:\n");
    printf("  GPU time: %.4f ms\n", metrics.gpu_time_ms);
    printf("  CPU time: %.4f ms\n", metrics.cpu_time_ms);
    printf("  Speedup: %.2fx\n", metrics.speedup);
    
    printf("\n💾 Memory Performance:\n");
    printf("  Data size: %zu elements (%.2f MB)\n", 
           metrics.total_elements, metrics.total_elements * sizeof(float) / (1024.0f * 1024.0f));
    printf("  Memory bandwidth: %.2f GB/s\n", metrics.memory_bandwidth_gb_s);
    printf("  Peak bandwidth: %.2f GB/s\n", gpu_info.peak_memory_bandwidth_gb_s);
    printf("  Bandwidth utilization: %.1f%%\n", metrics.memory_utilization_percent);
    
    printf("\n⚡ Compute Performance:\n");
    printf("  Threads per block: %d\n", metrics.threads_per_block);
    printf("  Blocks per grid: %d\n", metrics.blocks_per_grid);
    printf("  Total threads: %d\n", metrics.threads_per_block * metrics.blocks_per_grid);
    printf("  Compute utilization: %.1f%%\n", metrics.compute_utilization_percent);
    
    printf("\n🎯 Performance Rating:\n");
    if (metrics.memory_utilization_percent < 10.0f) {
        printf("  Memory: 🔴 Needs optimization (%.1f%%)\n", metrics.memory_utilization_percent);
    } else if (metrics.memory_utilization_percent < 50.0f) {
        printf("  Memory: 🟡 Moderate utilization (%.1f%%)\n", metrics.memory_utilization_percent);
    } else {
        printf("  Memory: 🟢 Good utilization (%.1f%%)\n", metrics.memory_utilization_percent);
    }
    
    if (metrics.speedup > 10.0f) {
        printf("  Speedup: 🟢 Excellent (%.1fx)\n", metrics.speedup);
    } else if (metrics.speedup > 5.0f) {
        printf("  Speedup: 🟡 Good (%.1fx)\n", metrics.speedup);
    } else {
        printf("  Speedup: 🔴 Needs improvement (%.1fx)\n", metrics.speedup);
    }
}

// Enhanced measure_kernel_ms function with multiple signatures
template<typename KernelFunc>
float measure_kernel_ms(KernelFunc kernel, int iterations = 2) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm up
    kernel();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        kernel();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / iterations;
}

// Specialized version for softmax kernels
float measure_kernel_ms(void (*kernel)(float*, float*, int, int),
                        float *d_input, float *d_output,
                        int blocks, int threads, size_t shared_mem,
                        int batch_size, int dim, int iterations = 2) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm up
    // kernel<<<blocks, threads, shared_mem>>>(d_input, d_output, batch_size, dim);
    // CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        kernel<<<blocks, threads, shared_mem>>>(d_input, d_output, batch_size, dim);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / iterations;
}

// Specialized version for FP16 softmax kernels
float measure_kernel_ms(void (*kernel)(half*, half*, int, int),
                        half *d_input, half *d_output,
                        int blocks, int threads, size_t shared_mem,
                        int batch_size, int dim, int iterations = 2) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm up
    // kernel<<<blocks, threads, shared_mem>>>(d_input, d_output, batch_size, dim);
    // CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        kernel<<<blocks, threads, shared_mem>>>(d_input, d_output, batch_size, dim);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / iterations;
}

// Benchmark different configurations
void benchmark_configurations(void (*kernel)(float*, float*, int, int), 
                             const GPUInfo& gpu_info) {
    printf("\n=== Configuration Benchmarking ===\n");
    
    const int configs[][3] = {
        {1, 64, 1},      // Small
        {1, 256, 2},     // Medium  
        {1, 1024, 3},    // Large
        {4, 512, 4},     // Batch
        {16, 128, 5},    // Multi-batch
        {32, 256, 6},    // Training-like
        {2048, 512, 7}   // Original
    };
    const char* config_names[] = {
        "Small (1x64)",
        "Medium (1x256)", 
        "Large (1x1024)",
        "Batch (4x512)",
        "Multi-batch (16x128)",
        "Training-like (32x256)",
        "Original (2048x512)"
    };
    const int num_configs = sizeof(configs) / sizeof(configs[0]);
    
    for (int i = 0; i < num_configs; i++) {
        int batch_size = configs[i][0];
        int dim = configs[i][1];
        int config_id = configs[i][2];
        
        printf("Config %d: %s\n", config_id, config_names[i]);
        
        size_t total_elements = batch_size * dim;
        size_t bytes = total_elements * sizeof(float);
        
        // Allocate host memory
        float *h_input = (float*)malloc(bytes);
        float *h_output_gpu = (float*)malloc(bytes);
        
        // Initialize with random data
        for (size_t j = 0; j < total_elements; j++) {
            h_input[j] = (float)(rand() % 200) / 10.0f - 10.0f;
        }
        
        // Allocate device memory
        float *d_input, *d_output;
        CUDA_CHECK(cudaMalloc(&d_input, bytes));
        CUDA_CHECK(cudaMalloc(&d_output, bytes));
        
        // Copy input data to device
        CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
        
        // CPU reference timing
        auto cpu_start = std::chrono::high_resolution_clock::now();
        // Note: CPU reference would need to be implemented separately
        auto cpu_end = std::chrono::high_resolution_clock::now();
        float cpu_time = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
        
        // GPU optimized version
        int threads_per_block = (256 < dim) ? 256 : dim;
        if (threads_per_block > 1024) threads_per_block = 1024;
        int blocks_per_grid = batch_size;
        size_t shared_mem = threads_per_block * sizeof(float);
        
        // Measure GPU execution time using the new function
        float gpu_time = measure_kernel_ms(kernel, d_input, d_output, 
                                         blocks_per_grid, threads_per_block, shared_mem,
                                         batch_size, dim);
        
        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(h_output_gpu, d_output, bytes, cudaMemcpyDeviceToHost));
        
        // Calculate metrics
        PerformanceMetrics metrics = calculate_metrics("Benchmark Kernel", gpu_time, cpu_time, total_elements, dim,
                                                      threads_per_block, blocks_per_grid, gpu_info);
        
        printf("  GPU: %.4f ms, CPU: %.4f ms, Speedup: %.2fx, BW: %.2f GB/s (%.1f%%), Compute Utilization: %.1f%%\n",
                metrics.gpu_time_ms, metrics.cpu_time_ms, metrics.speedup, 
                metrics.memory_bandwidth_gb_s, metrics.memory_utilization_percent,
                metrics.compute_utilization_percent);
        
        // Cleanup
        free(h_input);
        free(h_output_gpu);
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
    }
}

// Verify results between GPU and CPU implementations
bool verify_results(const float* gpu_result, const float* cpu_result, size_t total_elements, float tolerance = 1e-5f) {
    for (size_t i = 0; i < total_elements; i++) {
        if (fabsf(gpu_result[i] - cpu_result[i]) > tolerance) {
            printf("Mismatch at index %zu: GPU=%.6f, CPU=%.6f, diff=%.6f\n", 
                   i, gpu_result[i], cpu_result[i], fabsf(gpu_result[i] - cpu_result[i]));
            return false;
        }
    }
    return true;
}

