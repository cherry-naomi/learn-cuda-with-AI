#ifndef PERF_H
#define PERF_H

#include <cuda_runtime.h>

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

// Function declarations
GPUInfo get_gpu_info();
PerformanceMetrics calculate_metrics(const char* kernel_name, float gpu_time_ms, float cpu_time_ms, 
                                   size_t total_elements, int dim,
                                   int threads_per_block, int blocks_per_grid,
                                   const GPUInfo& gpu_info);
void print_performance_analysis(const PerformanceMetrics& metrics, const GPUInfo& gpu_info);

// Enhanced measure_kernel_ms function
template<typename KernelFunc>
float measure_kernel_ms(KernelFunc kernel, int iterations = 2);

// Specialized version for softmax kernels
float measure_kernel_ms(void (*kernel)(float*, float*, int, int),
                        float *d_input, float *d_output,
                        int blocks, int threads, size_t shared_mem,
                        int batch_size, int dim, int iterations = 2);

void benchmark_configurations(void (*kernel)(float*, float*, int, int), 
                             const GPUInfo& gpu_info);
bool verify_results(const float* gpu_result, const float* cpu_result, size_t total_elements, float tolerance = 1e-5f);

// SFU utilization calculation
float calculate_sfu_utilization(int batch_size, int dim, float gpu_time_ms, const GPUInfo& gpu_info);

#endif // PERF_H
