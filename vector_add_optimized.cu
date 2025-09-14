#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <math.h>

// Original basic vector add kernel
__global__ void vectorAddBasic(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Optimized version 1: Vectorized loads (float4)
__global__ void vectorAddVectorized(float4 *a, float4 *b, float4 *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float4 va = a[idx];
        float4 vb = b[idx];
        float4 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;
        c[idx] = vc;
    }
}

// Optimized version 2: Grid-stride loop for better occupancy
__global__ void vectorAddGridStride(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

// Optimized version 3: Shared memory caching (for repeated access)
__global__ void vectorAddSharedCache(float *a, float *b, float *c, int n) {
    __shared__ float cache_a[512];
    __shared__ float cache_b[512];
    
    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + tid;
    
    // Load to shared memory
    if (global_idx < n) {
        cache_a[tid] = a[global_idx];
        cache_b[tid] = b[global_idx];
    }
    
    __syncthreads();
    
    // Compute from shared memory
    if (global_idx < n) {
        c[global_idx] = cache_a[tid] + cache_b[tid];
    }
}

// Helper function for timing
class GpuTimer {
public:
    cudaEvent_t start, stop;
    
    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    
    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    void Start() {
        cudaEventRecord(start);
    }
    
    float Stop() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed;
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

void printDeviceInfo() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("=== GPU DEVICE INFO ===\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0*1024.0*1024.0));
    printf("Memory Clock Rate: %.2f GHz\n", prop.memoryClockRate / 1e6);
    printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("Peak Memory Bandwidth: %.2f GB/s\n", 
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Shared memory per block: %d KB\n", prop.sharedMemPerBlock / 1024);
    printf("\n");
}

float calculateBandwidth(int n, float time_ms) {
    // Vector add: reads 2 arrays, writes 1 array = 3 * N * sizeof(float) bytes
    float bytes = 3.0f * n * sizeof(float);
    float bandwidth_gb_s = (bytes / (1024*1024*1024)) / (time_ms / 1000.0f);
    return bandwidth_gb_s;
}

void runBenchmark(const char* name, int n, int blocks, int threads, 
                  void (*kernel)(float*, float*, float*, int), 
                  float *d_a, float *d_b, float *d_c) {
    
    GpuTimer timer;
    const int iterations = 100;
    
    // Warmup
    kernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    timer.Start();
    for (int i = 0; i < iterations; i++) {
        kernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
    }
    float time_ms = timer.Stop();
    
    float avg_time = time_ms / iterations;
    float bandwidth = calculateBandwidth(n, avg_time);
    
    printf("%-20s: %8.3f ms, %8.2f GB/s\n", name, avg_time, bandwidth);
}

void runVectorizedBenchmark(const char* name, int n, int blocks, int threads,
                           float4 *d_a4, float4 *d_b4, float4 *d_c4) {
    
    GpuTimer timer;
    const int iterations = 100;
    int n4 = n / 4;
    
    // Warmup
    vectorAddVectorized<<<blocks, threads>>>(d_a4, d_b4, d_c4, n4);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    timer.Start();
    for (int i = 0; i < iterations; i++) {
        vectorAddVectorized<<<blocks, threads>>>(d_a4, d_b4, d_c4, n4);
    }
    float time_ms = timer.Stop();
    
    float avg_time = time_ms / iterations;
    float bandwidth = calculateBandwidth(n, avg_time);
    
    printf("%-20s: %8.3f ms, %8.2f GB/s\n", name, avg_time, bandwidth);
}

int main() {
    printf("=== CUDA VECTOR ADD PERFORMANCE ANALYSIS ===\n\n");
    
    printDeviceInfo();
    
    // Test different sizes
    const int sizes[] = {1024*1024, 16*1024*1024, 64*1024*1024, 256*1024*1024};
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    for (int s = 0; s < num_sizes; s++) {
        int N = sizes[s];
        printf("=== VECTOR SIZE: %d elements (%.2f MB) ===\n", 
               N, N * sizeof(float) / (1024.0*1024.0));
        
        size_t size = N * sizeof(float);
        
        // Allocate host memory
        float *h_a = (float*)malloc(size);
        float *h_b = (float*)malloc(size);
        float *h_c = (float*)malloc(size);
        
        // Initialize data
        for (int i = 0; i < N; i++) {
            h_a[i] = 1.0f;
            h_b[i] = 2.0f;
        }
        
        // Allocate device memory
        float *d_a, *d_b, *d_c;
        CUDA_CHECK(cudaMalloc(&d_a, size));
        CUDA_CHECK(cudaMalloc(&d_b, size));
        CUDA_CHECK(cudaMalloc(&d_c, size));
        
        // Copy to device
        CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));
        
        // Test different block sizes
        int block_sizes[] = {128, 256, 512, 1024};
        int num_block_sizes = sizeof(block_sizes) / sizeof(block_sizes[0]);
        
        printf("\nBlock Size Optimization:\n");
        for (int bs = 0; bs < num_block_sizes; bs++) {
            int threads = block_sizes[bs];
            int blocks = (N + threads - 1) / threads;
            
            char name[50];
            sprintf(name, "Basic (%d threads)", threads);
            runBenchmark(name, N, blocks, threads, vectorAddBasic, d_a, d_b, d_c);
        }
        
        // Optimal configuration for remaining tests
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        
        printf("\nKernel Optimizations:\n");
        runBenchmark("Basic", N, blocks, threads, vectorAddBasic, d_a, d_b, d_c);
        
        // Grid-stride version
        int grid_stride_blocks = min(blocks, 1024); // Limit blocks for grid-stride
        runBenchmark("Grid-Stride", N, grid_stride_blocks, threads, vectorAddGridStride, d_a, d_b, d_c);
        
        // Shared memory version (only for smaller sizes due to shared mem limits)
        if (threads <= 512) {
            runBenchmark("Shared Cache", N, blocks, threads, vectorAddSharedCache, d_a, d_b, d_c);
        }
        
        // Vectorized version (float4)
        if (N % 4 == 0) {
            float4 *d_a4 = (float4*)d_a;
            float4 *d_b4 = (float4*)d_b;
            float4 *d_c4 = (float4*)d_c;
            int blocks4 = ((N/4) + threads - 1) / threads;
            runVectorizedBenchmark("Vectorized (float4)", N, blocks4, threads, d_a4, d_b4, d_c4);
        }
        
        // Memory transfer overhead analysis
        if (s == 0) { // Only for first size
            printf("\nMemory Transfer Analysis:\n");
            GpuTimer timer;
            
            // Host to Device
            timer.Start();
            CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
            float h2d_time = timer.Stop();
            
            // Device to Host  
            timer.Start();
            CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));
            float d2h_time = timer.Stop();
            
            printf("Host->Device: %8.3f ms, %8.2f GB/s\n", 
                   h2d_time, (size/(1024*1024*1024)) / (h2d_time/1000.0f));
            printf("Device->Host: %8.3f ms, %8.2f GB/s\n", 
                   d2h_time, (size/(1024*1024*1024)) / (d2h_time/1000.0f));
        }
        
        // Verify correctness
        CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));
        bool correct = true;
        for (int i = 0; i < min(1000, N); i++) {
            if (fabs(h_c[i] - 3.0f) > 1e-5) {
                correct = false;
                break;
            }
        }
        printf("Result verification: %s\n", correct ? "PASSED" : "FAILED");
        
        // Cleanup
        free(h_a); free(h_b); free(h_c);
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        
        printf("\n");
    }
    
    printf("=== PERFORMANCE OPTIMIZATION TIPS ===\n\n");
    
    printf("1. MEMORY BANDWIDTH OPTIMIZATION:\n");
    printf("   - Use vectorized loads (float2, float4) when possible\n");
    printf("   - Ensure coalesced memory access patterns\n");
    printf("   - Consider memory alignment\n\n");
    
    printf("2. OCCUPANCY OPTIMIZATION:\n");
    printf("   - Use 128-512 threads per block (multiples of 32)\n");
    printf("   - Balance shared memory usage vs occupancy\n");
    printf("   - Use grid-stride loops for large datasets\n\n");
    
    printf("3. INSTRUCTION OPTIMIZATION:\n");
    printf("   - Minimize divergent branches\n");
    printf("   - Use built-in math functions when possible\n");
    printf("   - Avoid expensive operations (division, sqrt)\n\n");
    
    printf("4. PROFILING TOOLS:\n");
    printf("   - nvprof: Legacy profiler\n");
    printf("   - Nsight Compute: Modern detailed profiler\n");
    printf("   - Nsight Systems: Timeline profiler\n\n");
    
    printf("Run with: nvprof ./vector_add_optimized\n");
    printf("Or: ncu ./vector_add_optimized\n");
    
    return 0;
}
