#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// CUDA kernel for vector addition
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Print thread information for the first few threads (for educational purposes)
    if (idx < 16) {
        printf("Thread %d: blockIdx.x=%d, threadIdx.x=%d, blockDim.x=%d, gridDim.x=%d\n",
               idx, blockIdx.x, threadIdx.x, blockDim.x, gridDim.x);
    }
    
    // Ensure we don't go out of bounds
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Helper function to check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

int main() {
    // Vector size
    const int N = 1024*1024;
    const size_t size = N * sizeof(float);
    
    // Host vectors
    float *h_a, *h_b, *h_c;
    
    // Device vectors
    float *d_a, *d_b, *d_c;
    
    // Allocate host memory
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);
    
    // Initialize host vectors
    printf("Initializing vectors...\n");
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMalloc(&d_c, size));
    
    // Copy vectors from host to device
    printf("Copying data to GPU...\n");
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));
    
    // Define grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("\nKernel configuration:\n");
    printf("Vector size: %d\n", N);
    printf("Threads per block: %d\n", threadsPerBlock);
    printf("Blocks per grid: %d\n", blocksPerGrid);
    printf("Total threads: %d\n", blocksPerGrid * threadsPerBlock);
    
    // Launch the kernel
    printf("\nLaunching kernel...\n");
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Wait for the kernel to complete
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    printf("Copying result back to CPU...\n");
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));
    
    // Verify the result
    printf("Verifying results...\n");
    bool success = true;
    for (int i = 0; i < N; i++) {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5) {
            printf("Error at index %d: expected %f, got %f\n", i, expected, h_c[i]);
            success = false;
            break;
        }
    }
    
    if (success) {
        printf("Vector addition completed successfully!\n");
        
        // Print first 10 results as examples
        printf("\nFirst 10 results:\n");
        for (int i = 0; i < 10; i++) {
            printf("a[%d] + b[%d] = %.1f + %.1f = %.1f\n", 
                   i, i, h_a[i], h_b[i], h_c[i]);
        }
    } else {
        printf("Vector addition failed!\n");
    }
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    
    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);
    
    return success ? 0 : 1;
}
