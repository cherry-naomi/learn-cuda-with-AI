#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// CUDA kernel for vector addition using 2D thread blocks
__global__ void vectorAdd2D(float *a, float *b, float *c, int n) {
    // Calculate global thread index using 2D thread blocks
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Alternative way to think about it with 2D blocks:
    // int row = blockIdx.y * blockDim.y + threadIdx.y;
    // int col = blockIdx.x * blockDim.x + threadIdx.x;
    // int idx = row * gridDim.x * blockDim.x + col;
    
    // Print detailed thread information for educational purposes
    if (idx < 32) {  // Print for first 32 threads
        printf("Thread %d: Block(%d,%d) Thread(%d,%d) BlockDim(%d,%d) GridDim(%d,%d)\n",
               idx, 
               blockIdx.x, blockIdx.y,
               threadIdx.x, threadIdx.y,
               blockDim.x, blockDim.y,
               gridDim.x, gridDim.y);
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

void demonstrateGridConfigurations(float *d_a, float *d_b, float *d_c, int N) {
    printf("\n=== Demonstrating Different Grid/Block Configurations ===\n");
    
    // Configuration 1: 1D blocks, 1D grid
    {
        dim3 blockSize(256);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
        
        printf("\nConfiguration 1 - 1D blocks, 1D grid:\n");
        printf("Block size: (%d, %d, %d)\n", blockSize.x, blockSize.y, blockSize.z);
        printf("Grid size: (%d, %d, %d)\n", gridSize.x, gridSize.y, gridSize.z);
        printf("Total threads: %d\n", gridSize.x * blockSize.x);
        
        vectorAdd2D<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Configuration 2: 2D blocks, 1D grid  
    {
        dim3 blockSize(16, 16);  // 256 threads total
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
        
        printf("\nConfiguration 2 - 2D blocks, 1D grid:\n");
        printf("Block size: (%d, %d, %d)\n", blockSize.x, blockSize.y, blockSize.z);
        printf("Grid size: (%d, %d, %d)\n", gridSize.x, gridSize.y, gridSize.z);
        printf("Total threads: %d\n", gridSize.x * blockSize.x * blockSize.y);
        
        vectorAdd2D<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Configuration 3: 1D blocks, 2D grid
    {
        dim3 blockSize(128);
        int blocksNeeded = (N + blockSize.x - 1) / blockSize.x;
        dim3 gridSize(blocksNeeded / 2 + blocksNeeded % 2, 2);
        
        printf("\nConfiguration 3 - 1D blocks, 2D grid:\n");
        printf("Block size: (%d, %d, %d)\n", blockSize.x, blockSize.y, blockSize.z);
        printf("Grid size: (%d, %d, %d)\n", gridSize.x, gridSize.y, gridSize.z);
        printf("Total threads: %d\n", gridSize.x * gridSize.y * blockSize.x);
        
        vectorAdd2D<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

int main() {
    // Vector size
    const int N = 1024;
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
    printf("Initializing vectors with %d elements...\n", N);
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMalloc(&d_c, size));
    
    // Copy vectors from host to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));
    
    // Demonstrate different grid/block configurations
    demonstrateGridConfigurations(d_a, d_b, d_c, N);
    
    // Copy result back to host for verification
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));
    
    // Verify the result
    printf("\nVerifying results...\n");
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
        printf("All configurations completed successfully!\n");
        
        // Print some results
        printf("\nSample results:\n");
        for (int i = 0; i < 5; i++) {
            printf("a[%d] + b[%d] = %.0f + %.0f = %.0f\n", 
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
