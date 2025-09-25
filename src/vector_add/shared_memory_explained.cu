#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Kernel to demonstrate shared memory within a block
__global__ void demonstrateSharedMemory(int *global_output, int N) {
    // SHARED MEMORY: All threads in THIS BLOCK can access this
    __shared__ int shared_data[256];  // Shared by all threads in this block
    __shared__ int block_sum;         // Shared variable for block-wide operations
    
    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory (only thread 0 does this)
    if (tid == 0) {
        block_sum = 0;
        printf("Block %d: Initializing shared memory (done by thread 0)\n", blockIdx.x);
    }
    
    // SYNCHRONIZATION: Wait for thread 0 to finish initialization
    __syncthreads();  // All threads in block wait here
    
    // Each thread writes to shared memory
    if (tid < 256) {
        shared_data[tid] = global_idx * 2;  // Each thread writes its data
    }
    
    // Synchronize again before reading
    __syncthreads();
    
    // Demonstrate shared memory access: each thread can read others' data
    if (tid < 16 && global_idx < N) {  // First 16 threads in each block
        printf("Block %d, Thread %d: I can see shared_data[%d] = %d (written by thread %d)\n",
               blockIdx.x, tid, (tid + 1) % blockDim.x, 
               shared_data[(tid + 1) % blockDim.x], (tid + 1) % blockDim.x);
    }
    
    __syncthreads();
    
    // Cooperative operation: block-wide sum using shared memory
    // This is a simple reduction - each thread adds its value
    atomicAdd(&block_sum, shared_data[tid]);
    
    __syncthreads();
    
    // Only thread 0 writes the final result
    if (tid == 0) {
        global_output[blockIdx.x] = block_sum;
        printf("Block %d: Final sum = %d (computed cooperatively)\n", blockIdx.x, block_sum);
    }
}

// Kernel showing what happens with different blocks (they DON'T share memory)
__global__ void showBlockIndependence(int *data) {
    __shared__ int block_id_storage;
    
    int tid = threadIdx.x;
    
    // Each block stores its own ID in shared memory
    if (tid == 0) {
        block_id_storage = blockIdx.x * 1000;  // Unique value per block
    }
    
    __syncthreads();
    
    // All threads in block can access the same shared variable
    if (tid < 8) {  // First 8 threads per block
        printf("Block %d, Thread %d: Our block's shared storage = %d\n", 
               blockIdx.x, tid, block_id_storage);
    }
}

// Advanced example: Matrix multiplication using shared memory
__global__ void matrixMultiplyShared(float *A, float *B, float *C, int N) {
    // Shared memory tiles - shared by all threads in this block
    __shared__ float tile_A[16][16];
    __shared__ float tile_B[16][16];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    // Calculate global position
    int row = by * 16 + ty;
    int col = bx * 16 + tx;
    
    float result = 0;
    
    // Loop over tiles
    for (int tile = 0; tile < (N + 15) / 16; tile++) {
        // Cooperative loading: each thread loads one element into shared memory
        if (row < N && (tile * 16 + tx) < N)
            tile_A[ty][tx] = A[row * N + tile * 16 + tx];
        else
            tile_A[ty][tx] = 0;
            
        if ((tile * 16 + ty) < N && col < N)
            tile_B[ty][tx] = B[(tile * 16 + ty) * N + col];
        else
            tile_B[ty][tx] = 0;
        
        // Synchronize: wait for all threads to finish loading
        __syncthreads();
        
        // Compute using shared memory (much faster than global memory!)
        for (int k = 0; k < 16; k++) {
            result += tile_A[ty][k] * tile_B[k][tx];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < N && col < N) {
        C[row * N + col] = result;
    }
}

void explainBlockConcept() {
    printf("=== WHAT IS A CUDA BLOCK? ===\n\n");
    
    printf("A BLOCK is defined by SHARED RESOURCES:\n");
    printf("1. 🏠 SHARED MEMORY: All threads in block can access same shared memory\n");
    printf("2. 🔄 SYNCHRONIZATION: Threads in block can synchronize with __syncthreads()\n");
    printf("3. 🎯 COOPERATION: Threads can work together on shared tasks\n");
    printf("4. 📍 LOCALITY: Block runs on single SM (close physical proximity)\n\n");
    
    printf("MEMORY HIERARCHY:\n");
    printf("┌─────────────────────────────────────────────┐\n");
    printf("│ GLOBAL MEMORY (entire GPU)                 │ ← Slowest, largest\n");
    printf("├─────────────────────────────────────────────┤\n");
    printf("│ SHARED MEMORY (per block)                  │ ← Fast, medium size\n");
    printf("├─────────────────────────────────────────────┤\n");
    printf("│ REGISTERS (per thread)                     │ ← Fastest, smallest\n");
    printf("└─────────────────────────────────────────────┘\n\n");
    
    printf("BLOCK BOUNDARIES:\n");
    printf("Block 0: Threads 0-255 → Share Shared Memory A\n");
    printf("Block 1: Threads 0-255 → Share Shared Memory B (DIFFERENT from A!)\n");
    printf("Block 2: Threads 0-255 → Share Shared Memory C (DIFFERENT from A & B!)\n\n");
    
    printf("KEY INSIGHT: Threads in DIFFERENT blocks CANNOT:\n");
    printf("❌ Access each other's shared memory\n");
    printf("❌ Synchronize with each other\n"); 
    printf("❌ Cooperate directly\n");
    printf("✅ Only communicate through global memory\n\n");
}

int main() {
    printf("=== CUDA BLOCKS AND SHARED MEMORY EXPLAINED ===\n\n");
    
    explainBlockConcept();
    
    // Demonstrate shared memory within blocks
    printf("=== DEMONSTRATION 1: Shared Memory Within Blocks ===\n");
    
    const int N = 64;
    int *d_output;
    cudaMalloc(&d_output, 4 * sizeof(int));  // 4 blocks output
    
    printf("Launching 4 blocks, 16 threads each:\n");
    printf("Each block will demonstrate shared memory access:\n\n");
    
    demonstrateSharedMemory<<<4, 16>>>(d_output, N);
    cudaDeviceSynchronize();
    
    // Get results
    int h_output[4];
    cudaMemcpy(h_output, d_output, 4 * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("\nBlock sums computed using shared memory:\n");
    for (int i = 0; i < 4; i++) {
        printf("Block %d sum: %d\n", i, h_output[i]);
    }
    
    printf("\n=== DEMONSTRATION 2: Block Independence ===\n");
    printf("Showing that different blocks have separate shared memory:\n\n");
    
    showBlockIndependence<<<3, 8>>>(nullptr);
    cudaDeviceSynchronize();
    
    printf("\n=== SHARED MEMORY BENEFITS ===\n");
    printf("1. SPEED: ~100x faster than global memory access\n");
    printf("2. COOPERATION: Enables efficient algorithms (reductions, matrix multiply)\n");
    printf("3. DATA REUSE: Cache frequently accessed data\n");
    printf("4. COMMUNICATION: Threads can share intermediate results\n\n");
    
    printf("=== TYPICAL SHARED MEMORY USAGE PATTERNS ===\n\n");
    
    printf("Pattern 1: DATA CACHING\n");
    printf("__shared__ float cache[BLOCK_SIZE];\n");
    printf("cache[threadIdx.x] = global_data[global_idx];  // Load to shared\n");
    printf("__syncthreads();                               // Wait for all\n");
    printf("result = cache[threadIdx.x] * 2;               // Use cached data\n\n");
    
    printf("Pattern 2: COOPERATIVE REDUCTION\n");
    printf("__shared__ float sdata[BLOCK_SIZE];\n");
    printf("sdata[tid] = input[tid];\n");
    printf("__syncthreads();\n");
    printf("// Parallel reduction in shared memory\n");
    printf("for (int s = 1; s < blockDim.x; s *= 2) {\n");
    printf("    if (tid %% (2*s) == 0) sdata[tid] += sdata[tid + s];\n");
    printf("    __syncthreads();\n");
    printf("}\n\n");
    
    printf("Pattern 3: TILED ALGORITHMS (like matrix multiply)\n");
    printf("__shared__ float tile_A[TILE_SIZE][TILE_SIZE];\n");
    printf("__shared__ float tile_B[TILE_SIZE][TILE_SIZE];\n");
    printf("// Load tile cooperatively\n");
    printf("// Compute using shared data\n");
    printf("// Repeat for all tiles\n\n");
    
    printf("=== MEMORY SIZE LIMITS ===\n");
    
    // Query shared memory info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Your GPU shared memory per block: %d KB\n", prop.sharedMemPerBlock / 1024);
    printf("Your GPU shared memory per SM: %d KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("Typical usage: 16-48 KB per block\n\n");
    
    printf("=== SYNCHRONIZATION RULES ===\n");
    printf("✅ __syncthreads(): Synchronizes all threads in SAME block\n");
    printf("❌ NO synchronization between different blocks\n");
    printf("❌ NO direct communication between blocks\n");
    printf("✅ Blocks are independent and can run in any order\n\n");
    
    printf("SUMMARY: A block = a team of threads that share workspace (shared memory)\n");
    printf("         and can coordinate with each other (__syncthreads())\n");
    
    cudaFree(d_output);
    return 0;
}
