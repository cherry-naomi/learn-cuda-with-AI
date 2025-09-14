#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Kernel to demonstrate hardware mapping concepts
__global__ void explainHardwareMapping(int *data, int N) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate warp ID within the block
    int warpId = threadIdx.x / 32;  // 32 threads per warp
    int laneId = threadIdx.x % 32;  // Position within warp (0-31)
    
    // Print detailed hardware mapping info for first few threads
    if (global_idx < 64) {  // Show first 64 threads (2 warps per block if blockDim=32+)
        printf("Thread %2d: Block %d, ThreadIdx %2d, Warp %d, Lane %2d\n",
               global_idx, blockIdx.x, threadIdx.x, warpId, laneId);
    }
    
    // Simulate some work
    if (global_idx < N) {
        data[global_idx] = global_idx * blockIdx.x + warpId;
    }
}

// Kernel to demonstrate warp divergence
__global__ void demonstrateWarpDivergence(int *data, int N) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    
    if (global_idx < N) {
        // BAD: This causes warp divergence!
        if (laneId % 2 == 0) {
            // Even lanes do this
            data[global_idx] = global_idx * 2;
        } else {
            // Odd lanes do this - DIFFERENT CODE PATH!
            data[global_idx] = global_idx * 3 + 1;
        }
        
        // Print warp divergence info for first warp
        if (global_idx < 32) {
            printf("Thread %2d (Lane %2d): %s path, result=%d\n", 
                   global_idx, laneId, 
                   (laneId % 2 == 0) ? "EVEN" : "ODD", 
                   data[global_idx]);
        }
    }
}

// Better version - no warp divergence
__global__ void efficientVersion(int *data, int N) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (global_idx < N) {
        // GOOD: All threads in warp execute same instruction
        int base = global_idx * 2;
        int extra = (global_idx % 2);  // 0 for even, 1 for odd
        data[global_idx] = base + extra;
    }
}

void printGPUInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        return;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("=== GPU Hardware Information ===\n");
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Number of SMs: %d\n", prop.multiProcessorCount);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("Warp size: %d\n", prop.warpSize);
    printf("Shared memory per block: %d KB\n", prop.sharedMemPerBlock / 1024);
    printf("Shared memory per SM: %d KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("\n");
}

void explainHardwareMapping() {
    printf("=== CUDA LOGICAL MODEL vs HARDWARE MAPPING ===\n\n");
    
    printf("LOGICAL MODEL (What you program):\n");
    printf("Grid → Blocks → Threads\n");
    printf("├── Block 0: Threads 0-255\n");  
    printf("├── Block 1: Threads 0-255\n");
    printf("├── Block 2: Threads 0-255\n");
    printf("└── Block 3: Threads 0-255\n\n");
    
    printf("HARDWARE MODEL (How GPU executes):\n");
    printf("GPU → SMs → Warps → Threads\n");
    printf("├── SM 0: Warp 0 (Threads 0-31), Warp 1 (Threads 32-63), ...\n");
    printf("├── SM 1: Warp 8 (Threads 0-31), Warp 9 (Threads 32-63), ...\n");
    printf("├── SM 2: Warp 16 (Threads 0-31), Warp 17 (Threads 32-63), ...\n");
    printf("└── ...\n\n");
    
    printf("KEY MAPPING RULES:\n");
    printf("1. Each BLOCK is assigned to ONE SM\n");
    printf("2. Each SM can run MULTIPLE blocks (if resources allow)\n");
    printf("3. Threads in a block are grouped into WARPS of 32 threads\n");
    printf("4. All threads in a warp execute the SAME instruction (SIMT)\n");
    printf("5. SM schedules warps, not individual threads\n\n");
}

void demonstrateBlockToSMMapping() {
    printf("=== BLOCK-TO-SM MAPPING EXAMPLE ===\n");
    printf("Scenario: 8 blocks, 4 SMs available\n\n");
    
    printf("Initial assignment:\n");
    printf("SM 0: Block 0, Block 4\n");
    printf("SM 1: Block 1, Block 5  \n");
    printf("SM 2: Block 2, Block 6\n");
    printf("SM 3: Block 3, Block 7\n\n");
    
    printf("When Block 0 finishes on SM 0:\n");
    printf("SM 0: Block 4 (continues), [new block if available]\n");
    printf("SM 1: Block 1, Block 5\n");
    printf("SM 2: Block 2, Block 6\n");
    printf("SM 3: Block 3, Block 7\n\n");
    
    printf("This is why blocks must be INDEPENDENT!\n\n");
}

void demonstrateWarpMapping() {
    printf("=== WARP MAPPING WITHIN A BLOCK ===\n");
    printf("Block with 128 threads → 4 warps:\n\n");
    
    printf("Warp 0: Threads  0-31  (threadIdx.x:  0-31)\n");
    printf("Warp 1: Threads 32-63  (threadIdx.x: 32-63)\n");  
    printf("Warp 2: Threads 64-95  (threadIdx.x: 64-95)\n");
    printf("Warp 3: Threads 96-127 (threadIdx.x: 96-127)\n\n");
    
    printf("Formula: warpId = threadIdx.x / 32\n");
    printf("         laneId = threadIdx.x %% 32\n\n");
}

int main() {
    printf("=== CUDA HARDWARE MAPPING: BLOCKS, THREADS, SMs, and WARPS ===\n\n");
    
    // Show GPU hardware info
    printGPUInfo();
    
    // Explain the mapping concepts
    explainHardwareMapping();
    demonstrateBlockToSMMapping();
    demonstrateWarpMapping();
    
    // Demonstrate with actual kernel execution
    printf("=== PRACTICAL DEMONSTRATION ===\n");
    
    const int N = 128;
    const int threadsPerBlock = 64;  // 2 warps per block
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;  // 2 blocks
    
    printf("Configuration:\n");
    printf("- %d elements to process\n", N);
    printf("- %d threads per block (= %d warps per block)\n", threadsPerBlock, threadsPerBlock/32);
    printf("- %d blocks total\n", blocksPerGrid);
    printf("- Total: %d warps across all blocks\n\n", (blocksPerGrid * threadsPerBlock) / 32);
    
    int *d_data;
    cudaMalloc(&d_data, N * sizeof(int));
    
    printf("Launching kernel to show thread-to-warp mapping:\n");
    explainHardwareMapping<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);
    cudaDeviceSynchronize();
    
    printf("\n=== WARP DIVERGENCE DEMONSTRATION ===\n");
    printf("This shows why warp divergence hurts performance:\n\n");
    
    printf("BAD EXAMPLE (with divergence):\n");
    demonstrateWarpDivergence<<<1, 32>>>(d_data, 32);  // 1 warp
    cudaDeviceSynchronize();
    
    printf("\nGOOD EXAMPLE (no divergence):\n");
    printf("All threads execute same instructions → efficient!\n");
    efficientVersion<<<1, 32>>>(d_data, 32);
    cudaDeviceSynchronize();
    
    printf("\n=== KEY INSIGHTS FOR PERFORMANCE ===\n");
    printf("1. OCCUPANCY: Fill SMs with enough warps to hide latency\n");
    printf("2. DIVERGENCE: Avoid different code paths within a warp\n");
    printf("3. MEMORY: Coalesced access patterns work best\n");
    printf("4. BLOCK SIZE: Multiple of 32 (warp size) is most efficient\n");
    printf("5. RESOURCE LIMITS: Shared memory, registers limit occupancy\n\n");
    
    printf("OPTIMAL BLOCK SIZES:\n");
    printf("- Common choices: 128, 256, 512 threads per block\n");
    printf("- Always multiple of 32 (warp size)\n");
    printf("- Balance: enough warps vs resource usage\n");
    
    cudaFree(d_data);
    return 0;
}
