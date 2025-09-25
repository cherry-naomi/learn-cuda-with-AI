#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Kernel to demonstrate block isolation even on same SM
__global__ void demonstrateBlockIsolation(int *global_output) {
    // Each block has its OWN shared memory - completely separate!
    __shared__ int block_shared_data[32];
    __shared__ int block_id_marker;
    __shared__ int attempt_counter;
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Initialize this block's shared memory
    if (tid == 0) {
        block_id_marker = bid * 1000;  // Unique marker per block
        attempt_counter = 0;
        printf("Block %d initializing on SM (shared memory address space: Block_%d)\n", bid, bid);
    }
    
    __syncthreads();  // Sync within THIS block only
    
    // Each thread writes to shared memory
    if (tid < 32) {
        block_shared_data[tid] = block_id_marker + tid;
    }
    
    __syncthreads();
    
    // Try to "communicate" with other blocks (THIS WON'T WORK!)
    if (tid == 0) {
        printf("Block %d: My shared memory marker = %d\n", bid, block_id_marker);
        printf("Block %d: My shared_data[0] = %d\n", bid, block_shared_data[0]);
        
        // This is IMPOSSIBLE - we cannot access other blocks' shared memory!
        // Even if Block 0 and Block 1 are on the same SM, their shared memory is separate
        printf("Block %d: I CANNOT see other blocks' shared memory (even on same SM)\n", bid);
        
        // Only way to "communicate" is through global memory
        global_output[bid] = block_id_marker;
        attempt_counter++;
    }
    
    __syncthreads();
    
    // Show that only threads in THIS block can see the counter
    if (tid < 4) {
        printf("Block %d, Thread %d: I can see my block's attempt_counter = %d\n", 
               bid, tid, attempt_counter);
    }
}

// Kernel showing why block independence is crucial for scalability
__global__ void showWhyIndependenceMatters(int *data, int N) {
    __shared__ int local_sum;
    __shared__ int block_data[256];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int global_idx = bid * blockDim.x + tid;
    
    // Initialize shared memory for THIS block
    if (tid == 0) {
        local_sum = 0;
        printf("Block %d starting work (could be on any SM, any time)\n", bid);
    }
    
    __syncthreads();
    
    // Load data into shared memory
    if (global_idx < N) {
        block_data[tid] = data[global_idx];
    } else {
        block_data[tid] = 0;
    }
    
    __syncthreads();
    
    // Each block computes its own local sum independently
    atomicAdd(&local_sum, block_data[tid]);
    
    __syncthreads();
    
    // Store result - each block works independently
    if (tid == 0) {
        data[bid] = local_sum;  // Store block's result
        printf("Block %d finished: sum = %d (independent of other blocks)\n", bid, local_sum);
    }
}

// Function to explain the hardware reality
void explainBlockIsolation() {
    printf("=== BLOCK ISOLATION: THE HARDWARE REALITY ===\n\n");
    
    printf("SCENARIO: Multiple blocks on the same SM\n");
    printf("┌─────────────── SM 0 ──────────────┐\n");
    printf("│ ┌──Block 0──┐  ┌──Block 1──┐     │\n");
    printf("│ │SharedMem A│  │SharedMem B│     │\n");
    printf("│ │Thread 0-31│  │Thread 0-31│     │\n");
    printf("│ └───────────┘  └───────────┘     │\n");
    printf("│                                  │\n");
    printf("│ Hardware ensures COMPLETE        │\n");
    printf("│ ISOLATION between SharedMem A    │\n");
    printf("│ and SharedMem B                  │\n");
    printf("└──────────────────────────────────┘\n\n");
    
    printf("KEY FACTS:\n");
    printf("1. 🚫 Block 0 CANNOT access Block 1's shared memory\n");
    printf("2. 🚫 Block 0 CANNOT synchronize with Block 1\n");
    printf("3. 🚫 This is true EVEN IF they're on the same SM\n");
    printf("4. ✅ Only communication: Global memory\n");
    printf("5. ✅ This enables scalability across different SMs\n\n");
    
    printf("WHY THIS DESIGN?\n");
    printf("🎯 SCALABILITY: Blocks can run on ANY SM, in ANY order\n");
    printf("🎯 SIMPLICITY: No complex inter-block synchronization\n");
    printf("🎯 HARDWARE: SM can efficiently manage isolated blocks\n");
    printf("🎯 PORTABILITY: Code works on GPUs with different SM counts\n\n");
}

void demonstrateCommunicationLimitations() {
    printf("=== WHAT BLOCKS CANNOT DO ===\n\n");
    
    printf("❌ CANNOT: Share memory directly\n");
    printf("   Block A: __shared__ int data[100];\n");
    printf("   Block B: // Cannot access Block A's data[]\n\n");
    
    printf("❌ CANNOT: Synchronize with each other\n");
    printf("   Block A: __syncthreads();  // Only syncs Block A's threads\n");
    printf("   Block B: __syncthreads();  // Only syncs Block B's threads\n");
    printf("   // No way to sync Block A with Block B!\n\n");
    
    printf("❌ CANNOT: Know execution order\n");
    printf("   // Block 0 might finish before Block 1, or vice versa\n");
    printf("   // Block 2 might start before Block 1 finishes\n");
    printf("   // NO GUARANTEES about timing!\n\n");
    
    printf("✅ CAN DO: Communicate through global memory\n");
    printf("   __global__ void kernel(int *global_data) {\n");
    printf("       // All blocks can read/write global_data\n");
    printf("       global_data[blockIdx.x] = result;\n");
    printf("   }\n\n");
    
    printf("✅ CAN DO: Use atomic operations on global memory\n");
    printf("   atomicAdd(&global_counter, 1);  // Safe across blocks\n\n");
}

int main() {
    printf("=== CUDA BLOCK ISOLATION DEMONSTRATION ===\n\n");
    
    explainBlockIsolation();
    demonstrateCommunicationLimitations();
    
    // Practical demonstration
    printf("=== PRACTICAL DEMONSTRATION ===\n\n");
    
    int *d_output;
    cudaMalloc(&d_output, 4 * sizeof(int));
    
    printf("Launching 4 blocks on your GPU:\n");
    printf("(Multiple blocks may run on same SM, but they're still isolated)\n\n");
    
    // Launch kernel with multiple blocks
    demonstrateBlockIsolation<<<4, 32>>>(d_output);
    cudaDeviceSynchronize();
    
    // Get results to show they worked independently
    int h_output[4];
    cudaMemcpy(h_output, d_output, 4 * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("\nResults from each block (proving independence):\n");
    for (int i = 0; i < 4; i++) {
        printf("Block %d result: %d\n", i, h_output[i]);
    }
    
    printf("\n=== INDEPENDENCE SCALABILITY TEST ===\n\n");
    
    const int N = 1024;
    int *d_data;
    cudaMalloc(&d_data, N * sizeof(int));
    
    // Initialize data
    int *h_data = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        h_data[i] = i + 1;
    }
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    
    printf("Testing block independence with realistic workload:\n");
    showWhyIndependenceMatters<<<4, 256>>>(d_data, N);
    cudaDeviceSynchronize();
    
    printf("\n=== KEY INSIGHTS ===\n\n");
    
    printf("1. ISOLATION IS HARDWARE-ENFORCED\n");
    printf("   - Even on same SM, blocks have separate shared memory spaces\n");
    printf("   - GPU hardware prevents any cross-block access\n\n");
    
    printf("2. THIS ENABLES SCALABILITY\n");
    printf("   - Your code works on 1 SM or 100 SMs\n");
    printf("   - Blocks can be scheduled flexibly\n");
    printf("   - No complex synchronization needed\n\n");
    
    printf("3. ONLY GLOBAL MEMORY FOR COMMUNICATION\n");
    printf("   - Use global arrays for inter-block communication\n");
    printf("   - Use atomic operations for safe concurrent access\n");
    printf("   - Accept that blocks are independent units\n\n");
    
    printf("4. DESIGN IMPLICATION\n");
    printf("   - Design algorithms where blocks work independently\n");
    printf("   - Use multiple kernel launches if you need global coordination\n");
    printf("   - Embrace the parallel, independent block model\n\n");
    
    printf("BOTTOM LINE: Blocks are like independent teams - even if they're\n");
    printf("in the same building (SM), they have separate offices (shared memory)\n");
    printf("and can only communicate by leaving notes (global memory)!\n");
    
    // Cleanup
    cudaFree(d_output);
    cudaFree(d_data);
    free(h_data);
    
    return 0;
}
