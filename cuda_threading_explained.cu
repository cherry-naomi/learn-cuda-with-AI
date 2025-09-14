#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Educational kernel to demonstrate CUDA threading concepts
__global__ void explainThreading(int *data, int N) {
    // These are the key variables you need to understand:
    
    // threadIdx.x: Position of this thread within its block (0 to blockDim.x-1)
    // blockIdx.x:  Position of this block within the grid (0 to gridDim.x-1)
    // blockDim.x:  Number of threads per block (set when launching kernel)
    // gridDim.x:   Number of blocks in the grid (set when launching kernel)
    
    // Calculate the global thread index (unique ID for each thread)
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Print detailed information for first few threads
    if (global_idx < 20) {
        printf("Thread %2d: blockIdx=%d, threadIdx=%d, blockDim=%d, gridDim=%d\n",
               global_idx, blockIdx.x, threadIdx.x, blockDim.x, gridDim.x);
    }
    
    // Do the actual work (if within bounds)
    if (global_idx < N) {
        data[global_idx] = global_idx * 2;  // Simple computation
    }
}

void visualizeThreadLayout(int threadsPerBlock, int blocksPerGrid) {
    printf("\n=== VISUAL REPRESENTATION ===\n");
    printf("Configuration: %d threads per block, %d blocks\n", threadsPerBlock, blocksPerGrid);
    printf("Total threads: %d\n\n", threadsPerBlock * blocksPerGrid);
    
    printf("GRID (contains %d blocks):\n", blocksPerGrid);
    printf("┌");
    for (int b = 0; b < blocksPerGrid; b++) {
        printf("─────────Block %d─────────", b);
        if (b < blocksPerGrid - 1) printf("┬");
    }
    printf("┐\n");
    
    // Show thread layout within each block
    printf("│");
    for (int b = 0; b < blocksPerGrid; b++) {
        printf(" Threads: ");
        for (int t = 0; t < threadsPerBlock && t < 8; t++) {  // Show max 8 threads
            printf("%d ", t);
        }
        if (threadsPerBlock > 8) printf("...");
        printf("│");
    }
    printf("\n");
    
    printf("│");
    for (int b = 0; b < blocksPerGrid; b++) {
        printf(" Global:  ");
        for (int t = 0; t < threadsPerBlock && t < 8; t++) {
            printf("%d ", b * threadsPerBlock + t);
        }
        if (threadsPerBlock > 8) printf("...");
        printf("│");
    }
    printf("\n");
    
    printf("└");
    for (int b = 0; b < blocksPerGrid; b++) {
        for (int i = 0; i < 25; i++) printf("─");
        if (b < blocksPerGrid - 1) printf("┴");
    }
    printf("┘\n\n");
}

int main() {
    printf("=== CUDA THREADING CONCEPTS EXPLAINED ===\n\n");
    
    // Example 1: Small scale to understand the concepts
    printf("EXAMPLE 1: Small scale (16 elements, 4 threads per block)\n");
    printf("═══════════════════════════════════════════════════════\n");
    
    const int N1 = 16;
    const int threadsPerBlock1 = 4;
    const int blocksPerGrid1 = (N1 + threadsPerBlock1 - 1) / threadsPerBlock1;  // = 4 blocks
    
    printf("Problem: Process %d elements\n", N1);
    printf("Solution: Use %d threads per block\n", threadsPerBlock1);
    printf("Result: Need %d blocks (ceiling(%d/%d) = %d)\n\n", 
           blocksPerGrid1, N1, threadsPerBlock1, blocksPerGrid1);
    
    visualizeThreadLayout(threadsPerBlock1, blocksPerGrid1);
    
    // Allocate and run
    int *d_data1;
    cudaMalloc(&d_data1, N1 * sizeof(int));
    
    printf("Kernel launch syntax explanation:\n");
    printf("vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(args...)\n");
    printf("          ^^^                ^^^^\n");
    printf("          |                  |\n");
    printf("          |                  +-- How many threads per block\n");
    printf("          +-- How many blocks total\n\n");
    
    printf("For our example:\n");
    printf("explainThreading<<<%d, %d>>>(d_data1, %d)\n\n", 
           blocksPerGrid1, threadsPerBlock1, N1);
    
    explainThreading<<<blocksPerGrid1, threadsPerBlock1>>>(d_data1, N1);
    cudaDeviceSynchronize();
    
    printf("\n");
    printf("\nEXAMPLE 2: Realistic scale (1024 elements, 256 threads per block)\n");
    printf("════════════════════════════════════════════════════════════════\n");
    
    const int N2 = 1024;
    const int threadsPerBlock2 = 256;
    const int blocksPerGrid2 = (N2 + threadsPerBlock2 - 1) / threadsPerBlock2;  // = 4 blocks
    
    printf("Problem: Process %d elements\n", N2);
    printf("Solution: Use %d threads per block\n", threadsPerBlock2);
    printf("Result: Need %d blocks\n\n", blocksPerGrid2);
    
    int *d_data2;
    cudaMalloc(&d_data2, N2 * sizeof(int));
    
    printf("Launching: explainThreading<<<%d, %d>>>(d_data2, %d)\n", 
           blocksPerGrid2, threadsPerBlock2, N2);
    printf("This creates %d total threads to process %d elements\n\n", 
           blocksPerGrid2 * threadsPerBlock2, N2);
    
    explainThreading<<<blocksPerGrid2, threadsPerBlock2>>>(d_data2, N2);
    cudaDeviceSynchronize();
    
    printf("\n=== KEY INSIGHTS ===\n");
    printf("1. threadIdx.x: Your position within your block (like seat number)\n");
    printf("2. blockIdx.x:  Which block you're in (like row number)\n");
    printf("3. blockDim.x:  How many threads per block (like seats per row)\n");
    printf("4. gridDim.x:   How many blocks total (like total rows)\n\n");
    
    printf("5. Global index calculation:\n");
    printf("   idx = blockIdx.x * blockDim.x + threadIdx.x\n");
    printf("   Think: (row * seats_per_row) + seat_in_row = unique_seat_number\n\n");
    
    printf("6. Kernel launch <<<blocks, threads>>>:\n");
    printf("   - First number: How many blocks to create\n");
    printf("   - Second number: How many threads per block\n");
    printf("   - Total threads = blocks × threads_per_block\n\n");
    
    printf("7. Why this design?\n");
    printf("   - Threads in same block can share memory and synchronize\n");
    printf("   - Blocks run independently for scalability\n");
    printf("   - GPU hardware maps this efficiently to physical cores\n");
    
    // Cleanup
    cudaFree(d_data1);
    cudaFree(d_data2);
    
    return 0;
}
