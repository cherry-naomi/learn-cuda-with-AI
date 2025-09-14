#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// PROBLEM: Insufficient threads - some data won't be processed!
__global__ void vectorAddProblematic(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // This is the PROBLEM! No bounds checking means some elements never get processed
    c[idx] = a[idx] + b[idx];  // Only processes first 64 elements!
    
    // Let's print which elements are being processed
    if (idx < 10) {  // Only first 10 threads print
        printf("Thread %d processing element %d\n", idx, idx);
    }
}

// SOLUTION 1: Bounds checking (basic fix)
__global__ void vectorAddBoundsCheck(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // BOUNDS CHECK: Only process if within array bounds
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
        
        if (idx < 10) {
            printf("Bounds-checked thread %d processing element %d\n", idx, idx);
        }
    } else {
        // Thread has no work to do
        if (idx < 70) {  // Print for first few extra threads
            printf("Thread %d has no work (idx >= %d)\n", idx, n);
        }
    }
}

// SOLUTION 2: Grid-stride loop (each thread processes multiple elements)
__global__ void vectorAddGridStride(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;  // Total number of threads
    
    // Each thread processes multiple elements
    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
        
        if (i < 10 || (i >= 60 && i < 70) || (i >= 120 && i < 128)) {
            printf("Thread %d processing element %d (stride loop)\n", idx, i);
        }
    }
}

// SOLUTION 3: Multiple kernel launches
__global__ void vectorAddChunked(float *a, float *b, float *c, int n, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
        
        if (idx < offset + 10) {
            printf("Chunk kernel: thread %d processing element %d (offset %d)\n", 
                   blockIdx.x * blockDim.x + threadIdx.x, idx, offset);
        }
    }
}

// Helper function to check results
bool verifyResults(float *h_c, int n, const char* method) {
    printf("\nVerifying results for %s:\n", method);
    bool correct = true;
    int errors = 0;
    
    for (int i = 0; i < n; i++) {
        float expected = (float)i + (float)(i * 2);  // a[i] + b[i]
        if (fabs(h_c[i] - expected) > 1e-5) {
            if (errors < 10) {  // Show first 10 errors
                printf("  Error at index %d: expected %.1f, got %.1f\n", i, expected, h_c[i]);
            }
            correct = false;
            errors++;
        }
    }
    
    printf("  Result: %s", correct ? "CORRECT" : "INCORRECT");
    if (errors > 0) {
        printf(" (%d errors)", errors);
    }
    printf("\n");
    
    return correct;
}

void demonstrateInsufficientThreads() {
    printf("=== INSUFFICIENT THREADS PROBLEM DEMONSTRATION ===\n\n");
    
    printf("SCENARIO:\n");
    printf("- Workload: 128 elements to process\n");
    printf("- Available: 4 blocks × 16 threads = 64 threads\n");
    printf("- Problem: 64 elements won't be processed!\n\n");
    
    const int N = 128;
    const int blocks = 4;
    const int threads = 16;
    const int total_threads = blocks * threads;
    
    printf("Configuration:\n");
    printf("- N = %d elements\n", N);
    printf("- Blocks = %d\n", blocks);
    printf("- Threads per block = %d\n", threads);
    printf("- Total threads = %d\n", total_threads);
    printf("- Coverage = %.1f%% (%d/%d elements)\n\n", 
           (float)total_threads / N * 100, total_threads, N);
}

int main() {
    demonstrateInsufficientThreads();
    
    const int N = 128;
    const int blocks = 4;
    const int threads = 16;
    
    // Host data
    float *h_a = (float*)malloc(N * sizeof(float));
    float *h_b = (float*)malloc(N * sizeof(float));
    float *h_c = (float*)malloc(N * sizeof(float));
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
        h_c[i] = -999.0f;  // Initialize to obvious wrong value
    }
    
    // Device data
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));
    
    // Copy to device
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
    
    printf("=== DEMONSTRATION 1: PROBLEMATIC VERSION (NO BOUNDS CHECK) ===\n");
    printf("This will only process first %d elements!\n\n", blocks * threads);
    
    // Reset output array
    cudaMemcpy(d_c, h_c, N * sizeof(float), cudaMemcpyHostToDevice);
    
    vectorAddProblematic<<<blocks, threads>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    
    // Copy back and verify
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    verifyResults(h_c, N, "Problematic (no bounds check)");
    
    printf("\nFirst 10 elements: ");
    for (int i = 0; i < 10; i++) {
        printf("%.0f ", h_c[i]);
    }
    printf("\nLast 10 elements: ");
    for (int i = N-10; i < N; i++) {
        printf("%.0f ", h_c[i]);
    }
    printf(" (should be -999 = unprocessed)\n");
    
    printf("\n=== DEMONSTRATION 2: BOUNDS CHECK (STILL WRONG!) ===\n");
    printf("Bounds checking doesn't create more threads - still only processes 64 elements!\n\n");
    
    // Reset output array
    for (int i = 0; i < N; i++) h_c[i] = -999.0f;
    cudaMemcpy(d_c, h_c, N * sizeof(float), cudaMemcpyHostToDevice);
    
    vectorAddBoundsCheck<<<blocks, threads>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    verifyResults(h_c, N, "Bounds check (still insufficient threads)");
    
    printf("\n=== DEMONSTRATION 3: GRID-STRIDE LOOP SOLUTION ===\n");
    printf("Each thread processes multiple elements!\n\n");
    
    // Reset output array
    for (int i = 0; i < N; i++) h_c[i] = -999.0f;
    cudaMemcpy(d_c, h_c, N * sizeof(float), cudaMemcpyHostToDevice);
    
    vectorAddGridStride<<<blocks, threads>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    verifyResults(h_c, N, "Grid-stride loop");
    
    printf("\n=== DEMONSTRATION 4: MULTIPLE KERNEL LAUNCHES ===\n");
    printf("Launch kernel multiple times with different offsets!\n\n");
    
    // Reset output array
    for (int i = 0; i < N; i++) h_c[i] = -999.0f;
    cudaMemcpy(d_c, h_c, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch multiple times to cover all data
    int elements_per_launch = blocks * threads;
    int num_launches = (N + elements_per_launch - 1) / elements_per_launch;
    
    printf("Launching %d kernels to cover %d elements:\n", num_launches, N);
    for (int launch = 0; launch < num_launches; launch++) {
        int offset = launch * elements_per_launch;
        int remaining = N - offset;
        printf("  Launch %d: offset %d, processing %d elements\n", 
               launch, offset, min(elements_per_launch, remaining));
        
        vectorAddChunked<<<blocks, threads>>>(d_a, d_b, d_c, N, offset);
        cudaDeviceSynchronize();
    }
    
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    verifyResults(h_c, N, "Multiple kernel launches");
    
    printf("\n=== SOLUTION COMPARISON ===\n\n");
    
    printf("1. NO BOUNDS CHECK (❌ WRONG):\n");
    printf("   - Only processes first %d elements\n", blocks * threads);
    printf("   - %d elements remain unprocessed\n", N - blocks * threads);
    printf("   - Result: INCORRECT\n\n");
    
    printf("2. BOUNDS CHECK (❌ STILL WRONG with insufficient threads):\n");
    printf("   - Only processes first %d elements\n", blocks * threads);
    printf("   - %d elements remain unprocessed\n", N - blocks * threads);
    printf("   - Bounds check prevents crashes but doesn't create more threads!\n");
    printf("   - Result: INCORRECT\n\n");
    
    printf("3. GRID-STRIDE LOOP (✅ OPTIMAL):\n");
    printf("   - Each thread processes ~%.1f elements\n", (float)N / (blocks * threads));
    printf("   - All threads utilized efficiently\n");
    printf("   - Single kernel launch\n");
    printf("   - Result: CORRECT\n\n");
    
    printf("4. MULTIPLE LAUNCHES (✅ CORRECT):\n");
    printf("   - %d kernel launches needed\n", num_launches);
    printf("   - Higher overhead due to multiple launches\n");
    printf("   - Good for very different chunk sizes\n");
    printf("   - Result: CORRECT\n\n");
    
    printf("=== GENERAL GUIDELINES ===\n\n");
    
    printf("IMPORTANT: Bounds checking alone is NOT enough!\n");
    printf("if (idx < N) {  // Prevents crashes but doesn't solve insufficient threads!\n");
    printf("    // Only processes elements 0 to (total_threads - 1)\n");
    printf("}\n\n");
    
    printf("REAL SOLUTION: Ensure sufficient threads OR use grid-stride:\n");
    
    printf("For LARGE datasets, prefer grid-stride loops:\n");
    printf("int stride = gridDim.x * blockDim.x;\n");
    printf("for (int i = idx; i < N; i += stride) {\n");
    printf("    // Process element i\n");
    printf("}\n\n");
    
    printf("For OPTIMAL performance:\n");
    printf("- Use enough blocks to fill GPU\n");
    printf("- Use 256-512 threads per block\n");
    printf("- Let each thread process multiple elements if needed\n");
    printf("- Calculate grid size: (N + blockSize - 1) / blockSize\n\n");
    
    printf("OPTIMAL launch for %d elements:\n", N);
    int optimal_threads = 256;
    int optimal_blocks = (N + optimal_threads - 1) / optimal_threads;
    printf("vectorAdd<<<%d, %d>>>(d_a, d_b, d_c, %d);\n", optimal_blocks, optimal_threads, N);
    
    // Cleanup
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    return 0;
}
