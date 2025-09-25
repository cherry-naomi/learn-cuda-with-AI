# CUDA Vector Addition Performance Guide

## 🚀 How to Profile and Optimize Vector Addition

### Quick Start

```bash
# Build all examples
make all

# Run performance analysis
make run7

# Run comprehensive profiling  
make profile
```

## 📊 Performance Analysis Tools

### 1. **Built-in Timing (Recommended for Learning)**
```bash
./vector_add_optimized
```
This shows:
- GPU device information
- Memory bandwidth utilization
- Different optimization techniques
- Block size optimization
- Kernel variants comparison

### 2. **NVPROF (Legacy Profiler)**
```bash
# Basic profiling
nvprof ./vector_add

# Detailed metrics
nvprof --metrics achieved_occupancy,gld_efficiency,gst_efficiency ./vector_add

# Memory bandwidth
nvprof --metrics dram_read_throughput,dram_write_throughput ./vector_add
```

### 3. **Nsight Compute (Modern Profiler)**
```bash
# Interactive analysis
ncu ./vector_add

# Full metric set
ncu --set full ./vector_add

# Memory-focused analysis
ncu --set memory ./vector_add

# Specific metrics
ncu --metrics sm__cycles_elapsed.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed ./vector_add
```

### 4. **Nsight Systems (Timeline Profiler)**
```bash
# Create timeline profile
nsys profile --output=timeline ./vector_add

# Open in GUI
nsys-ui timeline.nsys-rep
```

## 🎯 Key Performance Metrics

### Memory Bandwidth (Most Important for Vector Add)
```
Target: >80% of peak GPU memory bandwidth

Formula: (3 * N * sizeof(float)) / execution_time
- Reads: 2 arrays (A and B)
- Writes: 1 array (C)
- Total: 3 * N * 4 bytes

Example:
- 256M elements: 3 * 256M * 4 = 3 GB
- 10ms execution: 3 GB / 0.01s = 300 GB/s
```

### Occupancy
```
Target: >50% occupancy

What it measures:
- Ratio of active warps to maximum possible warps
- Higher occupancy = better latency hiding
- But diminishing returns beyond ~50%
```

### Memory Efficiency
```
Target: >90% efficiency

Global Load Efficiency:
- Measures coalesced memory access
- 100% = perfect coalescing
- <90% = scattered access pattern

Global Store Efficiency:
- Same for writes
- Vector add should achieve ~100%
```

## ⚡ Optimization Techniques

### 1. **Block Size Optimization**
```c
// Test different block sizes
int block_sizes[] = {128, 256, 512, 1024};

// Optimal for most GPUs: 256-512 threads
// Must be multiple of 32 (warp size)
```

**Why 256-512 is optimal:**
- Enough warps to hide memory latency
- Good resource utilization
- Not too large to limit occupancy

### 2. **Vectorized Memory Access**
```c
// Instead of float
__global__ void vectorAddBasic(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];  // 4 bytes per thread
    }
}

// Use float4 for 4x throughput
__global__ void vectorAddVectorized(float4 *a, float4 *b, float4 *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float4 va = a[idx];       // 16 bytes per thread
        float4 vb = b[idx];
        float4 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;
        c[idx] = vc;
    }
}
```

### 3. **Grid-Stride Loops**
```c
// For large datasets
__global__ void vectorAddGridStride(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    // Each thread processes multiple elements
    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}
```

**Benefits:**
- Better occupancy for large datasets
- Fewer blocks needed
- More flexible grid sizing

### 4. **Memory Alignment**
```c
// Ensure data is aligned for optimal access
float *aligned_malloc(size_t size) {
    void *ptr;
    posix_memalign(&ptr, 128, size);  // 128-byte alignment
    return (float*)ptr;
}
```

## 📈 Performance Expectations

### Theoretical Limits
```
RTX 3080 Example:
- Memory Bandwidth: ~760 GB/s
- Vector Add Transfer: 3 * N * 4 bytes
- Theoretical Min Time: (3 * N * 4) / 760e9

For 256M elements:
- Data Transfer: 3 GB
- Min Time: 3.95 ms
- Achievable: ~80% = 4.9 ms
```

### Real-World Results
```
GPU Class          | Expected Bandwidth | Vector Add Performance
-------------------|-------------------|----------------------
RTX 30xx/40xx     | 600-900 GB/s      | 5-15 ms (256M elements)
RTX 20xx          | 400-600 GB/s      | 8-20 ms
GTX 16xx          | 300-450 GB/s      | 12-25 ms
Older GPUs        | 150-300 GB/s      | 20-50 ms
```

## 🛠️ Optimization Workflow

### Step 1: Baseline Measurement
```bash
# Run basic version
./vector_add

# Note execution time and check correctness
```

### Step 2: Profile with Tools
```bash
# Quick profiling
nvprof ./vector_add

# Look for:
# - Kernel execution time
# - Memory bandwidth utilization
# - Occupancy
```

### Step 3: Optimize Block Size
```bash
# Test different configurations
./vector_add_optimized

# Find optimal threads per block for your GPU
```

### Step 4: Apply Advanced Optimizations
```c
// Try in order:
1. Vectorized loads (float4)
2. Grid-stride loops
3. Memory alignment
4. Multiple streams (for very large data)
```

### Step 5: Verify and Measure
```bash
# Comprehensive analysis
ncu --set memory ./vector_add_optimized

# Look for:
# - Memory efficiency >90%
# - Bandwidth utilization >80%
# - No warp divergence
```

## 🚫 Common Performance Pitfalls

### 1. **Wrong Block Size**
```c
❌ BAD: <<<N, 1>>>        // Too few threads
❌ BAD: <<<1, N>>>        // Single block
❌ BAD: <<<N/100, 100>>>  // Not warp-aligned

✅ GOOD: <<<(N+255)/256, 256>>>  // Optimal
```

### 2. **Memory Access Pattern**
```c
❌ BAD: Scattered access
for (int i = 0; i < N; i++) {
    c[random_indices[i]] = a[i] + b[i];  // Random writes
}

✅ GOOD: Sequential access
int idx = blockIdx.x * blockDim.x + threadIdx.x;
c[idx] = a[idx] + b[idx];  // Coalesced
```

### 3. **Unnecessary Shared Memory**
```c
❌ BAD: Shared memory without reuse
__shared__ float temp[256];
temp[tid] = a[global_idx];
__syncthreads();
c[global_idx] = temp[tid] + b[global_idx];  // No benefit!

✅ GOOD: Direct global access
c[global_idx] = a[global_idx] + b[global_idx];
```

### 4. **Poor Occupancy**
```c
❌ BAD: Large shared memory per block
__shared__ float big_array[8192];  // Limits occupancy

✅ GOOD: Reasonable resource usage
__shared__ float small_array[256]; // If needed at all
```

## 🎯 GPU-Specific Tips

### Ampere (RTX 30xx/40xx)
- Excellent memory bandwidth
- Use float4 vectorization
- Target 512 threads per block
- Focus on memory coalescing

### Turing (RTX 20xx)
- Good memory bandwidth
- 256-512 threads per block
- Vectorization helps significantly
- Watch out for occupancy limits

### Pascal (GTX 10xx)
- Memory bandwidth limited
- 256 threads per block often optimal
- Coalescing is critical
- Grid-stride loops beneficial

### Maxwell/Kepler (Older)
- Lower memory bandwidth
- 128-256 threads per block
- Focus on occupancy
- Simpler is often better

## 🔧 Debugging Poor Performance

### Memory Bandwidth Too Low?
```bash
# Check memory efficiency
ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.pct_of_peak_sustained_elapsed ./vector_add

# If low (<80%):
# 1. Check data alignment
# 2. Verify coalesced access
# 3. Try vectorized loads
```

### Low Occupancy?
```bash
# Check occupancy
nvprof --metrics achieved_occupancy ./vector_add

# If low (<50%):
# 1. Reduce threads per block
# 2. Reduce shared memory usage
# 3. Reduce register usage
```

### Kernel Too Slow?
```bash
# Profile detailed timing
ncu --metrics sm__cycles_elapsed.avg ./vector_add

# Compare with theoretical minimum:
# Min cycles = (data_bytes / memory_bandwidth) * clock_rate
```

## 📚 Advanced Topics

### Multiple GPU Streams
```c
// For very large datasets
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Overlap computation and memory transfer
```

### Memory Types
```c
// Pinned memory for faster transfers
cudaMallocHost(&h_a, size);

// Managed memory for simplicity
cudaMallocManaged(&a, size);
```

### Multiple GPUs
```c
// Scale across multiple GPUs
for (int gpu = 0; gpu < num_gpus; gpu++) {
    cudaSetDevice(gpu);
    // Launch on each GPU
}
```

## 🎓 Next Steps

1. **Master the Basics**: Get >80% memory bandwidth on vector add
2. **Learn Matrix Multiply**: Understand shared memory optimization
3. **Study Reductions**: Learn synchronization patterns
4. **Explore Libraries**: cuBLAS, Thrust, CUB for optimized algorithms
5. **Advanced Profiling**: Memory hierarchy analysis, instruction-level optimization

## 🏆 Performance Checklist

- [ ] Memory bandwidth >80% of peak
- [ ] Occupancy >50%
- [ ] Global memory efficiency >90%
- [ ] Block size multiple of 32
- [ ] No warp divergence
- [ ] Coalesced memory access
- [ ] Correct answer verification
- [ ] Profiling data collected
- [ ] Optimization techniques tested
- [ ] Performance compared to theoretical

Remember: Vector addition is **memory-bound**, so focus on memory optimization, not computation! 🚀
