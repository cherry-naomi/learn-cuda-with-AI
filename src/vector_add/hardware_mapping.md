# CUDA Hardware Mapping: Blocks, Threads, SMs, and Warps

## 🏗️ The Complete Picture: Logical Model → Hardware Reality

### Logical Model (What You Program)
```
GRID
├── BLOCK 0 (256 threads: 0-255)
├── BLOCK 1 (256 threads: 0-255)  
├── BLOCK 2 (256 threads: 0-255)
└── BLOCK 3 (256 threads: 0-255)
```

### Hardware Reality (How GPU Executes)
```
GPU
├── SM 0 
│   ├── WARP 0 (Block 0, Threads 0-31)
│   ├── WARP 1 (Block 0, Threads 32-63)
│   ├── WARP 2 (Block 1, Threads 0-31)
│   └── WARP 3 (Block 1, Threads 32-63)
├── SM 1
│   ├── WARP 0 (Block 2, Threads 0-31)
│   ├── WARP 1 (Block 2, Threads 32-63)
│   ├── WARP 2 (Block 3, Threads 0-31)
│   └── WARP 3 (Block 3, Threads 32-63)
└── ...
```

## 🎯 Key Hardware Components

### 1. **SM (Streaming Multiprocessor)** 
- The **actual compute unit** on the GPU
- Modern GPUs have 10s to 100s of SMs
- Each SM can execute multiple warps concurrently
- **Analogy**: Think of SMs as individual CPU cores, but much more powerful

### 2. **Warp**
- **Fundamental execution unit** = 32 threads that execute together
- All threads in a warp execute the **same instruction** (SIMT - Single Instruction, Multiple Thread)
- **Hardware reality**: GPU doesn't schedule individual threads, it schedules warps
- **Analogy**: Like a marching band - all 32 musicians play the same note at the same time

### 3. **Block-to-SM Mapping**
- Each **block is assigned to exactly ONE SM**
- Each **SM can run multiple blocks** (if it has enough resources)
- Blocks are **independent** - they can run on different SMs without communication

## 📐 Warp Formation Rules

### How Threads Become Warps
```c
// For any block, threads are grouped into warps of 32:
warpId = threadIdx.x / 32;
laneId = threadIdx.x % 32;  // Position within warp (0-31)
```

### Example: Block with 128 threads
```
Block (128 threads)
├── Warp 0: Threads  0-31  (threadIdx.x:  0-31)
├── Warp 1: Threads 32-63  (threadIdx.x: 32-63)
├── Warp 2: Threads 64-95  (threadIdx.x: 64-95)
└── Warp 3: Threads 96-127 (threadIdx.x: 96-127)
```

## 🎪 SM Resource Management

### What Limits How Many Blocks Fit on an SM?

1. **Max threads per SM** (e.g., 2048 threads)
2. **Max blocks per SM** (e.g., 32 blocks)  
3. **Shared memory per SM** (e.g., 96 KB)
4. **Registers per SM** (e.g., 65536 registers)

### Example Calculation
```
GPU: RTX 3080 (example)
- Max threads per SM: 1536
- Block size: 256 threads
- Max blocks on one SM: 1536 ÷ 256 = 6 blocks
```

## ⚡ Performance Implications

### 1. **Warp Divergence** (PERFORMANCE KILLER! 💀)

**BAD Example:**
```c
if (threadIdx.x % 2 == 0) {
    result = data[idx] * 2;     // Even threads do this
} else {
    result = data[idx] * 3;     // Odd threads do this - DIFFERENT PATH!
}
```

**What happens:** Warp executes BOTH paths sequentially!
- First: Execute path 1 (even threads active, odd threads idle)
- Then: Execute path 2 (odd threads active, even threads idle)
- **Result: 2× slower!**

**GOOD Example:**
```c
// All threads execute same instructions
int multiplier = 2 + (threadIdx.x % 2);  // 2 or 3
result = data[idx] * multiplier;
```

### 2. **Occupancy** (Keeping SMs Busy)

**Low Occupancy (Bad):**
```
SM has capacity for 1536 threads
Your blocks use 1024 threads each
1536 ÷ 1024 = 1.5 → Only 1 block fits
Only 1024/1536 = 67% of SM utilized
```

**High Occupancy (Good):**
```  
SM has capacity for 1536 threads
Your blocks use 256 threads each
1536 ÷ 256 = 6 blocks fit perfectly
100% of SM utilized
```

### 3. **Memory Coalescing**

**Efficient Pattern (Coalesced):**
```
Thread 0 → Memory[0]
Thread 1 → Memory[1]  
Thread 2 → Memory[2]
Thread 3 → Memory[3]
...
```
All threads in warp access consecutive memory → **1 transaction**

**Inefficient Pattern (Scattered):**
```
Thread 0 → Memory[0]
Thread 1 → Memory[100]
Thread 2 → Memory[200]  
Thread 3 → Memory[300]
...
```
Random access pattern → **32 separate transactions**

## 🎯 Block-to-SM Assignment Example

### Scenario: 8 Blocks, 4 SMs Available

**Initial Assignment:**
```
SM 0: [Block 0] [Block 4]
SM 1: [Block 1] [Block 5]
SM 2: [Block 2] [Block 6]  
SM 3: [Block 3] [Block 7]
```

**When Block 0 Finishes:**
```
SM 0: [Block 4] [New Block if available]
SM 1: [Block 1] [Block 5]
SM 2: [Block 2] [Block 6]
SM 3: [Block 3] [Block 7]
```

**Key Insight:** This is why blocks must be **independent**! They can finish in any order.

## 🛠️ Optimal Configuration Guidelines

### Block Size Rules
```
✅ GOOD block sizes: 128, 256, 512 threads
✅ Always multiple of 32 (warp size)
✅ Balance: enough warps vs resource usage

❌ AVOID: 31, 33, 100, 200 (not warp-aligned)
❌ AVOID: Very small (< 64) or very large (> 1024)
```

### Grid Size Rules
```
✅ GOOD: More blocks than SMs (for load balancing)
✅ GOOD: Blocks >> SMs for hiding latency
✅ Example: 1000 blocks on 80 SMs = good distribution

❌ AVOID: Fewer blocks than SMs (underutilization)
```

## 🔍 Debugging Tools

### Check Occupancy
```bash
nvprof --metrics achieved_occupancy ./your_program
```

### Check Warp Efficiency  
```bash
nvprof --metrics warp_execution_efficiency ./your_program
```

### Visual Profiler
```bash
nvvp ./your_program  # See SM utilization, warp efficiency
```

## 💡 Mental Model Summary

Think of it like a **factory production line**:

- **SM** = Factory floor
- **Warp** = Production team (32 workers who must do the same task)
- **Block** = Job order (assigned to one factory floor)
- **Thread** = Individual worker

**Key Rules:**
1. All workers in a team (warp) must do the same task simultaneously
2. Job orders (blocks) can't be split between factory floors (SMs)
3. Factory floors (SMs) try to keep multiple teams (warps) busy
4. If teams do different tasks (divergence), they must work sequentially → slower

**Performance Tips:**
1. Keep factory floors busy (high occupancy)
2. Avoid teams doing different tasks (avoid divergence)
3. Use efficient material handling (coalesced memory access)
4. Right team size (block size multiple of 32)

This hardware understanding is crucial for writing high-performance CUDA code! 🚀
