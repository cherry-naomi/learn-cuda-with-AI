# CUDA Threading Concepts Explained

## The Theater Analogy 🎭

Think of CUDA threading like a movie theater:

```
GRID = The entire theater
├── BLOCK 0 = Row 1 (has multiple seats)
├── BLOCK 1 = Row 2 (has multiple seats)  
├── BLOCK 2 = Row 3 (has multiple seats)
└── BLOCK 3 = Row 4 (has multiple seats)
```

## Key Variables Explained

### 1. `threadIdx.x` - Your seat number within your row
- Range: 0 to (blockDim.x - 1)
- Example: If you're in seat 5 of your row, threadIdx.x = 5

### 2. `blockIdx.x` - Which row you're sitting in  
- Range: 0 to (gridDim.x - 1)
- Example: If you're in row 2, blockIdx.x = 2

### 3. `blockDim.x` - How many seats per row
- Set when launching the kernel
- Example: If each row has 8 seats, blockDim.x = 8

### 4. `gridDim.x` - How many rows in the theater
- Calculated based on how many blocks you launch
- Example: If theater has 4 rows, gridDim.x = 4

## Visual Example: 16 elements, 4 threads per block

```
Data to process: [0][1][2][3][4][5][6][7][8][9][10][11][12][13][14][15]

GRID (4 blocks total):
┌─────Block 0─────┬─────Block 1─────┬─────Block 2─────┬─────Block 3─────┐
│ Threads: 0 1 2 3│ Threads: 0 1 2 3│ Threads: 0 1 2 3│ Threads: 0 1 2 3│
│ Global:  0 1 2 3│ Global:  4 5 6 7│ Global:  8 9 10 │ Global: 12 13 14│
│                 │                 │         11      │         15      │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘

blockIdx.x:    0           1           2           3
blockDim.x:    4           4           4           4  
gridDim.x:     4           4           4           4
threadIdx.x: 0,1,2,3     0,1,2,3     0,1,2,3     0,1,2,3
```

## Global Index Calculation

The magic formula that maps each thread to unique data:

```c
int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
```

### Step by step for Thread in Block 1, Position 2:
1. `blockIdx.x = 1` (we're in block 1)
2. `blockDim.x = 4` (4 threads per block)  
3. `threadIdx.x = 2` (we're thread 2 within our block)
4. `global_idx = 1 * 4 + 2 = 6`

So this thread processes data element 6!

## Kernel Launch Syntax

```c
vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(args...);
           ^^^              ^^^
           |                |
           |                +-- Threads per block (blockDim.x)
           +-- Number of blocks (gridDim.x)
```

### Example Launch
```c
int N = 1024;                                    // Elements to process
int threadsPerBlock = 256;                       // Threads per block
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;  // = 4 blocks

vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
//         <<<4,            256>>>
```

This creates:
- 4 blocks
- 256 threads per block  
- 4 × 256 = 1024 total threads
- Perfect for processing 1024 elements!

## Why This Design?

### 1. **Scalability** 🚀
- More data? Launch more blocks
- GPU automatically distributes blocks across available cores

### 2. **Cooperation** 🤝  
- Threads in same block can:
  - Share fast shared memory
  - Synchronize with each other
  - Communicate efficiently

### 3. **Independence** 🎯
- Different blocks run independently
- No communication between blocks
- Perfect for parallel algorithms

## Common Patterns

### Pattern 1: One thread per element
```c
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < N) {
    output[idx] = input[idx] * 2;  // Each thread processes one element
}
```

### Pattern 2: Multiple elements per thread  
```c
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;  // Total number of threads

for (int i = idx; i < N; i += stride) {
    output[i] = input[i] * 2;  // Each thread processes multiple elements
}
```

## Memory Access Pattern

```
Thread 0 → data[0]
Thread 1 → data[1]  
Thread 2 → data[2]
Thread 3 → data[3]
...
Thread N-1 → data[N-1]
```

This creates **coalesced memory access** - very efficient on GPU! 🚄

## Quick Reference

| Variable | Meaning | Range | Set by |
|----------|---------|-------|--------|
| `threadIdx.x` | Thread position in block | 0 to blockDim.x-1 | Hardware |
| `blockIdx.x` | Block position in grid | 0 to gridDim.x-1 | Hardware |  
| `blockDim.x` | Threads per block | 1 to 1024 | You (launch) |
| `gridDim.x` | Blocks in grid | 1 to 65535 | You (launch) |

Remember: **Global Index = blockIdx.x × blockDim.x + threadIdx.x** 🔑
