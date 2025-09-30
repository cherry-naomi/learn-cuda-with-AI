#!/usr/bin/env python3
"""
Triton Softmax Kernel Performance Test
Compare with CUDA implementations
"""

import torch
import triton
import triton.language as tl
import time
import numpy as np

@triton.jit
def triton_softmax_kernel(
    in_ptr, in_row_stride: tl.constexpr,
    out_ptr, out_row_stride: tl.constexpr,
    n_rows: tl.constexpr, n_cols: tl.constexpr,
    block_size: tl.constexpr,
):
    """Triton Softmax kernel implementation"""
    row_idx = tl.program_id(0)  # Index of the row to process
    
    # Load a whole row of the input
    col_offsets = tl.arange(0, block_size)  # We expect block_size >= n_cols
    mask = col_offsets < n_cols
    
    # Compute input row pointers
    in_row_ptrs = in_ptr + row_idx * in_row_stride + col_offsets
    row = tl.load(in_row_ptrs, mask=mask, other=float("-inf"))

    # 1. Max reduction over a row
    row_max = tl.max(row, axis=0)

    # 2. Stable exponential of a row
    numerator = tl.exp(row - row_max)
    
    # 3. Sum reduction of the exponentiated row
    denominator = tl.sum(numerator, axis=0)
    
    # 4. Compute and store the result of softmax
    res = numerator / denominator
    
    # Compute output row pointers
    out_row_ptrs = out_ptr + row_idx * out_row_stride + col_offsets
    tl.store(out_row_ptrs, res, mask=mask)

def triton_softmax(x):
    """Triton Softmax wrapper function"""
    n_rows, n_cols = x.shape
    block_size = triton.next_power_of_2(n_cols)
    
    # Allocate output tensor
    y = torch.empty_like(x)
    
    # Launch kernel
    grid = (n_rows,)
    triton_softmax_kernel[grid](
        x, x.stride(0),
        y, y.stride(0),
        n_rows, n_cols,
        block_size=block_size,
    )
    
    return y

def torch_softmax(x):
    """PyTorch reference implementation"""
    return torch.softmax(x, dim=-1)

def benchmark_triton_softmax(batch_size, dim, num_iterations=100):
    """Benchmark Triton Softmax implementation"""
    print(f"=== Triton Softmax Benchmark ===")
    print(f"Shape: [{batch_size}, {dim}]")
    print(f"Iterations: {num_iterations}")
    
    # Create random input data
    x = torch.randn(batch_size, dim, device='cuda', dtype=torch.float32)
    
    # Warmup
    for _ in range(10):
        _ = triton_softmax(x)
    torch.cuda.synchronize()
    
    # Benchmark Triton
    start_time = time.time()
    for _ in range(num_iterations):
        triton_result = triton_softmax(x)
    torch.cuda.synchronize()
    triton_time = (time.time() - start_time) / num_iterations * 1000  # Convert to ms
    
    # Benchmark PyTorch
    start_time = time.time()
    for _ in range(num_iterations):
        torch_result = torch_softmax(x)
    torch.cuda.synchronize()
    torch_time = (time.time() - start_time) / num_iterations * 1000  # Convert to ms
    
    # Verify correctness
    diff = torch.abs(triton_result - torch_result).max().item()
    is_correct = diff < 1e-3
    
    # Calculate memory bandwidth
    memory_size = x.numel() * 4 * 2  # 4 bytes per float, 2x for read+write
    memory_size_gb = memory_size / (1024**3)
    triton_bandwidth = memory_size_gb / (triton_time / 1000)  # GB/s
    
    print(f"Triton Softmax Time: {triton_time:.4f} ms")
    print(f"PyTorch Softmax Time: {torch_time:.4f} ms")
    print(f"Triton Speedup: {torch_time / triton_time:.2f}x")
    print(f"Triton Memory Bandwidth: {triton_bandwidth:.2f} GB/s")
    print(f"Correctness Check: {'✅ PASS' if is_correct else '❌ FAIL'} (max diff: {diff:.2e})")
    print()
    
    return {
        'triton_time_ms': triton_time,
        'torch_time_ms': torch_time,
        'speedup': torch_time / triton_time,
        'bandwidth_gb_s': triton_bandwidth,
        'correct': is_correct,
        'max_diff': diff
    }

def main():
    """Main benchmark function"""
    print("🚀 Triton Softmax Kernel Performance Test")
    print("=" * 60)
    
    # Test cases matching CUDA benchmark
    test_cases = [
        (49152, 128),
        (49152, 1024),
        (49152, 16384),
    ]
    
    results = []
    
    for batch_size, dim in test_cases:
        try:
            result = benchmark_triton_softmax(batch_size, dim, num_iterations=100)
            result['batch_size'] = batch_size
            result['dim'] = dim
            results.append(result)
        except Exception as e:
            print(f"❌ Error testing [{batch_size}, {dim}]: {e}")
            print()
    
    # Print summary table
    print("=" * 80)
    print("PERFORMANCE SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Shape':<20} {'Triton(ms)':<12} {'Torch(ms)':<12} {'Speedup':<10} {'Bandwidth(GB/s)':<15} {'Status':<10}")
    print("-" * 80)
    
    for result in results:
        shape_str = f"[{result['batch_size']}, {result['dim']}]"
        status = "✅ PASS" if result['correct'] else "❌ FAIL"
        print(f"{shape_str:<20} {result['triton_time_ms']:<12.4f} {result['torch_time_ms']:<12.4f} "
              f"{result['speedup']:<10.2f} {result['bandwidth_gb_s']:<15.2f} {status:<10}")
    
    print("=" * 80)
    
    # Compare with CUDA results (from previous benchmarks)
    print("\n📊 Comparison with CUDA Implementations:")
    print("Based on previous CUDA benchmarks on A100:")
    print()
    print("Dim=128:")
    print("  CUDA Warp Shuffle:    0.11 ms  (405.2x speedup)")
    print("  CUDA Shared Memory:   0.56 ms  (81.9x speedup)")
    print("  Triton:               See above results")
    print()
    print("Dim=1024:")
    print("  CUDA Warp Shuffle:    1.16 ms  (312.8x speedup)")
    print("  CUDA Shared Memory:   0.69 ms  (526.3x speedup)")
    print("  Triton:               See above results")
    print()
    print("Dim=16384:")
    print("  CUDA Shared Memory:   10.33 ms (575.9x speedup)")
    print("  Triton:               See above results")

if __name__ == "__main__":
    main()
