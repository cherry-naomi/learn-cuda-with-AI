# NVIDIA Nsight Compute (ncu) 使用指南

## 1. 基本概念

NVIDIA Nsight Compute (ncu) 是专门用于分析 CUDA 内核详细硬件指标的工具，可以分析：

- **DRAM 带宽利用率**: `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- **内存访问模式**: `dram__bytes_read.sum`, `dram__bytes_write.sum`
- **计算利用率**: `sm__throughput.avg.pct_of_peak_sustained_elapsed`
- **Warp 效率**: `smsp__warps_active.avg.pct_of_peak_sustained_elapsed`
- **寄存器使用**: `launch__registers_per_thread`
- **共享内存使用**: `launch__shared_mem_per_block`

## 2. 权限设置

### 解决权限问题

如果遇到 `ERR_NVGPUCTRPERM` 错误，需要设置权限：

```bash
# 方法1: 设置设备权限
sudo nvidia-ml-py3 -c "import pynvml; pynvml.nvmlInit(); print('Permission set')"

# 方法2: 使用 sudo 运行
sudo ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./your_program

# 方法3: 设置环境变量
export CUDA_PROFILER_TIMESTAMP=1
sudo -E ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./your_program
```

### 检查权限状态

```bash
# 检查当前权限
nvidia-smi -q -d SUPPORTED_CLOCKS

# 或者
nvidia-smi -q -d PERFORMANCE
```

## 3. 基本用法

### 语法格式
```bash
ncu [选项] --metrics <指标列表> <可执行文件> [程序参数]
```

### 常用选项
- `--metrics`: 指定要收集的指标
- `--target-processes all`: 分析所有进程
- `--kernel-name <name>`: 只分析特定内核
- `--kernel-base-name <name>`: 分析匹配基名的内核
- `--launch-skip <count>`: 跳过前几次内核启动
- `--launch-count <count>`: 只分析指定次数的内核启动

## 4. 重要的硬件指标

### DRAM 带宽指标
```bash
# DRAM 带宽利用率（百分比）
dram__throughput.avg.pct_of_peak_sustained_elapsed

# DRAM 读取字节数
dram__bytes_read.sum

# DRAM 写入字节数  
dram__bytes_write.sum

# DRAM 读取吞吐量 (GB/s)
dram__throughput.avg.pct_of_peak_sustained_elapsed

# DRAM 延迟
dram__latency.avg.pct_of_peak_sustained_elapsed
```

### 计算利用率指标
```bash
# SM 计算利用率
sm__throughput.avg.pct_of_peak_sustained_elapsed

# Warp 活跃度
smsp__warps_active.avg.pct_of_peak_sustained_elapsed

# 指令吞吐量
smsp__inst_executed.avg.per_cycle_elapsed

# 算术指令利用率
smsp__inst_executed.avg.pct_of_peak_sustained_elapsed
```

### 内存访问模式指标
```bash
# 全局内存访问
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum

# 共享内存访问
l1tex__t_sectors_pipe_lsu_mem_shared_op_ld.sum
l1tex__t_sectors_pipe_lsu_mem_shared_op_st.sum

# L1 缓存命中率
l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum
l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum
```

## 5. 实际使用示例

### 分析 softmax_optimized 程序

```bash
# 基本 DRAM 带宽分析
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum ./softmax_optimized

# 详细的内存分析
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum ./softmax_optimized

# 计算利用率分析
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,smsp__warps_active.avg.pct_of_peak_sustained_elapsed ./softmax_optimized

# 综合分析（推荐）
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,smsp__warps_active.avg.pct_of_peak_sustained_elapsed,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum ./softmax_optimized
```

### 只分析特定内核

```bash
# 只分析 softmax_optimized 内核
ncu --kernel-base-name softmax_optimized --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./softmax_optimized

# 只分析 warp primitives 内核
ncu --kernel-base-name softmax_warp_primitives --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./softmax_optimized
```

### 跳过初始启动

```bash
# 跳过前5次内核启动，只分析后面的
ncu --launch-skip 5 --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./softmax_optimized
```

## 6. 输出结果解读

### DRAM 带宽利用率示例
```
dram__throughput.avg.pct_of_peak_sustained_elapsed
  softmax_optimized(float *, float *, int, int), 2024-09-19 02:45:12
    (100.00%) 45.23%
```

**解读**:
- `45.23%`: 实际 DRAM 带宽利用率
- 表示只使用了峰值带宽的 45.23%
- 有优化空间，可以尝试提高内存访问效率

### 计算利用率示例
```
sm__throughput.avg.pct_of_peak_sustained_elapsed
  softmax_optimized(float *, float *, int, int), 2024-09-19 02:45:12
    (100.00%) 12.34%
```

**解读**:
- `12.34%`: SM 计算利用率
- 表示计算单元利用率较低
- 可能需要优化算法或增加计算密度

## 7. 性能优化建议

### 基于 DRAM 带宽的优化

1. **低带宽利用率 (< 50%)**:
   - 检查内存访问模式
   - 使用内存合并访问
   - 考虑使用共享内存
   - 增加计算密度

2. **高带宽利用率 (> 80%)**:
   - 内存带宽接近瓶颈
   - 考虑算法优化
   - 减少内存传输
   - 使用流式处理

### 基于计算利用率的优化

1. **低计算利用率 (< 30%)**:
   - 增加计算密度
   - 减少分支分歧
   - 优化线程配置
   - 使用更高效的算法

2. **高计算利用率 (> 70%)**:
   - 计算单元充分利用
   - 关注内存访问优化
   - 考虑数据并行化

## 8. 批量分析脚本

创建一个分析脚本来比较不同内核：

```bash
#!/bin/bash
# analyze_kernels.sh

echo "=== CUDA 内核性能分析 ==="

# 分析不同内核的 DRAM 带宽利用率
echo "1. 分析 softmax_optimized 内核"
ncu --kernel-base-name softmax_optimized \
    --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed \
    ./softmax_optimized > analysis_optimized.txt 2>&1

echo "2. 分析 softmax_warp_primitives 内核"
ncu --kernel-base-name softmax_warp_primitives \
    --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed \
    ./softmax_optimized > analysis_warp.txt 2>&1

echo "3. 分析 softmax_fused 内核"
ncu --kernel-base-name softmax_fused \
    --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed \
    ./softmax_optimized > analysis_fused.txt 2>&1

echo "分析完成！查看 *_analysis.txt 文件获取结果。"
```

## 9. 与其他工具结合使用

### 结合 nsys 使用
```bash
# 先用 nsys 获取整体性能
nsys profile -o overall_profile ./softmax_optimized

# 再用 ncu 分析特定内核的详细指标
ncu --kernel-base-name softmax_optimized \
    --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed \
    ./softmax_optimized
```

### 生成报告
```bash
# 生成 HTML 报告
ncu --export analysis_report.html \
    --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed \
    ./softmax_optimized
```

## 10. 常见问题解决

### 权限问题
```bash
# 检查权限
nvidia-smi -q -d SUPPORTED_CLOCKS

# 如果失败，使用 sudo
sudo ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./program
```

### 没有内核被分析
```bash
# 使用 --target-processes all
ncu --target-processes all --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./program

# 或者指定内核名称
ncu --kernel-base-name your_kernel_name --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./program
```

### 指标不存在
```bash
# 查看可用指标
ncu --list-metrics

# 查看特定指标详情
ncu --query-metrics dram__throughput.avg.pct_of_peak_sustained_elapsed
```

## 11. 总结

ncu 是分析 CUDA 内核硬件性能的强有力工具，特别适合：

1. **精确的带宽分析**: 了解实际 DRAM 使用情况
2. **计算利用率分析**: 评估 SM 使用效率
3. **内存访问优化**: 识别内存访问瓶颈
4. **内核比较**: 对比不同实现的硬件效率

结合 nsys 和 ncu，可以全面了解 CUDA 程序的性能特征。

