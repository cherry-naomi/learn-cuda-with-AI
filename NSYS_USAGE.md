# NVIDIA Nsight Systems (nsys) 使用指南

## 1. 基本概念

nsys 是 NVIDIA 提供的系统级性能分析工具，可以分析：
- CUDA 内核执行时间
- 内存传输性能
- CPU-GPU 同步开销
- API 调用时间
- 系统调用分析

## 2. 基本语法

```bash
nsys profile [选项] <可执行文件> [程序参数]
```

## 3. 常用选项

### 输出控制
- `-o <filename>`: 指定输出文件名（不包含扩展名）
- `--force-overwrite true`: 强制覆盖已存在的输出文件

### 数据收集控制
- `--trace=cuda`: 只收集 CUDA 相关数据（默认包含）
- `--trace=cuda,nvtx`: 收集 CUDA 和 NVTX 标记数据
- `--trace=cuda,osrt`: 收集 CUDA 和操作系统运行时数据
- `--trace=cuda,osrt,nvtx`: 收集所有数据

### 采样控制
- `--sample=none`: 不进行 CPU 采样
- `--sample=cpu`: 进行 CPU 采样（默认）
- `--cuda-memory-usage true`: 跟踪 CUDA 内存使用

### 性能开销控制
- `--cuda-memory-usage=true`: 跟踪内存使用（会增加开销）
- `--stats=true`: 显示统计信息

## 4. 实际使用示例

### 基本分析
```bash
# 最简单的使用方式
nsys profile ./my_cuda_program

# 指定输出文件名
nsys profile -o my_analysis ./my_cuda_program

# 收集完整数据
nsys profile --trace=cuda,osrt,nvtx -o full_analysis ./my_cuda_program
```

### 分析我们的 softmax 程序
```bash
# 基本分析
nsys profile -o softmax_basic ./softmax_optimized

# 详细分析（包含操作系统调用）
nsys profile --trace=cuda,osrt -o softmax_detailed ./softmax_optimized

# 内存使用分析
nsys profile --cuda-memory-usage=true -o softmax_memory ./softmax_optimized
```

## 5. 查看分析结果

### 命令行查看统计信息
```bash
# 查看所有统计信息
nsys stats profile_name.nsys-rep

# 只查看 CUDA 内核统计
nsys stats --report gpukernsum profile_name.nsys-rep

# 只查看内存操作统计
nsys stats --report gpumemtimesum profile_name.nsys-rep

# 只查看 CUDA API 统计
nsys stats --report cudaapisum profile_name.nsys-rep
```

### 生成报告
```bash
# 生成 HTML 报告
nsys export --type=html -o report.html profile_name.nsys-rep

# 生成 JSON 报告
nsys export --type=json -o report.json profile_name.nsys-rep
```

## 6. 图形界面分析

### 启动 Nsight Systems GUI
```bash
# 打开分析文件
nsys-ui profile_name.nsys-rep

# 或者直接启动 GUI
nsys-ui
```

## 7. 高级用法

### 使用 NVTX 标记
```cpp
#include <nvtx3/nvtx3.hpp>

// 标记代码段
nvtx3::scoped_range range("Softmax Kernel");

// 标记函数
nvtx3::function_attributes attr("MyFunction");
nvtx3::scoped_range range(attr);
```

### 条件分析
```bash
# 只在特定 NVTX 范围内分析
nsys profile --capture-range=nvtx --capture-range-nvtx-name="MyRange" ./program

# 使用 CUDA Profiler API
nsys profile --capture-range=cudaProfilerApi ./program
```

### 批量分析
```bash
# 分析多个配置
for config in config1 config2 config3; do
    nsys profile -o analysis_${config} ./program --config=${config}
done
```

## 8. 分析结果解读

### CUDA 内核摘要 (gpukernsum)
- **Time (%)**: 内核执行时间占总时间的百分比
- **Instances**: 内核被调用的次数
- **Avg (ns)**: 平均执行时间
- **GridXYZ**: 网格维度
- **BlockXYZ**: 块维度

### 内存操作摘要 (gpumemtimesum)
- **DtoH**: Device to Host 内存传输
- **HtoD**: Host to Device 内存传输
- **时间分布**: 显示内存传输的时间开销

### CUDA API 摘要 (cudaapisum)
- **cudaMalloc**: 内存分配时间
- **cudaMemcpy**: 内存拷贝时间
- **cudaLaunchKernel**: 内核启动时间

## 9. 性能优化建议

### 基于 nsys 结果的优化策略

1. **内核执行时间过长**
   - 检查内核的线程配置
   - 优化内存访问模式
   - 使用共享内存

2. **内存传输开销大**
   - 减少 Host-Device 数据传输
   - 使用流式传输
   - 考虑使用统一内存

3. **同步开销大**
   - 减少 cudaDeviceSynchronize 调用
   - 使用 CUDA 流进行异步执行

## 10. 实际案例分析

### 分析 softmax_optimized 程序

从我们的分析结果可以看到：

1. **内核执行时间**:
   - `softmax_optimized`: 34.3μs (主要内核)
   - `softmax_fused`: 23.9μs (融合版本更快)
   - `softmax_warp_primitives`: 22.4μs (warp 原语最快)

2. **内存操作**:
   - DtoH 传输: 72.3% 的时间
   - HtoD 传输: 27.7% 的时间
   - 总内存传输: ~25MB

3. **API 调用**:
   - `cudaMalloc`: 89.8% 的 API 时间（正常，一次性分配）
   - `cudaMemcpy`: 4.3% 的 API 时间
   - `cudaEventSynchronize`: 3.6% 的 API 时间

### 优化建议

1. **减少内存传输**: 当前 DtoH 传输占用大量时间，可以考虑：
   - 减少结果回传频率
   - 使用流式处理
   - 在 GPU 上进行更多计算

2. **内核优化**: 三种内核中，warp primitives 版本最快，可以考虑：
   - 进一步优化 warp primitives 版本
   - 使用更小的输入尺寸时优先使用 warp 版本

## 11. 常用命令总结

```bash
# 基本分析
nsys profile -o analysis ./program

# 详细分析
nsys profile --trace=cuda,osrt,nvtx -o detailed ./program

# 查看统计
nsys stats analysis.nsys-rep

# 生成 HTML 报告
nsys export --type=html -o report.html analysis.nsys-rep

# 打开图形界面
nsys-ui analysis.nsys-rep
```

这个工具对于 CUDA 程序性能优化非常有用，可以帮助识别瓶颈并进行针对性优化。
