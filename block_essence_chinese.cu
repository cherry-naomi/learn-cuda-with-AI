#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// 演示CUDA Block的本质：独立的协作单元
__global__ void demonstrateBlockEssence(int *global_data, int N) {
    // Block的本质特征1：共享内存空间
    __shared__ int block_workspace[256];  // 这个Block的专属工作区
    __shared__ int block_id_storage;      // Block内所有线程共享
    __shared__ int cooperation_result;    // 协作计算结果
    
    int tid = threadIdx.x;  // 线程在Block内的位置
    int bid = blockIdx.x;   // Block在Grid中的位置
    int global_idx = bid * blockDim.x + tid;  // 线程的全局索引
    
    // 特征1：Block内线程可以协作初始化
    if (tid == 0) {
        block_id_storage = bid * 1000;  // Block标识
        cooperation_result = 0;         // 初始化协作结果
        printf("Block %d: 我是一个独立的协作单元，有%d个线程协同工作\n", 
               bid, blockDim.x);
    }
    
    // 特征2：Block内同步 - 这是Block存在的核心原因
    __syncthreads();  // 等待Block内所有线程完成初始化
    
    // 特征3：共享内存访问 - Block定义了共享边界
    if (tid < 256 && global_idx < N) {
        block_workspace[tid] = global_data[global_idx] + block_id_storage;
        printf("Block %d, Thread %d: 访问共享工作区[%d] = %d\n", 
               bid, tid, tid, block_workspace[tid]);
    }
    
    __syncthreads();  // 等待所有线程写入完成
    
    // 特征4：Block内协作计算
    // 每个线程贡献自己的值到协作结果
    if (tid < 256 && global_idx < N) {
        atomicAdd(&cooperation_result, block_workspace[tid]);
    }
    
    __syncthreads();  // 等待协作计算完成
    
    // 特征5：Block级别的结果输出
    if (tid == 0) {
        printf("Block %d: 协作计算完成，结果 = %d\n", bid, cooperation_result);
        global_data[bid] = cooperation_result;  // 写回全局结果
    }
    
    // 重要：不同Block之间无法直接通信或同步！
    // Block是完全独立的执行单元
}

// 演示Block的物理映射到硬件
__global__ void demonstrateBlockHardwareMapping() {
    __shared__ int sm_identifier;  // 模拟SM标识
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    if (tid == 0) {
        // 每个Block被分配到一个SM上执行
        sm_identifier = bid % 8;  // 假设有8个SM
        printf("Block %d: 被分配到SM %d上执行\n", bid, sm_identifier);
        printf("Block %d: 我的%d个线程将被组织成%d个warp\n", 
               bid, blockDim.x, (blockDim.x + 31) / 32);
    }
    
    __syncthreads();
    
    // 计算这个线程属于哪个warp
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    if (tid < 64) {  // 只打印前64个线程的信息
        printf("Block %d, Thread %d: 属于Warp %d, Lane %d, 运行在SM %d\n",
               bid, tid, warp_id, lane_id, sm_identifier);
    }
}

// 演示Block的独立性
__global__ void demonstrateBlockIndependence(int *counter) {
    __shared__ int local_counter;
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    if (tid == 0) {
        local_counter = 0;
        printf("Block %d: 开始独立工作\n", bid);
    }
    
    __syncthreads();
    
    // 每个线程对Block内计数器加1
    atomicAdd(&local_counter, 1);
    
    __syncthreads();
    
    if (tid == 0) {
        printf("Block %d: 本地计数器 = %d\n", bid, local_counter);
        // 只能通过全局内存与其他Block通信
        atomicAdd(counter, local_counter);
    }
}

void explainBlockEssence() {
    printf("=== CUDA BLOCK的本质解析 ===\n\n");
    
    printf("Block是什么？\n");
    printf("1. 硬件抽象：Block是GPU硬件SM(流多处理器)的软件抽象\n");
    printf("2. 协作单元：一组可以协作的线程的集合\n");
    printf("3. 共享边界：定义了哪些线程可以共享内存和同步\n");
    printf("4. 独立单位：每个Block完全独立，可以并行执行\n");
    printf("5. 调度单位：GPU以Block为单位进行任务调度\n\n");
    
    printf("Block的核心特征：\n");
    printf("┌─────────────────────────────────────────────┐\n");
    printf("│ Block = 协作 + 共享 + 独立 + 同步           │\n");
    printf("├─────────────────────────────────────────────┤\n");
    printf("│ 协作：线程可以一起完成复杂任务              │\n");
    printf("│ 共享：线程共享同一块shared memory          │\n");
    printf("│ 独立：不同Block完全独立，无法直接通信      │\n");
    printf("│ 同步：__syncthreads()只能同步Block内线程   │\n");
    printf("└─────────────────────────────────────────────┘\n\n");
    
    printf("硬件映射关系：\n");
    printf("软件层面：Grid → Block → Thread\n");
    printf("硬件层面：GPU  → SM    → Core/ALU\n");
    printf("         ↓      ↓       ↓\n");
    printf("映射关系：整个程序→ 一个Block → 一个线程\n");
    printf("         被分配到一个SM上执行\n\n");
}

void explainBlockDesignPrinciples() {
    printf("=== Block设计原则 ===\n\n");
    
    printf("1. 为什么Block必须独立？\n");
    printf("   - 可扩展性：代码在不同GPU上都能运行\n");
    printf("   - 调度灵活性：GPU可以灵活调度Block到不同SM\n");
    printf("   - 容错性：一个Block失败不影响其他Block\n");
    printf("   - 简化编程：不需要考虑复杂的Block间通信\n\n");
    
    printf("2. 为什么Block内需要协作？\n");
    printf("   - 数据局部性：充分利用shared memory的高速访问\n");
    printf("   - 算法复杂性：实现需要协作的并行算法\n");
    printf("   - 硬件效率：一个SM上的线程天然适合协作\n");
    printf("   - 同步成本：Block内同步成本远低于全局同步\n\n");
    
    printf("3. Block大小如何选择？\n");
    printf("   - 硬件限制：必须是32的倍数(warp大小)\n");
    printf("   - 资源限制：shared memory, registers的限制\n");
    printf("   - 占用率：平衡线程数量和资源使用\n");
    printf("   - 常用选择：128, 256, 512线程\n\n");
}

int main() {
    printf("=== CUDA BLOCK 本质探究 ===\n\n");
    
    explainBlockEssence();
    explainBlockDesignPrinciples();
    
    // 实际演示
    printf("=== 实际演示 ===\n\n");
    
    const int N = 64;
    const int blocks = 4;
    const int threads = 16;
    
    int *h_data = (int*)malloc(N * sizeof(int));
    int *d_data, *d_counter;
    
    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_data[i] = i + 1;
    }
    
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMalloc(&d_counter, sizeof(int));
    
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    
    int zero = 0;
    cudaMemcpy(d_counter, &zero, sizeof(int), cudaMemcpyHostToDevice);
    
    printf("演示1：Block的协作特性\n");
    printf("启动配置：%d个Block，每个Block %d个线程\n\n", blocks, threads);
    
    demonstrateBlockEssence<<<blocks, threads>>>(d_data, N);
    cudaDeviceSynchronize();
    
    printf("\n演示2：Block的硬件映射\n");
    demonstrateBlockHardwareMapping<<<blocks, threads>>>();
    cudaDeviceSynchronize();
    
    printf("\n演示3：Block的独立性\n");
    demonstrateBlockIndependence<<<blocks, threads>>>(d_counter);
    cudaDeviceSynchronize();
    
    // 检查全局计数器结果
    int final_counter;
    cudaMemcpy(&final_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    printf("\n全局计数器最终值：%d (应该等于总线程数 %d)\n", 
           final_counter, blocks * threads);
    
    printf("\n=== Block本质总结 ===\n\n");
    
    printf("🏗️ 架构层面：\n");
    printf("Block是GPU并行计算架构的基本构建块\n");
    printf("它在软件抽象和硬件实现之间提供了完美的平衡\n\n");
    
    printf("🤝 协作层面：\n");
    printf("Block定义了线程协作的边界和能力\n");
    printf("共享内存 + 同步原语 = 强大的协作能力\n\n");
    
    printf("🎯 调度层面：\n");
    printf("Block是GPU调度器的最小调度单位\n");
    printf("独立性保证了调度的灵活性和可扩展性\n\n");
    
    printf("💡 编程层面：\n");
    printf("Block提供了直观的并行编程模型\n");
    printf("程序员只需关注Block内的协作逻辑\n\n");
    
    printf("Block的本质 = 硬件抽象 + 协作单元 + 调度单位 + 编程模型\n");
    printf("这就是为什么Block是CUDA编程的核心概念！\n");
    
    // 清理
    free(h_data);
    cudaFree(d_data);
    cudaFree(d_counter);
    
    return 0;
}
