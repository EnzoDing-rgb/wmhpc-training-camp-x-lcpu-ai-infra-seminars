// 问题 4.7：访存模式与带宽。
// 同一个 kernel，只改读取的步长。
//
// ═════════════════════ 实验报告 · 第一性原理版（复习用）════════════════════
// 实测环境：RTX PRO 6000 Blackwell（sm_120），n = 16M 元素，reps = 20
//
//   实测数据（读写各 4 字节）：
//     stride    1       2       4       8       16      32
//     GB/s   2552.9  3136.2  2914.1  1827.3  1846.5   987.9
//
//   ── 三个底层事实（不用任何术语）──
//   1) 一个 warp = 32 个线程同时发同一个 load：32 个地址同一时刻递到内存单元。
//   2) DRAM 物理上要"激活整行"才能读，激活很贵，所以硬件按块搬运，
//      最小单位 32 字节（sector），cache line 是 128 字节（4 sector）。
//      → 你只要 1 个 float（4B），硬件也搬回 32B，多搬的是"无效字节"。
//   3) 所以"合并访存"= 让 warp 的 32 个请求尽量落进同一条 line / sector。
//
//   ── 逐档推演：32 个请求摊开了几条 line？──
//     stride=1  → 1 条 line（128B 连续），100% 有用
//     stride=2  → 2 条 line，每条约一半有用 → 50%
//     stride=4  → 4 条 line → 25%
//     stride=8  → 每个请求独占一个 32B sector → 12.5%
//     stride=32 → 每个请求独占一整条 line → ~3%
//   有用的数据恒为"每 warp 128B"，硬件却要搬 128×stride 字节。
//   内存原始吞吐固定 → 按有用数据计数的带宽 ≈ 除以 stride。
//
//   ── "唯一数据越读越少"是怎么来的？（把公式拆开看）──
//   j = (long)i * stride & (n-1)，n = 2^24，&(n-1) 只保留低 24 位。
//   当 stride = 2^k 时，i * stride = i << k，最低 k 位恒为 0，
//   再 & (n-1) 后这 k 位仍是 0 → j 永远是一个 stride 的倍数。
//   小例子验证：n=16、stride=4 时，j = i*4 & 15 只能取 {0,4,8,12}，
//   in[1..3] 这类位置任何线程都读不到。
//   → 全数组只有 1/stride 的位置被读过：
//       stride=1 → 64MB    stride=8 → 8MB    stride=32 → 2MB
//
//   ── 所以曲线 = 两股相反方向的力在拉扯 ──
//   拖慢力（合并访存被破坏）：stride 翻倍，每 warp 的 32 个请求散到的
//     sector/line 数翻倍（stride=8 起每个 float 独占一个 sector），
//     内存单元要处理的事务数变多。
//   救场力（能读的位置变少）：stride 越大，真正从 DRAM 搬的读数据越少
//     （64 → 32 → 16MB），省下的时间抵消了一部分事务开销。
//   合起来：stride≤4 贴峰值；8 开始掉；16 因"读的数据减半"与 8 基本持平；
//   到 32，请求撒满全数组（每 128B line 才 1 个 float），DRAM 行局部性
//   彻底消失，两股力一起往下压，又砍半（988）。
//
//   ── 顺带区分 bank conflict（同源问题，不同硬件单元）──
//   合并访存  = 32 个 global 请求撞"cache line"，散太开 → 搬无效字节。
//   bank conflict = 32 个 shared 请求撞"bank"：shared 只有 32 个 bank，
//   每周期每 bank 只服务 1 个词；两个线程地址映射到同一 bank（相距 128B，
//   即 32 个 float 的整数倍）就要串行排队。例：smem[tid] vs smem[tid+32]。
//   两个概念是一个模型的两面：并行请求撞上同一硬件单元 → 被迫串行 → 浪费。
//
//   ── 一句话记忆 ──
//   内存搬数的最小单位（32/128B）比线程想要的大（4B）→ 32 个请求撒得越开，
//   搬运的无效字节越多，有效带宽越低。让请求挤进尽量少的 line 就是合并访存。
// ═══════════════════════════════════════════════════════════════
#include "common.h"

// stride = 1 时是连续访问；stride 变大后，warp 里相邻线程读的地址
// 相距 stride 个 float。n 是 2 的幂，& (n-1) 等价于取模。
// 如果是隔着访问的话，会导致 Cache Line 一次取的这些数的数据利用率更低
__global__ void strided_copy(const float *in, float *out, int n, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int j = (long)i * stride & (n - 1);
        out[i] = in[j];
    }
}

int main() {
    const int n = 1 << 24;  // 16M 元素，2 的幂
    size_t bytes = (size_t)n * sizeof(float);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemset(d_in, 1, bytes));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    strided_copy<<<blocks, threads>>>(d_in, d_out, n, 1);  // 热身
    CUDA_CHECK_KERNEL();

    const int reps = 20;
    int strides[] = {1, 2, 4, 8, 16, 32};
    printf("%8s %12s %12s\n", "stride", "ms", "GB/s");
    for (int s : strides) {
        GpuTimer timer;
        timer.start();
        for (int r = 0; r < reps; r++)
            strided_copy<<<blocks, threads>>>(d_in, d_out, n, s);
        float ms = timer.stop_ms() / reps;
        CUDA_CHECK_KERNEL();
        // 读 + 写各 4 字节。
        double gbps = 2.0 * bytes / (ms * 1e-3) / 1e9;
        printf("%8d %12.4f %12.1f\n", s, ms, gbps);
    }
    return 0;
}
