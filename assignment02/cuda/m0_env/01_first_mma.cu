// 问题 0.1:最小的 tensor core 程序,不需要修改。
//
// 单个 warp 发一条 mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32:
// D[16x8] = A[16x16] × B[16x8] + C。A/B 用小整数填充(fp16 下精确,
// f32 累加也精确),host 端用 CPU 循环对拍,所以判测是严格相等。
//
// fragment 的装载按 PTX 文档的公式写成了下标计算的形式,课件 P2.2
// 推导的就是这组公式;模块 1 会让你对另一个形状把它们重新推一遍。
//
// 运行:make run/m0_env/01_first_mma
// 题面 (b) 问会用到 Makefile 的 ptx 目标和 assignment01 的
// sassonly/ptxonly 实验,见题面。
//
// ============================================================================
// 词汇表(读代码前先扫一遍;后面 inline 注释还会再点一次)
// ============================================================================
// Tensor Core
//   GPU 里专门做小矩阵乘加(MMA)的硬件单元。本文件不碰 CUDA Core 上的
//   标量 FMA 循环,而是让整个 warp 协作,把一块小矩阵乘交给 Tensor Core。
//
// MMA (Matrix Multiply-Accumulate)
//   D = A × B + C。本例形状固定为 M=16, N=8, K=16,即
//   A 是 16×16, B 是 16×8(按 K×N 看), C/D 是 16×8。
//
// warp / lane
//   warp = 32 条硬件线程一起锁步执行的一组。
//   lane = 这条线程在 warp 里的编号,通常就是 threadIdx.x % 32。
//   本 kernel 只开 <<<1, 32>>>,所以 lane == threadIdx.x ∈ [0,31]。
//
// fragment(碎片 / 矩阵片段)
//   MMA 的操作数不整块放在一个数组里给硬件看,而是被“切开”分给 32 个
//   lane:每个 lane 只在自己的寄存器里拿矩阵的一小块。这一小块就叫
//   fragment。
//
// group / tig  (本文件的命名习惯;PTX 文档常写成类似的拆分)
//   把 lane 拆成两个坐标,正好对应 fragment 布局里的“行组”和“组内列”:
//     group = lane >> 2 = lane / 4   → 取值 0..7  (8 个组)
//     tig   = lane &  3 = lane % 4   → 取值 0..3  (Thread-In-Group)
//   例如 lane=13: 13=4*3+1 → group=3, tig=1。
//   为什么这样拆?因为 m16n8k16 里每个 lane 拿的 A 是"2 行 × 4 列"的小块:
//   2 行 → 16 行 ÷ 2 = 8 个行组(group);4 列 → 16 列 ÷ 4 = 4 个列组(tig)。
//   5 位的 lane 号正好 = 3 位 group + 2 位 tig,位拆开就是这个二维坐标。
//
// row.col / 布局修饰符
//   PTX 里 `.row.col` 表示:A 按 row-major 解释进 fragment,B 按
//   column-major 解释进 fragment(注意:这是 MMA 指令对操作数布局的约定,
//   不等于你在 global memory 里怎么存。本文件 B 在内存仍按 [k][n]
//   行主序存,装载时用下标把它“看成”col 布局需要的样子)。
//
// m16n8k16.f32.f16.f16.f32
//   形状 M/N/K,以及 D/A/B/C 的 dtype:累加与 C/D 用 f32,A/B 用 f16。
//
// sync / aligned
//   sync:这条 MMA 是 warp 级同步协作指令——32 个 lane 必须一起执行,
//   有 lane 发散则行为未定义(题面 0.3(b))。
//   aligned:操作数寄存器按文档要求对齐地交给指令。
// ============================================================================

#include <cuda_fp16.h>
#include "../common.h"

__global__ void mma_demo(const __half* A, const __half* B, float* D) {
    // ---- 身份:我是 warp 里的哪条 lane?再拆成 (group, tig) ----
    int lane = threadIdx.x;     // 本 block 只有 32 线程 ⇒ lane ∈ [0,31]
    int group = lane >> 2;      // 行方向的 8 个组  (0..7);等价于 lane / 4
    int tig = lane & 3;         // 组内 4 个线程    (0..3);等价于 lane % 4
    //
    // 为什么拆成 (group, tig)?因为硬件规定的 fragment 布局里,每个 lane
    // 拿的 A 是"2 行 × 4 列"的小块:
    //   - 2 行 → 16 行 ÷ 2 = 8 个行组 → group(lane 的【高 3 位】,0..7)
    //   - 4 列 → 16 列 ÷ 4 = 4 个列组 → tig (lane 的【低 2 位】,0..3)
    // 5 位的 lane 号 = 3 位 group + 2 位 tig,位拆开正好就是这两个坐标。
    //
    // 行归属(group 决定"我管哪 2 行"):
    //   group 0: lanes 0,1,2,3  → 行 {0, 0+8} = 第 0 行 与 第 8 行
    //   group 1: lanes 4,5,6,7  → 行 {1, 1+8} = 第 1 行 与 第 9 行
    //   ...
    //   group 7: lanes 28..31   → 行 {7, 7+8} = 第 7 行 与 第 15 行
    // 为什么每组是"两行、且相隔 8"?因为 16 行被 8 个组平分(每组 2 行),
    // 硬件把第 g 组的两行放在第 g 行和第 g+8 行。列方向由 tig 再分(见 A 装载)。

    // A fragment:每线程 8 个 fp16 = 4 个 b32 寄存器。
    // 8 个元素在矩阵里是"2 行 × 4 列":
    //   行 = {group, group+8};列 = {2t, 2t+1, 2t+8, 2t+9}(左半 2 个 + 右半 2 个)
    //
    // 寄存器宽度(为什么是 4 个寄存器):
    //   一个 b32 = 32 位 = 2 个 fp16(16 位×2)= 1 个 __half2。
    //   8 个元素 ÷ 每寄存器 2 个 = 4 个寄存器 a[0..3]。
    //   所以用 unsigned 声明(裸 b32),再 reinterpret 成 __half2 来读写。
    //
    // 内存公式:A[row * 16 + col],*16 是行步长(16 列),单位是【元素】不是字节。
    //
    // 四个寄存器各自的位置(a[i] 内含相邻两列):
    //     a[0] → 行 group,     列 2t, 2t+1      (K 前半 · 上半行)
    //     a[1] → 行 group+8,   列 2t, 2t+1      (K 前半 · 下半行)
    //     a[2] → 行 group,     列 2t+8, 2t+9    (K 后半 · 上半行)
    //     a[3] → 行 group+8,   列 2t+8, 2t+9    (K 后半 · 下半行)
    //   两个 "+8" 含义不同:行上的 +8 是 M 方向切两半;列上的 +8 是 K 方向切两半。
    //   "2t" 来自 __half2:一次装 2 个相邻元素,所以列从 2t 起步、步进 2。
    unsigned a[4];
    __half2* ah = reinterpret_cast<__half2*>(a);
    ah[0] = __halves2half2(A[(group)*16 + tig * 2], A[(group)*16 + tig * 2 + 1]);
    ah[1] = __halves2half2(A[(group + 8) * 16 + tig * 2],
                           A[(group + 8) * 16 + tig * 2 + 1]);
    ah[2] = __halves2half2(A[(group)*16 + tig * 2 + 8],
                           A[(group)*16 + tig * 2 + 9]);
    ah[3] = __halves2half2(A[(group + 8) * 16 + tig * 2 + 8],
                           A[(group + 8) * 16 + tig * 2 + 9]);

    // B fragment(col 布局,B 在内存里仍按 [k][n] 行主序存):
    //
    // 为什么只有 2 个寄存器?B 是 16×8 = 128 个元素 ÷ 32 lane = 4 个/lane
    //   = 2 个 __half2 = 2 个 b32(B 比 A 瘦:N=8 < K=16)。
    //
    // 每个 lane 拿的 4 个元素 = "固定列 n=group,沿 k 取 4 个 {2t,2t+1,2t+8,2t+9}"。
    // 这是 `.col` 布局的含义:硬件要算 D[r][n] = Σ_k A[r][k]·B[k][n] 的点积,
    // 需要 B 的【一整列】(16 个 k),所以 B 按列切给线程;同一 group 的 4 个 tig
    // 合起来正好凑满那一列的 16 个 k。
    //
    // 内存公式:B[k * 8 + n],*8 是行步长(8 列)。
    // 注意:这里 group 扮演的是 *N 方向* 的列号(0..7),而在 A/D 里
    //   group 扮演的是 *M 方向* 的行号——同一名字在不同操作数里轴不同,
    //   这是 fragment 布局的约定,不是笔误。
    unsigned b[2];
    __half2* bh = reinterpret_cast<__half2*>(b);
    bh[0] = __halves2half2(B[(tig * 2) * 8 + group], B[(tig * 2 + 1) * 8 + group]);
    bh[1] = __halves2half2(B[(tig * 2 + 8) * 8 + group],
                           B[(tig * 2 + 9) * 8 + group]);

    // C/D fragment:每 lane 4 个 f32。本 demo 令 C=0,所以 D = A×B。
    // PTX 操作数顺序:D, A, B, C —— 对应 asm 里四组花括号。
    float c[4] = {0.f, 0.f, 0.f, 0.f}, d[4];
    asm volatile(
        // 指令名拆读:
        //   mma.sync.aligned          warp 协作的 MMA
        //   .m16n8k16                 形状
        //   .row.col                  A row / B col
        //   .f32.f16.f16.f32          D.A.B.C 的 dtype
        // 输出 d[0..3],输入 a[0..3], b[0..1], c[0..3]。
        // "=f" / "r" / "f" 是 GCC 扩展 asm 的约束:f32 输出、u32 输入、f32 输入。
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));

    // D fragment:d[0..3] = 2 行 × 2 列 = 行 {group, group+8} × 列 {2t, 2t+1}。
    // mma 输出的 4 个 f32,写回的位置必须按 fragment 布局,不能“lane i 写第 i 个元素”:
    //   d[0] → D[group][tig*2]      d[1] → D[group][tig*2+1]
    //   d[2] → D[group+8][tig*2]    d[3] → D[group+8][tig*2+1]
    // 内存是 row-major 的 16×8:下标 = row * 8 + col。
    // 32 lane × 4 = 128 = 16×8,恰好盖满整个 D,无重无漏。
    D[(group)*8 + tig * 2] = d[0];
    D[(group)*8 + tig * 2 + 1] = d[1];
    D[(group + 8) * 8 + tig * 2] = d[2];
    D[(group + 8) * 8 + tig * 2 + 1] = d[3];
}

int main() {
    // ---- Host:构造小整数输入 + CPU 参考答案,再和 GPU 结果严格相等比对 ----
    __half hA[16 * 16], hB[16 * 8];
    float ref[16 * 8] = {};
    // A[r,k] = (r+k)%5 - 2 ∈ {-2,-1,0,1,2},fp16 与 f32 都能精确表示。
    for (int r = 0; r < 16; r++)
        for (int k = 0; k < 16; k++) hA[r * 16 + k] = __float2half((r + k) % 5 - 2);
    // B[k,n] = (k*n)%3 - 1 ∈ {-1,0,1},同样精确。
    for (int k = 0; k < 16; k++)
        for (int n = 0; n < 8; n++) hB[k * 8 + n] = __float2half((k * n) % 3 - 1);
    // 朴素三重循环:ref[r,n] = Σ_k A[r,k] * B[k,n]。
    for (int r = 0; r < 16; r++)
        for (int n = 0; n < 8; n++)
            for (int k = 0; k < 16; k++)
                ref[r * 8 + n] += __half2float(hA[r * 16 + k]) *
                                  __half2float(hB[k * 8 + n]);

    __half *dA, *dB;
    float* dD;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(hA)));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(hB)));
    CUDA_CHECK(cudaMalloc(&dD, 16 * 8 * 4));  // 16*8 个 float = 512 B
    CUDA_CHECK(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));
    // 1 个 block、32 个线程 = 恰好 1 个 warp,发一条 mma.sync。
    mma_demo<<<1, 32>>>(dA, dB, dD);
    CUDA_CHECK_KERNEL();
    float got[16 * 8];
    CUDA_CHECK(cudaMemcpy(got, dD, sizeof(got), cudaMemcpyDeviceToHost));

    long bad = 0;
    // 因为输入是小整数且累加用 f32,期望 bit-exact;任何 fragment 装载错误
    // 都会让 bad > 0(这就是后面 debug fragment 题的判据形态)。
    for (int i = 0; i < 16 * 8; i++) bad += got[i] != ref[i];
    printf("D[0][0]=%.0f D[0][7]=%.0f D[15][0]=%.0f D[15][7]=%.0f\n", got[0],
           got[7], got[15 * 8], got[15 * 8 + 7]);
    if (bad)
        printf("FAIL: %ld mismatches\n", bad);
    else
        printf("PASS\n");
    return bad != 0;
}
