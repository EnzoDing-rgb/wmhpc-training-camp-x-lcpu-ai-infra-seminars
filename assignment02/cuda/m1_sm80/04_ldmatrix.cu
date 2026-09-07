// 问题 1.4:把 1.3 的手工装载换成 ldmatrix,两条路径共存、都要 PASS。
//
// 把你 1.3 的 kernel 拆成两个装载函数移植进来:
//   load_manual:1.3 的逐 byte 手工装载(公式来自 1.1)
//   load_ldsm:  用 ldmatrix 装载。A 是 fp8——ldmatrix 的元素是 16 个
//               原始 bit,不关心类型;1.1 附加问的打包方向在这里起作用。
//               变体(.x1/.x2/.x4、是否 .trans)自己从 PTX 文档选。
// 数据已在 smem(main 里先从 global 拷入),两条路径都从 smem 装载。
//
// 都 PASS 之后:make ptx/m1_sm80/04_ldmatrix 或 nvdisasm 反汇编,
// 数两条路径 smem->fragment 段的指令构成(装载条数、地址算术条数),
// 报告里回答:ldmatrix 消掉的是哪部分工作?为什么手工路径绕不开它?
//
// 运行:make run/m1_sm80/04_ldmatrix(内部两条路径各跑多 seed)
#include <cuda_fp8.h>
#include <cstdlib>
#include <random>
#include "../common.h"

// smem 布局:sA 按 [16][32] 行主序;B 备了两种布局——sBk 按 [32][8]
// (k-major,1.3 用的就是它),sBn 按 [8][32](n-major,每个 n 的 32
// 个 k 字节连续)。手工路径用哪种都行;ldmatrix 的每个"行地址"要求
// 16 byte 连续,B 的 fragment 需要 k 方向相邻的字节成对进 b16——
// 想清楚哪种布局能满足它。
//
// TODO: 实现两个装载函数。
__device__ void load_manual(const uint8_t* sA, const uint8_t* sBk,
                            const uint8_t* sBn, unsigned (&a)[4],
                            unsigned (&b)[2]) {
    (void)sBk;
    int lane = threadIdx.x;
    int gid = lane / 4;  // 一组 4 个 lane（32 / N=8）
    int tig = lane % 4;

    // A 四象限,每寄存器 4 个沿 K 的 fp8: r0 左上, r1 左下, r2 右上, r3 右下
    a[0] = *(const unsigned*)&sA[gid * 32 + 4 * tig];
    a[1] = *(const unsigned*)&sA[(gid + 8) * 32 + 4 * tig];
    a[2] = *(const unsigned*)&sA[gid * 32 + 4 * tig + 16];
    a[3] = *(const unsigned*)&sA[(gid + 8) * 32 + 4 * tig + 16];
    // B 用 sBn(n-major,K 连续)。r=0 上半 K, r=1 下半 K(+16)
    b[0] = *(const unsigned*)&sBn[gid * 32 + 4 * tig];
    b[1] = *(const unsigned*)&sBn[gid * 32 + 4 * tig + 16];
}

__device__ void load_ldsm(const uint8_t* sA, const uint8_t* sBk,
                          const uint8_t* sBn, unsigned (&a)[4],
                          unsigned (&b)[2]) {
    (void)sBk;
    int lane = threadIdx.x;
    // ------------------------------------------------------------------
    // ldmatrix 每人只交一个「16 byte 连续行」的行首。交完就完事;硬件按
    // fragment 图拆进 a[]/b[]。
    //
    // 对着 A 的四色图(绿a[0]/紫a[1]/蓝a[2]/橙a[3]):每个色块是 8 行 ×
    // 半段 K。.x4 = 一次装齐四个色块 → 需要 4×8 = 32 个行首 → 正好 32 lane
    // 一人交一条。
    //
    //   quad        = 我负责哪个色块(= 图上的 r,也是写进 a[quad] 的那个)
    //   row_in_quad = 这个色块里第几行(0..7)——图上每个色块恰好 8 行
    //
    // 本题 e4m3、K=32:一行 32 byte,半段 K = 16 byte = 一条 ldmatrix 行。
    // (课上 fp16/K=16 的图半段是 8 个 fp16 = 16 byte,同构。)
    // ------------------------------------------------------------------
    int quad = lane / 8;        // 0绿 1紫 2蓝 3橙
    int row_in_quad = lane % 8; // 色块内行号 0..7

    // A: sA[16][32] 行主序。行随上下半色块 +0/+8;列随左右半色块 +0/+16。
    int a_row = row_in_quad + 8 * (quad % 2);  // 紫/橙在下半
    int a_col = 16 * (quad / 2);               // 蓝/橙在 K 右半
    uint32_t aAddr = __cvta_generic_to_shared(&sA[a_row * 32 + a_col]);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                 "{%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(aAddr));

    // B: sBn[8][32],一行 = 某个 n 的 32 个 k。.x2 只要两个「8 行」半区
    // (b[0] 上半 K、b[1] 下半 K)→ 16 个行首;lane 0..15 有用,16..31 填合法地址即可。
    // 这里的「行」是 n:row_in_quad 当 n;quad%2 选 K 半区(+0 / +16)。
    int b_n = row_in_quad;
    int b_k = 16 * (quad % 2);
    uint32_t bAddr = __cvta_generic_to_shared(&sBn[b_n * 32 + b_k]);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
                 "{%0,%1}, [%2];\n"
                 : "=r"(b[0]), "=r"(b[1])
                 : "r"(bAddr));
}

template <bool USE_LDSM>
__global__ void mma_kernel(const uint8_t* A, const uint8_t* B, float* D) {
    __shared__ uint8_t sA[16 * 32], sBk[32 * 8], sBn[8 * 32];
    for (int i = threadIdx.x; i < 16 * 32; i += 32) sA[i] = A[i];
    for (int i = threadIdx.x; i < 32 * 8; i += 32) {
        sBk[i] = B[i];
        sBn[(i & 7) * 32 + (i >> 3)] = B[i];  // 转成 n-major
    }
    __syncwarp();
    unsigned a[4], b[2];
    if constexpr (USE_LDSM)
        load_ldsm(sA, sBk, sBn, a, b);
    else
        load_manual(sA, sBk, sBn, a, b);
    float c[4] = {0, 0, 0, 0}, d[4];
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
    int group = threadIdx.x >> 2, tig = threadIdx.x & 3;
    D[group * 8 + tig * 2] = d[0];
    D[group * 8 + tig * 2 + 1] = d[1];
    D[(group + 8) * 8 + tig * 2] = d[2];
    D[(group + 8) * 8 + tig * 2 + 1] = d[3];
}

static int run_path(bool ldsm, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 15);
    uint8_t hA[16 * 32], hB[32 * 8];
    float fA[16 * 32], fB[32 * 8], ref[16 * 8] = {};
    for (int i = 0; i < 16 * 32; i++) {
        __nv_fp8_e4m3 v = __nv_fp8_e4m3((float)(dist(rng) - 8));
        hA[i] = *(uint8_t*)&v;
        fA[i] = float(v);
    }
    for (int i = 0; i < 32 * 8; i++) {
        __nv_fp8_e4m3 v = __nv_fp8_e4m3((float)(dist(rng) - 8));
        hB[i] = *(uint8_t*)&v;
        fB[i] = float(v);
    }
    for (int r = 0; r < 16; r++)
        for (int n = 0; n < 8; n++)
            for (int k = 0; k < 32; k++)
                ref[r * 8 + n] += fA[r * 32 + k] * fB[k * 8 + n];
    uint8_t *dA, *dB;
    float* dD;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(hA)));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(hB)));
    CUDA_CHECK(cudaMalloc(&dD, 16 * 8 * 4));
    CUDA_CHECK(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));
    if (ldsm)
        mma_kernel<true><<<1, 32>>>(dA, dB, dD);
    else
        mma_kernel<false><<<1, 32>>>(dA, dB, dD);
    CUDA_CHECK_KERNEL();
    float got[16 * 8];
    CUDA_CHECK(cudaMemcpy(got, dD, sizeof(got), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < 16 * 8; i++) bad += got[i] != ref[i];
    cudaFree(dA); cudaFree(dB); cudaFree(dD);
    return bad;
}

int main() {
    long total = 0;
    for (unsigned s : {1u, 7u, 42u}) {
        int bm = run_path(false, s), bl = run_path(true, s);
        printf("seed=%-6u manual %s(%d)  ldsm %s(%d)\n", s,
               bm ? "FAIL" : "PASS", bm, bl ? "FAIL" : "PASS", bl);
        total += bm + bl;
    }
    return total != 0;
}
