// ============================================================================
// 问题 1.3 (FROM-SCRATCH): 手写单 tile fp8 mma
//
// 单 tile m16n8k32 fp8 mma: D(16×8 f32) = A(16×32 fp8) × B(32×8 fp8)
//   - A 行主序存; B 存转置 B^T[n][k](K 连续),让 B 也能整 b32 读
//   - fragment 装载全部手动,本题不用 ldmatrix
//   - 输入 [-4,4] 小整数,e4m3 精确 → 与 CPU 参考严格相等比较
//
// fragment 公式(1.1):
//   lane = gid*4 + tig,  gid = lane/4 (0..7),  tig = lane%4 (0..3)
//   A: row = gid + 8*(r%2),  col = 4*tig + 16*(r/2) + j
//   B(转置后): n = gid,  k = 4*tig + 16*r + j   → 整 b32 读 B^T[n][k]
//   D: row = gid + 8*(r/2),  col = 2*tig + (r%2)
//
// 判测: cd assignment02/cuda/m1_sm80
//   srun -p lcpu-infra --gpus=1 env ARCH=120a ./judge_mma_fp8.sh 03_mma_fp8.cu
// 单 seed 调试: cd assignment02/cuda
//   make ARCH=120a bin/m1_sm80/03_mma_fp8 && srun -p lcpu-infra --gpus=1 ./bin/m1_sm80/03_mma_fp8 123
// ============================================================================
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include "../common.h"

__global__ void mma_fp8_kernel(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* B,
                               float* D) {
    int lane = threadIdx.x;
    int gid = lane >> 2;
    int tig = lane & 3;

    // ---------- A fragment: 4 个 b32,每个装 4 个沿 K 方向连续的 fp8 ----------
    // 四象限: r0 左上, r1 左下(+8 行), r2 右上(+16 列), r3 右下。
    unsigned ra[4] = {
        *(const unsigned*)&A[gid * 32 + 4 * tig],
        *(const unsigned*)&A[(gid + 8) * 32 + 4 * tig],
        *(const unsigned*)&A[gid * 32 + 4 * tig + 16],
        *(const unsigned*)&A[(gid + 8) * 32 + 4 * tig + 16],
    };

    // ---------- B fragment: 2 个 b32 ----------
    // B 存的是转置 B^T[n][k](K 连续),所以和 A 一样整 b32 读;
    // "行"是 n=gid,"列"是 k。r=0 → K 左半(4*tig), r=1 → K 右半(4*tig+16)。
    unsigned rb[2] = {
        *(const unsigned*)&B[gid * 32 + 4 * tig],
        *(const unsigned*)&B[gid * 32 + 4 * tig + 16],
    };

    // ---------- mma: m16n8k32 fp8 e4m3, f32 累加 ----------
    float c[4] = {0.f, 0.f, 0.f, 0.f}, d[4];
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb[0]),
          "r"(rb[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));

    // ---------- D 写回: 2×2 小块(行 = gid/gid+8, 列 = 2*tig/2*tig+1) ----------
    D[gid * 8 + tig * 2] = d[0];
    D[gid * 8 + tig * 2 + 1] = d[1];
    D[(gid + 8) * 8 + tig * 2] = d[2];
    D[(gid + 8) * 8 + tig * 2 + 1] = d[3];
}

int main(int argc, char* argv[]) {
    int seed = argc > 1 ? atoi(argv[1]) : 0;
    std::mt19937 rng(seed);

    // A: 16×32 行主序; B 存转置 hB[n*32+k] = B(k,n)(K 连续)。
    // 都取 [-4,4] 小整数:e4m3 精确、乘积/部分和都是小整数 fp32 精确,
    // 所以能跟 CPU 参考严格相等。
    __nv_fp8_e4m3 hA[16 * 32];
    __nv_fp8_e4m3 hB[32 * 8];  // B^T: 8×32
    for (int i = 0; i < 16 * 32; i++)
        hA[i] = __nv_fp8_e4m3((int)(rng() % 9) - 4);
    for (int n = 0; n < 8; n++)
        for (int k = 0; k < 32; k++)
            hB[n * 32 + k] = __nv_fp8_e4m3((int)(rng() % 9) - 4);

    // CPU 参考: B(k,n) = hB[n*32+k]
    float ref[16 * 8] = {};
    for (int r = 0; r < 16; r++)
        for (int n = 0; n < 8; n++)
            for (int k = 0; k < 32; k++)
                ref[r * 8 + n] += static_cast<float>(hA[r * 32 + k]) *
                                  static_cast<float>(hB[n * 32 + k]);

    __nv_fp8_e4m3 *dA, *dB;
    float* dD;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(hA)));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(hB)));
    CUDA_CHECK(cudaMalloc(&dD, 16 * 8 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));
    mma_fp8_kernel<<<1, 32>>>(dA, dB, dD);
    CUDA_CHECK_KERNEL();

    float got[16 * 8];
    CUDA_CHECK(cudaMemcpy(got, dD, sizeof(got), cudaMemcpyDeviceToHost));

    long bad = 0;
    for (int r = 0; r < 16; r++)
        for (int n = 0; n < 8; n++)
            if (got[r * 8 + n] != ref[r * 8 + n]) {
                if (bad < 4)
                    printf("MISMATCH D[%d][%d]: got %.0f, want %.0f\n", r, n,
                           got[r * 8 + n], ref[r * 8 + n]);
                bad++;
            }
    if (bad)
        printf("FAIL: %ld / %d mismatches\n", bad, 16 * 8);
    else
        printf("PASS\n");
    return bad != 0;
}
