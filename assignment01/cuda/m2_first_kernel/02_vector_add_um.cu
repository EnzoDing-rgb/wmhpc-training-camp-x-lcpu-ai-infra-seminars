// 问题 2.3：把显式内存管理改成 Unified Memory（ MODIFY ）。
// 下面是一份完整可运行的显式管理版本。任务：
//   0. 先按原样跑一次，记下耗时——这一版会被你的改动覆盖掉，
//      第 4 步的对比要拿它做基准；
//   1. 用 cudaMallocManaged 替换 cudaMalloc + malloc；
//   2. 删掉所有 cudaMemcpy，kernel 直接读写同一组指针，CPU 也直接读；
//   3. 想清楚哪里需要 cudaDeviceSynchronize；
//   4. 对比两版的耗时。两版的计时窗口要保持一致：分配和填数据都在窗口
//      外，窗口从"数据已经在内存里备好"开始，到 CPU 把结果全部读完为止
//      （下面用一个累加校验和的循环代表"CPU 读完全部结果"，别把它删了）。
// 改完仍要 PASS。

// 本实现在一个 main 里依次跑显式版和 UM 版，一次运行直接对比。
#include <chrono>
#include <cstring>
#include "common.h"

__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main() {
    const int n = 1 << 24;  // 16M 元素
    size_t bytes = (size_t)n * sizeof(float);

    // 先把 CUDA context 建起来。首次调用 CUDA API 要花几百毫秒初始化，
    // 放进计时窗口会把要观察的差距完全淹掉。
    CUDA_CHECK(cudaFree(0));

    // ---------- 两份输入数据，两版共用，在计时窗口外备好 ----------
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    fill_random(h_a, n, 1);
    fill_random(h_b, n, 2);

    // 期望的校验和，host 上先算好，同样不计入计时。
    double want = 0;
    for (int i = 0; i < n; i++) want += (double)(h_a[i] + h_b[i]);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    double explicit_ms = 0, um_ms = 0;
    bool explicit_ok = false, um_ok = false;

    // ==================== 显式内存管理（baseline） ====================
    {
        // 6 次分配：host 3 + device 3
        float *h_c = (float *)malloc(bytes);
        float *d_a, *d_b, *d_c;
        CUDA_CHECK(cudaMalloc(&d_a, bytes));
        CUDA_CHECK(cudaMalloc(&d_b, bytes));
        CUDA_CHECK(cudaMalloc(&d_c, bytes));

        // ===== 计时窗口开始 =====
        auto t0 = std::chrono::steady_clock::now();

        CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

        vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);
        CUDA_CHECK_KERNEL();

        // 这个一定是等前面搬完了才能开始搬的，因此这里的数据搬运点，实际承担了隐式同步的功能
        CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

        double got = 0;
        for (int i = 0; i < n; i++) got += (double)h_c[i];

        auto t1 = std::chrono::steady_clock::now();
        // ===== 计时窗口结束 =====

        explicit_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        explicit_ok = fabs(got - want) <= 1e-3 * (1.0 + fabs(want));

        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));
        free(h_c);
    }

    // ==================== Unified Memory ====================
    {
        // 3 次分配：代替上面的 3×malloc + 3×cudaMalloc
        float *a, *b, *c;
        CUDA_CHECK(cudaMallocManaged(&a, bytes));
        CUDA_CHECK(cudaMallocManaged(&b, bytes));
        CUDA_CHECK(cudaMallocManaged(&c, bytes));

        // 把输入数据拷进 UM 分配（CPU 写 → 页在 host，在计时窗口外）
        std::memcpy(a, h_a, bytes);
        std::memcpy(b, h_b, bytes);

        // ===== 计时窗口开始 =====
        auto t0 = std::chrono::steady_clock::now();

        // kernel 读写 a, b, c —— 缺页时驱动自动搬运
        vectorAdd<<<blocks, threads>>>(a, b, c, n);
        // kernel 启动是异步的，必须等它跑完 CPU 才能读 c
        CUDA_CHECK(cudaGetLastError());
        // 这里是真正需要同步的同步点
        CUDA_CHECK(cudaDeviceSynchronize());

        // CPU 读全部结果 —— 这一步才把 c 的页从 device 逐页搬回 host
        double got = 0;
        for (int i = 0; i < n; i++) got += (double)c[i];

        auto t1 = std::chrono::steady_clock::now();
        // ===== 计时窗口结束 =====

        um_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        um_ok = fabs(got - want) <= 1e-3 * (1.0 + fabs(want));

        CUDA_CHECK(cudaFree(a));
        CUDA_CHECK(cudaFree(b));
        CUDA_CHECK(cudaFree(c));
    }

    free(h_a);
    free(h_b);

    // ---------- 对比输出 ----------
    printf("Explicit:   %.1f ms  %s\n", explicit_ms,
           explicit_ok ? "PASS" : "FAIL");
    printf("Unified:    %.1f ms  %s\n", um_ms,
           um_ok ? "PASS" : "FAIL");
    printf("Speedup (explicit / unified): %.2fx\n", explicit_ms / um_ms);

    REPORT(explicit_ok && um_ok);
    return 0;
}
