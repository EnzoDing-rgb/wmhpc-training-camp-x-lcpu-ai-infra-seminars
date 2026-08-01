// 问题 2.9：从零写 SAXPY —— y ← 2.0 · x + y（单精度）。
// 要求：不 include common.h，错误检查宏和 cudaEvent 计时全部自己写。
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// ==================== 错误检查宏（手写） ====================
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                    \
                    cudaGetErrorName(err_), __FILE__, __LINE__,                \
                    cudaGetErrorString(err_));                                 \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CUDA_CHECK_KERNEL()                                                    \
    do {                                                                       \
        CUDA_CHECK(cudaGetLastError());                                        \
        CUDA_CHECK(cudaDeviceSynchronize());                                   \
    } while (0)

// ==================== GpuTimer（RAII 封装 cudaEvent，手写） ====================
struct GpuTimer {
    cudaEvent_t start_, stop_;
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    float stop_ms() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

// ==================== kernel ====================
__global__ void saxpy(const float *x, float *y, int n, float a) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] = a * x[idx] + y[idx];
}

// ==================== main ====================
int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <n>\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    if (n < 0) {
        fprintf(stderr, "n must be >= 0\n");
        return 1;
    }

    // n = 0 特判：0 个 block 的 launch 是非法的
    if (n == 0) {
        printf("SUM=0\n");
        return 0;
    }

    size_t bytes = (size_t)n * sizeof(float);

    // ---------- host 端分配 + 按公式填充 x, y ----------
    float *h_x = (float *)malloc(bytes);
    float *h_y = (float *)malloc(bytes);
    for (int i = 0; i < n; i++) {
        h_x[i] = ((i % 2048) - 1024) * 0.5f;
        h_y[i] = (float)((i % 1024) - 512);
    }

    // ---------- device 端分配 + 搬运 ----------
    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    // ---------- launch kernel（GPU 侧计时） ----------
    GpuTimer timer;
    timer.start();

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    saxpy<<<blocks, threads>>>(d_x, d_y, n, 2.0f);
    CUDA_CHECK_KERNEL();

    float ms = timer.stop_ms();

    // ---------- 拷回结果 ----------
    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));

    // ---------- double 累加 ----------
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += (double)h_y[i];

    // ---------- 输出 ----------
    printf("SUM=%.0f  n=%d  kernel=%.3f ms\n", sum, n, ms);

    // ---------- 清理 ----------
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    free(h_x);
    free(h_y);

    return 0;
}
