"""7.7 (Optional)：你的 TileLang softmax vs torch.softmax 性能对拍。

运行（集群、有 GPU，在 assignment01/ 目录下）：
    srun -p lcpu-infra --time=00:10:00 --gres=gpu:1 .venv/bin/python bench_softmax.py

--------------------------------------------------------------------------
先想清楚为什么这是带宽瓶颈，再看数字（否则数据没有解释力）：
--------------------------------------------------------------------------
softmax 对每个元素只做几件算术（一次 max 归约、一次 exp、一次 sum 归约、
一次除法），FLOP 量几乎为零。但每个元素必然经历：
    从全局读一次（X → fragment）  +  写回一次（fragment → Y）
所以内存搬运量是确定且不可省掉的：
    总搬运字节 = 2 × M × N × 4        （fp32，每元素 4 字节）

既然算术不吃力、搬运是硬开销，理论下限就是带宽说了算，不是 FLOP：
    t_limit = 2·M·N·4 / BW_peak

RTX 5090 的 HBM 带宽标称约 1792 GB/s。于是：
    - 每个 shape 都能先口算出一个"理论上限时间"（下面表里会打出来）；
    - 拿实测时间反推"有效带宽"，看你的 kernel 吃到了理论峰值的百分之几；
    - 95% 以上 = 已经贴着墙跑，再优化空间很小；
    - 明显低于 80% = 多半有多余的读/写（比如把结果从 shared/fragment
      写回全局又读回来），去找那些多余的搬运。

参照系：torch.softmax 是厂商手调的 kernel（向量化 load、更好的 unroll），
它大概率比你的快一点。但注意方向——它是被用来证明"你的实现离带宽墙
还有多远"的尺子，不是用来碾压你的。两者都该在几十微秒量级。
--------------------------------------------------------------------------
"""

import torch
import triton

from kernels.tilelang_softmax import softmax

# RTX 5090 理论显存带宽（GB/s）。可改：想验证不同假设就用不同值。
BW_GBS = 1792.0

# 三个 shape，M×N 都约等于 4.2M 个元素（总搬运量一致），
# 这样行宽变化的影响可以直接横向比较，而不会被数据量差异干扰。
SHAPES = [
    (16384, 256),   # 行多列窄
    (4096, 1024),   # 行宽 1024
    (1024, 4096),   # 行宽 4096（N_pad = 4096，block_M 退化为 1）
]


def traffic_bytes(M: int, N: int) -> int:
    """一次 softmax 调用必须搬运的最小字节数：读 X 一遍 + 写 Y 一遍。"""
    return 2 * M * N * 4  # fp32


def main() -> None:
    print(f"理论带宽假设：{BW_GBS:.0f} GB/s\n")
    header = (f"{'shape':>14} | {'实测 ms':>8} | {'有效 GB/s':>10} "
              f"| {'占理论%':>7} | {'理论上限 ms':>10}")
    print(header)
    print("-" * len(header))

    for M, N in SHAPES:
        torch.manual_seed(0)
        x = torch.randn(M, N, device="cuda")

        # 1) 正确性对拍（编译第一次会慢，正常）
        y = softmax(x)
        torch.testing.assert_close(y, torch.softmax(x, dim=-1), atol=1e-5, rtol=1e-5)

        # 2) 实测（do_bench 内部自带 warmup，编译耗时不会计入）
        ms_tl = triton.testing.do_bench(lambda: softmax(x))
        ms_torch = triton.testing.do_bench(lambda: torch.softmax(x, dim=-1))

        tb = traffic_bytes(M, N)
        t_lim_ms = tb / (BW_GBS * 1e9) * 1e3  # 理论上限，毫秒
        bw_tl = tb / (ms_tl * 1e-3) / 1e9      # 实测反推有效带宽
        bw_torch = tb / (ms_torch * 1e-3) / 1e9

        print(f"{str((M, N)):>14} | {ms_tl:8.3f} | {bw_tl:10.1f} "
              f"| {bw_tl / BW_GBS * 100:6.1f}% | {t_lim_ms:10.3f}")
        print(f"{'torch':>14} | {ms_torch:8.3f} | {bw_torch:10.1f} "
              f"| {bw_torch / BW_GBS * 100:6.1f}% | ")

    print()
    print("怎么读这张表：")
    print("  - '理论上限 ms' 是带宽墙——任何实现都不可能低于它。")
    print("  - '占理论%' 越高越贴墙；你的实现和 torch 谁高谁低不重要，")
    print("    重要的是两者都该在 90% 上下，而不是一个 95% 一个 30%。")
    print("  - 若你的明显偏低，优先怀疑：有没有多读了 X、多写了 Y，")


if __name__ == "__main__":
    main()
