#!/usr/bin/env python3
"""Bonus（EXPERIMENT · Optional）：matmul 性能复现脚本。

把 handout 里 Bonus 的三段一次跑全：
  (a) naive CUDA    cuda/bonus/matmul.cu        （BS = 8/16/32）
  (b) Tiled Triton  kernels/matmul_triton.py    （多组 tile + cuBLAS 对照）
  (c) TileLang      kernels/tilelang_matmul.py  （即补全的 7.6）

需要 GPU + tilelang，在 assignment01/ 目录下运行：
    srun -p lcpu-infra --time=00:10:00 --gres=gpu:1 .venv/bin/python scripts/bench_bonus_matmul.py

预期输出怎么读：
  - naive 三段只有个位数 TFLOPS（5090 fp16 峰值约 210 TFLOPS，差着两个量级）；
  - Triton / TileLang 都到 150+ TFLOPS 量级，这就是"线程到 tile"重写带来的差距；
  - Triton bench 末尾自带 cuBLAS 基线，当"厂商天花板"参照；
  - TileLang bench 最后一条配置 (128,256,64,256,3) 会崩：144KB shared >
    单 block 上限（5090 约 100KB）。这是预期行为，本脚本会捕获，前面的数据仍有效。

注意：naive 每段要先在 CPU 上算 1024^3 次乘加做参考对拍，各会卡几秒，不是死机。
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CUDA_DIR = ROOT / "cuda"

# 让 `kernels` 包能被 import（脚本在 scripts/ 下，默认 sys.path 只含脚本目录）
sys.path.insert(0, str(ROOT))

# 5090 = sm_120。换机器时改成对应 arch，或用 -arch=native。
ARCH = "sm_120"
NAIVE_BS = [8, 16, 32]


def run_naive() -> None:
    """(a) 编译并运行 naive CUDA，BS 取 8/16/32（输出统一换算成 TFLOPS）。"""
    print("=" * 60, flush=True)
    print("(a) naive CUDA  cuda/bonus/matmul.cu  (BS = 8/16/32)", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)
    (CUDA_DIR / "bin" / "bonus").mkdir(parents=True, exist_ok=True)
    src = CUDA_DIR / "bonus" / "matmul.cu"
    for bs in NAIVE_BS:
        exe = CUDA_DIR / "bin" / "bonus" / f"matmul_bs{bs}"
        compile_cmd = ["nvcc", "-O2", "-std=c++17", "-I.",
                       f"-arch={ARCH}", f"-DBS={bs}",
                       "-o", str(exe), str(src)]
        subprocess.run(compile_cmd, cwd=CUDA_DIR, check=True)
        print(f"--- BS={bs} ---", flush=True)
        # matmul.cu 原生打印 GFLOPS，这里换算成 TFLOPS 与后面两段统一
        out = subprocess.run([str(exe)], cwd=CUDA_DIR, check=True,
                             capture_output=True, text=True).stdout
        out = re.sub(r"([0-9]+\.?[0-9]*) GFLOPS",
                     lambda m: f"{float(m.group(1)) / 1000:.1f} TFLOPS", out)
        sys.stdout.write(out)
        sys.stdout.flush()


def run_triton() -> None:
    """(b) Triton bench（含 cuBLAS 基线）。"""
    print("=" * 60, flush=True)
    print("(b) Tiled Triton  kernels/matmul_triton.py", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)
    from kernels.matmul_triton import bench as bench_triton
    bench_triton()


def run_tilelang() -> None:
    """(c) TileLang bench（最后一条配置会因 shared 超限崩溃，属预期）。"""
    print("=" * 60, flush=True)
    print("(c) TileLang  kernels/tilelang_matmul.py  (即补全的 7.6)", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)
    from kernels.tilelang_matmul import bench as bench_tilelang
    try:
        bench_tilelang()
    except Exception as e:  # noqa: BLE001 —— 预期捕获 shared 超限
        print()
        print(f"[提示] TileLang bench 在此处中断：{type(e).__name__}: {e}")
        print("这是最后一条配置 (128,256,64,256,3) 超出单 block shared 上限的预期崩溃，")
        print("上方已打印的各配置数据仍然有效。")


def main() -> None:
    run_naive()
    run_triton()
    run_tilelang()


if __name__ == "__main__":
    main()
