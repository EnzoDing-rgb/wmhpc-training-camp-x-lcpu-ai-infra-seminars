"""问题 1.6（选做）：SIMT Simulator —— 一个 warp 的执行模拟器。

不需要 GPU

contract: 实现 run(program) -> (regs, cycles)
- warp 固定 32 个 lane，lane i 的寄存器初值为 i（int）；
- program 是指令列表，指令是元组，共三种：
    ("add", k)   active lanes 的 reg += k，1 cycle
    ("mul", k)   active lanes 的 reg *= k，1 cycle
    ("if_lt", t, then_prog, else_prog)
        reg < t 的 lane 走 then_prog，其余走 else_prog。
        模拟器先带 mask 执行 then_prog，再带 mask 的补集执行
        else_prog，然后汇合。某一支没有 active lane 时整支跳过、
        不计拍。嵌套指令照常计拍（divergence 的代价就在这里）。
        if_lt 这条指令本身不计拍，拍数只来自实际执行到的 add / mul。
- 返回值 regs 是 32 个 lane 的最终寄存器值（list），cycles 是总拍数。

通过 pytest tests/test_simt_sim.py 即为完成。
"""

from typing import List, Tuple

# ══════════════════════════════════════════════════════════════════════
#  类型别名
# ══════════════════════════════════════════════════════════════════════

# Mask: 长度为 32 的布尔列表，表示"哪些 lane 当前参与执行"。
#       mask[i] == True  → lane i 在干活
#       mask[i] == False → lane i 被屏蔽，指令对它无效，也不为它计拍
#
#  例: mask = [True] * 32           # 全部 32 个 lane 都活跃（程序入口）
#  例: mask = [True, False, True, False, ...]  # 只有部分 lane 活跃（divergence 后）
Mask = List[bool]

# Prog: 指令序列。每条指令是一个 tuple，具体形式见文件头 docstring。
Prog = List[Tuple]


# ══════════════════════════════════════════════════════════════════════
#  run()
# ══════════════════════════════════════════════════════════════════════

def run(program: Prog) -> Tuple[List[int], int]:
    """在 32-lane warp 上执行 program，返回 (regs, cycles)。

    参数:
        program: 指令列表，如 [("add", 3), ("mul", 2)]

    返回:
        regs:   长度为 32 的 int 列表，regs[i] 是 lane i 的最终寄存器值
        cycles: 总共消耗的拍数（if_lt 本身不计拍）
    """

    # regs[i] = lane i 的寄存器当前值，初始值 = 所在 lane 编号
    # 例: 初始时 regs = [0, 1, 2, 3, ..., 31]
    regs: List[int] = list(range(32))

    # ── 内部递归函数 ──────────────────────────────────────────────

    def exec_prog(prog: Prog, mask: Mask) -> int:
        """在给定 mask 下执行指令序列 prog，返回消耗的拍数。

        这是整个模拟器的核心：递归地在当前活跃 lane 子集上执行代码。
        - 对于 add/mul：只修改 mask[i]==True 的那些 lane
        - 对于 if_lt：把 mask 拆成两支，分别递归执行，然后汇合

        参数:
            prog: 要执行的指令列表
            mask: 当前活跃的 lane 掩码————只有 mask[i]==True 的 lane 才被修改

        返回:
            本段 prog 消耗的总拍数（if_lt 自身不计拍，只计内部的 add/mul）
        """
        cycles: int = 0

        for insn in prog:
            op = insn[0]

            # ── add / mul：统一处理 ──────────────────────────────
            if op == "add" or op == "mul":
                k: int = insn[1]

                # any_active: 这条指令 "有没有至少一个 lane 在执行"。
                #
                # 为什么要检查？因为 GPU 硬件上，如果一条指令所有 lane
                # 都被 mask 屏蔽了，这条指令不会被发射，也就不消耗 cycle。
                #
                # 例: mask = [True, False, False, ...] （只有 lane 0 活跃）
                #     → lane 0 执行 add/mul，any_active 变成 True，+1 cycle
                #
                # 例: mask = [False] * 32 （全部被屏蔽）
                #     → 循环里一次 mask[i] 都没命中，any_active 保持 False
                #     → 不累加 cycles（这条指令被跳过）
                any_active: bool = False

                for i in range(32):
                    if mask[i]:
                        any_active = True
                        if op == "add":
                            regs[i] += k
                        else:  # op == "mul"
                            regs[i] *= k

                if any_active:
                    cycles += 1

            # ── if_lt：条件分支，SIMT divergence 的来源 ──────────
            elif op == "if_lt":
                _, t, then_prog, else_prog = insn

                # 根据 "regs[i] < t ?" 把当前 mask 拆成两份:
                #
                # then_mask[i] = mask[i] AND (regs[i] < t)
                #   → 条件成立的 lane，走 then 分支
                #
                # else_mask[i] = mask[i] AND (regs[i] >= t)
                #   → 条件不成立的 lane，走 else 分支
                #
                # 注意: already-inactive 的 lane (mask[i]==False) 两边都不参与
                #
                # 例: regs = [0, 1, 2, ..., 31], t = 10, mask = [True]*32
                #   then_mask: lane  0~9  → True,  lane 10~31 → False
                #   else_mask: lane  0~9  → False, lane 10~31 → True
                #
                # 例: regs = [0, 1, ...], mask 中只有 lane 0~15 活跃, t = 8
                #   then_mask: lane  0~7  → True, 其余 → False
                #   else_mask: lane  8~15 → True, 其余 → False
                #   (lane 16~31 本来就不活跃，两边都是 False——正确)
                then_mask: Mask = [mask[i] and regs[i] < t for i in range(32)]
                else_mask: Mask = [mask[i] and not then_mask[i] for i in range(32)]

                # 先执行 then 分支，再执行 else 分支————这就是 divergence
                # 的代价：原本并行执行的 32 个 lane 不得不串行执行两个分支。
                #
                # 如果某一分支没有活跃 lane（any(...) == False），整支
                # 跳过、不计拍——这是 GPU 硬件的优化：空分支不发射指令。
                if any(then_mask):
                    cycles += exec_prog(then_prog, then_mask)
                if any(else_mask):
                    cycles += exec_prog(else_prog, else_mask)

            else:
                raise ValueError(f"unknown op: {op}")

        return cycles

    # ── 入口：全体 32 个 lane 初始都活跃 ─────────────────────────
    # 例: exec_prog([("add", 1), ("mul", 2)], [True]*32)
    #     → 全部 lane reg += 1（1 cycle），全部 lane reg *= 2（1 cycle）
    #     → 共 2 cycles
    total_cycles: int = exec_prog(program, [True] * 32)
    return regs, total_cycles
