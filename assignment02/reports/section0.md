# Section 0 · 环境与峰值 —— 0.1 实验记录 + 0.2 峰值推导

> 配套代码:`cuda/m0_env/01_first_mma.cu`
> 完整讲解(心智模型 + 逐行拆解 + 交互图):`reports/section0.html`
> 本文 = 实验记录 + 现象解释,讲解细节见 HTML。

## 0. 实验环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA RTX 5090(sm_120a,32GB GDDR7) |
| 驱动 | 580.142 · CUDA 13.0 |
| 编译 | `cuda/Makefile`,显式 `-gencode arch=compute_$(ARCH),code=sm_$(ARCH)` |
| 拿卡 | `srun --partition=lcpu-infra --gres=gpu:1 --time=00:15:00 --pty bash` |

> 注:本集群只有 5090,没有 B300(sm_100)。所以只测了 **5090 的正确架构(120a)** 和 **默认架构(100f)的对照**;B300 部分无法在本环境完成。

---

## 1. 正确架构:`ARCH=120a`(5090 匹配)→ PASS

```bash
cd /home/lcpu/39112061/ai-infra/wmhpc-training-camp-x-lcpu-ai-infra-seminars/assignment02/cuda
ARCH=120a make -B run/m0_env/01_first_mma
```

实际输出:

```
nvcc -O2 -std=c++17 -I. --expt-relaxed-constexpr -gencode arch=compute_120a,code=sm_120a -o bin/m0_env/01_first_mma m0_env/01_first_mma.cu
./bin/m0_env/01_first_mma
D[0][0]=2 D[0][7]=2 D[15][0]=2 D[15][7]=2
PASS
```

**说明**:fatbin 里装的是 `sm_120a` 的 SASS,与 5090 精确匹配 → 驱动直接运行,输出 `PASS`。
这一步证明了 **工具链 + 5090 都能发出 Tensor Core 指令**——后面所有实现的性能达成率都以它为准。

---

## 2. 不匹配架构:默认 `ARCH=100f`(B300 系)→ 报错

```bash
make -B run/m0_env/01_first_mma     # 不带 ARCH = 默认 100f
```

实际输出:

```
nvcc -O2 -std=c++17 -I. --expt-relaxed-constexpr -gencode arch=compute_100f,code=sm_100f -o bin/m0_env/01_first_mma m0_env/01_first_mma.cu
./bin/m0_env/01_first_mma
CUDA error cudaErrorNoKernelImageForDevice at m0_env/01_first_mma.cu:185: no kernel image is available for execution on the device
make: *** [Makefile:42: run/m0_env/01_first_mma] Error 1
```

**现象**:`cudaErrorNoKernelImageForDevice` —— "设备上找不到可执行的 kernel 镜像"。
这不是程序逻辑错(同一个源码,120a 能 PASS),而是**这份二进制里没有 5090 能用的东西**。

---

## 3. 用 `cuobjdump` 拆开 fatbin,找证据

```bash
cuobjdump --list-elf bin/m0_env/01_first_mma            # fatbin 里装了哪些架构的 SASS
cuobjdump --list-ptx bin/m0_env/01_first_mma            # 有没有内嵌 PTX
cuobjdump -sass      bin/m0_env/01_first_mma | grep HMMA  # SASS 汇编里的矩阵乘指令
```

实际输出:

```
$ cuobjdump --list-elf bin/m0_env/01_first_mma
ELF file    1: 01_first_mma.1.sm_100.cubin
ELF file    2: 01_first_mma.2.sm_100.cubin

$ cuobjdump --list-ptx bin/m0_env/01_first_mma
cuobjdump info : No PTX file found to extract from '.../assignment02/cuda/bin/m0_env/01_first_mma'.

$ cuobjdump -sass bin/m0_env/01_first_mma | grep HMMA
        /*0230*/                   HMMA.16816.F32 R4, R4, R12, RZ ;          /* 0x0000000c0404723c */
```

**读证据,三句话**:

1. `--list-elf`:fatbin 里**只有 `sm_100`(B300 系)的 SASS**,没有 sm_120a;
2. `--list-ptx`:**没有内嵌任何 PTX**;
3. `-sass`:**`mma.sync` 在 SASS 层的真面目就是 `HMMA.16816.F32`** —— Tensor Core 的机器指令(`.16816` = 16×8×16)。

所以 100f 编译的 fatbin = "**只有 B300 的机器码 + 没有可 JIT 的 PTX**"。5090 运行时:① 找不到 sm_120a 的 SASS;② 没有 PTX 可以现场 JIT → 直接第 ③ 种结局:报错。

---

## 4. 看 PTX 层:`make ptx/...`

```bash
ARCH=120a make -B ptx/m0_env/01_first_mma
grep -n mma m0_env/01_first_mma.ptx
```

实际输出:

```
83:	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%f1,%f2,%f3,%f4}, {%r1,%r2,%r3,%r4}, {%r5,%r6}, {%f8,%f8,%f8,%f8};
```

`.cu` 里那行内联汇编 → 编译成这条 **PTX**(可移植汇编)→ 再被 `ptxas` 编成 **SASS**(`HMMA.16816.F32`)→ 装进 fatbin。三层亲眼可见。

---

## 5. 结论:fatbin / PTX / SASS / JIT 心智模型

```
.cu(内联 asm:mma.sync ...)
  │ nvcc
  ▼
PTX   —— 可移植汇编,不是机器码(mma.sync 在这一层;本工程 ptx/ 目标单独生成)
  │ ptxas
  ▼
SASS  —— 真机器码(HMMA.16816.F32),显卡 SM 直接执行,装进 fatbin
  │ 运行时,驱动拆 fatbin:
  ├─ 有当前卡的 SASS    → 直接跑
  ├─ 没 SASS 但有 PTX    → JIT:驱动现场把 PTX 编成 SASS 再跑(能跑,启动慢)
  └─ 都没有              → cudaErrorNoKernelImageForDevice
```

**本实验的关键点(第一层)**:Makefile 用 `-gencode arch=compute_$(ARCH),code=sm_$(ARCH)`,这个写法**只嵌 SASS、不嵌 PTX**。
所以默认 build 的 fatbin 里没有 PTX 兜底 → ARCH 不匹配就直接报错。

**补充实验(第二层,更深)**:即使嵌了 PTX,`compute_100f` 的 PTX 是 **family-specific 虚拟架构(B300 家族专用)**,只在 **sm_100 家族**的卡上能被 JIT。5090 是 **sm_120a(消费级 Blackwell 家族)**,跨家族不认——所以"SASS+PTX 都嵌"或"只嵌 PTX"的 100f 版本在 5090 上**照样报错**。只有**同家族**的 PTX(`compute_120a`)才能被 5090 JIT。

为了验证 JIT 确实存在,编译了"只嵌 `compute_120a` PTX、零 SASS"的版本(`-gencode arch=compute_120a,code=compute_120a`),`cuobjdump --list-elf` 确认无任何 cubin,但运行 **PASS** —— 零机器码也能跑,唯一解释就是驱动现场 JIT。**这是 JIT 的铁证。**

四种 build 的完整对照:

| build | fatbin 内容 | 在 5090(sm_120a)上 | 原因 |
|---|---|---|---|
| `ARCH=120a`(SASS only) | sm_120a SASS | ✅ **PASS** | 精确匹配 |
| 默认 `100f`(SASS only) | sm_100 SASS,无 PTX | ❌ 报错 | 不匹配 + 无 PTX |
| `100f` SASS+PTX | sm_100 SASS + compute_100f PTX | ❌ **报错** | PTX 跨家族不 JIT |
| `100f` PTX only | compute_100f PTX | ❌ **报错** | 同上 |
| `120a` PTX only | compute_120a PTX | ✅ **PASS** | 同家族 JIT |

**结论**:带后缀的家族架构(100f/120a)**连 PTX 兜底都跨不了家族**——ARCH 必须和显卡精确匹配,一个家族是一个世界。这也印证了 Makefile 注释里的坑:为什么用显式 `-gencode` 而不用 `-arch=sm_XXXa` 简写(简写会把 PTX target 展开错,ptxas 直接拒绝)。**ARCH 选错 = 二进制对这块卡是废的。**

---

*实验时间:2026-08-25 · gj-5090-1 · 全程占用 1 块 RTX 5090*

---

## 0.2 · Tensor Core 理论峰值推导(DERIVE)

> 方法:参考课件 S018–S019 的 A100 推导法——**从 Tensor Core 结构出发**:
> **峰值 = SM 数 × (每 SM 的 TC 数 × 每 TC 每拍的 MAC 数) × 时钟频率**,再与 datasheet 对照。
> 环境说明:本集群只有 5090,B300 部分按官方 datasheet 计算,无实测。

### 0. 口径声明(开算前钉死,避免 2 倍级错位)

| 决策 | 采用 | 说明 |
|---|---|---|
| dense / sparse | **dense** | 2:4 稀疏只在 datasheet 对照时出现 |
| boost / base 频率 | **boost** | 官方数字按 boost 算;换 base 会差 ~1.19× |
| FMA 计 FLOP | **2**(乘 + 加各一) | 按 1 算则所有峰值减半 |
| fp8 / fp4 峰值 | 按 dtype 宽度关系估算 | 官方只公布"稀疏 fp4"一个数 |

### 1. 结果总表

| 量 | 5090 | B300 |
|---|---|---|
| bf16 FLOP/cycle/SM | 1024 | 3.5e15 ÷ (160×f),f 未公布 |
| bf16 峰值(TFLOPS) | **419** | **3,500**(3.5 PFLOPS) |
| fp8 峰值(TFLOPS) | 838 | 7,000 |
| fp4 峰值(TFLOPS) | 1,676 | 15,000 |
| datasheet 对照与口径差异 | 官方 3,352 TOPS = FP4+2:4 稀疏,dense 即 1,676;fp8/bf16 为宽度关系估算 | 官方 dense 值;fp4=15 而非严格 2×7=14;时钟未公布 |
| 显存带宽(GB/s) | 1,792 | 8,000 |
| 机器平衡点(FLOP/byte,bf16) | ≈234 | ≈437 |

### 2. 推导方法模板:A100 的 312 是怎么从 Tensor Core 结构推出来的

先按 S018–S019 把 A100 完整推一遍,作为 5090 的模板。核心是不拿"每周期 FLOP"当已知数,而是从**单个 Tensor Core 的阵列结构**出发:

```
硬件:108 SM、每 SM 4 个 Tensor Core、boost 1.41 GHz。

① 单个 TC 的阵列:8 行 × 8 列 = 64 个 MAC 单元,每拍(cycle)沿 K 吞 4 个值
   → 64 × 4 = 256 次 fp16 乘加 = 512 FLOP/cycle/TC。

② 用指令形状对账(证明结构自洽):
   mma.m16n8k16 = 16×8×16 = 2048 FMA = 4096 FLOP(一条 warp 指令)
   单 TC 上:M=16 分 2 趟(阵列 8 行深)、K=16 分 4 拍(每拍 4 个 k)→ 8 拍
   8 拍 × 256 MAC/拍 = 2048 MAC ✓  正好等于指令的 2048 FMA

③ 每 SM = 4 个 TC 并行 → 4 × 256 = 1024 MAC/cycle = 2048 FLOP/cycle/SM

④ 峰值 = 108 SM × 2048 FLOP/cycle × 1.41e9 = 311.9e12 ≈ 312 TFLOPS ✓(官方 312)

注:312 是纯 Tensor Core 的数——CUDA Core 的 FP32 只有 19.5 TFLOPS
(108×64×2×1.41e9),不在这个峰值里。
```

### 3. 5090(GB202 · sm_120a)计算过程

```
硬件:170 SM、每 SM 4 个 Tensor Core、boost 2.407 GHz(base 2.017)、GDDR7 1,792 GB/s。

① 单个 TC:消费级 Blackwell 第 5 代,每 SM 每拍 512 次 fp16 乘加(4 个 TC,每 TC 128 次)
   = 1024 FLOP/cycle/SM。
   "翻倍"有依据:4090(Ada)每 SM 每拍 256 MAC = 512 FLOP/cycle
   (165e12 ÷ 128 SM ÷ 2.52e9 = 512),Blackwell 消费级把它翻倍 → 1024 FLOP/cycle/SM。

② 正向推导:
   170 SM × 1024 FLOP/cycle × 2.407e9 = 419.0 TFLOPS ✓
   (base 口径:170 × 1024 × 2.017e9 ≈ 351 TFLOPS)

③ datasheet 交叉验证(反方向,两条路撞同一个数):
   官方 3,352 TOPS = FP4 + 2:4 稀疏
   ÷2(去稀疏)→ FP4 dense = 1,676 → ÷2 → FP8 = 838 → ÷2 → BF16 = 419
   419e12 ÷ (170 × 2.407e9) = 1024 FLOP/cycle/SM ← 和 ② 完全一致

④ 指令级对账:m16n8k16 = 4096 FLOP,每 SM 每 4 拍退休一条(4096 ÷ 1024)。
```

**datasheet 对照与口径差异**:官方只给 "3,352 AI TOPS"(FP4、带 2:4 稀疏);
dense 链(1676/838/419)是"去稀疏 + 宽度关系"的结果,2 倍差距全部来自 2:4 稀疏因子;
官方不公布消费级 fp8/bf16 的 dense 数,838/419 是估算值;③ 的反推和 ② 的正向推导撞出同一个 1024,说明口径自洽。

### 4. B300(sm_100)计算过程

硬件参数:160 SM、HBM3e 8 TB/s、288 GB。官方 dense 数(无实测):

```
BF16 = 3.5 PFLOPS
FP8  = 7 PFLOPS   (≈ 3.5 × 2)
FP4  = 15 PFLOPS  (≈ 7 × 2;官方口径 15 而非严格 14)
```

每 SM 每周期:B300 的时钟 NVIDIA 未公布,只能反解
`bf16 FLOP/cycle/SM = 3.5e15 ÷ (160 × f)`;
若取 f ≈ 2.0 GHz(第三方常见口径)→ ≈ 10,938 FLOP/cycle/SM。

**口径差异**:fp4 官方值 15 与严格 2× 关系(14)有出入;所有数字若开稀疏再 ×2。

### 5. 机器平衡点(bf16)

```
5090:419e12 ÷ 1.792e12 ≈ 234 FLOP/byte
B300:3.5e15 ÷ 8e12     ≈ 437 FLOP/byte
```

### 6. 差距的含义:为什么 M2–M4 要从数据供给路径优化

与单条 mma 的计算强度(3.2 FLOP/byte,完整推导见文末第 6 节)比较:

```
5090:234 ÷ 3.2 ≈ 73 倍
B300:437 ÷ 3.2 ≈ 137 倍
```

含义:机器平衡点 = 硬件"每搬运 1 字节能支撑的 FLOP"。单条 mma 自己搬的数据量
只能支撑 3.2 FLOP/byte,比硬件能力低两个数量级——如果每条 mma 都直接从 global
取自己的数据,带宽必然卡死算力,tensor core 每周期都在等数据。
解法分两层:① **分块 tiling**,让同一份数据被多条 mma 复用,摊薄每字节的搬运次数;
② 让搬移本身更快、和计算重叠:M2 的 descriptor/swizzle(smem 供数无 bank 冲突)、
M3 的 TMA(异步搬运)、M4 的 pipeline(搬运与计算重叠)。

### 7. 重点:单条 mma 的计算强度 3.2 FLOP/byte 是怎么算出来的

> 这就是 0.2 里那个"3.2"——它是 **FLOP/byte(计算强度)**,不是 TFLOPS。
> 单条指令没有"每秒",只有"每搬运 1 字节能支撑几次运算"。

指令:`mma.sync.m16n8k16` fp16(A、B 是 fp16,D 是 fp32 累加)。
S016 口径:**分子 = 2·M·N·K(FLOP),分母 = A、B 读入 + D 写回的字节总和**。

```
分子:2·M·N·K = 2 × 16 × 8 × 16 = 4096 FLOP     (FMA 计 2 FLOP)

分母:
  A 读:16×16 个 fp16 = 256 元素 × 2 B = 512 B
  B 读:16×8 个 fp16  = 128 元素 × 2 B = 256 B
  D 写:16×8 个 fp32  = 128 元素 × 4 B = 512 B
  ────────────────────────────────────────────
  合计 = 1280 B

强度 = 4096 FLOP ÷ 1280 B = 3.2 FLOP/byte
```

**两点澄清(答辩会被问):**

1. **分母不含 C**。C 是累加器预载(16×8×4 B = 512 B),S016 把它算进 D 的寄存器状态,
   不单独计搬运;如果把 C 也计入,强度 = 4096/1792 ≈ 2.29,就和 S016 的 3.2 对不上了;
2. **它和机器平衡点的口径一致**才能比较:平衡点也是"FLOP/byte",
   所以 3.2(单条指令,无复用)与 234/437(整机)的差距就是"需要靠复用摊薄的倍数"。

---

## 0.3 · 概念判断(CONCEPT)

判断下列说法是否正确,并给出一句理由。

### (a) 一条 mma 的计算强度,分子是 2MNK,分母按 A、B 读入与 D 写回的字节总和计(S016 的口径)。

**正确。**

理由:S016 口径正是 `2MNK / (A 读 + B 读 + D 写)`,m16n8k16 fp16 算出来就是 4096/1280 = **3.2 FLOP/byte**(推导见 0.2 第 7 节);分母不含 C——C 是累加器预载,算进 D 的寄存器状态,不单独计搬运。

### (b) mma.sync 是 warp 级协作指令:32 个 lane 各持 fragment 的一部分,要求全 warp 一致地执行这条指令;有 lane 发散时行为未定义。

**正确。**

展开——为什么"必须是 warp 级协作":

1. **fragment 本身就是碎的**:m16n8k16 里每个 lane 只拿 A 的 2×4、B 的 2×2、D 的 2×2(见 0.1 / 模块 1 的 fragment 布局)。任何一块单独看都是残片,只有 32 个 lane 的数据拼在一起,才凑成一个完整的 16×8×16 矩阵乘。
2. **`sync` 就是这个"必须一起走"的语义**:32 个 lane 要在同一时刻、以同一条指令,把自己的残片喂给 tensor core 阵列。指令本身是 warp 级同步点——它隐含要求执行这条指令时 warp 是收敛的(converged / aligned)。
3. **发散 = 未定义**:如果 mma.sync 出现在按 lane 分叉的分支里(比如 `if (lane < 16)` 内部),部分 lane 不在这条指令上,行为未定义(PTX 文档明文)。

与实验的呼应:0.1 的 `01_first_mma` 开 `<<<1, 32>>>` 一个 warp,32 个 lane 一起发一条 m16n8k16,才得到正确的 D[16×8]。把这条指令包进按 lane 分叉的 `if` 里,结果就不可信。

### (c) 增大 mma 的形状 M/N/K 能提高单条指令的计算强度,而且没有代价,所以指令形状越大越好。

**错(前半句对,结论错)。**

展开——前半句为什么对(增大形状确实提高单条强度):

单条强度 = 2MNK / (A 读 + B 读 + D 写)。fp16、f32 累加时,分母 = `2MK(A) + 2KN(B) + 4MN(D)`。
分子是 M·N·K 的"**体积**",分母是各矩阵的"**表面积**"——体积比表面积长得快:

```
m16n8k16: 4096 / 1280        = 3.2
m16n8k32: K 翻倍 → 8192/2048  = 4.0    ↑
m32n8k16: M 翻倍 → 8192/2304  ≈ 3.56   ↑
```

展开——后半句为什么错:

1. **形状由 ISA/硬件固定**:你能选的只有指令集列出的那几个形状(m16n8k8/16/32…),不能自定义;tensor core 阵列宽度固定,每周期 MAC 数是硬顶(A100 每 SM 每周期 2048 FLOP、5090 是 1024)——更大的指令只是覆盖更大的 tile、占更多周期,**不改变每周期算力**。
2. **有真实代价**:更大的 fragment = 每个 lane 更多寄存器、tile 要更多共享内存、累加器更多、寄存器压力升高 → 占用下降,可能更慢。
3. 所以正确的是:大形状能**摊薄指令发射/寻址这类固定开销**、提高单条指令强度,但要在"tile 够大"和"资源装得下"之间权衡——这正是 M4 里选 tile 尺寸要做的事。

### (d) 只要单条 mma 的计算强度低于机器平衡点,GEMM kernel 就不可能逼近计算峰值。

**错。**

展开——它拿错了比较对象。有两个"计算强度":

```
① 单条 mma 的强度:3.2 FLOP/byte —— 一条指令自己的 A/B/D 搬运,零复用。
② kernel 的强度(arithmetic intensity):整段 GEMM 从全局内存搬的字节 / 总 FLOP。
   数据在寄存器/smem 里被多条 mma 复用,每搬 1 字节对应很多 FLOP。
```

Roofline 的结论:能不能逼近峰值,看的是 **②**,不是 ①——需要 kernel 强度 ≥ 机器平衡点(5090 是 234、B300 是 437)才可能打满算力。

分块 tiling 怎样把 ② 从 3.2 拉上去——先用一个能写完的小例子,再放大到 128。

**① 小例子:A 是 2×3,B 是 3×4,算出 D 是 2×4**

```
A (2行×3列)          B (3行×4列)              D (2行×4列)
[ a00 a01 a02 ]       [ b00 b01 b02 b03 ]      [ d00 d01 d02 d03 ]
[ a10 a11 a12 ]   ×   [ b10 b11 b12 b13 ]  =   [ d10 d11 d12 d13 ]
                      [ b20 b21 b22 b23 ]
```

这里 TM=2,TN=4,TK=3。先只盯 **第一行第一个输出** `d00`:

```
d00 = a00·b00 + a01·b10 + a02·b20
```

3 次乘加 → 3×2 = 6 FLOP。整张 D 有 2×4=8 个输出,每个都要走完 3 个 k:

```
总 FLOP = 8 × 6 = 48 = 2 · TM · TN · TK = 2·2·4·3
```

**② 复用长什么样(对着具体字母数)**

把 A 里的 `a00` 圈出来。算第一行四个输出时:

```
d00 用到 a00·b00
d01 用到 a00·b01
d02 用到 a00·b02
d03 用到 a00·b03
```

`a00` 被用了 **4 次**(= TN)。从 global 读进 smem/寄存器一次就够,四个点积共用。

再圈 B 里的 `b00`。算第一列两个输出时:

```
d00 用到 a00·b00
d10 用到 a10·b00
```

`b00` 被用了 **2 次**(= TM)。同样只付一次 global 读。

D 的 8 个格子在 K 上加完后各写回一次 → 写 8 次。

对照「零复用」:若每算一个 `dij` 都重新从 global 拉一整行 A、一整列 B,那 `a00` 会被读 4 遍、`b00` 会被读 2 遍,分母大很多。tiling 就是先把整块 A(2×3=6 个数)、整块 B(3×4=12 个数)搬进来再算。

**③ 这个小例子的强度(仍用 fp16 读 / f32 写的口径)**

```
分子:48 FLOP

分母:
  A 读:2·3 · 2 = 12 B
  B 读:3·4 · 2 = 24 B
  D 写:2·4 · 4 = 32 B
  合计 = 68 B

强度 = 48 / 68 ≈ 0.71 FLOP/byte
```

数字小是因为 tile 太小;结构已经和正式公式一样:

```
强度 = 2·TM·TN·TK / (TM·TK·2 + TK·TN·2 + TM·TN·4)
```

**④ 放大到作业里的数:TM=TN=TK=128**

同一套公式,只是 2、4、3 换成 128:

```
分子:2 · 128 · 128 · 128 = 4_194_304 FLOP

分母:
  A 读:128·128 · 2 = 32_768 B     (像例子里的 12 B)
  B 读:128·128 · 2 = 32_768 B     (像例子里的 24 B)
  D 写:128·128 · 4 = 65_536 B     (像例子里的 32 B)
  合计 = 131_072 B

强度 = 4_194_304 / 131_072 = 32 FLOP/byte
```

和单条 mma 的 3.2 比:32 / 3.2 = **10 倍**。倍率来自例子里已经看见的事——每个 A 元素摊在 TN=128 个输出列上,每个 B 元素摊在 TM=128 个输出行上;global 仍只为每个元素付一次读。

tile 再放大(或数据更多留在 smem/L2、少回 global),分子按「体积」涨、分母按「表面积」涨,强度还会继续升。要把 ② 抬到机器平衡点(5090≈234)附近,单靠这一层 tile 往往不够,还要更深的层级复用和更快的供数路径。

结论:单条 mma 强度低(3.2)只说明「每条指令各自从 global 取数一定喂不满」;kernel 用 tiling 把同一份数据复用起来,把 ② 抬上去,才可能逼近峰值。M2 的 descriptor/swizzle、M3 的 TMA、M4 的 pipeline,都是在抬「实际达到的 ②」或让搬和算重叠。
