# Section 1 · Problem 1.1 —— m16n8k32 e4m3 的 fragment 映射

> 配套代码:`cuda/m1_sm80/01_fragment_map.cu`
> 判测:`make run/m1_sm80/01_fragment_map`(纯 host)→ **PASS**

四个函数回答的是:给定 lane 和它手里第几个元素,这个元素落在 A/B 的哪一格。后面 1.3 的手工装载、1.4 的 `ldmatrix` 都直接用这套公式;映射错一处,随机数据上的 MMA 会算错。

---

## 1. 这条指令在算什么

```
mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
D[16×8] = A[16×32] × B[32×8] + C
```

- A:16 行(M) × 32 列(K),元素类型 e4m3
- B:32 行(K) × 8 列(N),元素类型 e4m3
- C/D:f32 累加

一个 warp 的 32 个 lane 合拿完整的 A 和 B。均分之后:

```
A:16×32 = 512 元素 / 32 lane = 每 lane 16 个元素 → 4 个 b32 寄存器
B:32×8  = 256 元素 / 32 lane = 每 lane  8 个元素 → 2 个 b32 寄存器
```

`i` 是 **这个 lane 手里第几个元素**,按寄存器打包顺序编号(A:0..15, B:0..7)。它标识 fragment 里的槽位。本题 e4m3 每个元素宽度恰好是 8 bit,所以「第几个元素」和「fragment 里第几个 8-bit 槽」数值重合;换 fp16 时每人仍按元素计数,只是每个元素占 16 bit。

一个 32-bit 寄存器能装 `32 / (元素位宽)` 个元素。e4m3 是 8 bit,所以每寄存器 **4** 个元素:

```
r = i / 4    第几个寄存器
j = i % 4    这个寄存器里第几格(0,1,2,3)
```

这里的 4 来自 **32 bit ÷ 8 bit**,和下面 lane 分组里的 4 是两件不同的事。

---

## 1.1 `gid` / `tig` 里的 4 从哪来

指令形状是 **m16n8k32**:D 和 B 的 N 维是 **8 列**。32 个 lane 要覆盖这 8 列:

```
32 个 lane / 8 列 = 每列 4 个 lane
```

硬件把 32 个 lane 编成 **8 组、每组 4 人**,组内 4 人合管 B 的同一列(也合管 A 的同一组行):

```
列 n=0:  lane  0, 1, 2, 3
列 n=1:  lane  4, 5, 6, 7
…
列 n=7:  lane 28,29,30,31
```

所以:

```
gid = lane / 4     我在第几组 = 我管 B 的哪一列 n(0..7)
tig = lane % 4     我在组里是第几人(0..3)
lane = gid * 4 + tig
```

这个 4 来自 **`32 / N`**,而 N=8 写在指令名的 `n8` 里。课上 m16n8k16 fp16 的图是同一套分组:T0–T3 管 B 的第 0 列,T4–T7 管第 1 列。dtype 和 K 变了,N 仍是 8,所以仍是除以 4、对 4 取余。

A 用同一套分组,角色对调一下:

- `gid`(0..7)对应半边 8 行里的哪一行(`M=16` 切成上下各 8 行,所以 gid 到 7)
- `tig`(0..3)对应这一行上,K 被切成的第几段

K 半边宽度 ÷ 每寄存器沿 K 装的元素数,也等于 4(本题 16/4=4;课上 fp16 是 8/2=4)。所以 A 的「一行半边要 4 个人」和 B 的「一列要 4 个人」是同一个 4,都锁在 `m16n8` 这个形状上。

---

## 2. 公式:从课上 k16 fp16 改两个常数

课上 m16n8k16 fp16 的图(四象限,每个寄存器沿 K 跨 2 个元素,右半 +8):

```
A:  row = gid + 8 * (r % 2)
    col = 2 * tig + 8 * (r / 2)

B:  n = lane / 4
    k = 2 * (lane % 4) + 8 * r
```

本题 K=32、每寄存器沿 K 装 4 个 e4m3,所以跨度 2→4、半边 8→16,再写出寄存器内第几格 `j`:

```
A:  row = gid + 8 * (r % 2)
    col = 4 * tig + 16 * (r / 2) + j

B:  n = lane / 4
    k = 4 * (lane % 4) + 16 * r + j
```

A 的四个寄存器对应四个象限:`r=0` 左上,`r=1` 左下,`r=2` 右上,`r=3` 右下。B 两个寄存器对应一列的上半 K / 下半 K。

实现见 `01_fragment_map.cu`。host 真值表用 `row*32+col`(A)和 `k*8+n`(B)把一对坐标收成一个整数再比对,比的是格子位置。

---

## 3. 附加问题

> A 的同一个 b32 寄存器中的 4 个 fp8 元素沿矩阵哪个方向相邻?
> 这个布局对 1.4 中使用 ldmatrix load 有什么影响?

### 3.1 结论

**沿 K 相邻。** A 上是同一行连续 4 列;B 上是同一列连续 4 行。换行、换左右半、换 n,都发生在不同寄存器之间。

### 3.2 从公式读出相邻方向

把 A 的公式里 `r` 钉死(同一个寄存器),只让 `j` 走 0,1,2,3:

```
row = gid + 8*(r%2)                 ← 不含 j,四个元素行号相同
col = 4*tig + 16*(r/2) + j          ← 只在 col 上 +0,+1,+2,+3
```

这四格是 `(同一行, 起始列+0/1/2/3)`。A 的列就是 K。

lane 0 的四个寄存器:

```
a[0]  j=0..3 → A[0][0..3]
a[1]  j=0..3 → A[8][0..3]     另一行,仍沿 K 走 4 格
a[2]  j=0..3 → A[0][16..19]
a[3]  j=0..3 → A[8][16..19]
```

B:钉死 `r` 之后 `n = lane/4` 固定,`k = 4*tig + 16*r + j` 只在 k 上连走 4 步,同样沿 K。

MMA 沿 K 做内积,寄存器里打包的就是这段 K。课上 fp16:一个寄存器跨 2 个 K;e4m3 每个元素 8 bit,同样 32 bit 跨 **4 个 K**。

### 3.3 这对 1.4 的 ldmatrix 意味着什么

`ldmatrix` 的接口是:

- 每条「行地址」指向 **16 byte 连续** 的一段 smem
- 搬移单位是 **b16**(16 bit)。e4m3 下,两个沿 K 的元素合成一个 b16,四个合成一个 b32
- `.x1/.x2/.x4` 决定每个 lane 拿到几个 b32;`.trans` 决定这 16 byte 按哪条轴灌进寄存器

1.1 的打包方向接到这条接口上,有三件具体的事。

**(1) 沿 K 相邻,和「16 byte 连续行」对齐的方式。**

A 按 `[16][32]` 行主序放 smem 时,一行就是 K 方向 32 个连续元素:

- 一个 b32 要的 4 个元素 = 一行上连续 4 格
- 一条 ldmatrix 行地址要的 16 byte = 一行上连续 16 个 K,覆盖「左半或右半」一整段(`col` 公式里的 `+16*(r/2)`)

A 用行主序 smem、配合 `ldmatrix` 的默认轴,「16 byte 连续行」就对上「沿 K 打包的 fragment」。手工路径里按公式算 `A[row*32+col]` 的那一大段地址算术,会被这次集体搬移消掉——这是 1.4 要数指令的原因。

**(2) B 要用能让 K 连续的 smem 布局,所以 1.4 备了 `sBn`。**

B 的 fragment 同样要沿 K 的 4 个相邻元素进同一个 b32。`[32][8]` 的 k-major 下,固定 n、k 加 1,步长是 8 个元素,沿 K 的邻居在内存里是跳着的;ldmatrix 的行地址要的是步长为 1 的 16 byte,两者要靠把 K 摆成连续维才能合上。

转成 n-major `[8][32]`:每个 n 的 32 个 k 变成连续 32 个元素。沿 K 相邻 = 这一行里连续的元素,ldmatrix 的 16 byte 行地址成立。文件头说「B 的 fragment 需要 k 方向相邻的元素成对进 b16」:b16 是 ldmatrix 的原子,两个沿 K 的 e4m3 合成一个 b16;布局要让这些 K 向邻居在 smem 里贴在一起。

**(3) `.trans` 选的是轴,要和「沿 K 打包」一致。**

`.trans` 把连续 16 byte 解释成另一条轴。A 已经是 K 向连续、fragment 也要 K 向打包,用直搬(无 `.trans`)即可。B 在 n-major 下 K 已经是连续维,同样用直搬。骨架提供 `sBn`,就是把 K 摆成连续维,让 ldmatrix 的行地址直接对上 fragment。

一句话:**fragment 规定「一个寄存器吃一段 K」;ldmatrix 规定「一次搬 16 个连续 byte」。两边对齐时,smem 的连续维是 K。** 1.1 把相邻方向钉死之后,1.4 的布局和 `.trans` 就有了选择依据。

---

# Section 1 · Problem 1.2 —— m16n8k16 fp16：A 下半行装错了会怎样

> 配套代码:`cuda/m1_sm80/02_bug_fragment.cu`
> 判测:`make ARCH=120a bin/m1_sm80/02_bug_fragment && srun -p lcpu-infra --gpus=1 ./bin/m1_sm80/02_bug_fragment`
> 修好后 → **PASS**(仓库里当前版本已修)

`m16n8k16` fp16 MMA。B 的装载和 D 的写回都是对的;bug 只在 A 的下半行。对着下面两段代码看就够了。

---

## 1. 错的 vs 对的

四象限提醒:`a0,a1` / `a4,a5` 管行 `group`;`a2,a3` / `a6,a7` 管行 `group+8`。

**原来(错):**`a2,a3,a6,a7` 又读了一遍上行,漏了 `(group+8)`。

```
a0 = A[group * 16 + tig * 2];
a1 = A[group * 16 + tig * 2 + 1];
a2 = A[group * 16 + tig * 2];          // 错
a3 = A[group * 16 + tig * 2 + 1];      // 错
a4 = A[group * 16 + tig * 2 + 8];
a5 = A[group * 16 + tig * 2 + 9];
a6 = A[group * 16 + tig * 2 + 8];      // 错
a7 = A[group * 16 + tig * 2 + 9];      // 错
```

**修好(对):**只有这四行改成 `(group + 8) * 16 + …`,其余不动。

```
a0 = A[group * 16 + tig * 2];
a1 = A[group * 16 + tig * 2 + 1];
a2 = A[(group + 8) * 16 + tig * 2];
a3 = A[(group + 8) * 16 + tig * 2 + 1];
a4 = A[group * 16 + tig * 2 + 8];
a5 = A[group * 16 + tig * 2 + 9];
a6 = A[(group + 8) * 16 + tig * 2 + 8];
a7 = A[(group + 8) * 16 + tig * 2 + 9];
```

差就差在:`a2,a3,a6,a7` 的行号是不是 `group+8`。

---

## 2. (a) 症状

跑 buggy 版本:

- `D` 上半行 `0..7` 全对
- 下半行 `8..15` 整块等于上半:`got[r+8][n] == got[r][n]`
- 相对 ref 大约 `59/128` 个 mismatch(少数格碰巧 `A[r]·B == A[r+8]·B`)

host 故意让 A 上下半不同,所以「下行装成上行」会在 D 上直接变成复印件。

---

## 3. (b) 为什么是这个症状

`d0,d1` 写 `D[group][…]`,靠的是上行 A(`a0,a1,a4,a5`)→ 上半对。  
`d2,d3` 写 `D[group+8][…]`,靠的是下行 A(`a2,a3,a6,a7`)→ 错装成上行之后,算出来的正好是 `D[group]` 那一份,所以下半 = 上半。

---

## 4. D 写回长什么样(顺手对照)

K 在 MMA 里已经消掉了;每 lane 4 个 f32 是 4 个算完的 `D[m][n]`:

```
d[0], d[1]  →  同一行、相邻两列
d[2], d[3]  →  行号 +8、同样两列
```

```
D[group * 8 + tig * 2]           = d[0];  // 行 group,   列 tig*2
D[group * 8 + tig * 2 + 1]       = d[1];
D[(group + 8) * 8 + tig * 2]     = d[2];  // 行 group+8, 列 tig*2
D[(group + 8) * 8 + tig * 2 + 1] = d[3];
```

固定某个 `group`,`tig=0..3` 四人合起来:

```
        列 0,1    2,3    4,5    6,7
行 g      tig0   tig1   tig2   tig3     ← d0,d1
行 g+8    tig0   tig1   tig2   tig3     ← d2,d3
```

---

# Section 1 · Problem 1.4 —— ldmatrix

> 配套代码:`cuda/m1_sm80/04_ldmatrix.cu`
> 判测:两条路径各跑 3 个 seed(1,7,42),全 PASS 才算过
> 运行:
> ```
> cd assignment02/cuda
> make ARCH=120a bin/m1_sm80/04_ldmatrix && srun -p lcpu-infra --gpus=1 ./bin/m1_sm80/04_ldmatrix
> ```

## 0. 矩阵形状

```
mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
D[16×8] = A[16×32] × B[32×8] + C
```

| 矩阵 | 形状 | smem |
|------|------|------|
| A | 16×32 e4m3 | `sA` 行主序 |
| B | 32×8 e4m3 | `sBn` n-major(本题 ldmatrix 用这个) |
| D | 16×8 f32 | 写回 global |

`load_ldsm` 的工作:从 smem 把 A、B 装进每个 lane 的 `a[0..3]`、`b[0..1]`,布局对齐课上的 fragment 四色图,然后发 `mma`。

---

## 1. ldmatrix 在干什么

写法跟课上一致:每个 lane 提供一个 8×8(以 b16 计)小矩阵某一行的首地址;指令读行并按 fragment 图写入寄存器。

### 1.1 对着四色图

| lane | 色块 | 寄存器 | 行首 |
|------|------|--------|------|
| 0..7 | 绿左上 | `a[0]` | `&sA[row*32 + 0]`,row=0..7 |
| 8..15 | 紫左下 | `a[1]` | `&sA[row*32 + 0]`,row=8..15 |
| 16..23 | 蓝右上 | `a[2]` | `&sA[row*32 + 16]`,row=0..7 |
| 24..31 | 橙右下 | `a[3]` | `&sA[row*32 + 16]`,row=8..15 |

蓝 / 橙的行首在 **K 右半**(`col=16` 起的 16 byte)。绿 / 紫在 `col=0`。

课上 fp16、K=16 是 `col = (lane>>4)*8`、stride 16;本题 e4m3、K=32 半段仍是 16 byte,改成 `*16`、stride 32,结构相同。

### 1.2 代码(课上模板,本题用 `/` `%`)

```
lane = threadIdx.x % 32
row  = lane % 16
col  = (lane / 16) * 16
aAddr = __cvta_generic_to_shared(&sA[row * 32 + col])
ldmatrix.x4 → a[0..3]

n = lane % 8
k = ((lane / 8) % 2) * 16
bAddr = __cvta_generic_to_shared(&sBn[n * 32 + k])
ldmatrix.x2 → b[0..1]
```

---

## 2. 数据从哪到哪

```
global → smem(sA, sBn) → ldmatrix → 寄存器 a[]/b[] → mma → d[] → global
```

`ldmatrix` = load from shared memory:从 smem 读进寄存器。

B 用 `sBn`(n-major):每个 n 的 K 连续,才能取出 16 byte 的一行。`sBk` 沿 K 步长是 8,拼不成这种行。

---

## 3. PTX 对照(报告两问的证据)

文件:`make ARCH=120a ptx/m1_sm80/04_ldmatrix` → `m1_sm80/04_ldmatrix.ptx`  
口径:**`bar.warp.sync` 之后、`mma.sync` 之前**(拷贝循环、写回 D、C 清零的 `mov.f32` 都不算)。

- **装载** = `ld.shared.*` / `ldmatrix.*`
- **地址算术** = 从 `%tid.x`(`%r1`)算出 smem 字节偏移再加到基址的 `shl`/`and`/`or`/`add`(基址 `mov.u32 sA/sBn` 单独记,不进算术条数)

`mma_kernel<true>` = ldmatrix,`mma_kernel<false>` = 手工。

### 3.1 手工路径(`$L__BB1_6`)

```
bar.warp.sync  -1;
shl.b32  %r22, %r1, 3
and.b32  %r23, %r22, 8160      // gid*32 相关
shl.b32  %r24, %r1, 2
and.b32  %r25, %r24, 12        // 4*tig 相关
or.b32   %r26, %r23, %r25      // offset = gid*32 + 4*tig
mov.u32  %r27, sA
add.s32  %r28, %r27, %r26
ld.shared.u32  %r16, [%r28]        // a[0] 左上
ld.shared.u32  %r17, [%r28+256]    // a[1] 左下  (+8 行 = 256B 折进立即数)
ld.shared.u32  %r18, [%r28+16]     // a[2] 右上  (+16 列)
ld.shared.u32  %r19, [%r28+272]    // a[3] 右下
mov.u32  %r29, sBn
add.s32  %r30, %r29, %r26          // B 复用同一 offset
ld.shared.u32  %r20, [%r30]        // b[0]
ld.shared.u32  %r21, [%r30+16]     // b[1]
mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
    {%f1,%f2,%f3,%f4}, {%r16,%r17,%r18,%r19}, {%r20,%r21}, {%f8,%f8,%f8,%f8};
```

| 类 | 条数 | 明细 |
|----|------|------|
| 装载 | **6** | 6× `ld.shared.u32` |
| 地址算术 | **7** | 5 条算 `%r26` + 2 条 `base+offset` |
| 基址 mov | 2 | `sA`、`sBn` |

六个 b32(A 四象限 + B 两半区)各对应一条 `ld.shared`;`+256/+16/+272` 是编译器把固定象限位移折进寻址立即数。

### 3.2 ldmatrix 路径(`$L__BB0_6`)

```
bar.warp.sync  -1;
and.b32  %r30, %r1, 16
shl.b32  %r31, %r1, 5
and.b32  %r32, %r31, 480
or.b32   %r33, %r32, %r30       // A: row*32 + col 的偏移
mov.u32  %r34, sA
add.s32  %r20, %r34, %r33
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r16,%r17,%r18,%r19}, [%r20];
shl.b32  %r35, %r1, 1
and.b32  %r36, %r35, 16
and.b32  %r37, %r31, 224        // 复用上面的 %r31
or.b32   %r38, %r37, %r36       // B: n*32 + k 半区
mov.u32  %r39, sBn
add.s32  %r23, %r39, %r38
ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%r21,%r22}, [%r23];
mma.sync ... {%r16..%r19}, {%r21,%r22}, ...
```

| 类 | 条数 | 明细 |
|----|------|------|
| 装载 | **2** | 1× `.x4` + 1× `.x2` |
| 地址算术 | **10** | A:4 算偏移+1 `add`;B:4 算偏移+1 `add`(`%r31` 两边共用) |
| 基址 mov | 2 | `sA`、`sBn` |

`.x4` 的四个目标寄存器 `{%r16..%r19}` 就是 `a[0..3]`;`.x2` 的 `{%r21,%r22}` 就是 `b[0],b[1]`。随后 `mma` 直接吃这六个寄存器——和手工路径装完后的寄存器角色相同。

### 3.3 对照表

| 路径 | 装载指令 | 地址计算指令 |
|------|----------|--------------|
| 手工 | **6** (`ld.shared.u32`) | **7** |
| ldmatrix | **2** (`ldmatrix` ×2) | **10** |

### 3.4 报告两问

**(a) `ldmatrix` 省掉了手工装载中的哪些工作?**

看 PTX:手工在 `bar` 与 `mma` 之间是 **6 条** `ld.shared.u32`,每条只往 **本 lane** 填一个 b32(`%r16`…`%r21`)。ldmatrix 路径同区间只剩 **2 条** 装载指令:

- `ldmatrix...x4 ... {%r16,%r17,%r18,%r19}, [%r20]` → 一次写齐 A 的四个色块寄存器
- `ldmatrix...x2 ... {%r21,%r22}, [%r23]` → 一次写齐 B 的两个半区寄存器

省掉的是「按 fragment 寄存器一条条 `ld.shared`」这 **6→2** 的取数。  
地址算术 **没有**变少(7→10):每个 lane 仍要算出自己交给 `ldmatrix` 的那条 16 byte 行首(`[%r20]` / `[%r23]`),A、B 两套公式,固定象限位移也折不进 `ldmatrix` 的寻址立即数。

**(b) 为什么这些工作在手工路径里绕不开?**

手工路径的契约是:`ld.shared.u32` 把 **本 lane** 指针上的 4 byte 写进 **本 lane** 的一个寄存器。A+B 一共 6 个 b32,PTX 里就出现 6 次 load;「一次读多行、再按 fragment 图拆到各 lane」没有对应的普通 load 形态。

`ldmatrix` 把这件事收成一条 warp 指令:输入是各 lane 的行首,输出已经是 `{%r16..%r19}` / `{%r21,%r22}` 这种可直接喂给 `mma` 的 fragment 寄存器(两条路径的 `mma` 源寄存器表一致)。没有这条指令时,软件只能继续用逐寄存器 `ld.shared`(或自己 load + shuffle),条数随 fragment 寄存器个数线性涨——这就是手工路径在 PTX 里绕不开 6 条 `ld.shared` 的原因。

---

## 4. 小结

```
发 mma 前:两个路径的 a[0..3]、b[0..1] 都已是 fragment 布局
手工 PTX:6× ld.shared.u32 + 7 条地址算术
ldmatrix PTX:1× .x4 + 1× .x2 + 10 条地址算术
省的是装载条数(6→2),不是算行首
```