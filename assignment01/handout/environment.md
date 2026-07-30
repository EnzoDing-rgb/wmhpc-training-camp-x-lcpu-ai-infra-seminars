# 环境配置说明

> 主机：`4p1r1pd2gec8r-0` | Docker 容器 | 更新：2026-07-30

**核心事项：**

1. **GPU 驱动升级** — 当前 550（CUDA 12.4），需升到 570+（CUDA 13.x）。驱动不升级，PyTorch 等所有上层包都被卡住。
2. **vLLM 无负载** — Qwen 3.6 27B 常驻显存（77.5 GB / 80 GB），但 GPU 利用率为 0%，无推理请求。监控脚本 `python3 /Lishun/scripts/server_info.py` 可一键查看全貌。
3. **计算资源** — 需向老师申请更多计算资源。

---

<style>
  .cmp { width: 100%; border-collapse: collapse; font-size: 14px; }
  .cmp th { background: #1e3a5f; color: #fff; padding: 10px 12px; text-align: left; }
  .cmp td { padding: 10px 12px; vertical-align: top; border-bottom: 1px solid #e0e0e0; }
  .cmp tr:nth-child(even) td { background: #f7f9fc; }
  .cmp .best  { color: #2e7d32; }
  .cmp .now   { color: #e65100; }
  .cmp .tag   { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }
  .cmp .tag-best  { background: #e8f5e9; color: #2e7d32; }
  .cmp .tag-now   { background: #fff3e0; color: #e65100; }
  .cmp .tag-admin { background: #e3f2fd; color: #1565c0; }
  /* 管理员列：需介入 = 红，不需介入 = 灰 */
  .cmp tr.need-admin td:last-child  { background: #ffebee; color: #c62828; font-weight: 500; }
  .cmp tr.no-admin   td:last-child  { background: #f5f5f5; color: #9e9e9e; }
</style>

<table class="cmp">
<thead>
<tr>
  <th style="width:15%">组件 / 配置项</th>
  <th style="width:28%"><span class="tag tag-best">最佳实践</span></th>
  <th style="width:28%"><span class="tag tag-now">我们当前的妥协</span></th>
  <th style="width:29%"><span class="tag tag-admin">管理员操作建议</span></th>
</tr>
</thead>
<tbody>

<tr class="need-admin">
<td><strong>CUDA 驱动</strong></td>
<td class="best">570+（CUDA 13.x）</td>
<td class="now">550.90.07（CUDA 12.4）</td>
<td class="admin">升级到 570+<br>PyTorch 2.13 起只支持 CUDA 13.x</td>
</tr>

<tr class="need-admin">
<td><strong>CUDA Toolkit<br>(nvcc 编译器)</strong></td>
<td class="best">nvcc 12.x / 13.x<br>与驱动版本对齐</td>
<td class="now">nvcc 11.8（2022 年版本）<br><code>/usr/local/cuda/bin/nvcc</code></td>
<td class="admin">升级到 12.x 或 13.x<br>解锁新 C++ 特性、新 CUDA API</td>
</tr>

<tr class="need-admin">
<td><strong>PyTorch</strong></td>
<td class="best">2.13.0+cu130</td>
<td class="now">2.6.0+cu124<br>（驱动上限，cu130 装不上）</td>
<td class="admin">驱动升级后自然解决</td>
</tr>

<tr class="need-admin">
<td><strong>Triton</strong></td>
<td class="best">与 torch 严格匹配</td>
<td class="now">3.7.1（torch 要求 3.2.0，但 3.2.0 有 bug）</td>
<td class="admin">驱动升级→torch 升级→triton 版本对齐</td>
</tr>

<tr class="no-admin">
<td><strong>NumPy</strong></td>
<td class="best">最新稳定版（2.4.6）</td>
<td class="now">1.26.4（停在 1.x，避免 2.x API 变动）</td>
<td class="admin">无需管理介入<br>pip 即可升级</td>
</tr>

<tr class="no-admin">
<td><strong>Python 版本</strong></td>
<td class="best">3.12 或 3.13</td>
<td class="now">3.11.5（conda base）</td>
<td class="admin">无需管理介入<br>conda 可自行管理</td>
</tr>

<tr class="no-admin">
<td><strong>Python 包管理</strong></td>
<td class="best">每项目独立 venv<br>不依赖全局，避免污染</td>
<td class="now">conda base 装共享大库<br>项目 venv 用 --system-site-packages<br>（省磁盘，但有全局污染）</td>
<td class="admin">无需管理介入</td>
</tr>

<tr class="no-admin">
<td><strong>pip 镜像</strong></td>
<td class="best">就近镜像 / 代理加速</td>
<td class="now">清华源<br>已写入 ~/.pip/pip.conf 和 uv.toml</td>
<td class="admin">无需管理介入</td>
</tr>

<tr class="no-admin">
<td><strong>vLLM 模型服务</strong></td>
<td class="best">按需自动扩缩</td>
<td class="now">Qwen 3.6 27B 常驻显存<br>GPU 利用率 0%，无推理负载<br>监控脚本覆盖：<code>python3 /Lishun/scripts/server_info.py</code></td>
<td class="admin">无需管理介入</td>
</tr>

</tbody>
</table>

---

## 1. 管理员操作建议

以下内容建议提交给容器/集群管理员，按优先级排序。

### 高优先级：CUDA 驱动升级

**现状**：NVIDIA 驱动版本 550.90.07，对应 CUDA 12.4。最新 PyTorch（2.13.0）已要求 CUDA 13.x 驱动，当前驱动无法运行。

**请求**：将驱动升级到 **570 及以上**（即 CUDA 13.x 系列）。

**收益**：
- PyTorch 可从 2.6.0+cu124 升级到最新 2.13.0+cu130
- 消除 triton 版本冲突（当前 3.7.1 与 torch 2.6.0 不严格匹配）
- 后续新项目不用再为驱动版本妥协

### 中优先级：CUDA Toolkit（nvcc）升级

**现状**：nvcc 版本 11.8（2022 年发布），路径 `/usr/local/cuda/bin/nvcc`。

**请求**：升级到 **12.x 或 13.x**，与驱动版本对齐。

**收益**：
- 支持更新的 C++ 标准特性
- CUDA 新 API（如 `cudaMallocAsync`、graph API 增强等）可用
- 编译出的 PTX/SASS 代码和驱动版本一致，减少兼容隐患

### 低优先级（暂不需要）

- Ubuntu 基础镜像升级（20.04 → 24.04）：暂不需要
- GPU 硬件升级：A800 80GB 足够当前使用

---

## 2. 当前妥协版最佳实践

### 核心约束

CUDA 驱动 12.4 是硬上限——驱动在容器内无法升级，所有软件版本选择围绕它来折中。

### 软件版本锁定

| 组件 | 版本 | 说明 |
|------|------|------|
| PyTorch (torch) | 2.6.0+cu124 | CUDA 12.4 下能用的最新版；不能装 cu130 版本 |
| Triton | 3.7.1 | 覆盖了 torch 自带的 3.2.0（后者有 bug），有 pip 依赖警告但正常运行 |
| NumPy | 1.26.4 | 停留在 1.x，避免 2.x 的 API 变动 |
| Pytest | 9.1.1 | 满足项目要求 ≥8.0 |
| TileLang | 0.1.12 | 项目特定依赖 |
| nvcc | 11.8 | 编译 CUDA 练习用，目标架构 `sm_80` |
| Python | 3.11.5 | conda base 提供 |

### Python 包管理：两级复用

```text
conda base（全局，所有项目共享）
  ├── torch, triton, numpy, pytest  ← 通用大库，只装一份
  └── 不装项目特定包

项目 .venv（--system-site-packages 继承 conda）
  └── tilelang  ← 仅装本项目独有的库
```

### pip/uv 镜像

清华源已配置为全局默认（`~/.pip/pip.conf` 和 `~/.config/uv/uv.toml`），`pip install` 不再需要手动指定 `--index-url`。

---

## 3. vLLM 模型服务

### 运行状态

| 项目 | 值 |
|------|-----|
| PID | 31127 |
| 状态 | RUNNING，端口 30019 已监听 |
| 模型 | Qwen 3.6 27B（权重路径 `/Lishun/models/Qwen/qwen3.6-27b`） |
| 对外名称 | `qwen3.5`（兼容旧 API 名） |
| 上下文长度 | 262144 tokens |
| 显存利用率 | 0.95（约 77.5 GB / 80 GB，占 94.6%） |
| GPU 利用率 | 当前 0%（空闲等待请求） |
| GPU 温度 | 39°C |
| 访问地址 | `http://127.0.0.1:30019/v1/models` |

### 并发估算（256k 上下文）

| 文本长度 | 并发数 |
|----------|--------|
| 4k（短） | ≈ 5 |
| 16k（中） | ≈ 1 |
| 64k（长） | ≈ 1 |
| 256k（超长） | ≈ 1 |

模型已常驻显存（94.6%），处于待命状态，当前无推理请求。

---

## 4. 监控脚本

`/Lishun/scripts/server_info.py` — 一键可视化服务器全貌。

运行方式：

```bash
python3 /Lishun/scripts/server_info.py
```

覆盖内容：
- 运行环境检测（Docker/物理机/K8s）
- GPU 状态（型号、显存、利用率、温度，带彩色进度条）
- CPU 信息（型号、核心数、频率、SIMD 指令集、NUMA 拓扑）
- 内存/磁盘使用
- vLLM 模型服务详情（进程状态、并发估算、模型路径）
- 热点进程排行

输出带颜色和图标，适合日常巡检和给他人截图分享。
