# Problem 1.3 · fragment 图（m16n8k32 e4m3）

> 课件原图是 **m16n8k16 fp16**（每标签横跨 2 列 / 2 行，半区宽 8）。
> 1.3 换成 **m16n8k32 e4m3**：每寄存器 4 个 fp8，每标签横跨 **4** 列（A）或 **4** 行（B），半区宽 **16**。
> 格子里的数字仍是持有该元素的 **lane 编号**。用 Markdown 预览或浏览器打开本文件。

<style>

.fig { font-family: "Segoe UI", "PingFang SC", "Noto Sans SC", sans-serif; max-width: 1100px; margin: 2rem auto 3.5rem; color: #222; }
.fig h2 { font-size: 1.35rem; margin: 0 0 .4rem; }
.fig .lead { color: #444; font-size: .92rem; margin: 0 0 .8rem; line-height: 1.45; }
.fig table.grid { border-collapse: collapse; font-size: 11px; margin: .4rem 0; }
.fig table.grid th, .fig table.grid td { border: 1px solid #9aa; text-align: center; padding: 2px 3px; min-width: 22px; line-height: 1.15; }
.fig table.grid th { background: #f4f4f4; font-weight: 600; color: #555; }
.fig table.grid td.rh { background: #f4f4f4; font-weight: 600; color: #555; min-width: 28px; }
.g0 { background: #e8f6e8; }
.g1 { background: #e6d9f2; }
.g2 { background: #d6e8fb; }
.g3 { background: #fde6c8; }
.fig .axis { font-size: .85rem; color: #333; margin: .15rem 0 .35rem; }
.fig .legend { display: flex; flex-wrap: wrap; gap: .6rem 1.2rem; align-items: center; margin: .6rem 0; font-size: .9rem; }
.fig .sw { display: inline-block; width: 18px; height: 14px; border: 1px solid #888; vertical-align: middle; margin-right: 6px; }
.fig .box { background: #fff; border: 1px solid #ccc; padding: .7rem .9rem; margin-top: .6rem; font-family: ui-monospace, Menlo, Consolas, monospace; font-size: .86rem; line-height: 1.55; white-space: pre-wrap; }
.fig .call { float: right; width: 280px; background: #fff; border: 1px solid #bbb; padding: .65rem .8rem; margin: 0 0 .8rem 1rem; font-size: .88rem; line-height: 1.45; }
.fig .clear { clear: both; }
.circ { outline: 2px solid #c00; outline-offset: -1px; }

</style>

<div class="fig">

## Fragment 图的规律（A）

<p class="lead">分成四个象限，每个标签横跨 <b>四列</b>（一个 b32 里 4 个 e4m3，沿 K 相邻）。
T0 的 a[0] 占列 0–3 → a 的 4 个 fp8 在 (0,0)(0,1)(0,2)(0,3)。每个线程 4 个寄存器的分工：</p>
<p class="axis">row (M) ↓ &nbsp;&nbsp; col (K) → &nbsp; 格子里是持有该元素的 lane 编号。A = 16×32。</p>

<table class="grid">
<tr><th></th><th>0</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th><th>6</th><th>7</th><th>8</th><th>9</th><th>10</th><th>11</th><th>12</th><th>13</th><th>14</th><th>15</th><th>16</th><th>17</th><th>18</th><th>19</th><th>20</th><th>21</th><th>22</th><th>23</th><th>24</th><th>25</th><th>26</th><th>27</th><th>28</th><th>29</th><th>30</th><th>31</th></tr>
<tr>
<td class="rh">0</td>
<td class="g0" colspan="4">0</td>
<td class="g0" colspan="4">1</td>
<td class="g0" colspan="4">2</td>
<td class="g0" colspan="4">3</td>
<td class="g2" colspan="4">0</td>
<td class="g2" colspan="4">1</td>
<td class="g2" colspan="4">2</td>
<td class="g2" colspan="4">3</td>
</tr>
<tr>
<td class="rh">1</td>
<td class="g0" colspan="4">4</td>
<td class="g0" colspan="4">5</td>
<td class="g0" colspan="4">6</td>
<td class="g0" colspan="4">7</td>
<td class="g2" colspan="4">4</td>
<td class="g2" colspan="4">5</td>
<td class="g2" colspan="4">6</td>
<td class="g2" colspan="4">7</td>
</tr>
<tr>
<td class="rh">2</td>
<td class="g0" colspan="4">8</td>
<td class="g0" colspan="4">9</td>
<td class="g0" colspan="4">10</td>
<td class="g0" colspan="4">11</td>
<td class="g2" colspan="4">8</td>
<td class="g2" colspan="4">9</td>
<td class="g2" colspan="4">10</td>
<td class="g2" colspan="4">11</td>
</tr>
<tr>
<td class="rh">3</td>
<td class="g0" colspan="4">12</td>
<td class="g0" colspan="4">13</td>
<td class="g0" colspan="4">14</td>
<td class="g0" colspan="4">15</td>
<td class="g2" colspan="4">12</td>
<td class="g2" colspan="4">13</td>
<td class="g2" colspan="4">14</td>
<td class="g2" colspan="4">15</td>
</tr>
<tr>
<td class="rh">4</td>
<td class="g0" colspan="4">16</td>
<td class="g0" colspan="4">17</td>
<td class="g0" colspan="4">18</td>
<td class="g0" colspan="4">19</td>
<td class="g2" colspan="4">16</td>
<td class="g2" colspan="4">17</td>
<td class="g2" colspan="4">18</td>
<td class="g2" colspan="4">19</td>
</tr>
<tr>
<td class="rh">5</td>
<td class="g0" colspan="4">20</td>
<td class="g0" colspan="4">21</td>
<td class="g0" colspan="4">22</td>
<td class="g0" colspan="4">23</td>
<td class="g2" colspan="4">20</td>
<td class="g2" colspan="4">21</td>
<td class="g2" colspan="4">22</td>
<td class="g2" colspan="4">23</td>
</tr>
<tr>
<td class="rh">6</td>
<td class="g0" colspan="4">24</td>
<td class="g0" colspan="4">25</td>
<td class="g0" colspan="4">26</td>
<td class="g0" colspan="4">27</td>
<td class="g2" colspan="4">24</td>
<td class="g2" colspan="4">25</td>
<td class="g2" colspan="4">26</td>
<td class="g2" colspan="4">27</td>
</tr>
<tr>
<td class="rh">7</td>
<td class="g0" colspan="4">28</td>
<td class="g0" colspan="4">29</td>
<td class="g0" colspan="4">30</td>
<td class="g0" colspan="4">31</td>
<td class="g2" colspan="4">28</td>
<td class="g2" colspan="4">29</td>
<td class="g2" colspan="4">30</td>
<td class="g2" colspan="4">31</td>
</tr>
<tr>
<td class="rh">8</td>
<td class="g1" colspan="4">0</td>
<td class="g1" colspan="4">1</td>
<td class="g1" colspan="4">2</td>
<td class="g1" colspan="4">3</td>
<td class="g3" colspan="4">0</td>
<td class="g3" colspan="4">1</td>
<td class="g3" colspan="4">2</td>
<td class="g3" colspan="4">3</td>
</tr>
<tr>
<td class="rh">9</td>
<td class="g1" colspan="4">4</td>
<td class="g1" colspan="4">5</td>
<td class="g1" colspan="4">6</td>
<td class="g1" colspan="4">7</td>
<td class="g3" colspan="4">4</td>
<td class="g3" colspan="4">5</td>
<td class="g3" colspan="4">6</td>
<td class="g3" colspan="4">7</td>
</tr>
<tr>
<td class="rh">10</td>
<td class="g1" colspan="4">8</td>
<td class="g1" colspan="4">9</td>
<td class="g1" colspan="4">10</td>
<td class="g1" colspan="4">11</td>
<td class="g3" colspan="4">8</td>
<td class="g3" colspan="4">9</td>
<td class="g3" colspan="4">10</td>
<td class="g3" colspan="4">11</td>
</tr>
<tr>
<td class="rh">11</td>
<td class="g1" colspan="4">12</td>
<td class="g1" colspan="4">13</td>
<td class="g1" colspan="4">14</td>
<td class="g1" colspan="4">15</td>
<td class="g3" colspan="4">12</td>
<td class="g3" colspan="4">13</td>
<td class="g3" colspan="4">14</td>
<td class="g3" colspan="4">15</td>
</tr>
<tr>
<td class="rh">12</td>
<td class="g1" colspan="4">16</td>
<td class="g1" colspan="4">17</td>
<td class="g1" colspan="4">18</td>
<td class="g1" colspan="4">19</td>
<td class="g3" colspan="4">16</td>
<td class="g3" colspan="4">17</td>
<td class="g3" colspan="4">18</td>
<td class="g3" colspan="4">19</td>
</tr>
<tr>
<td class="rh">13</td>
<td class="g1" colspan="4">20</td>
<td class="g1" colspan="4">21</td>
<td class="g1" colspan="4">22</td>
<td class="g1" colspan="4">23</td>
<td class="g3" colspan="4">20</td>
<td class="g3" colspan="4">21</td>
<td class="g3" colspan="4">22</td>
<td class="g3" colspan="4">23</td>
</tr>
<tr>
<td class="rh">14</td>
<td class="g1" colspan="4">24</td>
<td class="g1" colspan="4">25</td>
<td class="g1" colspan="4">26</td>
<td class="g1" colspan="4">27</td>
<td class="g3" colspan="4">24</td>
<td class="g3" colspan="4">25</td>
<td class="g3" colspan="4">26</td>
<td class="g3" colspan="4">27</td>
</tr>
<tr>
<td class="rh">15</td>
<td class="g1" colspan="4">28</td>
<td class="g1" colspan="4">29</td>
<td class="g1" colspan="4">30</td>
<td class="g1" colspan="4">31</td>
<td class="g3" colspan="4">28</td>
<td class="g3" colspan="4">29</td>
<td class="g3" colspan="4">30</td>
<td class="g3" colspan="4">31</td>
</tr>
</table>

<div class="legend">
  <span><i class="sw g0"></i>a[0] = 上行、K 左半（列 0–15）</span>
  <span><i class="sw g1"></i>a[1] = 下行、K 左半</span>
  <span><i class="sw g2"></i>a[2] = 上行、K 右半（列 16–31）</span>
  <span><i class="sw g3"></i>a[3] = 下行、K 右半</span>
</div>

<div class="box">lane = gid * 4 + tig              // 这个 4 = 一组 4 个 lane（32 / N=8，来自指令的 n8）
gid = lane / 4          (0..7)   // 同上：按 4 人一组取组号
tig = lane % 4          (0..3)   // 同上：组内第几人
row = gid + 8 * (r % 2)           // 8 = M 半区行数（16/2）
col = 4 * tig + 16 * (r / 2) + j
      // 4*tig 的 4 = 每寄存器沿 K 装 4 个 e4m3（32 bit / 8 bit）
      // 16 = K 半区宽度（32/2）
      // j=0..3 还是这同一个 4：寄存器内第几个 fp8</div>

</div>

<div class="fig">

## Fragment 图的规律（B）

<p class="lead">分成上下两个半区，每个标签横跨 <b>四行</b>（一个 b32 里 4 个 e4m3，沿 K 相邻）。
T0 的 b[0] 占 k=0–3、n=0 → 4 个 fp8 在 (k,n) = (0,0)(1,0)(2,0)(3,0)。
每个线程 2 个寄存器的分工：</p>
<p class="axis">k (K) ↓ &nbsp;&nbsp; n (N) → &nbsp; 格子里是持有该元素的 lane 编号。B = 32×8。</p>

<table class="grid">
<tr><th></th><th>0</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th><th>6</th><th>7</th></tr>
<tr>
<td class="rh">0</td>
<td class="g0" rowspan="4">0</td>
<td class="g0" rowspan="4">4</td>
<td class="g0" rowspan="4">8</td>
<td class="g0" rowspan="4">12</td>
<td class="g0" rowspan="4">16</td>
<td class="g0" rowspan="4">20</td>
<td class="g0" rowspan="4">24</td>
<td class="g0" rowspan="4">28</td>
</tr>
<tr>
<td class="rh">1</td>
</tr>
<tr>
<td class="rh">2</td>
</tr>
<tr>
<td class="rh">3</td>
</tr>
<tr>
<td class="rh">4</td>
<td class="g0" rowspan="4">1</td>
<td class="g0" rowspan="4">5</td>
<td class="g0" rowspan="4">9</td>
<td class="g0" rowspan="4">13</td>
<td class="g0" rowspan="4">17</td>
<td class="g0" rowspan="4">21</td>
<td class="g0" rowspan="4">25</td>
<td class="g0" rowspan="4">29</td>
</tr>
<tr>
<td class="rh">5</td>
</tr>
<tr>
<td class="rh">6</td>
</tr>
<tr>
<td class="rh">7</td>
</tr>
<tr>
<td class="rh">8</td>
<td class="g0" rowspan="4">2</td>
<td class="g0" rowspan="4">6</td>
<td class="g0" rowspan="4">10</td>
<td class="g0" rowspan="4">14</td>
<td class="g0" rowspan="4">18</td>
<td class="g0" rowspan="4">22</td>
<td class="g0" rowspan="4">26</td>
<td class="g0" rowspan="4">30</td>
</tr>
<tr>
<td class="rh">9</td>
</tr>
<tr>
<td class="rh">10</td>
</tr>
<tr>
<td class="rh">11</td>
</tr>
<tr>
<td class="rh">12</td>
<td class="g0" rowspan="4">3</td>
<td class="g0" rowspan="4">7</td>
<td class="g0" rowspan="4">11</td>
<td class="g0" rowspan="4">15</td>
<td class="g0" rowspan="4">19</td>
<td class="g0" rowspan="4">23</td>
<td class="g0" rowspan="4">27</td>
<td class="g0" rowspan="4">31</td>
</tr>
<tr>
<td class="rh">13</td>
</tr>
<tr>
<td class="rh">14</td>
</tr>
<tr>
<td class="rh">15</td>
</tr>
<tr>
<td class="rh">16</td>
<td class="g1" rowspan="4">0</td>
<td class="g1" rowspan="4">4</td>
<td class="g1" rowspan="4">8</td>
<td class="g1" rowspan="4">12</td>
<td class="g1" rowspan="4">16</td>
<td class="g1" rowspan="4">20</td>
<td class="g1" rowspan="4">24</td>
<td class="g1" rowspan="4">28</td>
</tr>
<tr>
<td class="rh">17</td>
</tr>
<tr>
<td class="rh">18</td>
</tr>
<tr>
<td class="rh">19</td>
</tr>
<tr>
<td class="rh">20</td>
<td class="g1" rowspan="4">1</td>
<td class="g1" rowspan="4">5</td>
<td class="g1" rowspan="4">9</td>
<td class="g1" rowspan="4">13</td>
<td class="g1" rowspan="4">17</td>
<td class="g1" rowspan="4">21</td>
<td class="g1" rowspan="4">25</td>
<td class="g1" rowspan="4">29</td>
</tr>
<tr>
<td class="rh">21</td>
</tr>
<tr>
<td class="rh">22</td>
</tr>
<tr>
<td class="rh">23</td>
</tr>
<tr>
<td class="rh">24</td>
<td class="g1" rowspan="4">2</td>
<td class="g1" rowspan="4">6</td>
<td class="g1" rowspan="4">10</td>
<td class="g1" rowspan="4">14</td>
<td class="g1" rowspan="4">18</td>
<td class="g1" rowspan="4">22</td>
<td class="g1" rowspan="4">26</td>
<td class="g1" rowspan="4">30</td>
</tr>
<tr>
<td class="rh">25</td>
</tr>
<tr>
<td class="rh">26</td>
</tr>
<tr>
<td class="rh">27</td>
</tr>
<tr>
<td class="rh">28</td>
<td class="g1" rowspan="4">3</td>
<td class="g1" rowspan="4">7</td>
<td class="g1" rowspan="4">11</td>
<td class="g1" rowspan="4">15</td>
<td class="g1" rowspan="4">19</td>
<td class="g1" rowspan="4">23</td>
<td class="g1" rowspan="4">27</td>
<td class="g1" rowspan="4">31</td>
</tr>
<tr>
<td class="rh">29</td>
</tr>
<tr>
<td class="rh">30</td>
</tr>
<tr>
<td class="rh">31</td>
</tr>
</table>

<div class="legend">
  <span><i class="sw g0"></i>b[0] = K 上半（行 0–15），每格 4 个沿 K 的 fp8</span>
  <span><i class="sw g1"></i>b[1] = K 下半（行 16–31）</span>
</div>

<div class="box">lane = gid * 4 + tig              // 这个 4 = 一组 4 个 lane（32 / N=8，来自指令的 n8）
gid = lane / 4          (0..7)   // 同上：组号 = B 的列号 n
tig = lane % 4          (0..3)   // 同上：组内第几人
n   = gid
k   = 4 * tig + 16 * r + j
      // 4*tig 的 4 = 每寄存器沿 K 装 4 个 e4m3（32 bit / 8 bit）
      // 16 = K 半区高度（32/2）；r=0 上半 / r=1 下半
      // j=0..3 还是这同一个 4：寄存器内第几个 fp8

T0（gid=0, tig=0）：b[0] 在 k=0..3、n=0；b[1] 在 k=16..19、n=0。
同一列由 4 个连续 lane 合管（T0–T3 管 n=0，T4–T7 管 n=1，…）——这里的 4 仍是「一组 4 人」。

</div>

<div class="fig">

## Fragment 图的规律（B 转置后 B^T）

<p class="lead">B 存成转置 <b>B^T[n][k]</b>(K 连续)后,每个 lane 的 2 个寄存器落在<b>同一行</b>(n=gid)上,只差 <b>K 左右半区</b>——没有上下半区之分。
每格横跨 <b>4 列</b>(一个 b32 里 4 个 e4m3,沿 K 相邻)。T0 的 b[0] 占 k=0–3、b[1] 占 k=16–19。</p>
<p class="axis">n (N) ↓ &nbsp;&nbsp; k (K) → &nbsp; 格子里是持有该元素的 lane 编号。B^T = 8×32。</p>

<table class="grid">
<tr><th></th><th>0</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th><th>6</th><th>7</th><th>8</th><th>9</th><th>10</th><th>11</th><th>12</th><th>13</th><th>14</th><th>15</th><th>16</th><th>17</th><th>18</th><th>19</th><th>20</th><th>21</th><th>22</th><th>23</th><th>24</th><th>25</th><th>26</th><th>27</th><th>28</th><th>29</th><th>30</th><th>31</th></tr>
<tr>
<td class="rh">0</td>
<td class="g0" colspan="4">0</td>
<td class="g0" colspan="4">1</td>
<td class="g0" colspan="4">2</td>
<td class="g0" colspan="4">3</td>
<td class="g1" colspan="4">0</td>
<td class="g1" colspan="4">1</td>
<td class="g1" colspan="4">2</td>
<td class="g1" colspan="4">3</td>
</tr>
<tr>
<td class="rh">1</td>
<td class="g0" colspan="4">4</td>
<td class="g0" colspan="4">5</td>
<td class="g0" colspan="4">6</td>
<td class="g0" colspan="4">7</td>
<td class="g1" colspan="4">4</td>
<td class="g1" colspan="4">5</td>
<td class="g1" colspan="4">6</td>
<td class="g1" colspan="4">7</td>
</tr>
<tr>
<td class="rh">2</td>
<td class="g0" colspan="4">8</td>
<td class="g0" colspan="4">9</td>
<td class="g0" colspan="4">10</td>
<td class="g0" colspan="4">11</td>
<td class="g1" colspan="4">8</td>
<td class="g1" colspan="4">9</td>
<td class="g1" colspan="4">10</td>
<td class="g1" colspan="4">11</td>
</tr>
<tr>
<td class="rh">3</td>
<td class="g0" colspan="4">12</td>
<td class="g0" colspan="4">13</td>
<td class="g0" colspan="4">14</td>
<td class="g0" colspan="4">15</td>
<td class="g1" colspan="4">12</td>
<td class="g1" colspan="4">13</td>
<td class="g1" colspan="4">14</td>
<td class="g1" colspan="4">15</td>
</tr>
<tr>
<td class="rh">4</td>
<td class="g0" colspan="4">16</td>
<td class="g0" colspan="4">17</td>
<td class="g0" colspan="4">18</td>
<td class="g0" colspan="4">19</td>
<td class="g1" colspan="4">16</td>
<td class="g1" colspan="4">17</td>
<td class="g1" colspan="4">18</td>
<td class="g1" colspan="4">19</td>
</tr>
<tr>
<td class="rh">5</td>
<td class="g0" colspan="4">20</td>
<td class="g0" colspan="4">21</td>
<td class="g0" colspan="4">22</td>
<td class="g0" colspan="4">23</td>
<td class="g1" colspan="4">20</td>
<td class="g1" colspan="4">21</td>
<td class="g1" colspan="4">22</td>
<td class="g1" colspan="4">23</td>
</tr>
<tr>
<td class="rh">6</td>
<td class="g0" colspan="4">24</td>
<td class="g0" colspan="4">25</td>
<td class="g0" colspan="4">26</td>
<td class="g0" colspan="4">27</td>
<td class="g1" colspan="4">24</td>
<td class="g1" colspan="4">25</td>
<td class="g1" colspan="4">26</td>
<td class="g1" colspan="4">27</td>
</tr>
<tr>
<td class="rh">7</td>
<td class="g0" colspan="4">28</td>
<td class="g0" colspan="4">29</td>
<td class="g0" colspan="4">30</td>
<td class="g0" colspan="4">31</td>
<td class="g1" colspan="4">28</td>
<td class="g1" colspan="4">29</td>
<td class="g1" colspan="4">30</td>
<td class="g1" colspan="4">31</td>
</tr>
</table>

<div class="legend">
  <span><i class="sw g0"></i>b[0] = K 左半（k=0–15），每格 4 个沿 K 的 fp8</span>
  <span><i class="sw g1"></i>b[1] = K 右半（k=16–31）</span>
</div>

<div class="box">lane = gid * 4 + tig              // 同 A/B
gid = lane / 4          (0..7)   // B^T 行号 = n
tig = lane % 4          (0..3)
n   = gid
k   = 4 * tig + 16 * r + j   // r=0 左半 / r=1 右半;j=0..3 寄存器内
      // 转置后无上下半区:同一个 n 行里,rb[0]/rb[1] 只差 K 半区
      // 半区偏移 16 = 4 个 lane × 4 个元素(即 4*tig+16 里 16 的来源)
T0（gid=0, tig=0）：b[0] 在 (n=0, k=0..3)；b[1] 在 (n=0, k=16..19)。
同一行(n=gid)的 K 由 4 个 lane 各管一个 4 元素组。</div>

</div>

<div class="fig">

## Fragment 图的规律（D）

<p class="lead">输出 D 是 <b>16×8 f32</b>。每个 lane 拿 4 个 f32（d[0..3]），铺成 <b>2×2</b> 的小块：
d0/d1 在上半行（row=gid）、d2/d3 在下半行（row=gid+8）；列方向每 lane 占 <b>2</b> 列（2*tig, 2*tig+1）。
注意与 A/B 不同：D 的每个"寄存器"只装 <b>1 个</b> f32，4 个 d 是 4 个独立 float，不是 K 方向的连续 chunk。</p>
<p class="axis">row (M) ↓ &nbsp;&nbsp; n (N) → &nbsp; 格子里是持有该元素的 lane 编号。D = 16×8。</p>

<table class="grid">
<tr><th></th><th>0</th><th>1</th><th>2</th><th>3</th><th>4</th><th>5</th><th>6</th><th>7</th></tr>
<tr>
<td class="rh">0</td>
<td class="g0">0</td>
<td class="g1">0</td>
<td class="g0">1</td>
<td class="g1">1</td>
<td class="g0">2</td>
<td class="g1">2</td>
<td class="g0">3</td>
<td class="g1">3</td>
</tr>
<tr>
<td class="rh">1</td>
<td class="g0">4</td>
<td class="g1">4</td>
<td class="g0">5</td>
<td class="g1">5</td>
<td class="g0">6</td>
<td class="g1">6</td>
<td class="g0">7</td>
<td class="g1">7</td>
</tr>
<tr>
<td class="rh">2</td>
<td class="g0">8</td>
<td class="g1">8</td>
<td class="g0">9</td>
<td class="g1">9</td>
<td class="g0">10</td>
<td class="g1">10</td>
<td class="g0">11</td>
<td class="g1">11</td>
</tr>
<tr>
<td class="rh">3</td>
<td class="g0">12</td>
<td class="g1">12</td>
<td class="g0">13</td>
<td class="g1">13</td>
<td class="g0">14</td>
<td class="g1">14</td>
<td class="g0">15</td>
<td class="g1">15</td>
</tr>
<tr>
<td class="rh">4</td>
<td class="g0">16</td>
<td class="g1">16</td>
<td class="g0">17</td>
<td class="g1">17</td>
<td class="g0">18</td>
<td class="g1">18</td>
<td class="g0">19</td>
<td class="g1">19</td>
</tr>
<tr>
<td class="rh">5</td>
<td class="g0">20</td>
<td class="g1">20</td>
<td class="g0">21</td>
<td class="g1">21</td>
<td class="g0">22</td>
<td class="g1">22</td>
<td class="g0">23</td>
<td class="g1">23</td>
</tr>
<tr>
<td class="rh">6</td>
<td class="g0">24</td>
<td class="g1">24</td>
<td class="g0">25</td>
<td class="g1">25</td>
<td class="g0">26</td>
<td class="g1">26</td>
<td class="g0">27</td>
<td class="g1">27</td>
</tr>
<tr>
<td class="rh">7</td>
<td class="g0">28</td>
<td class="g1">28</td>
<td class="g0">29</td>
<td class="g1">29</td>
<td class="g0">30</td>
<td class="g1">30</td>
<td class="g0">31</td>
<td class="g1">31</td>
</tr>
<tr>
<td class="rh">8</td>
<td class="g2">0</td>
<td class="g3">0</td>
<td class="g2">1</td>
<td class="g3">1</td>
<td class="g2">2</td>
<td class="g3">2</td>
<td class="g2">3</td>
<td class="g3">3</td>
</tr>
<tr>
<td class="rh">9</td>
<td class="g2">4</td>
<td class="g3">4</td>
<td class="g2">5</td>
<td class="g3">5</td>
<td class="g2">6</td>
<td class="g3">6</td>
<td class="g2">7</td>
<td class="g3">7</td>
</tr>
<tr>
<td class="rh">10</td>
<td class="g2">8</td>
<td class="g3">8</td>
<td class="g2">9</td>
<td class="g3">9</td>
<td class="g2">10</td>
<td class="g3">10</td>
<td class="g2">11</td>
<td class="g3">11</td>
</tr>
<tr>
<td class="rh">11</td>
<td class="g2">12</td>
<td class="g3">12</td>
<td class="g2">13</td>
<td class="g3">13</td>
<td class="g2">14</td>
<td class="g3">14</td>
<td class="g2">15</td>
<td class="g3">15</td>
</tr>
<tr>
<td class="rh">12</td>
<td class="g2">16</td>
<td class="g3">16</td>
<td class="g2">17</td>
<td class="g3">17</td>
<td class="g2">18</td>
<td class="g3">18</td>
<td class="g2">19</td>
<td class="g3">19</td>
</tr>
<tr>
<td class="rh">13</td>
<td class="g2">20</td>
<td class="g3">20</td>
<td class="g2">21</td>
<td class="g3">21</td>
<td class="g2">22</td>
<td class="g3">22</td>
<td class="g2">23</td>
<td class="g3">23</td>
</tr>
<tr>
<td class="rh">14</td>
<td class="g2">24</td>
<td class="g3">24</td>
<td class="g2">25</td>
<td class="g3">25</td>
<td class="g2">26</td>
<td class="g3">26</td>
<td class="g2">27</td>
<td class="g3">27</td>
</tr>
<tr>
<td class="rh">15</td>
<td class="g2">28</td>
<td class="g3">28</td>
<td class="g2">29</td>
<td class="g3">29</td>
<td class="g2">30</td>
<td class="g3">30</td>
<td class="g2">31</td>
<td class="g3">31</td>
</tr>
</table>

<div class="legend">
  <span><i class="sw g0"></i>d[0] = 上行、偶数列（row gid, col 2*tig）</span>
  <span><i class="sw g1"></i>d[1] = 上行、奇数列（row gid, col 2*tig+1）</span>
  <span><i class="sw g2"></i>d[2] = 下行、偶数列（row gid+8, col 2*tig）</span>
  <span><i class="sw g3"></i>d[3] = 下行、奇数列（row gid+8, col 2*tig+1）</span>
</div>

<div class="box">lane = gid * 4 + tig              // 同 A/B
gid = lane / 4          (0..7)
tig = lane % 4          (0..3)
row = gid + 8 * (r / 2)         // 8 = M 半区行数（16/2）；r=0,1 上半 / r=2,3 下半
col = 2 * tig + (r % 2)         // 每 lane 占 2 列；r%2 选左/右
      // 每 lane 4 个寄存器 = 4 个独立 f32，不是连续 chunk
      // 写回：d[0]→(gid,2tig)  d[1]→(gid,2tig+1)  d[2]→(gid+8,2tig)  d[3]→(gid+8,2tig+1)
T0（gid=0, tig=0）：d[0..3] = (0,0)(0,1)(8,0)(8,1)。</div>

</div>
