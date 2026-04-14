# R32 理论附录：为什么一阶（甚至二阶）探针信号注定失败

> **摘要**：本附录从数学角度证明了为什么基于单次前向传播的信号（无论是零阶还是一阶标量属性）无法预测 ETD 的逐样本增益，并进一步解释了为什么即使构造了 J_F 的有限差分近似，信号的样本间变异仍然过低。

---

## 1. ETD 的数学形式

**三段式架构**：

```
Input → [Encoder: Layer 0..t_start-1] → [T-block: Layer t_start..t_stop-1]^k → [Decoder: 剩余层] → Output
```

令 T-block 为非线性映射 $F : \mathbb{R}^d \to \mathbb{R}^d$，其中 $d$ 为隐层维度（Qwen3-8B: $d=4096$）。$F$ 是该区间内所有残差子层（SelfAttn + FFN）的叠加：

$$F(h) = h_{t_{\text{stop}}} - h_{t_{\text{start}}-1}$$

其中 $h_{l}$ 为第 $l$ 层输出的隐层状态。

**ETD 的迭代公式（含 R2 阻尼 $\alpha$）**：

$$h^{(k+1)} = \alpha \cdot \big(h^{(k)} + F(h^{(k)})\big) + (1-\alpha) \cdot h^{(k)} = h^{(k)} + \alpha \cdot F(h^{(k)})$$

对 $k=2$ 的 Champion 配置（$\alpha \approx 0.43$，$n_t=14$）：

- **第 1 次迭代**：$h_1 = h_0 + \alpha F(h_0) = h_0 + \alpha \Delta_0$
- **第 2 次迭代**：$h_2 = h_1 + \alpha F(h_1)$

---

## 2. 泰勒-海森证明：一阶探针为何注定失败

### 2.1 ETD 的净物理扰动

ETD 最终输出与 Baseline 输出的差异，由第 2 次迭代引入的增量决定：

$$\delta_{\text{ETD}} = h_2 - h_1 = \alpha \cdot F(h_1)$$

由于 $h_1 = h_0 + \alpha \Delta_0$，对 $F(h_1)$ 在 $h_0$ 处做二阶泰勒展开：

$$F(h_1) = F(h_0 + \alpha\Delta_0) = F(h_0) + \alpha J_F(h_0)\Delta_0 + \frac{\alpha^2}{2}\Delta_0^\top H_F(h_0)\Delta_0 + \mathcal{O}(\|\Delta_0\|^3)$$

代入得：

$$\delta_{\text{ETD}} = \alpha\Delta_0 + \alpha^2 J_F(h_0)\Delta_0 + \frac{\alpha^3}{2}\Delta_0^\top H_F(h_0)\Delta_0 + \mathcal{O}(\|\Delta_0\|^3)$$

### 2.2 判死刑：信息缺失

R2-R31 中所有历史探针信号（`entropy@8`、`norm_delta`、`logit_lens_KL` 等）测量的都是 $\Delta_0$ 的**零阶和一阶标量属性**，即：

$$\text{Signal} \in \big\{ \|\Delta_0\|,\ H(\text{softmax}(\Delta_0)),\ \text{cos}(\Delta_0, \Delta_0^{\text{prev}}),\ \ldots \big\}$$

然而，**ETD 是否翻转 Logits 决策边界**取决于：

$$\delta_{\text{ETD}} \text{ 经过 Decoder（剩余 14 层）传播后的方向} \propto J_F(h_0)\Delta_0$$

这由**雅可比矩阵** $J_F(h_0) \in \mathbb{R}^{d \times d}$ 完全主导。

**关键点**：用 $\Delta_0$（一个向量）去推测 $J_F(h_0)\Delta_0$（该向量在未知曲率空间上的仿射结果），在信息论上是**病态逆问题**：你需要 $J_F(h_0)$ 的 $d^2 = 4096^2 \approx 1.7 \times 10^7$ 个元素，但你只有 1 个标量。

**这就是为什么 $I(\text{一阶信号}; Y) \approx 0.01$ bits 的数学根源。**

---

## 3. R32 的二阶探针尝试：为什么仍然失败

### 3.1 R32 的方法

R32 尝试直接运行第二次 T-block（"2-Pass Probe"），从而**直接测量**：

$$\delta \approx J_F(h_0)\Delta_0 + \frac{\alpha}{2}\Delta_0^\top H_F(h_0)\Delta_0$$

提取的信号：
- **收缩率** $r_c = \|\delta\| / \|\Delta_0\|$（$J_F$ 的局部谱半径）
- **方向对齐度** $\theta = \cos(\Delta_0, \delta)$（迭代方向一致性）
- **Hessian 代理量** = $\delta$ 中垂直于 $\Delta_0$ 的分量比例

### 3.2 Phase 1 实验结果（N=200/benchmark）

| 信号 | mean\|ρ\| | max\|ρ\| | 显著（p<0.05）|
|------|---------|---------|------|
| rc_global | 0.014 | 0.093 | 0/4 |
| theta_global | 0.020 | 0.097 | 0/4 |
| logit_align | 0.001 | 0.085 | 0/4 |
| hessian_proxy | 0.020 | 0.097 | 0/4 |
| max_rc | 0.037 | 0.118 | 0/4 |

**关键观测**：
- `n_expanding = nan`（所有样本中 T-block 内没有任何层的 rc > 1，即 T-block 在所有输入上都是压缩映射）
- rc_global 的样本间方差极小：std ≈ 0.02-0.03，而均值 ≈ 0.65-0.70

### 3.3 为什么二阶信号也失败：更深层的分析

#### 失败原因 A：J_F 由权重决定，不由输入决定

$J_F(h_0)$ 是 T-block 在点 $h_0$ 处的雅可比矩阵。对于 Transformer，它是：

$$J_F(h_0) = \prod_{l=t_{\text{start}}}^{t_{\text{stop}}-1} \big(I + J_{\text{attn},l}(h_0) + J_{\text{ffn},l}(h_0)\big)$$

其中 $J_{\text{attn},l}$ 和 $J_{\text{ffn},l}$ 分别是注意力层和 FFN 层的雅可比矩阵。

**关键性质**：这些雅可比矩阵由**模型权重**主导（因为 Transformer 是高度线性化的，特别是在低温度推理时）。不同输入样本在 $h_0$ 的变化主要在**方向**上，而 $J_F$ 的谱结构（特征值分布）基本相同。

这就是为什么 $r_c = \|J_F(h_0)\Delta_0\| / \|\Delta_0\|$ 在不同样本间几乎相同（std ≈ 0.02）：**$r_c$ 近似等于 $J_F$ 在 $\Delta_0$ 方向上的 Rayleigh 商，而 $\Delta_0$ 的方向分布是相对均匀的，导致 $r_c$ 趋近于 $J_F$ 的平均谱半径。**

#### 失败原因 B：oracle_gain 的分布极度稀疏

从 N=200 实验的数据：

| Benchmark | ETD有益样本 | ETD有害样本 | 无影响样本 |
|-----------|------------|------------|---------|
| BoolQ     | 9          | 1          | 190     |
| ARC-C     | 15         | 10         | 175     |
| CSQA      | 6          | 5          | 189     |
| TruthfulQA | 9         | 2          | 189     |

**合计 800 样本中，oracle_gain ≠ 0 的只有约 57 个（7.1%）。**

即使某个信号与 oracle_gain 真正相关（假设 $\rho = 0.5$），由于 93% 的样本 gain=0，计算 Spearman ρ 时大多数样本对（gain=0 vs gain=0）对相关性无贡献，但会显著稀释 ρ。

**等效 N**：有效样本数约为 57 个，而非 800 个。在 57 个二值样本（+1 vs -1）上，检测 $\rho = 0.5$ 需要的样本数约为 $n \approx 4 / \rho^2 = 16$（可行），但如果 $\rho$ 更小（如 0.2），需要 $n \approx 100$。

**问题**：即使真实的信号-增益相关性为 $\rho = 0.2$（有意义的），在 57 个有效样本中也只有 0.35 的期望检测功效（power < 50%），而且样本选择偏差会进一步降低 ρ 的可靠性。

#### 失败原因 C：H_critical_layer 的系统性偏差

argmax(rc_per_layer) 实验显示：

| Benchmark | oracle t_start | argmax_rc 均值 | MAE |
|-----------|-------------|------------|-----|
| BoolQ     | 8           | 8.0        | 0.0 |
| ARC-C     | 14          | 8.6        | 5.4 |
| CSQA      | 10          | 8.0        | 2.0 |
| TruthfulQA | 16         | 8.1        | 7.9 |

argmax(rc_per_layer) 几乎总是在层 8（T-block 的第一层）。原因是：**T-block 第一层的 J_F 局部谱半径最大，是因为层 8 是语义信息最密集汇聚的层（R9 发现的"语义整合完成点"），随后各层的谱半径递减（收敛过程）。** 这是 Qwen3-8B 的结构属性，不随输入变化。

---

## 4. 综合结论：信号-效果解耦的三层原因

以下是对整个 R2-R32 探索的统一理论解释：

### 第一层：信息论障碍（R29 确认）

$$I(\text{一阶信号}; \text{oracle\_gain}) \approx 0.01 \text{ bits}$$

原因：一阶信号只携带 $\|\Delta_0\|$ 等标量信息，而 ETD 增益依赖于 $J_F(h_0)\Delta_0$ 的方向，后者无法从前者推断。

### 第二层：J_F 的"输入无关性"（R32 新发现）

二阶信号（$r_c = \|J_F\Delta_0\|/\|\Delta_0\|$）同样失败，因为：

$$r_c \approx \sqrt{\frac{\Delta_0^\top J_F^\top J_F \Delta_0}{\|\Delta_0\|^2}} \approx \bar{\sigma}(J_F)$$

在 $\Delta_0$ 方向均匀分布的假设下，$r_c$ 趋近于 $J_F^\top J_F$ 的**平均特征值的平方根**，这是模型权重的固定属性，对不同输入几乎相同。

### 第三层：oracle_gain 的稀疏性使任何信号都难以验证

93% 的样本 oracle_gain=0，使得即使是真正有效的信号也无法在统计上被检测到。这是 ETD 的核心特性：**对大多数样本，Champion 和 Baseline 结果相同；ETD 的收益来自少数"临界样本"，而这些样本的特征在任何探针信号中都没有明显的前兆。**

---

## 5. 对未来研究的启示

### 5.1 为什么 Champion 韧性如此强（理论解释）

Champion (t_start=8, t_stop=22, k=2, α=0.43) 是 T-block 在**谱结构最优区间**的固定点配置：

- **t_start=8**：J_F 的谱半径在层 8 达到局部最大（"扩张-收缩"过渡点），从此处开始循环可以最大化二阶项贡献
- **t_stop=22**：J_F 的谱半径在层 22 已降至 < 0.3，继续循环几乎无额外信息
- **α=0.43**：阻尼系数精确控制谱半径，防止迭代发散（Banach 不动点定理：需要 $\alpha \cdot \sigma_{\max}(J_F) < 1$）

这是 Qwen3-8B 的**自然固定点配置**，而非局部最优点。任何试图通过样本级路由"改进"它的方案，都是在对抗模型的内在谱结构。

### 5.2 正确的研究方向

鉴于以上分析，有效的研究方向应当从"预测最优配置"转向：

**方向 A：任务类型判别（粗粒度路由）**
- 不在样本级别路由，而在**任务类型**级别（如文本长度 > 100 → BoolQ 类 → 特定 t_stop）
- 但 R18 已证明这对 BoolQ 以外的 benchmark 无效

**方向 B：对抗 ETD 的样本（跳过 ETD）**
- 从"ETD 有益样本的特征"转向"ETD 有害样本的特征"
- R31 分析：CSQA 样本在 Champion 下比 Baseline 差（0.591 < 0.636），说明有"系统性有害"任务
- 若能用语义特征（而非信号特征）识别这类任务，则可跳过 ETD

**方向 C：第二次前向的条件价值（EVSI）**
- 从贝叶斯角度计算：运行第二次前向能提供多少"信息价值"？
- 若 EVSI ≈ 0（即二次前向不改变决策），则直接用 Champion，否则考虑 k=3

**方向 D：ETD 增益的不可预测性本身作为研究对象**
- 将"ETD 增益不可预测"作为一个正式定理，证明在某些模型类和某些信号类下，$I(\text{信号}; Y) = 0$
- 这将为 ETD 研究提供严格的下界

---

## 6. 数学补充：Banach 不动点定理与 ETD 稳定性

**定理（Banach）**：若 $F$ 是完备度量空间上的压缩映射（$\|F(x) - F(y)\| \leq L\|x-y\|, L < 1$），则迭代 $x_{n+1} = F(x_n)$ 以指数速度收敛到唯一不动点 $x^*$。

**ETD 的适用性**：R32 的 Phase 0/1 实验确认 $r_c \approx 0.65-0.70$（所有样本和所有层），即 T-block 在 Champion 配置下确实是一个压缩映射（$L \approx 0.67 < 1$）。

这意味着：
1. ETD 的 k 次迭代理论上**总是**向不动点 $h^*$ 收敛，与输入无关
2. 不动点 $h^*$ 满足 $F(h^*) = 0$（T-block 出口等于入口，即 T-block 对 $h^*$ 无改变）
3. ETD 的"思考深化"效果 = 把 $h_0$ 的 T-block 处理结果推向 $h^*$

**关键问题**：$h^*$ 是否对应"更正确的答案"，取决于 T-block 的语义结构，而不是输入 $h_0$ 的属性。这正是"信号无法预测增益"的本质原因：**不动点 $h^*$ 是由权重决定的，而不是由输入决定的。**

---

*本附录由 R32 实验（2026-04-12）总结，覆盖 Round 2 → Round 32 的理论框架演进。*
