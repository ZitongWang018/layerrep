# ETD 信号探索深度分析：为什么所有信号都「不显著」？

> 本文档记录 R30-R32 期间对 20+ 种逐层信号的实证剖面结果，汇总三组绘图实验的核心发现，并提出一个根源性假设来解释为何信号路由持续失败。

## 1. 实验概览

### 1.1 三组信号绘图

| 批次 | 信号数量 | 来源与目录 | 信号列表 |
|------|---------|-----------|---------|
| **R30 原始信号** | 10 | `figures/r30_optimal_by_layer/` | `attn_entropy`, `ffn_gate_norm`, `layer_sim`, `head_specialization`, `logit_lens_KL`, `attention_locality`, `residual_write_norm`, `participation_ratio`, `prediction_flip_rate`, `attn_sink_ratio` |
| **扩展理论信号** | 10 | `figures/proposed_signals_by_layer/` | `residual_delta_l2`, `contraction_ratio`, `logit_lens_jsd_vel`, `logit_lens_jsd_curv`, `erank`, `delta_erank`, `attn_consensus`, `delta_norm_to_tstart`, `attn_entropy`, `logit_top1_margin` |
| **ETD 五机制信号** | 5 | `figures/etd_five_signals_by_layer/` | 逐层 `contraction_ratio`(含块标量 CR_block), `logit_lens_jsd_vel`, `delta_erank`, `attn_consensus`, 逐层 `delta_norm_to_tstart`(含块标量 FPR_simple) |

基准：ARC-C、TruthfulQA、CSQA、MMLU-HS-Math，各 N=20 样本。
R30 最优 T-block 标注：ARC-C (14,20)、TruthfulQA (16,19)、CSQA (10,22)、MMLU-HS-Math (10,18)。

### 1.2 五机制信号的理论锚点

| 信号 | 理论基础 | 参考文献 |
|------|---------|---------|
| CR_block / contraction_ratio | Banach 不动点定理；收缩映射条件 K<1 时迭代收敛 | Bai et al., NeurIPS 2019 (DEQ); Ke et al., ICLR 2026 (Fixed-Point Iterations in DNNs) |
| JSD Velocity | 信息几何；统计流形上的分布变化速率 | Amari 2016; DistillLens (arXiv 2602.13567) |
| ΔeRank | 有效秩 = exp(H(σ̃))；信息瓶颈理论中的探索→压缩相变 | Roy & Vetterli 2007; "Layer by Layer" ICML 2025; "Attention Layers Add Into Low-Dimensional Residual Subspaces" ICLR 2024 |
| ACI (注意力共识指数) | 自注意力的粒子系统共识动力学；多头间共识度 | Geshkovski et al. 2023; "Consensus Is All You Get" ICML 2025; Krause Synchronization Transformers (Liu et al. 2026) |
| FPR_simple / delta_norm_to_tstart | DEQ 不动点残差；收缩轨迹上距稳态的距离 | Bai et al. 2019; "Consistency DEQ" (arXiv 2602.03024); RKSP (Kim et al. 2026) |

## 2. 核心观察：三个层面的「不显著」

### 2.1 跨基准：信号曲线形态几乎完全一致

**这是最致命的发现。** 对比四个基准的同一信号：

- **contraction_ratio**：层 5-6 有一个尖峰（~20-50），随后快速衰减到 ~1 附近，之后几乎不变。四个基准的形态**几乎重合**。
- **JSD Velocity**：层 0-5 快速下降，层 8-15 出现若干次级峰，层 20 后趋近 0。四基准形态**一致**。
- **ACI**：从 ~0.55 单调上升到 ~0.95。四基准形态**一致**。
- **erank / delta_erank**：层 0-6 急剧上升/下降，之后波动减小。四基准形态**一致**。
- **delta_norm_to_tstart**：从 1.0 单调上升（指数型）。四基准形态**一致**。

但四个基准的 R30 最优 T-block **完全不同**——ARC-C (14,20)、TruthfulQA (16,19)、CSQA (10,22)、MMLU-HS-Math (10,18)。

**结论**：信号刻画的是 **Qwen3-8B 的架构计算流**，不是任务特异的推理行为。信号没有信息量来区分「哪个 T-block 在这个基准上最优」，更遑论逐样本区分。

### 2.2 跨样本：样本间方差极小

观察所有图中淡蓝色个体曲线与黑色均值曲线的关系：

- **contraction_ratio**：除层 5-6 的尖峰外，20 条样本曲线在层 8+ 几乎完全重合于均值。
- **JSD velocity**：样本散布仅在层 10-20 略微增大，但散布的量级远小于 **信号本身在层轴上的变化量**。
- **ACI**：20 条曲线几乎不可区分。
- **logit_top1_margin**：在层 8-15（ETD 关键区间）样本间方差极其微小。

**数值化**：CR_block 在 ARC-C 上的 20 个样本值从 1.006 到 1.061，标准差 ~0.014。对比均值 ~1.04，变异系数仅 **1.3%**。MMLU-HS-Math 的 CR_block 从 0.993 到 1.028，标准差 ~0.010，变异系数 **1.0%**。

**结论**：在同一基准内，信号 **几乎没有逐样本区分度**。这意味着基于这些信号的任何门控/路由规则，在同一基准的不同样本上将给出 **几乎相同的决策**——退化为基准级别（而非样本级别）的路由。

### 2.3 CR_block 的唯一例外：任务级别的差异

CR_block 块标量在 **跨基准** 维度上显示了唯一有意义的分离：

| 基准 | CR_block 均值 | FPR_simple 均值 | 物理含义 |
|------|-------------|----------------|---------|
| **TruthfulQA** | **0.795** | **0.634** | T-block **收缩**：出口写入远小于入口 |
| MMLU-HS-Math | 1.011 | 1.090 | T-block **临界**：写入几乎不变 |
| ARC-C | 1.042 | 1.237 | T-block **轻度扩张** |
| CSQA | 1.066 | 2.018 | T-block **显著扩张** |

TruthfulQA 的 CR_block < 1（均值 0.795）是唯一真正「收缩」的基准。这与 TruthfulQA 在 R31 中 ETD 始终有效的事实一致——Banach 不动点理论预测：收缩映射的迭代收敛到不动点，第二次通过 T-block 使表示更接近稳态。

CSQA 的 FPR_simple ≈ 2.0 意味着块出口的写入是入口的两倍——T-block 在 **放大** 扰动。这与 CSQA 上 Champion 本身就不如 Baseline 的事实一致——对该任务，T-block 层在第一次通过时就已经在「发散」。

**但**：这是 **任务级别** 的差异，不是样本级别的。TruthfulQA 的 20 个样本的 CR_block 全部 < 1（0.72-0.84），没有交叉。无法用它在 TruthfulQA 内部区分不同样本。

## 3. 根源假设：信号-效果解耦定理

### 3.1 表述

> **对于一个冻结的、未经循环训练的 Transformer，T-block 重复对最终预测的影响由 T-block 出口隐藏状态附近的 *局部损失曲面曲率* 决定。这个曲率从第一次前向传播的信号中 *在原理上就不可观测*。**

### 3.2 推导

设第一次前向传播得到隐藏状态轨迹 \(\{h_0, h_1, \ldots, h_L\}\)，其中 \(h_l = F_l(h_{l-1})\)。

ETD 在 T-block \([t_s, t_e)\) 上做第二次通过：

\[
h'_{t_e} = \alpha \cdot F_{t_s \to t_e}(h^{(1)}_{t_e}) + (1-\alpha) \cdot h^{(1)}_{t_e}
\]

ETD 改变最终预测 ⇔ \(\arg\max_v P(v \mid h'_L) \neq \arg\max_v P(v \mid h_L)\)。

这取决于 \(\Delta h = h'_L - h_L\) 是否跨越了 argmax 的决策边界。而 \(\Delta h\) 的大小和方向由：

1. **\(F_{t_s \to t_e}\) 的 Jacobian**（在 \(h^{(1)}_{t_e}\) 处）决定 \(h'\) 偏离 \(h^{(1)}\) 的方向
2. **后续层 \([t_e, L)\) 的传播** 将偏移 amplify 或 attenuate 到 \(h_L\)
3. **lm_head 的决策边界几何** 决定这个偏移是否翻转 argmax

我们测量的所有信号——无论是 `residual_write_norm`、`JSD velocity`、`erank` 还是 `ACI`——都是 \(\{h_0, \ldots, h_L\}\) 的函数。它们捕捉的是 **第一次通过** 的动力学。但 ETD 的效果取决于 **第二次通过** 创造的 \(\Delta h\)，这是一个 \(F\) 的 **二阶性质**（Jacobian / Hessian），而非一阶统计量。

**类比**：预测「推一辆球会不会翻过山坡」需要知道山坡那一侧的坡度（二阶信息），而不是球当前的速度（一阶信息）。我们的所有信号都在测量球的速度。

### 3.3 为什么跨样本方差如此小？

这也有一个原理性的解释：

**权重共享 + 残差连接 ⟹ 表示同质化。**

Transformer 的每一层 \(F_l\) 的参数对所有输入是共享的。残差连接 \(h_l = h_{l-1} + \text{Attn}(h_{l-1}) + \text{FFN}(\ldots)\) 意味着 \(h_l\) 的主分量始终是 \(h_{l-1}\)，增量部分相对于主分量很小。因此：

- \(\|h_l - h_{l-1}\| / \|h_{l-1}\| \ll 1\)（residual_write_norm 通常 < 0.15）
- \(\cos(h_l, h_{l-1}) \approx 0.95+\)（layer_sim 几乎为 1）
- \(\delta_l / \delta_{l-1} \approx 1\)（contraction_ratio 在 T-block 内几乎恒定）

这些比值由 **层的参数范数** 主导，而非由 **输入内容** 主导。只有在模型做出 argmax 决策的最后几层（层 30+），输入差异才通过 lm_head 的放大作用显现——但那时已经超出了 T-block 的范围。

我们的信号采集跨所有 token 做均值（attn_entropy 是 B×H×S 的平均，residual_delta_l2 是 B×S 的平均），进一步压缩了本就微弱的逐样本差异。

### 3.4 反例检验：TruthfulQA 为何「看起来不同」？

TruthfulQA 的 CR_block < 1 与其他基准的 CR_block > 1 形成了对比。但这不是「TruthfulQA 的样本使 T-block 收缩」——而是 TruthfulQA 的 R30 最优 T-block **(16, 19) 只有 3 层**，而 ARC-C 是 6 层、CSQA 是 12 层。

层数越少的 T-block 更容易表现出收缩，因为：
- δ 序列在层 16-18 的自然趋势（由架构决定）恰好是递减的
- 跨越更多层会包含 δ 递增的区段，使几何平均 > 1

换言之，CR_block 的跨基准差异 **不是因为不同基准的样本在物理上使 T-block 收缩/扩张**，而是因为 **不同基准的 R30 最优 T-block 落在了 δ 序列的不同区段**。这是一个 **循环论证**：最优 T-block 的定义已经包含了「哪个区段有益」的信息。

## 4. 不可行性论证：为什么「更好的信号」不是答案

### 4.1 已穷尽的信号空间

经过 R29-R32 的三轮探索，我们已经覆盖了以下信号类别：

| 类别 | 具体信号 | 刻画维度 | 结果 |
|------|---------|---------|------|
| **输出分布** | logit_lens_entropy, logit_lens_KL, prediction_flip_rate, logit_lens_jsd_vel, logit_lens_jsd_curv, logit_top1_margin | 模型预测的确定性 / 稳定性 / 变化率 | 跨样本方差极小 |
| **残差流动力学** | residual_write_norm, residual_delta_l2, contraction_ratio, delta_norm_to_tstart, layer_sim, CR_block, FPR_simple | 层间写入幅度 / 收缩率 / 相对距离 | 主要反映架构常数 |
| **注意力结构** | attn_entropy, head_specialization, attention_locality, attn_sink_ratio, attn_consensus (ACI) | 注意力模式的集中度 / 多样性 / 共识 | 跨样本方差极小 |
| **表示几何** | participation_ratio, erank, delta_erank | 隐藏状态的有效维度 | 主要反映架构常数 |
| **FFN 活动** | ffn_gate_norm | 前馈网络的激活强度 | 完全无区分度 |

总计 **22 个独立信号**，覆盖了从输出空间到表示空间、从一阶到二阶统计量、从全序列到末 token、从单层到块级的所有维度。**没有一个信号在样本级别上显示出与 ETD 有效性的稳定关联。**

### 4.2 信息论论证

设 \(X\) = 第一次前向的信号向量，\(Y\) = ETD 是否改变答案（二值）。

R29 Phase 0 已发现：在 Champion 配置下，~90% 的样本 ETD 不改变答案。即 \(H(Y) \approx 0.47\) bits（极低基础熵）。

即使存在完美信号，\(I(X; Y) \leq H(Y) \approx 0.47\) bits。但从 R29 的 Pearson 相关分析，所有信号的 \(|r| \leq 0.14\)，对应 \(I(X;Y) \approx 0.01\) bits——信号携带的关于 ETD 效果的信息不到理论上限的 **2%**。

这不是「信号不够多」的问题。是 **信息根本不在第一次前向传播的可观测量中**。

### 4.3 为什么 LoopFormer 成功但 ETD 信号路由失败

LoopFormer (Jeddi et al., 2026) 通过 **shortcut-consistency training** 解决了同样的问题——让模型在不同循环深度下都产出高质量表示。关键区别：

| | ETD (本研究) | LoopFormer |
|---|---|---|
| 模型 | 冻结的预训练 Qwen3-8B | 从头训练或持续训练 |
| 循环决策 | 第一次前向的信号 → 启发式规则 | 模型内置的时间/步长条件化 |
| 保证 | 无（信号与效果解耦） | 训练目标包含一致性损失 |
| 本质 | 给不会游泳的人挑救生衣 | 教人游泳 |

**ETD 的根本困难在于：Qwen3-8B 从未被训练为循环架构。** 重复 T-block 是对模型的一种 *域外扰动*（OOD perturbation）。第一次前向传播的信号无法预测 OOD 扰动的效果，正如你无法通过看一个人的正常步态来预测他在结冰路面上会不会摔倒。

## 5. 前进方向

### 5.1 放弃信号路由，转向任务/输入类别路由

CR_block 数据显示，ETD 效果的方差来源主要是 **任务类别**，而非样本个体。因此：

| 输入特征 | 推荐操作 | 依据 |
|---------|---------|------|
| 事实核查型（TruthfulQA 类：短问句 + 问号 + 无上下文段落） | 强制 ETD | CR_block < 1；R31 中所有 ETD 变体均超过 BL |
| 数学/计算型（MMLU-Math 类：数学符号密度高） | 跳过 ETD | CR_block ≈ 1 且 ETD 无增益 (R27: −0.011) |
| 推理/常识型（ARC-C/CSQA 类：需推理链） | 使用固定 Champion | CR_block > 1 但总体有增益 |

这是一个 **三路分类器**，输入特征为：问题长度、数学符号比例、是否包含上下文段落等 **文本表面特征**（非模型内部信号）。

### 5.2 如果仍要做信号路由：唯一可能的方向

如果一定要用模型内部信号做逐样本决策，需要 **跨越信号-效果解耦**：

**方案 A：Two-Pass 差分信号**  
在第一次前向（probe）和第二次前向（ETD）中 **都** 收集信号，然后比较差异。如果 ETD 通过后信号剧变（如 JSD velocity 在 ETD 后骤降），说明 ETD 产生了实质性效果。但这需要先执行 ETD，失去了「提前决定是否做 ETD」的意义——除非用它来训练一个轻量级预测器（需要标注数据）。

**方案 B：直接训练 Skip Gate（需要少量标注）**  
收集 ~500-1000 条样本的 (probe_signals, etd_outcome) 数据对，训练一个简单的 logistic regression 或小 MLP 来预测「ETD 是否会改变答案」。这绕开了信号-效果解耦问题，因为模型通过有监督学习来捕捉信号与效果之间的（可能非线性、高维的）关联。

### 5.3 更根本的方向：从 ETD 走向 Looped Transformer

ETD 是一种 **推理时干预**（inference-time intervention），将循环机制硬嫁接到非循环模型上。根据本分析，这种嫁接的天花板非常低（macro 提升 ~0.02，且不稳定）。

如果目标是让 Transformer 具备自适应计算深度，正确的路径是：
1. **LoopFormer 范式**（Jeddi et al., 2026）：在预训练/后训练中引入 shortcut-consistency 损失
2. **DEQ 范式**（Bai et al., 2019）：将中间层建模为隐式不动点方程
3. **Early Exit 范式**：训练每层的退出分类头

这些方向都需要 **修改训练过程**，而非在冻结模型上做推理时搜索。

## 6. 总结

| 问题 | 回答 |
|------|------|
| 22 个信号中有显著信号吗？ | **没有**，在样本级别上。有一个任务级别的差异（CR_block < 1 ⟺ TruthfulQA），但它的信息量仅限于任务分类，无法用于逐样本路由。 |
| 是信号设计的问题吗？ | **不是**。已覆盖输出分布、残差动力学、注意力结构、表示几何四大类，从一阶到二阶，22 个维度。信号空间已近饱和。 |
| 有更本质的原因吗？ | **有**。对冻结模型做层循环是一种 OOD 扰动；扰动的效果取决于损失曲面的局部曲率（二阶/高阶信息），而非第一次前向可观测的一阶统计量。这就是「信号-效果解耦」。 |
| 下一步该做什么？ | (a) 放弃逐样本信号路由，改用 **任务/输入类别** 级别的简单分类；(b) 如需信号路由，走有监督 Skip Gate（需标注数据）；(c) 更根本地，走 LoopFormer / DEQ 方向做训练时干预。 |

---

## 附录：关键文献

| 文献 | 关联 |
|------|------|
| Bai et al., NeurIPS 2019 "Deep Equilibrium Models" | DEQ 理论基础；不动点迭代收敛条件 |
| Ke et al., submitted to ICLR 2026 "Advancing the understanding of fixed point iterations in deep neural networks" | 循环网络可存在 2^d 个不动点 |
| Roy & Vetterli, 2007 "The Effective Rank" | erank 定义 |
| Nait Saada et al., ICML 2025 "Mind the Gap: Spectral Analysis of Rank Collapse" | 注意力层谱间隙导致秩坍缩 |
| Geshkovski et al., 2023 + Chen et al., ICML 2025 "Consensus Is All You Get" | 自注意力的共识动力学 |
| Jeddi et al., 2026 "LoopFormer: Elastic-Depth Looped Transformers" | 弹性深度循环 Transformer + shortcut-consistency 训练 |
| Kim et al., 2026 "Residual Koopman Spectral Profiling" | Koopman 谱分析诊断 Transformer 稳定性 |
| "Consistency DEQ" (arXiv 2602.03024, 2026) | DEQ 一致性蒸馏与轨迹锚定 |

---

*本文档与 `plan_etd_five_signals.md`、`plan_proposed_signals_experiment.md` 属于同一实验周期。最后更新：2026-04-11。*
