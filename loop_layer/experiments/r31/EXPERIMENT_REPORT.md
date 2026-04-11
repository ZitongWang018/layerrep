# R31 实验报告：信号路由自适应 ETD（中等规模复现）
- **生成时间**：2026-04-10 18:51 UTC
- **模型**：`/root/autodl-tmp/model_qwen`
- **每基准样本数**：44；Oracle 子集每基准前 18 题
- **Phase1 墙钟**：356.23 s；**Phase2 墙钟**：398.33 s；**合计**：754.56 s

## 1. Phase1：与固定基线对比（准确率）
| Benchmark | n | baseline | Champion | macro_top1 | adaptive_p2 | adaptive_p1_rule |
|-----------|---|----------|----------|------------|-------------|------------------|
| ARC-C | 44 | 0.4318 | 0.5455 | 0.4091 | 0.2727 | 0.2500 |
| TruthfulQA | 44 | 0.1591 | 0.2045 | 0.2273 | 0.2727 | 0.2500 |
| CSQA | 44 | 0.6364 | 0.5909 | 0.5682 | 0.2273 | 0.2273 |
| MMLU-HS-Math | 44 | 0.3636 | 0.4091 | 0.4545 | 0.2727 | 0.2727 |
| **macro 平均** | — | 0.3977 | 0.4375 | 0.4148 | 0.2614 | 0.2500 |

- **adaptive_p2**：Phase2 风格路由器 + H2/H3（与 `predict_mc_adaptive` 一致）
- **adaptive_p1_rule**：`route_phase1_style`（H1 阈值 1.0 + 2 层结晶）

### Phase1 辅助指标
| Benchmark | oracle_hit_rate | routing_acc (vs prior) | t_start MAE (oracle 子集) |
|-----------|-----------------|-------------------------|----------------------------|
| ARC-C | 0.666667 | 0.386364 | 7.0 |
| TruthfulQA | 0.277778 | 0.227273 | 8.2 |
| CSQA | 0.833333 | 0.886364 | 7.4667 |
| MMLU-HS-Math | 0.666667 | 0.613636 | 7.25 |

## 2. Phase2：消融（macro 平均准确率）
| 变体 | macro avg |
|------|----------|
| Champion | 0.4375 |
| MacroTop1 | 0.4148 |
| Baseline | 0.3977 |
| AdaptiveH2only | 0.2898 |
| AdaptiveH2H3 | 0.2614 |
| AdaptiveH2H3Dual | 0.2614 |
| AdaptiveH3only | 0.2102 |

### Phase2 分基准
| 变体 | ARC-C | TruthfulQA | CSQA | MMLU-HS-Math |
|---|---|---|---|---|
| Baseline | 0.4318 | 0.1591 | 0.6364 | 0.3636 |
| Champion | 0.5455 | 0.2045 | 0.5909 | 0.4091 |
| MacroTop1 | 0.4091 | 0.2273 | 0.5682 | 0.4545 |
| AdaptiveH3only | 0.2045 | 0.3409 | 0.1136 | 0.1818 |
| AdaptiveH2only | 0.3182 | 0.2500 | 0.2500 | 0.3409 |
| AdaptiveH2H3 | 0.2727 | 0.2727 | 0.2273 | 0.2727 |
| AdaptiveH2H3Dual | 0.2727 | 0.2727 | 0.2273 | 0.2727 |

## 3. 图

### Phase1：预测 t_start vs Oracle-lite
![](../figures/r31_t_start_prediction_scatter.png)

### Phase1：路由混淆矩阵（相对 benchmark 先验）
![](../figures/r31_routing_confusion_matrix.png)

### Phase2：各变体 macro 柱状图
![](../figures/r31_adaptive_vs_fixed_bars.png)

### Phase2：分基准分组柱
![](../figures/r31_phase2_per_benchmark.png)

## 4. 结论（基于本次运行）
1. **macro 准确率**：Phase1 的 adaptive_p2=0.2614，低于 baseline=0.3977（更低）、Champion=0.4375（更低）与 macro_top1=0.4148（更低）。 说明在本配置（每基准 44 题、当前路由与 ETD 网格）下，自适应路由尚未带来整体收益，反而明显拉低 macro。
2. **Phase2 消融**：完整自适应 AdaptiveH2H3=0.2614；仅 H2=0.2898；仅 H3=0.2102。 AdaptiveH3only 最弱，符合「仅调 t_stop、不调 t_start」信息不足的直觉。 AdaptiveH2H3 (0.2614) 与 AdaptiveH2H3Dual (0.2614)：二者数值相同，本次运行中双候选合并未带来可测增益。
3. **分任务现象**：TruthfulQA 上 adaptive_p2=0.2727 高于 baseline=0.1591；CSQA 上 adaptive_p2=0.2273 远低于 baseline=0.6364；ARC-C 上 adaptive_p2=0.2727 低于 baseline=0.4318。 macro 被后两类大幅拉低。路由辅助指标在 CSQA 上 oracle_hit 与 routing_acc 较高，在 TruthfulQA 上偏低，与分任务表现分化一致。
4. **耗时**：Phase1+Phase2 墙钟见文首；若需更接近 15 分钟预算，可在保持环境不变的前提下微调每基准样本数（例如略减 `samples_per_bench`）。
