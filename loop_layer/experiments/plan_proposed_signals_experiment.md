# 实验计划：R32 提议信号（理论驱动）剖面采集与可视化

## 1. 目标

在 **与 R30 最优 T-block 标注一致** 的前提下，对四个基准各 **N=20** 条样本做单次前向探针，提取文献讨论中提出的 **扩展信号**，并生成与 `figures/r30_optimal_by_layer/` **同版式** 的逐层曲线图（个体样本淡线 + 均值黑线 + R30 `t_start`/`t_stop` 竖线与区间浅黄底）。

输出目录：**`experiments/figures/proposed_signals_by_layer/`**  
元数据：**`experiments/results/proposed_signals_by_layer_plot_meta.json`**

## 2. 背景与动机（摘要）

- R29/R31 表明：现有 10 个 Lite Probe 信号与 oracle 增益相关性极弱；**预测最优 (t_start,t_stop)** 在 macro 上失败。
- 本实验不先验声称这些新信号能路由成功，而是 **可复现地刻画** 其在层轴上的形态，为后续 **Skip-Gate / 任务条件化** 假设提供数据。
- 数学对应关系见下节；完整推导见对话中的「CR / JSD / eRank / ACI / FPR」论述及引文（DEQ、有效秩、Mind the Gap、LoopFormer 等）。

## 3. 本批实现的信号（逐层标量）

| 键名 | 定义要点 | 理论锚点 |
|------|-----------|----------|
| `residual_delta_l2` | 全序列 token 上 \(\mathbb{E}\|h_\ell-h_{\ell-1}\|_2\)（与 R29 `residual_write_norm` 的分子同阶，但未除以 \(\|h_{\ell-1}\|\)） | 残差流写入幅度 |
| `contraction_ratio` | \(\delta_\ell / (\delta_{\ell-1}+\varepsilon)\)，\(\ell\ge1\) | Banach / 局部收缩直觉 |
| `logit_lens_jsd_vel` | 末 token：\(\mathrm{JSD}(P_\ell\|P_{\ell-1})\)，\(P_\ell=\mathrm{softmax}(\mathrm{lm\_head}(\mathrm{norm}(h_\ell))))\) | 信息几何，较熵更保结构 |
| `logit_lens_jsd_curv` | \(\mathrm{jsd\_vel}(\ell)-\mathrm{jsd\_vel}(\ell-1)\)，\(\ell\ge2\) | 「减速/加速」二阶形态 |
| `erank` | 对 \(h_\ell\) 的子采样 token 矩阵做 SVD，对归一化奇异值谱算 \(\exp(-\sum p_i\log p_i)\) | Roy & Vetterli 有效秩 |
| `delta_erank` | \(\mathrm{erank}(\ell)-\mathrm{erank}(\ell-1)\) | 探索↔压缩相变 |
| `attn_consensus` | 末 query 位置上，各头注意力分布两两 JSD 的均值，再映射为 \(1-\mathrm{mean\_jsd}/\ln 2\in[0,1]\) | 多头共识/分歧 |
| `delta_norm_to_tstart` | \(\delta_\ell / (\delta_{t^{\mathrm{R30}}_{\mathrm{start}}}+\varepsilon)\)，按基准给定 R30 最优起点 | 相对 R30 块的写入标尺 |
| `attn_entropy` | 与 R29 相同：对注意力权重在 key 维熵再均值 | 与 ACI 互补 |
| `logit_top1_margin` | 末 token 分布：\(p_{(1)}-p_{(2)}\) | 判别置信（非熵标量） |

**说明：**

- `delta_norm_to_tstart` 依赖各基准 R30 最优 `t_start`（与 `r30_top_configs.txt` 一致），在 **后处理** 阶段由 `residual_delta_l2` 与 `t_start` 生成。
- `erank` 对长序列仅 **均匀下采样至多 64 个 token** 行参与 SVD，控制成本；这是计划内近似，非理论更改。
- 实现文件：`proposed_signals_probe.py`（独立探针，不修改 `r29/probe_forward.py`）。

## 4. 实验配置（与 R30 图脚本对齐）

- 模型：`R29_MODEL_PATH` 或默认 `/root/autodl-tmp/model_qwen`
- `attn_implementation="eager"`，与 R29 一致
- 基准：`ARC-C`, `TruthfulQA`, `CSQA`, `MMLU-HS-Math`
- R30 最优 `(t_start,t_stop)`：与 `plot_r30_optimal_signals_by_layer.py` 中 `R30_OPTIMAL` 相同
- `N_PER_BENCH=20`

## 5. 成功标准

- 四个基准各生成一张 `proposed_signals_vs_layer_<Bench>.png`（2×5 子图）。
- `proposed_signals_by_layer_plot_meta.json` 记录信号列表、样本数、耗时、模型路径。
- 脚本可单次运行完成（数据集可走本地缓存；网络失败不阻断）。

## 6. 后续（本计划文档范围外）

- 与 `oracle_gain` / 准确率做相关与分桶（需另开评测脚本）。
- Tuned Lens 版 JSD（需训练或加载 lens）。
- 完整序列 SVD 或随机投影估计 erank（精度—成本权衡）。

## 7. 运行方式

脚本在未设置 `HF_ENDPOINT` 时默认使用 `https://hf-mirror.com`，与本地已缓存的 `datasets` 兼容；若已配置代理或官方端点，可先 `export HF_ENDPOINT=https://huggingface.co`。

```bash
cd /root/autodl-tmp/loop_layer/experiments
# 可选：export HF_ENDPOINT=https://hf-mirror.com
python plot_proposed_signals_by_layer.py
```

---

*文档版本：与 `plot_proposed_signals_by_layer.py` / `proposed_signals_probe.py` 同步更新。*
