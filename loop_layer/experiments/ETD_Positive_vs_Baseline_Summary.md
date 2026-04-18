# ETD 优于 Baseline 的实验结果摘录（R37–R41）

本文**只展开**在已保存的 JSON 结果中，**某 ETD 选窗/信号方法的准确率严格高于同设置 Baseline** 的情形；若未超过 Baseline 或整体为负向，则**合并为末节极短概述**，不展开表格细节。

**约定**：$\Delta = \text{accuracy}_{\text{method}} - \text{accuracy}_{\text{baseline}}$。插图路径相对于本文件所在目录 `experiments/`。

---

## 1. Qwen3-8B：R39C 多选题主评测（`r39c_final_qwen3.json`）

评测规模：各 benchmark **N=100**（TruthfulQA 为 **N=50**）。方法含义：`neg_cos_am_calib`（负 $\cos(a,m)$ 标定窗）、`emp_logit_fixed`（标定集 logit 增益搜窗）、`neg_cos_am_ps_nt`（逐样本负 $\cos(a,m)$ + 架构内 $n_t$ 搜索）。`sweep_best` 为 R30 扫参最优固定窗（上界参照，**非**无参方法）。

### 1.1 相对 Baseline 有增益的方法（逐项）

下表**仅保留**至少一种信号法满足 $\Delta>0$ 的 benchmark；数值保留两位小数。

| Benchmark | Baseline | 最佳信号法（若有） | Acc | $\Delta$ | 备注 |
|-----------|----------|-------------------|-----|-----------|------|
| BoolQ | 0.82 | `emp_logit_fixed` | **0.90** | **+0.08** | 同时优于 `sweep_best`（0.87） |
| BoolQ | 0.82 | `neg_cos_am_calib` | 0.86 | +0.04 | |
| BoolQ | 0.82 | `neg_cos_am_ps_nt` | 0.85 | +0.03 | |
| TruthfulQA | 0.32 | `emp_logit_fixed` / `neg_cos_am_ps_nt` | **0.34** | **+0.02** | `neg_cos_am_calib` 与 baseline 持平 |
| CSQA | 0.64 | `neg_cos_am_calib` | **0.65** | **+0.01** | `emp_logit_fixed` 与 baseline 持平 |
| MMLU-HS-Math | 0.40 | `neg_cos_am_calib` | **0.41** | **+0.01** | `sweep_best` 为 +0.03；`emp_logit_fixed` 低于 baseline |
| AGIEval-Gaokao-MathQA | 0.52 | `emp_logit_fixed` / `neg_cos_am_ps_nt` | **0.57** | **+0.05** | `neg_cos_am_calib` 与 `sweep_best` 同为 0.54（+0.02） |
| LogiQA | 0.36 | `emp_logit_fixed` | **0.42** | **+0.06** | |
| LogiQA | 0.36 | `neg_cos_am_ps_nt` | 0.39 | +0.03 | |
| LogiQA | 0.36 | `neg_cos_am_calib` | 0.38 | +0.02 | |

**ARC-Challenge**：`neg_cos_am_calib` 与 `emp_logit_fixed` 与 Baseline 同为 0.56；仅 `sweep_best` 为 0.58（+0.02）。**GPQA-Diamond**：三种信号法均 **低于** Baseline（0.38），不展开。

### 1.2 图示（Qwen3 / R39C）

![Qwen3 R39C 各 Benchmark 准确率条形图](figures/r39c_final_qwen3/01_accuracy_bars.png)

![Qwen3 相对 Baseline 的 $\Delta$acc 热力图](figures/r39c_final_qwen3/04_heatmap.png)

---

## 2. Qwen3-8B：R38 全基准信号法（`r38_final_results.json` 摘要）

**明显优于 Baseline 的亮点**：MMLU-HS-Math 上 `calib_onset8` 达到 **0.45**，相对 Baseline **0.40** 为 **+0.05**，且高于该任务上的 `sweep_best`（0.43）。CSQA 上 `persample_cos8` **0.68**、`calib_global8` **0.67** 均高于 Baseline **0.64**。

其余 benchmark 上各信号法多为持平或低于 `sweep_best`；整体不另占篇幅。

### 2.1 图示（R38）

![R38 全 Benchmark 条形图](figures/r38_signal_full/all_benchmark_bars.png)

![R38 $\Delta$acc 热力图](figures/r38_signal_full/final_heatmap.png)

---

## 3. Qwen3-8B：R40 BBH（6 子任务 × N=50）与 GSM8K（N=50）

### 3.1 BBH：子任务层面 $\Delta>0$ 汇总

| 子任务（简名） | Baseline | 方法 | Acc | $\Delta$ |
|----------------|------------|------|-----|-----------|
| boolean_expressions | 0.90 | `neg_cos_am_calib` | 0.92 | +0.02 |
| boolean_expressions | 0.90 | `emp_logit_fixed` | **0.94** | **+0.04** |
| causal_judgement | 0.58 | `emp_logit_fixed` | **0.62** | **+0.04** |
| disambiguation_qa | 0.48 | `neg_cos_am_calib` | 0.50 | +0.02 |
| disambiguation_qa | 0.48 | `emp_logit_fixed` | **0.58** | **+0.10** |
| logical_deduction_three_objects | 0.82 | `neg_cos_am_calib` | 0.86 | +0.04 |
| logical_deduction_three_objects | 0.82 | `emp_logit_fixed` | 0.86 | +0.04 |
| object_counting | 0.48 | `neg_cos_am_calib` | **0.52** | **+0.04** |

`date_understanding` 上三种 ETD 法均未超过 Baseline（0.74）；`causal_judgement` / `object_counting` 上 `neg_cos_am_ps_nt` 未过 Baseline——**略**。

### 3.2 GSM8K（exact_match）

| 方法 | exact_match | $\Delta$ vs Baseline |
|------|-------------|----------------------|
| Baseline | 0.90 | — |
| **`neg_cos_am_calib`** | **0.94** | **+0.04** |
| `emp_logit_fixed` / `neg_cos_am_ps_nt` | 0.90 | 0 |

### 3.3 图示（R40 / Qwen3）

![GSM8K exact_match 对比](figures/r40_bbh_gsm8k/r40_gsm8k_exact_match.png)

![BBH 各子任务准确率热力图（含 baseline 与三法）](figures/r40_bbh_gsm8k/r40_bbh_accuracy_heatmaps.png)

![BBH 按模型分组的 Accuracy](figures/r40_bbh_gsm8k/r40_bbh_accuracy_by_model.png)

---

## 4. Qwen3-8B：R41（MC + BBH，小样本）

R41 中 **BBH 六子任务宏平均**（`r41_qwen3_reflux_jac.json`，各子任务 N=12）：

| 方法 | BBH 宏平均 acc | $\Delta$ vs Baseline |
|------|----------------|----------------------|
| Baseline | 0.6528 | — |
| **`neg_cos_am_calib`** | **0.7361** | **+0.0833** |
| **`emp_logit_fixed`** | **0.7083** | **+0.0556** |
| **`neg_cos_am_prop_attn`** | **0.6944** | **+0.0417** |
| `reflux_rho_gate` | 0.6667 | +0.0139 |

MC 三任务（BoolQ / ARC-C / GPQA，N=25）上多数法与 Baseline 持平或方差大；其中 **ARC-C** 上 `emp_logit_fixed` **0.56** 相对 Baseline **0.52** 为 **+0.04**，`neg_cos_am_calib` 低于 Baseline（不展开）。

![R41 各条件准确率条形图](figures/r41_qwen3/r41_accuracy_comparison.png)

---

## 5. Llama3-8B / Gemma2-2B（效果不佳：极短概述）

**Llama3-8B（`r39c_final_llama3.json`）**：多数 benchmark 上信号法 **未稳定超过** `sweep_best`；相对 Baseline 的 **单独正增益** 仅零星出现（例如 GPQA 上 `emp_logit_fixed` 0.39 vs baseline 0.29 等），整体以扫参窗为参照更稳妥。

**Gemma2-2B（`r39c_final_gemma2.json`）**：`neg_cos_am_calib` 在 BoolQ 等任务上 **明显低于** Baseline；`emp_logit_fixed` 多数为持平。**不展开**。

---

## 6. 早期轮次（R32–R36）与「未过 Baseline」的合并说明

R32–R33 在 **per-sample Spearman** 上未得到稳定显著相关；R34 主要贡献为 **几何信号**（如 $\cos(a,m)$）与 T-block 的定性对齐。R35 交换子 **范数** 与 T-block 不对齐；R36 的 `directional_advantage` **未**形成可靠的「过 Baseline 选层」产品结论。上述内容**不另附表**；若需完整推导与图表，见 `ETD_R32_to_R41_Comprehensive_Report.md`。

---

## 数据与复现

- 主要来源：`experiments/results/r39c_final_qwen3.json`、`r38_final_results.json`、`r40_bbh_gsm8k_qwen3_8b.json`、`r41_qwen3_reflux_jac.json`。
- PDF 生成：`bash experiments/export_positive_vs_baseline_pdf.sh`（与主报告共用 `pandoc` + `xelatex` + Noto CJK）。
