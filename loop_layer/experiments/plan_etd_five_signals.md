# 实验计划：ETD 五类机制信号（与理论表一一对应）

## 1. 目标

针对 R30 各基准最优 T-block \([t_{\mathrm{start}}, t_{\mathrm{stop}})\)（`t_stop` 为开区间），每基准 **N=20** 样本做单次 eager-attention 前向，提取下表 **5 个信号**，并生成与 `figures/r30_optimal_by_layer/` **相同画布尺寸**（2×5，仅用前 5 个子图）的逐层/块标量图。

| 信号 | 图中表现 | 数学定义（本实现） |
|------|-----------|-------------------|
| **CR（块 + 局部）** | 子图 1：逐层 **contraction_ratio** \(\delta_\ell/\delta_{\ell-1}\) 曲线；灰线 ratio=1；橙点划线 **mean CR_block**（块内几何平均，与曲线纵轴含义不同，仅作对照） | **CR_block** \(=\bigl(\delta_{t_{\mathrm{stop}}-1}/\delta_{t_{\mathrm{start}}}\bigr)^{1/(t_{\mathrm{stop}}-t_{\mathrm{start}}-1)}\)；\(\delta_\ell=\mathbb{E}\|h_\ell-h_{\ell-1}\|_2\) |
| **JSD Velocity** | 逐层曲线（末 token 相邻层 logit-lens 分布 JSD） | \(\mathrm{JSD}(P_\ell\|P_{\ell-1})\) |
| **ΔeRank** | 逐层曲线 | \(\mathrm{erank}(H_\ell)-\mathrm{erank}(H_{\ell-1})\)（erank 见 Roy–Vetterli，token 子采样后 SVD） |
| **ACI** | 逐层曲线 | 末 query 位置上多头注意力分布两两 JSD 的均值映射到 \([0,1]\) 共识指数 |
| **FPR（轨迹 + 标量）** | 子图 5：逐层 **delta_norm_to_tstart** \(=\delta_\ell/\delta_{t_{\mathrm{start}}}\)；紫竖线 **layer = t_stop-1**；均值曲线在该层的红点对应 **mean FPR_simple** | **FPR_simple** \(=\delta_{t_{\mathrm{stop}}-1}/\delta_{t_{\mathrm{start}}}\)（即轨迹在出口层的纵坐标） |

**说明：** 块标量 **CR_block**、**FPR_simple** 仍写入 `etd_five_signals_by_layer_plot_meta.json`；layer–x 轴上改为画**逐层动力学**，避免整块常数误读为「信号无变化」。

## 2. 输出路径

- 图：`experiments/figures/etd_five_signals_by_layer/etd_five_signals_vs_layer_<Bench>.png`
- 元数据：`experiments/results/etd_five_signals_by_layer_plot_meta.json`（含每样本 `cr_block`、`fpr_simple` 列表，便于后续相关分析）

## 3. 实现文件

- `etd_five_signals_metrics.py`：`compute_cr_block`、`compute_fpr_simple`、`residual_delta_series`
- `proposed_signals_probe.py`：复用 `collect_proposed_signals`（δ、JSD、erank、ACI）
- `plot_etd_five_signals_by_layer.py`：加载模型与数据、采集、计算块标量、绑图

## 4. 配置（与 R30 图脚本一致）

- 基准与 \((t_{\mathrm{start}},t_{\mathrm{stop}})\)：同 `plot_r30_optimal_signals_by_layer.py` / `r30_top_configs.txt`
- 模型：`R29_MODEL_PATH` 或 `/root/autodl-tmp/model_qwen`，`attn_implementation=eager`
- 未设置 `HF_ENDPOINT` 时默认 `https://hf-mirror.com`（在绘图脚本内设置）
- 模型设备：默认 `ETD_DEVICE_MAP=auto`；探针内已将 `input_ids` 对齐到 embedding 设备、`h_prev` 对齐到 `h`，logit 概率在 CPU 上算 JSD。独占 GPU 且显存足够时可设 `ETD_DEVICE_MAP=cuda0` 整卡加载。

## 5. 运行

```bash
cd /root/autodl-tmp/loop_layer/experiments
python plot_etd_five_signals_by_layer.py
# 快速冒烟（每基准 2 条）：
# ETD_FIVE_N_PER_BENCH=2 python plot_etd_five_signals_by_layer.py
```

---

*与 `plot_etd_five_signals_by_layer.py` 保持同步。*
