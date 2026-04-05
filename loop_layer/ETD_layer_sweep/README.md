# ETD 层区间网格搜索（BoolQ + ARC）

## 做什么

- 对思考块 **T** 的 **起点 `t_start`**（[5,15]）与 **终点 `t_end`**（[20,28]）做**全组合**（`t_start ≤ t_end`，共 **99** 格）。
- 每一格：**E** = 层 `0..t_start-1`，**T** = `t_start..t_end`，**D** = `t_end+1..35`。
- 同一格内评测 **baseline**、**k=2**、**k=3**；**BoolQ 与 ARC 使用完全相同的样本子集**（各取前 `BOOLQ_LIMIT` / `ARC_LIMIT` 条，顺序固定）。
- 不做角距离采集。

## 如何运行（你来执行）

```bash
cd /root/autodl-tmp/loop_layer/ETD_layer_sweep
chmod +x run_sweep.sh
./run_sweep.sh
```

生成：

- `artifacts/sweep_results.csv`：逐格准确率、正确数、耗时。
- `figures/*.png`：英文标题的热力图与相对 baseline 的差分图。
- `EXPERIMENT_REPORT.md`：汇总表 + 中文摘要 + 插图。

## 环境变量（可选）

| 变量 | 含义 |
|------|------|
| `MODEL` | 模型目录，默认 `/root/autodl-tmp/model_qwen` |
| `BOOLQ_LIMIT` | BoolQ 条数（validation 前缀顺序），默认 `500` |
| `ARC_LIMIT` | ARC-Challenge 条数（test 前缀顺序），默认 `500` |
| `OUT_CSV` | 输出 CSV 路径，默认 `artifacts/sweep_results.csv` |
| `MAX_CELLS` | 仅跑前 N 格（调试），不设则跑满 99 格 |
| `RESUME=1` | 断点续跑：跳过 CSV 已有 `(t_start,t_end)` |
| `SKIP_SWEEP=1` | 只跑分析与出图（需已有 CSV） |

## 仅分析与出图

```bash
SKIP_SWEEP=1 ./run_sweep.sh
```

或：

```bash
python3 analyze_plots.py --csv artifacts/sweep_results.csv --report EXPERIMENT_REPORT.md
```

## 依赖

见 `requirements.txt`（含 `pandas`、`matplotlib`）。
