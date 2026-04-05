# Plan: T-block sweep

1. 固定 BoolQ / ARC 样本列表（与 `t_start,t_end` 无关）。
2. 枚举 `t_start∈[5,15]`、`t_end∈[20,28]`、`t_start≤t_end`，共 99 组。
3. 每组：`n_e=t_start`，`n_t=t_end-t_start+1`，`n_d=36-n_e-n_t`；依次跑 baseline、k=2、k=3 于两数据集。
4. 写 CSV；`analyze_plots.py` 出英文图与 `EXPERIMENT_REPORT.md`。

由用户在目标机器上执行 `run_sweep.sh`。
