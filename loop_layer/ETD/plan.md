# ETD 复现计划（Qwen3-8B）

1. 从 BoolQ / ARC 训练集各采样少量文本，前向取 `output_hidden_states`，对**最后一个非 pad token** 计算相邻层平均角距离序列。
2. 对距离序列做 Kneedle（正向 + 反向）得到 `N_E, N_T, N_D`，写入 `artifacts/*_layers.json`。
3. 用 ETD 前向 `E → T^k → D`（`k∈{2,3}`）在验证集上做多选题 log 似然；与标准一次前向（baseline）对比。
4. BoolQ 使用 **validation**（test 无 gold）；ARC 使用 **test**。

确认后可运行 `./run_all.sh`（或分步运行 `collect_layers.py` / `evaluate_etd.py`）。
