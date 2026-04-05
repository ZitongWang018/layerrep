# ETD / BoolQ

- `aps/super_glue` 的 BoolQ **test** 划分 `label=-1`，无法算准确率；评测请用 **validation**。
- ETD 前向在 `k=1` 且 `N_E+N_T+N_D=L` 时与标准 `Qwen3Model` 前向数值一致（已用 max abs diff 校验）。
