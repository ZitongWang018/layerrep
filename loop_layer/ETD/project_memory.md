# ETD / BoolQ

- `aps/super_glue` 的 BoolQ **test** 划分 `label=-1`，无法算准确率；评测请用 **validation**。
- ETD 前向在 `k=1` 且 `N_E+N_T+N_D=L` 时与标准 `Qwen3Model` 前向数值一致（已用 max abs diff 校验）。
- `experiments/hf_hub_network_env.sh`：先 `unset` 代理后探测 `hf-mirror.com`；可达则用镜像并**保持无代理**；不可达则 `source /etc/network_turbo`，再试镜像，仍失败则 `HF_ENDPOINT=https://huggingface.co`。扫参入口脚本已 `source` 该文件。
- LogiQA：`datasets>=3` 不再加载带 `logiqa.py` 的 `EleutherAI/logiqa`；`hard_mc_benchmark_loaders.load_logiqa` 在 `RuntimeError` 时回退到 **parquet** 数据集 `fireworks-ai/logiqa`（选项去掉 `A.` 前缀后与 lm-eval 模板一致）。
