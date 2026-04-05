# Weights & Biases（W&B）— 个人账号 + Qwen3-8B 评测

## 个人账号（默认情况）

1. **安装**（在 lm-evaluation-harness 目录）：

   ```bash
   cd /root/autodl-tmp/lm-evaluation-harness
   pip install -e ".[hf,math]"
   pip install "wandb>=0.13.6"
   ```

2. **登录一次**：

   ```bash
   wandb login
   ```

   在 <https://wandb.ai/authorize> 复制 API Key 粘贴即可。

3. **不要设置** `WANDB_ENTITY`。个人空间下 run 会自动归到你的用户名；脚本里只有当你加入**团队/组织项目**时才需要 `export WANDB_ENTITY=...`。

4. **跑评测**（默认工程名为 `lm-eval-qwen3-8b`，便于在个人主页里集中查看）：

   ```bash
   cd /root/autodl-tmp/eval_tools
   ./run_eval_qwen.sh
   ```

   打开浏览器：<https://wandb.ai/> → 你的用户名 → Project **`lm-eval-qwen3-8b`**，即可看到各次 run、指标与分组。

5. **数据集下载失败（SSL 握手超时 / ConnectTimeout）**  
   评测会从 Hugging Face 拉取各 benchmark 数据（不是模型）。`run_eval_qwen.sh` 已自动 `source hf_download_env.sh`：默认使用 **`HF_ENDPOINT=https://hf-mirror.com`**，并加大 **`HF_HUB_DOWNLOAD_TIMEOUT`**。  
   若你**在海外**或镜像异常，可：`export USE_HF_MIRROR=0` 再运行。  
   若机器配置了**有问题的 HTTP 代理**导致握手超时，可临时：`unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY`，再试。

6. **HumanEval 报错 `HF_ALLOW_CODE_EVAL`**  
   跑 `humaneval` 时，HuggingFace `code_eval` 会要求设置环境变量。`run_eval_qwen.sh` 已通过 `hf_download_env.sh` 默认 `export HF_ALLOW_CODE_EVAL=1`；若你自行调用 `lm-eval`，请先执行 `export HF_ALLOW_CODE_EVAL=1`（表示已阅读风险提示并同意在本地执行模型生成的代码进行单测）。

5. **多轮实验对比**：多次运行时用**相同** `WANDB_GROUP`，在 W&B 里用 Group 视图对比：

   ```bash
   export WANDB_GROUP="exp-2025-04"
   export RUN_NAME="run-with-lr-sweep-1"
   ./run_eval_qwen.sh
   ```

---

## 你会看到什么

- **终端**：每个 task 的进度与最终汇总表。
- **W&B 网页**：默认 **Project** = `lm-eval-qwen3-8b`，**Run name** 默认 `qwen3-8b-eval-时间戳`。Config 里会带 `model_name=Qwen3-8B` 便于筛选。
- **说明**：lm-eval 多在**每个任务结束后**把该任务指标写入 W&B；超长单任务期间以终端日志为主。

---

## 可选：团队项目

若 run 要进**组织**而非个人空间，再设置：

```bash
export WANDB_ENTITY=你的团队名
```

---

## 离线（不上传）

```bash
export WANDB_MODE=offline
./run_eval_qwen.sh
```

之后可用 `wandb sync` 同步到云端。
