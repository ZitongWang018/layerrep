#!/usr/bin/env bash
# R40：BBH（6 子任务×50）+ GSM8K（5-shot + 贪婪生成），三模型 × 四条件
# 环境：HF 镜像（HF_ENDPOINT）；默认允许在线拉取 BBH/GSM8K（HF_DATASETS_OFFLINE=0）
# 使用镜像时**不要**走 HTTP(S) 代理：TLS 经代理常握手超时；此处统一 unset。
set -euo pipefail

ROOT="/root/autodl-tmp/loop_layer"
cd "$ROOT"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
# 默认 0：首次需经镜像下载 SaylorTwift/bbh；已全量缓存后可改为 1
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export PYTHONPATH="/root/autodl-tmp/lm-evaluation-harness:${PYTHONPATH:-}"

# 避免 libgomp 对空/非法 OMP_NUM_THREADS 报警
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

mkdir -p experiments/results
LOG="experiments/results/r40_bbh_gsm8k_run.log"

{
  echo "========================================"
  echo "R40 BBH + GSM8K + ETD  开始 $(date -Is)"
  echo "日志: $ROOT/$LOG"
  echo "粗估：单卡 8B 级每模型约 1–6h，三模型顺序约 3–18h（进度条内 ETA_total 为动态校准）"
  echo "HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE}（=1 时需已缓存 SaylorTwift/bbh 与 openai/gsm8k）"
  echo "========================================"

  for PRESET in qwen3-8b llama3-8b gemma2-2b; do
    echo ""
    echo "---------- preset=${PRESET} $(date -Is) ----------"
    python3 experiments/exp_r40_bbh_gsm8k_etd.py \
      --preset "${PRESET}" \
      --bbh-limit 50 \
      --gsm-limit 50
  done

  echo ""
  echo "========================================"
  echo "R40 全部完成 $(date -Is)"
  echo "结果: experiments/results/r40_bbh_gsm8k_qwen3_8b.json"
  echo "      experiments/results/r40_bbh_gsm8k_llama3_8b.json"
  echo "      experiments/results/r40_bbh_gsm8k_gemma2_2b.json"
  echo "========================================"
} 2>&1 | tee -a "$LOG"
