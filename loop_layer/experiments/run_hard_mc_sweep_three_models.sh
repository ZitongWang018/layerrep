#!/usr/bin/env bash
# ETD layer sweep with hard multiple-choice benchmarks (lm-eval–aligned):
#   GPQA-Diamond, AGIEval-Gaokao-MathQA, LogiQA
# Requires: huggingface-cli login + accept GPQA terms for GPQA-Diamond (see hard_mc_benchmark_loaders.py).
# Network: sources experiments/hf_hub_network_env.sh — clears proxy if hf-mirror is reachable, else sources /etc/network_turbo.
# First-time dataset download: HF_DATASETS_OFFLINE=0 HF_HUB_OFFLINE=0 bash experiments/run_hard_mc_sweep_three_models.sh
set -euo pipefail
cd /root/autodl-tmp/loop_layer
# shellcheck source=hf_hub_network_env.sh
source experiments/hf_hub_network_env.sh
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

RUN_LLAMA="${RUN_LLAMA:-1}"
RUN_GEMMA="${RUN_GEMMA:-1}"
RUN_QWEN="${RUN_QWEN:-1}"

ARGS=(--bench-profile hard_mc --resume)

if [[ "$RUN_LLAMA" == "1" ]]; then
  python experiments/multimodel_etd_layer_sweep.py --preset llama3-8b "${ARGS[@]}"
fi
if [[ "$RUN_GEMMA" == "1" ]]; then
  python experiments/multimodel_etd_layer_sweep.py --preset gemma2-2b "${ARGS[@]}"
fi
if [[ "$RUN_QWEN" == "1" ]]; then
  python experiments/multimodel_etd_layer_sweep.py --preset qwen3-8b "${ARGS[@]}"
fi

echo "Outputs under experiments/<model>/results/hard_mc/ and figures/hard_mc/"
