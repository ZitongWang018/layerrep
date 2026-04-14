#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/loop_layer
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
RUN_LLAMA=0 RUN_QWEN=0 bash experiments/run_hard_mc_sweep_three_models.sh
