#!/usr/bin/env bash
# Runs only the Gemma2-2B sweep (same as experiments/run_multimodel_etd_layer_sweep.sh with RUN_LLAMA=0).
set -euo pipefail
cd /root/autodl-tmp/loop_layer
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
RUN_LLAMA=0 RUN_GEMMA=1 bash experiments/run_multimodel_etd_layer_sweep.sh
