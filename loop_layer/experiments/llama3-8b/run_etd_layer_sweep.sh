#!/usr/bin/env bash
# Runs only the Llama3-8B sweep (same as experiments/run_multimodel_etd_layer_sweep.sh with RUN_GEMMA=0).
set -euo pipefail
cd /root/autodl-tmp/loop_layer
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
RUN_LLAMA=1 RUN_GEMMA=0 bash experiments/run_multimodel_etd_layer_sweep.sh
