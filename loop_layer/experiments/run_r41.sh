#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
# Allow dataset download for BBH unless user set offline
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export PYTHONUNBUFFERED=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true

LOG="${1:-experiments/results/r41_run.log}"
mkdir -p "$(dirname "$LOG")"

echo "R41 log: $(readlink -f "$LOG" 2>/dev/null || echo "$LOG")"
# Pass extra args to python, e.g. --n-samples 25 --bbh-limit 12
python3 experiments/exp_r41_reflux_jac_etd.py "$@" 2>&1 | tee "$LOG"
