#!/usr/bin/env bash
# R29: Phase 0 then Phase 1 (single GPU — two concurrent model loads would OOM).
# Override: R29_N=24 R29_MODEL_PATH=... HF_DATASETS_OFFLINE=1
set -euo pipefail
export PYTHONPATH="/root/autodl-tmp/loop_layer/experiments:${PYTHONPATH:-}"
cd /root/autodl-tmp/loop_layer/experiments
N="${R29_N:-20}"
echo "=== R29 Phase 0 (n_per_bench=$N) ==="
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" python3 exp_round29_phase0.py --n-per-bench "$N" "$@"
echo "=== R29 Phase 1 (n_per_bench=$N) ==="
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" python3 exp_round29_phase1.py --n-per-bench "$N" "$@"
echo "Done. Results: experiments/results/round29_phase*.json"
