#!/usr/bin/env bash
# R38-Multimodel: Llama3-8B + Gemma2-2B 全 8 benchmark 信号验证（与 R38 条件对齐）
set -euo pipefail
ROOT="/root/autodl-tmp/loop_layer"
cd "$ROOT/experiments"
export PYTHONPATH="$ROOT/experiments:$ROOT/ETD:$ROOT:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

LOG_DIR="$ROOT/experiments/results"
mkdir -p "$LOG_DIR"

echo "=== R38 Multimodel: Llama3-8B ==="
python exp_r38_multimodel_signal.py --preset llama3-8b 2>&1 | tee "$LOG_DIR/r38_multimodel_llama3.log"

echo ""
echo "=== R38 Multimodel: Gemma2-2B ==="
python exp_r38_multimodel_signal.py --preset gemma2-2b 2>&1 | tee "$LOG_DIR/r38_multimodel_gemma2.log"

echo ""
echo "Done. JSON:"
echo "  $LOG_DIR/r38_multimodel_llama3_signal.json"
echo "  $LOG_DIR/r38_multimodel_gemma2_signal.json"
