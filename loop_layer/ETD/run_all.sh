#!/usr/bin/env bash
# 端到端：角距离采集 -> BoolQ / ARC 评测（固定 k=2,3，可选 baseline）
set -euo pipefail
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODEL="${MODEL:-/root/autodl-tmp/model_qwen}"
MAX_SAMPLES="${MAX_SAMPLES:-128}"
EVAL_LIMIT="${EVAL_LIMIT:-0}"

python3 collect_layers.py --dataset boolq --model-path "$MODEL" --max-samples "$MAX_SAMPLES" --out artifacts/boolq_layers.json
python3 collect_layers.py --dataset arc --model-path "$MODEL" --max-samples "$MAX_SAMPLES" --out artifacts/arc_layers.json

# BoolQ：官方 test 无标签，使用 validation
python3 evaluate_etd.py --dataset boolq --model-path "$MODEL" --layer-json artifacts/boolq_layers.json \
  --k 2 3 --baseline --limit "$EVAL_LIMIT" --out artifacts/eval_boolq_validation.json

python3 evaluate_etd.py --dataset arc --model-path "$MODEL" --layer-json artifacts/arc_layers.json \
  --k 2 3 --baseline --split test --limit "$EVAL_LIMIT" --out artifacts/eval_arc_test.json

echo "Done. See artifacts/"
