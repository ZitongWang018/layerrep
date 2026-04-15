#!/usr/bin/env bash
# run_r37.sh — R37: 信号引导的 ETD 循环层选择 (MMLU-HS-Math / GPQA-Diamond / AGIEval)
# 预估运行时间：~30-50 min (N=100 × 3 benchmarks × 7 conditions)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$SCRIPT_DIR/results/r37_run.log"

mkdir -p "$SCRIPT_DIR/results"
mkdir -p "$SCRIPT_DIR/figures/r37_signal_loop"

echo "============================================================"
echo " R37: Signal-Guided ETD Layer Selection"
echo " 开始时间: $(date)"
echo " 日志: $LOG_FILE"
echo "============================================================"

export HF_ENDPOINT="https://hf-mirror.com"
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="$ROOT:$SCRIPT_DIR:$ROOT/ETD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

cd "$SCRIPT_DIR"
python -u exp_r37_signal_guided_loop.py 2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo " R37 完成: $(date)"
echo " 结果: $SCRIPT_DIR/results/r37_signal_loop_results.json"
echo " 图表: $SCRIPT_DIR/figures/r37_signal_loop/"
echo "============================================================"
