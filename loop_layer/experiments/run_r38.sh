#!/bin/bash
# R38: 全 Benchmark 信号引导 ETD 优化实验
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$SCRIPT_DIR/results"
FIGURES_DIR="$SCRIPT_DIR/figures/r38_signal_full"
LOG_FILE="$RESULTS_DIR/r38_run.log"

mkdir -p "$RESULTS_DIR"
mkdir -p "$FIGURES_DIR"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTHONPATH="$SCRIPT_DIR:$ROOT_DIR/ETD:$ROOT_DIR:${PYTHONPATH:-}"

echo "============================================================"
echo "R38: Full Benchmark Signal-Guided ETD Experiment"
echo "Started: $(date)"
echo "Results → $RESULTS_DIR/r38_signal_full_bench_results.json"
echo "Figures → $FIGURES_DIR/"
echo "Log     → $LOG_FILE"
echo "============================================================"

python "$SCRIPT_DIR/exp_r38_signal_full_bench.py" 2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo "R38 Finished: $(date)"
echo "============================================================"
