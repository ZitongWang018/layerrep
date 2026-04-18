#!/bin/bash
# R39 信号猎杀实验运行脚本
# 用法：bash run_r39.sh [qwen3-8b|llama3-8b|gemma2-2b|all]
set -e

PRESET=${1:-qwen3-8b}
SCRIPT="$(dirname "$0")/exp_r39_signal_hunt.py"
LOGDIR="$(dirname "$0")/results"
mkdir -p "$LOGDIR"

run_one() {
    local preset=$1
    local logfile="$LOGDIR/r39_${preset}_run.log"
    echo "============================================"
    echo "R39 preset=$preset → $logfile"
    echo "============================================"
    python3 "$SCRIPT" --preset "$preset" 2>&1 | tee "$logfile"
    echo "Done: $preset"
}

if [ "$PRESET" = "all" ]; then
    run_one qwen3-8b
    run_one llama3-8b
    run_one gemma2-2b
else
    run_one "$PRESET"
fi
