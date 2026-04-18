#!/bin/bash
# R39C 三模型完整运行脚本
set -e
cd /root/autodl-tmp/loop_layer
RESULTS="experiments/results"
LOG="$RESULTS"

run_model() {
    local preset=$1
    local logfile="$LOG/r39c_${preset//-/_}_run.log"
    echo "====================================================="
    echo "[R39C] 开始运行 $preset  $(date)"
    echo "====================================================="
    python3 experiments/exp_r39c_final.py --preset "$preset" > "$logfile" 2>&1
    echo "[R39C] $preset 完成  $(date)"
    echo "最后10行日志："
    tail -10 "$logfile"
    echo ""
}

# Qwen3-8B 已在运行（PID 17849），等待完成
if pgrep -f "r39c_final.*qwen3" > /dev/null; then
    echo "[INFO] Qwen3-8B R39C 已在运行，等待完成..."
    wait $(pgrep -f "r39c_final.*qwen3") 2>/dev/null || true
    echo "[INFO] Qwen3-8B 已完成"
fi

# Llama3-8B
run_model "llama3-8b"

# Gemma2-2B
run_model "gemma2-2b"

echo ""
echo "====================================================="
echo "[R39C] 全部三个模型运行完成！"
echo "====================================================="
ls -la "$RESULTS/r39c_final_*.json"
