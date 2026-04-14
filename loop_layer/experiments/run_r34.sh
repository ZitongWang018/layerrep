#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# R34: 基于方向与交叉交互的 FFN-Attention 信号探针
#
# 用法：
#   bash run_r34.sh          # 等待 1 小时后自动运行
#   bash run_r34.sh --now    # 立即运行（跳过等待）
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

WAIT_SECONDS=1  # 1 小时

# --now 参数：跳过等待
if [[ "${1:-}" == "--now" ]]; then
    WAIT_SECONDS=0
fi

# ─── 环境 ────────────────────────────────────────────────────────────────────
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTHONPATH="/root/autodl-tmp/loop_layer:/root/autodl-tmp/loop_layer/ETD:/root/autodl-tmp/loop_layer/experiments:${PYTHONPATH:-}"

LOG_FILE="$SCRIPT_DIR/results/r34_run.log"
mkdir -p "$SCRIPT_DIR/results"

echo "============================================================"
echo "  R34 Cross-Memory Signal Probe"
echo "============================================================"
echo "  时间：$(date '+%Y-%m-%d %H:%M:%S')"
echo "  等待：${WAIT_SECONDS}s ($(( WAIT_SECONDS / 60 ))min) 后开始"
echo "  日志：$LOG_FILE"
echo "============================================================"

if [[ $WAIT_SECONDS -gt 0 ]]; then
    echo ""
    echo "当前有 GPU 进程运行中，等待 ${WAIT_SECONDS}s ..."
    echo "（可 Ctrl+C 取消，稍后用 bash run_r34.sh --now 立即运行）"
    echo ""

    # 每 60 秒打印一次倒计时
    REMAINING=$WAIT_SECONDS
    while [[ $REMAINING -gt 0 ]]; do
        MINS=$(( REMAINING / 60 ))
        SECS=$(( REMAINING % 60 ))
        printf "\r  剩余等待：%02d:%02d " $MINS $SECS
        SLEEP_STEP=60
        if [[ $REMAINING -lt 60 ]]; then
            SLEEP_STEP=$REMAINING
        fi
        sleep $SLEEP_STEP
        REMAINING=$(( REMAINING - SLEEP_STEP ))
    done
    echo ""
    echo "等待完毕，开始执行！$(date '+%Y-%m-%d %H:%M:%S')"
fi

echo ""
echo ">>> 检查 GPU 状态 ..."
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "(nvidia-smi 不可用)"

echo ""
echo ">>> 启动 R34 实验 ..."
echo "    输出同时写入终端和 $LOG_FILE"
echo ""

python3 "$SCRIPT_DIR/exp_r34_cross_memory_probe.py" 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "============================================================"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "  R34 实验完成！$(date '+%Y-%m-%d %H:%M:%S')"
    echo "  图表：$SCRIPT_DIR/figures/r34_cross_memory/"
    echo "  数据：$SCRIPT_DIR/results/r34_cross_memory_data_full.json"
    echo "  统计：$SCRIPT_DIR/results/r34_cross_memory_stats.json"
else
    echo "  R34 实验失败 (exit code=$EXIT_CODE)"
    echo "  日志：$LOG_FILE"
fi
echo "============================================================"

exit $EXIT_CODE
