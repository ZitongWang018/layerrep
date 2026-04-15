#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# R35: Attention-FFN 非对易交换子探针
#
# 用法：
#   bash run_r35.sh          # 立即运行
#   bash run_r35.sh --wait   # 等待 1 小时后运行（当前有进程时使用）
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

WAIT_SECONDS=0

if [[ "${1:-}" == "--wait" ]]; then
    WAIT_SECONDS=3600
fi

# ─── 环境 ────────────────────────────────────────────────────────────────────
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTHONPATH="/root/autodl-tmp/loop_layer:/root/autodl-tmp/loop_layer/ETD:/root/autodl-tmp/loop_layer/experiments:${PYTHONPATH:-}"

LOG_FILE="$SCRIPT_DIR/results/r35_run.log"
mkdir -p "$SCRIPT_DIR/results"

echo "============================================================"
echo "  R35 Attention-FFN Commutator Probe"
echo "============================================================"
echo "  时间：$(date '+%Y-%m-%d %H:%M:%S')"
if [[ $WAIT_SECONDS -gt 0 ]]; then
    echo "  等待：${WAIT_SECONDS}s ($(( WAIT_SECONDS / 60 ))min) 后开始"
fi
echo "  日志：$LOG_FILE"
echo "============================================================"

if [[ $WAIT_SECONDS -gt 0 ]]; then
    echo ""
    echo "当前有 GPU 进程运行中，等待 ${WAIT_SECONDS}s ..."
    echo "（可 Ctrl+C 取消，稍后用 bash run_r35.sh 立即运行）"
    echo ""

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
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null || echo "(nvidia-smi 不可用)"

echo ""
echo ">>> 启动 R35 实验（精确 Attention-FFN 交换子）..."
echo "    输出同时写入终端和 $LOG_FILE"
echo "    预计耗时：~100-130s（含每层额外 MLP + Attention 重跑）"
echo ""

python3 "$SCRIPT_DIR/exp_r35_commutator_probe.py" 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "============================================================"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "  R35 实验完成！$(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "  图表目录：$SCRIPT_DIR/figures/r35_commutator/"
    echo "  完整数据：$SCRIPT_DIR/results/r35_commutator_data_full.json"
    echo "  统计摘要：$SCRIPT_DIR/results/r35_commutator_stats.json"
    echo ""
    echo "  关键图表："
    echo "    *_r35_commutator_vs_layer.png   -- 每 benchmark 的 10 个交换子信号"
    echo "    *_r35_vs_r34_comparison.png     -- R35 vs R34 对比（每 benchmark）"
    echo "    r35_all_overlay.png             -- 全 benchmark 叠图"
    echo "    r35_vs_r34_comparison.png       -- 全 bench 对比（H1/H2 检验）"
    echo "    r35_scatter_commutator_vs_delta.png  -- Phase 2 散点图（H3 检验）"
else
    echo "  R35 实验失败 (exit code=$EXIT_CODE)"
    echo "  日志：$LOG_FILE"
fi
echo "============================================================"

exit $EXIT_CODE
