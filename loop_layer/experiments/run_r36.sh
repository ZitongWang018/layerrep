#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# R36: 方向特异性传播增益实验
#
# 用法：
#   bash run_r36.sh          # 立即运行
#   bash run_r36.sh --wait   # 等待 1 小时后运行（当前有进程时使用）
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

LOG_FILE="$SCRIPT_DIR/results/r36_run.log"
mkdir -p "$SCRIPT_DIR/results"

echo "============================================================"
echo "  R36 方向特异性传播增益实验"
echo "============================================================"
echo "  时间：$(date '+%Y-%m-%d %H:%M:%S')"
if [[ $WAIT_SECONDS -gt 0 ]]; then
    echo "  等待：${WAIT_SECONDS}s ($(( WAIT_SECONDS / 60 ))min) 后开始"
fi
echo "  日志：$LOG_FILE"
echo ""
echo "  关键新信号："
echo "    prop_sens           JSD(h_l + ε·Ĉ_l) vs baseline"
echo "    rand_sens           JSD(h_l + ε·r̂)   vs baseline (随机基线)"
echo "    directional_advantage  prop/rand — 核心 H2 假设"
echo "    etd_effective       cos(C_l,Δh_l) × DA"
echo "    comm_persist        cos(C_l, C_{l+1})"
echo ""
echo "  Probe 层：3,6,9,12,15,18,21,24,27,30,33 (11 层 × 2 forward 每 sample)"
echo "  预计耗时：~12-18min（N=100：22 额外 forwards/sample × 7 bench；视 GPU 而定）"
echo "============================================================"

if [[ $WAIT_SECONDS -gt 0 ]]; then
    echo ""
    echo "当前有 GPU 进程运行中，等待 ${WAIT_SECONDS}s ..."
    echo "（可 Ctrl+C 取消，稍后用 bash run_r36.sh 立即运行）"
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
echo ">>> 启动 R36 实验（方向特异性传播增益）..."
echo "    输出同时写入终端和 $LOG_FILE"
echo ""

python3 "$SCRIPT_DIR/exp_r36_propagation_etd.py" 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "============================================================"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "  R36 实验完成！$(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "  图表目录：$SCRIPT_DIR/figures/r36_propagation/"
    echo "  完整数据：$SCRIPT_DIR/results/r36_propagation_data_full.json"
    echo "  统计摘要：$SCRIPT_DIR/results/r36_propagation_stats.json"
    echo ""
    echo "  关键图表："
    echo "    {bench}_r36_prop_vs_layer.png       — 每 benchmark 6 个传播信号（H1-H6 对照）"
    echo "    r36_individual_samples_{bench}.png  — 每 benchmark 个体样本 DA 曲线（H4）"
    echo "    r36_all_overlay.png                 — 全 benchmark 叠图"
    echo "    r36_sample_variance.png             — var(DA) 按层（H4）"
    echo "    r36_scatter_da_vs_delta.png         — DA vs ETD Δacc 散点（H3）"
    echo "    r36_late_vs_tblock.png              — T-block vs 后期层对比（H1/H5）"
else
    echo "  R36 实验失败 (exit code=$EXIT_CODE)"
    echo "  日志：$LOG_FILE"
fi
echo "============================================================"

exit $EXIT_CODE
