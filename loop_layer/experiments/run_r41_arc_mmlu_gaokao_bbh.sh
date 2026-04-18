#!/usr/bin/env bash
# R41 子集：ARC-C、MMLU-HS-Math、AGIEval-Gaokao-MathQA、BBH（6 子任务）、GSM8K（贪婪生成 EM）
# 条件：baseline、sweep_best（R39C Qwen3 固定窗）、neg_cos_am_calib、neg_cos_am_prop_attn（无 emp / reflux）
#
# 默认：直接跑正式评测（不跑 smoke）。若需先冒烟： RUN_SMOKE=1 ./experiments/run_r41_arc_mmlu_gaokao_bbh.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export PYTHONUNBUFFERED=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY 2>/dev/null || true

LOG="${LOG:-experiments/results/r41_arc_mmlu_gaokao_bbh.log}"
RUN_SMOKE="${RUN_SMOKE:-0}"
# 断点续跑：从同一 output-json 恢复已完成 benchmark（n 须与本次一致才跳过）
RESUME="${RESUME:-0}"
mkdir -p "$(dirname "$LOG")"
N_MC="${N_MC:-50}"
N_GSM="${N_GSM:-$N_MC}"
# 每子任务条数（与 BBH_TOTAL 二选一：若设置 BBH_TOTAL，则六子任务合计为该数并均分）
N_BBH="${N_BBH:-50}"
BBH_TOTAL="${BBH_TOTAL:-}"
# 空格分隔；默认含 gsm8k。若不要 GSM8K： BENCHMARKS="arc-c mmlu-hs-math gaokao bbh"
BENCHMARKS="${BENCHMARKS:-arc-c mmlu-hs-math gaokao bbh gsm8k}"
read -r -a BENCH_ARR <<< "$BENCHMARKS"

echo "Log: $(readlink -f "$LOG" 2>/dev/null || echo "$ROOT/$LOG")"
echo "RUN_SMOKE=$RUN_SMOKE (set RUN_SMOKE=1 to run smoke first)"
echo "RESUME=$RESUME (set RESUME=1 to continue from existing JSON)"

if [[ -n "${BBH_TOTAL}" ]]; then
  PHASE_MSG="Full run (MC n=$N_MC / BBH total=$BBH_TOTAL across 6 subtasks)"
  EXTRA_BBH=(--bbh-total "$BBH_TOTAL")
else
  PHASE_MSG="Full run (MC n=$N_MC / BBH per-subtask=$N_BBH)"
  EXTRA_BBH=(--bbh-limit "$N_BBH")
fi

RESUME_ARGS=()
if [[ "$RESUME" == "1" ]]; then
  RESUME_ARGS=(--resume)
fi
OUT_JSON="${OUT_JSON:-experiments/results/r41_qwen3_arc_mmlu_gaokao_bbh.json}"

if [[ "$RUN_SMOKE" == "1" ]]; then
  echo ""
  echo "========== [1/2] Smoke (1 sample each) =========="
  python3 experiments/exp_r41_reflux_jac_etd.py \
    --benchmarks "${BENCH_ARR[@]}" \
    --smoke \
    --output-json experiments/results/r41_qwen3_smoke.json \
    2>&1 | tee "$LOG"
  echo ""
  echo "========== [2/2] $PHASE_MSG =========="
  python3 experiments/exp_r41_reflux_jac_etd.py \
    --benchmarks "${BENCH_ARR[@]}" \
    --n-samples "$N_MC" \
    --gsm-limit "$N_GSM" \
    "${EXTRA_BBH[@]}" \
    "${RESUME_ARGS[@]}" \
    --output-json "$OUT_JSON" \
    2>&1 | tee -a "$LOG"
else
  echo ""
  echo "========== $PHASE_MSG =========="
  python3 experiments/exp_r41_reflux_jac_etd.py \
    --benchmarks "${BENCH_ARR[@]}" \
    --n-samples "$N_MC" \
    --gsm-limit "$N_GSM" \
    "${EXTRA_BBH[@]}" \
    "${RESUME_ARGS[@]}" \
    --output-json "$OUT_JSON" \
    2>&1 | tee "$LOG"
fi

echo ""
echo "Outputs:"
echo "  JSON: $OUT_JSON"
echo "  Figures: experiments/figures/r41_qwen3/r41_accuracy_comparison.png"
echo "           experiments/figures/r41_qwen3/neg_cos_am_profiles.png"
