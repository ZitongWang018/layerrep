#!/usr/bin/env bash
# Layer sweep: all T ranges + BoolQ/ARC (same examples per cell) + baseline / k=2 / k=3.
# Then aggregate plots and EXPERIMENT_REPORT.md (English figure titles).
#
# Usage:
#   chmod +x run_sweep.sh
#   ./run_sweep.sh
#
# Optional environment variables:
#   MODEL          - default /root/autodl-tmp/model_qwen
#   BOOLQ_LIMIT    - default 500
#   ARC_LIMIT      - default 500
#   OUT_CSV        - default artifacts/sweep_results.csv
#   SKIP_SWEEP=1   - only run analysis/plots (requires existing CSV)
#   MAX_CELLS      - e.g. 2 for a short debug run (default empty = full 99 cells)
#   RESUME=1       - append to CSV and skip existing (t_start,t_end) rows

set -euo pipefail
cd "$(dirname "$0")"

export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

MODEL="${MODEL:-/root/autodl-tmp/model_qwen}"
BOOLQ_LIMIT="${BOOLQ_LIMIT:-500}"
ARC_LIMIT="${ARC_LIMIT:-500}"
OUT_CSV="${OUT_CSV:-artifacts/sweep_results.csv}"
MAX_CELLS="${MAX_CELLS:-}"

SWEEP_ARGS=(
  --model-path "$MODEL"
  --boolq-limit "$BOOLQ_LIMIT"
  --arc-limit "$ARC_LIMIT"
  --out-csv "$OUT_CSV"
)

if [[ -n "$MAX_CELLS" ]]; then
  SWEEP_ARGS+=(--max-cells "$MAX_CELLS")
fi

if [[ "${RESUME:-0}" == "1" ]]; then
  SWEEP_ARGS+=(--resume)
fi

if [[ "${SKIP_SWEEP:-0}" != "1" ]]; then
  python3 run_sweep.py "${SWEEP_ARGS[@]}"
else
  echo "SKIP_SWEEP=1: skipping run_sweep.py"
fi

python3 analyze_plots.py --csv "$OUT_CSV" --report EXPERIMENT_REPORT.md

echo "Done. See EXPERIMENT_REPORT.md and figures/"
