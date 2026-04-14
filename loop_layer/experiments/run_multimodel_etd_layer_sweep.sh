#!/usr/bin/env bash
# One-click R30-style ETD (t_start, t_stop) grid sweep for Llama3-8B and Gemma2-2B.
# Sample counts: 100 per benchmark, TruthfulQA=50 (same as experiments/results/r30_sweep_results.json).
# Grid sizes: Llama3-8B 246 cells (t_start 7–24, t_stop 13–30); Gemma2-2B 165 cells (5–18, 10–24).
# Optional: RUN_LLAMA=0 or RUN_GEMMA=0 to run only one model. Uses --resume (safe to interrupt).
# If you previously ran a debug sweep (--max-cells / small -n), remove the JSON under
# experiments/<model>/results/ before the full run, or those cells will be skipped with stale counts.
set -euo pipefail
cd /root/autodl-tmp/loop_layer
# shellcheck source=hf_hub_network_env.sh
source experiments/hf_hub_network_env.sh
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

RUN_LLAMA="${RUN_LLAMA:-1}"
RUN_GEMMA="${RUN_GEMMA:-1}"

if [[ "$RUN_LLAMA" == "1" ]]; then
  python experiments/multimodel_etd_layer_sweep.py --preset llama3-8b --resume
fi
if [[ "$RUN_GEMMA" == "1" ]]; then
  python experiments/multimodel_etd_layer_sweep.py --preset gemma2-2b --resume
fi

echo "Done. Outputs:"
echo "  llama3-8b:  experiments/llama3-8b/results/etd_layer_sweep_r30style.json"
echo "              experiments/llama3-8b/figures/*.png (heatmaps + baseline bar charts)"
echo "  gemma2-2b: experiments/gemma2-2b/results/etd_layer_sweep_r30style.json"
echo "             experiments/gemma2-2b/figures/*.png"
echo "Re-plot only (no GPU):"
echo "  python experiments/multimodel_sweep_visualize.py --json experiments/llama3-8b/results/etd_layer_sweep_r30style.json"
echo "  python experiments/multimodel_sweep_visualize.py --json experiments/gemma2-2b/results/etd_layer_sweep_r30style.json"
echo "Qwen3-8B (R30 import, same figure style):"
echo "  bash experiments/qwen3-8b/run_import_r30_and_plot.sh"
echo "Hard MC benchmarks (GPQA + Gaokao MathQA + LogiQA), three models:"
echo "  bash experiments/run_hard_mc_sweep_three_models.sh"
echo "  (GPQA needs HF gated access; see experiments/hard_mc_benchmark_loaders.py)"
