#!/usr/bin/env bash
# Re-import R30 sweep from experiments/results/r30_sweep_results.json and redraw figures (no GPU).
set -euo pipefail
cd /root/autodl-tmp/loop_layer
python experiments/qwen3-8b/import_r30_sweep_and_plot.py
