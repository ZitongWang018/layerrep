# R29 experiment run log

## Outputs

| File | Description |
|------|-------------|
| `round29_phase0_profiles.json` | Per-sample per-layer signals + oracle gain (champion correct − baseline correct) |
| `round29_phase0_correlation.json` | Pearson r vs oracle gain per signal × layer |
| `round29_phase1_results.json` | Summary + per-sample correctness and (t_start, t_stop) per strategy |
| `../figures/r29_phase0_*.png` | Mean profiles, correlation heatmap (English labels) |
| `../figures/r29_phase1_*.png` | Accuracy delta bars, B1_6sig boundary scatter (English labels) |

## How to run

```bash
chmod +x /root/autodl-tmp/loop_layer/experiments/run_r29.sh
R29_N=20 HF_DATASETS_OFFLINE=1 /root/autodl-tmp/loop_layer/experiments/run_r29.sh
```

Single phase:

```bash
cd /root/autodl-tmp/loop_layer/experiments
HF_DATASETS_OFFLINE=1 python3 exp_round29_phase0.py --n-per-bench 24
HF_DATASETS_OFFLINE=1 python3 exp_round29_phase1.py --n-per-bench 16
```

## Note on “two experiments in parallel”

This machine has **one** GPU. Loading Qwen3-8B twice simultaneously risks OOM, so Phase 0 and Phase 1 run **sequentially** in `run_r29.sh`. With two GPUs you could run:

```bash
CUDA_VISIBLE_DEVICES=0 python3 exp_round29_phase0.py ... &
CUDA_VISIBLE_DEVICES=1 python3 exp_round29_phase1.py ... &
wait
```

## Boundary constraints (important)

Profile detectors use `l_min=8` and `apply_boundary_constraints(min_start=8)` so **`t_start` is never below 8** on Qwen3-8B (matches champion encoder depth and avoids R21-style shallow T-blocks). **`t_stop` remains signal-driven.** For other models, change these defaults in `r29/profile_analysis.py`.

## Latest run (2026-04-09)

- Command: `R29_N=12 HF_DATASETS_OFFLINE=1 bash run_r29.sh`
- Samples: 12 per benchmark × 5 benchmarks = 60 (Phase 0 and Phase 1).
- Phase 0 wall time ~45 s (after model load); Phase 1 ~130 s.
- **Macro mean accuracy** (5 benchmarks): Baseline 0.650, Champion 0.650, **B1_6sig 0.633**, B3_consensus 0.617, PA1_layer_sim 0.633.
- Full log: `experiments/results/r29_full_run.log`

## Code layout

- `experiments/r29/signal_funcs.py` — signal formulas
- `experiments/r29/probe_forward.py` — hooks + one eager forward
- `experiments/r29/profile_analysis.py` — PA1 / B1_6sig / B3
- `experiments/r29_common.py` — datasets + paths
- `experiments/exp_round29_phase0.py` — Phase 0 driver
- `experiments/exp_round29_phase1.py` — Phase 1 driver

Model path: env `R29_MODEL_PATH` or default `/root/autodl-tmp/model_qwen`.  
Attention: **eager** (required for attention weights).
