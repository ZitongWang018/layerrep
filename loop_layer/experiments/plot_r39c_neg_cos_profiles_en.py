#!/usr/bin/env python3
"""
Redraw R39C 03_neg_cos_am_profiles.png from results JSON:
- x: layer index, y: mean neg_cos_am (calibration)
- shaded: sweep_best + neg_cos_am_calib only (no emp_logit)
- English legend and titles only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BENCH_ORDER = [
    "BoolQ",
    "ARC-C",
    "TruthfulQA",
    "CSQA",
    "MMLU-HS-Math",
    "GPQA-Diamond",
    "AGIEval-Gaokao-MathQA",
    "LogiQA",
]

PRESET_LABEL = {
    "qwen3-8b": "Qwen3-8B",
    "llama3-8b": "Llama3-8B",
    "gemma2-2b": "Gemma2-2B",
}
# Match exp_r39c_final.py PRESETS["*"]["out_fig"] folder names
PRESET_FIG_SUBDIR = {
    "qwen3-8b": "r39c_final_qwen3",
    "llama3-8b": "r39c_final_llama3",
    "gemma2-2b": "r39c_final_gemma2",
}


def plot_neg_cos_profiles_en(
    all_results: dict,
    fig_dir: Path,
    preset_key: str,
) -> Path:
    fig_dir.mkdir(parents=True, exist_ok=True)
    label = PRESET_LABEL.get(preset_key, preset_key)
    benches = [b for b in BENCH_ORDER if b in all_results]
    n_b = len(benches)
    n_cols = 4
    n_rows = (n_b + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, b in enumerate(benches):
        ax = axes_flat[idx]
        res = all_results[b]
        prof = {int(k): v for k, v in res.get("mean_profile", {}).items()}
        if not prof:
            ax.set_title(b[:16])
            continue
        layers = sorted(prof.keys())
        vals = [prof[l] for l in layers]
        ax.plot(layers, vals, "r-o", markersize=4, linewidth=1.5, label="neg_cos_am (calib. mean)")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        sw = res["sweep_best_window"]
        ax.axvspan(
            sw[0],
            sw[1],
            alpha=0.18,
            color="#1f77b4",
            label=f"Sweep-best window [{sw[0]}, {sw[1]})",
        )
        cw = res.get("neg_cos_am_calib_win")
        if cw:
            ax.axvspan(
                cw[0],
                cw[1],
                alpha=0.15,
                color="#9467bd",
                label=f"Calib window [{cw[0]}, {cw[1]})",
            )
        ax.set_title(
            f"{b[:18]}  ps_t_start={res.get('ps_tstart_mean', 0):.1f}±{res.get('ps_tstart_std', 0):.1f}",
            fontsize=8,
        )
        ax.set_xlabel("Layer", fontsize=7)
        ax.set_ylabel("neg_cos_am", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, loc="best")

    for idx in range(len(benches), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        f"{label}: neg_cos_am layer profile (calibration mean) vs. windows",
        fontsize=10,
    )
    plt.tight_layout()
    p = fig_dir / "03_neg_cos_am_profiles.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close()
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--json",
        type=Path,
        default=EXP / "results" / "r39c_final_qwen3.json",
        help="Path to r39c_final_*.json",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Figure directory (default: figures/r39c_final_<preset>)",
    )
    args = ap.parse_args()

    data = json.loads(args.json.read_text(encoding="utf-8"))
    preset = data.get("preset", "qwen3-8b")
    all_results = data.get("results", data)
    sub = PRESET_FIG_SUBDIR.get(preset, f"r39c_final_{preset.replace('-', '_')}")
    out_dir = args.out_dir or (EXP / "figures" / sub)
    p = plot_neg_cos_profiles_en(all_results, out_dir, preset)
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
