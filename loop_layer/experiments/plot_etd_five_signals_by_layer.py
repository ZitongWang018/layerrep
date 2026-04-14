#!/usr/bin/env python3
"""
Five ETD mechanism signals (table-aligned):
  (1) Local contraction δ_ℓ/δ_{ℓ-1} + block scalar CR_block in JSON;
  (2–4) JSD velocity, ΔeRank, ACI;
  (5) δ_ℓ/δ[t_start] trajectory + block scalar FPR_simple = value at ℓ=t_stop-1.
Same layout footprint as r30_optimal_by_layer: 2×5 subplots (rows 1–5 used, row 2 hidden).

Outputs: experiments/figures/etd_five_signals_by_layer/
Plan: experiments/plan_etd_five_signals.md
"""
from __future__ import annotations

import json
import os

if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path("/root/autodl-tmp/loop_layer")
EXP = ROOT / "experiments"
ETD = ROOT / "ETD"
for p in (str(ROOT), str(EXP), str(ETD)):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from etd_five_signals_metrics import (  # noqa: E402
    compute_cr_block,
    compute_fpr_simple,
    residual_delta_series,
)
from exp_round27_main import load_benchmark  # noqa: E402
from proposed_signals_probe import (  # noqa: E402
    add_delta_norm_to_tstart,
    collect_proposed_signals,
    proposed_signals_to_lists,
)

R30_OPTIMAL = {
    "ARC-C": {"t_start": 14, "t_stop": 20},
    "TruthfulQA": {"t_start": 16, "t_stop": 19},
    "CSQA": {"t_start": 10, "t_stop": 22},
    "MMLU-HS-Math": {"t_start": 10, "t_stop": 18},
}

CURVE_KEYS_MID = ["logit_lens_jsd_vel", "delta_erank", "attn_consensus"]
# 探针里一并拉出，用于子图 1 与子图 5 的逐层曲线
LAYER_KEYS_EXTRA = ["contraction_ratio", "delta_norm_to_tstart"]
PANEL_TITLES = [
    "contraction_ratio\nδ_ℓ/δ_{ℓ-1} (local); CR_block in meta",
    "logit_lens_jsd_vel\n(JSD velocity)",
    "delta_erank\n(Δ effective rank)",
    "attn_consensus\n(ACI)",
    "delta_norm_to_tstart\nFPR_simple = y at layer t_stop-1 (purple vline)",
]

MODEL_PATH = os.environ.get("R29_MODEL_PATH", "/root/autodl-tmp/model_qwen")
FIGURES_DIR = EXP / "figures" / "etd_five_signals_by_layer"
RESULTS_DIR = EXP / "results"
DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_PER_BENCH = int(os.environ.get("ETD_FIVE_N_PER_BENCH", "20"))
BENCHMARKS = list(R30_OPTIMAL.keys())


def load_model_eager():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    # 默认 auto 以适配多卡/显存紧张；若独占 GPU 可设 ETD_DEVICE_MAP=cuda0。
    dm_raw = os.environ.get("ETD_DEVICE_MAP", "auto").strip().lower()
    if dm_raw in ("cuda0", "0", "single") and torch.cuda.is_available():
        device_map = {"": 0}
    elif torch.cuda.is_available():
        device_map = "auto"
    else:
        device_map = None
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=DTYPE,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    if device_map is None:
        model = model.to(DEVICE)
    model.eval()
    return tok, model


def collect_benchmark(
    tok, model, bench: str, n_layers: int
) -> tuple[list[dict], list[dict[str, float]]]:
    """
    Returns (curve_rows per sample, block_scalars per sample)
    curve_rows: per-layer lists for mid three signals + contraction_ratio + delta_norm_to_tstart
    block_scalars: {cr_block, fpr_simple} (T-block aggregates, saved to meta)
    """
    opt = R30_OPTIMAL[bench]
    t0, t1 = opt["t_start"], opt["t_stop"]
    examples = load_benchmark(bench, N_PER_BENCH)
    curves: list[dict] = []
    scalars: list[dict[str, float]] = []
    for i, ex in enumerate(examples[:N_PER_BENCH]):
        pref = tok(ex["prompt"], return_tensors="pt", add_special_tokens=False)
        input_ids = pref["input_ids"].to(DEVICE)
        attn = pref.get("attention_mask")
        attn = attn.to(DEVICE) if attn is not None else None
        try:
            sig = collect_proposed_signals(model, input_ids, attn, n_layers)
            add_delta_norm_to_tstart(sig, n_layers, t0)
            delta = residual_delta_series(sig, n_layers)
            cr = compute_cr_block(delta, t0, t1)
            fp = compute_fpr_simple(delta, t0, t1)
            lists = proposed_signals_to_lists(
                sig, n_layers, CURVE_KEYS_MID + LAYER_KEYS_EXTRA
            )
            curves.append(lists)
            scalars.append({"cr_block": cr, "fpr_simple": fp})
        except Exception as e:
            print(f"  [{bench} #{i}] probe error: {e}")
            continue
        if (i + 1) % 5 == 0:
            print(f"    {bench}: {i+1}/{min(N_PER_BENCH, len(examples))}")
    return curves, scalars


def plot_benchmark(
    bench: str,
    curve_rows: list[dict],
    block_scalars: list[dict[str, float]],
    n_layers: int,
    opt: dict,
):
    if not curve_rows or len(curve_rows) != len(block_scalars):
        print(f"  Skip plot {bench}: mismatched or empty data")
        return
    layers = np.arange(n_layers)
    t_start, t_stop = opt["t_start"], opt["t_stop"]
    xlim = (-0.5, n_layers - 0.5)

    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    axes = axes.flatten()

    # --- Panel 0: per-layer contraction ratio (not the scalar CR_block horizontal line) ---
    ax = axes[0]
    stack_cr = []
    for row in curve_rows:
        y = np.array(
            row.get("contraction_ratio", [np.nan] * n_layers), dtype=np.float64
        )
        if y.shape[0] != n_layers:
            continue
        stack_cr.append(y)
        ax.plot(layers, y, color="C0", alpha=0.18, linewidth=1.0, zorder=1)
    if stack_cr:
        mean_cr = np.nanmean(np.stack(stack_cr, axis=0), axis=0)
        ax.plot(layers, mean_cr, color="black", linewidth=2.0, label="mean", zorder=3)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.2, label="ratio=1", zorder=2)
    crs = [s["cr_block"] for s in block_scalars]
    finite_cb = [y for y in crs if np.isfinite(y)]
    if finite_cb:
        mcb = float(np.mean(finite_cb))
        ax.axhline(
            mcb,
            color="#ff7f0e",
            linestyle="-.",
            linewidth=1.8,
            label=f"mean CR_block={mcb:.3f}",
            zorder=4,
        )
    ax.axvline(t_start, color="#2ca02c", linestyle="--", linewidth=2.0, label=f"R30 t_start={t_start}")
    ax.axvline(t_stop, color="#d62728", linestyle="--", linewidth=2.0, label=f"R30 t_stop={t_stop}")
    ax.axvspan(t_start, t_stop, alpha=0.08, color="gold", zorder=0)
    ax.set_title(PANEL_TITLES[0], fontsize=9)
    ax.set_xlabel("layer")
    ax.set_ylabel("δ_ℓ / δ_{ℓ-1}")
    ax.set_xlim(*xlim)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=6)

    # --- Panels 1–3: curves ---
    for j, key in enumerate(CURVE_KEYS_MID):
        ax = axes[1 + j]
        stack = []
        for row in curve_rows:
            y = np.array(row.get(key, [np.nan] * n_layers), dtype=np.float64)
            if y.shape[0] != n_layers:
                continue
            stack.append(y)
            ax.plot(layers, y, color="C0", alpha=0.18, linewidth=1.0, zorder=1)
        if stack:
            mean_y = np.nanmean(np.stack(stack, axis=0), axis=0)
            ax.plot(layers, mean_y, color="black", linewidth=2.0, label="mean", zorder=3)
        ax.axvline(t_start, color="#2ca02c", linestyle="--", linewidth=2.0, zorder=4)
        ax.axvline(t_stop, color="#d62728", linestyle="--", linewidth=2.0, zorder=4)
        ax.axvspan(t_start, t_stop, alpha=0.08, color="gold", zorder=0)
        ax.set_title(PANEL_TITLES[1 + j], fontsize=9)
        ax.set_xlabel("layer")
        ax.set_ylabel(key.split("_")[-1] if key else "value")
        ax.set_xlim(*xlim)
        ax.grid(True, alpha=0.25)
        if j == 0:
            ax.legend(loc="upper right", fontsize=7)

    # --- Panel 4: δ_ℓ/δ[t_start] vs layer; FPR_simple = y at ℓ = t_stop-1 ---
    ax = axes[4]
    stack_fp = []
    for row in curve_rows:
        y = np.array(
            row.get("delta_norm_to_tstart", [np.nan] * n_layers), dtype=np.float64
        )
        if y.shape[0] != n_layers:
            continue
        stack_fp.append(y)
        ax.plot(layers, y, color="C0", alpha=0.18, linewidth=1.0, zorder=1)
    if stack_fp:
        mean_fp = np.nanmean(np.stack(stack_fp, axis=0), axis=0)
        ax.plot(layers, mean_fp, color="black", linewidth=2.0, label="mean", zorder=3)
        li_exit = t_stop - 1
        if 0 <= li_exit < n_layers:
            ax.scatter(
                [li_exit],
                [mean_fp[li_exit]],
                color="#d62728",
                s=36,
                zorder=5,
                label="mean @ t_stop-1 (FPR)",
            )
    ax.axvline(t_start, color="#2ca02c", linestyle="--", linewidth=2.0, label=f"R30 t_start={t_start}")
    ax.axvline(t_stop, color="#d62728", linestyle="--", linewidth=2.0, label=f"R30 t_stop={t_stop}")
    ax.axvline(
        t_stop - 1,
        color="purple",
        linestyle=":",
        linewidth=1.5,
        label="layer = t_stop-1",
        zorder=4,
    )
    ax.axvspan(t_start, t_stop, alpha=0.08, color="gold", zorder=0)
    ax.set_title(PANEL_TITLES[4], fontsize=9)
    ax.set_xlabel("layer")
    ax.set_ylabel("δ_ℓ / δ[t_start]")
    ax.set_xlim(*xlim)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=6)

    for k in range(5, 10):
        axes[k].axis("off")

    fig.suptitle(
        f"{bench} — {len(curve_rows)} samples | ETD five signals | "
        f"R30 T-block [{t_start}, {t_stop})",
        fontsize=12,
    )
    fig.tight_layout()
    out_path = FIGURES_DIR / f"etd_five_signals_vs_layer_{bench.replace(' ', '_')}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    meta: dict = {
        "n_per_bench": N_PER_BENCH,
        "benchmarks": BENCHMARKS,
        "r30_optimal": R30_OPTIMAL,
        "panels": PANEL_TITLES,
        "curve_keys_mid": CURVE_KEYS_MID,
        "layer_keys_extra": LAYER_KEYS_EXTRA,
        "model": MODEL_PATH,
        "plan_doc": str(EXP / "plan_etd_five_signals.md"),
    }
    print(json.dumps({k: meta[k] for k in meta if k != "per_benchmark_scalars"}, indent=2))

    t0 = time.time()
    tok, model = load_model_eager()
    n_layers = model.config.num_hidden_layers
    print(f"n_layers={n_layers}, device={DEVICE}")

    per_bench_curves: dict[str, list] = {}
    per_bench_scalars: dict[str, list] = {}
    for bench in BENCHMARKS:
        curves, scalars = collect_benchmark(tok, model, bench, n_layers)
        per_bench_curves[bench] = curves
        per_bench_scalars[bench] = scalars
        plot_benchmark(bench, curves, scalars, n_layers, R30_OPTIMAL[bench])

    meta["elapsed_sec"] = time.time() - t0
    meta["counts"] = {b: len(per_bench_curves[b]) for b in BENCHMARKS}
    meta["per_benchmark_block_scalars"] = {
        b: per_bench_scalars[b] for b in BENCHMARKS
    }

    meta_path = RESULTS_DIR / "etd_five_signals_by_layer_plot_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {meta_path}")
    print(f"Done in {meta['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
