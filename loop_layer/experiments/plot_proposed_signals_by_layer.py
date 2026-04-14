#!/usr/bin/env python3
"""
Per-benchmark: 20 samples × proposed (theory-motivated) signals vs layer.
Marks R30 optimal t_start / t_stop (same as r30_top_configs.txt).

Outputs: experiments/figures/proposed_signals_by_layer/
Plan: experiments/plan_proposed_signals_experiment.md
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

# Hugging Face Hub：未显式设置时使用国内镜像，便于 load_dataset 走缓存/拉取。
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

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

SIGNAL_KEYS = [
    "residual_delta_l2",
    "contraction_ratio",
    "logit_lens_jsd_vel",
    "logit_lens_jsd_curv",
    "erank",
    "delta_erank",
    "attn_consensus",
    "delta_norm_to_tstart",
    "attn_entropy",
    "logit_top1_margin",
]

MODEL_PATH = os.environ.get("R29_MODEL_PATH", "/root/autodl-tmp/model_qwen")
FIGURES_DIR = EXP / "figures" / "proposed_signals_by_layer"
RESULTS_DIR = EXP / "results"
DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_PER_BENCH = 20
BENCHMARKS = list(R30_OPTIMAL.keys())


def load_model_eager():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    return tok, model


def collect_samples(
    tok, model, n_layers: int
) -> dict[str, list[dict[str, list[float]]]]:
    out: dict[str, list[dict[str, list[float]]]] = {}
    for bench in BENCHMARKS:
        examples = load_benchmark(bench, N_PER_BENCH)
        if len(examples) < N_PER_BENCH:
            print(f"  [WARN] {bench}: only {len(examples)} examples (requested {N_PER_BENCH})")
        curves: list[dict[str, list[float]]] = []
        t0 = R30_OPTIMAL[bench]["t_start"]
        for i, ex in enumerate(examples[:N_PER_BENCH]):
            pref = tok(ex["prompt"], return_tensors="pt", add_special_tokens=False)
            input_ids = pref["input_ids"].to(DEVICE)
            attn = pref.get("attention_mask")
            attn = attn.to(DEVICE) if attn is not None else None
            try:
                sig = collect_proposed_signals(model, input_ids, attn, n_layers)
                add_delta_norm_to_tstart(sig, n_layers, t0)
                lists = proposed_signals_to_lists(sig, n_layers, SIGNAL_KEYS)
                curves.append(lists)
            except Exception as e:
                print(f"  [{bench} #{i}] probe error: {e}")
                continue
            if (i + 1) % 5 == 0:
                print(f"    {bench}: {i+1}/{min(N_PER_BENCH, len(examples))}")
        out[bench] = curves
    return out


def plot_benchmark(bench: str, sample_curves: list[dict[str, list[float]]], n_layers: int, opt: dict):
    if not sample_curves:
        print(f"  Skip plot {bench}: no curves")
        return
    layers = np.arange(n_layers)
    t_start, t_stop = opt["t_start"], opt["t_stop"]

    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    axes = axes.flatten()
    for ax_i, sk in enumerate(SIGNAL_KEYS):
        ax = axes[ax_i]
        stack = []
        for sc in sample_curves:
            y = np.array(sc.get(sk, [np.nan] * n_layers), dtype=np.float64)
            if y.shape[0] != n_layers:
                continue
            stack.append(y)
            ax.plot(layers, y, color="C0", alpha=0.18, linewidth=1.0, zorder=1)
        if stack:
            mean_y = np.nanmean(np.stack(stack, axis=0), axis=0)
            ax.plot(layers, mean_y, color="black", linewidth=2.0, label="mean", zorder=3)
        ax.axvline(
            t_start,
            color="#2ca02c",
            linestyle="--",
            linewidth=2.0,
            label=f"R30 t_start={t_start}",
            zorder=4,
        )
        ax.axvline(
            t_stop,
            color="#d62728",
            linestyle="--",
            linewidth=2.0,
            label=f"R30 t_stop={t_stop}",
            zorder=4,
        )
        ax.axvspan(t_start, t_stop, alpha=0.08, color="gold", zorder=0)
        ax.set_title(sk, fontsize=10)
        ax.set_xlabel("layer")
        ax.set_ylabel("value")
        ax.set_xlim(-0.5, n_layers - 0.5)
        ax.grid(True, alpha=0.25)
        if ax_i == 0:
            ax.legend(loc="upper right", fontsize=7)
    fig.suptitle(
        f"{bench} — {len(sample_curves)} samples | proposed signals | "
        f"R30 T-block [{t_start}, {t_stop}) (t_stop exclusive)",
        fontsize=12,
    )
    fig.tight_layout()
    out_path = FIGURES_DIR / f"proposed_signals_vs_layer_{bench.replace(' ', '_')}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "n_per_bench": N_PER_BENCH,
        "benchmarks": BENCHMARKS,
        "r30_optimal": R30_OPTIMAL,
        "signals": SIGNAL_KEYS,
        "model": MODEL_PATH,
        "plan_doc": str(EXP / "plan_proposed_signals_experiment.md"),
    }
    print(json.dumps(meta, indent=2))

    t0 = time.time()
    tok, model = load_model_eager()
    n_layers = model.config.num_hidden_layers
    print(f"n_layers={n_layers}, device={DEVICE}")

    data = collect_samples(tok, model, n_layers)
    meta["elapsed_probe_sec"] = time.time() - t0
    meta["counts"] = {b: len(data[b]) for b in BENCHMARKS}

    for bench in BENCHMARKS:
        plot_benchmark(bench, data[bench], n_layers, R30_OPTIMAL[bench])

    meta_path = RESULTS_DIR / "proposed_signals_by_layer_plot_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {meta_path}")
    print(f"Done in {meta['elapsed_probe_sec']:.1f}s")


if __name__ == "__main__":
    main()
