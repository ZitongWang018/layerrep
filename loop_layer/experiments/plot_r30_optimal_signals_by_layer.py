#!/usr/bin/env python3
"""
Per-benchmark: 20 samples × probe signals vs layer (x=layer index, y=signal).
Marks R30 sweep #1 optimal t_start / t_stop from r30_top_configs.txt.

Outputs under experiments/figures/r30_optimal_by_layer/
"""
from __future__ import annotations

import json
import os
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

from exp_round27_main import load_benchmark  # noqa: E402
from r29.probe_forward import collect_probe_signals, signals_dict_to_lists  # noqa: E402

# R30 top-1 per benchmark (r30_top_configs.txt, grid sweep on same 4 benches)
R30_OPTIMAL = {
    "ARC-C": {"t_start": 14, "t_stop": 20},
    "TruthfulQA": {"t_start": 16, "t_stop": 19},
    "CSQA": {"t_start": 10, "t_stop": 22},
    "MMLU-HS-Math": {"t_start": 10, "t_stop": 18},
}

SIGNAL_KEYS = [
    "attn_entropy",
    "ffn_gate_norm",
    "layer_sim",
    "head_specialization",
    "logit_lens_KL",
    "attention_locality",
    "residual_write_norm",
    "participation_ratio",
    "prediction_flip_rate",
    "attn_sink_ratio",
]

MODEL_PATH = os.environ.get("R29_MODEL_PATH", "/root/autodl-tmp/model_qwen")
FIGURES_DIR = EXP / "figures" / "r30_optimal_by_layer"
RESULTS_DIR = EXP / "results"
DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_PER_BENCH = 20
BENCHMARKS = list(R30_OPTIMAL.keys())


def load_model_eager():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    return tok, model


def collect_samples_signals(tok, model, n_layers: int) -> dict[str, list[dict[str, list[float]]]]:
    """
    Returns bench -> list of length N, each item is {signal_name: [L floats]}
    """
    out: dict[str, list[dict[str, list[float]]]] = {}
    for bench in BENCHMARKS:
        examples = load_benchmark(bench, N_PER_BENCH)
        if len(examples) < N_PER_BENCH:
            print(f"  [WARN] {bench}: only {len(examples)} examples (requested {N_PER_BENCH})")
        curves: list[dict[str, list[float]]] = []
        for i, ex in enumerate(examples[:N_PER_BENCH]):
            pref = tok(ex["prompt"], return_tensors="pt", add_special_tokens=False)
            input_ids = pref["input_ids"].to(DEVICE)
            attn = pref.get("attention_mask")
            attn = attn.to(DEVICE) if attn is not None else None
            try:
                sig = collect_probe_signals(model, input_ids, attn, n_layers)
                lists = signals_dict_to_lists(sig, n_layers, SIGNAL_KEYS)
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
    t0, t1 = opt["t_start"], opt["t_stop"]

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
        ax.axvline(t0, color="#2ca02c", linestyle="--", linewidth=2.0, label=f"R30 t_start={t0}", zorder=4)
        ax.axvline(t1, color="#d62728", linestyle="--", linewidth=2.0, label=f"R30 t_stop={t1}", zorder=4)
        ax.axvspan(t0, t1, alpha=0.08, color="gold", zorder=0)
        ax.set_title(sk, fontsize=10)
        ax.set_xlabel("layer")
        ax.set_ylabel("value")
        ax.set_xlim(-0.5, n_layers - 0.5)
        ax.grid(True, alpha=0.25)
        if ax_i == 0:
            ax.legend(loc="upper right", fontsize=7)
    fig.suptitle(
        f"{bench} — {len(sample_curves)} samples | R30 optimal T-block [{t0}, {t1}) (t_stop exclusive)",
        fontsize=12,
    )
    fig.tight_layout()
    out_path = FIGURES_DIR / f"r30_signals_vs_layer_{bench.replace(' ', '_')}.png"
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
    }
    print(json.dumps(meta, indent=2))

    t0 = time.time()
    tok, model = load_model_eager()
    n_layers = model.config.num_hidden_layers
    print(f"n_layers={n_layers}, device={DEVICE}")

    data = collect_samples_signals(tok, model, n_layers)
    meta["elapsed_probe_sec"] = time.time() - t0
    meta["counts"] = {b: len(data[b]) for b in BENCHMARKS}

    for bench in BENCHMARKS:
        plot_benchmark(bench, data[bench], n_layers, R30_OPTIMAL[bench])

    meta_path = RESULTS_DIR / "r30_optimal_by_layer_plot_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {meta_path}")
    print(f"Done in {meta['elapsed_probe_sec']:.1f}s")


if __name__ == "__main__":
    main()
