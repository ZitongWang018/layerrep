#!/usr/bin/env python3
"""
R29 Phase 0: collect full-layer signal profiles on prompt tokens; oracle ETD gain; correlations; plots (English).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

sys.path.insert(0, "/root/autodl-tmp/loop_layer")
sys.path.insert(0, "/root/autodl-tmp/loop_layer/experiments")
sys.path.insert(0, "/root/autodl-tmp/loop_layer/ETD")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from etd_forward import baseline_forward_logits, etd_forward_logits, loglikelihood_continuation  # noqa: E402
from r29.probe_forward import collect_probe_signals  # noqa: E402
from r29_common import FIGURES_DIR, MODEL_PATH, RESULTS_DIR, load_benchmarks  # noqa: E402

DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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
BENCHES_DEFAULT = ["BoolQ", "ARC-C", "ARC-Easy", "CSQA", "TruthfulQA"]
CHAMP_T_START = 8
CHAMP_T_STOP = 22
CHAMP_K = 2


def load_model_eager():
    print("Loading model (eager attention for weights) …")
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


@torch.no_grad()
def mc_correct_baseline_champion(tok, model, ex: dict) -> tuple[bool, bool]:
    """Whether baseline / champion MC prediction matches gold."""
    prefix = ex["prompt"]
    choices_sp = [" " + c for c in ex["choices"]]
    gold = ex["answer"].strip().lower()
    pref = tok(prefix, return_tensors="pt", add_special_tokens=False)
    prompt_len = pref["input_ids"].shape[1]
    input_ids_pref = pref["input_ids"].to(DEVICE)
    scores_b = []
    scores_c = []
    for ch in choices_sp:
        full = tok(prefix + ch, return_tensors="pt", add_special_tokens=False)
        ids = full["input_ids"].to(DEVICE)
        am = full.get("attention_mask")
        am = am.to(DEVICE) if am is not None else None
        lb = baseline_forward_logits(model, ids, am)
        scores_b.append(loglikelihood_continuation(lb, ids, prompt_len))
        n_t = CHAMP_T_STOP - CHAMP_T_START
        lc = etd_forward_logits(
            model,
            ids,
            am,
            CHAMP_T_START,
            n_t,
            CHAMP_K,
            alpha="auto",
        )
        scores_c.append(loglikelihood_continuation(lc, ids, prompt_len))
    pred_b = ex["choices"][int(np.argmax(scores_b))].strip().lower()
    pred_c = ex["choices"][int(np.argmax(scores_c))].strip().lower()
    return pred_b == gold, pred_c == gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-bench", type=int, default=int(os.environ.get("R29_N", "32")))
    ap.add_argument("--benches", type=str, default=",".join(BENCHES_DEFAULT))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    benches = [b.strip() for b in args.benches.split(",") if b.strip()]

    meta = {
        "phase": "R29_phase0",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "n_per_bench": args.n_per_bench,
        "benches": benches,
        "model": MODEL_PATH,
    }
    print(json.dumps(meta, indent=2))

    data = load_benchmarks(benches, args.n_per_bench)
    if not data:
        print("No data loaded, exit.")
        return

    tok, model = load_model_eager()
    n_layers = model.config.num_hidden_layers

    all_samples: list[dict] = []
    t0 = time.time()

    for bench, examples in data.items():
        for i, ex in enumerate(examples):
            pref = tok(ex["prompt"], return_tensors="pt", add_special_tokens=False)
            input_ids = pref["input_ids"].to(DEVICE)
            attn = pref.get("attention_mask")
            attn = attn.to(DEVICE) if attn is not None else None
            try:
                bl_ok, ch_ok = mc_correct_baseline_champion(tok, model, ex)
                gain = int(ch_ok) - int(bl_ok)  # +1 if champion fixes, -1 if breaks, 0 else
                sig = collect_probe_signals(model, input_ids, attn, n_layers)
            except Exception as e:
                print(f"  [{bench} #{i}] error: {e}")
                continue
            row = {
                "benchmark": bench,
                "sample_index": i,
                "input_length": int(input_ids.shape[1]),
                "baseline_correct": bl_ok,
                "champion_correct": ch_ok,
                "oracle_etd_gain": gain,
                "signals": {str(k): v for k, v in sig.items()},
            }
            all_samples.append(row)
            if (i + 1) % 10 == 0:
                print(f"  {bench}: {i+1}/{len(examples)} samples, elapsed {time.time()-t0:.0f}s")

    meta["elapsed_sec"] = time.time() - t0
    meta["n_samples_total"] = len(all_samples)
    meta["finished_utc"] = datetime.now(timezone.utc).isoformat()

    out_path = os.path.join(RESULTS_DIR, "round29_phase0_profiles.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "samples": all_samples}, f, indent=2)
    print(f"Wrote {out_path}")

    # --- correlation: per-layer each signal vs oracle_etd_gain ---
    gains = np.array([s["oracle_etd_gain"] for s in all_samples], dtype=np.float64)
    corr_path = os.path.join(RESULTS_DIR, "round29_phase0_correlation.json")
    corr_mat = {}
    for sk in SIGNAL_KEYS:
        corr_mat[sk] = []
        for li in range(n_layers):
            vals = []
            for s in all_samples:
                v = s["signals"].get(str(li), {}).get(sk, float("nan"))
                vals.append(float(v))
            vals = np.array(vals, dtype=np.float64)
            m = np.isfinite(vals) & np.isfinite(gains)
            if m.sum() < 8:
                corr_mat[sk].append(None)
                continue
            c = np.corrcoef(vals[m], gains[m])[0, 1]
            corr_mat[sk].append(float(c) if np.isfinite(c) else None)
    with open(corr_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "pearson_r_vs_oracle_gain": corr_mat}, f, indent=2)
    print(f"Wrote {corr_path}")

    # --- plots (English) ---
    plot_mean_profiles(all_samples, n_layers, meta)
    plot_correlation_heatmap(corr_mat, n_layers, meta)
    print(f"Figures under {FIGURES_DIR}")


def plot_mean_profiles(samples: list[dict], n_layers: int, meta: dict):
    """V2-style: mean ± std per benchmark for each signal."""
    benches = sorted(set(s["benchmark"] for s in samples))
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    axes = axes.flatten()
    cmap = plt.cm.tab10(np.linspace(0, 0.9, len(benches)))
    for ax_i, sk in enumerate(SIGNAL_KEYS):
        ax = axes[ax_i]
        for bi, bench in enumerate(benches):
            sub = [s for s in samples if s["benchmark"] == bench]
            if not sub:
                continue
            prof = np.zeros((len(sub), n_layers))
            for si, s in enumerate(sub):
                for li in range(n_layers):
                    prof[si, li] = s["signals"].get(str(li), {}).get(sk, float("nan"))
            mean_p = np.nanmean(prof, axis=0)
            std_p = np.nanstd(prof, axis=0)
            layers = np.arange(n_layers)
            ax.plot(layers, mean_p, color=cmap[bi], label=bench, linewidth=1.5)
            ax.fill_between(layers, mean_p - std_p, mean_p + std_p, color=cmap[bi], alpha=0.12)
        ax.axvspan(CHAMP_T_START, CHAMP_T_STOP, color="yellow", alpha=0.12)
        ax.axvline(CHAMP_T_START, color="navy", linestyle="--", linewidth=0.8)
        ax.axvline(CHAMP_T_STOP, color="darkred", linestyle="--", linewidth=0.8)
        ax.set_title(sk, fontsize=9)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Value")
        if ax_i == 0:
            ax.legend(fontsize=6, loc="upper right")
    fig.suptitle("R29 Phase 0: Mean signal profiles by benchmark (shaded ±1 std)", fontsize=12)
    plt.tight_layout()
    p = os.path.join(FIGURES_DIR, "r29_phase0_mean_profiles.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {p}")


def plot_correlation_heatmap(corr_mat: dict, n_layers: int, meta: dict):
    mat = np.full((len(SIGNAL_KEYS), n_layers), np.nan, dtype=np.float64)
    for i, sk in enumerate(SIGNAL_KEYS):
        row = corr_mat.get(sk, [])
        for j in range(min(n_layers, len(row))):
            v = row[j]
            if v is not None:
                mat[i, j] = v
    fig, ax = plt.subplots(figsize=(16, 6))
    vmax = 0.35
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(len(SIGNAL_KEYS)))
    ax.set_yticklabels(SIGNAL_KEYS, fontsize=8)
    ax.set_xticks(range(0, n_layers, 2))
    ax.set_xlabel("Layer")
    ax.set_title("Pearson r: signal @ layer vs oracle ETD gain (champion correct − baseline correct)")
    ax.axvline(CHAMP_T_START - 0.5, color="blue", linewidth=1.5)
    ax.axvline(CHAMP_T_STOP - 0.5, color="red", linewidth=1.5)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    plt.tight_layout()
    p = os.path.join(FIGURES_DIR, "r29_phase0_correlation_heatmap.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {p}")


if __name__ == "__main__":
    main()
