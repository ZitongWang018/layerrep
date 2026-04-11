#!/usr/bin/env python3
"""
R29 Phase 1: single-signal and combo profile detectors vs Baseline/Champion; record boundaries and accuracy.
Plots in English.
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
from r29.profile_analysis import (  # noqa: E402
    SIGNAL_MODES,
    b1_weighted_combo,
    b3_median_consensus,
    pa1_single_signal,
)
from r29_common import FIGURES_DIR, MODEL_PATH, RESULTS_DIR, load_benchmarks  # noqa: E402

DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BENCHES_DEFAULT = ["BoolQ", "ARC-C", "ARC-Easy", "CSQA", "TruthfulQA"]
CHAMP_K = 2

SINGLE_SIGNALS = [
    "layer_sim",
    "attn_entropy",
    "ffn_gate_norm",
    "head_specialization",
    "logit_lens_KL",
    "attention_locality",
    "residual_write_norm",
    "participation_ratio",
    "prediction_flip_rate",
    "attn_sink_ratio",
]

B1_6SIG = {
    "layer_sim": "valley",
    "attn_entropy": "peak",
    "head_specialization": "peak",
    "logit_lens_KL": "peak",
    "attention_locality": "peak",
    "participation_ratio": "peak",
}


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


def bounds_for_strategy(name: str, signals: dict, n_layers: int) -> tuple[int, int, bool]:
    if name == "Champion":
        return 8, 22, False
    if name.startswith("PA1_"):
        sig = name.replace("PA1_", "")
        mode = SIGNAL_MODES.get(sig, "peak")
        return pa1_single_signal(signals, sig, mode, n_layers)
    if name == "B1_6sig":
        return b1_weighted_combo(signals, n_layers, B1_6SIG, None)
    if name == "B3_consensus":
        return b3_median_consensus(signals, n_layers, list(B1_6SIG.keys()))
    raise ValueError(name)


@torch.no_grad()
def score_mc(
    tok,
    model,
    ex: dict,
    t_start: int | None,
    t_stop: int | None,
    use_etd: bool,
) -> bool:
    prefix = ex["prompt"]
    gold = ex["answer"].strip().lower()
    choices_sp = [" " + c for c in ex["choices"]]
    pref = tok(prefix, return_tensors="pt", add_special_tokens=False)
    prompt_len = pref["input_ids"].shape[1]
    scores = []
    for ch in choices_sp:
        full = tok(prefix + ch, return_tensors="pt", add_special_tokens=False)
        ids = full["input_ids"].to(DEVICE)
        am = full.get("attention_mask")
        am = am.to(DEVICE) if am is not None else None
        if not use_etd or t_start is None or t_stop is None:
            logits = baseline_forward_logits(model, ids, am)
        else:
            n_t = max(t_stop - t_start, 1)
            logits = etd_forward_logits(
                model, ids, am, t_start, n_t, CHAMP_K, alpha="auto"
            )
        scores.append(loglikelihood_continuation(logits, ids, prompt_len))
    pred = ex["choices"][int(np.argmax(scores))].strip().lower()
    return pred == gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-bench", type=int, default=int(os.environ.get("R29_N", "24")))
    ap.add_argument("--benches", type=str, default=",".join(BENCHES_DEFAULT))
    args = ap.parse_args()
    benches = [b.strip() for b in args.benches.split(",") if b.strip()]

    strategies = ["Baseline", "Champion"]
    strategies += [f"PA1_{s}" for s in SINGLE_SIGNALS]
    strategies += ["B1_6sig", "B3_consensus"]

    meta = {
        "phase": "R29_phase1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "n_per_bench": args.n_per_bench,
        "benches": benches,
        "strategies": strategies,
        "model": MODEL_PATH,
    }
    print(json.dumps(meta, indent=2))

    data = load_benchmarks(benches, args.n_per_bench)
    if not data:
        return

    tok, model = load_model_eager()
    n_layers = model.config.num_hidden_layers

    per_sample: list[dict] = []
    summary: dict[str, dict[str, dict]] = {}  # bench -> strat -> stats

    t0 = time.time()
    for bench, examples in data.items():
        summary[bench] = {s: {"correct": 0, "n": 0, "t_starts": [], "t_stops": [], "fallback": 0} for s in strategies}
        for i, ex in enumerate(examples):
            pref = tok(ex["prompt"], return_tensors="pt", add_special_tokens=False)
            input_ids = pref["input_ids"].to(DEVICE)
            attn = pref.get("attention_mask")
            attn = attn.to(DEVICE) if attn is not None else None
            try:
                sig = collect_probe_signals(model, input_ids, attn, n_layers)
            except Exception as e:
                print(f"  probe fail {bench}#{i}: {e}")
                continue

            row = {"benchmark": bench, "sample_index": i, "strategies": {}}
            for strat in strategies:
                if strat == "Baseline":
                    ok = score_mc(tok, model, ex, None, None, False)
                    ts, te, fb = None, None, False
                else:
                    ts, te, fb = bounds_for_strategy(strat, sig, n_layers)
                    ok = score_mc(tok, model, ex, ts, te, True)
                row["strategies"][strat] = {
                    "correct": bool(ok),
                    "t_start": ts,
                    "t_stop": te,
                    "n_t": (te - ts) if ts is not None and te is not None else None,
                    "fallback": fb,
                }
                st = summary[bench][strat]
                st["correct"] += int(ok)
                st["n"] += 1
                if ts is not None:
                    st["t_starts"].append(ts)
                    st["t_stops"].append(te)
                if fb:
                    st["fallback"] += 1

            per_sample.append(row)
            if (i + 1) % 8 == 0:
                print(f"  {bench} {i+1}/{len(examples)} elapsed {time.time()-t0:.0f}s")

    meta["elapsed_sec"] = time.time() - t0
    meta["finished_utc"] = datetime.now(timezone.utc).isoformat()

    # aggregate accuracies
    acc_table = {}
    for bench in summary:
        acc_table[bench] = {}
        for strat, st in summary[bench].items():
            n = max(st["n"], 1)
            acc_table[bench][strat] = st["correct"] / n
            st["accuracy"] = st["correct"] / n
            st["t_start_mean"] = float(np.mean(st["t_starts"])) if st["t_starts"] else None
            st["t_stop_mean"] = float(np.mean(st["t_stops"])) if st["t_stops"] else None
            st["t_start_std"] = float(np.std(st["t_starts"])) if len(st["t_starts"]) > 1 else 0.0
            st["t_stop_std"] = float(np.std(st["t_stops"])) if len(st["t_stops"]) > 1 else 0.0
            del st["t_starts"]
            del st["t_stops"]

    out_path = os.path.join(RESULTS_DIR, "round29_phase1_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summary, "per_sample": per_sample, "accuracies": acc_table}, f, indent=2)
    print(f"Wrote {out_path}")

    plot_accuracy_bars(acc_table, strategies, meta)
    plot_boundary_scatter(per_sample, "B1_6sig", meta)
    print(f"Figures under {FIGURES_DIR}")


def plot_accuracy_bars(acc_table: dict, strategies: list[str], meta: dict):
    benches = list(acc_table.keys())
    fig, axes = plt.subplots(1, len(benches), figsize=(4 * len(benches), 5))
    if len(benches) == 1:
        axes = [axes]
    for ax, bench in zip(axes, benches):
        bl = acc_table[bench].get("Baseline", 0)
        x = np.arange(len(strategies))
        heights = [acc_table[bench].get(s, 0) - bl for s in strategies]
        colors = ["gray" if s == "Baseline" else "steelblue" for s in strategies]
        ax.bar(x, heights, color=colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("PA1_", "") for s in strategies], rotation=75, ha="right", fontsize=7)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(bench)
        ax.set_ylabel("Accuracy minus Baseline")
    fig.suptitle("R29 Phase 1: strategy accuracy delta vs Baseline", fontsize=12)
    plt.tight_layout()
    p = os.path.join(FIGURES_DIR, "r29_phase1_accuracy_delta.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {p}")


def plot_boundary_scatter(per_sample: list[dict], strat: str, meta: dict):
    benches = sorted(set(s["benchmark"] for s in per_sample))
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.cm.tab10(np.linspace(0, 0.9, len(benches)))
    for bi, bench in enumerate(benches):
        xs, ys = [], []
        for s in per_sample:
            if s["benchmark"] != bench:
                continue
            info = s["strategies"].get(strat, {})
            if info.get("t_start") is None:
                continue
            xs.append(info["t_start"])
            ys.append(info["t_stop"])
        if xs:
            ax.scatter(xs, ys, alpha=0.45, s=25, c=[cmap[bi]], label=bench)
    ax.scatter([8], [22], s=220, marker="*", c="black", zorder=5, label="Champion (8,22)")
    ax.set_xlabel("t_start (layer)")
    ax.set_ylabel("t_stop (layer)")
    ax.set_title(f"Detected boundaries: {strat}")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 36)
    ax.set_ylim(0, 36)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    p = os.path.join(FIGURES_DIR, "r29_phase1_boundaries_B1_6sig.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {p}")


if __name__ == "__main__":
    main()
