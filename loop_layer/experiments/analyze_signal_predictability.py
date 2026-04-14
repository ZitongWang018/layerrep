#!/usr/bin/env python3
"""
Signal Predictability Analysis for ETD Optimal Boundaries.

Core question: Do per-layer signal features predict the optimal t_start/t_stop
as determined by R30 grid sweep?

Methodology:
  1. Collect all signals (R29 + proposed) in ONE probe pass; save raw data.
  2. For each signal × benchmark, compute Spearman correlation between
     signal_mean[layer] and best_acc(t_start=layer) from R30 sweep.
  3. Identify signal features (derivative peaks, threshold crossings)
     and check alignment with R30 optimal boundaries.
  4. Generate overlay plots: signal curve vs accuracy landscape.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

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
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

from exp_round27_main import load_benchmark
from r29.signal_funcs import (
    attn_entropy_from_weights as attn_entropy_r29,
    attention_locality_from_weights,
    attn_sink_ratio as attn_sink_ratio_fn,
    ffn_gate_norm as ffn_gate_norm_fn,
    head_specialization_from_weights,
    layer_cosine_sim,
    logit_lens_kl_last_token,
    participation_ratio as participation_ratio_fn,
    prediction_flip_rate_last_token,
    residual_write_norm as residual_write_norm_fn,
)
from proposed_signals_probe import (
    _jsd_probs,
    attention_consensus_index,
    effective_rank_svd_rows,
    logit_probs_last_token,
    logit_top1_margin_last,
    residual_delta_l2_mean,
)

R30_OPTIMAL = {
    "ARC-C":        {"t_start": 14, "t_stop": 20},
    "TruthfulQA":   {"t_start": 16, "t_stop": 19},
    "CSQA":         {"t_start": 10, "t_stop": 22},
    "MMLU-HS-Math": {"t_start": 10, "t_stop": 18},
}

ALL_SIGNAL_KEYS = [
    "attn_entropy", "ffn_gate_norm", "layer_sim", "head_specialization",
    "logit_lens_KL", "attention_locality", "residual_write_norm",
    "participation_ratio", "prediction_flip_rate", "attn_sink_ratio",
    "residual_delta_l2", "contraction_ratio", "logit_lens_jsd_vel",
    "logit_lens_jsd_curv", "erank", "delta_erank", "attn_consensus",
    "logit_top1_margin",
]

MODEL_PATH = os.environ.get("R29_MODEL_PATH", "/root/autodl-tmp/model_qwen")
FIGURES_DIR = EXP / "figures" / "signal_predictability"
RESULTS_DIR = EXP / "results"
DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_PER_BENCH = int(os.environ.get("PRED_N_PER_BENCH", "20"))
BENCHMARKS = list(R30_OPTIMAL.keys())
RAW_DATA_PATH = RESULTS_DIR / "signal_raw_all_per_sample.json"


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


def collect_all_signals(model, input_ids, attention_mask, n_layers, erank_max_tokens=64):
    """One forward pass → all R29 + proposed signals. Returns {layer: {signal: float}}."""
    base = model.model
    ln_f = base.norm
    lm_head = model.lm_head
    emb_dev = next(base.embed_tokens.parameters()).device
    input_ids = input_ids.to(emb_dev)
    if attention_mask is not None:
        attention_mask = attention_mask.to(emb_dev)

    hidden = [None] * n_layers
    attn_w = [None] * n_layers
    gate_a = [None] * n_layers
    hooks = []

    def hid_hook(li):
        def fn(_m, _inp, out):
            hidden[li] = (out[0] if isinstance(out, tuple) else out).detach()
        return fn

    def attn_hook(li):
        def fn(_m, _inp, out):
            if isinstance(out, tuple) and len(out) > 1:
                attn_w[li] = out[1].detach() if out[1] is not None else None
        return fn

    def gate_hook(li):
        def fn(_m, _inp, out):
            gate_a[li] = out.detach()
        return fn

    for li in range(n_layers):
        hooks.append(base.layers[li].register_forward_hook(hid_hook(li)))
        hooks.append(base.layers[li].self_attn.register_forward_hook(attn_hook(li)))
        hooks.append(base.layers[li].mlp.act_fn.register_forward_hook(gate_hook(li)))

    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    for h in hooks:
        h.remove()

    with torch.no_grad():
        inputs_embeds = base.embed_tokens(input_ids)
    h_final = hidden[n_layers - 1]
    if h_final is None:
        raise RuntimeError("No final hidden state")

    out: dict[int, dict[str, float]] = {}
    probs_cache: list[torch.Tensor | None] = [None] * n_layers
    delta_l2_list: list[float] = []

    for li in range(n_layers):
        h = hidden[li]
        if h is None:
            continue
        h_prev = inputs_embeds if li == 0 else hidden[li - 1]
        if h_prev is None:
            h_prev = inputs_embeds
        if h_prev.device != h.device:
            h_prev = h_prev.to(h.device)
        if h_final.device != h.device:
            h_final_use = h_final.to(h.device)
        else:
            h_final_use = h_final

        rec: dict[str, float] = {}

        # R29 signals
        rec["layer_sim"] = layer_cosine_sim(h, h_prev)
        rec["residual_write_norm"] = residual_write_norm_fn(h, h_prev)
        rec["participation_ratio"] = participation_ratio_fn(h)
        rec["logit_lens_KL"] = logit_lens_kl_last_token(h, h_final_use, ln_f, lm_head)
        rec["prediction_flip_rate"] = prediction_flip_rate_last_token(h, h_prev, ln_f, lm_head)

        aw = attn_w[li]
        if aw is not None:
            rec["attn_entropy"] = attn_entropy_r29(aw)
            rec["head_specialization"] = head_specialization_from_weights(aw)
            rec["attention_locality"] = attention_locality_from_weights(aw)
            rec["attn_sink_ratio"] = attn_sink_ratio_fn(aw, sink_idx=0)
            rec["attn_consensus"] = attention_consensus_index(aw)
        else:
            for k in ["attn_entropy", "head_specialization", "attention_locality",
                       "attn_sink_ratio", "attn_consensus"]:
                rec[k] = float("nan")

        ga = gate_a[li]
        rec["ffn_gate_norm"] = ffn_gate_norm_fn(ga) if ga is not None else float("nan")

        # Proposed signals
        d2 = residual_delta_l2_mean(h, h_prev)
        delta_l2_list.append(d2)
        rec["residual_delta_l2"] = d2
        rec["contraction_ratio"] = d2 / (delta_l2_list[li - 1] + 1e-12) if li >= 1 else float("nan")

        probs_cache[li] = logit_probs_last_token(h, ln_f, lm_head)
        rec["logit_top1_margin"] = logit_top1_margin_last(h, ln_f, lm_head)

        if li >= 1 and probs_cache[li - 1] is not None:
            rec["logit_lens_jsd_vel"] = float(
                _jsd_probs(probs_cache[li], probs_cache[li - 1]).mean().item()
            )
        else:
            rec["logit_lens_jsd_vel"] = float("nan")

        eranks = [effective_rank_svd_rows(h[bi], max_tokens=erank_max_tokens)
                  for bi in range(h.shape[0])]
        rec["erank"] = float(sum(eranks) / max(len(eranks), 1))

        out[li] = rec

    # Deltas needing previous layer
    prev_erank, prev_jsd = float("nan"), float("nan")
    for li in range(n_layers):
        rec = out[li]
        e = rec["erank"]
        rec["delta_erank"] = (e - prev_erank) if not math.isnan(prev_erank) else float("nan")
        prev_erank = e
        jv = rec["logit_lens_jsd_vel"]
        rec["logit_lens_jsd_curv"] = (jv - prev_jsd) if not math.isnan(prev_jsd) else float("nan")
        prev_jsd = jv

    return out


def collect_benchmark_raw(tok, model, bench, n_layers):
    """Return list[dict[str, list[float]]] — one dict per sample, each mapping signal→[L floats]."""
    examples = load_benchmark(bench, N_PER_BENCH)
    if len(examples) < N_PER_BENCH:
        print(f"  [WARN] {bench}: only {len(examples)} examples (requested {N_PER_BENCH})")
    results = []
    for i, ex in enumerate(examples[:N_PER_BENCH]):
        pref = tok(ex["prompt"], return_tensors="pt", add_special_tokens=False)
        input_ids = pref["input_ids"].to(DEVICE)
        attn = pref.get("attention_mask")
        attn = attn.to(DEVICE) if attn is not None else None
        try:
            sig = collect_all_signals(model, input_ids, attn, n_layers)
            row = {k: [] for k in ALL_SIGNAL_KEYS}
            for li in range(n_layers):
                d = sig.get(li, {})
                for k in ALL_SIGNAL_KEYS:
                    row[k].append(float(d.get(k, float("nan"))))
            results.append(row)
        except Exception as e:
            print(f"  [{bench} #{i}] error: {e}")
            continue
        if (i + 1) % 5 == 0:
            print(f"    {bench}: {i+1}/{min(N_PER_BENCH, len(examples))}")
    return results


def load_r30_sweep():
    """Load R30 sweep results and compute marginal accuracy landscapes."""
    with open(RESULTS_DIR / "r30_sweep_results.json") as f:
        sweep = json.load(f)

    best_by_tstart: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    best_by_tstop: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    all_accs: dict[str, dict[tuple, float]] = defaultdict(dict)

    for row in sweep["results"]:
        ts, te = row["t_start"], row["t_stop"]
        for bench in BENCHMARKS:
            if bench in row:
                acc = row[bench]
                if acc > best_by_tstart[bench][ts]:
                    best_by_tstart[bench][ts] = acc
                if acc > best_by_tstop[bench][te]:
                    best_by_tstop[bench][te] = acc
                all_accs[bench][(ts, te)] = acc

    return dict(best_by_tstart), dict(best_by_tstop), dict(all_accs)


def compute_mean_curves(raw_data):
    """raw_data: {bench: [sample_dicts]}. Returns {bench: {signal: np.array of L floats}}."""
    out = {}
    for bench, samples in raw_data.items():
        if not samples:
            continue
        n_layers = len(samples[0][ALL_SIGNAL_KEYS[0]])
        means = {}
        for k in ALL_SIGNAL_KEYS:
            stack = np.array([s[k] for s in samples], dtype=np.float64)
            means[k] = np.nanmean(stack, axis=0)
        out[bench] = means
    return out


def compute_correlations(mean_curves, best_by_tstart, best_by_tstop):
    """For each signal × benchmark, Spearman corr between signal[layer] and accuracy landscape."""
    results = {}
    for bench in BENCHMARKS:
        if bench not in mean_curves:
            continue
        curves = mean_curves[bench]
        tstart_map = best_by_tstart.get(bench, {})
        tstop_map = best_by_tstop.get(bench, {})

        for sig_key in ALL_SIGNAL_KEYS:
            sig_vals = curves[sig_key]

            # Correlation with t_start accuracy
            layers_ts = sorted(tstart_map.keys())
            if len(layers_ts) >= 5:
                x = np.array([sig_vals[l] for l in layers_ts if l < len(sig_vals)], dtype=np.float64)
                y = np.array([tstart_map[l] for l in layers_ts if l < len(sig_vals)], dtype=np.float64)
                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() >= 5:
                    rho_ts, p_ts = spearmanr(x[mask], y[mask])
                else:
                    rho_ts, p_ts = float("nan"), float("nan")
            else:
                rho_ts, p_ts = float("nan"), float("nan")

            # Correlation with t_stop accuracy
            layers_te = sorted(tstop_map.keys())
            if len(layers_te) >= 5:
                x2 = np.array([sig_vals[l] for l in layers_te if l < len(sig_vals)], dtype=np.float64)
                y2 = np.array([tstop_map[l] for l in layers_te if l < len(sig_vals)], dtype=np.float64)
                mask2 = ~(np.isnan(x2) | np.isnan(y2))
                if mask2.sum() >= 5:
                    rho_te, p_te = spearmanr(x2[mask2], y2[mask2])
                else:
                    rho_te, p_te = float("nan"), float("nan")
            else:
                rho_te, p_te = float("nan"), float("nan")

            results[(bench, sig_key)] = {
                "rho_tstart": float(rho_ts) if not np.isnan(rho_ts) else None,
                "p_tstart": float(p_ts) if not np.isnan(p_ts) else None,
                "rho_tstop": float(rho_te) if not np.isnan(rho_te) else None,
                "p_tstop": float(p_te) if not np.isnan(p_te) else None,
            }
    return results


def find_derivative_peaks(sig_mean, layers=None):
    """Find layers where the signal derivative has local maxima/minima."""
    if layers is None:
        layers = np.arange(len(sig_mean))
    deriv = np.gradient(sig_mean)
    peaks_pos = []
    peaks_neg = []
    for i in range(1, len(deriv) - 1):
        if deriv[i] > deriv[i-1] and deriv[i] > deriv[i+1] and deriv[i] > 0:
            peaks_pos.append(int(layers[i]))
        if deriv[i] < deriv[i-1] and deriv[i] < deriv[i+1] and deriv[i] < 0:
            peaks_neg.append(int(layers[i]))
    return {"deriv_peaks_positive": peaks_pos, "deriv_peaks_negative": peaks_neg}


def feature_alignment(mean_curves, n_layers):
    """For each signal × benchmark, check if derivative peaks align with optimal boundaries."""
    results = {}
    for bench in BENCHMARKS:
        if bench not in mean_curves:
            continue
        curves = mean_curves[bench]
        t0 = R30_OPTIMAL[bench]["t_start"]
        t1 = R30_OPTIMAL[bench]["t_stop"]

        for sig_key in ALL_SIGNAL_KEYS:
            sig = curves[sig_key]
            peaks = find_derivative_peaks(sig)

            deriv = np.gradient(sig)
            sig_at_t0 = float(sig[t0]) if t0 < n_layers else float("nan")
            sig_at_t1 = float(sig[t1]) if t1 < n_layers else float("nan")
            deriv_at_t0 = float(deriv[t0]) if t0 < n_layers else float("nan")
            deriv_at_t1 = float(deriv[t1]) if t1 < n_layers else float("nan")

            valid = sig[~np.isnan(sig)]
            if len(valid) > 0:
                pct_t0 = float(np.nanmean(sig <= sig_at_t0)) * 100 if not np.isnan(sig_at_t0) else None
                pct_t1 = float(np.nanmean(sig <= sig_at_t1)) * 100 if not np.isnan(sig_at_t1) else None
            else:
                pct_t0 = pct_t1 = None

            closest_peak_to_t0 = None
            all_peaks = peaks["deriv_peaks_positive"] + peaks["deriv_peaks_negative"]
            if all_peaks:
                dists = [abs(p - t0) for p in all_peaks]
                closest_peak_to_t0 = all_peaks[int(np.argmin(dists))]

            results[(bench, sig_key)] = {
                "t_start": t0,
                "t_stop": t1,
                "sig_at_tstart": sig_at_t0,
                "sig_at_tstop": sig_at_t1,
                "deriv_at_tstart": deriv_at_t0,
                "deriv_at_tstop": deriv_at_t1,
                "percentile_tstart": pct_t0,
                "percentile_tstop": pct_t1,
                "closest_deriv_peak_to_tstart": closest_peak_to_t0,
                "dist_closest_peak_to_tstart": abs(closest_peak_to_t0 - t0) if closest_peak_to_t0 is not None else None,
                "all_deriv_peaks": all_peaks,
            }
    return results


def plot_overlay(mean_curves, raw_data, best_by_tstart, n_layers):
    """For each benchmark × signal: overlay signal mean + accuracy landscape (dual y)."""
    for bench in BENCHMARKS:
        if bench not in mean_curves:
            continue
        curves = mean_curves[bench]
        tstart_map = best_by_tstart.get(bench, {})
        t0 = R30_OPTIMAL[bench]["t_start"]
        t1 = R30_OPTIMAL[bench]["t_stop"]
        layers = np.arange(n_layers)

        n_sigs = len(ALL_SIGNAL_KEYS)
        n_cols = 6
        n_rows = (n_sigs + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 3.5))
        axes = axes.flatten()

        for ax_i, sk in enumerate(ALL_SIGNAL_KEYS):
            ax = axes[ax_i]
            sig = curves[sk]

            # Individual sample curves
            samples = raw_data.get(bench, [])
            for s in samples:
                y = np.array(s.get(sk, [np.nan] * n_layers), dtype=np.float64)
                if y.shape[0] == n_layers:
                    ax.plot(layers, y, color="C0", alpha=0.12, linewidth=0.7)

            ax.plot(layers, sig, color="black", linewidth=2.0, label="signal mean")
            ax.axvline(t0, color="#2ca02c", ls="--", lw=1.5, label=f"t_start={t0}")
            ax.axvline(t1, color="#d62728", ls="--", lw=1.5, label=f"t_stop={t1}")
            ax.axvspan(t0, t1, alpha=0.08, color="gold", zorder=0)

            # Accuracy landscape on secondary axis
            ax2 = ax.twinx()
            ts_layers = sorted(tstart_map.keys())
            ts_accs = [tstart_map[l] for l in ts_layers]
            ax2.step(ts_layers, ts_accs, color="#ff7f0e", linewidth=1.8, alpha=0.7,
                     where="mid", label="best_acc(t_start)")
            ax2.set_ylabel("accuracy", fontsize=7, color="#ff7f0e")
            ax2.tick_params(axis="y", labelcolor="#ff7f0e", labelsize=6)

            ax.set_title(sk, fontsize=9)
            ax.set_xlabel("layer", fontsize=7)
            ax.set_xlim(-0.5, n_layers - 0.5)
            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=6)

            if ax_i == 0:
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, fontsize=5, loc="upper right")

        for k in range(len(ALL_SIGNAL_KEYS), len(axes)):
            axes[k].axis("off")

        fig.suptitle(
            f"{bench}: Signal Mean (black) vs Accuracy Landscape (orange step)\n"
            f"R30 optimal [{t0}, {t1}) | {len(raw_data.get(bench, []))} samples",
            fontsize=12
        )
        fig.tight_layout()
        out_path = FIGURES_DIR / f"overlay_{bench.replace(' ', '_')}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Wrote {out_path}")


def plot_correlation_heatmap(corr_results):
    """Heatmap of Spearman rho(signal, acc) across benchmarks."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    for ax, key, title in [(ax1, "rho_tstart", "Spearman ρ: signal[L] vs best_acc(t_start=L)"),
                           (ax2, "rho_tstop", "Spearman ρ: signal[L] vs best_acc(t_stop=L)")]:
        matrix = np.full((len(ALL_SIGNAL_KEYS), len(BENCHMARKS)), np.nan)
        for bi, bench in enumerate(BENCHMARKS):
            for si, sig in enumerate(ALL_SIGNAL_KEYS):
                v = corr_results.get((bench, sig), {}).get(key)
                if v is not None:
                    matrix[si, bi] = v

        im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(BENCHMARKS)))
        ax.set_xticklabels(BENCHMARKS, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(ALL_SIGNAL_KEYS)))
        ax.set_yticklabels(ALL_SIGNAL_KEYS, fontsize=7)
        ax.set_title(title, fontsize=10)

        for si in range(len(ALL_SIGNAL_KEYS)):
            for bi in range(len(BENCHMARKS)):
                v = matrix[si, bi]
                if not np.isnan(v):
                    p_val = corr_results.get((BENCHMARKS[bi], ALL_SIGNAL_KEYS[si]), {}).get(
                        f"p_{key.split('_')[1]}"
                    )
                    star = ""
                    if p_val is not None and p_val < 0.05:
                        star = "*"
                    if p_val is not None and p_val < 0.01:
                        star = "**"
                    txt = f"{v:.2f}{star}"
                    ax.text(bi, si, txt, ha="center", va="center", fontsize=6,
                            color="white" if abs(v) > 0.6 else "black")

    fig.colorbar(im, ax=[ax1, ax2], shrink=0.6, label="Spearman ρ")
    fig.suptitle("Signal ↔ R30 Accuracy Correlation (* p<0.05, ** p<0.01)", fontsize=12)
    fig.tight_layout()
    out_path = FIGURES_DIR / "correlation_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}")


def plot_scatter_tstart(mean_curves, best_by_tstart, n_layers):
    """For top candidate signals, scatter: signal[L] vs acc(t_start=L), colored by benchmark."""
    fig, axes = plt.subplots(3, 6, figsize=(24, 10))
    axes = axes.flatten()

    for ax_i, sk in enumerate(ALL_SIGNAL_KEYS):
        ax = axes[ax_i]
        for bi, bench in enumerate(BENCHMARKS):
            if bench not in mean_curves:
                continue
            sig = mean_curves[bench][sk]
            tstart_map = best_by_tstart.get(bench, {})
            ts_layers = sorted(tstart_map.keys())
            x_vals = [sig[l] for l in ts_layers if l < n_layers]
            y_vals = [tstart_map[l] for l in ts_layers if l < n_layers]
            mask = [not (np.isnan(xi) or np.isnan(yi)) for xi, yi in zip(x_vals, y_vals)]
            x_clean = [xi for xi, m in zip(x_vals, mask) if m]
            y_clean = [yi for yi, m in zip(y_vals, mask) if m]
            ax.scatter(x_clean, y_clean, s=12, alpha=0.7, label=bench if ax_i == 0 else None)

        ax.set_title(sk, fontsize=8)
        ax.set_xlabel("signal value", fontsize=7)
        ax.set_ylabel("accuracy", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)

    for k in range(len(ALL_SIGNAL_KEYS), len(axes)):
        axes[k].axis("off")

    axes[0].legend(fontsize=6, loc="lower right")
    fig.suptitle("Scatter: signal[layer] vs best_acc(t_start=layer), colored by benchmark", fontsize=12)
    fig.tight_layout()
    out_path = FIGURES_DIR / "scatter_signal_vs_accuracy.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}")


def write_report(corr_results, feat_align, n_layers):
    """Write markdown summary of findings."""
    lines = ["# Signal Predictability Analysis Report\n"]
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}\n")
    lines.append(f"N samples per benchmark: {N_PER_BENCH}\n")
    lines.append(f"Total signals analyzed: {len(ALL_SIGNAL_KEYS)}\n\n")

    lines.append("## R30 Optimal Configurations\n")
    for bench in BENCHMARKS:
        opt = R30_OPTIMAL[bench]
        lines.append(f"- **{bench}**: t_start={opt['t_start']}, t_stop={opt['t_stop']}\n")
    lines.append("\n")

    # Correlation table
    lines.append("## Spearman Correlation: signal[layer] vs best_acc(t_start=layer)\n\n")
    lines.append("| Signal | " + " | ".join(BENCHMARKS) + " | Mean |ρ| |\n")
    lines.append("|--------|" + "|".join(["--------"] * len(BENCHMARKS)) + "|--------|\n")

    sig_mean_abs_rho = {}
    for sk in ALL_SIGNAL_KEYS:
        row_vals = []
        abs_vals = []
        for bench in BENCHMARKS:
            r = corr_results.get((bench, sk), {})
            rho = r.get("rho_tstart")
            p = r.get("p_tstart")
            if rho is not None:
                star = ""
                if p is not None and p < 0.05:
                    star = "*"
                if p is not None and p < 0.01:
                    star = "**"
                row_vals.append(f"{rho:+.3f}{star}")
                abs_vals.append(abs(rho))
            else:
                row_vals.append("N/A")
        mean_abs = np.mean(abs_vals) if abs_vals else 0
        sig_mean_abs_rho[sk] = mean_abs
        lines.append(f"| {sk} | " + " | ".join(row_vals) + f" | {mean_abs:.3f} |\n")
    lines.append("\n*p<0.05, **p<0.01\n\n")

    # Top signals by mean |ρ|
    ranked = sorted(sig_mean_abs_rho.items(), key=lambda x: -x[1])
    lines.append("## Top Signals by Mean |ρ| (t_start correlation)\n\n")
    for i, (sk, v) in enumerate(ranked[:5]):
        lines.append(f"{i+1}. **{sk}**: mean |ρ| = {v:.3f}\n")
    lines.append("\n")

    # t_stop correlation table
    lines.append("## Spearman Correlation: signal[layer] vs best_acc(t_stop=layer)\n\n")
    lines.append("| Signal | " + " | ".join(BENCHMARKS) + " | Mean |ρ| |\n")
    lines.append("|--------|" + "|".join(["--------"] * len(BENCHMARKS)) + "|--------|\n")

    for sk in ALL_SIGNAL_KEYS:
        row_vals = []
        abs_vals = []
        for bench in BENCHMARKS:
            r = corr_results.get((bench, sk), {})
            rho = r.get("rho_tstop")
            p = r.get("p_tstop")
            if rho is not None:
                star = ""
                if p is not None and p < 0.05:
                    star = "*"
                if p is not None and p < 0.01:
                    star = "**"
                row_vals.append(f"{rho:+.3f}{star}")
                abs_vals.append(abs(rho))
            else:
                row_vals.append("N/A")
        mean_abs = np.mean(abs_vals) if abs_vals else 0
        lines.append(f"| {sk} | " + " | ".join(row_vals) + f" | {mean_abs:.3f} |\n")
    lines.append("\n")

    # Feature alignment summary
    lines.append("## Feature Alignment: Derivative Peaks vs Optimal Boundaries\n\n")
    lines.append("| Benchmark | Signal | Optimal t_start | Closest Peak | Distance |\n")
    lines.append("|-----------|--------|----------------|-------------|----------|\n")
    for bench in BENCHMARKS:
        for sk in ALL_SIGNAL_KEYS:
            fa = feat_align.get((bench, sk))
            if fa is None:
                continue
            cp = fa.get("closest_deriv_peak_to_tstart")
            dist = fa.get("dist_closest_peak_to_tstart")
            if cp is not None and dist is not None and dist <= 3:
                lines.append(
                    f"| {bench} | {sk} | {fa['t_start']} | {cp} | {dist} |\n"
                )
    lines.append("\n(Only showing signals with derivative peak within 3 layers of optimal t_start)\n\n")

    # Inter-sample variance analysis
    lines.append("## Conclusion\n\n")
    lines.append("See correlation heatmap and overlay plots for detailed visual analysis.\n")

    report_path = RESULTS_DIR / "signal_predictability_report.md"
    with open(report_path, "w") as f:
        f.writelines(lines)
    print(f"  Wrote {report_path}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Collect or load raw data
    if RAW_DATA_PATH.exists():
        print(f"Loading cached raw data from {RAW_DATA_PATH}")
        with open(RAW_DATA_PATH) as f:
            raw_data = json.load(f)
        n_layers = len(raw_data[BENCHMARKS[0]][0][ALL_SIGNAL_KEYS[0]])
    else:
        print("Collecting signals (all R29 + proposed) in single pass...")
        t0 = time.time()
        tok, model = load_model_eager()
        n_layers = model.config.num_hidden_layers
        print(f"n_layers={n_layers}")

        raw_data = {}
        for bench in BENCHMARKS:
            print(f"  Collecting {bench}...")
            raw_data[bench] = collect_benchmark_raw(tok, model, bench, n_layers)
            print(f"    → {len(raw_data[bench])} samples collected")

        elapsed = time.time() - t0
        print(f"Signal collection done in {elapsed:.1f}s")

        with open(RAW_DATA_PATH, "w") as f:
            json.dump(raw_data, f)
        print(f"  Saved raw data to {RAW_DATA_PATH}")

        del model, tok
        torch.cuda.empty_cache()

    # Step 2: Compute mean curves
    print("Computing mean curves...")
    mean_curves = compute_mean_curves(raw_data)

    # Step 3: Load R30 sweep
    print("Loading R30 sweep data...")
    best_by_tstart, best_by_tstop, all_accs = load_r30_sweep()

    # Step 4: Correlations
    print("Computing correlations...")
    corr_results = compute_correlations(mean_curves, best_by_tstart, best_by_tstop)

    # Step 5: Feature alignment
    n_layers_val = len(mean_curves[BENCHMARKS[0]][ALL_SIGNAL_KEYS[0]])
    print("Analyzing feature alignment...")
    feat_align = feature_alignment(mean_curves, n_layers_val)

    # Step 6: Plots
    print("Generating overlay plots...")
    plot_overlay(mean_curves, raw_data, best_by_tstart, n_layers_val)

    print("Generating correlation heatmap...")
    plot_correlation_heatmap(corr_results)

    print("Generating scatter plots...")
    plot_scatter_tstart(mean_curves, best_by_tstart, n_layers_val)

    # Step 7: Report
    print("Writing report...")
    write_report(corr_results, feat_align, n_layers_val)

    # Save correlation results as JSON for programmatic access
    corr_json = {}
    for (bench, sig), vals in corr_results.items():
        corr_json[f"{bench}|{sig}"] = vals
    with open(RESULTS_DIR / "signal_correlations.json", "w") as f:
        json.dump(corr_json, f, indent=2)

    print("\n=== DONE ===")
    print(f"Figures: {FIGURES_DIR}")
    print(f"Report:  {RESULTS_DIR / 'signal_predictability_report.md'}")


if __name__ == "__main__":
    main()
