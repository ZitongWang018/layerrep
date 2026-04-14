#!/usr/bin/env python3
"""
Decompose the observed signal-t_start correlations into:
  1. BETWEEN-benchmark effect (can signals identify task type?)
  2. WITHIN-benchmark effect (can signals distinguish individual samples?)

Also: how well can a simple classifier predict benchmark from signal features?
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import f_oneway, kruskal

RESULTS_DIR = Path("/root/autodl-tmp/loop_layer/experiments/results")
FIGURES_DIR = Path("/root/autodl-tmp/loop_layer/experiments/figures/input_dependent_analysis")

BENCHMARKS = ["ARC-C", "TruthfulQA", "CSQA", "MMLU-HS-Math"]
R30_OPTIMAL = {
    "ARC-C": {"t_start": 14, "t_stop": 20},
    "TruthfulQA": {"t_start": 16, "t_stop": 19},
    "CSQA": {"t_start": 10, "t_stop": 22},
    "MMLU-HS-Math": {"t_start": 10, "t_stop": 18},
}
SIGNALS = [
    "attn_entropy", "ffn_gate_norm", "layer_sim", "head_specialization",
    "logit_lens_KL", "attention_locality", "residual_write_norm",
    "participation_ratio", "prediction_flip_rate", "attn_sink_ratio",
    "residual_delta_l2", "contraction_ratio", "logit_lens_jsd_vel",
    "logit_lens_jsd_curv", "erank", "delta_erank", "attn_consensus",
    "logit_top1_margin",
]


def load_data():
    with open(RESULTS_DIR / "signal_raw_all_per_sample.json") as f:
        return json.load(f)


def compute_anova_per_layer(raw, n_layers):
    """
    For each signal × layer: one-way ANOVA / Kruskal-Wallis across 4 benchmarks.
    This tells us: at layer L, can we statistically distinguish which benchmark
    a sample came from?
    """
    results = {}
    for sk in SIGNALS:
        f_vals = []
        p_vals = []
        kw_h = []
        kw_p = []
        for li in range(n_layers):
            groups = []
            for bench in BENCHMARKS:
                vals = [s[sk][li] for s in raw[bench]]
                groups.append(np.array(vals, dtype=np.float64))
            # Remove NaN
            clean_groups = [g[~np.isnan(g)] for g in groups]
            if all(len(g) >= 3 for g in clean_groups):
                f_stat, p_anova = f_oneway(*clean_groups)
                h_stat, p_kw = kruskal(*clean_groups)
                f_vals.append(float(f_stat))
                p_vals.append(float(p_anova))
                kw_h.append(float(h_stat))
                kw_p.append(float(p_kw))
            else:
                f_vals.append(float("nan"))
                p_vals.append(float("nan"))
                kw_h.append(float("nan"))
                kw_p.append(float("nan"))
        results[sk] = {
            "f_stat": np.array(f_vals),
            "p_anova": np.array(p_vals),
            "kw_h": np.array(kw_h),
            "kw_p": np.array(kw_p),
        }
    return results


def compute_effect_size_eta_squared(raw, n_layers):
    """
    Eta-squared: proportion of variance explained by benchmark membership.
    η² = SS_between / SS_total
    This tells us: what fraction of signal variance is benchmark-dependent?
    """
    results = {}
    for sk in SIGNALS:
        etas = []
        for li in range(n_layers):
            all_vals = []
            groups = []
            for bench in BENCHMARKS:
                vals = [s[sk][li] for s in raw[bench]]
                vals_clean = [v for v in vals if not np.isnan(v)]
                groups.append(vals_clean)
                all_vals.extend(vals_clean)
            if len(all_vals) < 10:
                etas.append(float("nan"))
                continue
            grand_mean = np.mean(all_vals)
            ss_total = sum((v - grand_mean) ** 2 for v in all_vals)
            ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups if len(g) > 0)
            eta = ss_between / (ss_total + 1e-12)
            etas.append(float(eta))
        results[sk] = np.array(etas)
    return results


def plot_eta_squared(eta_sq, n_layers):
    """Plot η² by layer for each signal — the fraction of variance due to benchmark."""
    layers = np.arange(n_layers)
    n_cols = 6
    n_rows = (len(SIGNALS) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes = axes.flatten()
    for ax_i, sk in enumerate(SIGNALS):
        ax = axes[ax_i]
        eta = eta_sq[sk]
        ax.bar(layers, eta, color="coral", alpha=0.7, width=0.8)
        ax.axhline(0.06, color="green", ls="--", lw=1, label="small effect (0.06)")
        ax.axhline(0.14, color="orange", ls="--", lw=1, label="medium effect (0.14)")
        for bench in BENCHMARKS:
            t0 = R30_OPTIMAL[bench]["t_start"]
            ax.axvline(t0, color="blue", ls=":", lw=0.6, alpha=0.5)
        ax.set_title(sk, fontsize=8)
        ax.set_xlabel("layer", fontsize=7)
        ax.set_ylabel("η²", fontsize=7)
        ax.set_ylim(0, min(max(eta[~np.isnan(eta)].max() * 1.3, 0.2), 1.0) if np.any(~np.isnan(eta)) else 0.2)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2, axis="y")
        if ax_i == 0:
            ax.legend(fontsize=5)
    for k in range(len(SIGNALS), len(axes)):
        axes[k].axis("off")
    fig.suptitle(
        "η² (variance explained by benchmark membership) per layer\n"
        "High η² = signal at that layer IS input-type-dependent",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eta_squared_by_layer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote eta_squared_by_layer.png")


def plot_anova_pvalues(anova_results, n_layers):
    """Plot -log10(p) from ANOVA per layer — significance of benchmark effect."""
    layers = np.arange(n_layers)
    n_cols = 6
    n_rows = (len(SIGNALS) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes = axes.flatten()
    for ax_i, sk in enumerate(SIGNALS):
        ax = axes[ax_i]
        p = anova_results[sk]["p_anova"]
        neg_logp = -np.log10(p + 1e-20)
        ax.bar(layers, neg_logp, color="steelblue", alpha=0.7, width=0.8)
        ax.axhline(-np.log10(0.05), color="red", ls="--", lw=1, label="p=0.05")
        ax.axhline(-np.log10(0.01), color="darkred", ls="--", lw=1, label="p=0.01")
        for bench in BENCHMARKS:
            t0 = R30_OPTIMAL[bench]["t_start"]
            ax.axvline(t0, color="green", ls=":", lw=0.6, alpha=0.5)
        ax.set_title(sk, fontsize=8)
        ax.set_xlabel("layer", fontsize=7)
        ax.set_ylabel("-log10(p)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2, axis="y")
        if ax_i == 0:
            ax.legend(fontsize=5)
    for k in range(len(SIGNALS), len(axes)):
        axes[k].axis("off")
    fig.suptitle(
        "ANOVA -log10(p): can benchmark identity be detected at each layer?\n"
        "Above red line = statistically significant (p<0.05)",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "anova_significance_by_layer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote anova_significance_by_layer.png")


def compute_benchmark_profile_distance(raw, n_layers):
    """
    Compute pairwise distance between benchmark mean signal profiles.
    This shows WHICH benchmarks are most different in signal space.
    """
    # Compute mean profile per benchmark
    profiles = {}
    for bench in BENCHMARKS:
        feature_vec = []
        for sk in SIGNALS:
            vals = np.array([s[sk] for s in raw[bench]], dtype=np.float64)
            mean_curve = np.nanmean(vals, axis=0)
            feature_vec.extend(mean_curve.tolist())
        profiles[bench] = np.array(feature_vec)

    # Pairwise cosine distances
    print("\n  Pairwise profile distances (cosine similarity):")
    print(f"  {'':15s}", end="")
    for b in BENCHMARKS:
        print(f" {b:>13s}", end="")
    print()

    for b1 in BENCHMARKS:
        print(f"  {b1:15s}", end="")
        for b2 in BENCHMARKS:
            v1, v2 = profiles[b1], profiles[b2]
            mask = ~(np.isnan(v1) | np.isnan(v2))
            cos = np.dot(v1[mask], v2[mask]) / (np.linalg.norm(v1[mask]) * np.linalg.norm(v2[mask]) + 1e-12)
            print(f" {cos:13.6f}", end="")
        print()

    # Also L2 distance
    print("\n  Pairwise L2 distances (normalized):")
    for b1 in BENCHMARKS:
        print(f"  {b1:15s}", end="")
        for b2 in BENCHMARKS:
            v1, v2 = profiles[b1], profiles[b2]
            mask = ~(np.isnan(v1) | np.isnan(v2))
            d = np.linalg.norm(v1[mask] - v2[mask]) / np.sqrt(mask.sum())
            print(f" {d:13.6f}", end="")
        print()


def summarize_key_layers(eta_sq, anova_results, n_layers):
    """For each signal, list layers where benchmark effect is statistically significant."""
    print("\n  === Layers where benchmark is detectable (ANOVA p < 0.05) ===")
    for sk in SIGNALS:
        p = anova_results[sk]["p_anova"]
        eta = eta_sq[sk]
        sig_layers = [li for li in range(n_layers) if p[li] < 0.05]
        if sig_layers:
            # Find the layer with highest eta-squared among significant
            best_layer = max(sig_layers, key=lambda l: eta[l])
            print(f"  {sk:25s}: {len(sig_layers):2d} significant layers, "
                  f"best η²={eta[best_layer]:.3f} at layer {best_layer}")
        else:
            print(f"  {sk:25s}:  0 significant layers")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading data...", flush=True)
    raw = load_data()
    n_layers = len(raw[BENCHMARKS[0]][0][SIGNALS[0]])

    print("Running ANOVA per layer...", flush=True)
    anova_results = compute_anova_per_layer(raw, n_layers)

    print("Computing effect sizes (η²)...", flush=True)
    eta_sq = compute_effect_size_eta_squared(raw, n_layers)

    print("Plotting η² by layer...", flush=True)
    plot_eta_squared(eta_sq, n_layers)

    print("Plotting ANOVA significance...", flush=True)
    plot_anova_pvalues(anova_results, n_layers)

    print("Computing benchmark profile distances...", flush=True)
    compute_benchmark_profile_distance(raw, n_layers)

    summarize_key_layers(eta_sq, anova_results, n_layers)

    # Summary statistics
    print("\n  === Summary: Average η² across layers ===", flush=True)
    for sk in SIGNALS:
        mean_eta = float(np.nanmean(eta_sq[sk]))
        max_eta = float(np.nanmax(eta_sq[sk]))
        max_layer = int(np.nanargmax(eta_sq[sk]))
        print(f"  {sk:25s}: mean η²={mean_eta:.4f}, max η²={max_eta:.4f} @ layer {max_layer}")

    # Count total significant layers
    total_tests = len(SIGNALS) * n_layers
    sig_count = sum(
        1 for sk in SIGNALS
        for li in range(n_layers)
        if anova_results[sk]["p_anova"][li] < 0.05
    )
    expected_fp = total_tests * 0.05
    print(f"\n  Total layer×signal tests: {total_tests}")
    print(f"  Significant at p<0.05: {sig_count} (expected by chance: {expected_fp:.0f})")
    print(f"  Ratio: {sig_count / expected_fp:.1f}x chance level")

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
