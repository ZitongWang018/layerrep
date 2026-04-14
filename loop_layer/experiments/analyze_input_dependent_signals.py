#!/usr/bin/env python3
"""
Input-Dependent Signal Analysis for ETD Optimal Boundaries.

Core shift: instead of looking at signal ABSOLUTE VALUES (architecture-inherent),
look at signal DEVIATIONS from the global mean (input-dependent).

Key analyses:
  1. Per-layer inter-sample variance profile: which layers are "input-sensitive"?
  2. Per-sample deviation vectors: does deviation pattern differ across benchmarks?
  3. Deviation-based features vs R30 optimal t_start correlation.
  4. "Input sensitivity depth" — how deep do input-driven differences penetrate?
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path("/root/autodl-tmp/loop_layer")
EXP = ROOT / "experiments"
for p in (str(ROOT), str(EXP)):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

RESULTS_DIR = EXP / "results"
FIGURES_DIR = EXP / "figures" / "input_dependent_analysis"
RAW_DATA_PATH = RESULTS_DIR / "signal_raw_all_per_sample.json"

BENCHMARKS = ["ARC-C", "TruthfulQA", "CSQA", "MMLU-HS-Math"]
R30_OPTIMAL = {
    "ARC-C":        {"t_start": 14, "t_stop": 20},
    "TruthfulQA":   {"t_start": 16, "t_stop": 19},
    "CSQA":         {"t_start": 10, "t_stop": 22},
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
    with open(RAW_DATA_PATH) as f:
        raw = json.load(f)
    with open(RESULTS_DIR / "r30_sweep_results.json") as f:
        sweep = json.load(f)
    return raw, sweep


def compute_per_layer_stats(raw):
    """
    For each benchmark × signal, compute per-layer:
      - mean, std, CV across samples
      - individual sample deviation from global mean (all benchmarks pooled)
    """
    n_layers = len(raw[BENCHMARKS[0]][0][SIGNALS[0]])

    # Global mean per signal per layer (pooled across all benchmarks)
    global_mean = {}
    for sk in SIGNALS:
        all_vals = []
        for bench in BENCHMARKS:
            for sample in raw[bench]:
                all_vals.append(sample[sk])
        stacked = np.array(all_vals, dtype=np.float64)
        global_mean[sk] = np.nanmean(stacked, axis=0)

    # Per-benchmark stats
    bench_stats = {}
    for bench in BENCHMARKS:
        samples = raw[bench]
        stats = {}
        for sk in SIGNALS:
            vals = np.array([s[sk] for s in samples], dtype=np.float64)
            mean = np.nanmean(vals, axis=0)
            std = np.nanstd(vals, axis=0)
            cv = std / (np.abs(mean) + 1e-12)
            # Deviation from global mean
            dev_from_global = mean - global_mean[sk]
            stats[sk] = {
                "mean": mean,
                "std": std,
                "cv": cv,
                "dev_from_global": dev_from_global,
                "per_sample_vals": vals,
            }
        bench_stats[bench] = stats

    return bench_stats, global_mean, n_layers


def plot_variance_profile(bench_stats, n_layers):
    """Plot 1: Inter-sample std/CV vs layer for each signal, colored by benchmark."""
    layers = np.arange(n_layers)
    n_sigs = len(SIGNALS)
    n_cols = 6
    n_rows = (n_sigs + n_cols - 1) // n_cols

    # --- STD plot ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes = axes.flatten()
    colors = {"ARC-C": "C0", "TruthfulQA": "C1", "CSQA": "C2", "MMLU-HS-Math": "C3"}
    for ax_i, sk in enumerate(SIGNALS):
        ax = axes[ax_i]
        for bench in BENCHMARKS:
            std = bench_stats[bench][sk]["std"]
            ax.plot(layers, std, color=colors[bench], linewidth=1.5, label=bench)
            # Mark optimal t_start
            t0 = R30_OPTIMAL[bench]["t_start"]
            ax.axvline(t0, color=colors[bench], ls=":", lw=0.8, alpha=0.6)
        ax.set_title(sk, fontsize=8)
        ax.set_xlabel("layer", fontsize=7)
        ax.set_ylabel("std (inter-sample)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)
        if ax_i == 0:
            ax.legend(fontsize=5, loc="upper right")
    for k in range(len(SIGNALS), len(axes)):
        axes[k].axis("off")
    fig.suptitle("Inter-sample STD per layer: where are inputs distinguishable?", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "variance_std_by_layer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote variance_std_by_layer.png")

    # --- CV plot ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes = axes.flatten()
    for ax_i, sk in enumerate(SIGNALS):
        ax = axes[ax_i]
        for bench in BENCHMARKS:
            cv = bench_stats[bench][sk]["cv"]
            ax.plot(layers, cv, color=colors[bench], linewidth=1.5, label=bench)
            t0 = R30_OPTIMAL[bench]["t_start"]
            ax.axvline(t0, color=colors[bench], ls=":", lw=0.8, alpha=0.6)
        ax.set_title(sk, fontsize=8)
        ax.set_xlabel("layer", fontsize=7)
        ax.set_ylabel("CV", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)
        if ax_i == 0:
            ax.legend(fontsize=5, loc="upper right")
    for k in range(len(SIGNALS), len(axes)):
        axes[k].axis("off")
    fig.suptitle("Inter-sample CV per layer: relative input sensitivity", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "variance_cv_by_layer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote variance_cv_by_layer.png")


def plot_deviation_from_global(bench_stats, global_mean, n_layers):
    """Plot 2: Benchmark-mean deviation from global-mean, per signal per layer."""
    layers = np.arange(n_layers)
    n_sigs = len(SIGNALS)
    n_cols = 6
    n_rows = (n_sigs + n_cols - 1) // n_cols
    colors = {"ARC-C": "C0", "TruthfulQA": "C1", "CSQA": "C2", "MMLU-HS-Math": "C3"}

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes = axes.flatten()
    for ax_i, sk in enumerate(SIGNALS):
        ax = axes[ax_i]
        gm = global_mean[sk]
        for bench in BENCHMARKS:
            dev = bench_stats[bench][sk]["dev_from_global"]
            # Normalize by global std for comparability
            gstd = np.nanstd(
                np.concatenate([bench_stats[b][sk]["mean"].reshape(1, -1)
                               for b in BENCHMARKS], axis=0),
                axis=0
            )
            normalized_dev = dev / (gstd + 1e-12)
            ax.plot(layers, normalized_dev, color=colors[bench], linewidth=1.5, label=bench)
            t0 = R30_OPTIMAL[bench]["t_start"]
            ax.axvline(t0, color=colors[bench], ls=":", lw=0.8, alpha=0.6)
        ax.axhline(0, color="gray", ls="-", lw=0.5)
        ax.set_title(sk, fontsize=8)
        ax.set_xlabel("layer", fontsize=7)
        ax.set_ylabel("normalized dev", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)
        if ax_i == 0:
            ax.legend(fontsize=5, loc="upper right")
    for k in range(len(SIGNALS), len(axes)):
        axes[k].axis("off")
    fig.suptitle(
        "Benchmark-mean deviation from global-mean (normalized)\n"
        "If benchmarks cluster differently → input matters", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "deviation_from_global.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote deviation_from_global.png")


def compute_input_sensitivity_depth(bench_stats, n_layers):
    """
    For each signal, find the "input sensitivity depth": the layer beyond which
    inter-sample std drops below a threshold (e.g., 10% of max std).
    """
    results = {}
    for sk in SIGNALS:
        per_bench = {}
        for bench in BENCHMARKS:
            std = bench_stats[bench][sk]["std"]
            max_std = np.nanmax(std)
            if max_std < 1e-12 or np.isnan(max_std):
                per_bench[bench] = 0
                continue
            threshold = 0.1 * max_std
            # Find the LAST layer where std > threshold
            above = np.where(std > threshold)[0]
            depth = int(above[-1]) if len(above) > 0 else 0
            per_bench[bench] = depth
        results[sk] = per_bench
    return results


def compute_per_sample_features(raw, bench_stats, global_mean, n_layers):
    """
    For each sample, compute input-dependent features:
    1. Early-layer deviation magnitude (L0-L6)
    2. Mid-layer deviation magnitude (L7-L17)
    3. Deviation slope (how fast does deviation decay with depth)
    4. Peak deviation layer
    These are features that VARY per sample (unlike the mean signal).
    """
    sample_features = {}
    for bench in BENCHMARKS:
        samples = raw[bench]
        features_list = []
        for si, sample in enumerate(samples):
            feats = {}
            for sk in SIGNALS:
                vals = np.array(sample[sk], dtype=np.float64)
                gm = global_mean[sk]
                dev = vals - gm

                # Feature 1: Early-layer deviation RMS (layers 0-6)
                early_dev = dev[:7]
                feats[f"{sk}_early_dev_rms"] = float(np.sqrt(np.nanmean(early_dev ** 2)))

                # Feature 2: Mid-layer deviation RMS (layers 7-17)
                mid_dev = dev[7:18]
                feats[f"{sk}_mid_dev_rms"] = float(np.sqrt(np.nanmean(mid_dev ** 2)))

                # Feature 3: Late-layer deviation RMS (layers 18-35)
                late_dev = dev[18:]
                feats[f"{sk}_late_dev_rms"] = float(np.sqrt(np.nanmean(late_dev ** 2)))

                # Feature 4: Deviation decay ratio (mid/early)
                early_rms = feats[f"{sk}_early_dev_rms"]
                mid_rms = feats[f"{sk}_mid_dev_rms"]
                feats[f"{sk}_dev_decay"] = mid_rms / (early_rms + 1e-12)

                # Feature 5: Peak absolute deviation layer
                abs_dev = np.abs(dev)
                feats[f"{sk}_peak_dev_layer"] = int(np.nanargmax(abs_dev))

                # Feature 6: Signal value itself at key layers
                for layer_idx in [5, 8, 10, 14, 16, 20]:
                    if layer_idx < n_layers:
                        feats[f"{sk}_val_L{layer_idx}"] = float(vals[layer_idx])

            features_list.append(feats)
        sample_features[bench] = features_list
    return sample_features


def plot_input_sensitivity_depth(sens_depth):
    """Heatmap: signal × benchmark → depth of input sensitivity."""
    matrix = np.zeros((len(SIGNALS), len(BENCHMARKS)))
    for si, sk in enumerate(SIGNALS):
        for bi, bench in enumerate(BENCHMARKS):
            matrix[si, bi] = sens_depth[sk][bench]

    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(BENCHMARKS)))
    ax.set_xticklabels(BENCHMARKS, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(SIGNALS)))
    ax.set_yticklabels(SIGNALS, fontsize=8)
    for si in range(len(SIGNALS)):
        for bi in range(len(BENCHMARKS)):
            ax.text(bi, si, f"{int(matrix[si, bi])}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, label="Deepest layer with std > 10% of max std")
    ax.set_title("Input Sensitivity Depth: how deep do input differences penetrate?", fontsize=11)

    # Mark optimal t_start for reference
    for bi, bench in enumerate(BENCHMARKS):
        t0 = R30_OPTIMAL[bench]["t_start"]
        ax.text(bi, -1.5, f"t*={t0}", ha="center", fontsize=8, color="red", fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "input_sensitivity_depth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote input_sensitivity_depth.png")


def benchmark_separability_analysis(bench_stats, n_layers):
    """
    At each layer, can we distinguish which benchmark a sample came from
    using signal values? Compute between-benchmark variance / within-benchmark variance
    (like a 1-way ANOVA F-ratio).
    """
    layers = np.arange(n_layers)
    f_ratios = {}
    for sk in SIGNALS:
        group_means = []
        within_vars = []
        for bench in BENCHMARKS:
            vals = bench_stats[bench][sk]["per_sample_vals"]  # [N_samples, L]
            group_means.append(np.nanmean(vals, axis=0))
            within_vars.append(np.nanvar(vals, axis=0))

        group_means = np.array(group_means)  # [4, L]
        within_vars = np.array(within_vars)  # [4, L]
        grand_mean = np.nanmean(group_means, axis=0)  # [L]

        between_var = np.nanmean((group_means - grand_mean) ** 2, axis=0)
        within_var = np.nanmean(within_vars, axis=0)
        f_ratio = between_var / (within_var + 1e-12)
        f_ratios[sk] = f_ratio

    # Plot F-ratio by layer
    n_cols = 6
    n_rows = (len(SIGNALS) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes = axes.flatten()
    for ax_i, sk in enumerate(SIGNALS):
        ax = axes[ax_i]
        f = f_ratios[sk]
        ax.bar(layers, f, color="steelblue", alpha=0.7, width=0.8)
        ax.set_title(sk, fontsize=8)
        ax.set_xlabel("layer", fontsize=7)
        ax.set_ylabel("F-ratio", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2, axis="y")
        # Mark all optimal t_starts
        for bench in BENCHMARKS:
            t0 = R30_OPTIMAL[bench]["t_start"]
            ax.axvline(t0, color="red", ls=":", lw=0.7, alpha=0.5)
    for k in range(len(SIGNALS), len(axes)):
        axes[k].axis("off")
    fig.suptitle(
        "Between-benchmark / Within-benchmark variance ratio (F-ratio) per layer\n"
        "High F = benchmarks are distinguishable at that layer", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "f_ratio_by_layer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote f_ratio_by_layer.png")

    return f_ratios


def deviation_correlation_with_sweep(sample_features, n_layers):
    """
    For each input-dependent feature, compute Spearman correlation with
    the benchmark's optimal t_start.
    Since we have 80 samples (20 × 4 benchmarks) each with a known optimal t_start,
    we can compute: does the sample's deviation feature predict its optimal t_start?
    """
    # Collect (feature_value, optimal_t_start) pairs across all samples
    all_t_starts = []
    all_feats_by_key = defaultdict(list)

    for bench in BENCHMARKS:
        t0 = R30_OPTIMAL[bench]["t_start"]
        for sample_feats in sample_features[bench]:
            all_t_starts.append(t0)
            for fk, fv in sample_feats.items():
                all_feats_by_key[fk].append(fv)

    all_t_starts = np.array(all_t_starts, dtype=np.float64)

    results = {}
    for fk in sorted(all_feats_by_key.keys()):
        fv = np.array(all_feats_by_key[fk], dtype=np.float64)
        mask = ~(np.isnan(fv) | np.isnan(all_t_starts))
        if mask.sum() >= 10:
            rho, p = spearmanr(fv[mask], all_t_starts[mask])
            results[fk] = {"rho": float(rho), "p": float(p)}

    return results


def write_comprehensive_report(bench_stats, global_mean, sens_depth,
                                f_ratios, dev_corr, n_layers):
    """Write detailed analysis report."""
    lines = []
    lines.append("# Input-Dependent Signal Analysis Report\n\n")

    # Section 1: Input Sensitivity Depth
    lines.append("## 1. Input Sensitivity Depth\n\n")
    lines.append("For each signal, the deepest layer where inter-sample std > 10% of max std:\n\n")
    lines.append("| Signal | " + " | ".join(BENCHMARKS) + " |\n")
    lines.append("|--------|" + "|".join(["--------"] * len(BENCHMARKS)) + "|\n")
    for sk in SIGNALS:
        row = f"| {sk} |"
        for bench in BENCHMARKS:
            d = sens_depth[sk][bench]
            row += f" {d} |"
        lines.append(row + "\n")
    ref_parts = [f"{b}={R30_OPTIMAL[b]['t_start']}" for b in BENCHMARKS]
    lines.append(f"\nOptimal t_start for reference: {', '.join(ref_parts)}\n\n")

    # Section 2: Peak F-ratio layers
    lines.append("## 2. Benchmark Separability (F-ratio)\n\n")
    lines.append("Top-3 layers per signal where benchmarks are most distinguishable:\n\n")
    lines.append("| Signal | Peak F-ratio Layers | Peak F values |\n")
    lines.append("|--------|--------------------|--------------|\n")
    for sk in SIGNALS:
        f = f_ratios[sk]
        top3_idx = np.argsort(f)[-3:][::-1]
        top3_vals = f[top3_idx]
        layers_str = ", ".join([str(i) for i in top3_idx])
        vals_str = ", ".join([f"{v:.3f}" for v in top3_vals])
        lines.append(f"| {sk} | {layers_str} | {vals_str} |\n")
    lines.append("\n")

    # Section 3: Deviation features correlated with optimal t_start
    lines.append("## 3. Input-Dependent Features vs Optimal t_start\n\n")
    lines.append("Spearman correlation of per-sample deviation features with optimal t_start:\n\n")

    # Sort by |rho|
    sorted_feats = sorted(dev_corr.items(), key=lambda x: -abs(x[1]["rho"]))
    lines.append("### Top 30 features by |ρ|\n\n")
    lines.append("| Feature | ρ | p-value | Significant? |\n")
    lines.append("|---------|-----|---------|-------------|\n")
    for fk, vals in sorted_feats[:30]:
        rho = vals["rho"]
        p = vals["p"]
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        lines.append(f"| {fk} | {rho:+.3f} | {p:.4f} | {sig} |\n")
    lines.append("\n")

    # Section 4: Key insights
    lines.append("## 4. Key Observations\n\n")

    # Count significant features
    sig_features = [(k, v) for k, v in sorted_feats if v["p"] < 0.05]
    sig_strong = [(k, v) for k, v in sorted_feats if v["p"] < 0.01]
    lines.append(f"- Total features tested: {len(dev_corr)}\n")
    lines.append(f"- Features with p < 0.05: {len(sig_features)}\n")
    lines.append(f"- Features with p < 0.01: {len(sig_strong)}\n")
    if sig_features:
        lines.append(f"- Expected false positives at p<0.05 by chance: ~{len(dev_corr)*0.05:.0f}\n")
    lines.append("\n")

    report_path = RESULTS_DIR / "input_dependent_analysis_report.md"
    with open(report_path, "w") as f:
        f.writelines(lines)
    print(f"  Wrote {report_path}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading raw signal data...")
    raw, sweep = load_data()

    print("Computing per-layer statistics...")
    bench_stats, global_mean, n_layers = compute_per_layer_stats(raw)

    print("Computing input sensitivity depth...")
    sens_depth = compute_input_sensitivity_depth(bench_stats, n_layers)

    print("Plotting variance profiles...")
    plot_variance_profile(bench_stats, n_layers)

    print("Plotting deviation from global mean...")
    plot_deviation_from_global(bench_stats, global_mean, n_layers)

    print("Plotting input sensitivity depth...")
    plot_input_sensitivity_depth(sens_depth)

    print("Computing benchmark separability (F-ratio)...")
    f_ratios = benchmark_separability_analysis(bench_stats, n_layers)

    print("Computing per-sample input-dependent features...")
    sample_features = compute_per_sample_features(raw, bench_stats, global_mean, n_layers)

    print("Correlating deviation features with optimal t_start...")
    dev_corr = deviation_correlation_with_sweep(sample_features, n_layers)

    print("Writing report...")
    write_comprehensive_report(bench_stats, global_mean, sens_depth,
                                f_ratios, dev_corr, n_layers)

    # Save deviation correlation results
    with open(RESULTS_DIR / "input_dependent_correlations.json", "w") as f:
        json.dump(dev_corr, f, indent=2)

    print(f"\n=== DONE ===")
    print(f"Figures: {FIGURES_DIR}")
    print(f"Report: {RESULTS_DIR / 'input_dependent_analysis_report.md'}")


if __name__ == "__main__":
    main()
