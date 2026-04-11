#!/usr/bin/env python3
"""
Comprehensive curve-plot figures for ETD Layer Sweep results.
Produces ONE figure per benchmark (BoolQ / ARC), each containing
all sweep results with clear baseline comparison.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent
CSV     = ROOT / "artifacts" / "sweep_results.csv"
FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── load data ───────────────────────────────────────────────────────────────
df = pd.read_csv(CSV)
df["n_t"] = df["t_end"] - df["t_start"] + 1   # thinking-block depth
df["delta_k2"] = df["boolq_k2"] - df["boolq_baseline"]
df["delta_k3"] = df["boolq_k3"] - df["boolq_baseline"]
df["arc_delta_k2"] = df["arc_k2"] - df["arc_baseline"]
df["arc_delta_k3"] = df["arc_k3"] - df["arc_baseline"]

T_STARTS = sorted(df["t_start"].unique())
T_ENDS   = sorted(df["t_end"].unique())

BASELINE_BOOLQ = df["boolq_baseline"].iloc[0]
BASELINE_ARC   = df["arc_baseline"].iloc[0]

# colour maps: one colour per t_start / t_end
cmap_start = cm.get_cmap("tab10", len(T_STARTS))
cmap_end   = cm.get_cmap("Set1",  len(T_ENDS))


# ── helpers ─────────────────────────────────────────────────────────────────
def draw_baseline(ax, value, label="Baseline", color="black",
                  lw=2.0, ls="--", zorder=10):
    ax.axhline(value, color=color, lw=lw, ls=ls, label=label, zorder=zorder)


def annotate_best(ax, xs, ys, marker="*", ms=14, color="gold", zorder=15):
    best_i = int(np.argmax(ys))
    ax.plot(xs[best_i], ys[best_i], marker=marker, ms=ms,
            color=color, zorder=zorder, markeredgecolor="black", markeredgewidth=0.8)


def finalize(ax, title, xlabel, ylabel, legend=True, ylim_pad=0.02,
             y_center=None, y_half=None):
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.3, ls=":")
    if legend:
        ax.legend(fontsize=6.5, ncol=2, loc="lower right",
                  framealpha=0.8, edgecolor="0.7")
    if y_center is not None and y_half is not None:
        ax.set_ylim(y_center - y_half, y_center + y_half)


# ════════════════════════════════════════════════════════════════════════════
#  UNIVERSAL FIGURE BUILDER
# ════════════════════════════════════════════════════════════════════════════
def make_figure(bench: str) -> None:
    """
    bench: 'boolq' or 'arc'
    Generates a 4×3 subplot figure with:
      Row-0  Acc vs t_end  (lines = t_start),  k=2 | k=3 | overlay k2+k3
      Row-1  Acc vs t_start(lines = t_end  ),  k=2 | k=3 | overlay k2+k3
      Row-2  Delta (ETD−baseline) vs t_end / vs t_start / vs n_t (scatter)
      Row-3  Per-t_start aggregates | per-t_end aggregates | summary bar
    """
    k2_col  = f"{bench}_k2"
    k3_col  = f"{bench}_k3"
    bl_col  = f"{bench}_baseline"
    d2_col  = f"{bench}_delta_k2" if bench == "arc" else "delta_k2"
    d3_col  = f"{bench}_delta_k3" if bench == "arc" else "delta_k3"
    BASELINE = BASELINE_BOOLQ if bench == "boolq" else BASELINE_ARC
    bench_label = "BoolQ" if bench == "boolq" else "ARC-Challenge"
    y_lo = 0.20 if bench == "boolq" else 0.15
    y_hi = 1.00 if bench == "boolq" else 0.60

    fig, axes = plt.subplots(4, 3, figsize=(18, 22))
    fig.suptitle(
        f"{bench_label}  —  Full ETD Layer Sweep (Qwen3-8B, 36 layers)\n"
        f"Baseline = {BASELINE:.4f}   |   ★ = global best in each panel",
        fontsize=13, fontweight="bold", y=0.995,
    )

    # ── ROW 0: Accuracy vs t_end, lines=t_start ─────────────────────────
    for col_idx, (metric_col, k_label) in enumerate(
        [(k2_col, "k=2"), (k3_col, "k=3")]
    ):
        ax = axes[0, col_idx]
        all_y = []
        for si, ts in enumerate(T_STARTS):
            sub = df[df["t_start"] == ts].sort_values("t_end")
            xs  = sub["t_end"].values
            ys  = sub[metric_col].values
            ax.plot(xs, ys, marker="o", ms=4, lw=1.5,
                    color=cmap_start(si), label=f"t_start={ts}")
            all_y.extend(ys.tolist())
        draw_baseline(ax, BASELINE)
        annotate_best(ax, df.sort_values("t_end")["t_end"].values,
                      [df[df["t_end"]==te][metric_col].max() for te in sorted(df["t_end"].unique())])
        finalize(ax,
                 f"Acc vs t_end  ({k_label})\nEach line = fixed t_start",
                 "t_end", "Accuracy")
        ax.set_ylim(y_lo, y_hi)

    # Overlay: k=2 vs k=3 mean across t_start
    ax = axes[0, 2]
    for metric_col, k_label, color in [(k2_col, "k=2 (mean)", "steelblue"),
                                        (k3_col, "k=3 (mean)", "tomato")]:
        means = df.groupby("t_end")[metric_col].mean()
        stds  = df.groupby("t_end")[metric_col].std()
        xs    = means.index.values
        ax.plot(xs, means.values, marker="o", ms=5, lw=2, color=color, label=k_label)
        ax.fill_between(xs, means-stds, means+stds, alpha=0.15, color=color)
        annotate_best(ax, xs, means.values)
    draw_baseline(ax, BASELINE)
    finalize(ax,
             "Mean Acc vs t_end  (k=2 vs k=3)\nShaded = ±1 std across t_starts",
             "t_end", "Accuracy")
    ax.set_ylim(y_lo, y_hi)

    # ── ROW 1: Accuracy vs t_start, lines=t_end ─────────────────────────
    for col_idx, (metric_col, k_label) in enumerate(
        [(k2_col, "k=2"), (k3_col, "k=3")]
    ):
        ax = axes[1, col_idx]
        for ei, te in enumerate(T_ENDS):
            sub = df[df["t_end"] == te].sort_values("t_start")
            xs  = sub["t_start"].values
            ys  = sub[metric_col].values
            ax.plot(xs, ys, marker="s", ms=4, lw=1.5,
                    color=cmap_end(ei), label=f"t_end={te}")
        draw_baseline(ax, BASELINE)
        annotate_best(ax,
                      [ts for ts in sorted(df["t_start"].unique())],
                      [df[df["t_start"]==ts][metric_col].max() for ts in sorted(df["t_start"].unique())])
        finalize(ax,
                 f"Acc vs t_start  ({k_label})\nEach line = fixed t_end",
                 "t_start", "Accuracy")
        ax.set_ylim(y_lo, y_hi)
        # mark phase transition
        ax.axvline(6.5, color="red", lw=1.2, ls=":", alpha=0.7, label="Phase boundary t_start=7")

    # Overlay: mean k=2 vs k=3 across t_end
    ax = axes[1, 2]
    for metric_col, k_label, color in [(k2_col, "k=2 (mean)", "steelblue"),
                                        (k3_col, "k=3 (mean)", "tomato")]:
        means = df.groupby("t_start")[metric_col].mean()
        stds  = df.groupby("t_start")[metric_col].std()
        xs    = means.index.values
        ax.plot(xs, means.values, marker="s", ms=5, lw=2, color=color, label=k_label)
        ax.fill_between(xs, means-stds, means+stds, alpha=0.15, color=color)
    draw_baseline(ax, BASELINE)
    ax.axvline(6.5, color="red", lw=1.2, ls=":", alpha=0.7, label="Phase boundary (t_start=7)")
    finalize(ax,
             "Mean Acc vs t_start  (k=2 vs k=3)\nShaded = ±1 std across t_ends",
             "t_start", "Accuracy")
    ax.set_ylim(y_lo, y_hi)

    # ── ROW 2: Delta (ETD − Baseline) ───────────────────────────────────
    # [2,0] delta vs t_end
    ax = axes[2, 0]
    for metric_col, k_label, color in [(d2_col, "Δ k=2", "steelblue"),
                                        (d3_col, "Δ k=3", "tomato")]:
        means = df.groupby("t_end")[metric_col].mean()
        stds  = df.groupby("t_end")[metric_col].std()
        xs    = means.index.values
        ax.plot(xs, means.values, marker="o", ms=5, lw=2, color=color, label=k_label)
        ax.fill_between(xs, means-stds, means+stds, alpha=0.15, color=color)
    ax.axhline(0, color="black", lw=1.5, ls="--", label="Baseline (Δ=0)")
    finalize(ax,
             "Mean Δ Acc (ETD−baseline) vs t_end\nPositive = ETD better",
             "t_end", "Δ Accuracy (ETD − baseline)")
    ax.set_ylim(-0.35, 0.10)

    # [2,1] delta vs t_start
    ax = axes[2, 1]
    for metric_col, k_label, color in [(d2_col, "Δ k=2", "steelblue"),
                                        (d3_col, "Δ k=3", "tomato")]:
        means = df.groupby("t_start")[metric_col].mean()
        stds  = df.groupby("t_start")[metric_col].std()
        xs    = means.index.values
        ax.plot(xs, means.values, marker="s", ms=5, lw=2, color=color, label=k_label)
        ax.fill_between(xs, means-stds, means+stds, alpha=0.15, color=color)
    ax.axhline(0, color="black", lw=1.5, ls="--", label="Baseline (Δ=0)")
    ax.axvline(6.5, color="red", lw=1.2, ls=":", alpha=0.7, label="Phase boundary")
    finalize(ax,
             "Mean Δ Acc (ETD−baseline) vs t_start\nPhase transition at t_start≈7",
             "t_start", "Δ Accuracy (ETD − baseline)")
    ax.set_ylim(-0.35, 0.10)

    # [2,2] delta vs n_t (thinking block depth)
    ax = axes[2, 2]
    for metric_col, k_label, color, marker in [
        (d2_col, "Δ k=2", "steelblue", "o"),
        (d3_col, "Δ k=3", "tomato", "s"),
    ]:
        sub_ok = df[df["t_start"] >= 7]  # only stable zone
        means = sub_ok.groupby("n_t")[metric_col].mean()
        stds  = sub_ok.groupby("n_t")[metric_col].std()
        xs    = means.index.values
        ax.plot(xs, means.values, marker=marker, ms=5, lw=2, color=color, label=k_label)
        ax.fill_between(xs, means-stds, means+stds, alpha=0.15, color=color)
    ax.axhline(0, color="black", lw=1.5, ls="--", label="Baseline (Δ=0)")
    finalize(ax,
             "Mean Δ Acc vs n_t (T-block depth)\n[stable zone: t_start≥7 only]",
             "n_t = t_end − t_start + 1", "Δ Accuracy (ETD − baseline)")
    ax.set_ylim(-0.35, 0.10)

    # ── ROW 3: Aggregated summaries ──────────────────────────────────────
    # [3,0]  Per-t_start: max k=2, max k=3, vs baseline bar chart
    ax = axes[3, 0]
    x_pos = np.arange(len(T_STARTS))
    width = 0.3
    max_k2 = [df[df["t_start"]==ts][k2_col].max() for ts in T_STARTS]
    max_k3 = [df[df["t_start"]==ts][k3_col].max() for ts in T_STARTS]
    bars1 = ax.bar(x_pos - width/2, max_k2, width, color="steelblue",
                   label="Best k=2", alpha=0.8, edgecolor="white")
    bars2 = ax.bar(x_pos + width/2, max_k3, width, color="tomato",
                   label="Best k=3", alpha=0.8, edgecolor="white")
    ax.axhline(BASELINE, color="black", lw=2, ls="--", label=f"Baseline={BASELINE:.3f}")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(ts) for ts in T_STARTS], fontsize=8)
    finalize(ax,
             "Best ETD Acc per t_start\n(max over all t_end)",
             "t_start", "Best Accuracy")
    ax.set_ylim(y_lo, y_hi)

    # [3,1]  Per-t_end: max k=2, max k=3
    ax = axes[3, 1]
    x_pos = np.arange(len(T_ENDS))
    max_k2 = [df[df["t_end"]==te][k2_col].max() for te in T_ENDS]
    max_k3 = [df[df["t_end"]==te][k3_col].max() for te in T_ENDS]
    ax.bar(x_pos - width/2, max_k2, width, color="steelblue",
           label="Best k=2", alpha=0.8, edgecolor="white")
    ax.bar(x_pos + width/2, max_k3, width, color="tomato",
           label="Best k=3", alpha=0.8, edgecolor="white")
    ax.axhline(BASELINE, color="black", lw=2, ls="--", label=f"Baseline={BASELINE:.3f}")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(te) for te in T_ENDS], fontsize=8)
    finalize(ax,
             "Best ETD Acc per t_end\n(max over all t_start)",
             "t_end", "Best Accuracy")
    ax.set_ylim(y_lo, y_hi)

    # [3,2]  Distribution: histogram of all ETD results vs baseline
    ax = axes[3, 2]
    all_k2 = df[k2_col].values
    all_k3 = df[k3_col].values
    bins = np.linspace(y_lo, y_hi, 30)
    ax.hist(all_k2, bins=bins, alpha=0.55, color="steelblue",
            label=f"k=2  (n={len(all_k2)})", edgecolor="white", lw=0.3)
    ax.hist(all_k3, bins=bins, alpha=0.55, color="tomato",
            label=f"k=3  (n={len(all_k3)})", edgecolor="white", lw=0.3)
    ax.axvline(BASELINE, color="black", lw=2.5, ls="--",
               label=f"Baseline = {BASELINE:.3f}")
    ax.set_xlabel("Accuracy", fontsize=9)
    ax.set_ylabel("Count (cells)", fontsize=9)
    ax.set_title(
        "Distribution of ETD Accuracies (all 99 cells)\nvs. fixed baseline",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=8, framealpha=0.8)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.3, ls=":")

    # ── shared colour-bar legend for t_start lines ───────────────────────
    sm_start = cm.ScalarMappable(
        cmap=cmap_start,
        norm=mcolors.BoundaryNorm(
            boundaries=[ts - 0.5 for ts in T_STARTS] + [T_STARTS[-1] + 0.5],
            ncolors=len(T_STARTS),
        ),
    )
    sm_start.set_array([])
    cbar = fig.colorbar(sm_start, ax=axes[0, :2], location="right",
                        fraction=0.012, pad=0.02, shrink=0.85)
    cbar.set_label("t_start value (Row-0 lines)", fontsize=8)
    cbar.set_ticks(T_STARTS)

    sm_end = cm.ScalarMappable(
        cmap=cmap_end,
        norm=mcolors.BoundaryNorm(
            boundaries=[te - 0.5 for te in T_ENDS] + [T_ENDS[-1] + 0.5],
            ncolors=len(T_ENDS),
        ),
    )
    sm_end.set_array([])
    cbar2 = fig.colorbar(sm_end, ax=axes[1, :2], location="right",
                         fraction=0.012, pad=0.02, shrink=0.85)
    cbar2.set_label("t_end value (Row-1 lines)", fontsize=8)
    cbar2.set_ticks(T_ENDS)

    plt.tight_layout(rect=[0, 0, 0.97, 0.995])
    out = FIGURES / f"comprehensive_{bench}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ════════════════════════════════════════════════════════════════════════════
#  Run
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    make_figure("boolq")
    make_figure("arc")
    print("Done.")
