#!/usr/bin/env python3
"""
Line-chart overview for layer sweep (BoolQ + ARC).

Each PNG: two panels (k=2, k=3). X = t_end; one colored curve per t_start (11 curves, 99 points each).
Baseline: thick dashed line + legend + text box with numeric value.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "artifacts" / "sweep_results.csv"
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_line_panel(
    ax,
    df: pd.DataFrame,
    acc_col: str,
    baseline: float,
    ts_sorted: list,
    title: str,
) -> None:
    """Accuracy vs t_end; one line per t_start. Baseline emphasized."""
    cmap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=min(ts_sorted), vmax=max(ts_sorted))

    all_y = []
    for tss in ts_sorted:
        sub = df[df["t_start"] == tss].sort_values("t_end")
        xs = sub["t_end"].values
        ys = sub[acc_col].values
        all_y.extend(ys.tolist())
        color = cmap(norm(tss))
        # slightly mute curves that stay clearly below baseline (optional visual hint)
        below = np.max(ys) < baseline - 0.005
        alpha = 0.38 if below else 0.88
        lw = 1.1 if below else 1.75
        ax.plot(
            xs,
            ys,
            marker="o",
            ms=3.8,
            lw=lw,
            color=color,
            alpha=alpha,
            zorder=2,
        )

    # Baseline — very clear
    ax.axhline(
        baseline,
        color="#1a1a1a",
        lw=2.8,
        ls=(0, (5, 4)),
        zorder=15,
        label=f"Baseline (no ETD): {baseline:.3f}",
    )
    ax.fill_between(
        [df["t_end"].min() - 0.5, df["t_end"].max() + 0.5],
        baseline - 0.0015,
        baseline + 0.0015,
        color="#1a1a1a",
        alpha=0.06,
        zorder=1,
    )

    # Best point
    best_row = df.loc[df[acc_col].idxmax()]
    bx = int(best_row["t_end"])
    by = float(best_row[acc_col])
    b_ts = int(best_row["t_start"])
    ax.plot(
        bx,
        by,
        marker="*",
        ms=15,
        color="#e6b800",
        markeredgecolor="0.2",
        markeredgewidth=0.9,
        zorder=20,
        label=f"Best ETD: ({b_ts},{bx}) = {by:.3f}",
    )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("t_end", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_xticks(sorted(df["t_end"].unique()))
    ax.grid(True, axis="y", alpha=0.35, ls=":")
    ax.grid(True, axis="x", alpha=0.2, ls=":")

    ymin = min(min(all_y), baseline) - 0.04
    ymax = max(max(all_y), baseline) + 0.04
    ax.set_ylim(ymin, ymax)

    # Baseline value box (redundant on purpose — easy to read)
    ax.text(
        0.02,
        0.98,
        f"Baseline = {baseline:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        family="sans-serif",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#333", linewidth=1.2, alpha=0.96),
        zorder=25,
    )

    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95, edgecolor="0.75")

    # Colorbar: which line is which t_start
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.055, pad=0.02)
    cbar.set_label("t_start (curve color)", fontsize=9)


def make_one_figure(
    k2_col: str,
    k3_col: str,
    baseline_col: str,
    out_name: str,
    bench_name: str,
) -> None:
    df = pd.read_csv(CSV)
    baseline = float(df[baseline_col].iloc[0])
    ts_sorted = sorted(df["t_start"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    fig.suptitle(bench_name, fontsize=14, fontweight="bold", y=1.02)

    draw_line_panel(axes[0], df, k2_col, baseline, ts_sorted, "k = 2")
    draw_line_panel(axes[1], df, k3_col, baseline, ts_sorted, "k = 3")

    out = OUT_DIR / out_name
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    make_one_figure(
        "boolq_k2",
        "boolq_k3",
        "boolq_baseline",
        "sweep_overview_boolq.png",
        "BoolQ",
    )
    make_one_figure(
        "arc_k2",
        "arc_k3",
        "arc_baseline",
        "sweep_overview_arc.png",
        "ARC-Challenge",
    )


if __name__ == "__main__":
    main()
