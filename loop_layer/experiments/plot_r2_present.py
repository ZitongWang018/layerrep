#!/usr/bin/env python3
"""
Presentation figures for R2 (damped ETD): minimal on-canvas text; full data, emphasis by alpha/size.
Reads: r2_full_validate.json, r2_c2_results.json, r2_generalize_results.json, n8_validate_results.json
Writes: experiments/figures/r2_present_*.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.use("Agg")

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# --- style: little ink for labels ---
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def load(name: str) -> dict:
    with open(RES / name, encoding="utf-8") as f:
        return json.load(f)


def fig1_full_validate() -> None:
    d = load("r2_full_validate.json")
    bl_b = d["baseline"]["boolq"]
    bl_a = d["baseline"]["arc"]
    order = [
        ("orig_opt", "(10,21) α=1"),
        ("r2_07_opt", "(10,21) α=0.7"),
        ("r2_05_opt", "(10,21) α=0.5"),
        ("r2_05_valley", "(10,24) α=0.5"),
        ("r2_03_valley", "(10,24) α=0.3"),
        ("arc_opt_orig", "(12,20) α=1"),
    ]
    highlight = "r2_05_opt"

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)

    for ax, task, bl in zip(axes, ["boolq", "arc"], [bl_b, bl_a]):
        xs = np.arange(len(order))
        heights = []
        colors = []
        zorders = []
        for key, _ in order:
            r = d["results"][key][task]["acc"]
            heights.append(r)
            is_hi = key == highlight
            colors.append("#2d6a8f" if is_hi else "#c5d4dc")
            zorders.append(10 if is_hi else 2)
        for i, (h, c, z) in enumerate(zip(heights, colors, zorders)):
            ax.bar(i, h, color=c, edgecolor="white", linewidth=0.8, zorder=z, width=0.72)
            if order[i][0] == highlight:
                ax.bar(
                    i,
                    h,
                    fill=False,
                    edgecolor="#e6a800",
                    linewidth=2.4,
                    zorder=11,
                    width=0.72,
                )
        ax.axhline(bl, color="#222", ls=(0, (4, 3)), lw=2.0, zorder=1)
        ax.set_xticks(xs)
        ax.set_xticklabels([lab for _, lab in order], rotation=28, ha="right")
        ax.set_ylabel("acc")
        ax.set_title("BoolQ" if task == "boolq" else "ARC")
        ax.set_ylim(bl - 0.12, max(heights + [bl]) + 0.06)
        ax.grid(axis="y", alpha=0.35)

    fig.suptitle("R2 · N=500", fontsize=12, fontweight="bold", y=1.02)
    out = FIG / "r2_present_fullvalidate.png"
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(out)


def fig2_alpha_sweeps() -> None:
    d = load("r2_c2_results.json")
    R = d["experiment_R2"]
    panels = [
        ("optimal_boolq", "(10,21)"),
        ("valley", "(10,24)"),
        ("optimal_arc", "(12,20)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9), sharey=False, constrained_layout=True)

    for ax, (pkey, title) in zip(axes, panels):
        block = R[pkey]
        keys_bq = block["boolq"]["accs_by_alpha"]
        alphas = sorted(float(k) for k in keys_bq.keys())

        def _ak(a: float) -> str:
            for k in keys_bq.keys():
                if abs(float(k) - a) < 1e-9:
                    return k
            return str(a)

        bq = [keys_bq[_ak(a)] for a in alphas]
        ar = [block["arc"]["accs_by_alpha"][_ak(a)] for a in alphas]

        ax.plot(alphas, bq, "-", color="#94a3b8", lw=1.2, alpha=0.55, zorder=1)
        ax.plot(alphas, ar, "-", color="#cbd5e1", lw=1.2, alpha=0.55, zorder=1)
        ax.scatter(alphas, bq, s=28, color="#94a3b8", alpha=0.45, zorder=2, edgecolors="none")
        ax.scatter(alphas, ar, s=28, color="#cbd5e1", alpha=0.45, zorder=2, edgecolors="none")

        ba = block["boolq"]["best_alpha"]
        bb = block["boolq"]["best_acc"]
        ax.scatter([ba], [bb], s=130, color="#1d4ed8", zorder=5, edgecolors="white", linewidths=1.2)

        aa = block["arc"]["best_alpha"]
        ab = block["arc"]["best_acc"]
        ax.scatter([aa], [ab], s=130, color="#ea580c", zorder=5, edgecolors="white", linewidths=1.2)

        ax.set_xlabel("α")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("acc")
    fig.text(0.01, 0.02, "N=100 · ● BoolQ peak  · ● ARC peak", fontsize=8, color="0.35")
    out = FIG / "r2_present_alpha_sweeps.png"
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(out)


def fig3_adaptive() -> None:
    d = load("r2_generalize_results.json")
    cfg = d["configs"]
    keys = list(cfg.keys())
    x = np.arange(len(keys))
    w = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 5.2), sharex=True, constrained_layout=True)

    for ax, task in zip(axes, ["boolq", "arc"]):
        o = [cfg[k][f"{task}_orig"] for k in keys]
        a = [cfg[k][f"{task}_adaptive"] for k in keys]
        ax.bar(x - w / 2, o, w, label="α=1", color="#d8dee9", edgecolor="none")
        ax.bar(x + w / 2, a, w, label="α=min(1,6/nₜ)", color="#88c0d0", edgecolor="none", alpha=0.85)
        bl = d["baseline"][task]
        ax.axhline(bl, color="#2e3440", ls=(0, (4, 3)), lw=1.5, alpha=0.85)
        ax.set_ylabel("acc")
        ax.set_title("BoolQ" if task == "boolq" else "ARC")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="lower right", frameon=False)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(keys, rotation=35, ha="right", fontsize=7)
    fig.suptitle("R2 adaptive α · N=100", fontsize=12, fontweight="bold", y=1.01)
    out = FIG / "r2_present_adaptive_generalize.png"
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(out)


def fig4_n8() -> None:
    d = load("n8_validate_results.json")
    bl_b = d["baseline"]["boolq"]
    bl_a = d["baseline"]["arc"]
    keys = list(d["configs"].keys())
    x = np.arange(len(keys))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)

    for ax, task, bl in zip(axes, ["boolq", "arc"], [bl_b, bl_a]):
        hs = [d["configs"][k][task]["acc"] for k in keys]
        cols = ["#b8c5d0"] * len(keys)
        if "n14_large_8_22" in keys:
            cols[keys.index("n14_large_8_22")] = "#5e81ac"
        ax.bar(x, hs, color=cols, edgecolor="white", linewidth=0.6)
        ax.axhline(bl, color="#222", ls=(0, (4, 3)), lw=1.8)
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=22, ha="right", fontsize=7)
        ax.set_ylabel("acc")
        ax.set_title("BoolQ" if task == "boolq" else "ARC")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("R2 · eff.step · N=500", fontsize=12, fontweight="bold", y=1.02)
    out = FIG / "r2_present_n8_validate.png"
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(out)


def fig5_combined_slide() -> None:
    """One 16:9 canvas: left = full validate; right top = alpha (small); right bottom = adaptive strip."""
    d = load("r2_full_validate.json")
    bl_b, bl_a = d["baseline"]["boolq"], d["baseline"]["arc"]
    order = [
        ("orig_opt", "α1"),
        ("r2_07_opt", "α.7"),
        ("r2_05_opt", "α.5"),
        ("r2_05_valley", "v.5"),
        ("r2_03_valley", "v.3"),
        ("arc_opt_orig", "12-20"),
    ]
    highlight = "r2_05_opt"

    fig = plt.figure(figsize=(13.5, 7.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.0, 0.85], wspace=0.22, hspace=0.35)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    for ax, task, bl in zip([ax0, ax1], ["boolq", "arc"], [bl_b, bl_a]):
        xs = np.arange(len(order))
        for i, (key, _) in enumerate(order):
            h = d["results"][key][task]["acc"]
            c = "#3b82f6" if key == highlight else "#e2e8f0"
            ax.bar(i, h, width=0.75, color=c, edgecolor="white", linewidth=0.5)
            if key == highlight:
                ax.bar(i, h, width=0.75, fill=False, edgecolor="#f59e0b", linewidth=2.0)
        ax.axhline(bl, color="#0f172a", lw=1.8, ls=(0, (5, 4)))
        ax.set_ylabel("acc")
        ax.grid(axis="y", alpha=0.25)
        ax.set_title("BoolQ" if task == "boolq" else "ARC", loc="left", fontsize=10)
    ax1.set_xticks(np.arange(len(order)))
    ax1.set_xticklabels([l for _, l in order])

    axa = fig.add_subplot(gs[:, 1])
    d2 = load("r2_c2_results.json")["experiment_R2"]["optimal_boolq"]
    keys_bq = d2["boolq"]["accs_by_alpha"]
    alphas = sorted(float(k) for k in keys_bq.keys())

    def _ak2(a: float) -> str:
        for k in keys_bq.keys():
            if abs(float(k) - a) < 1e-9:
                return k
        return str(a)

    bq = [keys_bq[_ak2(a)] for a in alphas]
    ar = [d2["arc"]["accs_by_alpha"][_ak2(a)] for a in alphas]
    axa.plot(alphas, bq, "-", color="#93c5fd", lw=2, alpha=0.45)
    axa.plot(alphas, ar, "-", color="#fdba74", lw=2, alpha=0.45)
    axa.scatter(alphas, bq, s=36, color="#93c5fd", alpha=0.35)
    axa.scatter(alphas, ar, s=36, color="#fdba74", alpha=0.35)
    ba, bb = d2["boolq"]["best_alpha"], d2["boolq"]["best_acc"]
    aa, ab = d2["arc"]["best_alpha"], d2["arc"]["best_acc"]
    axa.scatter([ba], [bb], s=140, color="#1d4ed8", zorder=5, edgecolor="white")
    axa.scatter([aa], [ab], s=140, color="#c2410c", zorder=5, edgecolor="white")
    axa.set_xlabel("α")
    axa.set_ylabel("acc")
    axa.set_title("(10,21) α-scan", fontweight="bold")
    axa.grid(True, alpha=0.25)

    fig.suptitle("R2 damped ETD", fontsize=14, fontweight="bold", y=0.98)
    fig.text(0.02, 0.02, "Left N=500  ·  Right N=100  ·  Gold = (10,21) α=0.5", fontsize=8, color="0.4")

    out = FIG / "r2_present_slide_combo.png"
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(out)


def main() -> None:
    fig1_full_validate()
    fig2_alpha_sweeps()
    fig3_adaptive()
    fig4_n8()
    fig5_combined_slide()
    print("Done.")


if __name__ == "__main__":
    main()
