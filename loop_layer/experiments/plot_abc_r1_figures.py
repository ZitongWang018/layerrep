#!/usr/bin/env python3
"""Generate figures for ABC+R1 mechanism experiments from abc_r1_results.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
JSON_PATH = ROOT / "results" / "abc_r1_results.json"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load() -> dict:
    with open(JSON_PATH, encoding="utf-8") as f:
        return json.load(f)


def plot_A(data: dict) -> None:
    """Hidden-state trajectory: delta_norm & cos_sim vs k (emphasis: k=2 exploration)."""
    A = data["experiment_A"]
    cfg = data["config"]
    k_list = A["boolq"]["k_range"]
    k_arr = np.array(k_list)

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True, constrained_layout=True)
    fig.suptitle(
        "Experiment A — Hidden-state trajectory (last token)\n"
        f"Optimal T-block: t_start={cfg['t_start_opt']}, t_end={cfg['t_end_opt']}  |  N={cfg['N_A']}/task",
        fontsize=11,
        fontweight="bold",
    )

    for ax_idx, task in enumerate(["boolq", "arc"]):
        ax = axes[ax_idx]
        ax2 = ax.twinx()
        t = task.upper()
        dn_m = np.array(A[task]["delta_norm_mean"])
        dn_s = np.array(A[task]["delta_norm_std"])
        cs_m = np.array(A[task]["cos_sim_mean"])
        cs_s = np.array(A[task]["cos_sim_std"])

        (ln1,) = ax.plot(
            k_arr,
            dn_m,
            "o-",
            color="#1f77b4",
            lw=2,
            ms=7,
            label=r"$\Delta$ norm $\|\delta_k\|/\|h_0\|$",
        )
        ax.fill_between(k_arr, dn_m - dn_s, dn_m + dn_s, color="#1f77b4", alpha=0.18)

        (ln2,) = ax2.plot(
            k_arr,
            cs_m,
            "s-",
            color="#d62728",
            lw=2,
            ms=6,
            label=r"cos$\angle(\delta_k,\delta_1)$",
        )
        ax2.fill_between(k_arr, cs_m - cs_s, cs_m + cs_s, color="#d62728", alpha=0.15)

        # Emphasize k=2: second T pass
        ax.axvline(2, color="#2ca02c", ls="--", lw=1.8, alpha=0.85)
        ax.annotate(
            "k=2: cos≈0.43\n(not aligned with 1st step)",
            xy=(2, dn_m[1]),
            xytext=(3.3, dn_m[1] + 0.25),
            fontsize=8,
            color="#1a1a1a",
            arrowprops=dict(arrowstyle="->", color="0.35", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#fffde7", edgecolor="#c9b87a", alpha=0.95),
        )

        ax.set_ylabel(r"$\Delta$ norm (normalized)", color="#1f77b4", fontsize=10)
        ax2.set_ylabel(r"Direction cosine vs 1st $\Delta$", color="#d62728", fontsize=10)
        ax.tick_params(axis="y", labelcolor="#1f77b4")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        ax.set_title(f"{t}  ({'BoolQ validation' if task == 'boolq' else 'ARC-Challenge test'})", fontsize=10, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.35)
        ax.set_xticks(k_arr)

        lns = [ln1, ln2]
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc="upper right", fontsize=8, framealpha=0.92)

    axes[-1].set_xlabel(r"Iteration $k$ (full T-block passes)", fontsize=10)
    out = FIG_DIR / "abc_r1_experiment_A_trajectory.png"
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def plot_B(data: dict) -> None:
    """Per-layer step size; highlight T-block and layer 24."""
    B = data["experiment_B"]
    cfg = data["config"]
    layers = np.array(B["layer_idx"])
    mean_s = np.array(B["mean_step_size"])
    z = np.array(B["z_scores"])

    fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    # Skip layer 0 (embedding jump) for y-scale readability; show as annotation
    m0 = mean_s[0]
    plot_layers = layers[1:36]
    plot_mean = mean_s[1:36]

    ax.plot(plot_layers, plot_mean, color="#444", lw=1.8, marker="o", ms=3.5, label="Mean step size (last token)")
    ax.fill_between(
        plot_layers,
        plot_mean - np.array(B["std_step_size"][1:36]),
        plot_mean + np.array(B["std_step_size"][1:36]),
        alpha=0.2,
        color="#444",
    )

    ts, te = cfg["t_start_opt"], cfg["t_end_opt"]
    ax.axvspan(ts, te, color="#a6d8ff", alpha=0.35, label=f"Optimal T-block [{ts},{te}]")
    ax.axvline(24, color="#ff7f0e", ls="--", lw=2, label="Layer 24 (t_end=24 valley)")
    z24 = z[24]
    ax.annotate(
        f"Layer 24\nstep={mean_s[24]:.3f}\nz={z24:.2f}",
        xy=(24, mean_s[24]),
        xytext=(26, 0.52),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=0.9),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ff7f0e", alpha=0.95),
    )

    ax.text(
        0.01,
        0.98,
        f"Layer 0 step = {m0:.2f} (embedding transition; omitted from x-axis start)\n"
        f"N_B = {cfg['N_B']} BoolQ samples  |  z-score flags: high >1.5 → {B['high_step_layers']}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", fc="white", ec="0.75", alpha=0.92),
    )

    ax.set_xlabel("Layer index (0-based, after embedding)", fontsize=10)
    ax.set_ylabel(r"Step size $\|\Delta h\|/\|h\|$ (last token)", fontsize=10)
    ax.set_title(
        "Experiment B — Per-layer step size (single forward, BoolQ)\n"
        "Valley at t_end=24 is not explained by layer-24 z-score alone",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlim(0.5, 35.5)
    ax.set_ylim(0.15, 0.72)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.35)

    out = FIG_DIR / "abc_r1_experiment_B_layer_step.png"
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def plot_C(data: dict) -> None:
    """Log-likelihood benefit for gold label: k=2 vs k=1, split by baseline correct/wrong."""
    C = data["experiment_C"]
    cfg = data["config"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    fig.suptitle(
        "Experiment C — Δ log-likelihood of gold answer  (ETD k=2 vs k=1)\n"
        f"Optimal T-block  |  N={cfg['N_C']}/task",
        fontsize=11,
        fontweight="bold",
    )

    for ax, task in zip(axes, ["boolq", "arc"]):
        bc = np.array(C[task]["benefits_correct"])
        bw = np.array(C[task]["benefits_wrong"])
        pos = [1, 2]
        bp = ax.boxplot(
            [bc, bw],
            positions=pos,
            widths=0.55,
            patch_artist=True,
            showfliers=True,
            medianprops=dict(color="black", lw=2),
        )
        bp["boxes"][0].set_facecolor("#b3cde3")
        bp["boxes"][1].set_facecolor("#fbb4ae")

        rng = np.random.default_rng(42)
        for i, arr in enumerate([bc, bw]):
            jitter = rng.uniform(-0.09, 0.09, size=len(arr))
            ax.scatter(np.full_like(arr, pos[i], dtype=float) + jitter, arr, alpha=0.35, s=14, color="0.25", zorder=3)

        mc = C[task]["mean_benefit_correct_group"]
        mw = C[task]["mean_benefit_wrong_group"]
        ax.plot([0.65, 1.35], [mc, mc], color="#08519c", lw=2.5, label=f"Mean (correct): {mc:+.3f}")
        ax.plot([1.65, 2.35], [mw, mw], color="#cb181d", lw=2.5, label=f"Mean (wrong): {mw:+.3f}")
        ax.axhline(0, color="0.4", ls=":", lw=1.2)

        ax.set_xticks(pos)
        ax.set_xticklabels(
            [
                f"Baseline correct\n(n={C[task]['n_correct_group']})",
                f"Baseline wrong\n(n={C[task]['n_wrong_group']})",
            ],
            fontsize=9,
        )
        ax.set_ylabel(r"$\Delta$ log p(gold)  (k=2 minus k=1)", fontsize=10)
        ax.set_title(task.upper(), fontsize=10, fontweight="bold")
        ax.legend(loc="upper right", fontsize=7.5)
        ax.grid(True, axis="y", alpha=0.35)

    out = FIG_DIR / "abc_r1_experiment_C_benefit.png"
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def plot_R1(data: dict) -> None:
    """Accuracy: standard ETD k=2 vs R1 residual."""
    R1 = data["experiment_R1"]
    cfg = data["config"]

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    labels = [
        f"Optimal\n({cfg['t_start_opt']},{cfg['t_end_opt']})\nBoolQ",
        f"Optimal\n({cfg['t_start_opt']},{cfg['t_end_opt']})\nARC",
        f"Valley\n({cfg['t_start_valley']},{cfg['t_end_valley']})\nBoolQ",
        f"Valley\n({cfg['t_start_valley']},{cfg['t_end_valley']})\nARC",
    ]
    x = np.arange(4)
    w = 0.36
    orig = [
        R1["optimal"]["boolq"]["acc_orig_k2"],
        R1["optimal"]["arc"]["acc_orig_k2"],
        R1["valley"]["boolq"]["acc_orig_k2"],
        R1["valley"]["arc"]["acc_orig_k2"],
    ]
    r1v = [
        R1["optimal"]["boolq"]["acc_r1_k2"],
        R1["optimal"]["arc"]["acc_r1_k2"],
        R1["valley"]["boolq"]["acc_r1_k2"],
        R1["valley"]["arc"]["acc_r1_k2"],
    ]

    b1 = ax.bar(x - w / 2, orig, w, label="ETD k=2 (standard)", color="#4c72b0", edgecolor="white")
    b2 = ax.bar(x + w / 2, r1v, w, label="R1: h + h_e", color="#dd8452", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Accuracy (N=100 / split)", fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.35)
    ax.set_title(
        "Experiment R1 — Global residual after T-block\n"
        "R1 hurts on optimal config; tiny lift on valley only",
        fontsize=11,
        fontweight="bold",
    )

    for i, (o, r) in enumerate(zip(orig, r1v)):
        d = r - o
        color = "#2ca02c" if d > 0 else "#c0392b"
        ax.annotate(
            f"Δ{d:+.2f}",
            xy=(i, max(o, r) + 0.02),
            ha="center",
            fontsize=8,
            fontweight="bold",
            color=color,
        )

    out = FIG_DIR / "abc_r1_experiment_R1_accuracy.png"
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    if not JSON_PATH.exists():
        raise SystemExit(f"Missing {JSON_PATH}; run exp_abc_r1.py first.")
    data = load()
    plot_A(data)
    plot_B(data)
    plot_C(data)
    plot_R1(data)
    print("Done.")


if __name__ == "__main__":
    main()
