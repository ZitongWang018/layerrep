#!/usr/bin/env python3
"""
Offline plots for R40 BBH + GSM8K JSON results (English labels only).

Reads:
  experiments/results/r40_bbh_gsm8k_{qwen3_8b,llama3_8b,gemma2_2b}.json

Writes to experiments/figures/r40_bbh_gsm8k/ by default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DEFAULT_OUT = ROOT / "figures" / "r40_bbh_gsm8k"

# Figures omit emp_logit_fixed for consistency with R39C publication plots.
METHOD_KEYS = (
    "baseline",
    "neg_cos_am_calib",
    "neg_cos_am_ps_nt",
)
METHOD_LABELS = {
    "baseline": "Baseline",
    "neg_cos_am_calib": "neg_cos_am (calib)",
    "neg_cos_am_ps_nt": "neg_cos_am (per-sample n_t)",
}

COLORS = {
    "baseline": "#37474F",
    "neg_cos_am_calib": "#1565C0",
    "neg_cos_am_ps_nt": "#2E7D32",
}

PRESET_FILES = [
    ("Qwen3-8B", "r40_bbh_gsm8k_qwen3_8b.json"),
    ("Llama3-8B", "r40_bbh_gsm8k_llama3_8b.json"),
    ("Gemma2-2B", "r40_bbh_gsm8k_gemma2_2b.json"),
]


def short_task(name: str) -> str:
    s = name.replace("leaderboard_bbh_", "")
    return s.replace("_", " ").title()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def plot_gsm8k(all_data: list[tuple[str, dict]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    n_groups = len(all_data)
    n_methods = len(METHOD_KEYS)
    width = 0.22
    x = np.arange(n_groups)
    for j, key in enumerate(METHOD_KEYS):
        offs = (j - (n_methods - 1) / 2) * width
        vals = []
        for _, d in all_data:
            em = d.get("gsm8k", {}).get("exact_match", {})
            vals.append(float(em.get(key, 0.0)))
        ax.bar(x + offs, vals, width, label=METHOD_LABELS[key], color=COLORS[key], edgecolor="white", linewidth=0.4)
    ax.set_ylabel("Exact match")
    ax.set_xlabel("Model")
    ax.set_title("GSM8K (5-shot CoT, n=50): exact match by method")
    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in all_data])
    ax.set_ylim(0, 1.05)
    ax.axhline(0, color="#ccc", lw=0.5)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_bbh_grouped(all_data: list[tuple[str, dict]], out: Path) -> None:
    task_order: list[str] | None = None
    for _, d in all_data:
        bbh = d.get("bbh", {})
        if bbh:
            task_order = list(bbh.keys())
            break
    if not task_order:
        return

    n_models = len(all_data)
    fig, axes = plt.subplots(n_models, 1, figsize=(14, 4.2 * n_models), sharey=True)
    if n_models == 1:
        axes = [axes]

    x = np.arange(len(task_order))
    width = 0.24
    n_methods = len(METHOD_KEYS)

    for ax, (preset_name, d) in zip(axes, all_data):
        bbh = d.get("bbh", {})
        for j, key in enumerate(METHOD_KEYS):
            offs = (j - (n_methods - 1) / 2) * width
            vals = [float(bbh[t]["accuracies"][key]) for t in task_order]
            ax.bar(x + offs, vals, width, label=METHOD_LABELS[key], color=COLORS[key], edgecolor="white", linewidth=0.35)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"BBH (6 subtasks × n=50): {preset_name}")
        ax.set_xticks(x)
        ax.set_xticklabels([short_task(t) for t in task_order], rotation=22, ha="right")
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="lower right", ncol=2, framealpha=0.95, fontsize=9)

    axes[-1].set_xlabel("Subtask")
    fig.suptitle("R40 BBH: accuracy by method vs baseline", y=1.01, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_bbh_delta(all_data: list[tuple[str, dict]], out: Path) -> None:
    task_order: list[str] | None = None
    for _, d in all_data:
        bbh = d.get("bbh", {})
        if bbh:
            task_order = list(bbh.keys())
            break
    if not task_order:
        return

    etd_keys = [k for k in METHOD_KEYS if k != "baseline"]
    n_models = len(all_data)
    fig, axes = plt.subplots(n_models, 1, figsize=(14, 4.0 * n_models), sharey=True)
    if n_models == 1:
        axes = [axes]

    x = np.arange(len(task_order))
    width = 0.22
    n_bars = len(etd_keys)

    for ax, (preset_name, d) in zip(axes, all_data):
        bbh = d.get("bbh", {})
        for j, key in enumerate(etd_keys):
            offs = (j - (n_bars - 1) / 2) * width
            deltas = []
            for t in task_order:
                acc = bbh[t]["accuracies"]
                b = float(acc["baseline"])
                deltas.append(float(acc[key]) - b)
            ax.bar(x + offs, deltas, width, label=METHOD_LABELS[key], color=COLORS[key], edgecolor="white", linewidth=0.35)
        ax.axhline(0, color="#222", lw=1)
        ax.set_ylabel(r"$\Delta$ accuracy vs baseline")
        ax.set_title(f"BBH: {preset_name}")
        ax.set_xticks(x)
        ax.set_xticklabels([short_task(t) for t in task_order], rotation=22, ha="right")
        ax.set_ylim(-0.55, 0.55)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="lower right", ncol=3, framealpha=0.95, fontsize=9)

    axes[-1].set_xlabel("Subtask")
    fig.suptitle("R40 BBH: ETD methods minus baseline per subtask", y=1.01, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_gsm8k_delta(all_data: list[tuple[str, dict]], out: Path) -> None:
    etd_keys = [k for k in METHOD_KEYS if k != "baseline"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    n_groups = len(all_data)
    width = 0.22
    x = np.arange(n_groups)
    n_bars = len(etd_keys)
    for j, key in enumerate(etd_keys):
        offs = (j - (n_bars - 1) / 2) * width
        vals = []
        for _, d in all_data:
            em = d.get("gsm8k", {}).get("exact_match", {})
            vals.append(float(em[key]) - float(em["baseline"]))
        ax.bar(x + offs, vals, width, label=METHOD_LABELS[key], color=COLORS[key], edgecolor="white", linewidth=0.4)
    ax.axhline(0, color="#222", lw=1)
    ax.set_ylabel(r"$\Delta$ exact match vs baseline")
    ax.set_xlabel("Model")
    ax.set_title("GSM8K (n=50): ETD methods minus baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in all_data])
    ax.set_ylim(-0.35, 0.35)
    ax.legend(loc="best", framealpha=0.95)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_bbh_heatmap_matrix(all_data: list[tuple[str, dict]], out: Path) -> None:
    """One heatmap per method (rows) × models (cols) for macro mean BBH accuracy."""
    task_order: list[str] | None = None
    for _, d in all_data:
        bbh = d.get("bbh", {})
        if bbh:
            task_order = list(bbh.keys())
            break
    if not task_order:
        return

    model_names = [n for n, _ in all_data]
    n_tasks = len(task_order)
    n_methods = len(METHOD_KEYS)

    fig, axes = plt.subplots(
        1, n_methods, figsize=(4.2 * n_methods, 6), sharey=True, constrained_layout=True
    )
    vmin, vmax = 0.0, 1.0
    ims = []
    for mi, key in enumerate(METHOD_KEYS):
        mat = np.zeros((len(model_names), n_tasks))
        for ri, (_, d) in enumerate(all_data):
            bbh = d["bbh"]
            for ci, t in enumerate(task_order):
                mat[ri, ci] = float(bbh[t]["accuracies"][key])
        ax = axes[mi]
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ims.append(im)
        ax.set_xticks(np.arange(n_tasks))
        ax.set_xticklabels([short_task(t) for t in task_order], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(model_names)))
        ax.set_yticklabels(model_names)
        ax.set_title(METHOD_LABELS[key])
    fig.colorbar(ims[-1], ax=list(axes), shrink=0.72, label="Accuracy")
    fig.suptitle("R40 BBH: accuracy heatmap (models × subtasks)", fontsize=12)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_bbh_macro_mean(all_data: list[tuple[str, dict]], out: Path) -> None:
    """Mean BBH accuracy across the 6 subtasks (equal weight)."""
    task_order: list[str] | None = None
    for _, d in all_data:
        bbh = d.get("bbh", {})
        if bbh:
            task_order = list(bbh.keys())
            break
    if not task_order:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    n_groups = len(all_data)
    n_methods = len(METHOD_KEYS)
    width = 0.22
    x = np.arange(n_groups)
    for j, key in enumerate(METHOD_KEYS):
        offs = (j - (n_methods - 1) / 2) * width
        means = []
        for _, d in all_data:
            bbh = d["bbh"]
            means.append(
                float(np.mean([float(bbh[t]["accuracies"][key]) for t in task_order]))
            )
        ax.bar(x + offs, means, width, label=METHOD_LABELS[key], color=COLORS[key], edgecolor="white", linewidth=0.4)
    ax.set_ylabel("Mean accuracy (6 BBH subtasks)")
    ax.set_xlabel("Model")
    ax.set_title("R40 BBH: macro-averaged accuracy (equal weight per subtask)")
    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in all_data])
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_runtime_summary(all_data: list[tuple[str, dict]], out: Path) -> None:
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.5))
    names = [n for n, _ in all_data]
    bbh_tot = []
    gsm_t = []
    for _, d in all_data:
        bbh = d.get("bbh", {})
        bbh_tot.append(sum(float(v.get("elapsed_s", 0)) for v in bbh.values()) / 60.0)
        gsm = d.get("gsm8k", {})
        gsm_t.append(float(gsm.get("elapsed_s", 0)) / 60.0)

    x = np.arange(len(names))
    ax0.bar(x, bbh_tot, color="#5C6BC0", edgecolor="white", label="BBH total wall time")
    ax0.set_xticks(x)
    ax0.set_xticklabels(names)
    ax0.set_ylabel("Wall time (minutes)")
    ax0.set_title("BBH stage (6 subtasks, all methods)")
    ax0.grid(axis="y", alpha=0.25)
    ax0.legend(loc="upper left", framealpha=0.95)

    ax1.bar(x, gsm_t, color="#00897B", edgecolor="white", label="GSM8K wall time")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylabel("Wall time (minutes)")
    ax1.set_title("GSM8K stage (n=50, all methods)")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(loc="upper left", framealpha=0.95)

    fig.suptitle("R40 approximate runtime from result JSON", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot R40 BBH + GSM8K JSON results.")
    p.add_argument("--results-dir", type=Path, default=RESULTS)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_data: list[tuple[str, dict]] = []
    for preset_label, fname in PRESET_FILES:
        path = args.results_dir / fname
        if not path.is_file():
            print(f"skip missing: {path}")
            continue
        all_data.append((preset_label, load_json(path)))

    if not all_data:
        raise SystemExit("No JSON files found under results dir.")

    plot_gsm8k(all_data, args.out_dir / "r40_gsm8k_exact_match.png")
    plot_gsm8k_delta(all_data, args.out_dir / "r40_gsm8k_delta_vs_baseline.png")
    plot_bbh_grouped(all_data, args.out_dir / "r40_bbh_accuracy_by_model.png")
    plot_bbh_delta(all_data, args.out_dir / "r40_bbh_delta_vs_baseline.png")
    plot_bbh_heatmap_matrix(all_data, args.out_dir / "r40_bbh_accuracy_heatmaps.png")
    plot_bbh_macro_mean(all_data, args.out_dir / "r40_bbh_macro_mean_by_model.png")
    plot_runtime_summary(all_data, args.out_dir / "r40_runtime_summary.png")

    print(f"Wrote figures under: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
