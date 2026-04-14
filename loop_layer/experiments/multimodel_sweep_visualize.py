#!/usr/bin/env python3
"""
Figures from etd_layer_sweep_r30style.json (no GPU): accuracy heatmaps, Δ vs baseline heatmaps,
and bar charts comparing baseline to best ETD configs.
All plot text in English.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CHAMP_K = 2


def macro_baseline(baseline: dict[str, float], benches: list[str]) -> float:
    vals = [float(baseline[b]) for b in benches if b in baseline]
    return float(np.mean(vals)) if vals else float("nan")


def fill_matrix(
    results: list[dict],
    key: str,
    t_start_min: int,
    t_start_max: int,
    t_stop_min: int,
    t_stop_max: int,
) -> np.ndarray:
    nrows = t_start_max - t_start_min + 1
    ncols = t_stop_max - t_stop_min + 1
    mat = np.full((nrows, ncols), np.nan, dtype=np.float64)
    for row in results:
        ts, te = int(row["t_start"]), int(row["t_stop"])
        if ts < t_start_min or ts > t_start_max or te < t_stop_min or te > t_stop_max:
            continue
        mat[ts - t_start_min, te - t_stop_min] = float(row[key])
    return mat


def plot_accuracy_heatmaps(
    results: list[dict],
    cfg: dict,
    benches: list[str],
    out_path: Path,
    title_prefix: str,
) -> None:
    ts0 = cfg["grid_bounds"]["t_start_min"]
    ts1 = cfg["grid_bounds"]["t_start_max"]
    te0 = cfg["grid_bounds"]["t_stop_min"]
    te1 = cfg["grid_bounds"]["t_stop_max"]
    nrows = ts1 - ts0 + 1
    ncols = te1 - te0 + 1

    def fm(key: str) -> np.ndarray:
        return fill_matrix(results, key, ts0, ts1, te0, te1)

    keys = ["macro_avg"] + benches
    labels = ["Macro average"] + benches
    n_panels = len(keys)
    n_cols_fig = 3
    n_rows_fig = (n_panels + n_cols_fig - 1) // n_cols_fig
    fig, axes = plt.subplots(n_rows_fig, n_cols_fig, figsize=(5 * n_cols_fig, 4.0 * n_rows_fig))
    axes_flat = np.atleast_1d(axes).ravel()
    extent = (te0 - 0.5, te1 + 0.5, ts0 - 0.5, ts1 + 0.5)

    for ax, key, lab in zip(axes_flat, keys, labels):
        mat = fm(key)
        im = ax.imshow(mat, origin="lower", aspect="auto", extent=extent, cmap="viridis")
        ax.set_xlabel("t_stop (exclusive)")
        ax.set_ylabel("t_start (inclusive)")
        ax.set_title(f"{title_prefix}: ETD accuracy — {lab}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(len(keys), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"{title_prefix} — ETD accuracy (k={CHAMP_K}, alpha=auto), T-block [t_start, t_stop)",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_delta_heatmaps(
    results: list[dict],
    baseline: dict[str, float],
    cfg: dict,
    benches: list[str],
    out_path: Path,
    title_prefix: str,
) -> None:
    ts0 = cfg["grid_bounds"]["t_start_min"]
    ts1 = cfg["grid_bounds"]["t_start_max"]
    te0 = cfg["grid_bounds"]["t_stop_min"]
    te1 = cfg["grid_bounds"]["t_stop_max"]
    mb = macro_baseline(baseline, benches)

    keys = ["macro_avg"] + benches
    labels = ["Macro average (ETD − mean baseline)"] + [f"{b} (ETD − baseline)" for b in benches]

    def delta_row_key(row: dict, key: str) -> float:
        if key == "macro_avg":
            etd_m = float(row["macro_avg"])
            return etd_m - mb
        return float(row[key]) - float(baseline[key])

    n_panels = len(keys)
    n_cols_fig = 3
    n_rows_fig = (n_panels + n_cols_fig - 1) // n_cols_fig
    fig, axes = plt.subplots(n_rows_fig, n_cols_fig, figsize=(5 * n_cols_fig, 4.0 * n_rows_fig))
    axes_flat = np.atleast_1d(axes).ravel()
    extent = (te0 - 0.5, te1 + 0.5, ts0 - 0.5, ts1 + 0.5)

    all_mats = []
    for key in keys:
        mat = np.full((ts1 - ts0 + 1, te1 - te0 + 1), np.nan, dtype=np.float64)
        for row in results:
            ts, te = int(row["t_start"]), int(row["t_stop"])
            if ts < ts0 or ts > ts1 or te < te0 or te > te1:
                continue
            mat[ts - ts0, te - te0] = delta_row_key(row, key)
        all_mats.append(mat)

    vmax = max(np.nanmax(np.abs(m)) for m in all_mats if np.any(np.isfinite(m)))
    if not np.isfinite(vmax) or vmax < 1e-9:
        vmax = 0.05

    for ax, mat, lab in zip(axes_flat, all_mats, labels):
        im = ax.imshow(
            mat,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_xlabel("t_stop (exclusive)")
        ax.set_ylabel("t_start (inclusive)")
        ax.set_title(f"{title_prefix}: {lab}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(len(keys), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"{title_prefix} — Gain vs baseline (positive = ETD better). Macro Δ uses mean of per-bench baselines.",
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_baseline_vs_best_bars(
    results: list[dict],
    baseline: dict[str, float],
    benches: list[str],
    out_path: Path,
    title_prefix: str,
) -> None:
    mb = macro_baseline(baseline, benches)

    best_per_bench: dict[str, dict] = {}
    for b in benches:
        best_per_bench[b] = max(results, key=lambda r: r[b])

    best_macro_row = max(results, key=lambda r: r["macro_avg"])

    x = np.arange(len(benches) + 1)
    w = 0.36
    base_vals = [baseline[b] for b in benches] + [mb]
    # Per-bench oracle: each benchmark's best cell
    oracle_vals = [best_per_bench[b][b] for b in benches] + [
        float(best_macro_row["macro_avg"])
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(x - w / 2, base_vals, width=w, label="Baseline (1× forward)", color="#4C72B0", edgecolor="white")
    ax1.bar(x + w / 2, oracle_vals, width=w, label="Best ETD per benchmark*", color="#DD8452", edgecolor="white")
    labels = list(benches) + ["Macro avg."]
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=22, ha="right")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, min(1.05, max(max(base_vals), max(oracle_vals), 0.1) * 1.15))
    ax1.legend(loc="upper right")
    ax1.set_title("Per-bench oracle: each bar pair uses the best (t_start,t_stop) for that benchmark")
    ax1.grid(axis="y", alpha=0.3)

    # Right: single cell that maximizes macro — compare all benches at once
    macro_cell_vals = [float(best_macro_row[b]) for b in benches] + [float(best_macro_row["macro_avg"])]
    ax2.bar(x - w / 2, base_vals, width=w, label="Baseline", color="#4C72B0", edgecolor="white")
    ax2.bar(x + w / 2, macro_cell_vals, width=w, label="ETD @ best-macro cell", color="#55A868", edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=22, ha="right")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, min(1.05, max(max(base_vals), max(macro_cell_vals), 0.1) * 1.15))
    ts, te = int(best_macro_row["t_start"]), int(best_macro_row["t_stop"])
    ax2.set_title(f"One shared cell: [{ts}, {te}) — maximizes macro average")
    ax2.legend(loc="upper right")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(f"{title_prefix} — Baseline vs ETD (k={CHAMP_K}, alpha=auto)", fontsize=12)
    fig.text(
        0.5,
        0.02,
        "* Rightmost macro column (orange): macro of the best-macro cell, not the mean of per-bench oracle accuracies.",
        ha="center",
        fontsize=8,
        style="italic",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_delta_bar_best_configs(
    results: list[dict],
    baseline: dict[str, float],
    benches: list[str],
    out_path: Path,
    title_prefix: str,
) -> None:
    """Bar chart of (best ETD − baseline) for emphasis on gains/losses."""
    mb = macro_baseline(baseline, benches)
    best_per_bench = {b: max(results, key=lambda r: r[b]) for b in benches}
    best_macro_row = max(results, key=lambda r: r["macro_avg"])

    x = np.arange(len(benches) + 1)
    w = 0.35
    d_oracle = [best_per_bench[b][b] - baseline[b] for b in benches] + [
        float(best_macro_row["macro_avg"]) - mb
    ]
    d_macro_cell = [float(best_macro_row[b]) - baseline[b] for b in benches] + [
        float(best_macro_row["macro_avg"]) - mb
    ]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - w / 2, d_oracle, width=w, label="Δ vs baseline: per-bench oracle ETD", color="#C44E52", edgecolor="white")
    ax.bar(x + w / 2, d_macro_cell, width=w, label="Δ vs baseline: ETD @ best-macro cell", color="#8172B2", edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    labels = list(benches) + ["Macro avg."]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylabel("Accuracy gain (ETD − baseline)")
    ax.legend(loc="best")
    ax.set_title(f"{title_prefix} — Where ETD helps or hurts vs standard forward")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def generate_all_figures(payload: dict, fig_dir: Path, title_prefix: str) -> list[Path]:
    cfg = payload["config"]
    baseline = payload["baseline"]
    benches = list(payload["benchmarks_used"])
    results = list(payload["results"])
    if not results:
        raise ValueError("No results rows in JSON; run sweep first.")

    written: list[Path] = []
    p1 = fig_dir / "etd_layer_sweep_heatmaps.png"
    plot_accuracy_heatmaps(results, cfg, benches, p1, title_prefix)
    written.append(p1)

    p2 = fig_dir / "etd_layer_sweep_delta_heatmaps.png"
    plot_delta_heatmaps(results, baseline, cfg, benches, p2, title_prefix)
    written.append(p2)

    p3 = fig_dir / "baseline_vs_best_etd_bars.png"
    plot_baseline_vs_best_bars(results, baseline, benches, p3, title_prefix)
    written.append(p3)

    p4 = fig_dir / "etd_gain_delta_bars.png"
    plot_delta_bar_best_configs(results, baseline, benches, p4, title_prefix)
    written.append(p4)

    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate sweep figures from JSON (no model).")
    ap.add_argument("--json", type=Path, required=True, help="Path to etd_layer_sweep_r30style.json")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Figure output directory (default: sibling figures/ next to JSON)",
    )
    args = ap.parse_args()

    payload = json.loads(args.json.read_text(encoding="utf-8"))
    preset = payload.get("config", {}).get("preset", "model")
    title_prefix = str(preset).replace("-", " ").upper()
    fig_dir = args.out_dir if args.out_dir is not None else args.json.parent.parent / "figures"
    paths = generate_all_figures(payload, fig_dir, title_prefix)
    for p in paths:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
