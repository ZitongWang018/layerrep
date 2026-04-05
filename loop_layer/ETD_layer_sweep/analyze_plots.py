#!/usr/bin/env python3
"""Aggregate sweep CSV, print summary stats, save English figures, generate experiment report."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sweep_config import ARTIFACTS, DEFAULT_ARC_LIMIT, DEFAULT_BOOLQ_LIMIT, FIGURES


def pivot_matrix(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.pivot(index="t_start", columns="t_end", values=col).sort_index(axis=0).sort_index(axis=1)


def plot_heatmap(mat: pd.DataFrame, title: str, cbar_label: str, out_path: Path, vmin=None, vmax=None) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    data = mat.values.astype(float)
    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        extent=[
            mat.columns.min() - 0.5,
            mat.columns.max() + 0.5,
            mat.index.min() - 0.5,
            mat.index.max() + 0.5,
        ],
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )
    ax.set_xlabel("T end layer index (inclusive)")
    ax.set_ylabel("T start layer index (inclusive)")
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_delta(mat_a: pd.DataFrame, mat_b: pd.DataFrame, title: str, out_path: Path) -> None:
    delta = mat_a - mat_b
    vmax = float(np.nanmax(np.abs(delta.values)))
    vmin = -vmax if vmax > 0 else None
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        delta.values.astype(float),
        aspect="auto",
        origin="lower",
        extent=[
            delta.columns.min() - 0.5,
            delta.columns.max() + 0.5,
            delta.index.min() - 0.5,
            delta.index.max() + 0.5,
        ],
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("T end layer index (inclusive)")
    ax.set_ylabel("T start layer index (inclusive)")
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Accuracy delta")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def best_cell(df: pd.DataFrame, col: str) -> tuple[float, int, int]:
    row = df.loc[df[col].idxmax()]
    return float(row[col]), int(row["t_start"]), int(row["t_end"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="Path to sweep_results.csv")
    parser.add_argument("--report", default=None, help="Output markdown report path")
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else ARTIFACTS / "sweep_results.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing {csv_path}; run run_sweep.py first.")

    df = pd.read_csv(csv_path)
    n = len(df)
    FIGURES.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("boolq_baseline", "BoolQ accuracy (baseline)", "Accuracy"),
        ("boolq_k2", "BoolQ accuracy (ETD, k=2)", "Accuracy"),
        ("boolq_k3", "BoolQ accuracy (ETD, k=3)", "Accuracy"),
        ("arc_baseline", "ARC-Challenge accuracy (baseline)", "Accuracy"),
        ("arc_k2", "ARC-Challenge accuracy (ETD, k=2)", "Accuracy"),
        ("arc_k3", "ARC-Challenge accuracy (ETD, k=3)", "Accuracy"),
    ]

    mats = {}
    for col, title, clabel in metrics:
        mat = pivot_matrix(df, col)
        mats[col] = mat
        fname = f"heatmap_{col}.png"
        plot_heatmap(mat, title, clabel, FIGURES / fname, vmin=0.0, vmax=1.0)
        print(f"Saved {FIGURES / fname}")

    # Deltas vs baseline
    for name, short in [("boolq", "BoolQ"), ("arc", "ARC-Challenge")]:
        b = mats[f"{name}_baseline"]
        for k in (2, 3):
            plot_delta(
                mats[f"{name}_k{k}"],
                b,
                f"{short}: accuracy delta (ETD k={k} minus baseline)",
                FIGURES / f"delta_{name}_k{k}_vs_baseline.png",
            )
            print(f"Saved {FIGURES / f'delta_{name}_k{k}_vs_baseline.png'}")

    # Summary stats
    lines: list[str] = []
    lines.append("# ETD Layer Sweep Experiment Report\n")
    lines.append("\n## 摘要（中文）\n\n")
    lines.append(
        "本报告对应 **T 思考块层区间** 网格搜索："
        "`t_start`∈[5,15]、`t_end`∈[20,28]、且 `t_start≤t_end`，共 99 组；"
        "每组在 **相同的 BoolQ / ARC 样本列表** 上对比 **baseline、ETD k=2、k=3**。"
        "图表标题与图例均为英文。\n\n"
    )
    lines.append("## Setup\n")
    lines.append(
        "- Model: Qwen3-8B (local path from sweep script).\n"
        "- Thinking block **T** uses contiguous layers `[t_start, t_end]` (inclusive, 0-based).\n"
        "- Encoder **E** = layers `0 .. t_start-1`, **D** = layers `t_end+1 .. 35` (36 layers total).\n"
        "- Sweep: `t_start` in [5, 15], `t_end` in [20, 28], all pairs with `t_start <= t_end`.\n"
        "- Modes per cell: **baseline** (standard forward), **ETD k=2**, **ETD k=3**.\n"
        "- **BoolQ** and **ARC** use the **same fixed example lists** for every cell (see limits below).\n"
    )
    lines.append(f"- Rows in CSV: **{n}** (full sweep expected **99**).\n")
    lines.append(f"- Default BoolQ limit: **{DEFAULT_BOOLQ_LIMIT}** (split: validation).\n")
    lines.append(f"- Default ARC limit: **{DEFAULT_ARC_LIMIT}** (split: test).\n")
    lines.append("\n## Summary statistics\n\n")
    lines.append("| Metric | Mean | Std | Max | Best t_start | Best t_end |\n")
    lines.append("|--------|------|-----|-----|--------------|------------|\n")
    for col, _, _ in metrics:
        mx, ts, te = best_cell(df, col)
        lines.append(
            f"| {col} | {df[col].mean():.4f} | {df[col].std():.4f} | {mx:.4f} | {ts} | {te} |\n"
        )

    lines.append("\n## Figures\n\n")
    for col, title, _ in metrics:
        lines.append(f"### {title}\n\n")
        lines.append(f"![{title}](figures/heatmap_{col}.png)\n\n")

    lines.append("### Accuracy delta (ETD minus baseline)\n\n")
    for name, short in [("boolq", "BoolQ"), ("arc", "ARC")]:
        for k in (2, 3):
            lines.append(f"**{short}, k={k} minus baseline**\n\n")
            lines.append(f"![delta {name} k={k}](figures/delta_{name}_k{k}_vs_baseline.png)\n\n")

    lines.append("\n## Runtime\n\n")
    if "seconds_total" in df.columns:
        lines.append(
            f"- Total wall time (sum over cells): **{df['seconds_total'].sum():.1f}** s.\n"
        )
        lines.append(
            f"- Mean seconds per cell (all modes, both benchmarks): **{df['seconds_total'].mean():.2f}** s.\n"
        )

    report_path = Path(args.report) if args.report else Path(__file__).resolve().parent / "EXPERIMENT_REPORT.md"
    report_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
