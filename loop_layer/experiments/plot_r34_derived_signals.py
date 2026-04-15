#!/usr/bin/env python3
"""
从 R34 全量 JSON 离线绘制派生曲线（无需 GPU）：

对每条原始信号 x（与 exp_r34 中 12 个 SIGNAL_SPECS 一致）：

1. demeaned_x[l] = mean_i x_i(l) - (1/L) * sum_l' mean_i x_i(l')
   即「相对全层均值的偏差」。

2. delta_x[l] = mean_i x_i(l) - mean_i x_i(l-1)（l>=1；l=0 为 nan）
   即「相邻层上均值曲线的差分」。

3. var_x[l] = Var_i x_i(l)（样本间方差，ddof=1）。

输出（每 benchmark 三张 3×4 子图 + 全 bench 叠图各三张）：
  figures/r34_cross_memory/derived/{bench}_r34_demeaned_vs_layer.png
  figures/r34_cross_memory/derived/{bench}_r34_delta_vs_layer.png
  figures/r34_cross_memory/derived/{bench}_r34_var_vs_layer.png
  figures/r34_cross_memory/derived/r34_all_demeaned_overlay.png
  figures/r34_cross_memory/derived/r34_all_delta_overlay.png
  figures/r34_cross_memory/derived/r34_all_var_overlay.png

用法：
  python3 plot_r34_derived_signals.py
  python3 plot_r34_derived_signals.py --json /path/to/r34_cross_memory_data_full.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/root/autodl-tmp/loop_layer")
EXP = ROOT / "experiments"
DEFAULT_JSON = EXP / "results" / "r34_cross_memory_data_full.json"
OUT_DIR = EXP / "figures" / "r34_cross_memory" / "derived"

# 与 exp_r34_cross_memory_probe.py 中 SIGNAL_SPECS 顺序一致
SIGNAL_SPECS: list[tuple[str, str]] = [
    ("attn_write_norm", "||a_l|| / ||h||"),
    ("ffn_write_norm", "||m_l|| / ||h||"),
    ("attn_ffn_balance", "||a||/(||a||+||m||)"),
    ("hidden_rotation_rate", "1-cos(h_out, h_in)"),
    ("ffn_direction_drift", "1-cos(m_l, m_{l-1})"),
    ("attn_direction_drift", "1-cos(a_l, a_{l-1})"),
    ("cross_cos_a_m", "cos(a_l, m_l)"),
    ("cross_attn_to_ffn_sens", "||FFN(h+a)-FFN(h)||/||FFN(h+a)||"),
    ("cross_attn_to_ffn_dirshift", "1-cos(FFN(h+a), FFN(h))"),
    ("logit_lens_jsd_vel", "JSD velocity"),
    ("prediction_flip_rate", "Flip rate"),
    ("residual_write_norm", "||delta h||/||h||"),
]

BENCH_COLORS = {
    "BoolQ": "#2196F3",
    "ARC-C": "#F44336",
    "CSQA": "#4CAF50",
    "TruthfulQA": "#FF9800",
    "MMLU-HS-Math": "#9C27B0",
    "GPQA-Diamond": "#00BCD4",
    "AGIEval-Gaokao-MathQA": "#795548",
    "LogiQA": "#607D8B",
}


def add_r30_marks(ax, r30: dict, bench: str) -> None:
    opt = r30.get(bench)
    if opt is None:
        return
    t0, t1 = int(opt["t_start"]), int(opt["t_stop"])
    ax.axvspan(t0, t1, alpha=0.12, color="gold", zorder=0)
    ax.axvline(t0, color="#2ca02c", ls="--", lw=1.5, zorder=4)
    ax.axvline(t1, color="#d62728", ls="--", lw=1.5, zorder=4)


def stack_per_layer(records: list[dict], key: str, n_layers: int) -> np.ndarray:
    """[n_samples, n_layers] float64."""
    rows = []
    for r in records:
        seq = r["per_layer"].get(key)
        if seq is None or len(seq) != n_layers:
            rows.append(np.full(n_layers, np.nan))
        else:
            rows.append(np.asarray(seq, dtype=np.float64))
    return np.stack(rows, axis=0)


def plot_one_benchmark_grid(
    bench: str,
    r30: dict,
    n_layers: int,
    demeaned: dict[str, np.ndarray],
    delta_m: dict[str, np.ndarray],
    var_x: dict[str, np.ndarray],
    color: str,
    out_dir: Path,
) -> None:
    layers = np.arange(n_layers)
    safe = bench.replace("/", "-")

    def _fig(title_suffix: str, data_map: dict[str, np.ndarray], fname: str) -> None:
        fig, axes = plt.subplots(3, 4, figsize=(22, 13))
        axes_flat = axes.flatten()
        for ax_i, (key, ylab) in enumerate(SIGNAL_SPECS):
            ax = axes_flat[ax_i]
            y = data_map[key]
            add_r30_marks(ax, r30, bench)
            ax.plot(layers, y, color=color, lw=2.2)
            ax.set_xlabel("Layer")
            ax.set_ylabel(ylab, fontsize=8)
            ax.set_title(key, fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.5, n_layers - 0.5)
        fig.suptitle(f"R34 {title_suffix} | {bench}", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    _fig("demeaned: x_mean(l) - mean_l x_mean(l)", demeaned, f"{safe}_r34_demeaned_vs_layer.png")
    _fig("delta: x_mean(l) - x_mean(l-1)", delta_m, f"{safe}_r34_delta_vs_layer.png")
    _fig("var across samples: Var_i x_i(l)", var_x, f"{safe}_r34_var_vs_layer.png")


def plot_overlay(
    all_demeaned: dict[str, dict[str, np.ndarray]],
    all_delta: dict[str, dict[str, np.ndarray]],
    all_var: dict[str, dict[str, np.ndarray]],
    n_layers: int,
    out_dir: Path,
) -> None:
    layers = np.arange(n_layers)

    def _overlay(data: dict[str, dict[str, np.ndarray]], title: str, fname: str) -> None:
        fig, axes = plt.subplots(3, 4, figsize=(22, 13))
        axes_flat = axes.flatten()
        for ax_i, (key, ylab) in enumerate(SIGNAL_SPECS):
            ax = axes_flat[ax_i]
            for bench, dmap in data.items():
                c = BENCH_COLORS.get(bench, "gray")
                # 叠图不画各 bench 的 T-block（区间因任务而异，叠在一起会混乱）
                ax.plot(layers, dmap[key], color=c, lw=1.4, label=bench, alpha=0.85)
            ax.set_xlabel("Layer")
            ax.set_ylabel(ylab, fontsize=8)
            ax.set_title(key, fontsize=10)
            if ax_i == 0:
                ax.legend(loc="upper right", fontsize=5, ncol=2)
            ax.grid(True, alpha=0.25)
            ax.set_xlim(-0.5, n_layers - 0.5)
        fig.suptitle(title, fontsize=13)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    _overlay(
        all_demeaned,
        "R34 demeaned (all benchmarks)",
        "r34_all_demeaned_overlay.png",
    )
    _overlay(all_delta, "R34 delta of layer-mean (all benchmarks)", "r34_all_delta_overlay.png")
    _overlay(all_var, "R34 per-layer sample variance (all benchmarks)", "r34_all_var_overlay.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON, help="r34_cross_memory_data_full.json")
    args = ap.parse_args()

    with open(args.json, encoding="utf-8") as f:
        payload = json.load(f)

    n_layers = int(payload["n_layers"])
    r30 = payload.get("r30_optimal", {})
    benches: dict[str, list] = payload.get("benches", {})

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    all_demeaned: dict[str, dict[str, np.ndarray]] = {}
    all_delta: dict[str, dict[str, np.ndarray]] = {}
    all_var: dict[str, dict[str, np.ndarray]] = {}

    for bench, records in benches.items():
        if not records:
            continue
        color = BENCH_COLORS.get(bench, "gray")
        demeaned: dict[str, np.ndarray] = {}
        delta_m: dict[str, np.ndarray] = {}
        var_x: dict[str, np.ndarray] = {}

        for key, _ in SIGNAL_SPECS:
            mat = stack_per_layer(records, key, n_layers)  # [N, L]
            x_mean = np.nanmean(mat, axis=0)
            xm = x_mean[np.isfinite(x_mean)]
            global_mean = float(np.mean(xm)) if xm.size > 0 else 0.0
            demeaned[key] = x_mean - global_mean

            d = np.full(n_layers, np.nan, dtype=np.float64)
            d[1:] = x_mean[1:] - x_mean[:-1]
            delta_m[key] = d

            # sample variance per layer (ddof=1); all-nan columns -> nan
            v = np.empty(n_layers, dtype=np.float64)
            for li in range(n_layers):
                col = mat[:, li]
                col = col[np.isfinite(col)]
                v[li] = float(np.var(col, ddof=1)) if col.size > 1 else np.nan
            var_x[key] = v

        all_demeaned[bench] = demeaned
        all_delta[bench] = delta_m
        all_var[bench] = var_x

        plot_one_benchmark_grid(bench, r30, n_layers, demeaned, delta_m, var_x, color, out_dir)
        print(f"Wrote {bench}: demeaned / delta / var (3 PNGs) -> {out_dir}/")

    plot_overlay(all_demeaned, all_delta, all_var, n_layers, out_dir)
    print(f"Wrote overlays -> {out_dir}/r34_all_*_overlay.png")
    print("Done.")


if __name__ == "__main__":
    main()
