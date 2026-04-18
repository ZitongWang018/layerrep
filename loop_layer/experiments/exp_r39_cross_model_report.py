#!/usr/bin/env python3
"""
R39 跨模型综合分析报告生成器

从 R39C 三个模型的 JSON 结果生成：
  1. 跨模型精度热力图（行=benchmark，列=method，分左中右三个子图）
  2. 三模型宏平均对比图
  3. 各模型每方法的 Δacc vs baseline / sweep_best 对比
  4. 方法稳健性分析（各方法在三个模型上的胜率）
  5. 最终结论总结

用法：python experiments/exp_r39_cross_model_report.py
      （自动读取 results/r39c_final_{qwen3,llama3,gemma2}.json）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

ROOT = Path("/root/autodl-tmp/loop_layer")
EXP  = ROOT / "experiments"
RES  = EXP / "results"
FIG  = EXP / "figures" / "r39_cross_model"
FIG.mkdir(parents=True, exist_ok=True)

MODEL_FILES = {
    "Qwen3-8B":  RES / "r39c_final_qwen3.json",
    "Llama3-8B": RES / "r39c_final_llama3.json",
    "Gemma2-2B": RES / "r39c_final_gemma2.json",
}
BENCH_ORDER = ["BoolQ","ARC-C","TruthfulQA","CSQA","MMLU-HS-Math",
               "GPQA-Diamond","AGIEval-Gaokao-MathQA","LogiQA"]
EVAL_COND_NAMES = ["baseline","sweep_best","neg_cos_am_calib","emp_logit_fixed","neg_cos_am_ps_nt"]
SIGNAL_CONDS    = ["neg_cos_am_calib","emp_logit_fixed","neg_cos_am_ps_nt"]
COND_COLORS = {
    "baseline":          "#888888",
    "sweep_best":        "#1f77b4",
    "neg_cos_am_calib":  "#9467bd",
    "emp_logit_fixed":   "#2ca02c",
    "neg_cos_am_ps_nt":  "#d62728",
}
COND_LABELS = {
    "baseline":          "Baseline",
    "sweep_best":        "Sweep Best",
    "neg_cos_am_calib":  "neg_cos_am Calib",
    "emp_logit_fixed":   "Empirical Logit",
    "neg_cos_am_ps_nt":  "Per-Sample n_t",
}


def load_all_data() -> dict[str, dict]:
    all_data = {}
    for model, path in MODEL_FILES.items():
        if path.exists():
            raw = json.loads(path.read_text())
            all_data[model] = raw.get("results", {})
            print(f"  Loaded {model}: {len(all_data[model])}/8 benchmarks")
        else:
            print(f"  [MISSING] {model}: {path}")
    return all_data


def plot_cross_model_heatmap(all_data: dict[str, dict]):
    """三模型 Δacc vs baseline 热力图（大图）"""
    models = [m for m in MODEL_FILES if m in all_data]
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models + 2, 7), sharey=False)
    if n_models == 1:
        axes = [axes]

    for ax_idx, model in enumerate(models):
        ax = axes[ax_idx]
        results = all_data[model]
        benches = [b for b in BENCH_ORDER if b in results]
        mat = np.zeros((len(SIGNAL_CONDS), len(benches)))
        for bi, b in enumerate(benches):
            a = results[b]["accuracies"]
            for ci, cn in enumerate(SIGNAL_CONDS):
                mat[ci, bi] = a.get(cn, 0) - a.get("baseline", 0)
        vmax = max(0.06, np.abs(mat).max())
        im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(benches)))
        ax.set_xticklabels([b[:12] for b in benches], rotation=30, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(SIGNAL_CONDS)))
        ax.set_yticklabels([COND_LABELS[c] for c in SIGNAL_CONDS], fontsize=8)
        ax.set_title(f"{model}\nΔacc vs Baseline", fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.7, label="Δacc")
        for ci in range(len(SIGNAL_CONDS)):
            for bi in range(len(benches)):
                v = mat[ci, bi]
                ax.text(bi, ci, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                        color="black" if abs(v) < vmax*0.6 else "white")

    plt.suptitle("R39C：三模型信号方法 Δacc vs Baseline 热力图\n（绿色=改善，红色=下降）",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = FIG / "01_cross_model_heatmap_vs_base.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  → {p}")


def plot_cross_model_heatmap_vs_sweep(all_data: dict[str, dict]):
    """三模型 Δacc vs sweep_best 热力图"""
    models = [m for m in MODEL_FILES if m in all_data]
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models + 2, 7), sharey=False)
    if n_models == 1:
        axes = [axes]

    for ax_idx, model in enumerate(models):
        ax = axes[ax_idx]
        results = all_data[model]
        benches = [b for b in BENCH_ORDER if b in results]
        mat = np.zeros((len(SIGNAL_CONDS), len(benches)))
        for bi, b in enumerate(benches):
            a = results[b]["accuracies"]
            for ci, cn in enumerate(SIGNAL_CONDS):
                mat[ci, bi] = a.get(cn, 0) - a.get("sweep_best", 0)
        vmax = max(0.06, np.abs(mat).max())
        im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(benches)))
        ax.set_xticklabels([b[:12] for b in benches], rotation=30, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(SIGNAL_CONDS)))
        ax.set_yticklabels([COND_LABELS[c] for c in SIGNAL_CONDS], fontsize=8)
        ax.set_title(f"{model}\nΔacc vs Sweep Best", fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.7, label="Δacc")
        for ci in range(len(SIGNAL_CONDS)):
            for bi in range(len(benches)):
                v = mat[ci, bi]
                ax.text(bi, ci, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                        color="black" if abs(v) < vmax*0.6 else "white")

    plt.suptitle("R39C：三模型信号方法 Δacc vs Sweep Best 热力图\n（绿色=超越最优，红色=不足）",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = FIG / "02_cross_model_heatmap_vs_sweep.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  → {p}")


def plot_macro_comparison(all_data: dict[str, dict]):
    """三模型宏平均对比（并列柱状图）"""
    models = [m for m in MODEL_FILES if m in all_data]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(models))
    n_c = len(EVAL_COND_NAMES)
    w = 0.9 / n_c

    for j, cn in enumerate(EVAL_COND_NAMES):
        vals = []
        for model in models:
            r = all_data[model]
            benches = [b for b in BENCH_ORDER if b in r]
            macro = np.mean([r[b]["accuracies"].get(cn, 0) for b in benches]) if benches else 0
            vals.append(macro)
        offs = (j - (n_c-1)/2.0) * w
        bars = ax.bar(x + offs, vals, w*0.92, label=COND_LABELS[cn],
                      color=COND_COLORS[cn], edgecolor="white", linewidth=0.4)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.002,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Macro-average Accuracy", fontsize=10)
    ax.set_title("R39C 三模型宏平均精度对比\n（neg_cos_am + 改进经验标定 vs baseline vs sweep_best）",
                 fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = FIG / "03_macro_comparison.png"
    plt.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  → {p}")


def plot_win_rate(all_data: dict[str, dict]):
    """方法胜率图（按模型和 vs baseline / vs sweep_best 分层）"""
    models = [m for m in MODEL_FILES if m in all_data]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    n_sc = len(SIGNAL_CONDS)
    x = np.arange(n_sc)
    w = 0.8 / len(models)

    for panel_idx, (ax, ref) in enumerate([(ax1, "baseline"), (ax2, "sweep_best")]):
        for mi, model in enumerate(models):
            r = all_data[model]
            benches = [b for b in BENCH_ORDER if b in r]
            win_rates = []
            for cn in SIGNAL_CONDS:
                wins = sum(1 for b in benches
                           if r[b]["accuracies"].get(cn, 0) > r[b]["accuracies"].get(ref, 0))
                win_rates.append(wins / max(len(benches), 1))
            offs = (mi - (len(models)-1)/2.0) * w
            ax.bar(x+offs, win_rates, w*0.9, label=model, edgecolor="white", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([COND_LABELS[c] for c in SIGNAL_CONDS], rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Win Rate (benchmarks)", fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
        ax.set_title(f"方法胜率 vs {ref}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("R39C：方法在各模型上的胜率\n（胜=方法精度 > 参考精度）", fontsize=11)
    plt.tight_layout()
    p = FIG / "04_win_rate.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  → {p}")


def plot_benchmark_violin(all_data: dict[str, dict]):
    """每个 benchmark 三模型 Δacc vs baseline 分布（单点 scatter）"""
    models = [m for m in MODEL_FILES if m in all_data]
    benches = BENCH_ORDER
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes_flat = axes.flatten()
    for bi, bench in enumerate(benches):
        ax = axes_flat[bi]
        for mi, model in enumerate(models):
            r = all_data.get(model, {})
            if bench not in r: continue
            a = r[bench]["accuracies"]
            base = a.get("baseline", 0)
            sw   = a.get("sweep_best", 0)
            for ci, cn in enumerate(SIGNAL_CONDS):
                v = a.get(cn, 0) - base
                ax.scatter(mi + ci*0.1 - 0.1, v,
                           color=COND_COLORS[cn], s=70, zorder=3, marker="o")
            ax.scatter(mi + 0.3, sw - base, color=COND_COLORS["sweep_best"], s=80,
                       zorder=4, marker="*")
        ax.axhline(0, color="black", linewidth=1.0, linestyle="--")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.split("-")[0] for m in models], fontsize=8)
        ax.set_title(bench[:16], fontsize=8.5)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0],[0],marker="o",color="w",markerfacecolor=COND_COLORS[c],
                               markersize=8,label=COND_LABELS[c]) for c in SIGNAL_CONDS]
    legend_elements.append(Line2D([0],[0],marker="*",color="w",markerfacecolor=COND_COLORS["sweep_best"],
                                   markersize=10,label="Sweep Best"))
    axes_flat[0].legend(handles=legend_elements, fontsize=7, loc="upper left")
    plt.suptitle("R39C：各 Benchmark Δacc vs Baseline（三模型×三方法）", fontsize=11)
    plt.tight_layout()
    p = FIG / "05_benchmark_breakdown.png"
    plt.savefig(p, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  → {p}")


def print_final_summary(all_data: dict[str, dict]):
    """综合数字摘要"""
    models = list(all_data.keys())
    print("\n" + "="*100)
    print("R39C 最终综合结果摘要")
    print("="*100)
    for model in models:
        results = all_data[model]
        benches = [b for b in BENCH_ORDER if b in results]
        print(f"\n{'─'*60}")
        print(f"  {model}  ({len(benches)} benchmarks)")
        print(f"{'─'*60}")
        header = f"  {'Benchmark':24}" + "".join(f"{c[:10]:>12}" for c in EVAL_COND_NAMES)
        print(header)
        print("  " + "-"*(len(header)-2))
        for b in benches:
            a = results[b]["accuracies"]
            sw, bl = a.get("sweep_best",0), a.get("baseline",0)
            row = f"  {b[:23]:24}"
            for c in EVAL_COND_NAMES:
                v = a.get(c, 0)
                if c not in ("baseline","sweep_best"):
                    mk = "★" if v > sw else ("+" if v > bl else " ")
                else:
                    mk = " "
                row += f"  {v:.3f}{mk}{'':5}"
            print(row)
        macro = {c: np.mean([results[b]["accuracies"].get(c,0) for b in benches])
                 for c in EVAL_COND_NAMES}
        sw_m, bl_m = macro["sweep_best"], macro["baseline"]
        row = f"  {'MACRO AVG':24}"
        for c in EVAL_COND_NAMES:
            v = macro[c]
            if c not in ("baseline","sweep_best"):
                mk = "★" if v>sw_m else ("+" if v>bl_m else " ")
            else:
                mk = " "
            row += f"  {v:.3f}{mk}{'':5}"
        print("  " + "-"*(len(row)-2))
        print(row)
        best_sig = max(SIGNAL_CONDS, key=lambda c: macro[c])
        best_val = macro[best_sig]
        print(f"\n  最优信号方法: {best_sig}  ({best_val:.3f})")
        print(f"  Δacc vs baseline:  {best_val - bl_m:+.3f}")
        print(f"  Δacc vs sweep_best:{best_val - sw_m:+.3f}")
        beat_bl = sum(1 for b in benches if results[b]["accuracies"].get(best_sig,0) > results[b]["accuracies"].get("baseline",0))
        beat_sw = sum(1 for b in benches if results[b]["accuracies"].get(best_sig,0) > results[b]["accuracies"].get("sweep_best",0))
        print(f"  超越 baseline：{beat_bl}/{len(benches)} benchmarks")
        print(f"  超越 sweep_best：{beat_sw}/{len(benches)} benchmarks")

    # 全局汇总
    print(f"\n{'═'*100}")
    print("全局摘要：")
    for model in models:
        results = all_data[model]
        benches = [b for b in BENCH_ORDER if b in results]
        if not benches: continue
        macro = {c: np.mean([results[b]["accuracies"].get(c,0) for b in benches])
                 for c in EVAL_COND_NAMES}
        best_sig = max(SIGNAL_CONDS, key=lambda c: macro[c])
        bl_m = macro["baseline"]
        sw_m = macro["sweep_best"]
        best_v = macro[best_sig]
        print(f"  {model:12} baseline={bl_m:.3f}  sweep={sw_m:.3f}  "
              f"best_sig={COND_LABELS[best_sig]}={best_v:.3f}  "
              f"Δbl={best_v-bl_m:+.3f}  Δsw={best_v-sw_m:+.3f}")


def main():
    print("加载三模型 R39C 数据...")
    all_data = load_all_data()
    if not all_data:
        print("[ERROR] 没有找到任何 R39C 结果文件！")
        sys.exit(1)

    print("\n生成可视化...")
    plot_cross_model_heatmap(all_data)
    plot_cross_model_heatmap_vs_sweep(all_data)
    plot_macro_comparison(all_data)
    plot_win_rate(all_data)
    plot_benchmark_violin(all_data)
    print_final_summary(all_data)
    print(f"\n所有图表已保存到: {FIG}")


if __name__ == "__main__":
    main()
