"""
R34: 基于方向与交叉交互的 FFN-Attention 信号探针
================================================================
核心改进（vs R33 Gini/spectral gap 失败）：
  - 方向漂移信号 替代 分布形状指标
  - 交叉模块余弦 + 有限差分灵敏度 直接测量 attention-FFN 耦合
  - 保留少数已证实有用的旧信号（logit_lens_jsd_vel, prediction_flip_rate）

覆盖全部 8 个 benchmark（5 R30 + 3 hard_mc），每个 N=20。
每个 benchmark 输出一张 3x4 逐层信号曲线图 + R30 最优 T-block 标注。

输出：
  figures/r34_cross_memory/{bench}_r34_signals_vs_layer.png
  results/r34_cross_memory_data_full.json
  results/r34_cross_memory_stats.json

派生图（离线，见 plot_r34_derived_signals.py；run_r34.sh 成功后自动调用）：
  figures/r34_cross_memory/derived/{bench}_r34_{demeaned,delta,var}_vs_layer.png
  figures/r34_cross_memory/derived/r34_all_{demeaned,delta,var}_overlay.png
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

warnings.filterwarnings("ignore")

ROOT = Path("/root/autodl-tmp/loop_layer")
EXP = ROOT / "experiments"
ETD = ROOT / "ETD"
for p in (str(ROOT), str(EXP), str(ETD)):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from r29.signal_funcs import (
    attn_write_norm_last,
    ffn_write_norm_last,
    ffn_direction_drift_last,
    attn_direction_drift_last,
    hidden_rotation_rate_last,
    cross_cos_a_m_last,
    attn_ffn_balance_last,
    cross_attn_to_ffn_sensitivity,
    cross_attn_to_ffn_dir_shift,
    residual_write_norm,
    prediction_flip_rate_last_token,
)

from hard_mc_benchmark_loaders import (
    load_gpqa_diamond,
    load_agieval_gaokao_mathqa,
    load_logiqa,
)

# ─── 配置 ──────────────────────────────────────────────────────────────────────
MODEL_PATH = "/root/autodl-tmp/model_qwen"
RESULTS_DIR = EXP / "results"
FIGURES_DIR = EXP / "figures" / "r34_cross_memory"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

R30_OPTIMAL = {
    "BoolQ":                  {"t_start": 8,  "t_stop": 22},
    "ARC-C":                  {"t_start": 14, "t_stop": 20},
    "TruthfulQA":             {"t_start": 16, "t_stop": 19},
    "CSQA":                   {"t_start": 10, "t_stop": 22},
    "MMLU-HS-Math":           {"t_start": 10, "t_stop": 18},
    "GPQA-Diamond":           {"t_start": 18, "t_stop": 20},
    "AGIEval-Gaokao-MathQA":  {"t_start": 13, "t_stop": 20},
    "LogiQA":                 {"t_start": 14, "t_stop": 19},
}

BENCH_COLORS = {
    "BoolQ": "#2196F3", "ARC-C": "#F44336", "CSQA": "#4CAF50",
    "TruthfulQA": "#FF9800", "MMLU-HS-Math": "#9C27B0",
    "GPQA-Diamond": "#00BCD4", "AGIEval-Gaokao-MathQA": "#795548",
    "LogiQA": "#607D8B",
}

SIGNAL_SPECS = [
    # (key, y_label, row)
    ("attn_write_norm",            "||a_l|| / ||h||",                  0),
    ("ffn_write_norm",             "||m_l|| / ||h||",                  0),
    ("attn_ffn_balance",           "||a||/(||a||+||m||)",              0),
    ("hidden_rotation_rate",       "1-cos(h_out, h_in)",              0),
    ("ffn_direction_drift",        "1-cos(m_l, m_{l-1})",             1),
    ("attn_direction_drift",       "1-cos(a_l, a_{l-1})",             1),
    ("cross_cos_a_m",              "cos(a_l, m_l)",                   1),
    ("cross_attn_to_ffn_sens",     "||FFN(h+a)-FFN(h)||/||FFN(h+a)||",1),
    ("cross_attn_to_ffn_dirshift", "1-cos(FFN(h+a), FFN(h))",         2),
    ("logit_lens_jsd_vel",         "JSD velocity",                     2),
    ("prediction_flip_rate",       "Flip rate",                        2),
    ("residual_write_norm",        "||delta h||/||h||",                2),
]


# ─── 数据加载 ─────────────────────────────────────────────────────────────────
def load_boolq(n):
    ds = load_dataset("aps/super_glue", "boolq")["validation"]
    out = []
    for x in ds:
        if int(x["label"]) < 0:
            continue
        out.append({
            "prompt": f"{x['passage']}\nQuestion: {x['question']}?\nAnswer:",
            "choices": ["no", "yes"], "label": int(x["label"]),
        })
        if len(out) >= n:
            break
    return out


def load_arc_c(n):
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")["test"]
    out = []
    for x in ds:
        lmap = {k: i for i, k in enumerate(x["choices"]["label"])}
        out.append({
            "prompt": f"Question: {x['question']}\nAnswer:",
            "choices": x["choices"]["text"],
            "label": lmap.get(x["answerKey"], 0),
        })
        if len(out) >= n:
            break
    return out


def load_csqa(n):
    ds = load_dataset("tau/commonsense_qa")["validation"]
    lmap = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    out = []
    for x in ds:
        out.append({
            "prompt": f"Question: {x['question']}\nAnswer:",
            "choices": x["choices"]["text"],
            "label": lmap.get(x["answerKey"], 0),
        })
        if len(out) >= n:
            break
    return out


def load_truthfulqa(n):
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice")["validation"]
    out = []
    for x in ds:
        labels = x["mc1_targets"]["labels"]
        out.append({
            "prompt": f"Question: {x['question']}\nAnswer:",
            "choices": x["mc1_targets"]["choices"],
            "label": int(np.argmax(labels)),
        })
        if len(out) >= n:
            break
    return out


def load_mmlu_hs_math(n):
    try:
        ds = load_dataset("cais/mmlu", "high_school_mathematics")["test"]
        out = []
        for x in ds:
            out.append({
                "prompt": f"Question: {x['question']}\nAnswer:",
                "choices": x["choices"], "label": int(x["answer"]),
            })
            if len(out) >= n:
                break
        return out
    except Exception as e:
        print(f"  [WARN] MMLU-HS-Math unavailable: {e}")
        return None


def _adapt_hard_mc(items):
    """Convert hard_mc format (valid_indices) to standard format (label)."""
    out = []
    for it in items:
        vi = it.get("valid_indices", [0])
        out.append({
            "prompt": it["prompt"],
            "choices": it["choices"],
            "label": vi[0],
        })
    return out


def load_gpqa(n):
    return _adapt_hard_mc(load_gpqa_diamond(n))


def load_agieval(n):
    return _adapt_hard_mc(load_agieval_gaokao_mathqa(n))


def load_logiqa_wrapped(n):
    return _adapt_hard_mc(load_logiqa(n))


BENCHMARK_LOADERS = [
    ("BoolQ",                  load_boolq),
    ("ARC-C",                  load_arc_c),
    ("CSQA",                   load_csqa),
    ("TruthfulQA",             load_truthfulqa),
    ("MMLU-HS-Math",           load_mmlu_hs_math),
    ("GPQA-Diamond",           load_gpqa),
    ("AGIEval-Gaokao-MathQA",  load_agieval),
    ("LogiQA",                 load_logiqa_wrapped),
]


# ─── 模型加载 ─────────────────────────────────────────────────────────────────
def load_model():
    print("Loading model ...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=DTYPE,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"  Loaded: {n_layers} layers, device={DEVICE}")
    return tok, model, n_layers


# ─── JSD for logit lens velocity ──────────────────────────────────────────────
def _jsd_probs(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-9) -> float:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
    return float((0.5 * (kl_pm + kl_qm)).mean().item())


def _logit_probs_last(h, ln_f, lm_head):
    dev = next(ln_f.parameters()).device
    h_last = h[:, -1:, :].to(dev)
    with torch.no_grad():
        logits = lm_head(ln_f(h_last)).float()
    return F.softmax(logits, dim=-1).squeeze(1).cpu()


# ─── 核心探针 ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def collect_r34_signals(model, input_ids, attention_mask, n_layers):
    """
    Single forward pass with sublayer hooks.
    Returns dict of signal_name -> list[float] (length n_layers).
    """
    base = model.model
    ln_f = base.norm
    lm_head = model.lm_head

    h_inputs = {}       # layer input (before layernorm + attn)
    a_outputs = {}      # self_attn output (before residual add)
    m_outputs = {}      # mlp output (before residual add)
    h_outputs = {}      # layer output

    hooks = []

    for li in range(n_layers):
        def make_pre_hook(idx):
            def fn(_m, args):
                t = args[0] if isinstance(args, tuple) else args
                h_inputs[idx] = t.detach()
            return fn

        def make_attn_hook(idx):
            def fn(_m, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                a_outputs[idx] = t.detach()
            return fn

        def make_mlp_hook(idx):
            def fn(_m, _inp, out):
                m_outputs[idx] = out.detach()
            return fn

        def make_layer_hook(idx):
            def fn(_m, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                h_outputs[idx] = t.detach()
            return fn

        hooks.append(base.layers[li].register_forward_pre_hook(make_pre_hook(li)))
        hooks.append(base.layers[li].self_attn.register_forward_hook(make_attn_hook(li)))
        hooks.append(base.layers[li].mlp.register_forward_hook(make_mlp_hook(li)))
        hooks.append(base.layers[li].register_forward_hook(make_layer_hook(li)))

    _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    for h in hooks:
        h.remove()

    # Build signals
    signals = {spec[0]: [] for spec in SIGNAL_SPECS}

    probs_cache = {}
    for li in range(n_layers):
        hi = h_inputs.get(li)
        al = a_outputs.get(li)
        ml = m_outputs.get(li)
        ho = h_outputs.get(li)

        if hi is None or al is None or ml is None or ho is None:
            for k in signals:
                signals[k].append(float("nan"))
            continue

        h_post_attn = hi + al

        # S1: attn_write_norm
        signals["attn_write_norm"].append(attn_write_norm_last(al, hi))

        # S2: ffn_write_norm
        signals["ffn_write_norm"].append(ffn_write_norm_last(ml, hi))

        # S3: ffn_direction_drift
        if li > 0 and m_outputs.get(li - 1) is not None:
            signals["ffn_direction_drift"].append(
                ffn_direction_drift_last(ml, m_outputs[li - 1])
            )
        else:
            signals["ffn_direction_drift"].append(float("nan"))

        # S4: attn_direction_drift
        if li > 0 and a_outputs.get(li - 1) is not None:
            signals["attn_direction_drift"].append(
                attn_direction_drift_last(al, a_outputs[li - 1])
            )
        else:
            signals["attn_direction_drift"].append(float("nan"))

        # S5: hidden_rotation_rate
        signals["hidden_rotation_rate"].append(hidden_rotation_rate_last(ho, hi))

        # S6: cross_cos_a_m
        signals["cross_cos_a_m"].append(cross_cos_a_m_last(al, ml))

        # S7: attn_ffn_balance
        signals["attn_ffn_balance"].append(attn_ffn_balance_last(al, ml))

        # S8: cross_attn_to_ffn_sensitivity (extra MLP forward)
        layer_mod = base.layers[li]
        signals["cross_attn_to_ffn_sens"].append(
            cross_attn_to_ffn_sensitivity(
                layer_mod.mlp, layer_mod.post_attention_layernorm,
                h_post_attn, hi,
            )
        )

        # S9: cross_attn_to_ffn_direction_shift
        signals["cross_attn_to_ffn_dirshift"].append(
            cross_attn_to_ffn_dir_shift(
                layer_mod.mlp, layer_mod.post_attention_layernorm,
                h_post_attn, hi,
            )
        )

        # S10: logit_lens_jsd_vel
        probs_cache[li] = _logit_probs_last(ho, ln_f, lm_head)
        if li > 0 and (li - 1) in probs_cache:
            signals["logit_lens_jsd_vel"].append(
                _jsd_probs(probs_cache[li], probs_cache[li - 1])
            )
        else:
            signals["logit_lens_jsd_vel"].append(float("nan"))

        # S11: prediction_flip_rate
        if li > 0 and h_outputs.get(li - 1) is not None:
            signals["prediction_flip_rate"].append(
                prediction_flip_rate_last_token(ho, h_outputs[li - 1], ln_f, lm_head)
            )
        else:
            signals["prediction_flip_rate"].append(float("nan"))

        # S12: residual_write_norm
        if li > 0 and h_outputs.get(li - 1) is not None:
            signals["residual_write_norm"].append(
                residual_write_norm(ho, h_outputs[li - 1])
            )
        else:
            signals["residual_write_norm"].append(float("nan"))

    return signals


# ─── 绘图 ─────────────────────────────────────────────────────────────────────
def add_r30_marks(ax, bench_name):
    opt = R30_OPTIMAL.get(bench_name)
    if opt is None:
        return
    t0, t1 = opt["t_start"], opt["t_stop"]
    ax.axvspan(t0, t1, alpha=0.12, color="gold", zorder=0)
    ax.axvline(t0, color="#2ca02c", ls="--", lw=2.0, zorder=4, label=f"t_start={t0}")
    ax.axvline(t1, color="#d62728", ls="--", lw=2.0, zorder=4, label=f"t_stop={t1}")


def plot_benchmark(bench_name, records, n_layers):
    if not records:
        return
    c = BENCH_COLORS.get(bench_name, "gray")
    layers = np.arange(n_layers)

    fig, axes = plt.subplots(3, 4, figsize=(22, 13))
    axes_flat = axes.flatten()

    for ax_i, (key, ylabel, _row) in enumerate(SIGNAL_SPECS):
        ax = axes_flat[ax_i]
        mat = np.array([r["per_layer"][key] for r in records], dtype=np.float64)
        mean_ = np.nanmean(mat, axis=0)
        std_ = np.nanstd(mat, axis=0)

        add_r30_marks(ax, bench_name)
        ax.plot(layers, mean_, color=c, linewidth=2.2, label="mean")
        ax.fill_between(layers, mean_ - std_, mean_ + std_, color=c, alpha=0.2)
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(key, fontsize=10)
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, n_layers - 0.5)

    opt = R30_OPTIMAL.get(bench_name, {})
    t_note = f"R30 T-block [{opt.get('t_start', '?')}, {opt.get('t_stop', '?')})"
    fig.suptitle(
        f"R34 Cross-Memory Signals + {t_note}  |  {bench_name}  N={len(records)}",
        fontsize=13,
    )
    fig.tight_layout()
    safe = bench_name.replace("/", "-")
    out_path = FIGURES_DIR / f"{safe}_r34_signals_vs_layer.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def plot_all_bench_overlay(all_data, n_layers):
    """All benchmarks overlaid on each signal (like R33 profile plots)."""
    layers = np.arange(n_layers)

    fig, axes = plt.subplots(3, 4, figsize=(22, 13))
    axes_flat = axes.flatten()

    for ax_i, (key, ylabel, _row) in enumerate(SIGNAL_SPECS):
        ax = axes_flat[ax_i]
        for bench_name, records in all_data.items():
            if not records:
                continue
            mat = np.array([r["per_layer"][key] for r in records], dtype=np.float64)
            mean_ = np.nanmean(mat, axis=0)
            std_ = np.nanstd(mat, axis=0)
            c = BENCH_COLORS.get(bench_name, "gray")
            ax.plot(layers, mean_, label=bench_name, color=c, linewidth=1.8)
            ax.fill_between(layers, mean_ - std_, mean_ + std_, color=c, alpha=0.1)
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(key, fontsize=10)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, n_layers - 0.5)

    fig.suptitle("R34 Cross-Memory Signals: All Benchmarks Overlay", fontsize=13)
    fig.tight_layout()
    out_path = FIGURES_DIR / "r34_all_benchmarks_overlay.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ─── 主流程 ───────────────────────────────────────────────────────────────────
def main():
    t0_wall = time.time()
    tok, model, n_layers = load_model()

    all_data = {}

    for bench_name, loader_fn in BENCHMARK_LOADERS:
        print(f"\n{'=' * 55}\n{bench_name}")
        try:
            samples = loader_fn(N_SAMPLES)
        except Exception as e:
            print(f"  [SKIP] Failed to load: {e}")
            continue
        if samples is None:
            print(f"  [SKIP] Data unavailable")
            continue
        print(f"  Loaded {len(samples)} samples")

        records = []
        for i, samp in enumerate(samples[:N_SAMPLES]):
            enc = tok(samp["prompt"], return_tensors="pt", add_special_tokens=False)
            iids = enc["input_ids"].to(DEVICE)
            amask = enc.get("attention_mask")
            amask = amask.to(DEVICE) if amask is not None else None

            sigs = collect_r34_signals(model, iids, amask, n_layers)
            rec = {"per_layer": sigs}
            records.append(rec)

            if (i + 1) % 5 == 0 or i == 0:
                sens_mid = sigs["cross_attn_to_ffn_sens"][n_layers // 2]
                cos_mid = sigs["cross_cos_a_m"][n_layers // 2]
                print(
                    f"  [{i+1:2d}/{min(N_SAMPLES, len(samples))}]  "
                    f"sens@mid={sens_mid:.4f}  cos_am@mid={cos_mid:.4f}"
                )

        all_data[bench_name] = records

        # Per-layer stats
        for key, _, _ in SIGNAL_SPECS[:4]:
            mat = np.array([r["per_layer"][key] for r in records], dtype=np.float64)
            valid = mat[:, n_layers // 2]
            valid = valid[np.isfinite(valid)]
            if len(valid) > 0:
                print(f"    {key}@mid: mean={np.mean(valid):.4f} std={np.std(valid):.4f} CV={np.std(valid)/(np.mean(valid)+1e-9):.3f}")

    # ── 保存数据 ──────────────────────────────────────────────────────────────
    full_path = RESULTS_DIR / "r34_cross_memory_data_full.json"
    with open(full_path, "w") as f:
        json.dump(
            {
                "n_layers": n_layers,
                "r30_optimal": R30_OPTIMAL,
                "n_samples": N_SAMPLES,
                "signal_keys": [s[0] for s in SIGNAL_SPECS],
                "benches": {
                    b: [r for r in recs]
                    for b, recs in all_data.items()
                },
            },
            f,
            indent=2,
        )
    print(f"\nSaved data -> {full_path}")

    # ── 统计摘要 ──────────────────────────────────────────────────────────────
    stats = {}
    for bench_name, records in all_data.items():
        if not records:
            continue
        bench_stats = {"n": len(records)}
        for key, _, _ in SIGNAL_SPECS:
            mat = np.array([r["per_layer"][key] for r in records], dtype=np.float64)
            for layer_idx in [n_layers // 4, n_layers // 2, 3 * n_layers // 4]:
                col = mat[:, layer_idx]
                valid = col[np.isfinite(col)]
                if len(valid) > 0:
                    bench_stats[f"{key}@L{layer_idx}_mean"] = float(np.mean(valid))
                    bench_stats[f"{key}@L{layer_idx}_std"] = float(np.std(valid))
        stats[bench_name] = bench_stats

    stats_path = RESULTS_DIR / "r34_cross_memory_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats -> {stats_path}")

    # ── 绘图 ─────────────────────────────────────────────────────────────────
    for bench_name, records in all_data.items():
        plot_benchmark(bench_name, records, n_layers)

    plot_all_bench_overlay(all_data, n_layers)

    elapsed = time.time() - t0_wall
    print(f"\nTotal elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
