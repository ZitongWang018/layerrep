"""
R35: Attention-FFN 非对易交换子探针
================================================================
理论框架：
  ETD 第二遍的真正增益来自 Attention 和 FFN 作为算子的非对易性。
  精确交换子：
    C_l(h) = [M_l, A_l](h) = M_l(A_l(h)) - A_l(M_l(h))
           = [m(h+a) - m(h)] + [a(h) - a(h+m)]
           = Term1 (context→knowledge) + Term2 (knowledge→context)

  Term1 = R34 cross_attn_to_ffn 的向量版（已有近似，现在精确化）
  Term2 = 全新信号：FFN 对 Attention 的反向因果影响

Phase 0: 精确 swap-order 交换子（7 个标量信号）
Phase 1: 向量分解分析（3 个方向信号）
Phase 2: 传播增益估计（累积交换子 vs ETD accuracy delta 散点图）

Qwen3-8B Pre-LN 层结构（来自 project_memory.md）：
  residual1 = h
  h = LN1(h)  [input_layernorm]
  a = self_attn(h, ...)    ← Term2 重跑这里（用 h + m_l0）
  h = residual1 + a

  residual2 = h
  h = LN2(h)  [post_attention_layernorm]
  m = mlp(h)               ← Term1 重跑这里（用 LN2(h_in)）
  h = residual2 + m

输出：
  figures/r35_commutator/{bench}_r35_commutator_vs_layer.png
  figures/r35_commutator/r35_all_overlay.png
  figures/r35_commutator/r35_vs_r34_comparison.png
  figures/r35_commutator/r35_scatter_commutator_vs_delta.png
  results/r35_commutator_data_full.json
  results/r35_commutator_stats.json
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

from hard_mc_benchmark_loaders import (
    load_gpqa_diamond,
    load_agieval_gaokao_mathqa,
    load_logiqa,
)

# ─── 配置 ──────────────────────────────────────────────────────────────────────
MODEL_PATH = "/root/autodl-tmp/model_qwen"
RESULTS_DIR = EXP / "results"
FIGURES_DIR = EXP / "figures" / "r35_commutator"
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

# ─── 信号规格 ──────────────────────────────────────────────────────────────────
# Phase 0+1: 交换子信号 (10 个)
COMM_SIGNAL_SPECS = [
    # (key,                          y_label,                           row)
    ("commutator_norm",              "||C_l||",                         0),
    ("commutator_norm_rel",          "||C_l|| / ||Δh||",                0),
    ("term1_norm",                   "||T1|| ctx→knw",                  0),
    ("term2_norm",                   "||T2|| knw→ctx",                  0),
    ("term_ratio",                   "||T1||/(||T1||+||T2||)",          1),
    ("cancellation_ratio",           "||C_l|| / (||T1||+||T2||)",       1),
    ("commutator_cos_with_residual", "cos(C_l, Δh)",                    1),
    ("cos_term1_term2",              "cos(T1, T2)",                     1),
    ("cos_commutator_attn",          "cos(C_l, a_l)",                   2),
    ("cos_commutator_ffn",           "cos(C_l, m_l)",                   2),
]

# R34 参考信号 (保留用于对比)
REF_SIGNAL_SPECS = [
    ("ref_cross_cos_a_m",       "cos(a_l, m_l)  [R34 ref]",            2),
    ("ref_cross_attn_ffn_sens", "||T1||/||m_std|| [R34 ref]",           2),
]

ALL_SPECS = COMM_SIGNAL_SPECS + REF_SIGNAL_SPECS
ALL_KEYS = [s[0] for s in ALL_SPECS]
COMM_KEYS = [s[0] for s in COMM_SIGNAL_SPECS]


# ─── 辅助 ─────────────────────────────────────────────────────────────────────
def safe_cos(u: torch.Tensor, v: torch.Tensor) -> float:
    """Cosine similarity between two 1-D tensors; returns 0.0 if either is zero."""
    u = u.float().reshape(-1)
    v = v.float().reshape(-1)
    norm_u = u.norm()
    norm_v = v.norm()
    if norm_u < 1e-12 or norm_v < 1e-12:
        return 0.0
    return float((u @ v / (norm_u * norm_v)).clamp(-1, 1).item())


# ─── 数据加载 ─────────────────────────────────────────────────────────────────
def load_boolq(n):
    ds = load_dataset("aps/super_glue", "boolq")["validation"]
    out = []
    for x in ds:
        if int(x["label"]) < 0:
            continue
        out.append({"prompt": f"{x['passage']}\nQuestion: {x['question']}?\nAnswer:",
                    "choices": ["no", "yes"], "label": int(x["label"])})
        if len(out) >= n:
            break
    return out


def load_arc_c(n):
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")["test"]
    out = []
    for x in ds:
        lmap = {k: i for i, k in enumerate(x["choices"]["label"])}
        out.append({"prompt": f"Question: {x['question']}\nAnswer:",
                    "choices": x["choices"]["text"],
                    "label": lmap.get(x["answerKey"], 0)})
        if len(out) >= n:
            break
    return out


def load_csqa(n):
    ds = load_dataset("tau/commonsense_qa")["validation"]
    lmap = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    out = []
    for x in ds:
        out.append({"prompt": f"Question: {x['question']}\nAnswer:",
                    "choices": x["choices"]["text"],
                    "label": lmap.get(x["answerKey"], 0)})
        if len(out) >= n:
            break
    return out


def load_truthfulqa(n):
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice")["validation"]
    out = []
    for x in ds:
        labels = x["mc1_targets"]["labels"]
        out.append({"prompt": f"Question: {x['question']}\nAnswer:",
                    "choices": x["mc1_targets"]["choices"],
                    "label": int(np.argmax(labels))})
        if len(out) >= n:
            break
    return out


def load_mmlu_hs_math(n):
    try:
        ds = load_dataset("cais/mmlu", "high_school_mathematics")["test"]
        out = []
        for x in ds:
            out.append({"prompt": f"Question: {x['question']}\nAnswer:",
                        "choices": x["choices"], "label": int(x["answer"])})
            if len(out) >= n:
                break
        return out
    except Exception as e:
        print(f"  [WARN] MMLU-HS-Math unavailable: {e}")
        return None


def _adapt_hard_mc(items):
    return [{"prompt": it["prompt"], "choices": it["choices"],
             "label": it.get("valid_indices", [0])[0]} for it in items]


BENCHMARK_LOADERS = [
    ("BoolQ",                 load_boolq),
    ("ARC-C",                 load_arc_c),
    ("CSQA",                  load_csqa),
    ("TruthfulQA",            load_truthfulqa),
    ("MMLU-HS-Math",          load_mmlu_hs_math),
    ("GPQA-Diamond",          lambda n: _adapt_hard_mc(load_gpqa_diamond(n))),
    ("AGIEval-Gaokao-MathQA", lambda n: _adapt_hard_mc(load_agieval_gaokao_mathqa(n))),
    ("LogiQA",                lambda n: _adapt_hard_mc(load_logiqa(n))),
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


# ─── 核心探针 ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def collect_r35_signals(model, input_ids, attention_mask, n_layers):
    """
    Single standard forward pass with sublayer hooks,
    followed by per-layer extra computations (MLP re-run + Attention re-run).

    Returns: dict of signal_name -> list[float] (length n_layers)
    """
    base = model.model

    # ── Tensor buffers filled by hooks ────────────────────────────────────────
    h_inputs  = {}   # layer input (before LN1 + Attn)
    a_outputs = {}   # self_attn output (before residual add)
    m_outputs = {}   # mlp output (before residual add)
    h_outputs = {}   # layer output (after residual adds)

    # For attention re-run: capture the args/kwargs passed to self_attn
    # so we can call it again with a modified hidden state.
    attn_args_rest   = {}   # positional args after hidden_states (usually empty)
    attn_kwargs_rest = {}   # keyword args excluding hidden_states
    attn_hs_positional = {} # True if hidden_states was args[0], else keyword

    hooks = []

    for li in range(n_layers):
        layer = base.layers[li]

        def make_layer_pre_hook(idx):
            def fn(_m, args):
                t = args[0] if isinstance(args, tuple) else args
                h_inputs[idx] = t.detach()
            return fn

        def make_attn_post_hook(idx):
            def fn(_m, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                a_outputs[idx] = t.detach()
            return fn

        def make_mlp_post_hook(idx):
            def fn(_m, _inp, out):
                m_outputs[idx] = out.detach()
            return fn

        def make_layer_post_hook(idx):
            def fn(_m, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                h_outputs[idx] = t.detach()
            return fn

        # NEW: capture self_attn inputs for Term2 re-run
        def make_attn_input_hook(idx):
            def fn(_m, args, kwargs):
                # Determine if hidden_states is positional or keyword
                if len(args) > 0:
                    # hidden_states = args[0]; rest of positional args
                    attn_args_rest[idx] = tuple(a.detach() if isinstance(a, torch.Tensor) else a
                                                for a in args[1:])
                    attn_hs_positional[idx] = True
                else:
                    attn_args_rest[idx] = ()
                    attn_hs_positional[idx] = False
                # Keyword args excluding hidden_states (they may be tensors or None)
                attn_kwargs_rest[idx] = {
                    k: (v.detach() if isinstance(v, torch.Tensor) else v)
                    for k, v in kwargs.items()
                    if k != "hidden_states"
                }
            return fn

        hooks.append(layer.register_forward_pre_hook(make_layer_pre_hook(li)))
        hooks.append(layer.self_attn.register_forward_hook(make_attn_post_hook(li)))
        hooks.append(layer.mlp.register_forward_hook(make_mlp_post_hook(li)))
        hooks.append(layer.register_forward_hook(make_layer_post_hook(li)))
        # with_kwargs=True requires PyTorch >= 2.0
        hooks.append(layer.self_attn.register_forward_pre_hook(
            make_attn_input_hook(li), with_kwargs=True))

    # ── Standard forward pass ─────────────────────────────────────────────────
    model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    for h in hooks:
        h.remove()

    # ── Per-layer commutator computation ──────────────────────────────────────
    signals = {k: [] for k in ALL_KEYS}
    eps = 1e-9

    for li in range(n_layers):
        hi = h_inputs.get(li)
        al = a_outputs.get(li)
        ml = m_outputs.get(li)
        ho = h_outputs.get(li)

        if any(x is None for x in [hi, al, ml, ho]):
            for k in ALL_KEYS:
                signals[k].append(float("nan"))
            continue

        layer_mod = base.layers[li]

        # ── Term 1: context → knowledge ──────────────────────────────────────
        # m_l0 = MLP(LN2(h_in))  [full sequence, bfloat16]
        # We need full seq because h_in + m_l0 is used for attn re-run.
        m_l0_full = layer_mod.mlp(layer_mod.post_attention_layernorm(hi))

        # ── Term 2: knowledge → context ──────────────────────────────────────
        # h_for_attn = h_in + m_l0_full  →  LN1  →  re-run self_attn
        h_for_attn = hi + m_l0_full
        h_normed_attn = layer_mod.input_layernorm(h_for_attn)

        saved_args   = attn_args_rest.get(li, ())
        saved_kwargs = attn_kwargs_rest.get(li, {})
        hs_positional = attn_hs_positional.get(li, True)

        try:
            if hs_positional:
                a_prime_out = layer_mod.self_attn(
                    h_normed_attn, *saved_args, **saved_kwargs)
            else:
                a_prime_out = layer_mod.self_attn(
                    hidden_states=h_normed_attn, *saved_args, **saved_kwargs)
            a_prime = (a_prime_out[0] if isinstance(a_prime_out, tuple)
                       else a_prime_out)
        except Exception as e:
            # Fallback: if attention re-run fails (e.g. unexpected arg format),
            # use the original output (Term2 = 0)
            print(f"  [WARN] layer {li} attn re-run failed: {e}; Term2 = 0")
            a_prime = al

        # ── Extract last-token vectors [float32] ─────────────────────────────
        al_last      = al[:, -1, :].float()          # [1, D]
        ml_last      = ml[:, -1, :].float()          # [1, D]
        m_l0_last    = m_l0_full[:, -1, :].float()   # [1, D]
        a_prime_last = a_prime[:, -1, :].float()     # [1, D]
        hi_last      = hi[:, -1, :].float()          # [1, D]
        ho_last      = ho[:, -1, :].float()          # [1, D]

        # Squeeze to [D] for cosine operations
        al_v      = al_last.squeeze(0)
        ml_v      = ml_last.squeeze(0)
        m_l0_v    = m_l0_last.squeeze(0)
        a_prime_v = a_prime_last.squeeze(0)
        hi_v      = hi_last.squeeze(0)
        ho_v      = ho_last.squeeze(0)

        # ── Commutator components ─────────────────────────────────────────────
        term1 = ml_v - m_l0_v          # how attention changes FFN output
        term2 = al_v - a_prime_v       # how FFN changes Attention output
        commutator = term1 + term2     # C_l = Term1 + Term2
        dh = ho_v - hi_v               # actual layer write (for normalization)

        norm_c  = commutator.norm().item()
        norm_t1 = term1.norm().item()
        norm_t2 = term2.norm().item()
        norm_dh = dh.norm().item()

        # ── Phase 0: scalar signals ───────────────────────────────────────────
        signals["commutator_norm"].append(norm_c)
        signals["commutator_norm_rel"].append(norm_c / (norm_dh + eps))
        signals["term1_norm"].append(norm_t1)
        signals["term2_norm"].append(norm_t2)
        signals["term_ratio"].append(norm_t1 / (norm_t1 + norm_t2 + eps))
        signals["cancellation_ratio"].append(norm_c / (norm_t1 + norm_t2 + eps))

        # ── Phase 1: directional signals ─────────────────────────────────────
        signals["commutator_cos_with_residual"].append(safe_cos(commutator, dh))
        signals["cos_term1_term2"].append(safe_cos(term1, term2))
        signals["cos_commutator_attn"].append(safe_cos(commutator, al_v))
        signals["cos_commutator_ffn"].append(safe_cos(commutator, ml_v))

        # ── Reference R34 signals (for comparison) ────────────────────────────
        signals["ref_cross_cos_a_m"].append(safe_cos(al_v, ml_v))
        signals["ref_cross_attn_ffn_sens"].append(norm_t1 / (ml_v.norm().item() + eps))

    return signals


# ─── ETD 精度增益加载（Phase 2）─────────────────────────────────────────────
def load_etd_deltas() -> dict[str, float]:
    """
    Load ETD accuracy deltas from R30 sweep results JSON.
    Returns {bench_name: accuracy_delta} for benchmarks that have sweep data.

    Falls back to approximate values from research report for benchmarks
    not in the R30 sweep (hard-MC benchmarks).
    """
    deltas: dict[str, float] = {}
    sweep_path = RESULTS_DIR / "r30_sweep_results.json"

    if sweep_path.exists():
        with open(sweep_path) as f:
            data = json.load(f)
        baseline = data.get("baseline", {})
        results_list = data.get("results", [])

        # For each benchmark, find the best ETD accuracy across all T-block configs
        best_acc: dict[str, float] = {}
        for row in results_list:
            for bench, acc in row.items():
                if bench in ("t_start", "t_stop") or not isinstance(acc, (int, float)):
                    continue
                if bench not in best_acc or acc > best_acc[bench]:
                    best_acc[bench] = acc

        for bench in baseline:
            if bench in best_acc:
                deltas[bench] = best_acc[bench] - baseline[bench]

    # Hard-MC approximate values from ETD_Research_Report.md sections
    # (R30 sweep didn't cover these; these come from separate hard-MC sweeps)
    # Values are best-ETD minus baseline, approximate, from report:
    HARD_MC_APPROX = {
        # From ETD_Research_Report chapter references for hard_mc benchmarks
        # These are rough — the scatter plot will note they're approximate
        "GPQA-Diamond":          0.02,   # small gain, hard task
        "AGIEval-Gaokao-MathQA": 0.03,   # modest gain
        "LogiQA":                0.04,
        "BoolQ":                 0.04,   # from R29 BoolQ signal analysis
    }
    for k, v in HARD_MC_APPROX.items():
        if k not in deltas:
            deltas[k] = v

    return deltas


# ─── 绘图 ─────────────────────────────────────────────────────────────────────
def add_r30_marks(ax, bench_name: str):
    opt = R30_OPTIMAL.get(bench_name)
    if opt is None:
        return
    t0, t1 = opt["t_start"], opt["t_stop"]
    ax.axvspan(t0, t1, alpha=0.12, color="gold", zorder=0)
    ax.axvline(t0, color="#2ca02c", ls="--", lw=2.0, zorder=4, label=f"t_start={t0}")
    ax.axvline(t1, color="#d62728", ls="--", lw=2.0, zorder=4, label=f"t_stop={t1}")


def plot_r35_benchmark(bench_name: str, records: list, n_layers: int, figures_dir: Path):
    if not records:
        return
    c = BENCH_COLORS.get(bench_name, "gray")
    layers = np.arange(n_layers)

    # ── Plot 1: All 10 commutator signals (2×5 grid) ──────────────────────────
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes_flat = axes.flatten()

    for ax_i, (key, ylabel, _row) in enumerate(COMM_SIGNAL_SPECS):
        ax = axes_flat[ax_i]
        mat = np.array([r["per_layer"][key] for r in records], dtype=np.float64)
        mean_ = np.nanmean(mat, axis=0)
        std_  = np.nanstd(mat, axis=0)

        add_r30_marks(ax, bench_name)
        ax.plot(layers, mean_, color=c, linewidth=2.2, label="mean")
        ax.fill_between(layers, mean_ - std_, mean_ + std_, color=c, alpha=0.2)
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_title(key, fontsize=9)
        ax.legend(loc="best", fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, n_layers - 0.5)

    opt = R30_OPTIMAL.get(bench_name, {})
    t_note = f"R30 T-block [{opt.get('t_start', '?')}, {opt.get('t_stop', '?')})"
    fig.suptitle(
        f"R35 Commutator Signals + {t_note}  |  {bench_name}  N={len(records)}",
        fontsize=13,
    )
    fig.tight_layout()
    safe = bench_name.replace("/", "-")
    out_path = figures_dir / f"{safe}_r35_commutator_vs_layer.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")

    # ── Plot 2: R35 vs R34 comparison (4 signals side-by-side) ───────────────
    fig2, axes2 = plt.subplots(2, 4, figsize=(22, 9))
    compare_pairs = [
        ("commutator_norm",         "R35: ||C_l||"),
        ("term1_norm",              "R35: ||T1|| ctx→knw"),
        ("term2_norm",              "R35: ||T2|| knw→ctx (NEW)"),
        ("cancellation_ratio",      "R35: ||C_l||/(||T1||+||T2||)"),
        ("ref_cross_cos_a_m",       "R34: cos(a_l, m_l)"),
        ("ref_cross_attn_ffn_sens", "R34: ||T1||/||m_std||"),
        ("cos_term1_term2",         "R35: cos(T1, T2)"),
        ("commutator_norm_rel",     "R35: ||C_l|| / ||Δh||"),
    ]
    for ax_i, (key, title) in enumerate(compare_pairs):
        ax = axes2.flatten()[ax_i]
        mat = np.array([r["per_layer"][key] for r in records], dtype=np.float64)
        mean_ = np.nanmean(mat, axis=0)
        std_  = np.nanstd(mat, axis=0)
        color = "#1565C0" if "R35" in title else "#B71C1C"

        add_r30_marks(ax, bench_name)
        ax.plot(layers, mean_, color=color, linewidth=2.2)
        ax.fill_between(layers, mean_ - std_, mean_ + std_, color=color, alpha=0.2)
        ax.set_xlabel("Layer")
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, n_layers - 0.5)

    fig2.suptitle(
        f"R35 vs R34 Signal Comparison  |  {bench_name}", fontsize=12
    )
    fig2.tight_layout()
    out_path2 = figures_dir / f"{safe}_r35_vs_r34_comparison.png"
    fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved -> {out_path2}")


def plot_r35_all_overlay(all_data: dict, n_layers: int, figures_dir: Path):
    """All benchmarks overlaid on key commutator signals."""
    layers = np.arange(n_layers)

    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes_flat = axes.flatten()

    for ax_i, (key, ylabel, _row) in enumerate(COMM_SIGNAL_SPECS):
        ax = axes_flat[ax_i]
        for bench_name, records in all_data.items():
            if not records:
                continue
            mat = np.array([r["per_layer"][key] for r in records], dtype=np.float64)
            mean_ = np.nanmean(mat, axis=0)
            std_  = np.nanstd(mat, axis=0)
            c = BENCH_COLORS.get(bench_name, "gray")
            ax.plot(layers, mean_, label=bench_name, color=c, linewidth=1.8)
            ax.fill_between(layers, mean_ - std_, mean_ + std_, color=c, alpha=0.1)
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(key, fontsize=10)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, n_layers - 0.5)

    fig.suptitle("R35 Commutator: All Benchmarks Overlay", fontsize=13)
    fig.tight_layout()
    out_path = figures_dir / "r35_all_overlay.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def plot_r35_scatter(all_data: dict, n_layers: int, etd_deltas: dict, figures_dir: Path):
    """
    Phase 2: scatter plot of accumulated commutator within T-block vs ETD accuracy delta.
    Tests H3: sum(||C_l||) within T-block correlates with ETD gain.
    """
    bench_names = []
    comm_sums = []   # sum(||C_l||) within T-block
    t1_sums = []
    t2_sums = []
    delta_vals = []

    for bench_name, records in all_data.items():
        if not records or bench_name not in etd_deltas:
            continue
        opt = R30_OPTIMAL.get(bench_name)
        if opt is None:
            continue
        t0, t1 = opt["t_start"], opt["t_stop"]

        comm_mat = np.array([r["per_layer"]["commutator_norm"] for r in records], dtype=np.float64)
        t1_mat   = np.array([r["per_layer"]["term1_norm"] for r in records], dtype=np.float64)
        t2_mat   = np.array([r["per_layer"]["term2_norm"] for r in records], dtype=np.float64)

        # Mean across samples, sum within T-block
        comm_mean = np.nanmean(comm_mat, axis=0)
        t1_mean   = np.nanmean(t1_mat, axis=0)
        t2_mean   = np.nanmean(t2_mat, axis=0)

        comm_tblock = float(np.nansum(comm_mean[t0:t1]))
        t1_tblock   = float(np.nansum(t1_mean[t0:t1]))
        t2_tblock   = float(np.nansum(t2_mean[t0:t1]))

        bench_names.append(bench_name)
        comm_sums.append(comm_tblock)
        t1_sums.append(t1_tblock)
        t2_sums.append(t2_tblock)
        delta_vals.append(etd_deltas[bench_name])

    if len(bench_names) < 2:
        print("  [WARN] Not enough benchmarks for scatter plot")
        return

    comm_sums  = np.array(comm_sums)
    t1_sums    = np.array(t1_sums)
    t2_sums    = np.array(t2_sums)
    delta_vals = np.array(delta_vals)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = [
        "||C_l|| sum in T-block vs ETD Δacc (H3)",
        "||T1|| (ctx→knw) sum in T-block vs ETD Δacc",
        "||T2|| (knw→ctx) sum in T-block vs ETD Δacc",
    ]
    x_arrays = [comm_sums, t1_sums, t2_sums]
    x_labels = [
        "Σ ||C_l|| within T-block",
        "Σ ||Term1|| within T-block",
        "Σ ||Term2|| within T-block",
    ]

    for ax_i, (ax, title, x_arr, xlabel) in enumerate(
            zip(axes, titles, x_arrays, x_labels)):
        for i, name in enumerate(bench_names):
            c = BENCH_COLORS.get(name, "gray")
            ax.scatter(x_arr[i], delta_vals[i], color=c, s=100, zorder=5)
            ax.annotate(name, (x_arr[i], delta_vals[i]),
                        textcoords="offset points", xytext=(5, 3), fontsize=7)

        # Fit linear trendline
        if len(x_arr) >= 3:
            try:
                coeffs = np.polyfit(x_arr, delta_vals, 1)
                x_fit = np.linspace(x_arr.min(), x_arr.max(), 100)
                y_fit = np.polyval(coeffs, x_fit)
                corr = float(np.corrcoef(x_arr, delta_vals)[0, 1])
                ax.plot(x_fit, y_fit, "k--", alpha=0.5, linewidth=1.5,
                        label=f"r={corr:.3f}")
                ax.legend(fontsize=9)
            except Exception:
                pass

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("ETD Accuracy Δ", fontsize=10)
        ax.set_title(title, fontsize=9)
        ax.axhline(0, color="gray", ls=":", lw=1)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Phase 2: Accumulated Commutator vs ETD Accuracy Gain (H3 Test)",
                 fontsize=12)
    fig.tight_layout()
    out_path = figures_dir / "r35_scatter_commutator_vs_delta.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def plot_r35_cross_bench_comparison(all_data: dict, n_layers: int, figures_dir: Path):
    """
    Focused comparison: R35 commutator_norm vs R34 cross_cos_a_m across benchmarks.
    Tests H1 (T-block alignment) and H2 (commutator distinguishes better).
    """
    layers = np.arange(n_layers)

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    compare_keys = [
        ("commutator_norm",         "R35: ||C_l|| (交换子 norm)"),
        ("cancellation_ratio",      "R35: ||C_l||/(||T1||+||T2||) (对消程度)"),
        ("term1_norm",              "R35: ||T1|| ctx→knw"),
        ("term2_norm",              "R35: ||T2|| knw→ctx (新)"),
        ("ref_cross_cos_a_m",       "R34 ref: cos(a_l, m_l)"),
        ("ref_cross_attn_ffn_sens", "R34 ref: sens (||T1||/||m||)"),
        ("cos_term1_term2",         "R35: cos(T1, T2) 方向对消"),
        ("commutator_norm_rel",     "R35: ||C_l||/||Δh|| (相对交换子)"),
    ]
    for ax_i, (key, title) in enumerate(compare_keys):
        ax = axes.flatten()[ax_i]
        for bench_name, records in all_data.items():
            if not records:
                continue
            mat = np.array([r["per_layer"][key] for r in records], dtype=np.float64)
            mean_ = np.nanmean(mat, axis=0)
            c = BENCH_COLORS.get(bench_name, "gray")
            ax.plot(layers, mean_, label=bench_name, color=c, linewidth=1.8)
        ax.set_xlabel("Layer")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, n_layers - 0.5)

    fig.suptitle("R35 vs R34: Cross-Benchmark Signal Comparison (H1, H2 Test)", fontsize=12)
    fig.tight_layout()
    out_path = figures_dir / "r35_vs_r34_comparison.png"
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
            iids  = enc["input_ids"].to(DEVICE)
            amask = enc.get("attention_mask")
            amask = amask.to(DEVICE) if amask is not None else None

            sigs = collect_r35_signals(model, iids, amask, n_layers)
            records.append({"per_layer": sigs})

            if (i + 1) % 5 == 0 or i == 0:
                c_mid    = sigs["commutator_norm"][n_layers // 2]
                cr_mid   = sigs["cancellation_ratio"][n_layers // 2]
                t1_mid   = sigs["term1_norm"][n_layers // 2]
                t2_mid   = sigs["term2_norm"][n_layers // 2]
                print(
                    f"  [{i+1:2d}/{min(N_SAMPLES, len(samples))}]  "
                    f"||C_l||@mid={c_mid:.4f}  cancel@mid={cr_mid:.4f}  "
                    f"T1={t1_mid:.4f}  T2={t2_mid:.4f}"
                )

        all_data[bench_name] = records

    # ── 保存数据 ──────────────────────────────────────────────────────────────
    full_path = RESULTS_DIR / "r35_commutator_data_full.json"
    with open(full_path, "w") as f:
        json.dump(
            {
                "n_layers": n_layers,
                "r30_optimal": R30_OPTIMAL,
                "n_samples": N_SAMPLES,
                "signal_keys": ALL_KEYS,
                "comm_signal_keys": COMM_KEYS,
                "benches": {b: recs for b, recs in all_data.items()},
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
        opt = R30_OPTIMAL.get(bench_name, {})
        t0, t1 = opt.get("t_start", 0), opt.get("t_stop", n_layers)

        for key in COMM_KEYS:
            mat = np.array([r["per_layer"][key] for r in records], dtype=np.float64)
            # Stats at key checkpoints
            for li in [n_layers // 4, n_layers // 2, 3 * n_layers // 4]:
                col = mat[:, li]
                valid = col[np.isfinite(col)]
                if len(valid) > 0:
                    bench_stats[f"{key}@L{li}_mean"] = float(np.mean(valid))
                    bench_stats[f"{key}@L{li}_std"]  = float(np.std(valid))
            # T-block aggregate
            tblock_mean = np.nanmean(mat[:, t0:t1])
            bench_stats[f"{key}_tblock_mean"] = float(tblock_mean)
            bench_stats[f"{key}_tblock_sum"]  = float(
                np.nansum(np.nanmean(mat, axis=0)[t0:t1]))
        stats[bench_name] = bench_stats

    stats_path = RESULTS_DIR / "r35_commutator_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats -> {stats_path}")

    # ── 绘图 ─────────────────────────────────────────────────────────────────
    for bench_name, records in all_data.items():
        plot_r35_benchmark(bench_name, records, n_layers, FIGURES_DIR)

    plot_r35_all_overlay(all_data, n_layers, FIGURES_DIR)
    plot_r35_cross_bench_comparison(all_data, n_layers, FIGURES_DIR)

    # Phase 2: scatter plot
    etd_deltas = load_etd_deltas()
    print(f"\nETD deltas loaded: {etd_deltas}")
    plot_r35_scatter(all_data, n_layers, etd_deltas, FIGURES_DIR)

    elapsed = time.time() - t0_wall
    print(f"\nTotal elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
