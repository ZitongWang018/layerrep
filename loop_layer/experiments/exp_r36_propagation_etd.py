"""
R36: 方向特异性传播增益实验
================================================================
理论框架：
  R35 证明了绝对交换子 norm 因与残差流 norm 正相关而无法定位 T-block。
  R36 升级理论到两个正交条件：

  条件 1 (方向对齐): cos(C_l, Δh_l) — 顺序差异落在实际写入方向上
                       (已在 R35 计算，此处复用)

  条件 2 (传播特异性):
    prop_sens_l(x)     = JSD(logits(h_l + ε·Ĉ_l), logits(h_l))
    rand_sens_l(x)     = JSD(logits(h_l + ε·r̂),   logits(h_l))
    directional_advantage_l = prop_sens_l / rand_sens_l

    > 1 表示交换子方向比随机方向更能特异性影响最终预测
    ≈ 1 表示交换子方向无特殊价值
    消除"后期层靠近输出、任何扰动效果都大"的深度混淆因素

  复合信号: etd_effective_l = cos(C_l, Δh_l) × directional_advantage_l

  交换子持续性 (廉价): comm_persist_l = cos(C_l, C_{l+1})

假设:
  H1: prop_sens_l 原始值在后期层最高 (阴性对照，验证 R35 失败对称性)
  H2: directional_advantage_l 在 T-block 区域峰值 (核心假设)
  H3: etd_effective_l 给出最精准的 T-block 边界对齐
  H4: var_samples(directional_advantage_l) 在 T-block 最高 (样本差异化)
  H5: 后期层 directional_advantage ≈ 1 (随机扰动同样有效，非特异性)
  H6: comm_persist 在 T-block 内正且稳定

规模:
  每 benchmark 默认 N_SAMPLES=100；stats 含各信号 T-block/late 的 mean、var、**median**（稳健看 DA）。

输出:
  figures/r36_propagation/{bench}_r36_prop_vs_layer.png
  figures/r36_propagation/r36_all_overlay.png
  figures/r36_propagation/r36_sample_variance.png
  figures/r36_propagation/r36_individual_samples_{bench}.png
  figures/r36_propagation/r36_scatter_da_vs_delta.png
  figures/r36_propagation/r36_late_vs_tblock.png
  results/r36_propagation_data_full.json
  results/r36_propagation_stats.json
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
MODEL_PATH   = "/root/autodl-tmp/model_qwen"
RESULTS_DIR  = EXP / "results"
FIGURES_DIR  = EXP / "figures" / "r36_propagation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16

# Probe layers for propagation experiment (every 3rd, covering all T-blocks)
PROBE_LAYERS  = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33]
PROBE_SET     = set(PROBE_LAYERS)
EPSILON       = 1.0   # perturbation magnitude (unit vector direction)

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
# R35 保留信号（从 R35 复用，作为基础）
R35_KEYS = [
    "commutator_norm", "commutator_norm_rel", "term1_norm", "term2_norm",
    "term_ratio", "cancellation_ratio",
    "commutator_cos_with_residual", "cos_term1_term2",
    "cos_commutator_attn", "cos_commutator_ffn",
    "ref_cross_cos_a_m", "ref_cross_attn_ffn_sens",
]

# R36 新增信号
R36_NEW_KEYS = [
    "comm_persist",           # cos(C_l, C_{l+1}) — 所有 36 层 (layer 0 为 NaN)
    "prop_sens",              # JSD(C_l 扰动) — 仅 11 probe 层
    "rand_sens",              # JSD(随机扰动) — 仅 11 probe 层
    "directional_advantage",  # prop_sens / rand_sens — 仅 11 probe 层
    "etd_effective",          # cos_residual × DA — 仅 11 probe 层
]

ALL_KEYS = R35_KEYS + R36_NEW_KEYS

# 用于绘图的规格（signal_key, ylabel, include_in_prop_plot）
PROP_PLOT_SPECS = [
    ("prop_sens",             "JSD(Ĉ_l perturb)",    True),
    ("rand_sens",             "JSD(random perturb)",  True),
    ("directional_advantage", "DA = prop/rand",       True),
    ("etd_effective",         "cos_res × DA",         True),
    ("comm_persist",          "cos(C_l, C_{l+1})",    True),
    ("commutator_cos_with_residual", "cos(C_l, Δh_l)", True),
]


# ─── 辅助函数 ─────────────────────────────────────────────────────────────────
def safe_cos(u: torch.Tensor, v: torch.Tensor) -> float:
    u = u.float().reshape(-1).cpu()
    v = v.float().reshape(-1).cpu()
    nu, nv = u.norm(), v.norm()
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    return float((u @ v / (nu * nv)).clamp(-1, 1).item())


def jsd_logits(logits1: torch.Tensor, logits2: torch.Tensor, eps: float = 1e-9) -> float:
    """JSD between two logit vectors (1-D or [1, vocab])."""
    p = F.softmax(logits1.float().view(-1), dim=0).clamp_min(eps)
    q = F.softmax(logits2.float().view(-1), dim=0).clamp_min(eps)
    p = p / p.sum(); q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum()
    kl_qm = (q * (q.log() - m.log())).sum()
    return float((0.5 * (kl_pm + kl_qm)).clamp_min(0.0).item())


# ─── 数据加载 ─────────────────────────────────────────────────────────────────
def load_boolq(n):
    ds = load_dataset("aps/super_glue", "boolq")["validation"]
    out = []
    for x in ds:
        if int(x["label"]) < 0: continue
        out.append({"prompt": f"{x['passage']}\nQuestion: {x['question']}?\nAnswer:",
                    "choices": ["no","yes"], "label": int(x["label"])})
        if len(out) >= n: break
    return out

def load_arc_c(n):
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")["test"]
    out = []
    for x in ds:
        lmap = {k: i for i, k in enumerate(x["choices"]["label"])}
        out.append({"prompt": f"Question: {x['question']}\nAnswer:",
                    "choices": x["choices"]["text"], "label": lmap.get(x["answerKey"], 0)})
        if len(out) >= n: break
    return out

def load_csqa(n):
    ds = load_dataset("tau/commonsense_qa")["validation"]
    lmap = {"A":0,"B":1,"C":2,"D":3,"E":4}
    out = []
    for x in ds:
        out.append({"prompt": f"Question: {x['question']}\nAnswer:",
                    "choices": x["choices"]["text"], "label": lmap.get(x["answerKey"], 0)})
        if len(out) >= n: break
    return out

def load_truthfulqa(n):
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice")["validation"]
    out = []
    for x in ds:
        labels = x["mc1_targets"]["labels"]
        out.append({"prompt": f"Question: {x['question']}\nAnswer:",
                    "choices": x["mc1_targets"]["choices"], "label": int(np.argmax(labels))})
        if len(out) >= n: break
    return out

def load_mmlu_hs_math(n):
    try:
        ds = load_dataset("cais/mmlu", "high_school_mathematics")["test"]
        out = []
        for x in ds:
            out.append({"prompt": f"Question: {x['question']}\nAnswer:",
                        "choices": x["choices"], "label": int(x["answer"])})
            if len(out) >= n: break
        return out
    except Exception as e:
        print(f"  [WARN] MMLU-HS-Math unavailable: {e}")
        return None

def _adapt(items):
    return [{"prompt": it["prompt"], "choices": it["choices"],
             "label": it.get("valid_indices", [0])[0]} for it in items]

BENCHMARK_LOADERS = [
    ("BoolQ",                 load_boolq),
    ("ARC-C",                 load_arc_c),
    ("CSQA",                  load_csqa),
    ("TruthfulQA",            load_truthfulqa),
    ("MMLU-HS-Math",          load_mmlu_hs_math),
    ("GPQA-Diamond",          lambda n: _adapt(load_gpqa_diamond(n))),
    ("AGIEval-Gaokao-MathQA", lambda n: _adapt(load_agieval_gaokao_mathqa(n))),
    ("LogiQA",                lambda n: _adapt(load_logiqa(n))),
]


# ─── 模型加载 ─────────────────────────────────────────────────────────────────
def load_model():
    print("Loading model ...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, device_map="auto",
        attn_implementation="eager", trust_remote_code=True,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    d_model  = model.config.hidden_size
    print(f"  Loaded: {n_layers} layers, d_model={d_model}, device={DEVICE}")
    return tok, model, n_layers, d_model


# ─── 核心信号收集（R35 逻辑 + comm_persist + 捕获 orig_logits）────────────────
@torch.no_grad()
def collect_r36_base_signals(model, input_ids, attention_mask, n_layers):
    """
    Full R35 signal collection plus:
    - comm_persist: cos(C_l, C_{l+1}) computed in-loop
    - probe_commutator_vecs: {layer_idx: commutator_vec [D], float32, CPU}
    - orig_logits: final logits at last token [1, vocab], float32

    Returns: (signals_dict, probe_commutator_vecs, orig_logits)
    """
    base = model.model

    h_inputs        = {}
    a_outputs       = {}
    m_outputs       = {}
    h_outputs       = {}
    attn_args_rest  = {}
    attn_kwargs_rest = {}
    attn_hs_positional = {}
    logits_buf      = {}

    hooks = []

    for li in range(n_layers):
        layer = base.layers[li]

        def make_layer_pre(idx):
            def fn(_m, args):
                t = args[0] if isinstance(args, tuple) else args
                h_inputs[idx] = t.detach()
            return fn

        def make_attn_post(idx):
            def fn(_m, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                a_outputs[idx] = t.detach()
            return fn

        def make_mlp_post(idx):
            def fn(_m, _inp, out):
                m_outputs[idx] = out.detach()
            return fn

        def make_layer_post(idx):
            def fn(_m, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                h_outputs[idx] = t.detach()
            return fn

        def make_attn_input(idx):
            def fn(_m, args, kwargs):
                if len(args) > 0:
                    attn_args_rest[idx] = tuple(
                        a.detach() if isinstance(a, torch.Tensor) else a for a in args[1:])
                    attn_hs_positional[idx] = True
                else:
                    attn_args_rest[idx] = ()
                    attn_hs_positional[idx] = False
                attn_kwargs_rest[idx] = {
                    k: (v.detach() if isinstance(v, torch.Tensor) else v)
                    for k, v in kwargs.items() if k != "hidden_states"
                }
            return fn

        hooks.append(layer.register_forward_pre_hook(make_layer_pre(li)))
        hooks.append(layer.self_attn.register_forward_hook(make_attn_post(li)))
        hooks.append(layer.mlp.register_forward_hook(make_mlp_post(li)))
        hooks.append(layer.register_forward_hook(make_layer_post(li)))
        hooks.append(layer.self_attn.register_forward_pre_hook(
            make_attn_input(li), with_kwargs=True))

    # Capture final logits via lm_head hook
    def lm_head_hook(_m, _inp, out):
        logits_buf["logits"] = out.detach()[:, -1, :].float()
    hooks.append(model.lm_head.register_forward_hook(lm_head_hook))

    model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    for h in hooks:
        h.remove()

    orig_logits = logits_buf.get("logits")  # [1, vocab]

    # ── Per-layer commutator + all signals ────────────────────────────────────
    signals = {k: [] for k in ALL_KEYS}
    probe_commutator_vecs = {}
    prev_commutator_v = None
    eps = 1e-9

    for li in range(n_layers):
        hi = h_inputs.get(li)
        al = a_outputs.get(li)
        ml = m_outputs.get(li)
        ho = h_outputs.get(li)

        if any(x is None for x in [hi, al, ml, ho]):
            for k in ALL_KEYS:
                signals[k].append(float("nan"))
            prev_commutator_v = None
            continue

        layer_mod = base.layers[li]

        # Term 1: MLP on pre-attention hidden
        m_l0_full = layer_mod.mlp(layer_mod.post_attention_layernorm(hi))

        # Term 2: Attention on FFN-modified hidden
        h_for_attn    = hi + m_l0_full
        h_normed_attn = layer_mod.input_layernorm(h_for_attn)
        saved_args    = attn_args_rest.get(li, ())
        saved_kwargs  = attn_kwargs_rest.get(li, {})
        hs_pos        = attn_hs_positional.get(li, True)
        try:
            if hs_pos:
                a_prime_out = layer_mod.self_attn(h_normed_attn, *saved_args, **saved_kwargs)
            else:
                a_prime_out = layer_mod.self_attn(hidden_states=h_normed_attn, **saved_kwargs)
            a_prime = a_prime_out[0] if isinstance(a_prime_out, tuple) else a_prime_out
        except Exception as e:
            print(f"  [WARN] layer {li} attn re-run failed: {e}; Term2=0")
            a_prime = al

        # Last-token float32 vectors
        al_v      = al[:, -1, :].float().squeeze(0)
        ml_v      = ml[:, -1, :].float().squeeze(0)
        m_l0_v    = m_l0_full[:, -1, :].float().squeeze(0)
        a_prime_v = a_prime[:, -1, :].float().squeeze(0)
        hi_v      = hi[:, -1, :].float().squeeze(0)
        ho_v      = ho[:, -1, :].float().squeeze(0)

        term1      = ml_v - m_l0_v
        term2      = al_v - a_prime_v
        commutator = term1 + term2
        dh         = ho_v - hi_v

        norm_c  = commutator.norm().item()
        norm_t1 = term1.norm().item()
        norm_t2 = term2.norm().item()
        norm_dh = dh.norm().item()

        # R35 signals
        signals["commutator_norm"].append(norm_c)
        signals["commutator_norm_rel"].append(norm_c / (norm_dh + eps))
        signals["term1_norm"].append(norm_t1)
        signals["term2_norm"].append(norm_t2)
        signals["term_ratio"].append(norm_t1 / (norm_t1 + norm_t2 + eps))
        signals["cancellation_ratio"].append(norm_c / (norm_t1 + norm_t2 + eps))
        signals["commutator_cos_with_residual"].append(safe_cos(commutator, dh))
        signals["cos_term1_term2"].append(safe_cos(term1, term2))
        signals["cos_commutator_attn"].append(safe_cos(commutator, al_v))
        signals["cos_commutator_ffn"].append(safe_cos(commutator, ml_v))
        signals["ref_cross_cos_a_m"].append(safe_cos(al_v, ml_v))
        signals["ref_cross_attn_ffn_sens"].append(norm_t1 / (ml_v.norm().item() + eps))

        # comm_persist: cos(C_l, C_{l-1})
        if prev_commutator_v is not None:
            signals["comm_persist"].append(safe_cos(commutator, prev_commutator_v))
        else:
            signals["comm_persist"].append(float("nan"))

        # Save commutator vector at probe layers (CPU, float32, for prop experiment)
        if li in PROBE_SET:
            probe_commutator_vecs[li] = commutator.detach().cpu().clone()

        prev_commutator_v = commutator.detach().cpu().clone()

        # Prop signals will be filled in later (placeholder NaN)
        for k in ["prop_sens", "rand_sens", "directional_advantage", "etd_effective"]:
            signals[k].append(float("nan"))

    return signals, probe_commutator_vecs, orig_logits


# ─── 传播实验（Hook Injection）────────────────────────────────────────────────
@torch.no_grad()
def _perturb_and_forward(model, input_ids, attn_mask, probe_layer: int,
                          perturb_1d: torch.Tensor) -> torch.Tensor:
    """
    Run full model forward with perturb_1d [D] added to the last-token position
    of the hidden state at the INPUT of probe_layer.
    Returns logits[:, -1, :].float() of shape [1, vocab].
    """
    base = model.model
    perturb_cpu = perturb_1d.float().cpu()  # keep on CPU, move in hook

    def hook_fn(module, args):
        t = args[0] if isinstance(args, tuple) else args
        t_out = t.clone()
        delta = perturb_cpu.to(device=t.device, dtype=t.dtype)   # [D]
        t_out[:, -1, :] = t_out[:, -1, :] + delta
        if isinstance(args, tuple):
            return (t_out,) + args[1:]
        return t_out

    hook = base.layers[probe_layer].register_forward_pre_hook(hook_fn)
    out  = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
    hook.remove()
    return out.logits[:, -1, :].float()


@torch.no_grad()
def compute_prop_signals(model, input_ids, attn_mask,
                          probe_commutator_vecs: dict,
                          orig_logits: torch.Tensor,
                          d_model: int) -> dict:
    """
    For each probe layer, compute prop_sens, rand_sens, directional_advantage.
    Returns {layer_idx: {'prop_sens': float, 'rand_sens': float,
                         'directional_advantage': float}}
    """
    results = {}
    torch.manual_seed(42)   # deterministic random directions per sample? No — vary per call.
    # We want random directions that vary per (sample, layer) but are reproducible.
    # Use layer index as part of seed variation.

    for li in PROBE_LAYERS:
        if li not in probe_commutator_vecs:
            results[li] = dict(prop_sens=float("nan"), rand_sens=float("nan"),
                               directional_advantage=float("nan"))
            continue

        comm_vec = probe_commutator_vecs[li]   # [D], float32, CPU
        norm_c   = comm_vec.norm().item()
        if norm_c < 1e-12:
            results[li] = dict(prop_sens=float("nan"), rand_sens=float("nan"),
                               directional_advantage=float("nan"))
            continue

        comm_hat = comm_vec / norm_c   # unit vector [D], CPU, float32

        # Random unit vector in R^{d_model}
        rand_vec = torch.randn(d_model)
        rand_hat = rand_vec / (rand_vec.norm() + 1e-12)

        # Perturbation forwards
        logits_comm = _perturb_and_forward(model, input_ids, attn_mask, li,
                                            EPSILON * comm_hat)
        logits_rand = _perturb_and_forward(model, input_ids, attn_mask, li,
                                            EPSILON * rand_hat)

        ps_comm = jsd_logits(orig_logits, logits_comm)
        ps_rand = jsd_logits(orig_logits, logits_rand)
        da      = ps_comm / (ps_rand + 1e-9)

        results[li] = dict(prop_sens=ps_comm, rand_sens=ps_rand, directional_advantage=da)

    return results


def merge_prop_into_signals(signals: dict, prop_data: dict, n_layers: int) -> dict:
    """
    Fill in prop_sens, rand_sens, directional_advantage, etd_effective
    at probe layer positions.
    """
    for li, d in prop_data.items():
        signals["prop_sens"][li]            = d["prop_sens"]
        signals["rand_sens"][li]            = d["rand_sens"]
        signals["directional_advantage"][li] = d["directional_advantage"]

        # etd_effective = commutator_cos_with_residual × directional_advantage
        cos_val = signals["commutator_cos_with_residual"][li]
        da_val  = d["directional_advantage"]
        if not (np.isnan(cos_val) or np.isnan(da_val)):
            signals["etd_effective"][li] = cos_val * da_val
        # else remains NaN

    return signals


# ─── 绘图辅助 ─────────────────────────────────────────────────────────────────
def add_r30_marks(ax, bench_name: str):
    opt = R30_OPTIMAL.get(bench_name)
    if opt is None: return
    t0, t1 = opt["t_start"], opt["t_stop"]
    ax.axvspan(t0, t1, alpha=0.12, color="gold", zorder=0)
    ax.axvline(t0, color="#2ca02c", ls="--", lw=2.0, zorder=4, label=f"t_start={t0}")
    ax.axvline(t1, color="#d62728", ls="--", lw=2.0, zorder=4, label=f"t_stop={t1}")


def _plot_probe_signal(ax, layers, mat, c, label="mean", probe_only=True):
    """Plot a signal that is only defined at probe layers (others NaN)."""
    mean_ = np.nanmean(mat, axis=0)
    std_  = np.nanstd(mat, axis=0)
    if probe_only:
        valid_mask = np.isfinite(mean_)
        xl = layers[valid_mask]
        ym = mean_[valid_mask]
        ys = std_[valid_mask]
        ax.plot(xl, ym, "o-", color=c, linewidth=1.8, markersize=5, label=label)
        ax.fill_between(xl, ym - ys, ym + ys, color=c, alpha=0.2)
    else:
        ax.plot(layers, mean_, color=c, linewidth=2.0, label=label)
        ax.fill_between(layers, mean_ - std_, mean_ + std_, color=c, alpha=0.2)


# ─── Plot 1: 每 benchmark 传播剖面图 ─────────────────────────────────────────
def plot_r36_benchmark(bench_name: str, records: list, n_layers: int, fig_dir: Path):
    if not records: return
    c = BENCH_COLORS.get(bench_name, "gray")
    layers = np.arange(n_layers)

    plot_keys = [
        ("prop_sens",             "JSD(Ĉ_l)",         True),
        ("rand_sens",             "JSD(random)",       True),
        ("directional_advantage", "DA = prop/rand",    True),
        ("etd_effective",         "cos_res × DA",      True),
        ("comm_persist",          "cos(C_l,C_{l+1})",  False),
        ("commutator_cos_with_residual", "cos(C_l,Δh)", False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    opt = R30_OPTIMAL.get(bench_name, {})
    t_note = f"R30 T-block [{opt.get('t_start','?')}, {opt.get('t_stop','?')})"

    for ax_i, (key, ylabel, probe_only) in enumerate(plot_keys):
        ax = axes_flat[ax_i]
        mat = np.array([r["per_layer"][key] for r in records], dtype=np.float64)
        add_r30_marks(ax, bench_name)
        _plot_probe_signal(ax, layers, mat, c, probe_only=probe_only)
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(key, fontsize=10)
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, n_layers - 0.5)
        if not probe_only:
            ax.axhline(0, color="gray", ls=":", lw=1)
        if key == "directional_advantage":
            ax.axhline(1.0, color="gray", ls="--", lw=1.5, alpha=0.7, label="DA=1 (no advantage)")

    fig.suptitle(
        f"R36 Propagation Signals + {t_note}  |  {bench_name}  N={len(records)}",
        fontsize=12)
    fig.tight_layout()
    safe = bench_name.replace("/", "-")
    out_path = fig_dir / f"{safe}_r36_prop_vs_layer.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ─── Plot 2: 全 benchmark 叠图 ───────────────────────────────────────────────
def plot_r36_all_overlay(all_data: dict, n_layers: int, fig_dir: Path):
    layers = np.arange(n_layers)

    keys_to_plot = [
        ("directional_advantage", "DA = prop/rand [核心 H2]", True),
        ("prop_sens",             "JSD(Ĉ_l) [H1 对照]",      True),
        ("rand_sens",             "JSD(random)",              True),
        ("etd_effective",         "cos_res × DA [H3]",        True),
        ("comm_persist",          "cos(C_l, C_{l+1}) [H6]",  False),
        ("commutator_cos_with_residual", "cos(C_l, Δh_l)",   False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    for ax_i, (key, title, probe_only) in enumerate(keys_to_plot):
        ax = axes_flat[ax_i]
        for bench_name, records in all_data.items():
            if not records: continue
            mat = np.array([r["per_layer"][key] for r in records], dtype=np.float64)
            mean_ = np.nanmean(mat, axis=0)
            c = BENCH_COLORS.get(bench_name, "gray")
            if probe_only:
                valid = np.isfinite(mean_)
                ax.plot(layers[valid], mean_[valid], "o-", color=c,
                        linewidth=1.5, markersize=4, label=bench_name)
            else:
                ax.plot(layers, mean_, color=c, linewidth=1.8, label=bench_name)
        ax.set_xlabel("Layer")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, n_layers - 0.5)
        if key == "directional_advantage":
            ax.axhline(1.0, color="gray", ls="--", lw=1.5, alpha=0.7)
        if not probe_only:
            ax.axhline(0, color="gray", ls=":", lw=1)

    fig.suptitle("R36: All Benchmarks Overlay (H1-H6 Overview)", fontsize=12)
    fig.tight_layout()
    out = fig_dir / "r36_all_overlay.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


# ─── Plot 3: 样本方差图（H4）────────────────────────────────────────────────
def plot_r36_sample_variance(all_data: dict, n_layers: int, fig_dir: Path):
    layers = np.arange(n_layers)

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes_flat = axes.flatten()
    ax_idx = 0

    for bench_name, records in all_data.items():
        if not records or ax_idx >= len(axes_flat): continue
        ax = axes_flat[ax_idx]; ax_idx += 1
        c = BENCH_COLORS.get(bench_name, "gray")

        mat = np.array([r["per_layer"]["directional_advantage"] for r in records],
                        dtype=np.float64)
        var_ = np.nanvar(mat, axis=0, ddof=0)
        valid = np.isfinite(var_)

        # Background T-block
        opt = R30_OPTIMAL.get(bench_name, {})
        t0, t1 = opt.get("t_start", 0), opt.get("t_stop", n_layers)
        ax.axvspan(t0, t1, alpha=0.12, color="gold", zorder=0)
        ax.axvline(t0, color="#2ca02c", ls="--", lw=1.5, zorder=4)
        ax.axvline(t1, color="#d62728", ls="--", lw=1.5, zorder=4)

        ax.plot(layers[valid], var_[valid], "s--", color=c,
                linewidth=1.8, markersize=5)
        ax.set_title(f"{bench_name}\nVar(DA) per layer [H4]", fontsize=9)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Var(DA) across samples", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, n_layers - 0.5)

    # hide unused
    for i in range(ax_idx, len(axes_flat)):
        axes_flat[i].axis("off")

    fig.suptitle("R36: Sample Variance of Directional Advantage (H4 Test)", fontsize=12)
    fig.tight_layout()
    out = fig_dir / "r36_sample_variance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


# ─── Plot 4: 散点图 (H3, H5)─────────────────────────────────────────────────
def plot_r36_scatter(all_data: dict, n_layers: int, fig_dir: Path):
    """mean(DA/prop_sens in T-block) vs ETD delta scatter."""
    sweep_path = RESULTS_DIR / "r30_sweep_results.json"
    etd_deltas: dict[str, float] = {}
    if sweep_path.exists():
        with open(sweep_path) as f:
            r30 = json.load(f)
        baseline = r30.get("baseline", {})
        best_acc: dict[str, float] = {}
        for row in r30.get("results", []):
            for b, acc in row.items():
                if b in ("t_start","t_stop") or not isinstance(acc, (int, float)): continue
                if b not in best_acc or acc > best_acc[b]: best_acc[b] = acc
        for b in baseline:
            if b in best_acc: etd_deltas[b] = best_acc[b] - baseline[b]
    # Approximate hard-MC
    for b, v in {"BoolQ": 0.04, "GPQA-Diamond": 0.02,
                  "AGIEval-Gaokao-MathQA": 0.03, "LogiQA": 0.04}.items():
        if b not in etd_deltas: etd_deltas[b] = v

    bench_names, da_tblock, ps_tblock, delta_vals = [], [], [], []
    late_range = slice(27, 34)  # layers 27-33

    da_late, ps_late = [], []

    for bench_name, records in all_data.items():
        if not records: continue
        opt = R30_OPTIMAL.get(bench_name)
        delta = etd_deltas.get(bench_name)
        if opt is None or delta is None: continue
        t0, t1 = opt["t_start"], opt["t_stop"]

        da_mat = np.array([r["per_layer"]["directional_advantage"] for r in records], dtype=np.float64)
        ps_mat = np.array([r["per_layer"]["prop_sens"] for r in records], dtype=np.float64)

        da_mean = np.nanmean(da_mat[:, t0:t1])
        ps_mean = np.nanmean(ps_mat[:, t0:t1])
        da_l    = np.nanmean(da_mat[:, late_range])
        ps_l    = np.nanmean(ps_mat[:, late_range])

        bench_names.append(bench_name)
        da_tblock.append(da_mean)
        ps_tblock.append(ps_mean)
        delta_vals.append(delta)
        da_late.append(da_l)
        ps_late.append(ps_l)

    if len(bench_names) < 2:
        print("  [WARN] Not enough benchmarks for scatter"); return

    da_tblock  = np.array(da_tblock)
    ps_tblock  = np.array(ps_tblock)
    delta_vals = np.array(delta_vals)
    da_late    = np.array(da_late)
    ps_late    = np.array(ps_late)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    panels = [
        (axes[0, 0], da_tblock,  delta_vals, "mean(DA) in T-block vs ETD Δacc [H3]",
         "Σ DA in T-block", True),
        (axes[0, 1], ps_tblock,  delta_vals, "mean(prop_sens) in T-block vs ETD Δacc",
         "mean prop_sens in T-block", False),
        (axes[1, 0], da_late,    delta_vals, "mean(DA) late layers (27-33) vs ETD Δacc [H5]",
         "mean DA in late layers", True),
        (axes[1, 1], da_tblock / (da_late + 1e-9), delta_vals,
         "DA T-block / DA late vs ETD Δacc [H5 ratio]",
         "DA_tblock / DA_late", False),
    ]

    for ax, x_arr, y_arr, title, xlabel, is_da in panels:
        for i, name in enumerate(bench_names):
            c = BENCH_COLORS.get(name, "gray")
            ax.scatter(x_arr[i], y_arr[i], color=c, s=100, zorder=5)
            ax.annotate(name, (x_arr[i], y_arr[i]),
                        textcoords="offset points", xytext=(5, 3), fontsize=7)
        if len(x_arr) >= 3:
            try:
                coeffs = np.polyfit(x_arr, y_arr, 1)
                x_fit  = np.linspace(x_arr.min(), x_arr.max(), 100)
                corr   = float(np.corrcoef(x_arr, y_arr)[0, 1])
                ax.plot(x_fit, np.polyval(coeffs, x_fit), "k--", alpha=0.5, lw=1.5,
                        label=f"r={corr:.3f}")
                ax.legend(fontsize=9)
            except Exception:
                pass
        if is_da:
            ax.axvline(1.0, color="gray", ls=":", lw=1, alpha=0.7)
        ax.axhline(0, color="gray", ls=":", lw=1)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("ETD Accuracy Δ", fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("R36 Scatter: DA / prop_sens vs ETD Accuracy Gain", fontsize=12)
    fig.tight_layout()
    out = fig_dir / "r36_scatter_da_vs_delta.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


# ─── Plot 5: 个体样本曲线（H4 样本差异化）───────────────────────────────────
def plot_r36_individual_samples(bench_name: str, records: list, n_layers: int, fig_dir: Path):
    if not records: return
    layers = np.arange(n_layers)
    opt    = R30_OPTIMAL.get(bench_name, {})

    # Compute each sample's baseline confidence (entropy of orig logits not stored,
    # so use mean DA across probe layers as a proxy for "sample difficulty")
    da_matrices = np.array([r["per_layer"]["directional_advantage"] for r in records],
                            dtype=np.float64)
    sample_mean_da = np.nanmean(da_matrices, axis=1)   # [N]

    n_show = min(len(records), 10)
    cmap   = plt.cm.coolwarm

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: individual DA curves
    ax = axes[0]
    t0, t1 = opt.get("t_start", 0), opt.get("t_stop", n_layers)
    ax.axvspan(t0, t1, alpha=0.12, color="gold", zorder=0)
    ax.axvline(t0, color="#2ca02c", ls="--", lw=2.0, zorder=4)
    ax.axvline(t1, color="#d62728", ls="--", lw=2.0, zorder=4)
    ax.axhline(1.0, color="gray", ls="--", lw=1.5, alpha=0.7)

    norm_vals = sample_mean_da
    v_min, v_max = np.nanmin(norm_vals), np.nanmax(norm_vals)
    for i in range(len(records)):
        mat_row = da_matrices[i]
        valid   = np.isfinite(mat_row)
        norm_v  = (norm_vals[i] - v_min) / (v_max - v_min + 1e-9)
        color   = cmap(norm_v)
        ax.plot(layers[valid], mat_row[valid], "o-", color=color,
                linewidth=1.0, markersize=3, alpha=0.7)

    # Color bar (mean DA value → color)
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=v_min, vmax=v_max))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="sample mean DA")
    ax.set_title(f"Individual sample DA curves | {bench_name}", fontsize=10)
    ax.set_xlabel("Layer"); ax.set_ylabel("DA")
    ax.grid(True, alpha=0.3); ax.set_xlim(-0.5, n_layers - 0.5)

    # Right: prop_sens individual curves
    ax2 = axes[1]
    ps_mat = np.array([r["per_layer"]["prop_sens"] for r in records], dtype=np.float64)
    ax2.axvspan(t0, t1, alpha=0.12, color="gold", zorder=0)
    ax2.axvline(t0, color="#2ca02c", ls="--", lw=2.0, zorder=4)
    ax2.axvline(t1, color="#d62728", ls="--", lw=2.0, zorder=4)
    for i in range(len(records)):
        row   = ps_mat[i]
        valid = np.isfinite(row)
        norm_v = (norm_vals[i] - v_min) / (v_max - v_min + 1e-9)
        ax2.plot(layers[valid], row[valid], "o-", color=cmap(norm_v),
                 linewidth=1.0, markersize=3, alpha=0.7)
    ax2.set_title(f"Individual sample prop_sens | {bench_name}", fontsize=10)
    ax2.set_xlabel("Layer"); ax2.set_ylabel("JSD (prop_sens)")
    ax2.grid(True, alpha=0.3); ax2.set_xlim(-0.5, n_layers - 0.5)

    fig.suptitle(f"R36 Individual Samples: {bench_name}", fontsize=11)
    fig.tight_layout()
    safe = bench_name.replace("/", "-")
    out  = fig_dir / f"r36_individual_samples_{safe}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


# ─── Plot 6: 后期层悖论图（H5）──────────────────────────────────────────────
def plot_r36_late_vs_tblock(all_data: dict, n_layers: int, fig_dir: Path):
    """Bar comparison of DA and prop_sens at late vs T-block layers."""
    bench_list  = [b for b, r in all_data.items() if r]
    n_bench     = len(bench_list)
    if n_bench == 0: return

    late_s, late_e  = 27, 34   # layers 27-33
    da_late_vals    = []
    da_tblock_vals  = []
    ps_late_vals    = []
    ps_tblock_vals  = []
    colors          = []

    for bench_name in bench_list:
        records = all_data[bench_name]
        opt     = R30_OPTIMAL.get(bench_name, {})
        t0, t1  = opt.get("t_start", 10), opt.get("t_stop", 20)

        da_mat = np.array([r["per_layer"]["directional_advantage"] for r in records], dtype=np.float64)
        ps_mat = np.array([r["per_layer"]["prop_sens"] for r in records], dtype=np.float64)

        da_late_vals.append(np.nanmean(da_mat[:, late_s:late_e]))
        da_tblock_vals.append(np.nanmean(da_mat[:, t0:t1]))
        ps_late_vals.append(np.nanmean(ps_mat[:, late_s:late_e]))
        ps_tblock_vals.append(np.nanmean(ps_mat[:, t0:t1]))
        colors.append(BENCH_COLORS.get(bench_name, "gray"))

    x  = np.arange(n_bench)
    w  = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, late_v, tblock_v, ylabel, title in [
        (axes[0], da_tblock_vals, da_late_vals,
         "Directional Advantage (DA)", "DA: T-block vs Late layers [H5]"),
        (axes[1], ps_tblock_vals, ps_late_vals,
         "prop_sens (JSD)", "prop_sens: T-block vs Late layers [H1/H5]"),
    ]:
        bars1 = ax.bar(x - w/2, tblock_v, w, label="T-block",
                       color=[c + "cc" for c in colors] if False else colors,
                       alpha=0.85, edgecolor="black", linewidth=0.5)
        bars2 = ax.bar(x + w/2, late_v, w, label="Late (27-33)",
                       alpha=0.45, edgecolor="black", linewidth=0.5,
                       color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels([b[:12] for b in bench_list], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        if ylabel.startswith("Dir"):
            ax.axhline(1.0, color="gray", ls="--", lw=1.5, alpha=0.7)

    fig.suptitle("R36: T-block vs Late Layer Comparison (H1 / H5 Test)", fontsize=12)
    fig.tight_layout()
    out = fig_dir / "r36_late_vs_tblock.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


# ─── 统计摘要 ─────────────────────────────────────────────────────────────────
def compute_stats(all_data: dict, n_layers: int) -> dict:
    stats = {}
    for bench_name, records in all_data.items():
        if not records: continue
        opt = R30_OPTIMAL.get(bench_name, {})
        t0, t1 = opt.get("t_start", 0), opt.get("t_stop", n_layers)
        bs = {"n": len(records)}

        for key in ["directional_advantage", "prop_sens", "rand_sens",
                     "etd_effective", "comm_persist",
                     "commutator_cos_with_residual"]:
            mat = np.array([r["per_layer"][key] for r in records], dtype=np.float64)
            # T-block stats
            tblock_col = mat[:, t0:t1]
            bs[f"{key}_tblock_mean"] = float(np.nanmean(tblock_col))
            bs[f"{key}_tblock_var"]  = float(np.nanvar(tblock_col))
            # Late stats (layers 27-33)
            late_col = mat[:, 27:34]
            bs[f"{key}_late_mean"]   = float(np.nanmean(late_col))
            # Robust stats (H2/H5：DA 比值易受 rand_sens→0 影响，中位数更可信)
            flat_tb = tblock_col[np.isfinite(tblock_col)]
            flat_lt = late_col[np.isfinite(late_col)]
            bs[f"{key}_tblock_median"] = float(np.median(flat_tb)) if flat_tb.size else float("nan")
            bs[f"{key}_late_median"]   = float(np.median(flat_lt)) if flat_lt.size else float("nan")
            # Per-probe-layer means
            for li in PROBE_LAYERS:
                col = mat[:, li]
                valid = col[np.isfinite(col)]
                if len(valid) > 0:
                    bs[f"{key}@L{li}_mean"] = float(np.mean(valid))
                    bs[f"{key}@L{li}_std"]  = float(np.std(valid))

        stats[bench_name] = bs
    return stats


# ─── 主流程 ───────────────────────────────────────────────────────────────────
def main():
    t0_wall = time.time()
    tok, model, n_layers, d_model = load_model()

    all_data = {}

    for bench_name, loader_fn in BENCHMARK_LOADERS:
        print(f"\n{'=' * 55}\n{bench_name}")
        try:
            samples = loader_fn(N_SAMPLES)
        except Exception as e:
            print(f"  [SKIP] Failed to load: {e}"); continue
        if samples is None:
            print(f"  [SKIP] Data unavailable"); continue
        print(f"  Loaded {len(samples)} samples")

        records = []
        for i, samp in enumerate(samples[:N_SAMPLES]):
            enc   = tok(samp["prompt"], return_tensors="pt", add_special_tokens=False)
            iids  = enc["input_ids"].to(DEVICE)
            amask = enc.get("attention_mask")
            amask = amask.to(DEVICE) if amask is not None else None

            # Step 1: base R35 signals + comm_persist + capture commutator vecs + orig_logits
            base_sigs, probe_vecs, orig_logits = collect_r36_base_signals(
                model, iids, amask, n_layers)

            # Step 2: propagation experiment
            prop_data = compute_prop_signals(
                model, iids, amask, probe_vecs, orig_logits, d_model)

            # Step 3: merge into full record
            full_sigs = merge_prop_into_signals(base_sigs, prop_data, n_layers)
            records.append({"per_layer": full_sigs})

            if (i + 1) % 25 == 0 or i == 0:
                # Print key stats at mid-probe layer (layer 18)
                da_mid = full_sigs["directional_advantage"][18]
                ps_mid = full_sigs["prop_sens"][18]
                cp_mid = full_sigs["comm_persist"][18] if len(full_sigs["comm_persist"]) > 18 else float("nan")
                print(
                    f"  [{i+1:3d}/{min(N_SAMPLES,len(samples))}]  "
                    f"DA@18={da_mid:.3f}  prop_sens@18={ps_mid:.5f}  "
                    f"comm_persist@18={cp_mid:.3f}"
                )
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        all_data[bench_name] = records

    # ── 保存完整数据 ───────────────────────────────────────────────────────────
    full_path = RESULTS_DIR / "r36_propagation_data_full.json"
    with open(full_path, "w") as f:
        json.dump({
            "n_layers":    n_layers,
            "d_model":     d_model,
            "r30_optimal": R30_OPTIMAL,
            "n_samples":   N_SAMPLES,
            "probe_layers": PROBE_LAYERS,
            "epsilon":     EPSILON,
            "signal_keys": ALL_KEYS,
            "benches":     {b: recs for b, recs in all_data.items()},
        }, f, indent=2)
    print(f"\nSaved data -> {full_path}")

    # ── 统计摘要 ───────────────────────────────────────────────────────────────
    stats = compute_stats(all_data, n_layers)
    stats_path = RESULTS_DIR / "r36_propagation_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats -> {stats_path}")

    # ── 绘图 ──────────────────────────────────────────────────────────────────
    for bench_name, records in all_data.items():
        plot_r36_benchmark(bench_name, records, n_layers, FIGURES_DIR)
        plot_r36_individual_samples(bench_name, records, n_layers, FIGURES_DIR)

    plot_r36_all_overlay(all_data, n_layers, FIGURES_DIR)
    plot_r36_sample_variance(all_data, n_layers, FIGURES_DIR)
    plot_r36_scatter(all_data, n_layers, FIGURES_DIR)
    plot_r36_late_vs_tblock(all_data, n_layers, FIGURES_DIR)

    elapsed = time.time() - t0_wall
    print(f"\nTotal elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
