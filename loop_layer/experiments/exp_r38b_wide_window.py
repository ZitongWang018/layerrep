"""
R38b: 宽窗口信号扩展实验
================================================================
基于 R38a 的分析发现，两个核心问题限制了信号方法的性能：
  1. min_start=9 排除了 BoolQ 最优 t_start=8
  2. n_t∈{4,6,8} 太窄，BoolQ 最优 n_t=14，CSQA 最优 n_t=12

本实验新增两个条件（在已有 R38a 结果基础上补充）：
  C7  persample_wide  — 逐样本 n_t∈{4,6,8,10,12,14}，min_start=6（扩展搜索空间）
  C8  calib_adaptive  — 标定 profile 决定 onset（t_start）+峰宽（n_t）
  C9  two_phase       — 标定 onset t_start + 逐样本在固定 t_start 上选最优 n_t

同时修复 LogiQA 加载（直接用 fireworks-ai/logiqa 避免 offline 异常）。

输出：
  results/r38_combined_results.json （R38a + R38b 条件合并）
  figures/r38_signal_full/（更新所有汇总图）
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
warnings.filterwarnings("ignore")

ROOT = Path("/root/autodl-tmp/loop_layer")
EXP  = ROOT / "experiments"
ETD  = ROOT / "ETD"
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
)
from etd_forward import etd_forward_logits, baseline_forward_logits

# ─── 配置（与 R38a 相同）──────────────────────────────────────────────────────
MODEL_PATH   = "/root/autodl-tmp/model_qwen"
RESULTS_DIR  = EXP / "results"
FIGURES_DIR  = EXP / "figures" / "r38_signal_full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE    = torch.bfloat16
N_CALIB  = 20
K_ETD    = 2

# R38b 扩展参数
MIN_START_WIDE = 6    # 从 9→6，捕获早期峰（BoolQ t_start=8）
MAX_START      = 22
NT_WIDE        = (4, 6, 8, 10, 12, 14)   # 扩展 n_t 候选，覆盖 BoolQ n_t=14

PROBE_LAYERS = list(range(6, 29, 2))  # [6,8,…,28]

SWEEP_BEST = {
    "BoolQ":                  (8,  22),
    "ARC-C":                  (14, 20),
    "TruthfulQA":             (16, 19),
    "CSQA":                   (10, 22),
    "MMLU-HS-Math":           (10, 18),
    "GPQA-Diamond":           (18, 20),
    "AGIEval-Gaokao-MathQA":  (13, 20),
    "LogiQA":                 (14, 19),
}

N_SAMPLES = {
    "BoolQ": 100, "ARC-C": 100, "TruthfulQA": 50, "CSQA": 100,
    "MMLU-HS-Math": 100, "GPQA-Diamond": 100,
    "AGIEval-Gaokao-MathQA": 100, "LogiQA": 100,
}

# R38b 新条件
NEW_COND_NAMES = ["persample_wide", "calib_adaptive", "two_phase"]
NEW_COND_COLORS = {
    "persample_wide": "#00BCD4",
    "calib_adaptive": "#E91E63",
    "two_phase":      "#FF5722",
}
NEW_COND_LABELS = {
    "persample_wide": "逐样本-宽搜索",
    "calib_adaptive": "标定自适应宽",
    "two_phase":      "两阶段选层",
}

# 全部条件（含 R38a）
ALL_COND_NAMES = [
    "baseline", "sweep_best",
    "persample_cos8", "persample_var",
    "onset_fixed8", "calib_onset8", "calib_global8",
    "persample_wide", "calib_adaptive", "two_phase",
]
ALL_COND_COLORS = {
    "baseline":        "#9E9E9E",
    "sweep_best":      "#2196F3",
    "persample_cos8":  "#4CAF50",
    "persample_var":   "#8BC34A",
    "onset_fixed8":    "#FF9800",
    "calib_onset8":    "#F44336",
    "calib_global8":   "#9C27B0",
    "persample_wide":  "#00BCD4",
    "calib_adaptive":  "#E91E63",
    "two_phase":       "#FF5722",
}
ALL_COND_LABELS = {
    "baseline":        "Baseline",
    "sweep_best":      "扫参最优",
    "persample_cos8":  "逐样本-8层",
    "persample_var":   "逐样本-变长",
    "onset_fixed8":    "固定Onset-8",
    "calib_onset8":    "标定Onset-8",
    "calib_global8":   "标定全局-8",
    "persample_wide":  "逐样本-宽搜索",
    "calib_adaptive":  "标定自适应宽",
    "two_phase":       "两阶段选层",
}

BENCH_COLORS = {
    "BoolQ": "#2196F3", "ARC-C": "#F44336", "CSQA": "#4CAF50",
    "TruthfulQA": "#FF9800", "MMLU-HS-Math": "#9C27B0",
    "GPQA-Diamond": "#00BCD4", "AGIEval-Gaokao-MathQA": "#795548",
    "LogiQA": "#607D8B",
}


# ─── 数据加载（含修复的 LogiQA）─────────────────────────────────────────────────
def load_boolq(n):
    ds = load_dataset("aps/super_glue", "boolq")["validation"]
    out = []
    for x in ds:
        if int(x["label"]) < 0: continue
        out.append({"prompt": f"{x['passage']}\nQuestion: {x['question']}?\nAnswer:",
                    "choices": ["no", "yes"], "label": int(x["label"])})
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
    lmap = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
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
    ds = load_dataset("cais/mmlu", "high_school_mathematics")["test"]
    out = []
    for x in ds:
        out.append({"prompt": f"Question: {x['question']}\nAnswer:",
                    "choices": x["choices"], "label": int(x["answer"])})
        if len(out) >= n: break
    return out

def load_logiqa_fixed(n):
    """直接使用 fireworks-ai/logiqa（已缓存），绕过 EleutherAI/logiqa 离线错误."""
    import re
    def _strip(opt):
        return re.sub(r"^[ABCDabcd]\.\s*", "", str(opt).strip())
    def _to_letter(r):
        lab = r.get("label") if r.get("label") is not None else r.get("answer")
        if isinstance(lab, (int, float)) and lab == int(lab):
            i = int(lab)
            if 0 <= i < 4: return "abcd"[i]
        s = str(lab).strip().lower()
        if s in "abcd": return s
        return None
    ds = load_dataset("fireworks-ai/logiqa", split="test")
    out = []
    for r in ds:
        label = _to_letter(r)
        if label is None: continue
        opts = r["options"]
        if hasattr(opts, "tolist"): opts = opts.tolist()
        choices = ["a", "b", "c", "d"]
        prompt = (f"Passage: {r['context']}\nQuestion: {r['question']}\nChoices:\n"
                  + "\n".join(f"{l.upper()}. {_strip(o)}" for l, o in zip(choices, opts))
                  + "\nAnswer:")
        out.append({"prompt": prompt, "choices": choices,
                    "label": choices.index(label)})
        if len(out) >= n: break
    return out

def _adapt_hard(items):
    return [{"prompt": it["prompt"], "choices": it["choices"],
             "label": it.get("valid_indices", [0])[0]} for it in items]

BENCH_LOADERS = {
    "BoolQ":                  load_boolq,
    "ARC-C":                  load_arc_c,
    "TruthfulQA":             load_truthfulqa,
    "CSQA":                   load_csqa,
    "MMLU-HS-Math":           load_mmlu_hs_math,
    "GPQA-Diamond":           lambda n: _adapt_hard(load_gpqa_diamond(n)),
    "AGIEval-Gaokao-MathQA":  lambda n: _adapt_hard(load_agieval_gaokao_mathqa(n)),
    "LogiQA":                 load_logiqa_fixed,
}


# ─── 工具函数 ──────────────────────────────────────────────────────────────────
def safe_cos(u: torch.Tensor, v: torch.Tensor) -> float:
    u = u.float().reshape(-1).cpu(); v = v.float().reshape(-1).cpu()
    nu, nv = u.norm(), v.norm()
    if nu < 1e-12 or nv < 1e-12: return 0.0
    return float((u @ v / (nu * nv)).clamp(-1, 1).item())

def loglikelihood_mc(logits, input_ids, prompt_len):
    total = 0.0
    for i in range(prompt_len, input_ids.shape[1]):
        logp = F.log_softmax(logits[0, i-1].float(), dim=-1)
        total += float(logp[input_ids[0, i]].item())
    return total


# ─── 模型加载 ──────────────────────────────────────────────────────────────────
def load_model():
    print("Loading Qwen3-8B …")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, device_map="auto",
        attn_implementation="eager", trust_remote_code=True,
    )
    mdl.eval()
    n_layers = mdl.config.num_hidden_layers
    print(f"  {n_layers} layers, device={DEVICE}")
    return tok, mdl, n_layers


# ─── 探针前向 ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def probe_forward_collect_cos_res(model, input_ids, attn_mask, n_layers):
    base = model.model
    h_inputs, a_outputs, m_outputs = {}, {}, {}
    hooks = []
    for li in range(n_layers):
        layer = base.layers[li]
        def make_pre(idx):
            def fn(_m, args):
                t = args[0] if isinstance(args, tuple) else args
                h_inputs[idx] = t[:, -1:, :].detach().clone()
            return fn
        def make_attn_post(idx):
            def fn(_m, _i, out):
                t = out[0] if isinstance(out, tuple) else out
                a_outputs[idx] = t[:, -1:, :].detach().clone()
            return fn
        def make_mlp_post(idx):
            def fn(_m, _i, out):
                m_outputs[idx] = out[:, -1:, :].detach().clone()
            return fn
        hooks += [layer.register_forward_pre_hook(make_pre(li)),
                  layer.self_attn.register_forward_hook(make_attn_post(li)),
                  layer.mlp.register_forward_hook(make_mlp_post(li))]
    model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
    for h in hooks: h.remove()

    cos_res: dict[int, float] = {}
    for li in PROBE_LAYERS:
        hi, al, ml = h_inputs.get(li), a_outputs.get(li), m_outputs.get(li)
        if hi is None or al is None or ml is None: continue
        try:
            layer = base.layers[li]
            m_l0 = layer.mlp(layer.post_attention_layernorm(hi))
            term1 = (ml - m_l0).squeeze()
            delta_h = (al + ml).squeeze()
            cos_res[li] = safe_cos(term1, delta_h)
        except Exception:
            pass
    return cos_res


# ─── 标定阶段 ─────────────────────────────────────────────────────────────────
def calibrate_benchmark_profile(items_calib, model, tok, n_layers):
    profile_accum = defaultdict(list)
    for item in items_calib[:N_CALIB]:
        enc = tok(item["prompt"], return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(DEVICE)
        amask = enc.get("attention_mask")
        if amask is not None: amask = amask.to(DEVICE)
        cos_res = probe_forward_collect_cos_res(model, ids, amask, n_layers)
        for li, v in cos_res.items(): profile_accum[li].append(v)
    return {li: float(np.mean(vs)) for li, vs in profile_accum.items() if vs}


def derive_adaptive_window(
    profile: dict[int, float],
    ratio_onset: float = 0.65,
    ratio_drop: float = 0.40,
    n_t_min: int = 4,
    n_t_max: int = 14,
    min_start: int = MIN_START_WIDE,
    max_start: int = MAX_START,
) -> tuple[int, int]:
    """
    自适应宽度窗口：
    1. threshold_onset = max(profile) × ratio_onset → 找 onset t_start
    2. threshold_drop  = max(profile) × ratio_drop  → 找 peak 宽度（onset 到首次降落处）
    3. n_t = max(n_t_min, min(n_t_max, width))
    """
    valid = {l: v for l, v in profile.items() if min_start <= l <= max_start}
    if not valid:
        return min_start, min_start + 8
    max_val = max(valid.values())
    thr_onset = max_val * ratio_onset
    thr_drop  = max_val * ratio_drop

    sorted_layers = sorted(valid)
    onset = None
    for l in sorted_layers:
        if valid[l] >= thr_onset:
            onset = l
            break
    if onset is None:
        onset = max(valid, key=valid.__getitem__)

    # 找 drop：onset 之后首次低于 thr_drop 的层
    drop = None
    for l in sorted_layers:
        if l > onset and valid.get(l, thr_drop) < thr_drop:
            drop = l
            break
    if drop is None:
        drop = max_start

    n_t = max(n_t_min, min(n_t_max, drop - onset))
    return onset, onset + n_t


# ─── 窗口选择（扩展版）──────────────────────────────────────────────────────────
def select_window_wide(
    cos_res: dict[int, float],
    nt_candidates: tuple[int, ...] = NT_WIDE,
    min_start: int = MIN_START_WIDE,
    max_start: int = MAX_START,
) -> tuple[int, int]:
    """扩展搜索：n_t∈{4,6,8,10,12,14}，min_start=6."""
    best_start, best_nt, best_score = min_start, nt_candidates[0], -999.0
    for n_t in nt_candidates:
        for start in range(min_start, max_start + 1):
            vals = [cos_res[l] for l in cos_res if start <= l < start + n_t]
            if len(vals) < 2: continue
            score = float(np.mean(vals))
            if score > best_score:
                best_score = score
                best_start = start
                best_nt = n_t
    return best_start, best_start + best_nt


def select_window_two_phase(
    cos_res: dict[int, float],
    calib_onset_tstart: int,
    nt_candidates: tuple[int, ...] = (4, 6, 8, 10, 12, 14),
    max_start: int = MAX_START,
) -> tuple[int, int]:
    """
    两阶段：t_start 由标定阶段 onset 固定，在该 t_start 基础上
    选使 mean(cos_res in [t_start, t_start+n_t]) 最高的 n_t。
    """
    best_nt, best_score = nt_candidates[0], -999.0
    for n_t in nt_candidates:
        stop = calib_onset_tstart + n_t
        vals = [cos_res[l] for l in cos_res if calib_onset_tstart <= l < stop]
        if len(vals) < 1: continue
        score = float(np.mean(vals))
        if score > best_score:
            best_score = score
            best_nt = n_t
    return calib_onset_tstart, calib_onset_tstart + best_nt


# ─── MC 评测 ─────────────────────────────────────────────────────────────────
def mc_predict(model, tok, item, n_e=None, n_t=None, k=K_ETD):
    prompt, choices = item["prompt"], item["choices"]
    scores = []
    for cont in choices:
        full = prompt + " " + cont
        enc = tok(full, return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(DEVICE)
        amask = enc.get("attention_mask")
        if amask is not None: amask = amask.to(DEVICE)
        plen = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
        if n_e is not None and n_t is not None and n_t > 0:
            alpha = min(1.0, 6.0 / max(n_t, 1))
            lgts = etd_forward_logits(model, ids, amask, n_e=n_e, n_t=n_t, k=k, alpha=alpha)
        else:
            lgts = baseline_forward_logits(model, ids, amask)
        scores.append(loglikelihood_mc(lgts, ids, plen))
    return int(np.argmax(scores))


# ─── 核心评测（仅新条件）─────────────────────────────────────────────────────────
def evaluate_benchmark_new_conds(bench, items, model, tok, n_layers):
    """仅跑 R38b 新条件（persample_wide, calib_adaptive, two_phase），结合标定阶段."""
    n_total = len(items)

    # 标定阶段
    print(f"  [标定] 聚合前 {N_CALIB} 样本的 cos_res profile …")
    t0c = time.time()
    mean_profile = calibrate_benchmark_profile(items, model, tok, n_layers)
    calib_adapt_win = derive_adaptive_window(mean_profile)
    calib_onset_tstart = calib_adapt_win[0]
    print(f"  [标定] 完成 ({time.time()-t0c:.1f}s)  calib_adaptive={calib_adapt_win}")

    correct = {c: 0 for c in NEW_COND_NAMES}
    selected_tstarts = {c: [] for c in NEW_COND_NAMES}

    t0 = time.time()
    for i, item in enumerate(items):
        label = item["label"]

        # 探针前向（逐样本）
        enc = tok(item["prompt"], return_tensors="pt", add_special_tokens=False)
        probe_ids = enc["input_ids"].to(DEVICE)
        probe_mask = enc.get("attention_mask")
        if probe_mask is not None: probe_mask = probe_mask.to(DEVICE)
        cos_res = probe_forward_collect_cos_res(model, probe_ids, probe_mask, n_layers)

        # 窗口选择
        wide_win    = select_window_wide(cos_res)
        tp_win      = select_window_two_phase(cos_res, calib_onset_tstart)
        adapt_win   = calib_adapt_win  # 标定固定窗口

        cond_windows = {
            "persample_wide": wide_win,
            "calib_adaptive": adapt_win,
            "two_phase":      tp_win,
        }
        for cn in NEW_COND_NAMES:
            selected_tstarts[cn].append(cond_windows[cn][0])
            t_start, t_stop = cond_windows[cn]
            n_layers_model = model.config.num_hidden_layers
            n_e_c = t_start
            n_t_c = t_stop - t_start
            n_d_c = n_layers_model - t_stop
            if n_d_c < 1 or n_t_c < 1:
                pred = mc_predict(model, tok, item)
            else:
                pred = mc_predict(model, tok, item, n_e=n_e_c, n_t=n_t_c, k=K_ETD)
            if pred == label: correct[cn] += 1

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_total - i - 1)
            line = f"  [{i+1:3d}/{n_total}] "
            for cn in NEW_COND_NAMES:
                line += f"{cn.split('_')[1][:4]}={correct[cn]/(i+1):.3f} "
            line += f"| {elapsed:.0f}s ETA {eta:.0f}s"
            print(line)

        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    accuracies = {c: correct[c] / n_total for c in NEW_COND_NAMES}
    win_stats = {
        c: {
            "t_start_mean": float(np.mean(selected_tstarts[c])) if selected_tstarts[c] else 0.0,
            "t_start_std":  float(np.std(selected_tstarts[c]))  if selected_tstarts[c] else 0.0,
            "t_start_list": selected_tstarts[c],
            "calib_adaptive_window": list(calib_adapt_win),
        }
        for c in NEW_COND_NAMES
    }
    return accuracies, win_stats, elapsed


# ─── 最终可视化（合并 R38a + R38b）─────────────────────────────────────────────
SIGNAL_CONDS_ALL = [
    "persample_cos8", "persample_var",
    "onset_fixed8", "calib_onset8", "calib_global8",
    "persample_wide", "calib_adaptive", "two_phase",
]

def plot_benchmark_bar_combined(bench, result):
    accs = result["accuracies"]
    base_acc  = accs["baseline"]
    sweep_acc = accs["sweep_best"]

    conds = ["baseline", "sweep_best"] + SIGNAL_CONDS_ALL
    # 只显示 accs 里存在的条件
    conds = [c for c in conds if c in accs]

    x = np.arange(len(conds))
    bars   = [accs[c] for c in conds]
    colors = [ALL_COND_COLORS[c] for c in conds]

    fig, ax = plt.subplots(figsize=(14, 4))
    brs = ax.bar(x, bars, color=colors, edgecolor="black", linewidth=0.5, alpha=0.88)
    for bar, v in zip(brs, bars):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(base_acc,  color="grey",    linestyle="--", linewidth=1.2, alpha=0.7,
               label=f"Baseline={base_acc:.3f}")
    ax.axhline(sweep_acc, color="#2196F3", linestyle=":",  linewidth=1.5, alpha=0.85,
               label=f"扫参最优={sweep_acc:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels([ALL_COND_LABELS.get(c, c) for c in conds], fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("Accuracy", fontsize=10)

    # 标注各窗口信息
    sw = result.get("sweep_best_window", "?")
    cg = result.get("calib_global8_window", "?")
    ca = result.get("calib_adaptive_window_r38b", "?")
    ax.set_title(
        f"R38 Combined: {bench}  (N={result['n']})\n"
        f"扫参={sw}  标定全局={cg}  自适应={ca}",
        fontsize=9,
    )
    ax.set_ylim(0, max(bars) * 1.22)
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    fname = FIGURES_DIR / f"combined_bar_{bench.replace('/', '_')}.png"
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {fname}")


def plot_heatmap_combined(all_results):
    signal_conds = [c for c in SIGNAL_CONDS_ALL
                    if any(c in res["accuracies"] for res in all_results.values())]
    bench_list = list(all_results.keys())

    delta_matrix = np.full((len(bench_list), len(signal_conds)), np.nan)
    acc_matrix   = np.full_like(delta_matrix, np.nan)
    for bi, bench in enumerate(bench_list):
        accs = all_results[bench]["accuracies"]
        base = accs["baseline"]
        for ci, cn in enumerate(signal_conds):
            if cn in accs:
                delta_matrix[bi, ci] = accs[cn] - base
                acc_matrix[bi, ci]   = accs[cn]

    vmax = max(np.nanmax(np.abs(delta_matrix)), 0.05)
    fig, ax = plt.subplots(figsize=(14, max(5, len(bench_list) * 0.9)))
    im = ax.imshow(delta_matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="Δacc vs Baseline")

    ax.set_xticks(range(len(signal_conds)))
    ax.set_xticklabels([ALL_COND_LABELS.get(c, c) for c in signal_conds],
                       fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(len(bench_list)))
    ax.set_yticklabels(bench_list, fontsize=9)

    for bi in range(len(bench_list)):
        for ci in range(len(signal_conds)):
            if np.isnan(delta_matrix[bi, ci]): continue
            delta = delta_matrix[bi, ci]
            acc   = acc_matrix[bi, ci]
            color = "white" if abs(delta) > vmax * 0.5 else "black"
            ax.text(ci, bi, f"{acc:.3f}\n({delta:+.3f})",
                    ha="center", va="center", fontsize=7, color=color)

    ax.set_title("R38 Combined 热力图：全条件 Accuracy 及 Δacc vs Baseline", fontsize=11)
    plt.tight_layout()
    fname = FIGURES_DIR / "combined_heatmap.png"
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {fname}")


def plot_summary_final_combined(all_results):
    """最终汇总：baseline / 最佳信号(R38a+b combined) / sweep_best."""
    bench_list = list(all_results.keys())
    n_bench = len(bench_list)
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(n_bench)
    w = 0.26

    baselines  = [all_results[b]["accuracies"]["baseline"]  for b in bench_list]
    sweep_accs = [all_results[b]["accuracies"]["sweep_best"] for b in bench_list]
    best_signal_accs = []
    best_signal_names = []
    for bench in bench_list:
        accs = all_results[bench]["accuracies"]
        available = [c for c in SIGNAL_CONDS_ALL if c in accs]
        if available:
            best_c = max(available, key=lambda c: accs[c])
            best_signal_accs.append(accs[best_c])
            best_signal_names.append(ALL_COND_LABELS.get(best_c, best_c))
        else:
            best_signal_accs.append(accs["baseline"])
            best_signal_names.append("N/A")

    b1 = ax.bar(x - w, baselines,         w, label="Baseline",   color="#9E9E9E", alpha=0.85)
    b2 = ax.bar(x,     best_signal_accs,  w, label="最佳信号方法", color="#F44336", alpha=0.85)
    b3 = ax.bar(x + w, sweep_accs,        w, label="扫参最优",   color="#2196F3", alpha=0.85)

    for bar, v in zip(b1, baselines):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)
    for bar, v, nm in zip(b2, best_signal_accs, best_signal_names):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{v:.3f}\n({nm[:5]})", ha="center", va="bottom", fontsize=6)
    for bar, v in zip(b3, sweep_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels([b[:14] for b in bench_list], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_title("R38 最终汇总（含 R38b）：Baseline vs 最佳信号 vs 扫参最优", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, min(1.0, max(max(sweep_accs), max(best_signal_accs)) * 1.25 + 0.05))
    plt.tight_layout()
    fname = FIGURES_DIR / "combined_summary_final.png"
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {fname}")


def print_combined_hypothesis(all_results):
    print("\n" + "=" * 70)
    print("R38 Combined 假设验证摘要（R38a + R38b）")
    print("=" * 70)
    signal_conds = [c for c in SIGNAL_CONDS_ALL
                    if any(c in res["accuracies"] for res in all_results.values())]

    for bench, res in all_results.items():
        accs = res["accuracies"]
        base  = accs["baseline"]
        sweep = accs["sweep_best"]
        avail = [c for c in signal_conds if c in accs]
        best_sig = max(avail, key=lambda c: accs[c]) if avail else "N/A"
        print(f"\n{bench}  (N={res['n']}):")
        print(f"  Baseline={base:.4f}  扫参最优={sweep:.4f}  Δsweep={sweep-base:+.4f}")
        for cn in avail:
            delta = accs[cn] - base
            frac  = accs[cn] / sweep if sweep > 0 else 0.0
            mk = "★" if cn == best_sig else " "
            print(f"  {mk} {cn:18s}: {accs[cn]:.4f}  Δ={delta:+.4f}  {frac:.1%} of sweep")

    n_bench = len(all_results)
    print(f"\n{'─'*70}")
    print("方法 | 赢 benchmark 数 | 优于 baseline 数 | 宏平均 Δacc")
    for cn in signal_conds:
        wins  = sum(1 for res in all_results.values()
                    if cn in res["accuracies"]
                    and max((res["accuracies"].get(c, -99) for c in signal_conds if c in res["accuracies"]),
                            default=-99) == res["accuracies"].get(cn, -99))
        beats = sum(1 for res in all_results.values()
                    if cn in res["accuracies"]
                    and res["accuracies"][cn] > res["accuracies"]["baseline"])
        available = [res for res in all_results.values() if cn in res["accuracies"]]
        macro = (np.mean([res["accuracies"][cn] - res["accuracies"]["baseline"]
                          for res in available]) if available else float("nan"))
        print(f"  {ALL_COND_LABELS.get(cn, cn):14s} | {wins}/{n_bench}         | {beats}/{len(available)}           | {macro:+.4f}")

    sweep_macro = np.mean([res["accuracies"]["sweep_best"] - res["accuracies"]["baseline"]
                           for res in all_results.values()])
    print(f"  {'扫参最优':14s} |              |              | {sweep_macro:+.4f}")


# ─── 主流程 ───────────────────────────────────────────────────────────────────
def main():
    t_total = time.time()
    print("=" * 70)
    print("R38b: Wide-Window Signal Extension")
    print(f"MIN_START_WIDE={MIN_START_WIDE}  NT_WIDE={NT_WIDE}")
    print(f"New conditions: {NEW_COND_NAMES}")
    print("=" * 70)

    # 加载 R38a 结果
    r38a_path = RESULTS_DIR / "r38_signal_full_bench_results.json"
    combined_path = RESULTS_DIR / "r38_combined_results.json"

    if combined_path.exists():
        with open(combined_path) as f:
            all_results = json.load(f)
        print(f"[恢复] 读取已有 combined 结果 ({len(all_results)} benchmark)")
    elif r38a_path.exists():
        with open(r38a_path) as f:
            all_results = json.load(f)
        print(f"[导入] 从 R38a 结果读取 ({len(all_results)} benchmark)")
    else:
        all_results = {}
        print("[初始化] 无已有结果")

    tok, model, n_layers = load_model()

    for bench, loader in BENCH_LOADERS.items():
        n = N_SAMPLES[bench]
        # 检查是否已有 R38b 新条件
        existing = all_results.get(bench, {}).get("accuracies", {})
        if all(c in existing for c in NEW_COND_NAMES):
            print(f"  [跳过] {bench}（R38b 新条件已存在）")
            continue

        print(f"\n{'─'*60}")
        print(f"Benchmark: {bench}  N={n}  [R38b 新条件]")
        print(f"{'─'*60}")

        try:
            items = loader(n)
        except Exception as e:
            print(f"  [ERROR] 加载 {bench} 失败: {e}")
            continue
        if not items:
            print(f"  [SKIP] {bench} 返回空数据")
            continue
        items = items[:n]
        print(f"  加载 {len(items)} 样本")

        new_accs, win_stats, elapsed = evaluate_benchmark_new_conds(
            bench, items, model, tok, n_layers
        )

        # 合并到已有结果
        if bench not in all_results:
            # 需要先跑 R38a baseline+sweep_best 作为参考
            sweep_win = SWEEP_BEST[bench]
            all_results[bench] = {
                "benchmark": bench, "n": n,
                "elapsed_s": 0.0,
                "accuracies": {"baseline": 0.0, "sweep_best": 0.0},
                "sweep_best_window": list(sweep_win),
            }
            print(f"  [NOTE] {bench} 无 R38a 基础结果，baseline/sweep_best 需单独运行")

        all_results[bench]["accuracies"].update(new_accs)
        all_results[bench].setdefault("window_stats", {}).update(win_stats)
        # 记录 calib_adaptive 窗口
        calib_adapt_win = win_stats["calib_adaptive"]["calib_adaptive_window"]
        all_results[bench]["calib_adaptive_window_r38b"] = calib_adapt_win
        all_results[bench]["elapsed_s_r38b"] = elapsed

        # 打印结果
        accs_all = all_results[bench]["accuracies"]
        base  = accs_all.get("baseline", 0.0)
        sweep = accs_all.get("sweep_best", 0.0)
        print(f"\n  === {bench} R38b 新条件 Results ===")
        for cn in NEW_COND_NAMES:
            delta = new_accs[cn] - base
            frac  = new_accs[cn] / sweep if sweep > 0 else 0.0
            print(f"    {cn:20s}: {new_accs[cn]:.4f}  Δbaseline={delta:+.4f}  {frac:.1%} of sweep")

        # 条形图
        plot_benchmark_bar_combined(bench, all_results[bench])

        # 保存中间结果
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  保存 → {combined_path}")

    # 汇总可视化
    print(f"\n{'='*60}")
    print("生成合并汇总可视化 …")
    if all_results:
        plot_heatmap_combined(all_results)
        plot_summary_final_combined(all_results)

    print_combined_hypothesis(all_results)

    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {combined_path}")

    elapsed = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"R38b 完成！总耗时 {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
