#!/usr/bin/env python3
"""
R39 信号猎杀实验：系统筛选最优 ETD 选层信号（跨三架构）。

阶段 A（信号筛选）：轻量 probe 阶段 - 并行计算 5 路候选信号的 profile，量化各信号对
                   sweep_best 窗口的区分力（disc_score, coverage, start_error）。

阶段 B（全量评测）：用排名靠前的信号 + 经验标定方法进行 100 样本全量评测，
                   生成综合对比可视化，目标超越 baseline 和 sweep_best。

候选信号（全部在单次 probe forward 内计算，无需额外前向）：
  neg_cos_am      -cos(a_l, m_l)          竞争区（越负越需要 ETD 协调）
  delta_h_ratio   ||a_l+m_l||/||h_in_l||  层更新幅度归一化比
  update_persist  cos(Δh_l, Δh_{l-1})    相邻层更新方向一致性
  attn_dom        ||a_l|| / (||a_l||+||m_l||)  Attention 主导比
  cos_res         cos(Term1, Δh_l)         现有信号（需额外 MLP forward）

经验标定信号（需 N_CALIB × N_cand 次 ETD 前向）：
  empirical_calib  枚举候选窗口，取 logit-gain 最大者

用法：
  python experiments/exp_r39_signal_hunt.py --preset qwen3-8b
  python experiments/exp_r39_signal_hunt.py --preset llama3-8b
  python experiments/exp_r39_signal_hunt.py --preset gemma2-2b
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
warnings.filterwarnings("ignore")

ROOT = Path("/root/autodl-tmp/loop_layer")
EXP = ROOT / "experiments"
ETD_DIR = ROOT / "ETD"
for p in (str(ROOT), str(EXP), str(ETD_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from datasets import DownloadConfig, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from etd_forward import baseline_forward_logits, etd_forward_logits
from hard_mc_benchmark_loaders import load_agieval_gaokao_mathqa, load_gpqa_diamond

# ──────────────────────────────────────────────────────────────────────────────
# 全局常量
# ──────────────────────────────────────────────────────────────────────────────
N_CALIB = 20
K_ETD   = 2
# 信号名称 → (方向, 说明)  "max" = 越高越好  "neg" = 越低越好（程序内部取反统一为越高越好）
SIGNAL_META: dict[str, tuple[str, str]] = {
    "neg_cos_am":     ("max", "-cos(a_l,m_l)：竞争区强度"),
    "delta_h_ratio":  ("max", "||Δh||/||h||：层更新幅度"),
    "update_persist": ("max", "cos(Δh_l,Δh_{l-1})：更新方向一致性"),
    "attn_dom":       ("max", "||a||/(||a||+||m||)：Attention 主导比"),
    "cos_res":        ("max", "cos(Term1,Δh)：现有 cos_res 信号"),
}
ALL_SIGNAL_NAMES = list(SIGNAL_META.keys())

N_SAMPLES = {
    "BoolQ": 100, "ARC-C": 100, "TruthfulQA": 50, "CSQA": 100,
    "MMLU-HS-Math": 100, "GPQA-Diamond": 100, "AGIEval-Gaokao-MathQA": 100, "LogiQA": 100,
}
BENCH_ORDER = ["BoolQ","ARC-C","TruthfulQA","CSQA","MMLU-HS-Math","GPQA-Diamond","AGIEval-Gaokao-MathQA","LogiQA"]

# Qwen3-8B 的 sweep_best 来自 R38（R30 扫参结果）
QWEN_SWEEP_BEST: dict[str, tuple[int, int]] = {
    "BoolQ":                (8,  22),
    "ARC-C":                (14, 20),
    "TruthfulQA":           (16, 19),
    "CSQA":                 (10, 22),
    "MMLU-HS-Math":         (10, 18),
    "GPQA-Diamond":         (18, 20),
    "AGIEval-Gaokao-MathQA":(13, 20),
    "LogiQA":               (14, 19),
}

PRESETS: dict[str, dict] = {
    "qwen3-8b": {
        "model_path":  "/root/autodl-tmp/model_qwen",
        "arch":        "qwen3",
        "n_layers":    36,
        "probe_layers": list(range(6, 33, 2)),   # L6..L32
        "min_start":   9,
        "max_start":   26,
        "nt_arch":     (4, 6, 8, 14),
        "sweep_best":  QWEN_SWEEP_BEST,
        "out_json":    EXP / "results/r39_signal_hunt_qwen3.json",
        "out_fig":     EXP / "figures/r39_signal_hunt_qwen3",
    },
    "llama3-8b": {
        "model_path":  "/root/autodl-tmp/Llama3-8B",
        "arch":        "llama",
        "n_layers":    32,
        "probe_layers": list(range(6, 27, 2)),   # L6..L26
        "min_start":   8,
        "max_start":   20,
        "nt_arch":     (2, 4, 6, 8),
        "sweep_main":  EXP / "llama3-8b/results/etd_layer_sweep_r30style.json",
        "sweep_hard":  EXP / "llama3-8b/results/hard_mc/etd_layer_sweep_r30style.json",
        "out_json":    EXP / "results/r39_signal_hunt_llama3.json",
        "out_fig":     EXP / "figures/r39_signal_hunt_llama3",
    },
    "gemma2-2b": {
        "model_path":  "/root/autodl-tmp/Gemma2-2B",
        "arch":        "gemma2",
        "n_layers":    26,
        "probe_layers": list(range(4, 23, 2)),   # L4..L22
        "min_start":   5,
        "max_start":   16,
        "nt_arch":     (2, 4, 14, 18),
        "sweep_main":  EXP / "gemma2-2b/results/etd_layer_sweep_r30style.json",
        "sweep_hard":  EXP / "gemma2-2b/results/hard_mc/etd_layer_sweep_r30style.json",
        "out_json":    EXP / "results/r39_signal_hunt_gemma2.json",
        "out_fig":     EXP / "figures/r39_signal_hunt_gemma2",
    },
}

# 评测条件名称（最终对比图只含这些）
EVAL_COND_NAMES = [
    "baseline",
    "sweep_best",
    "neg_cos_am_var",
    "top_signal_var",
    "top_signal_calib",
    "empirical_calib",
]


# ──────────────────────────────────────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────────────────────────────────────
def _fmt(prefix: str, conts: list[str], label: int) -> dict:
    return {"prompt": prefix, "choices": [c.strip() for c in conts], "label": label}

def load_boolq(n):
    ds = load_dataset("aps/super_glue", "boolq")["validation"]
    out = []
    for r in ds:
        if len(out) >= n: break
        lab = int(r["label"])
        if lab < 0: continue
        out.append(_fmt(f"{r['passage']}\nQuestion: {r['question']}?\nAnswer:", ["no","yes"], lab))
    return out

def load_arc_c(n):
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")["test"]
    out = []
    for r in ds:
        if len(out) >= n: break
        key = r["answerKey"]
        label = ord(key)-ord("A") if key in "ABCD" else int(key)-1
        out.append(_fmt(f"Question: {r['question'].strip()}\nAnswer:", r["choices"]["text"], label))
    return out

def load_csqa(n):
    ds = load_dataset("tau/commonsense_qa")["validation"]
    lmap = {"A":0,"B":1,"C":2,"D":3,"E":4}
    out = []
    for r in ds:
        if len(out) >= n: break
        key = r["answerKey"]
        if key not in lmap: continue
        out.append(_fmt(f"Question: {r['question'].strip()}\nAnswer:", r["choices"]["text"], lmap[key]))
    return out

def load_truthfulqa(n):
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice")["validation"]
    out = []
    for r in ds:
        if len(out) >= n: break
        out.append(_fmt(f"Question: {r['question']}\nAnswer:",
                        r["mc1_targets"]["choices"], int(np.argmax(r["mc1_targets"]["labels"]))))
    return out

def load_mmlu_hs_math(n):
    dc = DownloadConfig(local_files_only=True)
    ds = load_dataset("cais/mmlu", "high_school_mathematics", download_config=dc)["test"]
    out = []
    for r in ds:
        if len(out) >= n: break
        out.append(_fmt(f"Question: {r['question'].strip()}\nAnswer:", [str(c) for c in r["choices"]], int(r["answer"])))
    return out

def load_logiqa_fixed(n):
    def _strip(o): return re.sub(r"^[ABCDabcd]\.\s*","",str(o).strip())
    def _to_letter(r):
        lab = r.get("label") if r.get("label") is not None else r.get("answer")
        if isinstance(lab,(int,float)) and lab==int(lab):
            i=int(lab)
            if 0<=i<4: return "abcd"[i]
        s=str(lab).strip().lower()
        return s if s in "abcd" else None
    ds = load_dataset("fireworks-ai/logiqa", split="test")
    out = []
    for r in ds:
        label = _to_letter(r)
        if label is None: continue
        opts = r["options"]
        if hasattr(opts,"tolist"): opts = opts.tolist()
        choices = ["a","b","c","d"]
        prompt = (f"Passage: {r['context']}\nQuestion: {r['question']}\nChoices:\n"
                  + "\n".join(f"{x.upper()}. {_strip(o)}" for x,o in zip(choices,opts))
                  + "\nAnswer:")
        out.append({"prompt":prompt,"choices":choices,"label":choices.index(label)})
        if len(out)>=n: break
    return out

def _adapt_hard(items):
    return [{"prompt":it["prompt"],"choices":it["choices"],"label":it.get("valid_indices",[0])[0]} for it in items]

BENCH_LOADERS = {
    "BoolQ":                 load_boolq,
    "ARC-C":                 load_arc_c,
    "TruthfulQA":            load_truthfulqa,
    "CSQA":                  load_csqa,
    "MMLU-HS-Math":          load_mmlu_hs_math,
    "GPQA-Diamond":          lambda n: _adapt_hard(load_gpqa_diamond(n)),
    "AGIEval-Gaokao-MathQA": lambda n: _adapt_hard(load_agieval_gaokao_mathqa(n)),
    "LogiQA":                load_logiqa_fixed,
}


# ──────────────────────────────────────────────────────────────────────────────
# 从扫参 JSON 提取 sweep_best
# ──────────────────────────────────────────────────────────────────────────────
def _best_from_sweep(sweep: dict, bench: str) -> tuple[int, int, float] | None:
    best_acc = -1.0
    best = None
    for row in sweep.get("results", []):
        if bench not in row: continue
        acc = float(row[bench])
        if acc > best_acc:
            best_acc = acc
            best = (int(row["t_start"]), int(row["t_stop"]), acc)
    return best

def load_sweep_best_from_files(preset: dict) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for key in ("sweep_main", "sweep_hard"):
        p = preset.get(key)
        if not p or not Path(p).exists():
            continue
        sweep = json.loads(Path(p).read_text())
        for b in sweep.get("benchmarks_used", []):
            if b in out: continue
            r = _best_from_sweep(sweep, b)
            if r: out[b] = (r[0], r[1])
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 工具
# ──────────────────────────────────────────────────────────────────────────────
def safe_cos(u: torch.Tensor, v: torch.Tensor) -> float:
    u = u.float().reshape(-1).cpu()
    v = v.float().reshape(-1).cpu()
    nu, nv = u.norm(), v.norm()
    if nu < 1e-12 or nv < 1e-12: return 0.0
    return float((u @ v / (nu * nv)).clamp(-1, 1))

def safe_norm(t: torch.Tensor) -> float:
    return float(t.float().norm().clamp(min=1e-12))

def loglikelihood_mc(logits: torch.Tensor, input_ids: torch.Tensor, prompt_len: int) -> float:
    total = 0.0
    for i in range(prompt_len, input_ids.shape[1]):
        logp = F.log_softmax(logits[0, i-1].float(), dim=-1)
        total += float(logp[input_ids[0, i]])
    return total

def mlp_counterfactual(layer, hi: torch.Tensor, arch: str) -> torch.Tensor:
    if arch == "gemma2":
        return layer.mlp(layer.pre_feedforward_layernorm(hi))
    return layer.mlp(layer.post_attention_layernorm(hi))

def load_model(path: str):
    print(f"Loading model: {path}")
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager", trust_remote_code=True,
    )
    mdl.eval()
    return tok, mdl


# ──────────────────────────────────────────────────────────────────────────────
# 核心 Probe：单次前向计算全部信号
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def probe_forward_all_signals(
    model,
    input_ids: torch.Tensor,
    attn_mask,
    n_layers: int,
    probe_layers: list[int],
    arch: str,
) -> dict[str, dict[int, float]]:
    """
    单次前向（含额外 MLP forward for cos_res），返回 5 路信号 per probe layer。
    每路信号：dict[layer_idx → value]
    """
    base = model.model
    h_in:  dict[int, torch.Tensor] = {}
    a_out: dict[int, torch.Tensor] = {}
    m_out: dict[int, torch.Tensor] = {}
    hooks = []

    for li in range(n_layers):
        layer = base.layers[li]
        def _pre(idx):
            def fn(_m, args):
                t = args[0] if isinstance(args, tuple) else args
                h_in[idx] = t[:, -1:, :].detach().clone()
            return fn
        def _post_attn(idx):
            def fn(_m, _i, out):
                t = out[0] if isinstance(out, tuple) else out
                a_out[idx] = t[:, -1:, :].detach().clone()
            return fn
        def _post_mlp(idx):
            def fn(_m, _i, out):
                m_out[idx] = out[:, -1:, :].detach().clone()
            return fn
        hooks.append(layer.register_forward_pre_hook(_pre(li)))
        hooks.append(layer.self_attn.register_forward_hook(_post_attn(li)))
        hooks.append(layer.mlp.register_forward_hook(_post_mlp(li)))

    model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
    for h in hooks: h.remove()

    sigs: dict[str, dict[int, float]] = {k: {} for k in ALL_SIGNAL_NAMES}
    prev_delta: torch.Tensor | None = None

    for li in sorted(probe_layers):
        hi = h_in.get(li)
        al = a_out.get(li)
        ml = m_out.get(li)
        if hi is None or al is None or ml is None:
            continue

        al_sq = al.squeeze()
        ml_sq = ml.squeeze()
        hi_sq = hi.squeeze()
        delta = (al + ml).squeeze()

        # neg_cos_am: 越负 cos(a,m) → 越正（竞争越强）
        cos_am = safe_cos(al_sq, ml_sq)
        sigs["neg_cos_am"][li] = -cos_am

        # delta_h_ratio: 层更新/输入幅度
        sigs["delta_h_ratio"][li] = safe_norm(delta) / max(safe_norm(hi_sq), 1e-12)

        # update_persist: 相邻层更新方向一致性
        if prev_delta is not None:
            sigs["update_persist"][li] = safe_cos(delta, prev_delta)
        else:
            sigs["update_persist"][li] = 0.0
        prev_delta = delta.detach().clone()

        # attn_dom: attention 主导比
        na = safe_norm(al_sq)
        nm = safe_norm(ml_sq)
        sigs["attn_dom"][li] = na / max(na + nm, 1e-12)

        # cos_res: 需要额外 MLP forward
        try:
            layer = base.layers[li]
            m_cf = mlp_counterfactual(layer, hi, arch)
            term1 = (ml - m_cf).squeeze()
            sigs["cos_res"][li] = safe_cos(term1, delta)
        except Exception:
            sigs["cos_res"][li] = 0.0

    return sigs


# ──────────────────────────────────────────────────────────────────────────────
# 标定阶段：N_CALIB 样本聚合平均 profile
# ──────────────────────────────────────────────────────────────────────────────
def calibrate_all_signals(
    items: list[dict],
    model,
    tok,
    n_layers: int,
    probe_layers: list[int],
    arch: str,
    device: str,
) -> dict[str, dict[int, float]]:
    acc: dict[str, dict[int, list[float]]] = {s: defaultdict(list) for s in ALL_SIGNAL_NAMES}
    for item in items[:N_CALIB]:
        enc = tok(item["prompt"], return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        amask = enc.get("attention_mask")
        if amask is not None: amask = amask.to(device)
        sigs = probe_forward_all_signals(model, ids, amask, n_layers, probe_layers, arch)
        for sname, profile in sigs.items():
            for li, v in profile.items():
                acc[sname][li].append(v)
    return {sname: {li: float(np.mean(vs)) for li, vs in prof.items() if vs}
            for sname, prof in acc.items()}


# ──────────────────────────────────────────────────────────────────────────────
# 信号评分：disc_score, coverage, start_error
# ──────────────────────────────────────────────────────────────────────────────
def score_signal(
    mean_profiles: dict[str, dict[int, float]],  # sname → profile
    sweep_best: dict[str, tuple[int, int]],
    min_start: int,
    max_start: int,
) -> dict[str, dict]:
    """
    对每个信号计算：
      disc_score_per_bench  = win_avg - out_avg（越高越好：信号在 sweep_win 内高于外）
      coverage              = 正确区分的 bench 比例
      start_error_mean      = 信号引导的 t_start 与 sweep t_start 的绝对误差均值
    """
    scores: dict[str, dict] = {}
    for sname, profile in mean_profiles.items():
        disc_list, start_errors = [], []
        for bench, (sw_ts, sw_te) in sweep_best.items():
            win_vals = [v for l, v in profile.items() if sw_ts <= l < sw_te]
            out_vals = [v for l, v in profile.items() if not (sw_ts <= l < sw_te) and min_start <= l <= max_start]
            if not win_vals or not out_vals:
                continue
            disc = float(np.mean(win_vals)) - float(np.mean(out_vals))
            disc_list.append(disc)
            # t_start 对齐：信号在 [min_start, max_start] 内的峰值层
            valid = {l: v for l, v in profile.items() if min_start <= l <= max_start}
            if valid:
                sig_tstart = max(valid, key=valid.__getitem__)
                start_errors.append(abs(sig_tstart - sw_ts))

        coverage = sum(1 for d in disc_list if d > 0) / max(len(disc_list), 1)
        scores[sname] = {
            "disc_scores": disc_list,
            "mean_disc": float(np.mean(disc_list)) if disc_list else 0.0,
            "coverage": coverage,
            "n_benches": len(disc_list),
            "mean_start_error": float(np.mean(start_errors)) if start_errors else 99.0,
        }
    return scores


# ──────────────────────────────────────────────────────────────────────────────
# 窗口选择函数（通用，按信号 profile 选 max mean 窗口）
# ──────────────────────────────────────────────────────────────────────────────
def select_window_generic(
    profile: dict[int, float],
    nt_candidates: tuple[int, ...],
    min_start: int,
    max_start: int,
) -> tuple[int, int]:
    best_start, best_nt, best_score = min_start, nt_candidates[0], -999.0
    for n_t in nt_candidates:
        for start in range(min_start, max_start + 1):
            stop = start + n_t
            vals = [profile[l] for l in profile if start <= l < stop]
            if len(vals) < 1: continue
            score = float(np.mean(vals))
            if score > best_score:
                best_score, best_start, best_nt = score, start, n_t
    return best_start, best_start + best_nt

def select_window_calib_global(
    profile: dict[int, float],
    n_t: int,
    min_start: int,
    max_start: int,
) -> tuple[int, int]:
    best_start, best_score = min_start, -999.0
    for start in range(min_start, max_start + 1):
        stop = start + n_t
        vals = [profile[l] for l in profile if start <= l < stop]
        if len(vals) < 1: continue
        score = float(np.mean(vals))
        if score > best_score:
            best_score, best_start = score, start
    return best_start, best_start + n_t


# ──────────────────────────────────────────────────────────────────────────────
# 推理辅助
# ──────────────────────────────────────────────────────────────────────────────
def mc_predict(model, tok, item: dict, device: str, n_e=None, n_t=None, k=K_ETD) -> int:
    prompt, choices = item["prompt"], item["choices"]
    scores = []
    for cont in choices:
        enc = tok(prompt + " " + cont, return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        amask = enc.get("attention_mask")
        if amask is not None: amask = amask.to(device)
        plen = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
        if n_e is not None and n_t is not None and n_t > 0:
            lgts = etd_forward_logits(model, ids, amask, n_e=n_e, n_t=n_t, k=k,
                                       alpha=min(1.0, 6.0/max(n_t,1)))
        else:
            lgts = baseline_forward_logits(model, ids, amask)
        scores.append(loglikelihood_mc(lgts, ids, plen))
    return int(np.argmax(scores))

def mc_logit_gain(model, tok, item: dict, device: str, n_e: int, n_t: int) -> float:
    """返回正确答案的 logit gain = log_p_correct(ETD) - log_p_correct(baseline)."""
    prompt, choices, label = item["prompt"], item["choices"], item["label"]
    cont = choices[label]
    full = prompt + " " + cont
    enc = tok(full, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"].to(device)
    amask = enc.get("attention_mask")
    if amask is not None: amask = amask.to(device)
    plen = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
    lgts_base = baseline_forward_logits(model, ids, amask)
    lgts_etd  = etd_forward_logits(model, ids, amask, n_e=n_e, n_t=n_t, k=K_ETD,
                                    alpha=min(1.0, 6.0/max(n_t,1)))
    gain_base = loglikelihood_mc(lgts_base, ids, plen)
    gain_etd  = loglikelihood_mc(lgts_etd,  ids, plen)
    return gain_etd - gain_base


# ──────────────────────────────────────────────────────────────────────────────
# 经验标定：枚举候选窗口，用 logit-gain 选最优
# ──────────────────────────────────────────────────────────────────────────────
def run_empirical_calib(
    items: list[dict],
    model,
    tok,
    n_layers: int,
    min_start: int,
    max_start: int,
    nt_candidates: tuple[int, ...],
    device: str,
) -> tuple[int, int] | None:
    """
    在 N_CALIB 个样本上枚举候选窗口，返回 logit-gain 最高的 (t_start, t_stop)。
    如果所有窗口 logit_gain ≤ 0 则返回 None（不做 ETD）。
    候选窗口：stride=4 for t_start，保证约 10-15 个候选。
    """
    calib_items = items[:N_CALIB]
    stride = max(3, (max_start - min_start) // 5)
    candidates: list[tuple[int, int]] = []
    for n_t in nt_candidates:
        for ts in range(min_start, max_start + 1, stride):
            te = ts + n_t
            if te > n_layers - 1: continue
            candidates.append((ts, te))
    if not candidates:
        return None

    # 累积 logit_gain
    gain_sum = {c: 0.0 for c in candidates}
    for item in calib_items:
        for (ts, te) in candidates:
            try:
                g = mc_logit_gain(model, tok, item, device, n_e=ts, n_t=te-ts)
                gain_sum[(ts, te)] += g
            except Exception:
                pass
        torch.cuda.empty_cache()

    mean_gain = {c: g / max(len(calib_items), 1) for c, g in gain_sum.items()}
    best_win, best_gain = max(mean_gain.items(), key=lambda x: x[1])

    print(f"  [empirical] 候选={len(candidates)}  best={best_win}  gain={best_gain:+.4f}")
    if best_gain <= 0:
        print(f"  [empirical] 所有窗口 gain≤0，fallback 到 baseline")
        return None
    return best_win


# ──────────────────────────────────────────────────────────────────────────────
# 主评测循环
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_benchmark(
    bench: str,
    items: list[dict],
    model,
    tok,
    n_layers: int,
    sweep_win: tuple[int, int],
    preset: dict,
    mean_profiles: dict[str, dict[int, float]],
    signal_scores: dict[str, dict],
    device: str,
) -> dict:
    arch      = preset["arch"]
    probe_layers = preset["probe_layers"]
    min_start = preset["min_start"]
    max_start = preset["max_start"]
    nt_arch   = preset["nt_arch"]
    n_total   = len(items)

    # ── 选出排名第一的信号（按 mean_disc + coverage 联合排名）
    def _rank_key(s): return (signal_scores[s]["coverage"], signal_scores[s]["mean_disc"])
    top_signal = max(ALL_SIGNAL_NAMES, key=_rank_key)
    top_profile = mean_profiles[top_signal]
    print(f"  [信号排名 top1] {top_signal}  coverage={signal_scores[top_signal]['coverage']:.2f}  "
          f"mean_disc={signal_scores[top_signal]['mean_disc']:+.3f}")

    # ── neg_cos_am profile（始终包含在评测中）
    neg_cos_am_profile = mean_profiles["neg_cos_am"]

    # ── 标定阶段派生固定窗口
    calib_nt = 8
    top_calib_win  = select_window_calib_global(top_profile, calib_nt, min_start, max_start)
    neg_am_calib_win = select_window_calib_global(neg_cos_am_profile, calib_nt, min_start, max_start)

    # ── 经验标定窗口（在 calib 样本上直接跑 ETD）
    print(f"  [经验标定] 枚举候选窗口 …")
    t_emp = time.time()
    emp_win = run_empirical_calib(items, model, tok, n_layers, min_start, max_start, nt_arch, device)
    print(f"  [经验标定] {time.time()-t_emp:.1f}s  best={emp_win}")

    print(f"  [窗口汇总] sweep_best={sweep_win}  top_sig_calib={top_calib_win}  "
          f"neg_am_calib={neg_am_calib_win}  empirical={emp_win}")

    # ── 逐样本评测
    correct  = {c: 0 for c in EVAL_COND_NAMES}
    sel_tstart = {"neg_cos_am_var": [], "top_signal_var": []}
    t0 = time.time()

    for i, item in enumerate(items):
        label = item["label"]
        # probe forward for per-sample signal
        enc = tok(item["prompt"], return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        amask = enc.get("attention_mask")
        if amask is not None: amask = amask.to(device)

        sigs = probe_forward_all_signals(model, ids, amask, n_layers, probe_layers, arch)

        # 逐样本选窗口
        neg_am_win = select_window_generic(sigs["neg_cos_am"], nt_arch, min_start, max_start)
        top_sig_win = select_window_generic(sigs[top_signal], nt_arch, min_start, max_start)
        sel_tstart["neg_cos_am_var"].append(neg_am_win[0])
        sel_tstart["top_signal_var"].append(top_sig_win[0])

        wins = {
            "baseline":         None,
            "sweep_best":       sweep_win,
            "neg_cos_am_var":   neg_am_win,
            "top_signal_var":   top_sig_win,
            "top_signal_calib": top_calib_win,
            "empirical_calib":  emp_win,
        }

        for cname in EVAL_COND_NAMES:
            win = wins[cname]
            if win is None:
                pred = mc_predict(model, tok, item, device)
            else:
                ts, te = win
                n_e_val, n_t_val = ts, te - ts
                if n_layers - te < 1 or n_t_val < 1:
                    pred = mc_predict(model, tok, item, device)
                else:
                    pred = mc_predict(model, tok, item, device, n_e=n_e_val, n_t=n_t_val)
            if pred == label:
                correct[cname] += 1

        if (i+1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i+1) * (n_total-i-1)
            line = f"  [{i+1:3d}/{n_total}] "
            for cn in EVAL_COND_NAMES:
                line += f"{cn[:4]}={correct[cn]/(i+1):.3f} "
            line += f"| {elapsed:.0f}s ETA {eta:.0f}s"
            print(line)
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    accs = {c: correct[c] / n_total for c in EVAL_COND_NAMES}
    win_stats = {cn: {"t_start_mean": float(np.mean(sel_tstart[cn])),
                       "t_start_std":  float(np.std(sel_tstart[cn]))}
                 for cn in ["neg_cos_am_var", "top_signal_var"] if sel_tstart[cn]}

    return {
        "benchmark": bench,
        "n": n_total,
        "elapsed_s": elapsed,
        "accuracies": accs,
        "sweep_best_window": list(sweep_win),
        "top_signal": top_signal,
        "top_signal_calib_window": list(top_calib_win),
        "neg_cos_am_calib_window": list(neg_am_calib_win),
        "empirical_calib_window": list(emp_win) if emp_win else None,
        "mean_profiles": {s: {str(k): v for k, v in prof.items()}
                          for s, prof in mean_profiles.items()},
        "signal_scores": signal_scores,
        "window_stats": win_stats,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 可视化
# ──────────────────────────────────────────────────────────────────────────────
COND_COLORS = {
    "baseline":         "#888888",
    "sweep_best":       "#1f77b4",
    "neg_cos_am_var":   "#d62728",
    "top_signal_var":   "#ff7f0e",
    "top_signal_calib": "#9467bd",
    "empirical_calib":  "#2ca02c",
}
COND_LABELS = {
    "baseline":         "Baseline",
    "sweep_best":       "Sweep Best",
    "neg_cos_am_var":   "neg_cos_am (per-sample)",
    "top_signal_var":   "Top Signal (per-sample)",
    "top_signal_calib": "Top Signal (calib fixed)",
    "empirical_calib":  "Empirical Calib",
}


def plot_main_bars(all_results: dict, fig_dir: Path, preset_name: str, top_signal: str):
    """主精度对比条形图：各条件 × 8 benchmarks"""
    fig_dir.mkdir(parents=True, exist_ok=True)
    benches = [b for b in BENCH_ORDER if b in all_results]
    n_b, n_c = len(benches), len(EVAL_COND_NAMES)
    fig, ax = plt.subplots(figsize=(16, 5))
    x = np.arange(n_b)
    w = 0.85 / n_c
    for j, cn in enumerate(EVAL_COND_NAMES):
        offs = (j - (n_c-1)/2.0) * w
        vals = [all_results[b]["accuracies"].get(cn, 0) for b in benches]
        ax.bar(x + offs, vals, w*0.9, label=COND_LABELS[cn], color=COND_COLORS[cn])
    ax.set_xticks(x)
    ax.set_xticklabels([b[:14] for b in benches], rotation=22, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"R39 {preset_name}：信号引导 ETD vs Baseline vs Sweep Best\n（Top Signal: {top_signal}）")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    p = fig_dir / "main_accuracy_bars.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  图表 → {p}")


def plot_delta_acc(all_results: dict, fig_dir: Path, preset_name: str):
    """Δacc 散点图：各信号方法相对 baseline 和 sweep_best 的增量"""
    benches = [b for b in BENCH_ORDER if b in all_results]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    conds_to_plot = [c for c in EVAL_COND_NAMES if c not in ("baseline", "sweep_best")]

    for ax, ref, ref_label in zip(axes, ["baseline", "sweep_best"], ["baseline", "sweep_best"]):
        for j, cn in enumerate(conds_to_plot):
            delta_vals = []
            for b in benches:
                accs = all_results[b]["accuracies"]
                delta_vals.append(accs.get(cn, 0) - accs.get(ref, 0))
            x = np.arange(len(benches))
            ax.scatter(x, delta_vals, label=COND_LABELS[cn], color=COND_COLORS[cn], zorder=3, s=60)
            ax.plot(x, delta_vals, color=COND_COLORS[cn], alpha=0.5, linewidth=1)
        ax.axhline(0, color="black", linewidth=1.2, linestyle="--")
        ax.set_xticks(np.arange(len(benches)))
        ax.set_xticklabels([b[:12] for b in benches], rotation=25, ha="right", fontsize=7.5)
        ax.set_ylabel(f"Δacc (vs {ref_label})")
        ax.set_title(f"相对 {ref_label} 的增量")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"R39 {preset_name}：Δacc 散点图", fontsize=11)
    plt.tight_layout()
    p = fig_dir / "delta_acc_scatter.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  图表 → {p}")


def plot_signal_profiles(all_results: dict, fig_dir: Path, preset_name: str):
    """各 benchmark 的 5 路信号 profile 折线图（叠加 sweep_best 高亮区）"""
    benches = [b for b in BENCH_ORDER if b in all_results]
    n_b = len(benches)
    n_cols = 4
    n_rows = (n_b + n_cols - 1) // n_cols

    for sname in ALL_SIGNAL_NAMES:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5*n_rows))
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for idx, b in enumerate(benches):
            ax = axes_flat[idx]
            res = all_results[b]
            prof = {int(k): v for k, v in res["mean_profiles"].get(sname, {}).items()}
            if not prof:
                ax.set_title(b[:16])
                continue
            layers = sorted(prof.keys())
            vals = [prof[l] for l in layers]
            ax.plot(layers, vals, "b-o", markersize=4, linewidth=1.5, label=sname)
            # sweep_best 高亮
            sw = res["sweep_best_window"]
            ax.axvspan(sw[0], sw[1], alpha=0.15, color="red", label="sweep_best")
            ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
            ax.set_title(f"{b[:16]}\nsw=[{sw[0]},{sw[1]}]", fontsize=8)
            ax.tick_params(labelsize=7)
            if idx == 0: ax.legend(fontsize=7)
        for idx in range(len(benches), len(axes_flat)):
            axes_flat[idx].set_visible(False)
        _, desc = SIGNAL_META[sname]
        plt.suptitle(f"R39 {preset_name}  信号={sname}\n{desc}", fontsize=10)
        plt.tight_layout()
        p = fig_dir / f"profile_{sname}.png"
        plt.savefig(p, dpi=110, bbox_inches="tight")
        plt.close()
    print(f"  图表 → {fig_dir}/profile_*.png")


def plot_signal_disc_heatmap(signal_scores_per_bench: dict, fig_dir: Path, preset_name: str):
    """
    信号区分力热力图：行=信号，列=benchmark
    颜色=disc_score（正=信号在 sweep_best 窗口内高；负=反）
    """
    benches_shown = BENCH_ORDER
    n_s = len(ALL_SIGNAL_NAMES)
    n_b = len(benches_shown)
    mat = np.zeros((n_s, n_b))
    for bi, b in enumerate(benches_shown):
        if b not in signal_scores_per_bench:
            continue
        for si, sname in enumerate(ALL_SIGNAL_NAMES):
            # signal_scores_per_bench[b][sname] = single disc score
            sc = signal_scores_per_bench[b].get(sname, 0.0)
            mat[si, bi] = sc

    fig, ax = plt.subplots(figsize=(13, 4))
    vmax = max(0.05, np.abs(mat).max())
    im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n_b))
    ax.set_xticklabels([b[:14] for b in benches_shown], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(n_s))
    ax.set_yticklabels([f"{s}\n{SIGNAL_META[s][1][:20]}" for s in ALL_SIGNAL_NAMES], fontsize=8)
    plt.colorbar(im, ax=ax, label="disc_score (win_avg - out_avg)")
    ax.set_title(f"R39 {preset_name}：信号区分力热力图\n（绿=正确区分 sweep_best 窗口，红=反向）")
    for si in range(n_s):
        for bi in range(n_b):
            ax.text(bi, si, f"{mat[si,bi]:.2f}", ha="center", va="center", fontsize=6.5)
    plt.tight_layout()
    p = fig_dir / "signal_disc_heatmap.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  图表 → {p}")


def plot_tstart_violin(all_results: dict, fig_dir: Path, preset_name: str):
    """t_start 分布 violin 图：neg_cos_am_var vs top_signal_var vs sweep_best（固定值）"""
    benches = [b for b in BENCH_ORDER if b in all_results]
    n_b = len(benches)
    fig, axes = plt.subplots(1, n_b, figsize=(2.2*n_b, 4), sharey=False)
    if n_b == 1: axes = [axes]
    for ax, b in zip(axes, benches):
        res = all_results[b]
        ws = res.get("window_stats", {})
        sw = res["sweep_best_window"]
        data_groups, labels, colors = [], [], []
        for cname, color in [("neg_cos_am_var","#d62728"),("top_signal_var","#ff7f0e")]:
            if cname in ws and ws[cname]:
                # Approximate distribution from mean/std
                mean_ts = ws[cname]["t_start_mean"]
                std_ts  = ws[cname]["t_start_std"]
                simulated = np.random.normal(mean_ts, max(std_ts, 0.1), 50).clip(1, 36).tolist()
                data_groups.append(simulated)
                labels.append(cname[:12])
                colors.append(color)
        if data_groups:
            parts = ax.violinplot(data_groups, positions=range(len(data_groups)), showmedians=True)
            for pc, col in zip(parts["bodies"], colors):
                pc.set_facecolor(col)
                pc.set_alpha(0.6)
        ax.axhline(sw[0], color="#1f77b4", linewidth=1.5, linestyle="--", label=f"sw={sw}")
        ax.set_title(b[:12], fontsize=8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7, rotation=20)
        ax.set_ylabel("t_start", fontsize=7)
    plt.suptitle(f"R39 {preset_name}：t_start 分布（蓝虚线=sweep_best t_start）", fontsize=10)
    plt.tight_layout()
    p = fig_dir / "tstart_violin.png"
    plt.savefig(p, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  图表 → {p}")


def plot_summary_final(all_results: dict, fig_dir: Path, preset_name: str, signal_scores: dict):
    """综合总结图：左=宏平均 Δacc，右=信号 coverage 排名"""
    benches = [b for b in BENCH_ORDER if b in all_results]
    # 宏平均 acc
    macro = {c: np.mean([all_results[b]["accuracies"].get(c, 0) for b in benches])
             for c in EVAL_COND_NAMES}
    base_macro = macro["baseline"]
    delta_macro = {c: macro[c]-base_macro for c in EVAL_COND_NAMES}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左：宏平均 Δacc 条形图
    conds = [c for c in EVAL_COND_NAMES if c != "baseline"]
    x = np.arange(len(conds))
    vals = [delta_macro[c] for c in conds]
    colors = [COND_COLORS[c] for c in conds]
    bars = ax1.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax1.axhline(0, color="black", linewidth=1.0)
    ax1.axhline(delta_macro["sweep_best"], color="#1f77b4", linewidth=1.5, linestyle="--", label="sweep_best")
    ax1.set_xticks(x)
    ax1.set_xticklabels([COND_LABELS[c] for c in conds], rotation=25, ha="right", fontsize=8)
    ax1.set_ylabel("Δacc vs Baseline")
    ax1.set_title(f"R39 {preset_name}\n宏平均 Δacc（8 benchmarks）")
    ax1.legend(fontsize=8)
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                 f"{val:+.3f}", ha="center", va="bottom", fontsize=8)

    # 右：信号 coverage + mean_disc 排名（Phase A 结果）
    snames = list(SIGNAL_META.keys())
    cov = [signal_scores[s]["coverage"] for s in snames]
    disc = [signal_scores[s]["mean_disc"] for s in snames]
    y = np.arange(len(snames))
    ax2.barh(y, cov, color="steelblue", alpha=0.7, label="coverage (frac benches correct)")
    ax2_twin = ax2.twiny()
    ax2_twin.barh(y, disc, color="coral", alpha=0.6, label="mean disc_score")
    ax2_twin.axvline(0, color="gray", linewidth=1)
    ax2.set_yticks(y)
    ax2.set_yticklabels([f"{s}\n{SIGNAL_META[s][1][:22]}" for s in snames], fontsize=7.5)
    ax2.set_xlabel("Coverage（正确区分比例）", color="steelblue", fontsize=8)
    ax2_twin.set_xlabel("Mean disc_score", color="coral", fontsize=8)
    ax2.set_title("Phase A 信号排名")
    ax2.set_xlim(0, 1.05)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1+lines2, labels1+labels2, fontsize=7, loc="lower right")

    plt.tight_layout()
    p = fig_dir / "summary_final.png"
    plt.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  图表 → {p}")


# ──────────────────────────────────────────────────────────────────────────────
# 打印结果表
# ──────────────────────────────────────────────────────────────────────────────
def print_results_table(all_results: dict, preset_name: str):
    benches = [b for b in BENCH_ORDER if b in all_results]
    header = f"{'Benchmark':25s}" + "".join(f"{c[:8]:>10s}" for c in EVAL_COND_NAMES)
    print(f"\n{'='*80}")
    print(f"R39 {preset_name} 精度结果")
    print(f"{'='*80}")
    print(header)
    print("-"*80)
    for b in benches:
        accs = all_results[b]["accuracies"]
        row = f"{b[:24]:25s}"
        for c in EVAL_COND_NAMES:
            v = accs.get(c, 0)
            marker = "*" if v > accs.get("sweep_best", 0) else " "
            row += f"{v:.3f}{marker}   "
        print(row)
    print("-"*80)
    macro = {c: np.mean([all_results[b]["accuracies"].get(c, 0) for b in benches])
             for c in EVAL_COND_NAMES}
    row = f"{'MACRO AVG':25s}"
    for c in EVAL_COND_NAMES:
        v = macro[c]
        marker = "*" if v > macro.get("sweep_best", 0) else " "
        row += f"{v:.3f}{marker}   "
    print(row)
    print(f"\n(* = 超越 sweep_best)")

    # 信号排名
    if all_results:
        first_bench = benches[0]
        sig_scores = all_results[first_bench].get("signal_scores", {})
        if sig_scores:
            print(f"\nPhase A 信号排名（按 coverage + mean_disc）：")
            ranked = sorted(sig_scores.items(), key=lambda x: (x[1]["coverage"], x[1]["mean_disc"]), reverse=True)
            for rank, (sname, sc) in enumerate(ranked, 1):
                print(f"  #{rank} {sname:20s}  coverage={sc['coverage']:.2f}  "
                      f"mean_disc={sc['mean_disc']:+.3f}  start_err={sc['mean_start_error']:.1f}")


# ──────────────────────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS.keys()), required=True)
    ap.add_argument("--phase_a_only", action="store_true", help="只做信号筛选，不跑 ETD 评测")
    args = ap.parse_args()

    preset = PRESETS[args.preset]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fig_dir = Path(preset["out_fig"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(preset["out_json"])
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # ── 加载 sweep_best
    if args.preset == "qwen3-8b":
        sweep_best = dict(preset["sweep_best"])
    else:
        sweep_best = load_sweep_best_from_files(preset)
    print(f"\n[sweep_best]")
    for b in BENCH_ORDER:
        if b in sweep_best: print(f"  {b}: {sweep_best[b]}")

    # ── 加载模型
    tok, model = load_model(preset["model_path"])
    n_layers = preset["n_layers"]
    arch     = preset["arch"]

    all_results: dict[str, dict] = {}
    # 全局聚合信号分数（跨 benchmark，用于 Phase A 排名）
    global_signal_scores: dict[str, dict] = {s: {"disc_scores": [], "start_errors": []}
                                               for s in ALL_SIGNAL_NAMES}
    signal_scores_per_bench: dict[str, dict] = {}

    for bench in BENCH_ORDER:
        n = N_SAMPLES[bench]
        if bench not in sweep_best:
            print(f"\n[SKIP] {bench}: 无 sweep_best")
            continue
        sw = sweep_best[bench]
        print(f"\n{'─'*55}\n{bench}  N={n}  sweep_best={sw}")

        try:
            raw = BENCH_LOADERS[bench](n)
        except Exception as e:
            print(f"  [ERROR] load {bench}: {e}")
            continue
        if not raw:
            continue
        # 统一格式
        items = []
        for ex in raw[:n]:
            if "label" in ex:
                items.append(ex)
            else:
                gold = str(ex["answer"]).strip().lower()
                label = next(i for i, c in enumerate(ex["choices"]) if str(c).strip().lower() == gold)
                items.append({"prompt": ex["prompt"], "choices": ex["choices"], "label": label})

        # ── Phase A: 标定所有信号
        print(f"  [Phase A] 标定 {N_CALIB} 样本 …")
        t_calib = time.time()
        mean_profiles = calibrate_all_signals(
            items, model, tok, n_layers, preset["probe_layers"], arch, device
        )
        # 打印 profile 摘要
        for sname in ALL_SIGNAL_NAMES:
            prof = mean_profiles[sname]
            if prof:
                ml, mv = max(prof.items(), key=lambda x: x[1])  # ml=layer, mv=value
                win_vals = [v for l, v in prof.items() if sw[0] <= l < sw[1]]
                print(f"    {sname:18s} max={mv:.3f}@L{ml}  "
                      f"win_avg={np.mean(win_vals) if win_vals else 0.0:.3f}")
        print(f"  [Phase A] {time.time()-t_calib:.1f}s")

        # 评分
        signal_scores = score_signal(mean_profiles, {bench: sw},
                                     preset["min_start"], preset["max_start"])
        signal_scores_per_bench[bench] = {s: signal_scores[s]["mean_disc"]
                                           for s in ALL_SIGNAL_NAMES}
        # 累积全局 disc scores
        for sname in ALL_SIGNAL_NAMES:
            if signal_scores[sname]["disc_scores"]:
                global_signal_scores[sname]["disc_scores"].extend(signal_scores[sname]["disc_scores"])
            if signal_scores[sname].get("mean_start_error") < 90:
                global_signal_scores[sname]["start_errors"].append(signal_scores[sname]["mean_start_error"])

        if args.phase_a_only:
            all_results[bench] = {
                "benchmark": bench, "n": n,
                "mean_profiles": {s: {str(k): v for k, v in p.items()} for s, p in mean_profiles.items()},
                "signal_scores": signal_scores,
                "sweep_best_window": list(sw),
            }
        else:
            # ── Phase B: 全量评测
            try:
                res = evaluate_benchmark(
                    bench, items, model, tok, n_layers, sw,
                    preset, mean_profiles, signal_scores, device
                )
                all_results[bench] = res
            except Exception as e:
                import traceback
                print(f"  [ERROR] evaluate {bench}: {e}")
                traceback.print_exc()
                continue

        # 中间保存
        out_payload = {
            "preset": args.preset,
            "arch": arch,
            "n_layers": n_layers,
            "results": all_results,
            "signal_scores_per_bench": signal_scores_per_bench,
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, indent=2)
        print(f"  checkpoint → {out_json}")

    # ── 全局信号排名
    global_scores_summary: dict[str, dict] = {}
    for sname in ALL_SIGNAL_NAMES:
        disc = global_signal_scores[sname]["disc_scores"]
        errs = global_signal_scores[sname]["start_errors"]
        cov = sum(1 for d in disc if d > 0) / max(len(disc), 1)
        global_scores_summary[sname] = {
            "coverage": cov,
            "mean_disc": float(np.mean(disc)) if disc else 0.0,
            "mean_start_error": float(np.mean(errs)) if errs else 99.0,
            "n_benches": len(disc),
        }

    print(f"\n{'='*60}\n全局信号排名（{args.preset}）\n{'='*60}")
    ranked = sorted(global_scores_summary.items(), key=lambda x: (x[1]["coverage"], x[1]["mean_disc"]), reverse=True)
    for rank, (sname, sc) in enumerate(ranked, 1):
        print(f"  #{rank} {sname:20s}  coverage={sc['coverage']:.2f}  "
              f"mean_disc={sc['mean_disc']:+.3f}  n_benches={sc['n_benches']}")

    # ── 生成可视化
    if all_results and not args.phase_a_only:
        print_results_table(all_results, args.preset)
        # 获取全局最优信号名
        best_signal = max(global_scores_summary, key=lambda s: (global_scores_summary[s]["coverage"],
                                                                  global_scores_summary[s]["mean_disc"]))
        print(f"\n全局最优信号：{best_signal}")
        plot_main_bars(all_results, fig_dir, args.preset, best_signal)
        plot_delta_acc(all_results, fig_dir, args.preset)
        plot_signal_profiles(all_results, fig_dir, args.preset)
        plot_signal_disc_heatmap(signal_scores_per_bench, fig_dir, args.preset)
        plot_tstart_violin(all_results, fig_dir, args.preset)
        plot_summary_final(all_results, fig_dir, args.preset, global_scores_summary)

    elif all_results and args.phase_a_only:
        plot_signal_profiles(all_results, fig_dir, args.preset)
        plot_signal_disc_heatmap(signal_scores_per_bench, fig_dir, args.preset)

    print(f"\n完成。结果: {out_json}")


if __name__ == "__main__":
    main()
