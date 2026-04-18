#!/usr/bin/env python3
"""
R39B 精化实验：基于 R39A 结论（neg_cos_am coverage=1.00）改进窗口选择策略。

R39A 发现：
  - neg_cos_am（-cos(a_l,m_l) 竞争区）是最强信号（coverage=1.00，mean_disc=+0.151）
  - 主要失败原因：窗口宽度 n_t 选择不对（GPQA 最优 n_t=2，CSQA 最优 n_t=12，搜索空间未覆盖）
  - 经验标定对 GPQA 等难任务因噪声选出错误晚层窗口

改进策略（四种新条件）：
  1. neg_cos_am_adaptive     自适应 n_t：从 peak 向两侧扩展到 sig < 0.7*peak 为止，自然适配窄/宽窗口
  2. neg_cos_am_fine_nt      细粒度 n_t 候选 {2,4,6,8,12,14} + 逐样本选择
  3. empirical_acc30         经验标定 v2：N_CALIB=30，acc 作为目标（非 logit gain），避免难任务噪声
  4. hybrid_signal_emp       混合方案：neg_cos_am 信号定 t_start → empirical 在该 t_start ±3 范围内搜索 n_t

所有改进方案与 baseline、sweep_best 及 R39A 最佳条件对比。

用法：
  python experiments/exp_r39b_refined.py --preset qwen3-8b
  python experiments/exp_r39b_refined.py --preset llama3-8b
  python experiments/exp_r39b_refined.py --preset gemma2-2b
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
EXP  = ROOT / "experiments"
ETD_DIR = ROOT / "ETD"
for p in (str(ROOT), str(EXP), str(ETD_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import DownloadConfig, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from etd_forward import baseline_forward_logits, etd_forward_logits
from hard_mc_benchmark_loaders import load_agieval_gaokao_mathqa, load_gpqa_diamond

# ──────────────────────────────────────────────────────────────────────────────
N_CALIB_V2   = 30   # 经验标定 v2 使用更多样本
K_ETD        = 2
ADAPTIVE_THR = 0.65  # 自适应 n_t：扩展到 sig < threshold * peak 时停

N_SAMPLES = {
    "BoolQ": 100, "ARC-C": 100, "TruthfulQA": 50, "CSQA": 100,
    "MMLU-HS-Math": 100, "GPQA-Diamond": 100, "AGIEval-Gaokao-MathQA": 100, "LogiQA": 100,
}
BENCH_ORDER = ["BoolQ","ARC-C","TruthfulQA","CSQA","MMLU-HS-Math","GPQA-Diamond","AGIEval-Gaokao-MathQA","LogiQA"]

QWEN_SWEEP_BEST: dict[str, tuple[int, int]] = {
    "BoolQ": (8,22), "ARC-C": (14,20), "TruthfulQA": (16,19), "CSQA": (10,22),
    "MMLU-HS-Math": (10,18), "GPQA-Diamond": (18,20), "AGIEval-Gaokao-MathQA": (13,20),
    "LogiQA": (14,19),
}

# R39A best condition per benchmark (for inclusion in final plot)
R39A_BEST = {
    "qwen3-8b": EXP / "results/r39_signal_hunt_qwen3.json",
    "llama3-8b": EXP / "results/r39_signal_hunt_llama3.json",
    "gemma2-2b": EXP / "results/r39_signal_hunt_gemma2.json",
}

PRESETS: dict[str, dict] = {
    "qwen3-8b": {
        "model_path": "/root/autodl-tmp/model_qwen",
        "arch":  "qwen3",
        "n_layers": 36,
        "probe_layers": list(range(6, 33, 2)),
        "min_start": 9,
        "max_start": 26,
        "nt_fine":  (2, 4, 6, 8, 12, 14),   # 细粒度候选（含 n_t=2 for GPQA）
        "sweep_best": QWEN_SWEEP_BEST,
        "out_json": EXP / "results/r39b_refined_qwen3.json",
        "out_fig":  EXP / "figures/r39b_refined_qwen3",
    },
    "llama3-8b": {
        "model_path": "/root/autodl-tmp/Llama3-8B",
        "arch":  "llama",
        "n_layers": 32,
        "probe_layers": list(range(6, 27, 2)),
        "min_start": 8,
        "max_start": 20,
        "nt_fine":  (2, 4, 6, 8),
        "sweep_main": EXP / "llama3-8b/results/etd_layer_sweep_r30style.json",
        "sweep_hard": EXP / "llama3-8b/results/hard_mc/etd_layer_sweep_r30style.json",
        "out_json": EXP / "results/r39b_refined_llama3.json",
        "out_fig":  EXP / "figures/r39b_refined_llama3",
    },
    "gemma2-2b": {
        "model_path": "/root/autodl-tmp/Gemma2-2B",
        "arch":  "gemma2",
        "n_layers": 26,
        "probe_layers": list(range(4, 23, 2)),
        "min_start": 5,
        "max_start": 16,
        "nt_fine":  (2, 4, 8, 14, 18),
        "sweep_main": EXP / "gemma2-2b/results/etd_layer_sweep_r30style.json",
        "sweep_hard": EXP / "gemma2-2b/results/hard_mc/etd_layer_sweep_r30style.json",
        "out_json": EXP / "results/r39b_refined_gemma2.json",
        "out_fig":  EXP / "figures/r39b_refined_gemma2",
    },
}

EVAL_COND_NAMES = [
    "baseline", "sweep_best",
    "neg_cos_am_adaptive",
    "neg_cos_am_fine_nt",
    "empirical_acc30",
    "hybrid_signal_emp",
]

COND_COLORS = {
    "baseline":           "#888888",
    "sweep_best":         "#1f77b4",
    "neg_cos_am_adaptive":"#d62728",
    "neg_cos_am_fine_nt": "#ff7f0e",
    "empirical_acc30":    "#2ca02c",
    "hybrid_signal_emp":  "#9467bd",
}
COND_LABELS = {
    "baseline":           "Baseline",
    "sweep_best":         "Sweep Best",
    "neg_cos_am_adaptive":"neg_cos_am Adaptive n_t",
    "neg_cos_am_fine_nt": "neg_cos_am Fine n_t {2..14}",
    "empirical_acc30":    "Empirical Calib (acc, N=30)",
    "hybrid_signal_emp":  "Hybrid: signal t_start + emp n_t",
}


# ──────────────────────────────────────────────────────────────────────────────
# 数据加载（复用 R38 的加载器）
# ──────────────────────────────────────────────────────────────────────────────
def _fmt(prompt, choices, label):
    return {"prompt": prompt, "choices": [c.strip() for c in choices], "label": label}

def load_boolq(n):
    ds = load_dataset("aps/super_glue","boolq")["validation"]
    out = []
    for r in ds:
        if len(out)>=n: break
        lab = int(r["label"])
        if lab<0: continue
        out.append(_fmt(f"{r['passage']}\nQuestion: {r['question']}?\nAnswer:", ["no","yes"], lab))
    return out

def load_arc_c(n):
    ds = load_dataset("allenai/ai2_arc","ARC-Challenge")["test"]
    out = []
    for r in ds:
        if len(out)>=n: break
        key = r["answerKey"]
        label = ord(key)-ord("A") if key in "ABCD" else int(key)-1
        out.append(_fmt(f"Question: {r['question'].strip()}\nAnswer:", r["choices"]["text"], label))
    return out

def load_csqa(n):
    ds = load_dataset("tau/commonsense_qa")["validation"]
    lmap={"A":0,"B":1,"C":2,"D":3,"E":4}
    out=[]
    for r in ds:
        if len(out)>=n: break
        key=r["answerKey"]
        if key not in lmap: continue
        out.append(_fmt(f"Question: {r['question'].strip()}\nAnswer:", r["choices"]["text"], lmap[key]))
    return out

def load_truthfulqa(n):
    ds=load_dataset("truthfulqa/truthful_qa","multiple_choice")["validation"]
    out=[]
    for r in ds:
        if len(out)>=n: break
        out.append(_fmt(f"Question: {r['question']}\nAnswer:",
                        r["mc1_targets"]["choices"], int(np.argmax(r["mc1_targets"]["labels"]))))
    return out

def load_mmlu_hs_math(n):
    dc=DownloadConfig(local_files_only=True)
    ds=load_dataset("cais/mmlu","high_school_mathematics",download_config=dc)["test"]
    out=[]
    for r in ds:
        if len(out)>=n: break
        out.append(_fmt(f"Question: {r['question'].strip()}\nAnswer:", [str(c) for c in r["choices"]], int(r["answer"])))
    return out

def load_logiqa_fixed(n):
    def _strip(o): return re.sub(r"^[ABCDabcd]\.\s*","",str(o).strip())
    def _to_letter(r):
        lab=r.get("label") if r.get("label") is not None else r.get("answer")
        if isinstance(lab,(int,float)) and lab==int(lab):
            i=int(lab)
            if 0<=i<4: return "abcd"[i]
        s=str(lab).strip().lower()
        return s if s in "abcd" else None
    ds=load_dataset("fireworks-ai/logiqa",split="test")
    out=[]
    choices=["a","b","c","d"]
    for r in ds:
        label=_to_letter(r)
        if label is None: continue
        opts=r["options"]
        if hasattr(opts,"tolist"): opts=opts.tolist()
        prompt=(f"Passage: {r['context']}\nQuestion: {r['question']}\nChoices:\n"
                +"\n".join(f"{x.upper()}. {_strip(o)}" for x,o in zip(choices,opts))+"\nAnswer:")
        out.append({"prompt":prompt,"choices":choices,"label":choices.index(label)})
        if len(out)>=n: break
    return out

def _adapt_hard(items):
    return [{"prompt":it["prompt"],"choices":it["choices"],"label":it.get("valid_indices",[0])[0]} for it in items]

BENCH_LOADERS = {
    "BoolQ": load_boolq, "ARC-C": load_arc_c, "TruthfulQA": load_truthfulqa,
    "CSQA": load_csqa, "MMLU-HS-Math": load_mmlu_hs_math,
    "GPQA-Diamond": lambda n: _adapt_hard(load_gpqa_diamond(n)),
    "AGIEval-Gaokao-MathQA": lambda n: _adapt_hard(load_agieval_gaokao_mathqa(n)),
    "LogiQA": load_logiqa_fixed,
}


# ──────────────────────────────────────────────────────────────────────────────
# 从扫参 JSON 提取 sweep_best
# ──────────────────────────────────────────────────────────────────────────────
def load_sweep_best_from_files(preset: dict) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for key in ("sweep_main","sweep_hard"):
        p = preset.get(key)
        if not p or not Path(p).exists(): continue
        sweep = json.loads(Path(p).read_text())
        for b in sweep.get("benchmarks_used",[]):
            if b in out: continue
            best_acc = -1.0
            for row in sweep.get("results",[]):
                if b not in row: continue
                acc = float(row[b])
                if acc > best_acc:
                    best_acc = acc
                    out[b] = (int(row["t_start"]), int(row["t_stop"]))
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

def loglikelihood_mc(logits: torch.Tensor, input_ids: torch.Tensor, prompt_len: int) -> float:
    total = 0.0
    for i in range(prompt_len, input_ids.shape[1]):
        logp = F.log_softmax(logits[0, i-1].float(), dim=-1)
        total += float(logp[input_ids[0, i]])
    return total

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
# Probe forward: 只需 a_l 和 m_l（计算 neg_cos_am）
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def probe_neg_cos_am(
    model,
    input_ids: torch.Tensor,
    attn_mask,
    n_layers: int,
    probe_layers: list[int],
) -> dict[int, float]:
    """高效 probe：只捕获 a_l 和 m_l，计算 -cos(a_l, m_l)（无需额外 MLP forward）"""
    base = model.model
    a_out: dict[int, torch.Tensor] = {}
    m_out: dict[int, torch.Tensor] = {}
    hooks = []
    for li in range(n_layers):
        def _pa(idx):
            def fn(_m, _i, out):
                t = out[0] if isinstance(out, tuple) else out
                a_out[idx] = t[:, -1:, :].detach().clone()
            return fn
        def _pm(idx):
            def fn(_m, _i, out):
                m_out[idx] = out[:, -1:, :].detach().clone()
            return fn
        hooks.append(base.layers[li].self_attn.register_forward_hook(_pa(li)))
        hooks.append(base.layers[li].mlp.register_forward_hook(_pm(li)))
    model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
    for h in hooks: h.remove()
    result: dict[int, float] = {}
    for li in probe_layers:
        al = a_out.get(li)
        ml = m_out.get(li)
        if al is None or ml is None: continue
        result[li] = -safe_cos(al.squeeze(), ml.squeeze())  # neg_cos_am
    return result


def calibrate_neg_cos_am(
    items: list[dict],
    model, tok,
    n_layers: int,
    probe_layers: list[int],
    device: str,
    n_calib: int = 20,
) -> dict[int, float]:
    acc: dict[int, list[float]] = defaultdict(list)
    for item in items[:n_calib]:
        enc = tok(item["prompt"], return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        amask = enc.get("attention_mask")
        if amask is not None: amask = amask.to(device)
        sig = probe_neg_cos_am(model, ids, amask, n_layers, probe_layers)
        for li, v in sig.items(): acc[li].append(v)
    return {li: float(np.mean(vs)) for li, vs in acc.items() if vs}


# ──────────────────────────────────────────────────────────────────────────────
# 改进的窗口选择函数
# ──────────────────────────────────────────────────────────────────────────────
def select_adaptive_nt(
    profile: dict[int, float],
    min_start: int,
    max_start: int,
    threshold_ratio: float = ADAPTIVE_THR,
    max_nt: int = 18,
) -> tuple[int, int]:
    """
    自适应 n_t 选择：
    1. 找 [min_start, max_start] 内 profile 峰值层 t_peak
    2. 从 t_peak 向左扩展，直到信号降到 threshold_ratio * peak 以下 → t_start
    3. 从 t_peak 向右扩展，直到信号降到 threshold_ratio * peak 以下 → t_stop
    4. 限制最大 n_t ≤ max_nt
    """
    valid = {l: v for l, v in profile.items() if min_start <= l <= max_start}
    if not valid:
        return min_start, min_start + 4

    t_peak = max(valid, key=valid.__getitem__)
    peak_val = valid[t_peak]
    thr = threshold_ratio * peak_val

    layers_sorted = sorted(valid.keys())

    # 向左找 t_start
    t_start = t_peak
    for l in reversed(layers_sorted):
        if l >= t_peak: continue
        if valid[l] >= thr:
            t_start = l
        else:
            break

    # 向右找 t_stop
    t_stop = t_peak + 2  # 至少包含 peak 层
    for l in layers_sorted:
        if l <= t_peak: continue
        if valid[l] >= thr:
            t_stop = l + 2  # +2 since probe layers are every 2
        else:
            break

    # 限制 n_t
    n_t = min(t_stop - t_start, max_nt)
    n_t = max(n_t, 2)  # 至少 n_t=2

    return t_start, t_start + n_t


def select_fine_nt(
    profile: dict[int, float],
    nt_candidates: tuple[int, ...],
    min_start: int,
    max_start: int,
) -> tuple[int, int]:
    """细粒度 n_t 搜索（同 select_window_generic 但候选集更细）"""
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


def select_hybrid(
    profile: dict[int, float],
    items_for_emp: list[dict],
    model, tok,
    n_layers: int,
    min_start: int,
    max_start: int,
    nt_candidates: tuple[int, ...],
    device: str,
) -> tuple[int, int]:
    """
    混合选择：
    1. 用信号找 t_start（最高 neg_cos_am 均值窗口中心）
    2. 在 [t_start-3, t_start+3] × nt_candidates 内用 acc 做经验精调
    """
    # Step 1: signal t_start (best single-layer position)
    valid = {l: v for l, v in profile.items() if min_start <= l <= max_start}
    if not valid:
        sig_tstart = min_start
    else:
        sig_tstart = max(valid, key=valid.__getitem__)

    # Step 2: empirical search around sig_tstart
    search_starts = range(max(min_start, sig_tstart-3), min(max_start, sig_tstart+4))
    candidates = [(ts, ts+n_t) for ts in search_starts for n_t in nt_candidates
                  if ts + n_t <= n_layers - 2]

    if not candidates:
        return sig_tstart, sig_tstart + 4

    acc_sum = {c: 0 for c in candidates}
    n_items = len(items_for_emp[:N_CALIB_V2])
    for item in items_for_emp[:N_CALIB_V2]:
        # baseline predict (no ETD)
        label = item["label"]
        enc = tok(item["prompt"], return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        amask = enc.get("attention_mask")
        if amask is not None: amask = amask.to(device)
        plen = tok(item["prompt"], return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]

        for (ts, te) in candidates:
            n_t = te - ts
            try:
                scores = []
                for cont in item["choices"]:
                    full = item["prompt"] + " " + cont
                    enc2 = tok(full, return_tensors="pt", add_special_tokens=False)
                    ids2 = enc2["input_ids"].to(device)
                    amask2 = enc2.get("attention_mask")
                    if amask2 is not None: amask2 = amask2.to(device)
                    lgts = etd_forward_logits(model, ids2, amask2, n_e=ts, n_t=n_t, k=K_ETD,
                                              alpha=min(1.0, 6.0/max(n_t,1)))
                    scores.append(loglikelihood_mc(lgts, ids2, plen))
                pred = int(np.argmax(scores))
                if pred == label: acc_sum[(ts,te)] += 1
            except Exception:
                pass
        torch.cuda.empty_cache()

    best_win = max(acc_sum, key=acc_sum.__getitem__)
    best_acc = acc_sum[best_win] / max(n_items, 1)
    print(f"  [hybrid] sig_tstart={sig_tstart}  best={best_win}  calib_acc={best_acc:.3f}")
    return best_win


def run_empirical_acc(
    items: list[dict],
    model, tok,
    n_layers: int,
    min_start: int,
    max_start: int,
    nt_candidates: tuple[int, ...],
    device: str,
    n_calib: int = N_CALIB_V2,
) -> tuple[int, int] | None:
    """
    经验标定 v2：基于 acc（而非 logit gain），N_CALIB=30。
    同时限制 t_stop ≤ n_layers - min_decoder (min_decoder=5)。
    """
    stride = max(2, (max_start - min_start) // 6)
    candidates: list[tuple[int, int]] = []
    min_decoder = 5
    for n_t in nt_candidates:
        for ts in range(min_start, max_start + 1, stride):
            te = ts + n_t
            if te > n_layers - min_decoder: continue
            candidates.append((ts, te))
    if not candidates:
        return None

    acc_sum = {c: 0 for c in candidates}
    calib_items = items[:n_calib]
    for item in calib_items:
        label = item["label"]
        plen = tok(item["prompt"], return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
        for (ts, te) in candidates:
            n_t = te - ts
            try:
                scores = []
                for cont in item["choices"]:
                    full = item["prompt"] + " " + cont
                    enc = tok(full, return_tensors="pt", add_special_tokens=False)
                    ids = enc["input_ids"].to(device)
                    amask = enc.get("attention_mask")
                    if amask is not None: amask = amask.to(device)
                    lgts = etd_forward_logits(model, ids, amask, n_e=ts, n_t=n_t, k=K_ETD,
                                              alpha=min(1.0, 6.0/max(n_t,1)))
                    scores.append(loglikelihood_mc(lgts, ids, plen))
                pred = int(np.argmax(scores))
                if pred == label: acc_sum[(ts,te)] += 1
            except Exception:
                pass
        torch.cuda.empty_cache()

    # Compare with baseline
    bl_correct = 0
    for item in calib_items:
        label = item["label"]
        plen = tok(item["prompt"], return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
        scores = []
        for cont in item["choices"]:
            full = item["prompt"] + " " + cont
            enc = tok(full, return_tensors="pt", add_special_tokens=False)
            ids = enc["input_ids"].to(device)
            amask = enc.get("attention_mask")
            if amask is not None: amask = amask.to(device)
            lgts = baseline_forward_logits(model, ids, amask)
            scores.append(loglikelihood_mc(lgts, ids, plen))
        if int(np.argmax(scores)) == label: bl_correct += 1
        torch.cuda.empty_cache()

    best_win = max(acc_sum, key=acc_sum.__getitem__)
    best_acc = acc_sum[best_win] / max(len(calib_items), 1)
    bl_acc = bl_correct / max(len(calib_items), 1)

    print(f"  [emp_acc30] 候选={len(candidates)}  best={best_win}  "
          f"calib_acc={best_acc:.3f}  baseline_acc={bl_acc:.3f}")

    # Only use ETD if it improves over baseline
    if best_acc <= bl_acc:
        print(f"  [emp_acc30] ETD 未超 baseline on calib set → fallback baseline")
        return None
    return best_win


# ──────────────────────────────────────────────────────────────────────────────
# 推理辅助
# ──────────────────────────────────────────────────────────────────────────────
def mc_predict(model, tok, item: dict, device: str, n_e=None, n_t=None, k=K_ETD) -> int:
    prompt, choices = item["prompt"], item["choices"]
    plen = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
    scores = []
    for cont in choices:
        enc = tok(prompt + " " + cont, return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        amask = enc.get("attention_mask")
        if amask is not None: amask = amask.to(device)
        if n_e is not None and n_t is not None and n_t > 0:
            lgts = etd_forward_logits(model, ids, amask, n_e=n_e, n_t=n_t, k=k,
                                       alpha=min(1.0, 6.0/max(n_t,1)))
        else:
            lgts = baseline_forward_logits(model, ids, amask)
        scores.append(loglikelihood_mc(lgts, ids, plen))
    return int(np.argmax(scores))


# ──────────────────────────────────────────────────────────────────────────────
# 主评测
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_benchmark(
    bench: str,
    items: list[dict],
    model, tok,
    n_layers: int,
    sweep_win: tuple[int, int],
    preset: dict,
    device: str,
) -> dict:
    probe_layers = preset["probe_layers"]
    min_start    = preset["min_start"]
    max_start    = preset["max_start"]
    nt_fine      = preset["nt_fine"]
    n_total      = len(items)

    # ── 标定 neg_cos_am profile（N=20 for signal, N=30 for empirical）
    print(f"  [标定 neg_cos_am] N=20 …")
    t0c = time.time()
    mean_profile = calibrate_neg_cos_am(items, model, tok, n_layers, probe_layers, device, n_calib=20)
    print(f"  [标定完成] {time.time()-t0c:.1f}s  profile: "
          + " ".join(f"L{l}={v:.2f}" for l,v in sorted(mean_profile.items())))

    # ── 派生各条件固定窗口
    win_adaptive = select_adaptive_nt(mean_profile, min_start, max_start)
    win_fine_nt  = select_fine_nt(mean_profile, nt_fine, min_start, max_start)

    print(f"  [信号窗口] adaptive={win_adaptive}  fine_nt={win_fine_nt}  sweep_best={sweep_win}")

    # ── 经验标定 acc30
    print(f"  [经验标定 v2] N_CALIB={N_CALIB_V2}，acc-based …")
    t_emp = time.time()
    win_emp_acc = run_empirical_acc(
        items, model, tok, n_layers, min_start, max_start, nt_fine, device
    )
    print(f"  [经验标定 v2] {time.time()-t_emp:.1f}s  → {win_emp_acc}")

    # ── 混合方案
    print(f"  [混合方案] signal t_start + emp n_t …")
    t_hyb = time.time()
    win_hybrid = select_hybrid(
        mean_profile, items, model, tok, n_layers,
        min_start, max_start, nt_fine[:4], device,  # 只用前 4 个 n_t 候选加快速度
    )
    print(f"  [混合方案] {time.time()-t_hyb:.1f}s  → {win_hybrid}")

    wins_calib = {
        "neg_cos_am_adaptive": win_adaptive,
        "neg_cos_am_fine_nt":  win_fine_nt,
        "empirical_acc30":     win_emp_acc,
        "hybrid_signal_emp":   win_hybrid,
        "sweep_best":          sweep_win,
    }
    print(f"  [所有窗口] " + "  ".join(f"{k}={v}" for k,v in wins_calib.items()))

    # ── 逐样本评测
    correct = {c: 0 for c in EVAL_COND_NAMES}
    t0 = time.time()

    for i, item in enumerate(items):
        label = item["label"]
        # per-sample signal
        enc = tok(item["prompt"], return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        amask = enc.get("attention_mask")
        if amask is not None: amask = amask.to(device)

        ps_sig = probe_neg_cos_am(model, ids, amask, n_layers, probe_layers)

        # per-sample adaptive window
        ps_adaptive = select_adaptive_nt(ps_sig, min_start, max_start)
        ps_fine_nt  = select_fine_nt(ps_sig, nt_fine, min_start, max_start)

        wins = {
            "baseline":           None,
            "sweep_best":         sweep_win,
            "neg_cos_am_adaptive": ps_adaptive,
            "neg_cos_am_fine_nt":  ps_fine_nt,
            "empirical_acc30":     win_emp_acc,   # calib-fixed
            "hybrid_signal_emp":   win_hybrid,    # calib-fixed
        }

        for cname in EVAL_COND_NAMES:
            win = wins[cname]
            if win is None:
                pred = mc_predict(model, tok, item, device)
            else:
                ts, te = win
                n_t = te - ts
                if n_layers - te < 2 or n_t < 1:
                    pred = mc_predict(model, tok, item, device)
                else:
                    pred = mc_predict(model, tok, item, device, n_e=ts, n_t=n_t)
            if pred == label: correct[cname] += 1

        if (i+1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed/(i+1)*(n_total-i-1)
            line = f"  [{i+1:3d}/{n_total}] "
            for cn in EVAL_COND_NAMES:
                line += f"{cn[:4]}={correct[cn]/(i+1):.3f} "
            line += f"| {elapsed:.0f}s ETA {eta:.0f}s"
            print(line)
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    accs = {c: correct[c]/n_total for c in EVAL_COND_NAMES}

    return {
        "benchmark": bench,
        "n": n_total,
        "elapsed_s": elapsed,
        "accuracies": accs,
        "sweep_best_window": list(sweep_win),
        "win_adaptive": list(win_adaptive),
        "win_fine_nt":  list(win_fine_nt),
        "win_emp_acc":  list(win_emp_acc) if win_emp_acc else None,
        "win_hybrid":   list(win_hybrid),
        "mean_profile": {str(k): v for k, v in sorted(mean_profile.items())},
    }


# ──────────────────────────────────────────────────────────────────────────────
# 可视化
# ──────────────────────────────────────────────────────────────────────────────
def plot_main(all_results: dict, fig_dir: Path, preset_name: str):
    """主精度对比条形图（含 R39A 最佳条件对比）"""
    fig_dir.mkdir(parents=True, exist_ok=True)
    benches = [b for b in BENCH_ORDER if b in all_results]

    # 尝试载入 R39A 数据
    r39a_emp = {}
    r39a_json = R39A_BEST.get(preset_name)
    if r39a_json and Path(r39a_json).exists():
        try:
            r39a_data = json.loads(Path(r39a_json).read_text())
            for b, v in r39a_data.get("results",{}).items():
                r39a_emp[b] = v["accuracies"].get("empirical_calib", 0)
        except Exception:
            pass

    n_b = len(benches)
    conds_show = EVAL_COND_NAMES
    n_c = len(conds_show) + (1 if r39a_emp else 0)
    fig, ax = plt.subplots(figsize=(max(14, n_b*2), 5))
    x = np.arange(n_b)
    w = 0.9 / n_c
    for j, cn in enumerate(conds_show):
        offs = (j - (n_c-1)/2.0) * w
        vals = [all_results[b]["accuracies"].get(cn, 0) for b in benches]
        ax.bar(x+offs, vals, w*0.9, label=COND_LABELS[cn], color=COND_COLORS[cn])
    if r39a_emp:
        offs = ((len(conds_show)) - (n_c-1)/2.0) * w
        vals = [r39a_emp.get(b, 0) for b in benches]
        ax.bar(x+offs, vals, w*0.9, label="R39A empirical (v1)", color="#bcbd22", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([b[:14] for b in benches], rotation=22, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"R39B {preset_name}：改进信号引导 ETD vs Baseline vs Sweep Best")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    p = fig_dir / "main_accuracy_bars.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  → {p}")


def plot_delta_scatter(all_results: dict, fig_dir: Path, preset_name: str):
    benches = [b for b in BENCH_ORDER if b in all_results]
    conds = [c for c in EVAL_COND_NAMES if c not in ("baseline","sweep_best")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, ref in zip(axes, ["baseline","sweep_best"]):
        for cn in conds:
            delta = [all_results[b]["accuracies"].get(cn,0) - all_results[b]["accuracies"].get(ref,0)
                     for b in benches]
            x = np.arange(len(benches))
            ax.scatter(x, delta, label=COND_LABELS[cn], color=COND_COLORS[cn], s=70, zorder=3)
            ax.plot(x, delta, color=COND_COLORS[cn], alpha=0.4, linewidth=1)
        ax.axhline(0, color="black", linewidth=1.2, linestyle="--")
        ax.set_xticks(np.arange(len(benches)))
        ax.set_xticklabels([b[:12] for b in benches], rotation=25, ha="right", fontsize=7.5)
        ax.set_ylabel(f"Δacc vs {ref}")
        ax.set_title(f"相对 {ref} 的增量")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
    plt.suptitle(f"R39B {preset_name}：Δacc 散点图", fontsize=11)
    plt.tight_layout()
    p = fig_dir / "delta_acc_scatter.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  → {p}")


def plot_neg_cos_am_profiles(all_results: dict, fig_dir: Path, preset_name: str):
    benches = [b for b in BENCH_ORDER if b in all_results]
    n_b = len(benches)
    n_cols = 4
    n_rows = (n_b + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5*n_rows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for idx, b in enumerate(benches):
        ax = axes_flat[idx]
        res = all_results[b]
        prof = {int(k): v for k, v in res.get("mean_profile", {}).items()}
        if not prof:
            ax.set_title(b[:16]); continue
        layers = sorted(prof.keys())
        vals = [prof[l] for l in layers]
        ax.plot(layers, vals, "r-o", markersize=4, linewidth=1.5, label="neg_cos_am")
        sw = res["sweep_best_window"]
        ax.axvspan(sw[0], sw[1], alpha=0.15, color="blue", label="sweep_best")
        # Mark selected windows
        for win_key, col in [("win_adaptive","red"),("win_fine_nt","orange"),("win_hybrid","purple")]:
            w = res.get(win_key)
            if w: ax.axvspan(w[0], w[1], alpha=0.08, color=col)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_title(f"{b[:16]}\nsw={sw}", fontsize=8)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=7)
    for idx in range(len(benches), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    plt.suptitle(f"R39B {preset_name}：neg_cos_am profile\n（蓝=sweep_best，红=adaptive，橙=fine_nt，紫=hybrid）", fontsize=10)
    plt.tight_layout()
    p = fig_dir / "neg_cos_am_profiles.png"
    plt.savefig(p, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  → {p}")


def plot_summary_final(all_results: dict, fig_dir: Path, preset_name: str):
    benches = [b for b in BENCH_ORDER if b in all_results]
    macro = {c: np.mean([all_results[b]["accuracies"].get(c,0) for b in benches])
             for c in EVAL_COND_NAMES}
    base_macro = macro["baseline"]
    sw_macro   = macro["sweep_best"]
    delta_macro = {c: macro[c]-base_macro for c in EVAL_COND_NAMES if c != "baseline"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 左：宏平均对比
    conds = [c for c in EVAL_COND_NAMES if c != "baseline"]
    x = np.arange(len(conds))
    vals = [delta_macro[c] for c in conds]
    colors = [COND_COLORS[c] for c in conds]
    bars = ax1.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax1.axhline(0, color="black", linewidth=1.0)
    ax1.axhline(sw_macro-base_macro, color="#1f77b4", linewidth=2.0, linestyle="--", label="sweep_best level")
    ax1.set_xticks(x)
    ax1.set_xticklabels([COND_LABELS[c] for c in conds], rotation=28, ha="right", fontsize=7.5)
    ax1.set_ylabel("Δacc vs Baseline")
    ax1.set_title(f"R39B {preset_name}\n宏平均 Δacc（8 benchmarks）")
    ax1.legend(fontsize=8)
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                 f"{val:+.3f}", ha="center", va="bottom", fontsize=8.5)

    # 右：每个 benchmark 的 best method highlight
    beat_sweep = []
    for b in benches:
        a = all_results[b]["accuracies"]
        sw = a.get("sweep_best", 0)
        for cn in ["neg_cos_am_adaptive","neg_cos_am_fine_nt","empirical_acc30","hybrid_signal_emp"]:
            if a.get(cn, 0) > sw:
                beat_sweep.append(b)
                break

    y = np.arange(len(benches))
    for cn, col in [(c, COND_COLORS[c]) for c in EVAL_COND_NAMES if c not in ("baseline",)]:
        vals2 = [all_results[b]["accuracies"].get(cn,0) for b in benches]
        ax2.plot(vals2, y, "o-", color=col, label=COND_LABELS[cn], linewidth=1.2, markersize=5)
    ax2.set_yticks(y)
    ax2.set_yticklabels([f"{'★' if b in beat_sweep else ' '} {b[:18]}" for b in benches], fontsize=8)
    ax2.set_xlabel("Accuracy")
    ax2.set_title(f"各 benchmark 精度（★=有方法超 sweep_best）")
    ax2.legend(fontsize=7, loc="lower right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    p = fig_dir / "summary_final.png"
    plt.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  → {p}")


# ──────────────────────────────────────────────────────────────────────────────
# 打印结果表
# ──────────────────────────────────────────────────────────────────────────────
def print_results_table(all_results: dict, preset_name: str):
    benches = [b for b in BENCH_ORDER if b in all_results]
    conds = EVAL_COND_NAMES
    print(f"\n{'='*90}")
    print(f"R39B {preset_name} 精度结果")
    print(f"{'='*90}")
    header = f"{'Benchmark':25s}" + "".join(f"{c[:10]:>12s}" for c in conds)
    print(header)
    print("-"*90)
    for b in benches:
        a = all_results[b]["accuracies"]
        sw = a.get("sweep_best", 0)
        bl = a.get("baseline", 0)
        row = f"{b[:24]:25s}"
        for c in conds:
            v = a.get(c, 0)
            if c not in ("baseline","sweep_best"):
                if v > sw:   mark = "★"
                elif v > bl: mark = "+"
                else:        mark = " "
            else:
                mark = " "
            row += f"{v:.3f}{mark}{'':7}"
        print(row)
    print("-"*90)
    macro = {c: np.mean([all_results[b]["accuracies"].get(c,0) for b in benches]) for c in conds}
    sw_m = macro["sweep_best"]
    bl_m = macro["baseline"]
    row = f"{'MACRO AVG':25s}"
    for c in conds:
        v = macro[c]
        if c not in ("baseline","sweep_best"):
            if v > sw_m:   mark = "★"
            elif v > bl_m: mark = "+"
            else:          mark = " "
        else:
            mark = " "
        row += f"{v:.3f}{mark}{'':7}"
    print(row)
    print(f"\n(★=超越 sweep_best  +=超越 baseline)")

    # Window summary
    print(f"\n窗口选择摘要：")
    for b in benches:
        v = all_results[b]
        print(f"  {b[:22]:22} sw={str(v['sweep_best_window']):12} adpt={str(v['win_adaptive']):12} "
              f"fine={str(v['win_fine_nt']):12} emp={str(v['win_emp_acc']):12} hyb={str(v['win_hybrid'])}")


# ──────────────────────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS.keys()), required=True)
    args = ap.parse_args()

    preset  = PRESETS[args.preset]
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    fig_dir = Path(preset["out_fig"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(preset["out_json"])
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # 加载 sweep_best
    if args.preset == "qwen3-8b":
        sweep_best = dict(preset["sweep_best"])
    else:
        sweep_best = load_sweep_best_from_files(preset)
    print(f"\n[sweep_best] {sweep_best}")

    tok, model = load_model(preset["model_path"])
    n_layers = preset["n_layers"]
    all_results: dict[str, dict] = {}

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
        # 统一格式
        items = []
        for ex in raw[:n]:
            if "label" in ex:
                items.append(ex)
            else:
                gold = str(ex["answer"]).strip().lower()
                label = next(i for i,c in enumerate(ex["choices"]) if str(c).strip().lower()==gold)
                items.append({"prompt":ex["prompt"],"choices":ex["choices"],"label":label})

        try:
            res = evaluate_benchmark(bench, items, model, tok, n_layers, sw, preset, device)
            all_results[bench] = res
        except Exception as e:
            import traceback
            print(f"  [ERROR] evaluate {bench}: {e}")
            traceback.print_exc()
            continue

        out_payload = {
            "preset": args.preset,
            "arch":   preset["arch"],
            "n_layers": n_layers,
            "results": all_results,
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, indent=2)
        print(f"  checkpoint → {out_json}")

    if all_results:
        print_results_table(all_results, args.preset)
        plot_main(all_results, fig_dir, args.preset)
        plot_delta_scatter(all_results, fig_dir, args.preset)
        plot_neg_cos_am_profiles(all_results, fig_dir, args.preset)
        plot_summary_final(all_results, fig_dir, args.preset)

    print(f"\n完成。结果: {out_json}")


if __name__ == "__main__":
    main()
