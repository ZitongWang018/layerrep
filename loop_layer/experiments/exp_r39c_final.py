#!/usr/bin/env python3
"""
R39C 最终信号实验：基于 R39A/B 分析确定的最优方案，三架构全 8 benchmark 对比。

R39A/B 关键结论：
  - neg_cos_am (-cos(a_l,m_l)) 是 coverage=1.00 的最强信号（远超其他候选）
  - Logit-gain 比 acc-based 经验标定效果更好（R39A emp在BoolQ达0.90，R39B降至0.85）
  - GPQA 失败根因：经验标定选出 t_stop 超过安全上限的晚层窗口
  - GPQA 需要 n_t=2 的极窄窗口，原 {4,6,8} 候选集无法覆盖

本轮改进（三个方法 vs baseline vs sweep_best）：
  1. neg_cos_am_calib  标定阶段（20样本）推导 neg_cos_am 均值 profile → 全局固定窗口
                      优点：稳定，不受逐样本噪声影响；对 MMLU/GPQA 最优
  2. emp_logit_fixed   经验标定（logit-gain，N=20）+ 两项修复：
                        - min_decoder=8（禁止 t_stop > n_layers-8，GPQA 修复）
                        - n_t 候选含 {2,4,6,8,12}（加入 n_t=2 for GPQA）
  3. neg_cos_am_ps_nt  逐样本 neg_cos_am 信号 + 细粒度 n_t 候选 {2,4,6,8,12}

最终综合可视化：
  - 三模型 × 8 benchmark 精度汇总热力图
  - 各方法宏平均 Δacc 对比（vs baseline 和 sweep_best）
  - neg_cos_am profile 折线图（含窗口标注）
  - Δacc scatter（每 benchmark 每方法）

用法：
  python experiments/exp_r39c_final.py --preset qwen3-8b
  python experiments/exp_r39c_final.py --preset llama3-8b
  python experiments/exp_r39c_final.py --preset gemma2-2b
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

ROOT    = Path("/root/autodl-tmp/loop_layer")
EXP     = ROOT / "experiments"
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
N_CALIB      = 20
K_ETD        = 2
MIN_DECODER  = 8   # 修复 GPQA 晚层问题：t_stop ≤ n_layers - MIN_DECODER

N_SAMPLES = {
    "BoolQ": 100, "ARC-C": 100, "TruthfulQA": 50, "CSQA": 100,
    "MMLU-HS-Math": 100, "GPQA-Diamond": 100, "AGIEval-Gaokao-MathQA": 100, "LogiQA": 100,
}
BENCH_ORDER = ["BoolQ","ARC-C","TruthfulQA","CSQA","MMLU-HS-Math",
               "GPQA-Diamond","AGIEval-Gaokao-MathQA","LogiQA"]

QWEN_SWEEP_BEST: dict[str, tuple[int, int]] = {
    "BoolQ": (8,22), "ARC-C": (14,20), "TruthfulQA": (16,19), "CSQA": (10,22),
    "MMLU-HS-Math": (10,18), "GPQA-Diamond": (18,20), "AGIEval-Gaokao-MathQA": (13,20),
    "LogiQA": (14,19),
}

PRESETS: dict[str, dict] = {
    "qwen3-8b": {
        "model_path":  "/root/autodl-tmp/model_qwen",
        "arch":        "qwen3",
        "n_layers":    36,
        "probe_layers": list(range(6, 33, 2)),   # L6..L32
        "min_start":   9,
        "max_start":   26,
        "nt_candidates": (2, 4, 6, 8, 12),       # 含 n_t=2 for GPQA
        "calib_nt":    8,
        "sweep_best":  QWEN_SWEEP_BEST,
        "out_json":    EXP / "results/r39c_final_qwen3.json",
        "out_fig":     EXP / "figures/r39c_final_qwen3",
    },
    "llama3-8b": {
        "model_path":  "/root/autodl-tmp/Llama3-8B",
        "arch":        "llama",
        "n_layers":    32,
        "probe_layers": list(range(6, 27, 2)),
        "min_start":   8,
        "max_start":   20,
        "nt_candidates": (2, 4, 6, 8),
        "calib_nt":    6,
        "sweep_main":  EXP / "llama3-8b/results/etd_layer_sweep_r30style.json",
        "sweep_hard":  EXP / "llama3-8b/results/hard_mc/etd_layer_sweep_r30style.json",
        "out_json":    EXP / "results/r39c_final_llama3.json",
        "out_fig":     EXP / "figures/r39c_final_llama3",
    },
    "gemma2-2b": {
        "model_path":  "/root/autodl-tmp/Gemma2-2B",
        "arch":        "gemma2",
        "n_layers":    26,
        "probe_layers": list(range(4, 23, 2)),
        "min_start":   5,
        "max_start":   16,
        "nt_candidates": (2, 4, 8, 14),
        "calib_nt":    8,
        "sweep_main":  EXP / "gemma2-2b/results/etd_layer_sweep_r30style.json",
        "sweep_hard":  EXP / "gemma2-2b/results/hard_mc/etd_layer_sweep_r30style.json",
        "out_json":    EXP / "results/r39c_final_gemma2.json",
        "out_fig":     EXP / "figures/r39c_final_gemma2",
    },
}

EVAL_COND_NAMES = ["baseline", "sweep_best",
                   "neg_cos_am_calib", "emp_logit_fixed", "neg_cos_am_ps_nt"]

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
    "neg_cos_am_calib":  "neg_cos_am Calib Fixed",
    "emp_logit_fixed":   "Empirical Logit (fixed, n_t≥2)",
    "neg_cos_am_ps_nt":  "neg_cos_am Per-Sample n_t{2..12}",
}

# Publication figures: omit emp_logit_fixed (green bars) while JSON/eval still stores it.
FIGURE_COND_NAMES = [c for c in EVAL_COND_NAMES if c != "emp_logit_fixed"]
FIGURE_SIGNAL_CONDS = [c for c in FIGURE_COND_NAMES if c not in ("baseline", "sweep_best")]

PRESET_DISPLAY = {"qwen3-8b": "Qwen3-8B", "llama3-8b": "Llama3-8B", "gemma2-2b": "Gemma2-2B"}


# ──────────────────────────────────────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────────────────────────────────────
def _fmt(p, c, l): return {"prompt": p, "choices":[x.strip() for x in c], "label": l}

def load_boolq(n):
    ds = load_dataset("aps/super_glue","boolq")["validation"]
    out=[]
    for r in ds:
        if len(out)>=n: break
        lab=int(r["label"])
        if lab<0: continue
        out.append(_fmt(f"{r['passage']}\nQuestion: {r['question']}?\nAnswer:",["no","yes"],lab))
    return out

def load_arc_c(n):
    ds=load_dataset("allenai/ai2_arc","ARC-Challenge")["test"]
    out=[]
    for r in ds:
        if len(out)>=n: break
        key=r["answerKey"]
        label=ord(key)-ord("A") if key in "ABCD" else int(key)-1
        out.append(_fmt(f"Question: {r['question'].strip()}\nAnswer:",r["choices"]["text"],label))
    return out

def load_csqa(n):
    ds=load_dataset("tau/commonsense_qa")["validation"]
    lmap={"A":0,"B":1,"C":2,"D":3,"E":4}
    out=[]
    for r in ds:
        if len(out)>=n: break
        key=r["answerKey"]
        if key not in lmap: continue
        out.append(_fmt(f"Question: {r['question'].strip()}\nAnswer:",r["choices"]["text"],lmap[key]))
    return out

def load_truthfulqa(n):
    ds=load_dataset("truthfulqa/truthful_qa","multiple_choice")["validation"]
    out=[]
    for r in ds:
        if len(out)>=n: break
        out.append(_fmt(f"Question: {r['question']}\nAnswer:",r["mc1_targets"]["choices"],
                        int(np.argmax(r["mc1_targets"]["labels"]))))
    return out

def load_mmlu(n):
    dc=DownloadConfig(local_files_only=True)
    ds=load_dataset("cais/mmlu","high_school_mathematics",download_config=dc)["test"]
    out=[]
    for r in ds:
        if len(out)>=n: break
        out.append(_fmt(f"Question: {r['question'].strip()}\nAnswer:",[str(c) for c in r["choices"]],int(r["answer"])))
    return out

def load_logiqa(n):
    def _s(o): return re.sub(r"^[ABCDabcd]\.\s*","",str(o).strip())
    def _l(r):
        lab=r.get("label") if r.get("label") is not None else r.get("answer")
        if isinstance(lab,(int,float)) and lab==int(lab):
            i=int(lab)
            if 0<=i<4: return "abcd"[i]
        s=str(lab).strip().lower()
        return s if s in "abcd" else None
    ds=load_dataset("fireworks-ai/logiqa",split="test")
    out=[];choices=["a","b","c","d"]
    for r in ds:
        label=_l(r)
        if label is None: continue
        opts=r["options"]
        if hasattr(opts,"tolist"): opts=opts.tolist()
        prompt=(f"Passage: {r['context']}\nQuestion: {r['question']}\nChoices:\n"
                +"\n".join(f"{x.upper()}. {_s(o)}" for x,o in zip(choices,opts))+"\nAnswer:")
        out.append({"prompt":prompt,"choices":choices,"label":choices.index(label)})
        if len(out)>=n: break
    return out

def _adapt(items):
    return [{"prompt":it["prompt"],"choices":it["choices"],"label":it.get("valid_indices",[0])[0]} for it in items]

BENCH_LOADERS = {
    "BoolQ": load_boolq, "ARC-C": load_arc_c, "TruthfulQA": load_truthfulqa,
    "CSQA": load_csqa, "MMLU-HS-Math": load_mmlu,
    "GPQA-Diamond": lambda n: _adapt(load_gpqa_diamond(n)),
    "AGIEval-Gaokao-MathQA": lambda n: _adapt(load_agieval_gaokao_mathqa(n)),
    "LogiQA": load_logiqa,
}

def load_sweep_best_from_files(preset):
    out = {}
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
def safe_cos(u, v) -> float:
    u = u.float().reshape(-1).cpu(); v = v.float().reshape(-1).cpu()
    nu, nv = u.norm(), v.norm()
    if nu < 1e-12 or nv < 1e-12: return 0.0
    return float((u @ v / (nu * nv)).clamp(-1, 1))

def loglikelihood_mc(logits, input_ids, prompt_len) -> float:
    total = 0.0
    for i in range(prompt_len, input_ids.shape[1]):
        logp = F.log_softmax(logits[0, i-1].float(), dim=-1)
        total += float(logp[input_ids[0, i]])
    return total

def load_model(path):
    print(f"Loading model: {path}")
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager", trust_remote_code=True,
    )
    mdl.eval()
    return tok, mdl


# ──────────────────────────────────────────────────────────────────────────────
# Probe forward：高效单次捕获 a_l、m_l，计算 neg_cos_am
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def probe_neg_cos_am(model, input_ids, attn_mask, n_layers, probe_layers) -> dict[int, float]:
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
        al = a_out.get(li); ml = m_out.get(li)
        if al is None or ml is None: continue
        result[li] = -safe_cos(al.squeeze(), ml.squeeze())  # neg_cos_am（越高=竞争越强）
    return result


def calibrate(items, model, tok, n_layers, probe_layers, device) -> dict[int, float]:
    acc: dict[int, list] = defaultdict(list)
    for item in items[:N_CALIB]:
        enc = tok(item["prompt"], return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        amask = enc.get("attention_mask")
        if amask is not None: amask = amask.to(device)
        sig = probe_neg_cos_am(model, ids, amask, n_layers, probe_layers)
        for li, v in sig.items(): acc[li].append(v)
    return {li: float(np.mean(vs)) for li, vs in acc.items() if vs}


# ──────────────────────────────────────────────────────────────────────────────
# 窗口选择
# ──────────────────────────────────────────────────────────────────────────────
def select_calib_global(profile, n_t, min_start, max_start) -> tuple[int,int]:
    best_start, best_score = min_start, -999.0
    for start in range(min_start, max_start+1):
        vals = [profile[l] for l in profile if start <= l < start+n_t]
        if not vals: continue
        score = float(np.mean(vals))
        if score > best_score:
            best_score, best_start = score, start
    return best_start, best_start + n_t

def select_persample(profile, nt_candidates, min_start, max_start) -> tuple[int,int]:
    best_start, best_nt, best_score = min_start, nt_candidates[0], -999.0
    for n_t in nt_candidates:
        for start in range(min_start, max_start+1):
            vals = [profile[l] for l in profile if start <= l < start+n_t]
            if not vals: continue
            score = float(np.mean(vals))
            if score > best_score:
                best_score, best_start, best_nt = score, start, n_t
    return best_start, best_start + best_nt


# ──────────────────────────────────────────────────────────────────────────────
# 经验标定（logit-gain，含 min_decoder 约束）
# ──────────────────────────────────────────────────────────────────────────────
def run_empirical_logit(items, model, tok, n_layers, min_start, max_start,
                        nt_candidates, device) -> tuple[int,int] | None:
    max_t_stop = n_layers - MIN_DECODER
    stride = max(2, (max_start - min_start) // 5)
    candidates = []
    for n_t in nt_candidates:
        for ts in range(min_start, max_start+1, stride):
            te = ts + n_t
            if te > max_t_stop: continue
            candidates.append((ts, te))
    if not candidates:
        return None

    gain_sum = {c: 0.0 for c in candidates}
    calib_items = items[:N_CALIB]
    for item in calib_items:
        plen = tok(item["prompt"], return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
        label = item["label"]
        cont = item["choices"][label]
        full = item["prompt"] + " " + cont
        enc = tok(full, return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        amask = enc.get("attention_mask")
        if amask is not None: amask = amask.to(device)
        # baseline logit once per sample
        lgts_base = baseline_forward_logits(model, ids, amask)
        g_base = loglikelihood_mc(lgts_base, ids, plen)
        for (ts, te) in candidates:
            n_t = te - ts
            try:
                lgts_etd = etd_forward_logits(model, ids, amask, n_e=ts, n_t=n_t, k=K_ETD,
                                               alpha=min(1.0, 6.0/max(n_t,1)))
                gain_sum[(ts,te)] += loglikelihood_mc(lgts_etd, ids, plen) - g_base
            except Exception:
                pass
        torch.cuda.empty_cache()

    n = max(len(calib_items), 1)
    mean_gain = {c: g/n for c, g in gain_sum.items()}
    best_win, best_gain = max(mean_gain.items(), key=lambda x: x[1])
    print(f"  [emp_logit] candidates={len(candidates)}  best={best_win}  gain={best_gain:+.4f}")
    if best_gain <= 0:
        print(f"  [emp_logit] gain≤0, fallback baseline")
        return None
    return best_win


# ──────────────────────────────────────────────────────────────────────────────
# 推理
# ──────────────────────────────────────────────────────────────────────────────
def mc_predict(model, tok, item, device, n_e=None, n_t=None) -> int:
    prompt, choices = item["prompt"], item["choices"]
    plen = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
    scores = []
    for cont in choices:
        enc = tok(prompt+" "+cont, return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        amask = enc.get("attention_mask")
        if amask is not None: amask = amask.to(device)
        if n_e is not None and n_t is not None and n_t > 0:
            lgts = etd_forward_logits(model, ids, amask, n_e=n_e, n_t=n_t, k=K_ETD,
                                       alpha=min(1.0, 6.0/max(n_t,1)))
        else:
            lgts = baseline_forward_logits(model, ids, amask)
        scores.append(loglikelihood_mc(lgts, ids, plen))
    return int(np.argmax(scores))


# ──────────────────────────────────────────────────────────────────────────────
# 主评测
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_benchmark(bench, items, model, tok, n_layers, sweep_win, preset, device) -> dict:
    probe_layers = preset["probe_layers"]
    min_start    = preset["min_start"]
    max_start    = preset["max_start"]
    nt_cands     = preset["nt_candidates"]
    calib_nt     = preset["calib_nt"]
    n_total      = len(items)

    # Phase A: 标定 neg_cos_am profile
    print(f"  [Phase A] 标定 neg_cos_am (N={N_CALIB}) …")
    t0 = time.time()
    mean_profile = calibrate(items, model, tok, n_layers, probe_layers, device)
    win_calib = select_calib_global(mean_profile, calib_nt, min_start, max_start)

    # Profile 摘要
    if mean_profile:
        ml, mv = max(mean_profile.items(), key=lambda x: x[1])
        win_vals = [v for l,v in mean_profile.items() if sweep_win[0]<=l<sweep_win[1]]
        out_vals = [v for l,v in mean_profile.items() if not(sweep_win[0]<=l<sweep_win[1]) and min_start<=l<=max_start]
        disc = np.mean(win_vals) - np.mean(out_vals) if win_vals and out_vals else 0.0
        print(f"  [Phase A] {time.time()-t0:.1f}s  peak=L{ml}({mv:.3f})  "
              f"disc_vs_sweep={disc:+.3f}  calib_win={win_calib}")

    # Phase B: 经验标定（logit-gain + min_decoder 约束）
    print(f"  [Phase B] 经验标定 logit-gain (N={N_CALIB}) …")
    t0b = time.time()
    win_emp = run_empirical_logit(items, model, tok, n_layers, min_start, max_start, nt_cands, device)
    print(f"  [Phase B] {time.time()-t0b:.1f}s  win_emp={win_emp}  sweep={sweep_win}")

    # Phase C: 评测
    correct = {c: 0 for c in EVAL_COND_NAMES}
    sel_ts  = {"neg_cos_am_ps_nt": []}
    t_eval = time.time()

    for i, item in enumerate(items):
        label = item["label"]
        enc = tok(item["prompt"], return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        amask = enc.get("attention_mask")
        if amask is not None: amask = amask.to(device)

        # per-sample signal
        ps_sig = probe_neg_cos_am(model, ids, amask, n_layers, probe_layers)
        ps_win = select_persample(ps_sig, nt_cands, min_start, max_start)
        sel_ts["neg_cos_am_ps_nt"].append(ps_win[0])

        wins = {
            "baseline":          None,
            "sweep_best":        sweep_win,
            "neg_cos_am_calib":  win_calib,
            "emp_logit_fixed":   win_emp,
            "neg_cos_am_ps_nt":  ps_win,
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
            el = time.time()-t_eval
            eta = el/(i+1)*(n_total-i-1)
            line = f"  [{i+1:3d}/{n_total}] "
            for cn in EVAL_COND_NAMES:
                line += f"{cn[:4]}={correct[cn]/(i+1):.3f} "
            line += f"| {el:.0f}s ETA {eta:.0f}s"
            print(line)
        torch.cuda.empty_cache()

    elapsed = time.time() - t_eval
    accs = {c: correct[c]/n_total for c in EVAL_COND_NAMES}
    return {
        "benchmark": bench,
        "n": n_total,
        "elapsed_s": elapsed,
        "accuracies": accs,
        "sweep_best_window":   list(sweep_win),
        "neg_cos_am_calib_win": list(win_calib),
        "emp_logit_win":       list(win_emp) if win_emp else None,
        "ps_tstart_mean":  float(np.mean(sel_ts["neg_cos_am_ps_nt"])) if sel_ts["neg_cos_am_ps_nt"] else 0.0,
        "ps_tstart_std":   float(np.std(sel_ts["neg_cos_am_ps_nt"]))  if sel_ts["neg_cos_am_ps_nt"] else 0.0,
        "mean_profile": {str(k): v for k,v in sorted(mean_profile.items())},
    }


# ──────────────────────────────────────────────────────────────────────────────
# 可视化（6 种图）
# ──────────────────────────────────────────────────────────────────────────────
def plot_main_bars(all_results, fig_dir, preset_name):
    benches = [b for b in BENCH_ORDER if b in all_results]
    conds = FIGURE_COND_NAMES
    n_b, n_c = len(benches), len(conds)
    fig, ax = plt.subplots(figsize=(max(14, n_b*2), 5))
    x = np.arange(n_b)
    w = 0.88 / n_c
    for j, cn in enumerate(conds):
        offs = (j - (n_c-1)/2.0) * w
        vals = [all_results[b]["accuracies"].get(cn, 0) for b in benches]
        ax.bar(x+offs, vals, w*0.92, label=COND_LABELS[cn], color=COND_COLORS[cn],
               edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([b[:14] for b in benches], rotation=22, ha="right", fontsize=8.5)
    ax.set_ylabel("Accuracy", fontsize=10)
    disp = PRESET_DISPLAY.get(preset_name, preset_name)
    ax.set_title(
        f"R39C {disp}: accuracy by method (ETD signal-guided; emp_logit omitted in figure)",
        fontsize=11,
    )
    ax.legend(fontsize=7.5, ncol=2, loc="upper right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = fig_dir / "01_accuracy_bars.png"
    plt.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  → {p}")


def plot_delta_scatter(all_results, fig_dir, preset_name):
    benches = [b for b in BENCH_ORDER if b in all_results]
    signal_conds = list(FIGURE_SIGNAL_CONDS)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    disp = PRESET_DISPLAY.get(preset_name, preset_name)
    for ax, ref in zip(axes, ["baseline", "sweep_best"]):
        for cn in signal_conds:
            delta = [all_results[b]["accuracies"].get(cn,0) - all_results[b]["accuracies"].get(ref,0)
                     for b in benches]
            x = np.arange(len(benches))
            ax.scatter(x, delta, label=COND_LABELS[cn], color=COND_COLORS[cn],
                       s=80, zorder=3, marker="o")
            ax.plot(x, delta, color=COND_COLORS[cn], alpha=0.5, linewidth=1.2)
        ax.axhline(0, color="black", linewidth=1.5, linestyle="--")
        ax.set_xticks(np.arange(len(benches)))
        ax.set_xticklabels([b[:12] for b in benches], rotation=28, ha="right", fontsize=7.5)
        ax.set_ylabel(r"$\Delta$ accuracy vs " + ("baseline" if ref == "baseline" else "sweep best"), fontsize=9)
        ax.set_title(f"Δ accuracy vs {ref}", fontsize=10)
        ax.legend(fontsize=7.5, loc="lower right")
        ax.grid(True, alpha=0.3)
    plt.suptitle(f"R39C {disp}: Δ accuracy by benchmark (emp_logit omitted)", fontsize=11)
    plt.tight_layout()
    p = fig_dir / "02_delta_scatter.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  → {p}")


def plot_profiles(all_results, fig_dir, preset_name):
    benches = [b for b in BENCH_ORDER if b in all_results]
    n_b = len(benches)
    n_cols = 4; n_rows = (n_b+n_cols-1)//n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5*n_rows))
    axes_flat = axes.flatten() if hasattr(axes,"flatten") else [axes]
    for idx, b in enumerate(benches):
        ax = axes_flat[idx]
        res = all_results[b]
        prof = {int(k): v for k,v in res.get("mean_profile",{}).items()}
        if not prof:
            ax.set_title(b[:16]); continue
        layers = sorted(prof.keys())
        vals = [prof[l] for l in layers]
        ax.plot(layers, vals, "r-o", markersize=4, linewidth=1.5, label="neg_cos_am (calib. mean)")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        sw = res["sweep_best_window"]
        ax.axvspan(sw[0], sw[1], alpha=0.18, color="#1f77b4", label=f"Sweep-best window [{sw[0]}, {sw[1]})")
        cw = res.get("neg_cos_am_calib_win")
        if cw:
            ax.axvspan(cw[0], cw[1], alpha=0.15, color="#9467bd", label=f"Calib window [{cw[0]}, {cw[1]})")
        ax.set_title(f"{b[:16]}  ps_t_start={res.get('ps_tstart_mean',0):.1f}±{res.get('ps_tstart_std',0):.1f}", fontsize=8)
        ax.set_xlabel("Layer", fontsize=7)
        ax.set_ylabel("neg_cos_am", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, loc="best")
    for idx in range(len(benches), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    disp = PRESET_DISPLAY.get(preset_name, preset_name)
    plt.suptitle(f"R39C {disp}: neg_cos_am layer profile (calibration mean) vs. windows", fontsize=10)
    plt.tight_layout()
    p = fig_dir / "03_neg_cos_am_profiles.png"
    plt.savefig(p, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  → {p}")


def plot_summary_heatmap(all_results, fig_dir, preset_name):
    """Δacc 热力图：行=方法，列=benchmark"""
    benches = [b for b in BENCH_ORDER if b in all_results]
    conds = list(FIGURE_SIGNAL_CONDS)
    mat_vs_base = np.zeros((len(conds), len(benches)))
    mat_vs_sweep = np.zeros((len(conds), len(benches)))
    for bi, b in enumerate(benches):
        a = all_results[b]["accuracies"]
        for ci, cn in enumerate(conds):
            mat_vs_base[ci, bi]  = a.get(cn,0) - a.get("baseline",0)
            mat_vs_sweep[ci, bi] = a.get(cn,0) - a.get("sweep_best",0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6))
    disp = PRESET_DISPLAY.get(preset_name, preset_name)
    for ax, mat, title in [
        (ax1, mat_vs_base, "Δ accuracy vs baseline (RdYlGn: higher is better)"),
        (ax2, mat_vs_sweep, "Δ accuracy vs sweep best (RdYlGn: higher is better)"),
    ]:
        vmax = max(0.05, np.abs(mat).max())
        im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(benches)))
        ax.set_xticklabels([b[:14] for b in benches], rotation=25, ha="right", fontsize=8)
        ax.set_yticks(range(len(conds)))
        ax.set_yticklabels([COND_LABELS[c] for c in conds], fontsize=8)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Δ accuracy", fontsize=8)
        ax.set_title(title, fontsize=9)
        for ci in range(len(conds)):
            for bi in range(len(benches)):
                ax.text(bi, ci, f"{mat[ci,bi]:+.2f}", ha="center", va="center", fontsize=7.5)

    plt.suptitle(f"R39C {disp}: Δ accuracy heatmap (emp_logit omitted)", fontsize=11)
    plt.tight_layout()
    p = fig_dir / "04_heatmap.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  → {p}")


def plot_macro_summary(all_results, fig_dir, preset_name):
    """宏平均 Δacc 汇总图（最终展示图）"""
    benches = [b for b in BENCH_ORDER if b in all_results]
    macro = {c: np.mean([all_results[b]["accuracies"].get(c,0) for b in benches])
             for c in EVAL_COND_NAMES}
    base_m = macro["baseline"]
    sw_m   = macro["sweep_best"]

    signal_conds = list(FIGURE_SIGNAL_CONDS)
    disp = PRESET_DISPLAY.get(preset_name, preset_name)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左：宏平均绝对精度
    all_conds = FIGURE_COND_NAMES
    vals = [macro[c] for c in all_conds]
    colors = [COND_COLORS[c] for c in all_conds]
    bars = ax1.bar(range(len(all_conds)), vals, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_xticks(range(len(all_conds)))
    ax1.set_xticklabels([COND_LABELS[c] for c in all_conds], rotation=28, ha="right", fontsize=7.5)
    ax1.set_ylabel("Macro-average Accuracy", fontsize=9)
    ax1.set_title(
        f"R39C {disp}\nMacro-average accuracy ({len(benches)} benchmarks; emp_logit omitted)",
        fontsize=10,
    )
    ax1.set_ylim(min(vals)*0.95, max(vals)*1.05)
    ax1.axhline(base_m, color="gray", linewidth=1.2, linestyle=":", label="baseline")
    ax1.axhline(sw_m, color="#1f77b4", linewidth=1.5, linestyle="--", label="sweep_best")
    ax1.legend(fontsize=8)
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x()+bar.get_width()/2, val+0.002,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # 右：每 benchmark 各信号方法最佳 vs sweep_best
    y = np.arange(len(benches))
    beat_sw_count = 0
    for cn in signal_conds:
        accs = [all_results[b]["accuracies"].get(cn,0) for b in benches]
        ax2.plot(accs, y, "o-", color=COND_COLORS[cn], label=COND_LABELS[cn],
                 linewidth=1.5, markersize=6)
    sw_accs = [all_results[b]["accuracies"].get("sweep_best",0) for b in benches]
    ax2.plot(sw_accs, y, "s--", color="#1f77b4", label="Sweep Best", linewidth=1.5, markersize=7)
    bl_accs = [all_results[b]["accuracies"].get("baseline",0) for b in benches]
    ax2.plot(bl_accs, y, "^:", color="#888888", label="Baseline", linewidth=1.2, markersize=5)
    for bi, b in enumerate(benches):
        a = all_results[b]["accuracies"]
        best_sig = max(a.get(c,0) for c in signal_conds)
        if best_sig > a.get("sweep_best",0):
            beat_sw_count += 1
            ax2.annotate("★", (best_sig, bi), fontsize=11, color="gold",
                         ha="left", va="center")
    ax2.set_yticks(y)
    ax2.set_yticklabels([b[:18] for b in benches], fontsize=8)
    ax2.set_xlabel("Accuracy", fontsize=9)
    ax2.set_title(
        f"Per-benchmark accuracy (★ beats sweep best: {beat_sw_count}/{len(benches)})",
        fontsize=9,
    )
    ax2.legend(fontsize=7, loc="lower right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    p = fig_dir / "05_macro_summary.png"
    plt.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  → {p}")


def plot_window_alignment(all_results, fig_dir, preset_name):
    """窗口选择与 sweep_best 的对齐情况"""
    benches = [b for b in BENCH_ORDER if b in all_results]
    fig, ax = plt.subplots(figsize=(12, 5))
    y = np.arange(len(benches))
    bar_h = 0.28
    disp = PRESET_DISPLAY.get(preset_name, preset_name)
    for bi, b in enumerate(benches):
        res = all_results[b]
        sw = res["sweep_best_window"]
        cw = res.get("neg_cos_am_calib_win", sw)
        # Two tracks per row: sweep_best (lower), calib (upper); emp omitted in figure
        ax.broken_barh([(sw[0], sw[1]-sw[0])], (bi - 0.38, bar_h), facecolors=COND_COLORS["sweep_best"], alpha=0.75)
        ax.broken_barh([(cw[0], cw[1]-cw[0])], (bi + 0.10, bar_h), facecolors=COND_COLORS["neg_cos_am_calib"], alpha=0.75)
    ax.set_ylim(-0.55, len(benches) - 0.35)
    ax.set_yticks(y)
    ax.set_yticklabels([b[:18] for b in benches], fontsize=8)
    ax.set_xlabel("Layer index")
    ax.set_title(
        f"R39C {disp}: selected decoder windows vs sweep best\n(emp_logit not shown)",
        fontsize=10,
    )
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COND_COLORS["sweep_best"],       label="Sweep best window"),
        Patch(facecolor=COND_COLORS["neg_cos_am_calib"], label="neg_cos_am calib window"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = fig_dir / "06_window_alignment.png"
    plt.savefig(p, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  → {p}")


# ──────────────────────────────────────────────────────────────────────────────
# 打印结果表
# ──────────────────────────────────────────────────────────────────────────────
def print_results_table(all_results, preset_name):
    benches = [b for b in BENCH_ORDER if b in all_results]
    print(f"\n{'='*95}")
    print(f"R39C {preset_name} 最终精度结果")
    print(f"{'='*95}")
    print(f"  {'Benchmark':24}" + "".join(f"{c[:11]:>13}" for c in EVAL_COND_NAMES))
    print("-"*95)
    for b in benches:
        a = all_results[b]["accuracies"]
        sw, bl = a.get("sweep_best",0), a.get("baseline",0)
        row = f"  {b[:23]:24}"
        for c in EVAL_COND_NAMES:
            v = a.get(c, 0)
            if c not in ("baseline","sweep_best"):
                mk = "★" if v > sw else ("+" if v > bl else " ")
            else:
                mk = " "
            row += f"  {v:.3f}{mk}{'':7}"
        print(row)
    print("-"*95)
    macro = {c: np.mean([all_results[b]["accuracies"].get(c,0) for b in benches]) for c in EVAL_COND_NAMES}
    sw_m, bl_m = macro["sweep_best"], macro["baseline"]
    row = f"  {'MACRO AVG':24}"
    for c in EVAL_COND_NAMES:
        v = macro[c]
        mk = "★" if c not in ("baseline","sweep_best") and v>sw_m else ("+" if c not in ("baseline","sweep_best") and v>bl_m else " ")
        row += f"  {v:.3f}{mk}{'':7}"
    print(row)

    beat_sw = sum(1 for b in benches
                  for c in ["neg_cos_am_calib","emp_logit_fixed","neg_cos_am_ps_nt"]
                  if all_results[b]["accuracies"].get(c,0) > all_results[b]["accuracies"].get("sweep_best",0))
    beat_bl = sum(1 for b in benches
                  for c in ["neg_cos_am_calib","emp_logit_fixed","neg_cos_am_ps_nt"]
                  if all_results[b]["accuracies"].get(c,0) > all_results[b]["accuracies"].get("baseline",0))
    print(f"\n  超越 sweep_best：{beat_sw}/24 (benchmark × method)  "
          f"超越 baseline：{beat_bl}/24")

    print("\n  窗口选择：")
    for b in benches:
        v = all_results[b]
        print(f"    {b[:22]:22} sw={str(v['sweep_best_window']):12} "
              f"calib={str(v['neg_cos_am_calib_win']):12} "
              f"emp={str(v['emp_logit_win']) if v['emp_logit_win'] else 'fallback':12}")


# ──────────────────────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS.keys()), required=True)
    ap.add_argument(
        "--figures-only",
        action="store_true",
        help="Load existing results JSON and regenerate figures only (no model eval).",
    )
    args = ap.parse_args()

    preset   = PRESETS[args.preset]
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    fig_dir  = Path(preset["out_fig"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(preset["out_json"])
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if args.figures_only:
        if not out_json.is_file():
            raise SystemExit(f"--figures-only: missing results JSON: {out_json}")
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        all_results = payload.get("results", payload)
        preset_name = payload.get("preset", args.preset)
        print(f"\n[figures-only] loaded {out_json} ({len(all_results)} benchmarks)")
        if all_results:
            print_results_table(all_results, preset_name)
            plot_main_bars(all_results, fig_dir, preset_name)
            plot_delta_scatter(all_results, fig_dir, preset_name)
            plot_profiles(all_results, fig_dir, preset_name)
            plot_summary_heatmap(all_results, fig_dir, preset_name)
            plot_macro_summary(all_results, fig_dir, preset_name)
            plot_window_alignment(all_results, fig_dir, preset_name)
        print(f"\n完成。仅重绘图表。源: {out_json}")
        return

    # 加载 sweep_best
    if args.preset == "qwen3-8b":
        sweep_best = dict(preset["sweep_best"])
    else:
        sweep_best = load_sweep_best_from_files(preset)

    print(f"\n[sweep_best] " + ", ".join(f"{b}:{sw}" for b,sw in sweep_best.items()))

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
            print(f"  [ERROR] load {bench}: {e}"); continue
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
            import traceback; print(f"  [ERROR] {bench}: {e}"); traceback.print_exc(); continue

        out_payload = {"preset": args.preset, "arch": preset["arch"],
                       "n_layers": n_layers, "results": all_results}
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, indent=2)
        print(f"  checkpoint → {out_json}")

    if all_results:
        print_results_table(all_results, args.preset)
        plot_main_bars(all_results, fig_dir, args.preset)
        plot_delta_scatter(all_results, fig_dir, args.preset)
        plot_profiles(all_results, fig_dir, args.preset)
        plot_summary_heatmap(all_results, fig_dir, args.preset)
        plot_macro_summary(all_results, fig_dir, args.preset)
        plot_window_alignment(all_results, fig_dir, args.preset)

    print(f"\n完成。结果: {out_json}")


if __name__ == "__main__":
    main()
