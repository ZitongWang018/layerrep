"""
R38: 全 Benchmark 信号引导 ETD 优化实验
================================================================
目标：将 R37 的 cos(Term1, Δh) 信号引导选层方法扩展到全部 8 个 benchmark，
      通过"标定阶段"（前 N_CALIB=20 样本聚合 cos_res profile）解决新 benchmark
      无 R36 预计算数据的问题，迭代优化找到普适最佳信号。

Benchmark 和样本数（与 R30 sweep 对齐）：
  ARC-C, CSQA, BoolQ, LogiQA, MMLU-HS-Math, GPQA-Diamond,
  AGIEval-Gaokao-MathQA  →  N=100
  TruthfulQA              →  N=50

实验条件（7 个，无 oracle 命名）：
  C0  baseline        — 无循环标准前向
  C1  sweep_best      — R30 扫参最优固定窗口（取代 oracle）
  C2  persample_cos8  — 逐样本 cos_res 滑动窗口，n_t=8
  C3  persample_var   — 逐样本 n_t∈{4,6,8} 全搜索
  C4  onset_fixed8    — 固定阈值 0.28 onset，n_t=8
  C5  calib_onset8    — 标定自适应阈值 onset，n_t=8
  C6  calib_global8   — 标定均值最优全局窗口，n_t=8

标定机制（核心创新）：
  对每个 benchmark 的前 N_CALIB=20 个样本做探针前向，聚合 mean cos_res profile。
  calib_onset8 阈值 = max(mean_profile in [9,22]) × 0.65（自适应，免调参）
  calib_global8 = 均值最高 8 层滑动窗口（从 mean_profile 中搜索）

信号机制（Term1 近似交换子，与 R37 一致）：
  探针前向时用 hook 捕获：h_i（层输入），a_l（Attention 输出），m_l（FFN 输出）
  对每个探针层 l：
    m_l0 = mlp(post_attn_norm(h_i))     ← FFN 作用于预注意力状态
    term1 = m_l_actual - m_l0           ← 注意力对知识方向的改变
    cos_res = cos(term1, a_l + m_l)     ← 与实际残差更新的对齐度

输出：
  results/r38_signal_full_bench_results.json
  figures/r38_signal_full/summary_bar_{bench}.png
  figures/r38_signal_full/heatmap_delta_acc.png
  figures/r38_signal_full/delta_acc_scatter.png
  figures/r38_signal_full/calib_profiles.png
  figures/r38_signal_full/tstart_violin.png
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
import matplotlib.colors as mcolors
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
from etd_forward import etd_forward_logits, baseline_forward_logits

# ─── 配置 ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = "/root/autodl-tmp/model_qwen"
RESULTS_DIR = EXP / "results"
FIGURES_DIR = EXP / "figures" / "r38_signal_full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE    = torch.bfloat16
N_CALIB  = 20          # 标定阶段样本数（probe pass 聚合 mean cos_res profile）
K_ETD    = 2           # ETD 重复次数
MIN_START = 9          # 选层搜索下界
MAX_START = 22         # 选层搜索上界

# 探针层：每隔 2 层，覆盖 L6-L28（与 R37 一致）
PROBE_LAYERS = list(range(6, 29, 2))  # [6,8,10,12,14,16,18,20,22,24,26,28]

# ── 扫参最优窗口（来自 R36 R30_OPTIMAL，取代 oracle 命名）────────────────────
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

# ── 样本数（与 R30 sweep 一致）───────────────────────────────────────────────
N_SAMPLES = {
    "BoolQ":                  100,
    "ARC-C":                  100,
    "TruthfulQA":             50,
    "CSQA":                   100,
    "MMLU-HS-Math":           100,
    "GPQA-Diamond":           100,
    "AGIEval-Gaokao-MathQA":  100,
    "LogiQA":                 100,
}

# ── 条件颜色 / 标签 ────────────────────────────────────────────────────────────
COND_NAMES = [
    "baseline", "sweep_best",
    "persample_cos8", "persample_var",
    "onset_fixed8", "calib_onset8", "calib_global8",
]
COND_COLORS = {
    "baseline":       "#9E9E9E",
    "sweep_best":     "#2196F3",
    "persample_cos8": "#4CAF50",
    "persample_var":  "#8BC34A",
    "onset_fixed8":   "#FF9800",
    "calib_onset8":   "#F44336",
    "calib_global8":  "#9C27B0",
}
COND_LABELS = {
    "baseline":       "Baseline",
    "sweep_best":     "扫参最优",
    "persample_cos8": "逐样本-8层",
    "persample_var":  "逐样本-变长",
    "onset_fixed8":   "固定Onset-8",
    "calib_onset8":   "标定Onset-8",
    "calib_global8":  "标定全局-8",
}

BENCH_COLORS = {
    "BoolQ": "#2196F3", "ARC-C": "#F44336", "CSQA": "#4CAF50",
    "TruthfulQA": "#FF9800", "MMLU-HS-Math": "#9C27B0",
    "GPQA-Diamond": "#00BCD4", "AGIEval-Gaokao-MathQA": "#795548",
    "LogiQA": "#607D8B",
}

# ─── 数据加载 ──────────────────────────────────────────────────────────────────
def load_boolq(n: int) -> list[dict]:
    ds = load_dataset("aps/super_glue", "boolq")["validation"]
    out: list[dict] = []
    for x in ds:
        if int(x["label"]) < 0:
            continue
        out.append({
            "prompt": f"{x['passage']}\nQuestion: {x['question']}?\nAnswer:",
            "choices": ["no", "yes"],
            "label": int(x["label"]),
        })
        if len(out) >= n:
            break
    return out


def load_arc_c(n: int) -> list[dict]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")["test"]
    out: list[dict] = []
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


def load_csqa(n: int) -> list[dict]:
    ds = load_dataset("tau/commonsense_qa")["validation"]
    lmap = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    out: list[dict] = []
    for x in ds:
        out.append({
            "prompt": f"Question: {x['question']}\nAnswer:",
            "choices": x["choices"]["text"],
            "label": lmap.get(x["answerKey"], 0),
        })
        if len(out) >= n:
            break
    return out


def load_truthfulqa(n: int) -> list[dict]:
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice")["validation"]
    out: list[dict] = []
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


def load_mmlu_hs_math(n: int) -> list[dict]:
    try:
        ds = load_dataset("cais/mmlu", "high_school_mathematics")["test"]
        out: list[dict] = []
        for x in ds:
            out.append({
                "prompt": f"Question: {x['question']}\nAnswer:",
                "choices": x["choices"],
                "label": int(x["answer"]),
            })
            if len(out) >= n:
                break
        return out
    except Exception as e:
        print(f"  [WARN] MMLU-HS-Math failed: {e}")
        return []


def _adapt_hard(items: list[dict]) -> list[dict]:
    """将 hard_mc_benchmark_loaders 格式适配为统一 label 格式."""
    return [
        {
            "prompt": it["prompt"],
            "choices": it["choices"],
            "label": it.get("valid_indices", [0])[0],
        }
        for it in items
    ]


BENCH_LOADERS: dict[str, callable] = {
    "BoolQ":                  load_boolq,
    "ARC-C":                  load_arc_c,
    "TruthfulQA":             load_truthfulqa,
    "CSQA":                   load_csqa,
    "MMLU-HS-Math":           load_mmlu_hs_math,
    "GPQA-Diamond":           lambda n: _adapt_hard(load_gpqa_diamond(n)),
    "AGIEval-Gaokao-MathQA":  lambda n: _adapt_hard(load_agieval_gaokao_mathqa(n)),
    "LogiQA":                 lambda n: _adapt_hard(load_logiqa(n)),
}


# ─── 工具函数 ──────────────────────────────────────────────────────────────────
def safe_cos(u: torch.Tensor, v: torch.Tensor) -> float:
    u = u.float().reshape(-1).cpu()
    v = v.float().reshape(-1).cpu()
    nu, nv = u.norm(), v.norm()
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    return float((u @ v / (nu * nv)).clamp(-1, 1).item())


def loglikelihood_mc(logits: torch.Tensor, input_ids: torch.Tensor, prompt_len: int) -> float:
    total = 0.0
    for i in range(prompt_len, input_ids.shape[1]):
        logp = F.log_softmax(logits[0, i - 1].float(), dim=-1)
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


# ─── 探针前向：收集 cos_res ───────────────────────────────────────────────────
@torch.no_grad()
def probe_forward_collect_cos_res(
    model, input_ids: torch.Tensor, attn_mask: torch.Tensor | None, n_layers: int
) -> dict[int, float]:
    """
    单次前向 + 每个探针层的额外 MLP 调用，计算 Term1 近似 cos_res。
    返回 {layer_idx: cos_res_value}（仅 PROBE_LAYERS 中的层）。
    """
    base = model.model
    h_inputs: dict[int, torch.Tensor] = {}
    a_outputs: dict[int, torch.Tensor] = {}
    m_outputs: dict[int, torch.Tensor] = {}
    hooks: list = []

    for li in range(n_layers):
        layer = base.layers[li]

        def make_pre(idx: int):
            def fn(_m, args):
                t = args[0] if isinstance(args, tuple) else args
                h_inputs[idx] = t[:, -1:, :].detach().clone()
            return fn

        def make_attn_post(idx: int):
            def fn(_m, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                a_outputs[idx] = t[:, -1:, :].detach().clone()
            return fn

        def make_mlp_post(idx: int):
            def fn(_m, _inp, out):
                m_outputs[idx] = out[:, -1:, :].detach().clone()
            return fn

        hooks.append(layer.register_forward_pre_hook(make_pre(li)))
        hooks.append(layer.self_attn.register_forward_hook(make_attn_post(li)))
        hooks.append(layer.mlp.register_forward_hook(make_mlp_post(li)))

    model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
    for h in hooks:
        h.remove()

    cos_res_by_layer: dict[int, float] = {}
    for li in PROBE_LAYERS:
        hi = h_inputs.get(li)
        al = a_outputs.get(li)
        ml = m_outputs.get(li)
        if hi is None or al is None or ml is None:
            continue
        try:
            layer = base.layers[li]
            m_l0  = layer.mlp(layer.post_attention_layernorm(hi))
            term1 = (ml - m_l0).squeeze()
            delta_h = (al + ml).squeeze()
            cos_res_by_layer[li] = safe_cos(term1, delta_h)
        except Exception:
            pass

    return cos_res_by_layer


# ─── 标定阶段 ─────────────────────────────────────────────────────────────────
def calibrate_benchmark_profile(
    items_calib: list[dict],
    model,
    tok,
    n_layers: int,
) -> dict[int, float]:
    """
    对前 N_CALIB 样本运行探针前向，聚合出 mean cos_res profile。
    返回 {layer_idx: mean_cos_res}（PROBE_LAYERS 中的层）。
    """
    profile_accum: dict[int, list[float]] = defaultdict(list)
    for item in items_calib[:N_CALIB]:
        prompt = item["prompt"]
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(DEVICE)
        amask = enc.get("attention_mask")
        if amask is not None:
            amask = amask.to(DEVICE)
        cos_res = probe_forward_collect_cos_res(model, ids, amask, n_layers)
        for li, v in cos_res.items():
            profile_accum[li].append(v)

    mean_profile = {
        li: float(np.mean(vs))
        for li, vs in profile_accum.items()
        if vs
    }
    return mean_profile


def derive_global_window(
    profile: dict[int, float],
    n_t: int = 8,
    min_start: int = MIN_START,
    max_start: int = MAX_START,
) -> tuple[int, int]:
    """从 mean_profile 中找均值最高的 n_t 宽滑动窗口."""
    best_start = min_start
    best_score = -999.0
    for start in range(min_start, max_start + 1):
        stop = start + n_t
        vals = [profile[l] for l in profile if start <= l < stop]
        if len(vals) < 2:
            continue
        score = float(np.mean(vals))
        if score > best_score:
            best_score = score
            best_start = start
    return best_start, best_start + n_t


def derive_onset_adaptive(
    profile: dict[int, float],
    ratio: float = 0.65,
    n_t: int = 8,
    min_start: int = MIN_START,
    max_start: int = MAX_START,
) -> tuple[int, int]:
    """
    自适应阈值 onset：threshold = max(profile in [min_start, max_start]) * ratio
    找首个超过阈值的层作为 t_start；fallback 为全局最大值的层。
    """
    valid = {l: v for l, v in profile.items() if min_start <= l <= max_start}
    if not valid:
        return min_start, min_start + n_t
    max_val = max(valid.values())
    threshold = max_val * ratio
    for l in sorted(valid):
        if valid[l] >= threshold:
            return l, l + n_t
    # fallback：取最大值所在层
    t_start = max(valid, key=valid.__getitem__)
    return t_start, t_start + n_t


# ─── 每样本窗口选择 ─────────────────────────────────────────────────────────────
def select_window_persample(
    cos_res_by_layer: dict[int, float],
    n_t: int = 8,
    min_start: int = MIN_START,
    max_start: int = MAX_START,
) -> tuple[int, int]:
    best_start = min_start
    best_score = -999.0
    for start in range(min_start, max_start + 1):
        stop = start + n_t
        vals = [cos_res_by_layer[l] for l in cos_res_by_layer if start <= l < stop]
        if len(vals) < 2:
            continue
        score = float(np.mean(vals))
        if score > best_score:
            best_score = score
            best_start = start
    return best_start, best_start + n_t


def select_window_variable_nt(
    cos_res_by_layer: dict[int, float],
    nt_candidates: tuple[int, ...] = (4, 6, 8),
    min_start: int = MIN_START,
    max_start: int = MAX_START,
) -> tuple[int, int]:
    best_start = min_start
    best_nt = nt_candidates[0]
    best_score = -999.0
    for n_t in nt_candidates:
        for start in range(min_start, max_start + 1):
            stop = start + n_t
            vals = [cos_res_by_layer[l] for l in cos_res_by_layer if start <= l < stop]
            if len(vals) < 2:
                continue
            score = float(np.mean(vals))
            if score > best_score:
                best_score = score
                best_start = start
                best_nt = n_t
    return best_start, best_start + best_nt


def select_window_onset_fixed(
    cos_res_by_layer: dict[int, float],
    threshold: float = 0.28,
    n_t: int = 8,
    min_start: int = MIN_START,
    max_start: int = MAX_START,
) -> tuple[int, int]:
    sorted_layers = sorted(l for l in cos_res_by_layer if min_start <= l <= max_start)
    for l in sorted_layers:
        if cos_res_by_layer[l] >= threshold:
            return l, l + n_t
    return select_window_persample(cos_res_by_layer, n_t, min_start, max_start)


# ─── MC 评测 ─────────────────────────────────────────────────────────────────
def mc_predict(
    model, tok, item: dict,
    n_e: int | None = None,
    n_t: int | None = None,
    k: int = K_ETD,
) -> int:
    """对单道题所有选项打 log-likelihood 分，返回最高分选项下标."""
    prompt = item["prompt"]
    choices = item["choices"]
    scores: list[float] = []

    for cont in choices:
        full = prompt + " " + cont
        enc = tok(full, return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(DEVICE)
        amask = enc.get("attention_mask")
        if amask is not None:
            amask = amask.to(DEVICE)
        plen = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]

        if n_e is not None and n_t is not None and n_t > 0:
            alpha = min(1.0, 6.0 / max(n_t, 1))
            lgts = etd_forward_logits(model, ids, amask, n_e=n_e, n_t=n_t, k=k, alpha=alpha)
        else:
            lgts = baseline_forward_logits(model, ids, amask)

        scores.append(loglikelihood_mc(lgts, ids, plen))

    return int(np.argmax(scores))


# ─── 核心评测循环 ──────────────────────────────────────────────────────────────
def evaluate_benchmark(
    bench: str,
    items: list[dict],
    model,
    tok,
    n_layers: int,
) -> dict:
    """
    对单个 benchmark 评测 7 个条件，先执行标定阶段推导 calib 参数，
    然后对全部 N 个样本评测所有条件。
    """
    n_total = len(items)
    sweep_win = SWEEP_BEST[bench]

    # ── 标定阶段：用前 N_CALIB 样本聚合 mean cos_res profile ────────────────
    print(f"  [标定] 聚合前 {N_CALIB} 样本的 cos_res profile …")
    t_calib = time.time()
    mean_profile = calibrate_benchmark_profile(items, model, tok, n_layers)
    calib_global8_win = derive_global_window(mean_profile, n_t=8)
    calib_onset8_win  = derive_onset_adaptive(mean_profile, ratio=0.65, n_t=8)
    print(f"  [标定] 完成 ({time.time()-t_calib:.1f}s)")
    print(f"    calib_global8  : {calib_global8_win}")
    print(f"    calib_onset8   : {calib_onset8_win}")
    print(f"    sweep_best     : {sweep_win}")

    # ── 初始化计数器 ────────────────────────────────────────────────────────
    correct = {c: 0 for c in COND_NAMES}
    selected_tstarts: dict[str, list[int]] = {
        c: [] for c in ["persample_cos8", "persample_var", "onset_fixed8"]
    }

    t0 = time.time()
    for i, item in enumerate(items):
        label = item["label"]

        # 探针前向（一次，共用所有 persample 条件）
        prompt = item["prompt"]
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
        probe_ids  = enc["input_ids"].to(DEVICE)
        probe_mask = enc.get("attention_mask")
        if probe_mask is not None:
            probe_mask = probe_mask.to(DEVICE)
        cos_res = probe_forward_collect_cos_res(model, probe_ids, probe_mask, n_layers)

        # 每样本动态选窗
        ps8_win  = select_window_persample(cos_res, n_t=8)
        var_win  = select_window_variable_nt(cos_res, nt_candidates=(4, 6, 8))
        on8_win  = select_window_onset_fixed(cos_res, threshold=0.28, n_t=8)

        selected_tstarts["persample_cos8"].append(ps8_win[0])
        selected_tstarts["persample_var"].append(var_win[0])
        selected_tstarts["onset_fixed8"].append(on8_win[0])

        # 条件映射: (n_e, n_t) ← None 表示 baseline
        cond_windows: dict[str, tuple[int, int] | None] = {
            "baseline":       None,
            "sweep_best":     sweep_win,
            "persample_cos8": ps8_win,
            "persample_var":  var_win,
            "onset_fixed8":   on8_win,
            "calib_onset8":   calib_onset8_win,
            "calib_global8":  calib_global8_win,
        }

        for cname in COND_NAMES:
            win = cond_windows[cname]
            if win is None:
                pred = mc_predict(model, tok, item)
            else:
                t_start, t_stop = win
                n_e_c  = t_start
                n_t_c  = t_stop - t_start
                n_d_c  = n_layers - t_stop
                if n_d_c < 1 or n_t_c < 1:
                    pred = mc_predict(model, tok, item)
                else:
                    pred = mc_predict(model, tok, item, n_e=n_e_c, n_t=n_t_c, k=K_ETD)
            if pred == label:
                correct[cname] += 1

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_total - i - 1)
            line = f"  [{i+1:3d}/{n_total}] "
            for cn in COND_NAMES:
                line += f"{cn}={correct[cn]/(i+1):.3f} "
            line += f"| {elapsed:.0f}s elapsed, ETA {eta:.0f}s"
            print(line)

        torch.cuda.empty_cache()

    accuracies = {c: correct[c] / n_total for c in COND_NAMES}
    elapsed = time.time() - t0

    # 窗口统计
    win_stats: dict[str, dict] = {}
    for cn in selected_tstarts:
        tss = selected_tstarts[cn]
        win_stats[cn] = {
            "t_start_mean": float(np.mean(tss)) if tss else 0.0,
            "t_start_std":  float(np.std(tss))  if tss else 0.0,
            "t_start_hist": {str(t): int(tss.count(t)) for t in sorted(set(tss))},
            "t_start_list": tss,
        }

    return {
        "benchmark":           bench,
        "n":                   n_total,
        "elapsed_s":           elapsed,
        "accuracies":          accuracies,
        "sweep_best_window":   list(sweep_win),
        "calib_global8_window": list(calib_global8_win),
        "calib_onset8_window": list(calib_onset8_win),
        "mean_profile":        {str(k): v for k, v in sorted(mean_profile.items())},
        "window_stats":        win_stats,
    }


# ─── 可视化 ────────────────────────────────────────────────────────────────────
def plot_benchmark_bar(bench: str, result: dict):
    """每个 benchmark 单独的条形图（7 conditions）."""
    accs = result["accuracies"]
    base_acc  = accs["baseline"]
    sweep_acc = accs["sweep_best"]

    x = np.arange(len(COND_NAMES))
    bars = [accs[c] for c in COND_NAMES]
    colors = [COND_COLORS[c] for c in COND_NAMES]

    fig, ax = plt.subplots(figsize=(11, 4))
    brs = ax.bar(x, bars, color=colors, edgecolor="black", linewidth=0.5, alpha=0.88)

    for bar, v in zip(brs, bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{v:.3f}", ha="center", va="bottom", fontsize=7.5,
        )

    ax.axhline(base_acc,  color="grey",    linestyle="--", linewidth=1.2, alpha=0.7,
               label=f"Baseline={base_acc:.3f}")
    ax.axhline(sweep_acc, color="#2196F3", linestyle=":",  linewidth=1.5, alpha=0.85,
               label=f"扫参最优={sweep_acc:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABELS[c] for c in COND_NAMES], fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_title(
        f"R38: {bench}  (N={result['n']}, {result['elapsed_s']:.0f}s)\n"
        f"扫参最优窗口={result['sweep_best_window']}  "
        f"标定全局={result['calib_global8_window']}  "
        f"标定Onset={result['calib_onset8_window']}",
        fontsize=9,
    )
    ymax = max(bars) * 1.18
    ax.set_ylim(0, max(ymax, sweep_acc * 1.15 + 0.05))
    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    fname = FIGURES_DIR / f"bar_{bench.replace('/', '_')}.png"
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {fname}")


def plot_heatmap(all_results: dict[str, dict]):
    """
    热力图：8 benchmark × 5 信号方法（不含 baseline）
    单元格颜色 = Δacc vs baseline。
    """
    signal_conds = ["sweep_best", "persample_cos8", "persample_var",
                    "onset_fixed8", "calib_onset8", "calib_global8"]
    bench_list = list(all_results.keys())

    delta_matrix = np.zeros((len(bench_list), len(signal_conds)))
    acc_matrix   = np.zeros_like(delta_matrix)
    for bi, bench in enumerate(bench_list):
        accs = all_results[bench]["accuracies"]
        base = accs["baseline"]
        for ci, cn in enumerate(signal_conds):
            delta_matrix[bi, ci] = accs[cn] - base
            acc_matrix[bi, ci]   = accs[cn]

    fig, ax = plt.subplots(figsize=(12, max(5, len(bench_list) * 0.85)))
    vmax = max(abs(delta_matrix).max(), 0.05)
    im = ax.imshow(delta_matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="Δacc vs Baseline")

    ax.set_xticks(range(len(signal_conds)))
    ax.set_xticklabels([COND_LABELS[c] for c in signal_conds], fontsize=9, rotation=25, ha="right")
    ax.set_yticks(range(len(bench_list)))
    ax.set_yticklabels(bench_list, fontsize=9)

    for bi in range(len(bench_list)):
        for ci in range(len(signal_conds)):
            delta = delta_matrix[bi, ci]
            acc   = acc_matrix[bi, ci]
            color = "white" if abs(delta) > vmax * 0.5 else "black"
            ax.text(ci, bi, f"{acc:.3f}\n({delta:+.3f})",
                    ha="center", va="center", fontsize=7.5, color=color)

    ax.set_title("R38 热力图：各信号方法 Accuracy 及 Δacc vs Baseline", fontsize=11)
    plt.tight_layout()
    fname = FIGURES_DIR / "heatmap_delta_acc.png"
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {fname}")


def plot_delta_scatter(all_results: dict[str, dict]):
    """
    Δacc 散点图：x = sweep_best Δacc，y = 最佳信号方法 Δacc
    对角线 = sweep_best 等价线。
    """
    signal_conds = ["persample_cos8", "persample_var",
                    "onset_fixed8", "calib_onset8", "calib_global8"]
    bench_list = list(all_results.keys())

    fig, ax = plt.subplots(figsize=(7, 7))

    all_deltas: list[float] = []
    for bench in bench_list:
        accs = all_results[bench]["accuracies"]
        base = accs["baseline"]
        sweep_delta = accs["sweep_best"] - base
        all_deltas.append(sweep_delta)
        for cn in signal_conds:
            all_deltas.append(accs[cn] - base)

    lim = max(abs(min(all_deltas)), abs(max(all_deltas)), 0.05) + 0.02

    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1, alpha=0.5, label="与扫参相当")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.axvline(0, color="grey", linewidth=0.8, linestyle=":")

    for cn in signal_conds:
        xs, ys, labels_pt = [], [], []
        for bench in bench_list:
            accs = all_results[bench]["accuracies"]
            base = accs["baseline"]
            xs.append(accs["sweep_best"] - base)
            ys.append(accs[cn] - base)
            labels_pt.append(bench[:6])
        sc = ax.scatter(xs, ys, label=COND_LABELS[cn], alpha=0.8, s=70,
                        color=COND_COLORS[cn], edgecolors="black", linewidths=0.5)
        for xi, yi, lb in zip(xs, ys, labels_pt):
            ax.annotate(lb, (xi, yi), fontsize=6.5, ha="center", va="bottom",
                        xytext=(0, 3), textcoords="offset points")

    ax.set_xlabel("sweep_best Δacc vs Baseline", fontsize=10)
    ax.set_ylabel("信号方法 Δacc vs Baseline", fontsize=10)
    ax.set_title("R38 Δacc 散点图：各信号方法 vs 扫参最优", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    plt.tight_layout()
    fname = FIGURES_DIR / "delta_acc_scatter.png"
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {fname}")


def plot_calib_profiles(all_results: dict[str, dict]):
    """标定 profile 图：8 benchmark 的 mean cos_res 曲线叠加."""
    bench_list = list(all_results.keys())
    fig, ax = plt.subplots(figsize=(12, 5))

    for bench in bench_list:
        profile_raw = all_results[bench].get("mean_profile", {})
        if not profile_raw:
            continue
        profile = {int(k): v for k, v in profile_raw.items()}
        layers = sorted(profile)
        vals   = [profile[l] for l in layers]
        color  = BENCH_COLORS.get(bench, "gray")

        ax.plot(layers, vals, "o-", color=color, linewidth=1.5, markersize=4,
                alpha=0.85, label=bench[:16])

        # 标注 calib_onset tstart
        onset_win = all_results[bench].get("calib_onset8_window")
        if onset_win:
            t_s = onset_win[0]
            val_at_ts = profile.get(t_s, profile.get(min(profile, key=lambda l: abs(l - t_s)), 0))
            ax.axvline(t_s, color=color, linestyle=":", linewidth=1.2, alpha=0.6)
            ax.annotate(
                f"{bench[:4]}@{t_s}",
                (t_s, val_at_ts),
                fontsize=6,
                color=color,
                xytext=(2, 2),
                textcoords="offset points",
            )

    ax.axhline(0.28, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="固定阈值 0.28")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.3)
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Mean cos(Term1, Δh)", fontsize=10)
    ax.set_title("R38 标定 Profile：各 Benchmark 的 Mean cos_res 曲线（N_CALIB=20）", fontsize=11)
    ax.legend(fontsize=7.5, loc="upper right", ncol=2)
    ax.set_xlim(4, 30)
    plt.tight_layout()
    fname = FIGURES_DIR / "calib_profiles.png"
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {fname}")


def plot_tstart_violin(all_results: dict[str, dict]):
    """
    t_start 分布 violin 图：persample_cos8 和 persample_var 在各 benchmark 上的分布。
    """
    bench_list = list(all_results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, cname in zip(axes, ["persample_cos8", "persample_var"]):
        data  = []
        ticks = []
        sweep_tstarts = []
        for bench in bench_list:
            ws = all_results[bench].get("window_stats", {}).get(cname)
            if ws and ws.get("t_start_list"):
                data.append(ws["t_start_list"])
                ticks.append(bench[:10])
                sweep_tstarts.append(SWEEP_BEST[bench][0])
            else:
                data.append([0])
                ticks.append(bench[:10])
                sweep_tstarts.append(SWEEP_BEST[bench][0])

        if any(len(d) > 1 for d in data):
            vp = ax.violinplot(data, positions=range(len(ticks)),
                               showmedians=True, showextrema=True)
            for body in vp["bodies"]:
                body.set_alpha(0.65)
                body.set_facecolor(COND_COLORS[cname])
        else:
            for bi, (d, pos) in enumerate(zip(data, range(len(ticks)))):
                ax.scatter([pos] * len(d), d, alpha=0.7,
                           color=COND_COLORS[cname], s=30)

        for bi, (ts, pos) in enumerate(zip(sweep_tstarts, range(len(ticks)))):
            ax.scatter(pos, ts, marker="D", s=55, color="#2196F3",
                       zorder=5, label="扫参最优 t_start" if bi == 0 else "")

        ax.set_xticks(range(len(ticks)))
        ax.set_xticklabels(ticks, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Selected t_start", fontsize=9)
        ax.set_title(f"{COND_LABELS[cname]}\n(蓝◆=扫参最优 t_start)", fontsize=9)
        ax.legend(fontsize=8)

    fig.suptitle("R38 t_start 选层分布（Violin）", fontsize=11, y=1.02)
    plt.tight_layout()
    fname = FIGURES_DIR / "tstart_violin.png"
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {fname}")


def plot_summary_final(all_results: dict[str, dict]):
    """
    最终汇总图：每个 benchmark 3 条柱（baseline / 最佳信号 / sweep_best），
    方便直观对比"信号引导 vs 扫参最优 vs 无 ETD"。
    """
    signal_conds = ["persample_cos8", "persample_var",
                    "onset_fixed8", "calib_onset8", "calib_global8"]
    bench_list = list(all_results.keys())
    n_bench = len(bench_list)

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(n_bench)
    width = 0.26

    baselines  = [all_results[b]["accuracies"]["baseline"]  for b in bench_list]
    sweep_accs = [all_results[b]["accuracies"]["sweep_best"] for b in bench_list]
    # 最佳信号方法（按各 benchmark 赢家）
    best_signal_accs = []
    best_signal_names = []
    for bench in bench_list:
        accs = all_results[bench]["accuracies"]
        best_c = max(signal_conds, key=lambda c: accs[c])
        best_signal_accs.append(accs[best_c])
        best_signal_names.append(COND_LABELS[best_c])

    b1 = ax.bar(x - width, baselines,      width, label="Baseline",   color="#9E9E9E", alpha=0.85)
    b2 = ax.bar(x,         best_signal_accs, width, label="最佳信号方法", color="#F44336", alpha=0.85)
    b3 = ax.bar(x + width, sweep_accs,      width, label="扫参最优",   color="#2196F3", alpha=0.85)

    for bar, v in zip(b1, baselines):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)
    for bar, v, nm in zip(b2, best_signal_accs, best_signal_names):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{v:.3f}\n({nm[:4]})", ha="center", va="bottom", fontsize=6)
    for bar, v in zip(b3, sweep_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels([b[:14] for b in bench_list], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_title("R38 汇总：Baseline vs 最佳信号方法 vs 扫参最优", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, min(1.0, max(max(sweep_accs), max(best_signal_accs)) * 1.25 + 0.05))
    plt.tight_layout()
    fname = FIGURES_DIR / "summary_final.png"
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {fname}")


# ─── 主流程 ───────────────────────────────────────────────────────────────────
def print_hypothesis_check(all_results: dict[str, dict]):
    signal_conds = ["persample_cos8", "persample_var",
                    "onset_fixed8", "calib_onset8", "calib_global8"]
    print("\n" + "=" * 70)
    print("假设验证摘要（R38）")
    print("=" * 70)

    global_winner_counts: dict[str, int] = defaultdict(int)
    beats_baseline_counts: dict[str, int] = defaultdict(int)

    for bench, res in all_results.items():
        accs = res["accuracies"]
        base  = accs["baseline"]
        sweep = accs["sweep_best"]
        best_sig = max(signal_conds, key=lambda c: accs[c])
        print(f"\n{bench}  (N={res['n']}):")
        print(f"  Baseline={base:.4f}  扫参最优={sweep:.4f}  Δsweep={sweep-base:+.4f}")
        for cn in signal_conds:
            delta  = accs[cn] - base
            frac   = accs[cn] / sweep if sweep > 0 else 0.0
            win_mk = "★" if cn == best_sig else " "
            print(f"  {win_mk} {cn:18s}: {accs[cn]:.4f}  Δ={delta:+.4f}  {frac:.1%} of sweep")
            if accs[cn] > base:
                beats_baseline_counts[cn] += 1
            global_winner_counts[best_sig] += 1

    n_bench = len(all_results)
    print(f"\n{'─'*70}")
    print(f"方法 | 赢 benchmark 数 | 优于 baseline 的 benchmark 数")
    for cn in signal_conds:
        wins = sum(1 for res in all_results.values()
                   if max(signal_conds, key=lambda c: res["accuracies"][c]) == cn)
        beats = beats_baseline_counts[cn]
        print(f"  {COND_LABELS[cn]:12s} | {wins}/{n_bench}       | {beats}/{n_bench}")

    # Macro-average Δacc
    print(f"\n{'─'*70}")
    print("宏平均 Δacc（vs baseline）：")
    for cn in signal_conds:
        macro = np.mean([res["accuracies"][cn] - res["accuracies"]["baseline"]
                         for res in all_results.values()])
        print(f"  {COND_LABELS[cn]:12s} : {macro:+.4f}")
    sweep_macro = np.mean([res["accuracies"]["sweep_best"] - res["accuracies"]["baseline"]
                           for res in all_results.values()])
    print(f"  {'扫参最优':12s} : {sweep_macro:+.4f}")


def main():
    t_total = time.time()
    print("=" * 70)
    print("R38: Signal-Guided ETD — Full Benchmark Evaluation")
    print(f"N_CALIB={N_CALIB}  PROBE_LAYERS={PROBE_LAYERS}")
    print(f"Conditions: {COND_NAMES}")
    print("=" * 70)

    tok, model, n_layers = load_model()

    all_results: dict[str, dict] = {}

    # 检查是否有已有结果可以恢复
    results_path = RESULTS_DIR / "r38_signal_full_bench_results.json"
    if results_path.exists():
        try:
            with open(results_path) as f:
                all_results = json.load(f)
            print(f"  [恢复] 读取到 {len(all_results)} 个已完成 benchmark 的结果")
        except Exception:
            all_results = {}

    for bench, loader in BENCH_LOADERS.items():
        if bench in all_results:
            print(f"  [跳过] {bench}（已有结果）")
            continue

        n = N_SAMPLES[bench]
        print(f"\n{'─'*60}")
        print(f"Benchmark: {bench}  N={n}")
        print(f"  sweep_best = {SWEEP_BEST[bench]}")
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

        result = evaluate_benchmark(bench, items, model, tok, n_layers)
        all_results[bench] = result

        # 打印当前 benchmark 结果
        accs = result["accuracies"]
        print(f"\n  === {bench} Results ===")
        for cn in COND_NAMES:
            delta = accs[cn] - accs["baseline"]
            sweep_ratio = accs[cn] / accs["sweep_best"] if accs["sweep_best"] > 0 else 0
            print(f"    {cn:18s}: {accs[cn]:.4f}  Δ={delta:+.4f}  "
                  f"{sweep_ratio:.1%} of sweep_best")

        # 每 benchmark 生成条形图
        plot_benchmark_bar(bench, result)

        # 保存中间结果
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  保存中间结果 → {results_path}")

    # ── 汇总可视化 ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("生成汇总可视化 …")
    if all_results:
        plot_heatmap(all_results)
        plot_delta_scatter(all_results)
        plot_calib_profiles(all_results)
        plot_tstart_violin(all_results)
        plot_summary_final(all_results)

    print_hypothesis_check(all_results)

    # 保存最终结果
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {results_path}")

    elapsed = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"R38 完成！总耗时 {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
