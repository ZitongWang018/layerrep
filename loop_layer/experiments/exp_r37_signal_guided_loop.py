"""
R37: 信号引导的 ETD 循环层选择 — 硬推理 benchmark 评测
================================================================
目标：验证 commutator_cos_with_residual 信号能否替代人工扫参，
      在 MMLU-HS-Math / GPQA-Diamond / AGIEval 上自动找到接近最优的循环区间。

实验条件（每 benchmark 100 样本）：
  C0  baseline         — 无循环标准前向
  C1  oracle           — 先验最优窗口（R30 扫参或历史经验）
  C2  global_cos6      — 从 R36 benchmark 级平均 cos_res 数据预推导的固定窗口 n_t=6
  C3  global_cos8      — 同上，n_t=8
  C4  persample_cos6   — 每样本探针前向 + Term1 cos_res 选窗，n_t=6
  C5  persample_cos8   — 同上，n_t=8
  C6  persample_cos10  — 同上，n_t=10

假设：
  H1: global_cos6 > baseline   （信号先验优于无 ETD）
  H2: persample_cosX ≥ global_cosX  （每样本选层优于固定层）
  H3: max(C2..C6) ≥ 0.90 × oracle  （信号引导接近最优）

信号机制（Term1 近似交换子）：
  探针前向时用 hook 捕获：h_i, a_l（attention 贡献），m_l（FFN 贡献）
  对每个探针层 l 额外计算：
    m_l0 = layer.mlp(layer.post_attn_norm(h_i))   ← FFN 作用于预注意力隐状态
    term1 = m_l_actual - m_l0                      ← 注意力对知识方向的改变
    cos_res = cos(term1, a_l + m_l_actual)          ← 与实际残差更新的对齐度
  选择使 mean(cos_res) 最大的 n_t 连续层作为循环区间。

输出：
  results/r37_signal_loop_results.json
  figures/r37_signal_loop/summary_bar.png
  figures/r37_signal_loop/{bench}_r37_conds_vs_layer.png
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

# ─── 配置 ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = "/root/autodl-tmp/model_qwen"
RESULTS_DIR = EXP / "results"
FIGURES_DIR = EXP / "figures" / "r37_signal_loop"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 100
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE     = torch.bfloat16

# 探针层：每隔 2 层，覆盖 L6-L28
PROBE_LAYERS = list(range(6, 29, 2))  # [6,8,10,...,28]

# ── 先验最优窗口 (Oracle) ──────────────────────────────────────────────────────
# MMLU-HS-Math: R30 扫参最优 [10,18] 准确率 0.45 (+0.07 vs baseline 0.38)
# GPQA-Diamond / AGIEval: 历史实验经验值
ORACLE_WINDOWS = {
    "MMLU-HS-Math":          (10, 18),   # R30 已验证
    "GPQA-Diamond":          (15, 21),   # 信号分析推导（cos_res 中层峰值区 L15=0.341）
    "AGIEval-Gaokao-MathQA": (13, 20),   # R36 reference
}

# ── 从 R36 benchmark 级聚合 cos_res 数据推导的固定窗口 ─────────────────────────
# 选取探针层中 cos_res 均值最高的连续区间（排除前6层普遍初始化峰）
# MMLU-HS-Math: L15=0.406, L12=0.354, L18=0.375 → [12,18] (n_t=6) / [10,18] (n_t=8)
# GPQA-Diamond: L15=0.341 peak (excl L6=0.458) → [12,18]→[15,21] (n_t=6) / [12,20] (n_t=8)
# AGIEval:      L15=0.292, L18=0.318 → [15,21] (n_t=6) / [12,20] (n_t=8)
R36_COS_RES = {
    "MMLU-HS-Math": {
        3:0.276, 6:0.341, 9:0.284, 12:0.354, 15:0.406, 18:0.375,
        21:0.270, 24:0.224, 27:0.054, 30:0.214, 33:0.106,
    },
    "GPQA-Diamond": {
        3:0.258, 6:0.458, 9:0.165, 12:0.193, 15:0.341, 18:0.275,
        21:0.169, 24:0.152, 27:-0.029, 30:0.226, 33:-0.114,
    },
    "AGIEval-Gaokao-MathQA": {
        3:0.322, 6:0.269, 9:0.299, 12:0.198, 15:0.292, 18:0.318,
        21:0.066, 24:-0.022, 27:-0.039, 30:0.442, 33:0.105,
    },
}

def derive_global_window(bench: str, n_t: int, min_start: int = 9, max_start: int = 22) -> tuple[int, int]:
    """从 R36 聚合 cos_res 数据中找最优滑动窗口 [t_start, t_start+n_t]."""
    cos_res = R36_COS_RES[bench]
    best_start = min_start
    best_score = -999.0
    for start in range(min_start, max_start + 1):
        stop = start + n_t
        window_vals = [cos_res[l] for l in cos_res if start <= l < stop]
        if len(window_vals) < 2:
            continue
        score = float(np.mean(window_vals))
        if score > best_score:
            best_score = score
            best_start = start
    return best_start, best_start + n_t


# 预计算全局窗口
GLOBAL_WINDOWS = {}
for _bench in ["MMLU-HS-Math", "GPQA-Diamond", "AGIEval-Gaokao-MathQA"]:
    GLOBAL_WINDOWS[_bench] = {
        6:  derive_global_window(_bench, 6),
        8:  derive_global_window(_bench, 8),
    }

print("预计算全局窗口（来自 R36 数据）：")
for _b, _ws in GLOBAL_WINDOWS.items():
    print(f"  {_b}: n_t=6 → {_ws[6]}, n_t=8 → {_ws[8]}")


# ─── 工具函数 ──────────────────────────────────────────────────────────────────
def safe_cos(u: torch.Tensor, v: torch.Tensor) -> float:
    u = u.float().reshape(-1).cpu()
    v = v.float().reshape(-1).cpu()
    nu, nv = u.norm(), v.norm()
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    return float((u @ v / (nu * nv)).clamp(-1, 1).item())


def _last(t: torch.Tensor) -> torch.Tensor:
    """取最后一个 token 的向量 [D]."""
    return t[0, -1]


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


# ─── 数据加载 ──────────────────────────────────────────────────────────────────
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
        print(f"  [WARN] MMLU-HS-Math failed: {e}")
        return []

def _adapt(items):
    return [{"prompt": it["prompt"], "choices": it["choices"],
             "label": it.get("valid_indices", [0])[0]} for it in items]

BENCH_LOADERS = {
    "MMLU-HS-Math":          load_mmlu_hs_math,
    "GPQA-Diamond":          lambda n: _adapt(load_gpqa_diamond(n)),
    "AGIEval-Gaokao-MathQA": lambda n: _adapt(load_agieval_gaokao_mathqa(n)),
}


# ─── 探针前向：收集 cos_res（Term1 近似）────────────────────────────────────────
@torch.no_grad()
def probe_forward_collect_cos_res(model, input_ids, attn_mask, n_layers):
    """
    单次前向 + post-forward FFN 调用计算 Term1 近似 cos_res。
    返回：
      - orig_logits: [1, vocab] 最后一个 token 的 logits（作为 baseline 使用）
      - cos_res_by_layer: {layer_idx: float} 各探针层的 cos_res 值
    """
    base = model.model
    h_inputs = {}
    a_outputs = {}
    m_outputs = {}
    hooks = []

    for li in range(n_layers):
        layer = base.layers[li]

        def make_pre(idx):
            def fn(_m, args):
                t = args[0] if isinstance(args, tuple) else args
                h_inputs[idx] = t[:, -1:, :].detach().clone()   # [1,1,D]
            return fn

        def make_attn_post(idx):
            def fn(_m, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                a_outputs[idx] = t[:, -1:, :].detach().clone()  # [1,1,D]
            return fn

        def make_mlp_post(idx):
            def fn(_m, _inp, out):
                m_outputs[idx] = out[:, -1:, :].detach().clone()  # [1,1,D]
            return fn

        hooks.append(layer.register_forward_pre_hook(make_pre(li)))
        hooks.append(layer.self_attn.register_forward_hook(make_attn_post(li)))
        hooks.append(layer.mlp.register_forward_hook(make_mlp_post(li)))

    out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
    for h in hooks:
        h.remove()

    orig_logits = out.logits[:, -1:, :].float()  # [1,1,vocab]

    # 对探针层计算 Term1 cos_res
    cos_res_by_layer: dict[int, float] = {}
    for li in PROBE_LAYERS:
        hi = h_inputs.get(li)
        al = a_outputs.get(li)
        ml = m_outputs.get(li)
        if hi is None or al is None or ml is None:
            continue
        try:
            layer = base.layers[li]
            # Term1 = mlp(post_attn_norm(h_i)) - m_l_actual
            # （与 R35 一致：用 h_i 近似"未经注意力的"FFN 输入）
            m_l0 = layer.mlp(layer.post_attention_layernorm(hi))  # [1,1,D]
            term1  = (ml - m_l0).squeeze()    # [D]
            delta_h = (al + ml).squeeze()      # [D]
            cos_res_by_layer[li] = safe_cos(term1, delta_h)
        except Exception:
            pass

    return orig_logits, cos_res_by_layer


# ─── 每样本窗口选择 ─────────────────────────────────────────────────────────────
def select_window_persample(cos_res_by_layer: dict[int, float],
                             n_t: int,
                             min_start: int = 9,
                             max_start: int = 22) -> tuple[int, int]:
    """
    在 [min_start, max_start] 搜索 n_t 宽的滑动窗口，
    返回 mean(cos_res) 最高的 (t_start, t_stop)。
    """
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


def select_window_variable_nt(cos_res_by_layer: dict[int, float],
                               nt_candidates: tuple[int, ...] = (4, 6, 8),
                               min_start: int = 9,
                               max_start: int = 22) -> tuple[int, int]:
    """
    在所有 (t_start, n_t) 组合中找 mean(cos_res) 最高的，自动选 n_t。
    """
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


def select_window_onset(cos_res_by_layer: dict[int, float],
                         threshold: float = 0.28,
                         n_t: int = 8,
                         min_start: int = 9,
                         max_start: int = 22) -> tuple[int, int]:
    """
    Onset criterion: 找到首个 cos_res > threshold 的层作为 t_start。
    代表"最早进入强对齐区域"的层，比峰值启发更早开始循环。
    """
    sorted_layers = sorted(l for l in cos_res_by_layer if min_start <= l <= max_start)
    for l in sorted_layers:
        if cos_res_by_layer[l] >= threshold:
            return l, l + n_t
    # fallback: sliding window max
    return select_window_persample(cos_res_by_layer, n_t, min_start, max_start)


# ─── MC 评测（多选题 log-likelihood）──────────────────────────────────────────
def mc_predict_from_logits(
    model, tok, item: dict, logits_baseline: torch.Tensor | None = None,
    n_e: int | None = None, n_t: int | None = None, k: int = 2,
    device: str = "cuda",
) -> int:
    """
    对单道题的所有选项打分，返回最高得分的选项下标。
    若传入 n_e 和 n_t，则走 ETD 前向；否则走 baseline 前向。
    """
    prompt = item["prompt"]
    choices = item["choices"]
    scores = []

    for cont in choices:
        full = prompt + " " + cont
        enc = tok(full, return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        amask = enc.get("attention_mask")
        if amask is not None:
            amask = amask.to(device)
        plen = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]

        if n_e is not None and n_t is not None:
            alpha = min(1.0, 6.0 / max(n_t, 1))
            lgts = etd_forward_logits(model, ids, amask, n_e=n_e, n_t=n_t, k=k, alpha=alpha)
        else:
            lgts = baseline_forward_logits(model, ids, amask)

        scores.append(loglikelihood_mc(lgts, ids, plen))

    return int(np.argmax(scores))


# ─── 核心评测循环 ──────────────────────────────────────────────────────────────
def evaluate_benchmark(bench: str, items: list[dict], model, tok, n_layers: int) -> dict:
    """
    对单个 benchmark 评测 7 个条件，返回 results dict。
    C0: baseline
    C1: oracle (先验最优窗口)
    C2: global_cos6
    C3: global_cos8
    C4: persample_cos6
    C5: persample_cos8
    C6: persample_cos10
    """
    oracle = ORACLE_WINDOWS[bench]
    glb6   = GLOBAL_WINDOWS[bench][6]
    glb8   = GLOBAL_WINDOWS[bench][8]

    cond_names = ["baseline", "oracle", "global_cos6", "global_cos8",
                  "persample_cos6", "persample_cos8", "persample_cos10",
                  "persample_variable", "onset_cos8"]
    cond_windows = {
        "baseline":          None,
        "oracle":            oracle,
        "global_cos6":       glb6,
        "global_cos8":       glb8,
        "persample_cos6":    None,  # 每样本动态确定
        "persample_cos8":    None,
        "persample_cos10":   None,
        "persample_variable": None, # 变长 n_t 选择
        "onset_cos8":        None,  # Onset criterion, n_t=8
    }

    correct = {c: 0 for c in cond_names}
    total   = len(items)
    selected_windows = {c: [] for c in [
        "persample_cos6", "persample_cos8", "persample_cos10",
        "persample_variable", "onset_cos8"
    ]}

    t0 = time.time()
    for i, item in enumerate(items):
        label = item["label"]

        # ── 探针前向（同时给出 baseline logits）────────────────────────────────
        prompt = item["prompt"]
        choices = item["choices"]

        # Baseline scores via normal forward
        base_scores = []
        for cont in choices:
            full = prompt + " " + cont
            enc = tok(full, return_tensors="pt", add_special_tokens=False)
            ids = enc["input_ids"].to(DEVICE)
            amask = enc.get("attention_mask")
            if amask is not None: amask = amask.to(DEVICE)
            plen = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
            lgts = baseline_forward_logits(model, ids, amask)
            base_scores.append(loglikelihood_mc(lgts, ids, plen))

        # Use last choice's hidden states for cos_res probe (only need the prompt part)
        # Run probe on prompt only for signal collection
        enc_p = tok(prompt, return_tensors="pt", add_special_tokens=False)
        ids_p = enc_p["input_ids"].to(DEVICE)
        amask_p = enc_p.get("attention_mask")
        if amask_p is not None: amask_p = amask_p.to(DEVICE)

        _, cos_res = probe_forward_collect_cos_res(model, ids_p, amask_p, n_layers)

        # C0: Baseline
        if int(np.argmax(base_scores)) == label:
            correct["baseline"] += 1

        # C1-C3: Fixed windows
        for cname in ["oracle", "global_cos6", "global_cos8"]:
            win = cond_windows[cname]
            t_start, t_stop = win
            n_t = t_stop - t_start
            pred = mc_predict_from_logits(model, tok, item, n_e=t_start, n_t=n_t,
                                           device=DEVICE)
            if pred == label:
                correct[cname] += 1

        # C4-C6: Per-sample fixed-nt windows
        for cname, nt in [("persample_cos6", 6), ("persample_cos8", 8), ("persample_cos10", 10)]:
            t_start, t_stop = select_window_persample(cos_res, n_t=nt)
            selected_windows[cname].append((t_start, t_stop))
            n_t_val = t_stop - t_start
            pred = mc_predict_from_logits(model, tok, item, n_e=t_start, n_t=n_t_val,
                                           device=DEVICE)
            if pred == label:
                correct[cname] += 1

        # C7: persample_variable — variable n_t selection (4, 6, 8)
        t_start_v, t_stop_v = select_window_variable_nt(cos_res, nt_candidates=(4, 6, 8))
        selected_windows["persample_variable"].append((t_start_v, t_stop_v))
        n_t_v = t_stop_v - t_start_v
        pred = mc_predict_from_logits(model, tok, item, n_e=t_start_v, n_t=n_t_v,
                                       device=DEVICE)
        if pred == label:
            correct["persample_variable"] += 1

        # C8: onset_cos8 — onset criterion with n_t=8
        t_start_o, t_stop_o = select_window_onset(cos_res, threshold=0.28, n_t=8)
        selected_windows["onset_cos8"].append((t_start_o, t_stop_o))
        n_t_o = t_stop_o - t_start_o
        pred = mc_predict_from_logits(model, tok, item, n_e=t_start_o, n_t=n_t_o,
                                       device=DEVICE)
        if pred == label:
            correct["onset_cos8"] += 1

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            accs = {c: correct[c] / (i + 1) for c in cond_names}
            print(f"  [{bench}] {i+1}/{total}  {elapsed:.0f}s | "
                  f"base={accs['baseline']:.3f}  oracle={accs['oracle']:.3f}  "
                  f"glb6={accs['global_cos6']:.3f}  ps6={accs['persample_cos6']:.3f}  "
                  f"psvar={accs['persample_variable']:.3f}  onset={accs['onset_cos8']:.3f}")

        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    accs = {c: correct[c] / total for c in cond_names}

    # 统计每样本选窗分布
    window_stats = {}
    for cname in selected_windows:
        ws = selected_windows[cname]
        starts = [w[0] for w in ws]
        window_stats[cname] = {
            "t_start_mean": float(np.mean(starts)),
            "t_start_std":  float(np.std(starts)),
            "t_start_hist": {str(s): starts.count(s) for s in sorted(set(starts))},
        }

    return {
        "benchmark":     bench,
        "n":             total,
        "elapsed_s":     elapsed,
        "accuracies":    accs,
        "oracle_window": oracle,
        "global_cos6_window": glb6,
        "global_cos8_window": glb8,
        "window_stats":  window_stats,
    }


# ─── 绘图 ─────────────────────────────────────────────────────────────────────
COND_COLORS = {
    "baseline":            "#9E9E9E",
    "oracle":              "#2196F3",
    "global_cos6":         "#4CAF50",
    "global_cos8":         "#8BC34A",
    "persample_cos6":      "#FF9800",
    "persample_cos8":      "#FF5722",
    "persample_cos10":     "#E91E63",
    "persample_variable":  "#9C27B0",
    "onset_cos8":          "#00BCD4",
}
COND_LABELS = {
    "baseline":            "Baseline",
    "oracle":              "Oracle (R30 best)",
    "global_cos6":         "Global-cos6 (n_t=6)",
    "global_cos8":         "Global-cos8 (n_t=8)",
    "persample_cos6":      "PerSample-cos6",
    "persample_cos8":      "PerSample-cos8",
    "persample_cos10":     "PerSample-cos10",
    "persample_variable":  "PerSample-variable",
    "onset_cos8":          "Onset-cos8",
}


def plot_summary_bar(all_results: dict[str, dict]):
    cond_names = ["baseline", "oracle", "global_cos6", "global_cos8",
                  "persample_cos6", "persample_cos8", "persample_cos10"]
    benches = list(all_results.keys())

    fig, axes = plt.subplots(1, len(benches), figsize=(5 * len(benches), 5))
    if len(benches) == 1:
        axes = [axes]

    for ax, bench in zip(axes, benches):
        res = all_results[bench]
        accs = res["accuracies"]
        baseline_acc = accs["baseline"]
        oracle_acc   = accs["oracle"]

        bars = [accs[c] for c in cond_names]
        colors = [COND_COLORS[c] for c in cond_names]
        x = np.arange(len(cond_names))
        brs = ax.bar(x, bars, color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)

        # 标注数值
        for bar, v in zip(brs, bars):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

        # 参考线
        ax.axhline(baseline_acc, color="grey",  linestyle="--", linewidth=1, alpha=0.6)
        ax.axhline(oracle_acc,   color="#2196F3", linestyle=":",  linewidth=1.5, alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([COND_LABELS[c].replace(" ", "\n") for c in cond_names],
                           fontsize=6, rotation=0)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{bench}\n(N={res['n']}, t={res['elapsed_s']:.0f}s)")
        ax.set_ylim(0, min(1.0, oracle_acc * 1.25 + 0.05))

    fig.suptitle("R37: Signal-Guided ETD vs Baseline vs Oracle", fontsize=11, y=1.02)
    plt.tight_layout()
    fpath = FIGURES_DIR / "summary_bar.png"
    plt.savefig(fpath, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved → {fpath}")


def plot_window_distribution(bench: str, result: dict):
    """绘制每样本选窗的 t_start 直方图."""
    ws = result.get("window_stats", {})
    if not ws:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, cname in zip(axes, ["persample_cos6", "persample_cos8", "persample_cos10"]):
        if cname not in ws:
            ax.set_visible(False)
            continue
        hist = ws[cname]["t_start_hist"]
        starts = [int(k) for k in hist]
        counts = [hist[k] for k in hist]
        ax.bar(starts, counts, color=COND_COLORS[cname], edgecolor="black", linewidth=0.5)
        mean_s = ws[cname]["t_start_mean"]
        oracle_s = result["oracle_window"][0]
        ax.axvline(mean_s,   color="orange", linestyle="--", linewidth=1.5, label=f"mean={mean_s:.1f}")
        ax.axvline(oracle_s, color="blue",   linestyle=":",  linewidth=1.5, label=f"oracle t_s={oracle_s}")
        ax.set_xlabel("Selected t_start")
        ax.set_ylabel("Count")
        ax.set_title(f"{bench} | {COND_LABELS[cname]}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    fpath = FIGURES_DIR / f"{bench.replace('/', '_')}_window_dist.png"
    plt.savefig(fpath, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved → {fpath}")


def plot_cos_res_profile(bench: str, oracle_win: tuple, glb6_win: tuple, glb8_win: tuple):
    """绘制该 benchmark 的 R36 cos_res 曲线及各窗口标注."""
    cos_res = R36_COS_RES.get(bench, {})
    if not cos_res:
        return

    layers = sorted(cos_res)
    vals   = [cos_res[l] for l in layers]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(layers, vals, "ko-", lw=1.5, ms=5, label="cos_res (R36 aggregate)")
    ax.axhline(0, color="grey", lw=0.8, linestyle="--")

    # 标注窗口
    def shade_window(win, color, label):
        ax.axvspan(win[0], win[1], alpha=0.15, color=color, label=label)
        ax.axvline(win[0], color=color, linestyle="--", lw=1)
        ax.axvline(win[1], color=color, linestyle="--", lw=1)

    shade_window(oracle_win, "#2196F3", f"Oracle {oracle_win}")
    shade_window(glb6_win,   "#4CAF50", f"Global-cos6 {glb6_win}")
    shade_window(glb8_win,   "#8BC34A", f"Global-cos8 {glb8_win}")

    ax.set_xlabel("Layer")
    ax.set_ylabel("cos(Term1, Δh)")
    ax.set_title(f"{bench}: R36 cos_res profile + Derived Windows")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 36)

    plt.tight_layout()
    fpath = FIGURES_DIR / f"{bench.replace('/', '_')}_cos_res_windows.png"
    plt.savefig(fpath, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved → {fpath}")


# ─── 主流程 ───────────────────────────────────────────────────────────────────
def main():
    t_total = time.time()
    print("=" * 60)
    print("R37: Signal-Guided ETD Layer Selection")
    print(f"N_SAMPLES={N_SAMPLES}  PROBE_LAYERS={PROBE_LAYERS}")
    print("=" * 60)

    # 预绘制 cos_res 曲线
    for bench in ["MMLU-HS-Math", "GPQA-Diamond", "AGIEval-Gaokao-MathQA"]:
        plot_cos_res_profile(
            bench,
            ORACLE_WINDOWS[bench],
            GLOBAL_WINDOWS[bench][6],
            GLOBAL_WINDOWS[bench][8],
        )

    tok, model, n_layers = load_model()

    all_results: dict[str, dict] = {}

    for bench, loader in BENCH_LOADERS.items():
        print(f"\n{'─'*50}")
        print(f"Benchmark: {bench}")
        print(f"  Oracle:     {ORACLE_WINDOWS[bench]}")
        print(f"  Global-cos6:{GLOBAL_WINDOWS[bench][6]}")
        print(f"  Global-cos8:{GLOBAL_WINDOWS[bench][8]}")
        print(f"{'─'*50}")

        try:
            items = loader(N_SAMPLES)
        except Exception as e:
            print(f"  [ERROR] Failed to load {bench}: {e}")
            continue
        if not items:
            print(f"  [SKIP] {bench} returned empty")
            continue

        items = items[:N_SAMPLES]
        print(f"  Loaded {len(items)} samples")

        result = evaluate_benchmark(bench, items, model, tok, n_layers)
        all_results[bench] = result

        # 打印摘要
        accs = result["accuracies"]
        print(f"\n  === {bench} Results ===")
        for cname, acc in accs.items():
            delta = acc - accs["baseline"]
            oracle_frac = acc / accs["oracle"] if accs["oracle"] > 0 else 0
            print(f"    {cname:20s}: {acc:.4f}  (Δbaseline={delta:+.4f}, "
                  f"{oracle_frac:.2%} of oracle)")

        plot_window_distribution(bench, result)

        # 保存中间结果
        with open(RESULTS_DIR / "r37_signal_loop_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    # 最终绘图
    if all_results:
        plot_summary_bar(all_results)

    # 假设检验
    print("\n" + "=" * 60)
    print("假设验证摘要")
    print("=" * 60)
    for bench, res in all_results.items():
        accs = res["accuracies"]
        base = accs["baseline"]
        oracle = accs["oracle"]
        print(f"\n{bench}:")
        print(f"  Baseline={base:.4f}, Oracle={oracle:.4f}, Δ={oracle-base:+.4f}")

        # H1: global_cos6 > baseline
        h1 = accs["global_cos6"] > base
        print(f"  H1 (global_cos6 > baseline): {'✓' if h1 else '✗'}  "
              f"{accs['global_cos6']:.4f} vs {base:.4f}")

        # H2: persample best ≥ global best
        best_ps   = max(accs["persample_cos6"], accs["persample_cos8"], accs["persample_cos10"])
        best_glb  = max(accs["global_cos6"], accs["global_cos8"])
        h2 = best_ps >= best_glb
        print(f"  H2 (best_persample ≥ best_global): {'✓' if h2 else '✗'}  "
              f"{best_ps:.4f} vs {best_glb:.4f}")

        # H3: max signal-guided ≥ 0.9 × oracle
        best_signal = max(accs[c] for c in accs if c not in ["baseline", "oracle"])
        h3 = best_signal >= 0.9 * oracle if oracle > 0 else False
        print(f"  H3 (best_signal ≥ 90%% oracle): {'✓' if h3 else '✗'}  "
              f"best_signal={best_signal:.4f}, 90%%oracle={0.9*oracle:.4f}")

        # 额外: 列出所有方法 vs baseline
        for cname in ["persample_variable", "onset_cos8"]:
            delta = accs[cname] - base
            print(f"  [{cname}] acc={accs[cname]:.4f}  Δbaseline={delta:+.4f}")

    # 保存最终结果
    with open(RESULTS_DIR / "r37_signal_loop_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {RESULTS_DIR / 'r37_signal_loop_results.json'}")

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"R37 完成！总耗时 {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
