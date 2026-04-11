#!/usr/bin/env python3
"""
exp_round4_main.py  ─  第四轮综合实验（2026-04-06）

实验 A3: 自主层选择
  - 用 50 条 BoolQ/ARC 样本计算 delta_norm（k=1→k=2 的隐藏状态变化量）
  - delta_norm 最小的 t_end → 选为最优 T 块边界
  - 用 100 条样本在 4 个基准上验证选出的配置（k=2 和 k=3）

实验 A4: k=3 + R2 阻尼
  - 验证有效步长理论在 k=3 时是否成立
  - H1: 每次迭代步长 ≈ 6（alpha 不变）
  - H2: 总步长 ≈ 12，单次步长 = 12/(k×n_t)（随 k 缩减）
  - H3: 更小步长（探索）

实验 LP: 层属性剖面（面向零样本选层的基础研究）
  - 用 64 条样本的 baseline 单次前向，收集 36 层统计量
  - 指标: step_size, norm, cosine_sim, token_differentiation
  - 对比好 t_start (8,10) 与坏 t_start (5,6) 的层特征

基准: BoolQ + ARC-Challenge + CommonsenseQA + TruthfulQA-MC
样本规模: 选层=50, 评估=100, 剖面=64
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─── 路径配置 ─────────────────────────────────────────────────────────────────
MODEL_PATH    = "/root/autodl-tmp/model_qwen"
ETD_PATH      = Path("/root/autodl-tmp/loop_layer/ETD")
SWEEP_PATH    = Path("/root/autodl-tmp/loop_layer/ETD_layer_sweep")
RESULTS_DIR   = Path("/root/autodl-tmp/loop_layer/experiments/results")
FIGURES_DIR   = Path("/root/autodl-tmp/loop_layer/experiments/figures")
MEMORY_BANK   = Path("/root/self-evolving-researcher/memory-bank")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

for _p in [str(ETD_PATH), str(SWEEP_PATH)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from etd_forward import (  # noqa: E402
    etd_forward_logits,
    baseline_forward_logits,
    loglikelihood_continuation,
    _adaptive_alpha,
    _prepare_position_ids,
)
from data_cache import load_boolq_examples, load_arc_examples  # noqa: E402

# ─── 实验参数 ─────────────────────────────────────────────────────────────────
N_SELECT  = 50   # A3 选层阶段样本数
N_EVAL    = 100  # 准确率评估样本数
N_PROFILE = 64   # 层属性剖面样本数

# A3: 固定 t_start=8 (n_e=8)，扫描 n_t
A3_N_E = 8
A3_N_T_CANDIDATES = [6, 8, 10, 12, 14, 16, 18]  # 对应 t_end 14,16,18,20,22,24,26
A3_K_VALUES = [2, 3]

# A4: 最优配置 (n_e=10, n_t=11, 即 t_end=21) 和通用最优 (n_e=8, n_t=14)
A4_CONFIGS = [
    ("opt_10_21", 10, 11),  # t_start=10, t_end=21, n_t=11
    ("opt_8_22",   8, 14),  # t_start=8,  t_end=22, n_t=14
]
# k=2 已知最优 alpha: 6/n_t, k=3 测试四种 alpha 假设
# H1: alpha_k3 = 6/n_t (每次步长=6, 与k=2相同)
# H2: alpha_k3 = 4/n_t (总步长=12, k=3时单次步长=4)
# H3: alpha_k3 = 3/n_t (更小步长)
# H4: alpha_k3 = 2/n_t (更激进压缩)

# LP: 层属性分析 - 比较的 t_start 值
LP_GOOD_T_STARTS = [8, 10]   # 已知好的
LP_BAD_T_STARTS  = [5, 6]    # 已知危险的
LP_MID_T_STARTS  = [12, 14]  # 中等

BOOLQ_BASELINE = 0.862
ARC_BASELINE   = 0.532
BASELINES = {"boolq": 0.862, "arc": 0.532, "commonsenseqa": None, "truthfulqa": None}

# ─── 数据加载 ─────────────────────────────────────────────────────────────────

def load_commonsenseqa(limit: int) -> list[tuple[str, list[str], int]]:
    """CommonsenseQA: 5选项常识推理。split='validation'（1221条）"""
    from datasets import load_dataset
    ds = load_dataset("tau/commonsense_qa")["validation"]
    out: list[tuple[str, list[str], int]] = []
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    for ex in ds:
        if len(out) >= limit:
            break
        q = ex["question"].strip()
        prefix = f"Question: {q}\nAnswer:"
        conts = [f" {t}" for t in ex["choices"]["text"]]
        key = ex["answerKey"]
        if key not in label_map:
            continue
        out.append((prefix, conts, label_map[key]))
    return out


def load_truthfulqa(limit: int) -> list[tuple[str, list[str], int]]:
    """TruthfulQA MC1: 正确答案唯一的多选版本（validation，817条）。"""
    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice")["validation"]
    out: list[tuple[str, list[str], int]] = []
    for ex in ds:
        if len(out) >= limit:
            break
        q = ex["question"].strip()
        mc1 = ex["mc1_targets"]
        choices = mc1["choices"]
        labels = mc1["labels"]
        if 1 not in labels:
            continue
        label = labels.index(1)
        prefix = f"Question: {q}\nAnswer:"
        conts = [f" {c}" for c in choices]
        out.append((prefix, conts, label))
    return out


def load_all_benchmarks(limit: int) -> dict[str, list]:
    """加载所有可用基准数据集，返回字典。"""
    data: dict[str, list] = {}
    print(f"  加载 BoolQ (validation, {limit}条)…")
    data["boolq"] = load_boolq_examples("validation", limit)
    print(f"  加载 ARC-Challenge (test, {limit}条)…")
    data["arc"] = load_arc_examples("test", limit)
    try:
        print(f"  加载 CommonsenseQA (validation, {limit}条)…")
        data["commonsenseqa"] = load_commonsenseqa(limit)
        print(f"    → {len(data['commonsenseqa'])} 条")
    except Exception as e:
        print(f"    CommonsenseQA 加载失败: {e}")
    try:
        print(f"  加载 TruthfulQA-MC1 (validation, {limit}条)…")
        data["truthfulqa"] = load_truthfulqa(limit)
        print(f"    → {len(data['truthfulqa'])} 条")
    except Exception as e:
        print(f"    TruthfulQA 加载失败: {e}")
    return data


# ─── ETD 工具函数 ─────────────────────────────────────────────────────────────

@torch.inference_mode()
def _get_T_output(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    n_e: int,
    n_t: int,
    k: int,
    alpha: float,
) -> torch.Tensor:
    """
    运行 E 块 + T 块 k 次（含 R2 阻尼），返回 T 块后的隐藏状态。
    不运行 D 块和 lm_head，比完整 forward 更快。
    用于 A3 的 delta_norm 计算。
    """
    from transformers.masking_utils import create_causal_mask
    base = model.model
    cfg = model.config
    device = input_ids.device

    inputs_embeds = base.embed_tokens(input_ids)
    batch, seq_len = inputs_embeds.shape[:2]
    position_ids = _prepare_position_ids(attention_mask, 0, batch, seq_len, device)

    past_key_values = None
    mask_kwargs = {
        "config": cfg,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
    }
    causal_mask_map = {"full_attention": create_causal_mask(**mask_kwargs)}
    if getattr(base, "has_sliding_layers", False):
        from transformers.masking_utils import create_sliding_window_causal_mask
        causal_mask_map["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    position_embeddings = base.rotary_emb(inputs_embeds, position_ids)
    hidden_states = inputs_embeds

    def _run(idx: int, hs: torch.Tensor) -> torch.Tensor:
        ltype = cfg.layer_types[idx]
        out = base.layers[idx](
            hs,
            attention_mask=causal_mask_map[ltype],
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=False,
            position_embeddings=position_embeddings,
        )
        return out[0] if isinstance(out, tuple) else out

    # E 块
    for i in range(n_e):
        hidden_states = _run(i, hidden_states)

    # T 块，收集每次迭代后的状态
    iter_states: list[torch.Tensor] = [hidden_states.clone()]
    use_damping = alpha < 1.0
    for _ in range(k):
        h_prev = hidden_states
        for i in range(n_e, n_e + n_t):
            hidden_states = _run(i, hidden_states)
        if use_damping:
            hidden_states = alpha * hidden_states + (1.0 - alpha) * h_prev
        iter_states.append(hidden_states.clone())

    return iter_states  # list of k+1 tensors: [h_e, h_1, h_2, ...]


@torch.inference_mode()
def eval_mc(
    model,
    tokenizer,
    examples: list[tuple[str, list[str], int]],
    n_e: int,
    n_t: int,
    k: int,
    alpha: float,
    device: torch.device,
    desc: str = "",
) -> float:
    """多选题准确率评估（ETD 模式）。"""
    correct = 0
    for prefix, conts, label in tqdm(examples, desc=desc, leave=False):
        plen = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
        scores = []
        for cont in conts:
            enc = tokenizer(prefix + cont, return_tensors="pt", add_special_tokens=False)
            ids = enc["input_ids"].to(device)
            attn = enc.get("attention_mask")
            if attn is not None:
                attn = attn.to(device)
            logits = etd_forward_logits(model, ids, attn, n_e, n_t, k, alpha=alpha)
            scores.append(loglikelihood_continuation(logits, ids, plen))
        if max(range(len(scores)), key=lambda i: scores[i]) == label:
            correct += 1
    return correct / len(examples)


@torch.inference_mode()
def eval_mc_baseline(
    model,
    tokenizer,
    examples: list[tuple[str, list[str], int]],
    device: torch.device,
    desc: str = "",
) -> float:
    """多选题准确率评估（baseline 标准 forward）。"""
    correct = 0
    for prefix, conts, label in tqdm(examples, desc=desc, leave=False):
        plen = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
        scores = []
        for cont in conts:
            enc = tokenizer(prefix + cont, return_tensors="pt", add_special_tokens=False)
            ids = enc["input_ids"].to(device)
            attn = enc.get("attention_mask")
            if attn is not None:
                attn = attn.to(device)
            logits = baseline_forward_logits(model, ids, attn)
            scores.append(loglikelihood_continuation(logits, ids, plen))
        if max(range(len(scores)), key=lambda i: scores[i]) == label:
            correct += 1
    return correct / len(examples)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    t_global = time.time()
    print("=" * 70)
    print("第四轮综合实验: A3(自主层选择) + A4(k=3 R2) + LP(层属性剖面)")
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n加载模型: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"  完成。设备={device}，层数={n_layers}\n")

    print("加载数据集…")
    all_data = load_all_benchmarks(N_EVAL)
    # 用 N_SELECT 条做选层，取各基准的前 N_SELECT 条
    select_data = {k: v[:N_SELECT] for k, v in all_data.items()}
    benchmark_names = list(all_data.keys())
    print(f"  可用基准: {benchmark_names}\n")

    results: dict = {
        "a3_layer_select": {},
        "a4_k3_r2": {},
        "layer_profile": {},
        "baselines": {},
    }

    # ──────────────────────────────────────────────────────────────────────────
    # BASELINE 评估（每个基准的参照值）
    # ──────────────────────────────────────────────────────────────────────────
    print("=" * 50)
    print("BASELINE 评估")
    print("=" * 50)
    for bname in benchmark_names:
        ex = all_data[bname][:N_EVAL]
        acc = eval_mc_baseline(model, tokenizer, ex, device, desc=f"baseline/{bname}")
        results["baselines"][bname] = acc
        print(f"  {bname:<18}: {acc:.4f}")
    BASELINES.update(results["baselines"])
    print()

    # ──────────────────────────────────────────────────────────────────────────
    # 实验 A3：自主层选择
    # Phase 1: delta_norm 扫描（N_SELECT 条 BoolQ + ARC）
    # ──────────────────────────────────────────────────────────────────────────
    print("=" * 50)
    print("实验 A3 Phase 1: delta_norm 扫描")
    print(f"  固定 n_e={A3_N_E}, 扫描 n_t={A3_N_T_CANDIDATES}")
    print("=" * 50)

    a3_delta_norms: dict[str, dict[int, float]] = {bname: {} for bname in ["boolq", "arc"]}
    a3_cos_sims:   dict[str, dict[int, float]] = {bname: {} for bname in ["boolq", "arc"]}

    for n_t in A3_N_T_CANDIDATES:
        alpha = _adaptive_alpha(n_t)  # min(1.0, 6.0/n_t)
        t_end_label = A3_N_E + n_t   # 对应的 t_end（概念上的层编号）

        for task in ["boolq", "arc"]:
            examples = select_data[task]
            delta_list, cos_list = [], []

            for prefix, conts, _ in examples:
                # 用第一个 continuation 作为代表（计算层内轨迹）
                full = prefix + conts[0]
                enc = tokenizer(full, return_tensors="pt", add_special_tokens=False)
                ids = enc["input_ids"].to(device)
                attn = enc.get("attention_mask")
                if attn is not None:
                    attn = attn.to(device)

                # 收集 k=2 迭代的隐藏状态序列 [h_e, h_1, h_2]
                states = _get_T_output(model, ids, attn, A3_N_E, n_t, k=2, alpha=alpha)
                h_e, h_1, h_2 = states[0], states[1], states[2]

                # 取最后一个 token 的表示
                h_e_last = h_e[0, -1]
                h_1_last = h_1[0, -1]
                h_2_last = h_2[0, -1]

                # delta_norm: k=2 vs k=1 的变化量（相对）
                diff_21 = h_2_last - h_1_last
                diff_10 = h_1_last - h_e_last
                dn = diff_21.norm().item() / (h_1_last.norm().item() + 1e-8)
                delta_list.append(dn)

                # cos_sim: (h2-h1) 与 (h1-h0) 的方向一致性
                cs = F.cosine_similarity(diff_21.unsqueeze(0), diff_10.unsqueeze(0)).item()
                cos_list.append(cs)

            mean_dn = float(np.mean(delta_list))
            mean_cs = float(np.mean(cos_list))
            a3_delta_norms[task][n_t] = mean_dn
            a3_cos_sims[task][n_t] = mean_cs
            print(f"  n_t={n_t:2d} α={alpha:.3f}  [{task}]  delta_norm={mean_dn:.4f}  cos_sim={mean_cs:.4f}")

    # 选层：用 BoolQ 和 ARC 的平均 delta_norm
    avg_delta = {n_t: (a3_delta_norms["boolq"][n_t] + a3_delta_norms["arc"][n_t]) / 2
                 for n_t in A3_N_T_CANDIDATES}
    selected_n_t = min(avg_delta, key=avg_delta.get)
    selected_alpha = _adaptive_alpha(selected_n_t)
    print(f"\n  >>> 自主选层结果: n_t={selected_n_t}, alpha={selected_alpha:.3f}")
    print(f"      对应 t_end={A3_N_E + selected_n_t} (T块: 层{A3_N_E}~层{A3_N_E + selected_n_t - 1})")
    print(f"      有效步长 = {selected_alpha:.3f} × {selected_n_t} = {selected_alpha * selected_n_t:.2f}")

    results["a3_layer_select"]["delta_norms_boolq"] = {str(k): v for k, v in a3_delta_norms["boolq"].items()}
    results["a3_layer_select"]["delta_norms_arc"]   = {str(k): v for k, v in a3_delta_norms["arc"].items()}
    results["a3_layer_select"]["cos_sims_boolq"]    = {str(k): v for k, v in a3_cos_sims["boolq"].items()}
    results["a3_layer_select"]["cos_sims_arc"]      = {str(k): v for k, v in a3_cos_sims["arc"].items()}
    results["a3_layer_select"]["selected_n_t"]      = selected_n_t
    results["a3_layer_select"]["selected_alpha"]    = selected_alpha
    results["a3_layer_select"]["avg_delta"]         = {str(k): v for k, v in avg_delta.items()}

    # ──────────────────────────────────────────────────────────────────────────
    # 实验 A3 Phase 2: 验证选出配置（k=2 和 k=3）
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n实验 A3 Phase 2: 验证选出配置 (n_e={A3_N_E}, n_t={selected_n_t}, alpha={selected_alpha:.3f})")
    a3_validation: dict = {}
    for k_val in A3_K_VALUES:
        a3_validation[k_val] = {}
        for bname in benchmark_names:
            ex = all_data[bname][:N_EVAL]
            acc = eval_mc(model, tokenizer, ex, A3_N_E, selected_n_t, k_val, selected_alpha, device,
                          desc=f"A3-k{k_val}/{bname}")
            delta = acc - BASELINES.get(bname, 0)
            a3_validation[k_val][bname] = {"acc": acc, "delta": delta}
            print(f"  k={k_val}  {bname:<18}: {acc:.4f}  (Δ={delta:+.4f})")

    # 同时测试 known champion (n_e=8, n_t=14) 作为参照
    print("\n  参照: 已知最优 (n_e=8, n_t=14, k=2, alpha=auto)")
    a3_champion: dict = {}
    for bname in benchmark_names:
        ex = all_data[bname][:N_EVAL]
        acc = eval_mc(model, tokenizer, ex, 8, 14, 2, _adaptive_alpha(14), device,
                      desc=f"champion/{bname}")
        delta = acc - BASELINES.get(bname, 0)
        a3_champion[bname] = {"acc": acc, "delta": delta}
        print(f"  champion  {bname:<18}: {acc:.4f}  (Δ={delta:+.4f})")

    results["a3_layer_select"]["validation"] = a3_validation
    results["a3_layer_select"]["champion"]   = a3_champion

    # ── 绘图 A3 ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("A3: 自主层选择实验", fontsize=14, fontweight="bold")

    # 左图: delta_norm vs n_t（BoolQ & ARC）
    ax = axes[0]
    n_ts = A3_N_T_CANDIDATES
    ax.plot(n_ts, [a3_delta_norms["boolq"][n] for n in n_ts], "o-", color="#2196F3", label="BoolQ")
    ax.plot(n_ts, [a3_delta_norms["arc"][n]   for n in n_ts], "s-", color="#FF5722", label="ARC")
    ax.plot(n_ts, [avg_delta[n] for n in n_ts], "^--", color="#4CAF50", label="平均", linewidth=2)
    ax.axvline(selected_n_t, color="gray", linestyle=":", alpha=0.7, label=f"选出 n_t={selected_n_t}")
    ax.set_xlabel("n_t（T块层数）")
    ax.set_ylabel("delta_norm (k=2 vs k=1)")
    ax.set_title("delta_norm vs n_t\n（越小=迭代越稳定）")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_ts)

    # 中图: cos_sim vs n_t
    ax = axes[1]
    ax.plot(n_ts, [a3_cos_sims["boolq"][n] for n in n_ts], "o-", color="#2196F3", label="BoolQ")
    ax.plot(n_ts, [a3_cos_sims["arc"][n]   for n in n_ts], "s-", color="#FF5722", label="ARC")
    ax.axvline(selected_n_t, color="gray", linestyle=":", alpha=0.7)
    ax.axhline(0, color="black", linestyle="-", alpha=0.2)
    ax.set_xlabel("n_t（T块层数）")
    ax.set_ylabel("cos_sim (方向一致性)")
    ax.set_title("cos_sim vs n_t\n（cos>0=同向，<0=反向）")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_ts)

    # 右图: 选出配置 vs champion 的准确率对比
    ax = axes[2]
    bnames_short = [b[:8] for b in benchmark_names]
    x = np.arange(len(benchmark_names))
    width = 0.2
    baseline_vals = [BASELINES.get(b, 0) for b in benchmark_names]
    ax.bar(x - width, baseline_vals, width, label="Baseline", color="#9E9E9E", alpha=0.7)
    if 2 in a3_validation:
        vals_k2 = [a3_validation[2][b]["acc"] for b in benchmark_names]
        ax.bar(x, vals_k2, width, label=f"选层 k=2 (n_t={selected_n_t})", color="#2196F3", alpha=0.8)
    if 3 in a3_validation:
        vals_k3 = [a3_validation[3][b]["acc"] for b in benchmark_names]
        ax.bar(x + width, vals_k3, width, label=f"选层 k=3 (n_t={selected_n_t})", color="#FF9800", alpha=0.8)
    champ_vals = [a3_champion[b]["acc"] for b in benchmark_names]
    ax.plot(x + width * 2 + 0.05, champ_vals, "D", color="#4CAF50", markersize=8, label="Champion (8,22)")
    ax.set_xticks(x)
    ax.set_xticklabels(bnames_short, fontsize=8)
    ax.set_ylabel("准确率")
    ax.set_title("验证：选出配置 vs Champion vs Baseline")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "a3_layer_selection.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  图表已保存: {FIGURES_DIR}/a3_layer_selection.png")

    # ──────────────────────────────────────────────────────────────────────────
    # 实验 A4：k=3 + R2 阻尼
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("实验 A4: k=3 + R2 阻尼（有效步长理论推广）")
    print("=" * 50)

    # 每个假设对应的 alpha（以 n_t 为参数）
    def a4_alphas(n_t: int) -> dict[str, float]:
        return {
            "H0_k2": min(1.0, 6.0 / n_t),           # k=2 baseline (已知最优)
            "H1_k3": min(1.0, 6.0 / n_t),           # k=3, 每步 step=6 (同k=2)
            "H2_k3": min(1.0, 4.0 / n_t),           # k=3, 总step=12 → 单步=4
            "H3_k3": min(1.0, 3.0 / n_t),           # k=3, 单步=3
            "H4_k3": min(1.0, 2.0 / n_t),           # k=3, 单步=2
        }

    a4_results: dict = {}
    eval_tasks_a4 = ["boolq", "arc"]  # A4 只用2个任务节省时间

    for cfg_name, n_e, n_t in A4_CONFIGS:
        a4_results[cfg_name] = {}
        alphas = a4_alphas(n_t)
        print(f"\n  配置 {cfg_name}: n_e={n_e}, n_t={n_t}")
        print(f"  {'假设':<10} {'k':<4} {'alpha':>6}  {'eff_step':>8}", end="")
        for t in eval_tasks_a4:
            print(f"  {t[:6]:>8}", end="")
        print()
        print("  " + "-" * 65)

        for hyp, alpha_val in alphas.items():
            k_val = 2 if hyp == "H0_k2" else 3
            eff_step = alpha_val * n_t
            a4_results[cfg_name][hyp] = {"alpha": alpha_val, "k": k_val, "eff_step": eff_step}
            line = f"  {hyp:<10} {k_val:<4} {alpha_val:>6.3f}  {eff_step:>8.2f}"
            for task in eval_tasks_a4:
                ex = all_data[task][:N_EVAL]
                acc = eval_mc(model, tokenizer, ex, n_e, n_t, k_val, alpha_val, device,
                              desc=f"A4/{cfg_name}/{hyp}")
                delta = acc - BASELINES.get(task, 0)
                a4_results[cfg_name][hyp][task] = {"acc": acc, "delta": delta}
                line += f"  {acc:.4f}({delta:+.3f})"
            print(line)

    results["a4_k3_r2"] = a4_results

    # ── 绘图 A4 ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("A4: k=3 + R2 阻尼（有效步长理论推广）", fontsize=14, fontweight="bold")

    for ax_idx, (cfg_name, n_e, n_t) in enumerate(A4_CONFIGS):
        alphas = a4_alphas(n_t)
        for task_idx, task in enumerate(eval_tasks_a4):
            ax = axes[ax_idx][task_idx]
            hyps_k2 = [h for h in alphas if "H0" in h]
            hyps_k3 = [h for h in alphas if "H0" not in h]

            # k=2 baseline（水平线）
            h0_acc = a4_results[cfg_name]["H0_k2"][task]["acc"]
            ax.axhline(h0_acc, color="#2196F3", linestyle="-", linewidth=2,
                       label=f"k=2 α={alphas['H0_k2']:.3f} (已知最优)")
            ax.axhline(BASELINES.get(task, 0), color="#9E9E9E", linestyle="--",
                       linewidth=1.5, label="Baseline")

            # k=3 各假设
            k3_accs = [a4_results[cfg_name][h][task]["acc"] for h in hyps_k3]
            k3_steps = [alphas[h] * n_t for h in hyps_k3]
            k3_colors = ["#FF5722", "#FF9800", "#FFC107", "#4CAF50"]
            for i, (h, acc, step) in enumerate(zip(hyps_k3, k3_accs, k3_steps)):
                ax.scatter(step, acc, s=100, color=k3_colors[i], zorder=5,
                           label=f"k=3 {h}(step={step:.1f})")

            # k=2 已知最优点
            ax.scatter(alphas["H0_k2"] * n_t, h0_acc, s=150, marker="*",
                       color="#2196F3", zorder=6)

            ax.set_xlabel("有效步长 (alpha × n_t)")
            ax.set_ylabel(f"{task} 准确率")
            ax.set_title(f"{cfg_name} | {task}\n(n_e={n_e}, n_t={n_t})")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "a4_k3_r2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  图表已保存: {FIGURES_DIR}/a4_k3_r2.png")

    # ──────────────────────────────────────────────────────────────────────────
    # 实验 LP：层属性剖面（面向零样本选层）
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("实验 LP: 层属性剖面（36层统计量）")
    print("=" * 50)

    # 收集 baseline forward 的所有层隐藏状态
    boolq_ex_prof = load_boolq_examples("validation", N_PROFILE)
    arc_ex_prof   = load_arc_examples("test", N_PROFILE)

    # 每层统计：step_size, norm, cosine_sim, token_diff
    lp_stats_boolq: dict[str, list[float]] = {
        "step_size": [], "norm": [], "cosine_sim": [], "token_diff": []
    }
    lp_stats_arc = {k: [] for k in lp_stats_boolq}

    # 初始化：L+1 个列表（embedding + 36 layers）
    def collect_layer_stats(examples, stats_dict):
        all_step = [[] for _ in range(n_layers)]
        all_norm  = [[] for _ in range(n_layers)]
        all_csim  = [[] for _ in range(n_layers)]
        all_tdiff = [[] for _ in range(n_layers)]

        for prefix, conts, _ in tqdm(examples, desc="  层剖面", leave=False):
            full = prefix + conts[0]
            enc = tokenizer(full, return_tensors="pt", add_special_tokens=False)
            ids = enc["input_ids"].to(device)
            attn = enc.get("attention_mask")
            if attn is not None:
                attn = attn.to(device)

            with torch.inference_mode():
                out = model(
                    input_ids=ids,
                    attention_mask=attn,
                    output_hidden_states=True,
                    use_cache=False,
                )
            # out.hidden_states: tuple of (n_layers+1) tensors, shape (1, seq, dim)
            hs = out.hidden_states  # [embed, layer0_out, ..., layer35_out]

            for l_idx in range(n_layers):
                h_prev = hs[l_idx][0]     # (seq, dim), layer l_idx input (= embedding or prev layer out)
                h_curr = hs[l_idx + 1][0] # (seq, dim), layer l_idx output

                # step_size: 每层对隐藏状态的相对变化量
                diff = h_curr - h_prev
                s = diff.norm(dim=-1).mean().item() / (h_prev.norm(dim=-1).mean().item() + 1e-8)
                all_step[l_idx].append(s)

                # norm: 每层输出的平均范数
                n_val = h_curr.norm(dim=-1).mean().item()
                all_norm[l_idx].append(n_val)

                # cosine_sim: 与上一层的余弦相似度（均值）
                cs = F.cosine_similarity(h_curr, h_prev, dim=-1).mean().item()
                all_csim[l_idx].append(cs)

                # token_diff: 相邻 token 间的差异度（1 - 平均余弦相似度）
                # 衡量"每层是否让 token 表示更加区分"
                if h_curr.shape[0] > 1:
                    td = 1.0 - F.cosine_similarity(h_curr[:-1], h_curr[1:], dim=-1).mean().item()
                else:
                    td = 0.0
                all_tdiff[l_idx].append(td)

        stats_dict["step_size"] = [float(np.mean(v)) for v in all_step]
        stats_dict["norm"]      = [float(np.mean(v)) for v in all_norm]
        stats_dict["cosine_sim"]= [float(np.mean(v)) for v in all_csim]
        stats_dict["token_diff"]= [float(np.mean(v)) for v in all_tdiff]

    print("  收集 BoolQ 剖面…")
    collect_layer_stats(boolq_ex_prof, lp_stats_boolq)
    print("  收集 ARC 剖面…")
    collect_layer_stats(arc_ex_prof, lp_stats_arc)

    results["layer_profile"]["boolq"] = lp_stats_boolq
    results["layer_profile"]["arc"]   = lp_stats_arc

    # ── 绘图 LP ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("层属性剖面（Baseline 单次前向，36层）", fontsize=14, fontweight="bold")

    layer_indices = list(range(n_layers))
    metrics = ["step_size", "norm", "cosine_sim", "token_diff"]
    metric_labels = [
        "步长 (||Δh||/||h||)",
        "隐藏状态范数 ||h||",
        "层间余弦相似度 cos(h_l, h_{l-1})",
        "Token区分度 (1 - cos(h_i, h_{i+1}))"
    ]

    # 已知好/差区间注解
    good_spans = [(8, 10), (10, 12)]   # 已知好的 t_start
    bad_spans  = [(5, 7)]              # 已知危险的 t_start
    optimal_T  = [(8, 22)]             # 已知最优 T 块范围

    for idx, (metric, ylabel) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx // 2][idx % 2]
        vals_bq = lp_stats_boolq[metric]
        vals_ar = lp_stats_arc[metric]

        ax.plot(layer_indices, vals_bq, "-", color="#2196F3", linewidth=1.5, label="BoolQ")
        ax.plot(layer_indices, vals_ar, "-", color="#FF5722", linewidth=1.5, label="ARC")

        # 标注已知好/危险区间
        for span in good_spans:
            ax.axvspan(span[0] - 0.5, span[1] + 0.5, color="#4CAF50", alpha=0.15,
                       label=f"好 t_start" if span == good_spans[0] else "")
        for span in bad_spans:
            ax.axvspan(span[0] - 0.5, span[1] + 0.5, color="#FF5722", alpha=0.15,
                       label=f"危险 t_start")
        # 标注最优 T 块范围
        ax.axvspan(optimal_T[0][0] - 0.5, optimal_T[0][1] + 0.5, color="#9C27B0", alpha=0.08,
                   label="最优 T 块 (8~22)")

        ax.set_xlabel("层索引 (0-35)")
        ax.set_ylabel(ylabel)
        ax.set_title(metric)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, n_layers, 4))

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "lp_layer_profile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  图表已保存: {FIGURES_DIR}/lp_layer_profile.png")

    # ── 决策信号分析（Exp-4）：好/差 t_start 的层特征分布 ────────────────────
    print("\n  Exp-4: 好/差 t_start 的层特征比较")
    t_start_groups = {
        "bad(5-6)":  LP_BAD_T_STARTS,
        "good(8-10)": LP_GOOD_T_STARTS,
        "mid(12-14)": LP_MID_T_STARTS,
    }
    signal_results = {}
    for metric in ["step_size", "norm", "cosine_sim", "token_diff"]:
        signal_results[metric] = {}
        for gname, t_starts in t_start_groups.items():
            # 取对应层的 BoolQ+ARC 平均值
            vals_bq = [lp_stats_boolq[metric][l] for l in t_starts]
            vals_ar = [lp_stats_arc[metric][l]   for l in t_starts]
            avg = float(np.mean(vals_bq + vals_ar))
            signal_results[metric][gname] = avg
        print(f"  {metric:<18}:", {g: f"{v:.4f}" for g, v in signal_results[metric].items()})

    # 绘制决策信号对比图
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Exp-4: 好/差/中 t_start 对应层的统计特征\n（面向零样本层选择的判别信号）",
                 fontsize=13, fontweight="bold")

    colors = {"bad(5-6)": "#FF5722", "good(8-10)": "#4CAF50", "mid(12-14)": "#2196F3"}
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2][idx % 2]
        groups = list(signal_results[metric].keys())
        vals = [signal_results[metric][g] for g in groups]
        bars = ax.bar(groups, vals, color=[colors.get(g, "#9E9E9E") for g in groups], alpha=0.8, width=0.5)
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in vals], fontsize=9)
        ax.set_title(metric)
        ax.set_ylabel(metric_labels[idx])
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "lp_decision_signal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  图表已保存: {FIGURES_DIR}/lp_decision_signal.png")

    results["layer_profile"]["decision_signal"] = signal_results

    # ──────────────────────────────────────────────────────────────────────────
    # 综合结果汇总
    # ──────────────────────────────────────────────────────────────────────────
    total_time = time.time() - t_global
    results["meta"] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_sec": total_time,
        "n_select": N_SELECT,
        "n_eval": N_EVAL,
        "n_profile": N_PROFILE,
        "benchmarks": benchmark_names,
    }

    out_path = RESULTS_DIR / "round4_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n完整结果已保存: {out_path}")

    # ── 打印最终摘要 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("第四轮实验总结")
    print("=" * 70)
    print(f"总耗时: {total_time/60:.1f} 分钟")

    print("\n[A3] 自主层选择结果:")
    print(f"  选出 n_t={selected_n_t}，alpha={selected_alpha:.3f}，有效步长={selected_alpha * selected_n_t:.2f}")
    print(f"  选层准确率（k=2，{N_EVAL}样本）:")
    for bname in benchmark_names:
        r2 = a3_validation.get(2, {}).get(bname, {})
        bl = BASELINES.get(bname, 0)
        print(f"    {bname:<20}: {r2.get('acc', 0):.4f}  (Δ={r2.get('delta', 0):+.4f} vs baseline {bl:.4f})")

    print(f"\n[A4] k=3 最优假设验证:")
    for cfg_name, n_e, n_t in A4_CONFIGS:
        best_h = max(
            (h for h in a4_results[cfg_name] if "k3" in h.lower() or h != "H0_k2"),
            key=lambda h: sum(a4_results[cfg_name][h].get(t, {}).get("acc", 0) for t in eval_tasks_a4),
        )
        print(f"  {cfg_name}: 最优假设={best_h}, alpha={a4_results[cfg_name][best_h]['alpha']:.3f}, "
              f"有效步长={a4_results[cfg_name][best_h]['eff_step']:.2f}")

    print(f"\n[LP] 零样本选层信号预测（从层统计量推断好 t_start）:")
    for metric in metrics:
        sig = signal_results[metric]
        print(f"  {metric:<18}: bad={sig.get('bad(5-6)', 0):.4f}  "
              f"good={sig.get('good(8-10)', 0):.4f}  "
              f"mid={sig.get('mid(12-14)', 0):.4f}")

    print("\n生成的图表:")
    for p in sorted(FIGURES_DIR.glob("*.png")):
        print(f"  {p}")

    print("\n完成！")


if __name__ == "__main__":
    main()
