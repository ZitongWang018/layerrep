#!/usr/bin/env python3
"""
综合前置实验 A+B+C + R1 残差连接（一次性运行）

A: 隐层轨迹分析 —— T 块每次迭代中 hidden state 的 delta_norm, cos_sim
B: 逐层步长分析 —— 解释 t_end=24 全局 valley 的物理原因
C: 答案概率变化 —— 循环是"强化正确"还是"固化偏见"
R1: 全局残差连接 —— h_final = h_after_k_iters + h_e（最简单改进）

运行时间估算：A≈10min, B≈2min, C≈15min, R1≈25min → 总计约 50 分钟
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
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask

# ===== 路径配置 =====
MODEL_PATH = "/root/autodl-tmp/model_qwen"
ETD_SWEEP_PATH = Path("/root/autodl-tmp/loop_layer/ETD_layer_sweep")
ETD_PATH = Path("/root/autodl-tmp/loop_layer/ETD")
RESULTS_DIR = Path("/root/autodl-tmp/loop_layer/experiments/results")
MEMORY_BANK = Path("/root/self-evolving-researcher/memory-bank")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

for p in [str(ETD_SWEEP_PATH), str(ETD_PATH)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from data_cache import load_boolq_examples, load_arc_examples  # noqa: E402
from etd_forward import (  # noqa: E402
    baseline_forward_logits,
    etd_forward_logits,
    loglikelihood_continuation,
    predict_mc_choice,
)

# ===== 实验参数 =====
# 最优配置（BoolQ best: 0.870）
T_START_OPT, T_END_OPT = 10, 21
# 谷底配置（全局 valley）
T_START_VALLEY, T_END_VALLEY = 10, 24

K_MAX = 8    # 实验 A 最大迭代次数
N_A = 50     # 实验 A 每任务样本数
N_B = 64     # 实验 B 样本数
N_C = 100    # 实验 C 每任务样本数
N_R1 = 100   # R1 评估每任务样本数


# ===== 内部工具函数 =====

def _prepare_forward_context(model, input_ids, attention_mask):
    """构建 position_ids, causal_mask_mapping, position_embeddings"""
    base = model.model
    cfg = model.config
    device = input_ids.device
    inputs_embeds = base.embed_tokens(input_ids)
    batch, seq_len = inputs_embeds.shape[:2]

    if attention_mask is not None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
    else:
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    mask_kwargs = dict(
        config=cfg,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=position_ids,
    )
    cmm = {"full_attention": create_causal_mask(**mask_kwargs)}
    if getattr(base, "has_sliding_layers", False):
        from transformers.masking_utils import create_sliding_window_causal_mask
        cmm["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    pos_emb = base.rotary_emb(inputs_embeds, position_ids)
    return inputs_embeds, position_ids, cmm, pos_emb


def _layer_safe(layer, h, attn_mask, position_ids, pos_emb):
    """运行单层，兼容 tuple/tensor 返回值"""
    out = layer(
        h,
        attention_mask=attn_mask,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
        position_embeddings=pos_emb,
    )
    return out[0] if isinstance(out, tuple) else out


def _get_mask(cmm, cfg, layer_idx):
    return cmm[cfg.layer_types[layer_idx]]


# ===== R1 残差连接前向 =====

@torch.inference_mode()
def etd_forward_r1(model, input_ids, attention_mask, n_e, n_t, k):
    """R1：每次 T 迭代后加回 E 输出残差（注意：每次迭代都加，不是最后加一次）
    实验 R1-A：h_final = h_after_all_iters + h_e（最简：只在末尾加一次）
    """
    base = model.model
    cfg = model.config
    l_total = cfg.num_hidden_layers

    inputs_embeds, position_ids, cmm, pos_emb = _prepare_forward_context(
        model, input_ids, attention_mask
    )

    h = inputs_embeds
    for i in range(n_e):
        h = _layer_safe(base.layers[i], h, _get_mask(cmm, cfg, i), position_ids, pos_emb)

    h_e = h.clone()  # 保存 E 输出

    for _ in range(k):
        for i in range(n_e, n_e + n_t):
            h = _layer_safe(base.layers[i], h, _get_mask(cmm, cfg, i), position_ids, pos_emb)

    # R1-A: 末尾一次性加回（最简单）
    h = h + h_e

    for i in range(n_e + n_t, l_total):
        h = _layer_safe(base.layers[i], h, _get_mask(cmm, cfg, i), position_ids, pos_emb)

    h = base.norm(h)
    return model.lm_head(h)


def predict_mc_r1(model, tokenizer, prefix, conts, n_e, n_t, k, device):
    """使用 R1 残差连接的多选评估"""
    scores = []
    plen = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
    for cont in conts:
        full = prefix + cont
        enc = tokenizer(full, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask")
        attn = attn.to(device) if attn is not None else None
        logits = etd_forward_r1(model, input_ids, attn, n_e, n_t, k)
        scores.append(loglikelihood_continuation(logits, input_ids, plen))
    return max(range(len(scores)), key=lambda i: scores[i])


# ===== 实验 A：隐层轨迹分析 =====

@torch.inference_mode()
def _run_e_then_t_iters(model, input_ids, attention_mask, n_e, n_t, max_k):
    """返回 [h0, h1, ..., h_max_k] — 只保留最后 token 的 hidden state（降低内存）"""
    base = model.model
    cfg = model.config

    inputs_embeds, position_ids, cmm, pos_emb = _prepare_forward_context(
        model, input_ids, attention_mask
    )

    h = inputs_embeds
    for i in range(n_e):
        h = _layer_safe(base.layers[i], h, _get_mask(cmm, cfg, i), position_ids, pos_emb)

    hs_list = [h[:, -1, :].cpu().float()]  # h0 = E 输出（最后 token）

    for _ in range(max_k):
        for i in range(n_e, n_e + n_t):
            h = _layer_safe(base.layers[i], h, _get_mask(cmm, cfg, i), position_ids, pos_emb)
        hs_list.append(h[:, -1, :].cpu().float())  # h_k（最后 token）

    return hs_list  # list of (1, hidden_dim) tensors


def exp_A(model, tokenizer, device, boolq_ex, arc_ex):
    print("\n" + "=" * 60)
    print(f"实验 A：隐层轨迹分析 (t_start={T_START_OPT}, t_end={T_END_OPT}, k=1..{K_MAX})")
    print("=" * 60)

    n_e = T_START_OPT
    n_t = T_END_OPT - T_START_OPT + 1
    results = {}

    for task_name, examples in [("boolq", boolq_ex[:N_A]), ("arc", arc_ex[:N_A])]:
        delta_norms = [[] for _ in range(K_MAX)]
        cos_sims = [[] for _ in range(K_MAX)]

        for prefix, conts, label in tqdm(examples, desc=f"A-{task_name}"):
            enc = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)
            input_ids = enc["input_ids"].to(device)
            attn = enc.get("attention_mask")
            attn = attn.to(device) if attn is not None else None

            hs = _run_e_then_t_iters(model, input_ids, attn, n_e, n_t, K_MAX)
            # hs[0] = h0 (E output), hs[k] = T^k(h0)

            h0_norm = float(hs[0].norm().item())
            first_delta = (hs[1] - hs[0]).flatten()  # direction of first T step

            for k in range(1, K_MAX + 1):
                delta = hs[k] - hs[k - 1]
                dn = float(delta.norm().item()) / (h0_norm + 1e-8)
                delta_norms[k - 1].append(dn)

                if k == 1:
                    cos_sims[k - 1].append(1.0)
                else:
                    cs = float(F.cosine_similarity(
                        delta.flatten().unsqueeze(0),
                        first_delta.unsqueeze(0)
                    ).item())
                    cos_sims[k - 1].append(cs)

        results[task_name] = {
            "delta_norm_mean": [float(np.mean(v)) for v in delta_norms],
            "delta_norm_std": [float(np.std(v)) for v in delta_norms],
            "cos_sim_mean": [float(np.mean(v)) for v in cos_sims],
            "cos_sim_std": [float(np.std(v)) for v in cos_sims],
            "k_range": list(range(1, K_MAX + 1)),
        }

        print(f"\n  [{task_name}] (n={len(examples)}, n_e={n_e}, n_t={n_t})")
        print(f"  k    delta_norm(均值±std)    cos_sim(均值±std)")
        for k in range(K_MAX):
            dn_m = results[task_name]['delta_norm_mean'][k]
            dn_s = results[task_name]['delta_norm_std'][k]
            cs_m = results[task_name]['cos_sim_mean'][k]
            cs_s = results[task_name]['cos_sim_std'][k]
            print(f"  {k+1}    {dn_m:.4f}±{dn_s:.4f}           {cs_m:.4f}±{cs_s:.4f}")

    return results


# ===== 实验 B：逐层步长分析 =====

def exp_B(model, tokenizer, device, boolq_ex):
    print("\n" + "=" * 60)
    print("实验 B：逐层步长分析（output_hidden_states=True）")
    print("=" * 60)

    step_sizes = []

    for prefix, conts, label in tqdm(boolq_ex[:N_B], desc="B-layerstep"):
        enc = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask")
        attn = attn.to(device) if attn is not None else None

        with torch.inference_mode():
            out = model(
                input_ids=input_ids,
                attention_mask=attn,
                use_cache=False,
                output_hidden_states=True,
            )

        hs = out.hidden_states  # (n_layers+1,) tuple of (1, seq, hidden)

        sample_steps = []
        for l in range(36):
            h_before = hs[l][:, -1, :].float()
            h_after = hs[l + 1][:, -1, :].float()
            step = float((h_after - h_before).norm().item()) / (float(h_before.norm().item()) + 1e-8)
            sample_steps.append(step)
        step_sizes.append(sample_steps)

    step_arr = np.array(step_sizes)  # (N_B, 36)
    mean_steps = step_arr.mean(axis=0).tolist()
    std_steps = step_arr.std(axis=0).tolist()

    # 找异常层（z-score > 2）
    mean_arr = np.array(mean_steps)
    z_scores = (mean_arr - mean_arr.mean()) / (mean_arr.std() + 1e-8)

    print("\n  Layer  Step_Size(均值)  Z-Score  备注")
    for l in range(36):
        marker = ""
        if z_scores[l] > 1.5:
            marker = " ← 高步长"
        elif z_scores[l] < -1.0:
            marker = " ← 低步长"
        # 标记关键区间
        if l in [20, 21, 22, 23, 24, 25, 26, 27]:
            marker += " [T-end 区域]"
        print(f"  {l:2d}     {mean_steps[l]:.4f}         {z_scores[l]:.2f}  {marker}")

    return {
        "mean_step_size": mean_steps,
        "std_step_size": std_steps,
        "z_scores": z_scores.tolist(),
        "layer_idx": list(range(36)),
        "high_step_layers": [l for l in range(36) if z_scores[l] > 1.5],
        "low_step_layers": [l for l in range(36) if z_scores[l] < -1.0],
    }


# ===== 实验 C：答案概率变化 =====

def exp_C(model, tokenizer, device, boolq_ex, arc_ex):
    print("\n" + "=" * 60)
    print(f"实验 C：答案概率变化（baseline 正误分组，配置=({T_START_OPT},{T_END_OPT})）")
    print("=" * 60)

    n_e = T_START_OPT
    n_t = T_END_OPT - T_START_OPT + 1
    results = {}

    for task_name, examples in [("boolq", boolq_ex[:N_C]), ("arc", arc_ex[:N_C])]:
        benefits_correct = []
        benefits_wrong = []

        for prefix, conts, label in tqdm(examples, desc=f"C-{task_name}"):
            plen = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
            scores_k1 = []
            scores_k2 = []

            for cont in conts:
                full = prefix + cont
                enc = tokenizer(full, return_tensors="pt", add_special_tokens=False)
                input_ids = enc["input_ids"].to(device)
                attn = enc.get("attention_mask")
                attn = attn.to(device) if attn is not None else None

                with torch.inference_mode():
                    # k=1 等价于 baseline（完整一次 forward，不重复 T）
                    logits_k1 = etd_forward_logits(model, input_ids, attn, n_e, n_t, 1)
                    logits_k2 = etd_forward_logits(model, input_ids, attn, n_e, n_t, 2)

                scores_k1.append(loglikelihood_continuation(logits_k1, input_ids, plen))
                scores_k2.append(loglikelihood_continuation(logits_k2, input_ids, plen))

            pred_k1 = max(range(len(scores_k1)), key=lambda i: scores_k1[i])
            is_correct = (pred_k1 == label)
            benefit = scores_k2[label] - scores_k1[label]

            if is_correct:
                benefits_correct.append(benefit)
            else:
                benefits_wrong.append(benefit)

        n_c, n_w = len(benefits_correct), len(benefits_wrong)
        m_c = float(np.mean(benefits_correct)) if n_c > 0 else 0.0
        m_w = float(np.mean(benefits_wrong)) if n_w > 0 else 0.0

        print(f"\n  [{task_name}]:")
        print(f"  已做对组 ({n_c}条): mean_benefit = {m_c:+.4f} {'(循环强化正确✓)' if m_c > 0 else '(循环削弱正确✗)'}")
        print(f"  已做错组 ({n_w}条): mean_benefit = {m_w:+.4f} {'(循环能纠错✓)' if m_w > 0 else '(循环固化偏见✗)'}")

        results[task_name] = {
            "n_correct_group": n_c,
            "n_wrong_group": n_w,
            "mean_benefit_correct_group": m_c,
            "mean_benefit_wrong_group": m_w,
            "benefits_correct": [float(x) for x in benefits_correct],
            "benefits_wrong": [float(x) for x in benefits_wrong],
        }

    return results


# ===== R1 残差连接实验 =====

def exp_R1(model, tokenizer, device, boolq_ex, arc_ex):
    print("\n" + "=" * 60)
    print("R1 实验：全局残差连接 (h_final = h_k_iters + h_e)")
    print(f"样本数: {N_R1}/任务，配置: 最优({T_START_OPT},{T_END_OPT}) + 谷底({T_START_VALLEY},{T_END_VALLEY})")
    print("=" * 60)

    configs = [
        ("optimal", T_START_OPT, T_END_OPT),
        ("valley", T_START_VALLEY, T_END_VALLEY),
    ]

    results = {}

    for cfg_name, t_start, t_end in configs:
        n_e = t_start
        n_t = t_end - t_start + 1
        results[cfg_name] = {}
        print(f"\n  --- 配置: {cfg_name} (t_start={t_start}, t_end={t_end}, n_t={n_t}) ---")

        for task_name, examples in [("boolq", boolq_ex[:N_R1]), ("arc", arc_ex[:N_R1])]:
            c_orig = 0
            c_r1 = 0
            n = len(examples)

            for prefix, conts, label in tqdm(examples, desc=f"R1-{cfg_name}-{task_name}"):
                pred_orig = predict_mc_choice(model, tokenizer, prefix, conts, n_e, n_t, 2, device)
                pred_r1 = predict_mc_r1(model, tokenizer, prefix, conts, n_e, n_t, 2, device)

                if pred_orig == label:
                    c_orig += 1
                if pred_r1 == label:
                    c_r1 += 1

            acc_orig = c_orig / n
            acc_r1 = c_r1 / n
            delta = acc_r1 - acc_orig
            print(f"    {task_name}: ETD-k2={acc_orig:.4f}, R1-k2={acc_r1:.4f}, Δ={delta:+.4f} "
                  f"({'↑改善' if delta > 0.002 else '↓退步' if delta < -0.002 else '≈持平'})")

            results[cfg_name][task_name] = {
                "n": n,
                "acc_orig_k2": float(acc_orig),
                "acc_r1_k2": float(acc_r1),
                "delta": float(delta),
            }

    return results


# ===== 报告生成 =====

def generate_report(all_results) -> str:
    A = all_results["experiment_A"]
    B = all_results["experiment_B"]
    C = all_results["experiment_C"]
    R1 = all_results["experiment_R1"]

    lines = ["# 综合实验 A+B+C+R1 分析报告\n"]
    lines.append(f"实验时间：{time.strftime('%Y-%m-%d %H:%M')}\n")

    # ---- 实验 A 分析 ----
    lines.append("## 实验 A：隐层轨迹分析\n")
    lines.append(f"配置：最优 (t_start={T_START_OPT}, t_end={T_END_OPT}), k=1..{K_MAX}\n\n")

    for task in ["boolq", "arc"]:
        dn = A[task]["delta_norm_mean"]
        cs = A[task]["cos_sim_mean"]
        lines.append(f"### {task.upper()}\n")
        lines.append("| k | delta_norm | cos_sim |\n|---|---|---|\n")
        for i, (d, c) in enumerate(zip(dn, cs)):
            lines.append(f"| {i+1} | {d:.4f} | {c:.4f} |\n")

        # 判断收敛性
        if dn[-1] < dn[0] * 0.5:
            conv = "**快速收敛**（delta_norm 降低到初始的 50% 以下）"
        elif dn[-1] < dn[0]:
            conv = "**缓慢收敛**（delta_norm 下降，但不显著）"
        else:
            conv = "**不收敛/振荡**（delta_norm 未明显下降）"

        # 判断方向一致性
        avg_cs = float(np.mean(cs[1:]))
        if avg_cs > 0.7:
            dir_con = f"**方向稳定**（平均 cos_sim={avg_cs:.3f}）"
        elif avg_cs > 0.3:
            dir_con = f"**方向部分一致**（平均 cos_sim={avg_cs:.3f}）"
        else:
            dir_con = f"**方向振荡**（平均 cos_sim={avg_cs:.3f}）"

        lines.append(f"\n- 收敛性：{conv}\n")
        lines.append(f"- 方向一致性：{dir_con}\n\n")

    # ---- 实验 B 分析 ----
    lines.append("## 实验 B：逐层步长分析\n\n")
    high_layers = B["high_step_layers"]
    low_layers = B["low_step_layers"]
    mean_steps = B["mean_step_size"]

    lines.append(f"- **高步长层**（z-score > 1.5）：{high_layers}\n")
    lines.append(f"- **低步长层**（z-score < -1.0）：{low_layers}\n\n")

    # 分析 t_end=24 区域（层 24 本身）
    step_24 = mean_steps[24] if len(mean_steps) > 24 else 0
    z_24 = B["z_scores"][24] if len(B["z_scores"]) > 24 else 0
    lines.append(f"层 24 步长：{step_24:.4f}（z-score={z_24:.2f}）\n\n")

    if z_24 > 1.5:
        lines.append(
            "**结论**：层 24 是高步长层，重复 T 到层 24 时会产生过冲，"
            "这解释了 t_end=24 的 valley 现象。方案 R2（阻尼迭代）对此有针对性。\n\n"
        )
    elif z_24 > 0.5:
        lines.append("**结论**：层 24 步长偏高，可能部分造成 valley，但不是强异常。\n\n")
    else:
        lines.append(
            "**结论**：层 24 步长正常，t_end=24 的 valley 原因不在单层步长异常，"
            "可能是 position embedding 双重施加或其他因素。\n\n"
        )

    # ---- 实验 C 分析 ----
    lines.append("## 实验 C：答案概率变化\n\n")

    for task in ["boolq", "arc"]:
        m_c = C[task]["mean_benefit_correct_group"]
        m_w = C[task]["mean_benefit_wrong_group"]
        n_c = C[task]["n_correct_group"]
        n_w = C[task]["n_wrong_group"]

        lines.append(f"### {task.upper()}\n")
        lines.append(f"- 已做对组 ({n_c}条): mean_benefit = {m_c:+.4f}\n")
        lines.append(f"- 已做错组 ({n_w}条): mean_benefit = {m_w:+.4f}\n\n")

        # 解读
        if task == "boolq":
            if m_c > 0 and m_w < 0:
                interp = "✅ ETD 对 BoolQ 机制正常：**强化正确、固化偏见**。循环在已对题上有益，已错题上无益。"
            elif m_c > 0 and m_w > 0:
                interp = "⚠️ ETD 对 BoolQ：两组均有益，但循环对错题也有一定纠错能力（意外）。"
            elif m_c < 0:
                interp = "❌ ETD 对 BoolQ 有害：即使已对的题，循环也在削弱正确答案概率。"
            else:
                interp = "ℹ️ 结论不明确，需要更多样本。"
        else:  # arc
            if m_c < 0 and m_w < 0:
                interp = "✅ 验证了假设：ETD 对 ARC **两组均有害**，固化偏见（包括正确的偏见也被过度削弱）。"
            elif m_c > 0 and m_w < 0:
                interp = "⚠️ ARC 中 ETD 在已对题上有益，但已错题上固化偏见。"
            else:
                interp = f"ℹ️ ARC 结论：correct_benefit={m_c:+.4f}, wrong_benefit={m_w:+.4f}。"
        lines.append(f"**解读**：{interp}\n\n")

    # ---- R1 分析 ----
    lines.append("## R1 残差连接实验\n\n")
    lines.append("方案：`h_final = h_after_k_iters + h_e`（最末尾一次性加回 E 输出）\n\n")

    for cfg_name in ["optimal", "valley"]:
        t_s = T_START_OPT if cfg_name == "optimal" else T_START_VALLEY
        t_e = T_END_OPT if cfg_name == "optimal" else T_END_VALLEY
        lines.append(f"### 配置：{cfg_name} (t_start={t_s}, t_end={t_e})\n\n")

        for task in ["boolq", "arc"]:
            r = R1[cfg_name][task]
            delta = r["delta"]
            verdict = "**有效↑**" if delta > 0.005 else ("**有害↓**" if delta < -0.005 else "**持平≈**")
            lines.append(f"- {task.upper()}: ETD-k2={r['acc_orig_k2']:.4f} → R1={r['acc_r1_k2']:.4f} "
                         f"(Δ={delta:+.4f}) {verdict}\n")
        lines.append("\n")

    # 总结
    lines.append("## 综合结论与下一步行动\n\n")

    # R1 对 valley 是否有改善？
    valley_boolq_delta = R1.get("valley", {}).get("boolq", {}).get("delta", 0)
    valley_arc_delta = R1.get("valley", {}).get("arc", {}).get("delta", 0)

    if valley_boolq_delta > 0.005 or valley_arc_delta > 0.005:
        lines.append("- **R1 对谷底配置有改善**：残差连接确实缓解了过冲问题。\n")
        lines.append("  下一步：在更多配置上测试 R1，并尝试 R2（阻尼迭代）。\n")
    else:
        lines.append(
            "- **R1 对谷底配置改善有限**：valley 的根本原因不是漂移，需考虑其他机制。\n"
        )

    opt_boolq_delta = R1.get("optimal", {}).get("boolq", {}).get("delta", 0)
    if opt_boolq_delta > 0.002:
        lines.append(f"- **R1 在最优配置上进一步提升 BoolQ**（Δ={opt_boolq_delta:+.4f}），值得在完整 500 样本上验证。\n")
    elif opt_boolq_delta < -0.002:
        lines.append(f"- R1 在最优配置上损害 BoolQ（Δ={opt_boolq_delta:+.4f}）：E 输出加回导致表征失衡。\n")
        lines.append("  下一步：尝试 R2（阻尼迭代, alpha<1）而非直接相加。\n")

    lines.append("\n---\n")
    lines.append("*自动生成摘要（完整版见 abc_r1_report.md）*\n")

    return "".join(lines)


# ===== 更新 memory bank =====

def update_memory_bank(all_results, report_text):
    """将实验结果同步到 memory bank"""
    try:
        # 更新 experiments-log.md
        exp_log = MEMORY_BANK / "experiments-log.md"
        with open(exp_log, "r", encoding="utf-8") as f:
            content = f.read()

        date_str = time.strftime("%Y-%m-%d")
        new_row = (
            f"| {date_str} | etd-mechanism-v1 | exp-abc-r1 | "
            f"前置实验A+B+C + R1残差连接，各100样本 | "
            f"`/root/autodl-tmp/loop_layer/experiments/results/` |\n"
        )
        if "exp-abc-r1" not in content:
            content = content.replace(
                "| 日期 | campaign | id | 备注 | 产物路径 |",
                "| 日期 | campaign | id | 备注 | 产物路径 |",
            )
            # 在第一行数据前插入
            content = content.replace(
                "| 2026-04-06 |",
                new_row + "| 2026-04-06 |",
                1,
            )
            with open(exp_log, "w", encoding="utf-8") as f:
                f.write(content)

        # 写入报告到 memory bank
        report_path = MEMORY_BANK / "reflection" / "exp_abc_r1_findings.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        print(f"\nMemory bank 已更新：{report_path}")
    except Exception as e:
        print(f"\nMemory bank 更新失败（不影响实验）：{e}")


# ===== 主流程 =====

def main():
    print("=" * 60)
    print("综合实验 A+B+C+R1 — ETD 循环机制深度分析")
    print(f"最优配置: t_start={T_START_OPT}, t_end={T_END_OPT}")
    print(f"谷底配置: t_start={T_START_VALLEY}, t_end={T_END_VALLEY}")
    print(f"实验参数: N_A={N_A}, N_B={N_B}, N_C={N_C}, N_R1={N_R1}, K_MAX={K_MAX}")
    print("=" * 60)

    model, tokenizer, device = load_model(MODEL_PATH)

    print("\n加载评估数据...")
    max_n = max(N_A, N_B, N_C, N_R1)
    boolq_ex = load_boolq_examples("validation", max_n)
    arc_ex = load_arc_examples("test", max_n)
    print(f"  BoolQ: {len(boolq_ex)} 条，ARC: {len(arc_ex)} 条")

    t_total = time.time()

    print("\n[1/4] 运行实验 A...")
    t0 = time.time()
    results_A = exp_A(model, tokenizer, device, boolq_ex, arc_ex)
    print(f"  实验 A 完成，耗时 {time.time()-t0:.0f}s")

    print("\n[2/4] 运行实验 B...")
    t0 = time.time()
    results_B = exp_B(model, tokenizer, device, boolq_ex)
    print(f"  实验 B 完成，耗时 {time.time()-t0:.0f}s")

    print("\n[3/4] 运行实验 C...")
    t0 = time.time()
    results_C = exp_C(model, tokenizer, device, boolq_ex, arc_ex)
    print(f"  实验 C 完成，耗时 {time.time()-t0:.0f}s")

    print("\n[4/4] 运行 R1 实验...")
    t0 = time.time()
    results_R1 = exp_R1(model, tokenizer, device, boolq_ex, arc_ex)
    print(f"  R1 实验完成，耗时 {time.time()-t0:.0f}s")

    total_time = time.time() - t_total

    all_results = {
        "experiment_A": results_A,
        "experiment_B": results_B,
        "experiment_C": results_C,
        "experiment_R1": results_R1,
        "config": {
            "t_start_opt": T_START_OPT,
            "t_end_opt": T_END_OPT,
            "t_start_valley": T_START_VALLEY,
            "t_end_valley": T_END_VALLEY,
            "N_A": N_A, "N_B": N_B, "N_C": N_C, "N_R1": N_R1,
            "K_MAX": K_MAX,
        },
        "total_time_seconds": round(total_time, 1),
    }

    # 保存 JSON 结果
    results_file = RESULTS_DIR / "abc_r1_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ 原始结果已保存：{results_file}")

    # 简短自动摘要（详细版见 abc_r1_report.md，勿覆盖）
    report = generate_report(all_results)
    report_auto = RESULTS_DIR / "abc_r1_report_auto.md"
    with open(report_auto, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"✓ 自动摘要已保存：{report_auto}")
    print(f"  （完整图文报告请维护：{RESULTS_DIR / 'abc_r1_report.md'}）")

    try:
        from plot_abc_r1_figures import main as plot_abc_r1_main

        plot_abc_r1_main()
        print(f"✓ 机制图已更新：{RESULTS_DIR.parent / 'figures' / 'abc_r1_experiment_*.png'}")
    except Exception as e:
        print(f"⚠ 绘图跳过（可手动运行 experiments/plot_abc_r1_figures.py）：{e}")

    # 更新 memory bank
    update_memory_bank(all_results, report)

    print(f"\n总耗时：{total_time:.0f}s ({total_time/60:.1f} 分钟)")
    print("\n" + "=" * 60)
    print("报告预览：")
    print("=" * 60)
    print(report)


def load_model(model_path=MODEL_PATH):
    print("加载模型 Qwen3-8B...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"  模型已加载到 {device}，耗时 {time.time()-t0:.1f}s")
    return model, tokenizer, device


if __name__ == "__main__":
    main()
