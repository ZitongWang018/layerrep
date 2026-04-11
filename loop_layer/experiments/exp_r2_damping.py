#!/usr/bin/env python3
"""
R2 阻尼迭代实验

动机（基于 exp_abc_r1 的发现）：
  - F-05/F-06：T 块迭代方向快速随机化（k=2时 cos_sim=0.43），delta_norm 在 k=3 最低后振荡
  - 这说明 T 块每步的步幅过大，导致振荡而非收敛
  - R2 阻尼：h_new = alpha * T(h) + (1-alpha) * h，控制每步步幅

方案：
  - alpha ∈ {0.3, 0.5, 0.7, 0.9, 1.0(原版)}
  - 配置：最优 (10,21) + 谷底 (10,24) + ARC最优 (12,20)
  - 每个配置：k=2（与原版对齐）
  - 任务：BoolQ + ARC，各 100 条样本

同时运行修正版实验 C（margin benefit 分析）：
  - margin(k) = score_correct(k) - max(score_wrong(k))
  - benefit = margin(k2) - margin(k1)
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
    etd_forward_logits,
    loglikelihood_continuation,
    predict_mc_choice,
)

# ===== 实验参数 =====
ALPHAS = [0.3, 0.5, 0.7, 0.9, 1.0]  # 1.0 = 原版 ETD

CONFIGS = [
    ("optimal_boolq", 10, 21),   # BoolQ 最优：0.870 @ 500样本
    ("valley",        10, 24),   # 谷底配置：t_end=24 全局最差
    ("optimal_arc",   12, 20),   # ARC 最优：0.530 @ 500样本
]

N_EVAL = 100  # 每任务样本数


# ===== 工具函数 =====

def _prepare_ctx(model, input_ids, attention_mask):
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
        config=cfg, inputs_embeds=inputs_embeds,
        attention_mask=attention_mask, past_key_values=None, position_ids=position_ids,
    )
    cmm = {"full_attention": create_causal_mask(**mask_kwargs)}
    if getattr(base, "has_sliding_layers", False):
        from transformers.masking_utils import create_sliding_window_causal_mask
        cmm["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
    pos_emb = base.rotary_emb(inputs_embeds, position_ids)
    return inputs_embeds, position_ids, cmm, pos_emb


def _layer_safe(layer, h, mask, pid, pos_emb):
    out = layer(h, attention_mask=mask, position_ids=pid,
                past_key_values=None, use_cache=False, position_embeddings=pos_emb)
    return out[0] if isinstance(out, tuple) else out


def _get_mask(cmm, cfg, l):
    return cmm[cfg.layer_types[l]]


@torch.inference_mode()
def etd_forward_damped(model, input_ids, attention_mask, n_e, n_t, k, alpha):
    """R2 阻尼迭代：h = alpha * T(h) + (1-alpha) * h"""
    base = model.model
    cfg = model.config
    l_total = cfg.num_hidden_layers

    inputs_embeds, pid, cmm, pos_emb = _prepare_ctx(model, input_ids, attention_mask)

    h = inputs_embeds
    for i in range(n_e):
        h = _layer_safe(base.layers[i], h, _get_mask(cmm, cfg, i), pid, pos_emb)

    for _ in range(k):
        h_prev = h
        for i in range(n_e, n_e + n_t):
            h = _layer_safe(base.layers[i], h, _get_mask(cmm, cfg, i), pid, pos_emb)
        if abs(alpha - 1.0) > 1e-6:
            h = alpha * h + (1.0 - alpha) * h_prev

    for i in range(n_e + n_t, l_total):
        h = _layer_safe(base.layers[i], h, _get_mask(cmm, cfg, i), pid, pos_emb)

    h = base.norm(h)
    return model.lm_head(h)


def predict_mc_damped(model, tokenizer, prefix, conts, n_e, n_t, k, alpha, device):
    scores = []
    plen = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
    for cont in conts:
        full = prefix + cont
        enc = tokenizer(full, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask")
        attn = attn.to(device) if attn is not None else None
        logits = etd_forward_damped(model, input_ids, attn, n_e, n_t, k, alpha)
        scores.append(loglikelihood_continuation(logits, input_ids, plen))
    return max(range(len(scores)), key=lambda i: scores[i])


def score_all_conts_damped(model, tokenizer, prefix, conts, n_e, n_t, k, alpha, device):
    """返回所有 continuation 的 score 列表"""
    plen = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
    scores = []
    for cont in conts:
        full = prefix + cont
        enc = tokenizer(full, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask")
        attn = attn.to(device) if attn is not None else None
        logits = etd_forward_damped(model, input_ids, attn, n_e, n_t, k, alpha)
        scores.append(loglikelihood_continuation(logits, input_ids, plen))
    return scores


# ===== R2 主实验：多 alpha 扫描 =====

def exp_R2(model, tokenizer, device, boolq_ex, arc_ex):
    print("\n" + "=" * 60)
    print(f"R2 阻尼实验 alpha∈{ALPHAS}，3个配置×2个任务×{N_EVAL}样本")
    print("=" * 60)

    results = {}

    for cfg_name, t_start, t_end in CONFIGS:
        n_e = t_start
        n_t = t_end - t_start + 1
        results[cfg_name] = {"t_start": t_start, "t_end": t_end, "n_t": n_t}

        print(f"\n--- 配置: {cfg_name} (t_start={t_start}, t_end={t_end}, n_t={n_t}) ---")

        for task_name, examples in [("boolq", boolq_ex[:N_EVAL]), ("arc", arc_ex[:N_EVAL])]:
            results[cfg_name][task_name] = {}
            accs = {}

            for alpha in ALPHAS:
                correct = 0
                n = len(examples)
                desc = f"R2-{cfg_name}-{task_name}-a{alpha}"

                for prefix, conts, label in tqdm(examples, desc=desc, leave=False):
                    pred = predict_mc_damped(model, tokenizer, prefix, conts,
                                            n_e, n_t, 2, alpha, device)
                    if pred == label:
                        correct += 1

                acc = correct / n
                accs[alpha] = float(acc)

            # 找最优 alpha
            best_alpha = max(accs, key=lambda a: accs[a])

            print(f"  [{task_name}]:", end="")
            for alpha in ALPHAS:
                marker = " ← 最优" if alpha == best_alpha else ""
                print(f"  α={alpha}: {accs[alpha]:.4f}{marker}", end="")
            print()

            results[cfg_name][task_name] = {
                "accs_by_alpha": accs,
                "best_alpha": best_alpha,
                "best_acc": accs[best_alpha],
                "orig_acc": accs[1.0],
                "best_delta": accs[best_alpha] - accs[1.0],
            }

    return results


# ===== 修正版实验 C2：margin benefit =====

def exp_C2(model, tokenizer, device, boolq_ex, arc_ex, n_e=10, n_t=12):
    """
    margin(k) = score_correct(k) - max_{wrong j}(score_j(k))
    benefit = margin(k=2) - margin(k=1)
    按 baseline_correct 分组
    """
    print("\n" + "=" * 60)
    print(f"实验 C2（修正）：margin benefit 分析 (t_start={n_e}, t_end={n_e+n_t-1})")
    print("=" * 60)

    results = {}
    N_C2 = 100

    for task_name, examples in [("boolq", boolq_ex[:N_C2]), ("arc", arc_ex[:N_C2])]:
        margins_k1 = []
        margins_k2 = []
        is_correct_baseline = []

        for prefix, conts, label in tqdm(examples, desc=f"C2-{task_name}"):
            scores_k1 = score_all_conts_damped(
                model, tokenizer, prefix, conts, n_e, n_t, 1, 1.0, device
            )
            scores_k2 = score_all_conts_damped(
                model, tokenizer, prefix, conts, n_e, n_t, 2, 1.0, device
            )

            pred_k1 = max(range(len(scores_k1)), key=lambda i: scores_k1[i])
            is_correct_baseline.append(pred_k1 == label)

            wrong_indices = [i for i in range(len(conts)) if i != label]

            margin_k1 = scores_k1[label] - max(scores_k1[j] for j in wrong_indices)
            margin_k2 = scores_k2[label] - max(scores_k2[j] for j in wrong_indices)

            margins_k1.append(margin_k1)
            margins_k2.append(margin_k2)

        benefits = [m2 - m1 for m1, m2 in zip(margins_k1, margins_k2)]

        correct_idx = [i for i, c in enumerate(is_correct_baseline) if c]
        wrong_idx = [i for i, c in enumerate(is_correct_baseline) if not c]

        b_correct = [benefits[i] for i in correct_idx]
        b_wrong = [benefits[i] for i in wrong_idx]

        m_c = float(np.mean(b_correct)) if b_correct else 0.0
        m_w = float(np.mean(b_wrong)) if b_wrong else 0.0

        print(f"\n  [{task_name}]:")
        print(f"  已对组({len(correct_idx)}条): margin_benefit = {m_c:+.4f} "
              f"{'(循环拉大正确-错误差距✓)' if m_c > 0 else '(循环缩小差距✗)'}")
        print(f"  已错组({len(wrong_idx)}条): margin_benefit = {m_w:+.4f} "
              f"{'(循环有纠错效果✓)' if m_w > 0 else '(循环固化错误✗)'}")

        results[task_name] = {
            "n_correct": len(correct_idx),
            "n_wrong": len(wrong_idx),
            "mean_margin_benefit_correct": m_c,
            "mean_margin_benefit_wrong": m_w,
        }

    return results


# ===== 报告生成 =====

def generate_r2_report(r2_results, c2_results):
    lines = ["# R2 阻尼实验 + C2 修正报告\n"]
    lines.append(f"实验时间：{time.strftime('%Y-%m-%d %H:%M')}\n\n")

    lines.append("## R2 阻尼迭代扫描\n\n")
    lines.append(f"方案：`h = alpha * T(h) + (1-alpha) * h`，alpha∈{ALPHAS}，k=2\n\n")

    baseline_boolq = 0.862  # 500样本的 baseline

    for cfg_name, t_start, t_end in CONFIGS:
        lines.append(f"### 配置 {cfg_name} (t_start={t_start}, t_end={t_end})\n\n")
        for task in ["boolq", "arc"]:
            r = r2_results[cfg_name][task]
            lines.append(f"**{task.upper()}**:\n")
            lines.append("| alpha | acc | Δ vs orig |\n|---|---|---|\n")
            for alpha in ALPHAS:
                acc = r["accs_by_alpha"][alpha]
                delta = acc - r["orig_acc"]
                marker = " ← 最优" if alpha == r["best_alpha"] else ""
                lines.append(f"| {alpha} | {acc:.4f} | {delta:+.4f}{marker} |\n")
            lines.append(f"\n最优 alpha={r['best_alpha']}，acc={r['best_acc']:.4f}，"
                         f"相对原版 Δ={r['best_delta']:+.4f}\n\n")

    lines.append("## 修正版实验 C2：margin benefit 分析\n\n")
    lines.append("方案：margin(k)=score_correct(k)-max(score_wrong(k))，"
                 "benefit=margin(k2)-margin(k1)\n\n")

    for task in ["boolq", "arc"]:
        c = c2_results[task]
        lines.append(f"### {task.upper()}\n")
        lines.append(f"- 已对组({c['n_correct']}条): margin_benefit={c['mean_margin_benefit_correct']:+.4f}\n")
        lines.append(f"- 已错组({c['n_wrong']}条): margin_benefit={c['mean_margin_benefit_wrong']:+.4f}\n\n")

    lines.append("## 综合结论\n\n")

    # 判断 R2 是否有效
    best_boolq_delta = max(
        r2_results["optimal_boolq"]["boolq"]["accs_by_alpha"][a]
        - r2_results["optimal_boolq"]["boolq"]["accs_by_alpha"][1.0]
        for a in ALPHAS if a != 1.0
    )
    best_valley_boolq = max(
        r2_results["valley"]["boolq"]["accs_by_alpha"][a]
        for a in ALPHAS
    )
    orig_valley_boolq = r2_results["valley"]["boolq"]["accs_by_alpha"][1.0]

    if best_boolq_delta > 0.005:
        lines.append(f"- **R2 在最优配置上有效**：阻尼使 BoolQ 进一步提升 Δ={best_boolq_delta:+.4f}\n")
        lines.append("  → 下一步：在完整 500 样本上验证最优 alpha\n")
    elif best_boolq_delta > -0.005:
        lines.append("- R2 在最优配置上效果持平：阻尼没有改善，也没有明显损害\n")
    else:
        lines.append(f"- R2 在最优配置上也有害（Δ={best_boolq_delta:+.4f}）：阻尼不适合当前配置\n")
        lines.append("  → 振荡本身可能是有益的（伪随机探索），阻尼反而削弱了探索能力\n")
        lines.append("  → 下一步：重新思考 ETD 的改进方向，考虑自动选层而非修改前向\n")

    lines.append("\n---\n*自动生成于 exp_r2_damping.py*\n")
    return "".join(lines)


# ===== 主流程 =====

def main():
    print("=" * 60)
    print("R2 阻尼实验 + C2 修正版 margin benefit 分析")
    print(f"alpha∈{ALPHAS}, 配置数={len(CONFIGS)}, N_EVAL={N_EVAL}")
    print("=" * 60)

    print("\n加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype="auto", device_map="auto"
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"  已加载到 {device}")

    print("\n加载评估数据...")
    boolq_ex = load_boolq_examples("validation", N_EVAL)
    arc_ex = load_arc_examples("test", N_EVAL)
    print(f"  BoolQ: {len(boolq_ex)}, ARC: {len(arc_ex)}")

    t_total = time.time()

    print("\n[1/2] R2 阻尼扫描...")
    t0 = time.time()
    r2_results = exp_R2(model, tokenizer, device, boolq_ex, arc_ex)
    print(f"  R2 完成，耗时 {time.time()-t0:.0f}s")

    print("\n[2/2] C2 修正版 margin benefit 分析...")
    t0 = time.time()
    c2_results = exp_C2(model, tokenizer, device, boolq_ex, arc_ex)
    print(f"  C2 完成，耗时 {time.time()-t0:.0f}s")

    total_time = time.time() - t_total

    all_results = {
        "experiment_R2": r2_results,
        "experiment_C2": c2_results,
        "config": {
            "alphas": ALPHAS,
            "configs": [(n, ts, te) for n, ts, te in CONFIGS],
            "N_EVAL": N_EVAL,
        },
        "total_time_seconds": round(total_time, 1),
    }

    results_file = RESULTS_DIR / "r2_c2_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ 结果已保存：{results_file}")

    report = generate_r2_report(r2_results, c2_results)
    report_file = RESULTS_DIR / "r2_c2_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"✓ 报告已保存：{report_file}")

    # 更新 memory bank
    try:
        reflection_file = MEMORY_BANK / "reflection" / "r2_c2_findings.md"
        reflection_file.parent.mkdir(parents=True, exist_ok=True)
        with open(reflection_file, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"✓ Memory bank 已更新：{reflection_file}")
    except Exception as e:
        print(f"  Memory bank 更新失败：{e}")

    print(f"\n总耗时：{total_time:.0f}s ({total_time/60:.1f} 分钟)")
    print("\n" + "=" * 60)
    print(report)


if __name__ == "__main__":
    main()
