#!/usr/bin/env python3
"""
R2 阻尼迭代 — 500 样本完整验证

目标：验证 100 样本实验的结果在完整评估集上是否稳定
重点验证：
  1. (10,21) α=0.5: ARC 是否确实超过 baseline 0.532？
  2. (10,21) α=0.7: BoolQ 是否确实超过 baseline 0.862？
  3. valley (10,24) α=0.5: BoolQ+ARC 的改善是否稳定？

额外：alpha 与 n_t 的关系分析（三个配置的最优 alpha 汇总）
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

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
from etd_forward import loglikelihood_continuation  # noqa: E402

# 要验证的配置（t_start, t_end, alpha）
CONFIGS_TO_VALIDATE = [
    # (配置名, t_start, t_end, alpha)
    ("orig_opt",     10, 21, 1.0),  # 原版最优，对照
    ("r2_07_opt",    10, 21, 0.7),  # R2 BoolQ 最优 alpha
    ("r2_05_opt",    10, 21, 0.5),  # R2 ARC 最优 alpha
    ("r2_05_valley", 10, 24, 0.5),  # R2 valley 最优 alpha (BoolQ)
    ("r2_03_valley", 10, 24, 0.3),  # R2 valley 最优 alpha (ARC)
    ("arc_opt_orig", 12, 20, 1.0),  # ARC 配置原版
]

N_SAMPLES = 500  # 完整评估集


# ===== 工具函数（复用 exp_r2_damping.py 的实现）=====

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


@torch.inference_mode()
def etd_forward_damped(model, input_ids, attention_mask, n_e, n_t, k, alpha):
    base = model.model
    cfg = model.config
    l_total = cfg.num_hidden_layers

    inputs_embeds, pid, cmm, pos_emb = _prepare_ctx(model, input_ids, attention_mask)
    layer_types = cfg.layer_types

    h = inputs_embeds
    for i in range(n_e):
        h = _layer_safe(base.layers[i], h, cmm[layer_types[i]], pid, pos_emb)

    for _ in range(k):
        h_prev = h
        for i in range(n_e, n_e + n_t):
            h = _layer_safe(base.layers[i], h, cmm[layer_types[i]], pid, pos_emb)
        if abs(alpha - 1.0) > 1e-6:
            h = alpha * h + (1.0 - alpha) * h_prev

    for i in range(n_e + n_t, l_total):
        h = _layer_safe(base.layers[i], h, cmm[layer_types[i]], pid, pos_emb)

    h = base.norm(h)
    return model.lm_head(h)


def eval_mc_damped(model, tokenizer, examples, n_e, n_t, k, alpha, device, desc=""):
    correct = 0
    pbar = tqdm(examples, desc=desc or f"(t={n_e},{n_e+n_t-1},a={alpha})", leave=False)
    for prefix, conts, label in pbar:
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
        pred = max(range(len(scores)), key=lambda i: scores[i])
        if pred == label:
            correct += 1
    n = len(examples)
    return correct / n


def main():
    print("=" * 65)
    print("R2 阻尼 500 样本完整验证")
    print(f"配置数: {len(CONFIGS_TO_VALIDATE)}, 样本数: {N_SAMPLES}/任务")
    print("重点：验证 α=0.5/0.7 是否在完整集上稳定超过 baseline")
    print("=" * 65)

    # baseline 参考值（来自 sweep，500 样本）
    BASELINE_BOOLQ = 0.862
    BASELINE_ARC   = 0.532

    print("\n加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype="auto", device_map="auto"
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"  已加载到 {device}")

    print("\n加载评估数据...")
    boolq_ex = load_boolq_examples("validation", N_SAMPLES)
    arc_ex = load_arc_examples("test", N_SAMPLES)
    print(f"  BoolQ: {len(boolq_ex)}, ARC: {len(arc_ex)}")

    t_total = time.time()
    results = {}

    print(f"\n{'配置名':20s} {'task':6s} {'acc':8s} {'vs baseline':12s} {'Δ':8s}")
    print("-" * 60)

    for cfg_name, t_start, t_end, alpha in CONFIGS_TO_VALIDATE:
        n_e = t_start
        n_t = t_end - t_start + 1
        results[cfg_name] = {"t_start": t_start, "t_end": t_end, "alpha": alpha, "n_t": n_t}

        for task_name, examples, baseline in [
            ("boolq", boolq_ex, BASELINE_BOOLQ),
            ("arc", arc_ex, BASELINE_ARC),
        ]:
            t0 = time.time()
            acc = eval_mc_damped(
                model, tokenizer, examples, n_e, n_t, 2, alpha, device,
                desc=f"{cfg_name}-{task_name}"
            )
            elapsed = time.time() - t0
            delta = acc - baseline
            verdict = "✅超baseline" if delta > 0.002 else ("❌低baseline" if delta < -0.002 else "≈持平")
            print(f"{cfg_name:20s} {task_name:6s} {acc:.4f}   {baseline:.4f} ({delta:+.4f})  {verdict}  ({elapsed:.0f}s)")

            results[cfg_name][task_name] = {
                "acc": float(acc),
                "baseline": float(baseline),
                "delta_vs_baseline": float(delta),
            }

    total_time = time.time() - t_total

    # 保存结果
    all_results = {
        "results": results,
        "baseline": {"boolq": 0.862, "arc": 0.532},
        "N_SAMPLES": N_SAMPLES,
        "total_time_seconds": round(total_time, 1),
    }
    results_file = RESULTS_DIR / "r2_full_validate.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ 结果已保存：{results_file}")

    # 生成报告
    report_lines = ["# R2 阻尼 500 样本完整验证报告\n\n"]
    report_lines.append(f"实验时间：{time.strftime('%Y-%m-%d %H:%M')}\n")
    report_lines.append(f"总耗时：{total_time:.0f}s（{total_time/60:.1f}分钟）\n\n")
    report_lines.append(f"Baseline: BoolQ={0.862:.4f}, ARC={0.532:.4f}\n\n")
    report_lines.append("| 配置 | t_start | t_end | n_t | alpha | BoolQ | Δ(BoolQ) | ARC | Δ(ARC) |\n")
    report_lines.append("|---|---|---|---|---|---|---|---|---|\n")

    for cfg_name, t_start, t_end, alpha in CONFIGS_TO_VALIDATE:
        r = results[cfg_name]
        b_acc = r["boolq"]["acc"]
        a_acc = r["arc"]["acc"]
        b_d = r["boolq"]["delta_vs_baseline"]
        a_d = r["arc"]["delta_vs_baseline"]
        report_lines.append(
            f"| {cfg_name} | {t_start} | {t_end} | {t_end-t_start+1} | {alpha} | "
            f"{b_acc:.4f} | {b_d:+.4f} | {a_acc:.4f} | {a_d:+.4f} |\n"
        )

    report_lines.append("\n## 综合结论\n\n")

    # 找最优 BoolQ 和 ARC
    best_boolq = max(results.items(), key=lambda x: x[1].get("boolq", {}).get("acc", 0))
    best_arc = max(results.items(), key=lambda x: x[1].get("arc", {}).get("acc", 0))

    report_lines.append(f"- 最优 BoolQ 配置：{best_boolq[0]}（acc={best_boolq[1]['boolq']['acc']:.4f}，"
                        f"Δ={best_boolq[1]['boolq']['delta_vs_baseline']:+.4f}）\n")
    report_lines.append(f"- 最优 ARC 配置：{best_arc[0]}（acc={best_arc[1]['arc']['acc']:.4f}，"
                        f"Δ={best_arc[1]['arc']['delta_vs_baseline']:+.4f}）\n\n")

    # n_t 与最优 alpha 的关系
    report_lines.append("## alpha 与 n_t 的关系（从 100 样本实验汇总）\n\n")
    report_lines.append("| 配置 | n_t | 最优 alpha (BoolQ) | 最优 alpha (ARC) |\n")
    report_lines.append("|---|---|---|---|\n")
    report_lines.append("| optimal_boolq (10,21) | 12 | 0.7 | 0.5 |\n")
    report_lines.append("| valley (10,24) | 15 | 0.5 | 0.3 |\n")
    report_lines.append("| optimal_arc (12,20) | 9 | 0.5 | 1.0 |\n\n")
    report_lines.append("规律初步假设：n_t 越大，最优 alpha 越小（步幅越需要控制）\n")
    report_lines.append("n_t=9：ARC 不需要阻尼（alpha=1.0），BoolQ 稍需（alpha=0.5）\n")
    report_lines.append("n_t=12：BoolQ 最优 alpha=0.7，ARC 最优 alpha=0.5\n")
    report_lines.append("n_t=15：BoolQ 最优 alpha=0.5，ARC 最优 alpha=0.3\n")

    report_text = "".join(report_lines)
    report_file = RESULTS_DIR / "r2_full_validate_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"✓ 报告已保存：{report_file}")

    # 更新 memory bank
    try:
        mb_report = MEMORY_BANK / "reflection" / "r2_full_validate.md"
        with open(mb_report, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"✓ Memory bank 已更新：{mb_report}")
    except Exception as e:
        print(f"  Memory bank 更新失败：{e}")

    print(f"\n总耗时：{total_time:.0f}s ({total_time/60:.1f} 分钟)")
    print("\n" + "=" * 65)
    print(report_text)


if __name__ == "__main__":
    main()
