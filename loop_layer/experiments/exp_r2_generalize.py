#!/usr/bin/env python3
"""
R2 阻尼自适应规则泛化验证实验

目标：验证 alpha = min(1.0, 6.0/n_t) 自适应规则在更多 (t_start, t_end) 配置上的泛化能力。

设计：
  - 选取覆盖不同 n_t 值的 8 个配置（n_t = 6, 8, 10, 12, 14, 16, 18, 20）
  - 每个配置运行：baseline（alpha=1.0）+ adaptive（alpha=6/n_t）
  - 任务：BoolQ + ARC，各 100 样本
  - 输出：比较 adaptive 是否优于 baseline

已知锚点（来自 500 样本验证）：
  - (10,21) n_t=12, alpha=0.5: BoolQ=0.880(+0.018), ARC=0.574(+0.042) ← 冠军
  - (10,21) n_t=12, alpha=1.0: BoolQ=0.870(+0.008), ARC=0.520(-0.012) ← 原版

假设（待验证）：
  H1: 对于 n_t < 6 的配置，alpha=1.0 最优（无需阻尼）
  H2: 对于 n_t >= 12 的配置，alpha=6/n_t < 1.0 优于 alpha=1.0
  H3: 自适应规则在 ARC 上的改善幅度随 n_t 增大而增大
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
    baseline_forward_logits,
    loglikelihood_continuation,
    _adaptive_alpha,
)

N_EVAL = 100

# 选取覆盖不同 n_t 的多样化配置
# 格式：(name, t_start, t_end)
# n_t = t_end - t_start
CONFIGS = [
    ("n6_early",   8,  14),   # n_t=6,  小 T 块，早期层
    ("n8_mid",     8,  16),   # n_t=8,  中小 T 块
    ("n10_mid",    8,  18),   # n_t=10, 中等 T 块
    ("n12_best",   10, 22),   # n_t=12, 接近最优（类似已验证的10,21）
    ("n14_large",  8,  22),   # n_t=14, 大 T 块，早起始
    ("n14_mid",    10, 24),   # n_t=14? 不，t_end-t_start=14，但这是 valley 区域
    ("n16_late",   10, 26),   # n_t=16, 大 T 块，延伸到深层
    ("n18_deep",   8,  26),   # n_t=18, 超大 T 块
    ("n6_late",    16, 22),   # n_t=6,  后期层
    ("n10_early",  6,  16),   # n_t=10, 早期层开始
]

# 过滤确保 n_d >= 1（总层数 36）
TOTAL_LAYERS = 36
CONFIGS = [
    (name, ts, te) for name, ts, te in CONFIGS
    if te < TOTAL_LAYERS and ts >= 1 and te > ts
]


def _layer_safe(layer, hs, **kwargs):
    out = layer(hs, **kwargs)
    return out[0] if isinstance(out, tuple) else out


@torch.inference_mode()
def eval_config(model, tokenizer, examples, n_e, n_t, alpha, device, desc=""):
    """Evaluate accuracy for a single configuration.
    examples: list of (prefix, continuations, label) tuples.
    """
    correct = 0
    total = 0
    for prefix, conts, label in tqdm(examples[:N_EVAL], desc=desc, leave=False):
        prompt_len = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
        scores = []
        for cont in conts:
            full = prefix + cont
            enc = tokenizer(full, return_tensors="pt", add_special_tokens=False)
            input_ids = enc["input_ids"].to(device)
            attn = enc.get("attention_mask")
            attention_mask = attn.to(device) if attn is not None else None
            logits = etd_forward_logits(
                model, input_ids, attention_mask,
                n_e=n_e, n_t=n_t, k=2, alpha=alpha,
            )
            scores.append(loglikelihood_continuation(logits, input_ids, prompt_len))
        pred = max(range(len(scores)), key=lambda i: scores[i])
        if pred == label:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


def main():
    print("=" * 60)
    print("R2 阻尼自适应规则泛化验证实验")
    print("=" * 60)
    print(f"自适应规则：alpha = min(1.0, 6.0/n_t)")
    print(f"评估样本数：{N_EVAL} per task")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"加载模型: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    print(f"  已加载到 {device}\n")

    print("加载评估数据...")
    boolq = load_boolq_examples("validation", N_EVAL)
    arc = load_arc_examples("test", N_EVAL)
    print(f"  BoolQ: {len(boolq)}, ARC: {len(arc)}\n")

    # 先跑 baseline（直接模型推理）
    print("计算 100 样本 baseline...")
    t0 = time.time()
    boolq_baseline_scores = []
    for prefix, conts, label in tqdm(boolq, desc="baseline-boolq", leave=False):
        pref_len = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
        scores = []
        for cont in conts:
            full = prefix + cont
            enc = tokenizer(full, return_tensors="pt", add_special_tokens=False)
            ids = enc["input_ids"].to(device)
            attn = enc.get("attention_mask", None)
            if attn is not None:
                attn = attn.to(device)
            out = model(input_ids=ids, attention_mask=attn, use_cache=False)
            scores.append(loglikelihood_continuation(out.logits, ids, pref_len))
        boolq_baseline_scores.append(max(range(len(scores)), key=lambda i: scores[i]) == label)
    boolq_baseline = sum(boolq_baseline_scores) / len(boolq_baseline_scores)

    arc_baseline_scores = []
    for prefix, conts, label in tqdm(arc, desc="baseline-arc", leave=False):
        pref_len = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
        scores = []
        for cont in conts:
            full = prefix + cont
            enc = tokenizer(full, return_tensors="pt", add_special_tokens=False)
            ids = enc["input_ids"].to(device)
            attn = enc.get("attention_mask", None)
            if attn is not None:
                attn = attn.to(device)
            out = model(input_ids=ids, attention_mask=attn, use_cache=False)
            scores.append(loglikelihood_continuation(out.logits, ids, pref_len))
        arc_baseline_scores.append(max(range(len(scores)), key=lambda i: scores[i]) == label)
    arc_baseline = sum(arc_baseline_scores) / len(arc_baseline_scores)
    print(f"  Baseline (100样本): BoolQ={boolq_baseline:.4f}, ARC={arc_baseline:.4f}  ({time.time()-t0:.0f}s)\n")

    results = {
        "baseline": {"boolq": boolq_baseline, "arc": arc_baseline},
        "configs": {}
    }

    print(f"{'配置':<18} {'n_t':>4} {'alpha_ad':>8} "
          f"{'BQ_orig':>7} {'BQ_adap':>7} {'BQ_diff':>7}  "
          f"{'AR_orig':>7} {'AR_adap':>7} {'AR_diff':>7}")
    print("-" * 90)

    for name, t_start, t_end in CONFIGS:
        n_e = t_start
        n_t = t_end - t_start
        alpha_ad = _adaptive_alpha(n_t)

        t0 = time.time()

        # alpha=1.0 (原版 ETD)
        bq_orig = eval_config(
            model, tokenizer, boolq, n_e, n_t, 1.0, device,
            desc=f"{name}-boolq-orig"
        )
        arc_orig = eval_config(
            model, tokenizer, arc, n_e, n_t, 1.0, device,
            desc=f"{name}-arc-orig"
        )

        # adaptive alpha
        if alpha_ad < 1.0:
            bq_adap = eval_config(
                model, tokenizer, boolq, n_e, n_t, alpha_ad, device,
                desc=f"{name}-boolq-adap"
            )
            arc_adap = eval_config(
                model, tokenizer, arc, n_e, n_t, alpha_ad, device,
                desc=f"{name}-arc-adap"
            )
        else:
            bq_adap = bq_orig
            arc_adap = arc_orig

        bq_diff = bq_adap - bq_orig
        arc_diff = arc_adap - arc_orig
        elapsed = time.time() - t0

        bq_sym = "+" if bq_diff >= 0 else ""
        arc_sym = "+" if arc_diff >= 0 else ""

        print(
            f"{name:<18} {n_t:>4} {alpha_ad:>8.2f} "
            f"{bq_orig:>7.4f} {bq_adap:>7.4f} {bq_sym+f'{bq_diff:.4f}':>7}  "
            f"{arc_orig:>7.4f} {arc_adap:>7.4f} {arc_sym+f'{arc_diff:.4f}':>7}"
            f"  ({elapsed:.0f}s)"
        )

        results["configs"][name] = {
            "t_start": t_start, "t_end": t_end, "n_t": n_t,
            "alpha_orig": 1.0, "alpha_adaptive": alpha_ad,
            "boolq_orig": bq_orig, "boolq_adaptive": bq_adap, "boolq_diff": bq_diff,
            "arc_orig": arc_orig, "arc_adaptive": arc_adap, "arc_diff": arc_diff,
        }

    # 统计分析
    print("\n" + "=" * 90)
    print("统计分析：alpha=6/n_t 自适应规则 vs 原版 alpha=1.0")
    print("-" * 60)

    adaptive_better_boolq = 0
    adaptive_better_arc = 0
    total_with_damping = 0

    for name, data in results["configs"].items():
        if data["alpha_adaptive"] < 1.0:
            total_with_damping += 1
            if data["boolq_diff"] > 0:
                adaptive_better_boolq += 1
            if data["arc_diff"] > 0:
                adaptive_better_arc += 1

    print(f"触发阻尼的配置数（alpha_ad < 1.0）：{total_with_damping}/{len(results['configs'])}")
    print(f"阻尼改善 BoolQ 的配置数：{adaptive_better_boolq}/{total_with_damping}")
    print(f"阻尼改善 ARC 的配置数：{adaptive_better_arc}/{total_with_damping}")

    # 验证假设
    print("\n假设验证：")
    n_t_list = [(name, data["n_t"], data["arc_diff"])
                for name, data in results["configs"].items()
                if data["alpha_adaptive"] < 1.0]
    n_t_list.sort(key=lambda x: x[1])
    print(f"H2（n_t>=12 时 adaptive 优于 orig）：")
    for name, n_t, arc_diff in n_t_list:
        marker = "✅" if arc_diff > 0 else "❌"
        print(f"  {name} (n_t={n_t}): ARC diff={arc_diff:+.4f} {marker}")

    # 保存结果
    out_path = RESULTS_DIR / "r2_generalize_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {out_path}")

    # 生成报告
    report_lines = [
        "# R2 阻尼自适应规则泛化验证报告",
        f"\n实验时间：2026-04-06",
        f"N_EVAL={N_EVAL}，alpha=min(1.0, 6.0/n_t)",
        f"\nBaseline (100样本): BoolQ={boolq_baseline:.4f}, ARC={arc_baseline:.4f}",
        "\n## 各配置结果",
        "\n| 配置 | n_t | alpha_ad | BQ_orig | BQ_adap | BQ_diff | AR_orig | AR_adap | AR_diff |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, data in results["configs"].items():
        report_lines.append(
            f"| {name} | {data['n_t']} | {data['alpha_adaptive']:.2f} | "
            f"{data['boolq_orig']:.4f} | {data['boolq_adaptive']:.4f} | {data['boolq_diff']:+.4f} | "
            f"{data['arc_orig']:.4f} | {data['arc_adaptive']:.4f} | {data['arc_diff']:+.4f} |"
        )

    report_lines += [
        f"\n## 结论",
        f"- 阻尼改善 BoolQ: {adaptive_better_boolq}/{total_with_damping}",
        f"- 阻尼改善 ARC: {adaptive_better_arc}/{total_with_damping}",
    ]

    report_path = RESULTS_DIR / "r2_generalize_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"报告已保存: {report_path}")

    # 更新 memory bank
    update_memory_bank(results, boolq_baseline, arc_baseline,
                       adaptive_better_boolq, adaptive_better_arc, total_with_damping)

    return results


def update_memory_bank(results, boolq_baseline, arc_baseline,
                       better_boolq, better_arc, total_damping):
    experiments_log = MEMORY_BANK / "experiments-log.md"
    if experiments_log.exists():
        content = experiments_log.read_text()
        entry = f"""
## exp-r2-generalize（2026-04-06）

**目标**：验证 alpha=min(1.0, 6.0/n_t) 自适应规则在 {len(results['configs'])} 个不同 n_t 配置上的泛化能力

**结果（{N_EVAL} 样本）**：
- Baseline: BoolQ={boolq_baseline:.4f}, ARC={arc_baseline:.4f}
- 触发阻尼的配置：{total_damping}/{len(results['configs'])}
- 阻尼改善 BoolQ: {better_boolq}/{total_damping}
- 阻尼改善 ARC: {better_arc}/{total_damping}

详细结果：见 r2_generalize_results.json
"""
        content += entry
        experiments_log.write_text(content)
        print(f"已更新 memory bank: {experiments_log}")


if __name__ == "__main__":
    main()
