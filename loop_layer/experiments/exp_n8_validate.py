#!/usr/bin/env python3
"""
n8_mid 配置 500 样本完整验证

目标：验证 (t_start=8, t_end=16) alpha=0.75 在 500 样本上是否超越当前冠军
     (t_start=10, t_end=21) alpha=0.5（BoolQ=0.880, ARC=0.574）

背景：泛化验证实验（100样本）发现 n8_mid (8,16) alpha=0.75 的结果非常出色：
     BoolQ=0.89（+0.02 vs orig），ARC=0.60（+0.05 vs orig）
     但 100 样本的误差约 ±0.03，需要 500 样本确认。

同时验证几个有潜力的新配置：
  - (8,16) alpha=0.75 ← n8_mid 冠军候选
  - (8,22) alpha=0.43 ← n14_large 也表现优秀（BoolQ+0.02, ARC+0.06）
  - (10,21) alpha=0.5 ← 已知冠军（作为参照）
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
from etd_forward import etd_forward_logits, loglikelihood_continuation  # noqa: E402

N_SAMPLES = 500

CONFIGS_TO_VALIDATE = [
    # (name, t_start, t_end, alpha)
    ("champion_10_21",  10, 21, 0.5),   # 已知冠军（500样本验证过）
    ("n8_mid_8_16",      8, 16, 0.75),  # 泛化实验发现的候选
    ("n14_large_8_22",   8, 22, 6/14),  # n14 大 T 块候选
    ("n8_orig_8_16",     8, 16, 1.0),   # n8 原版（无阻尼对照）
]

BOOLQ_BASELINE = 0.862
ARC_BASELINE = 0.532


@torch.inference_mode()
def eval_config(model, tokenizer, examples, t_start, t_end, alpha, device, desc=""):
    n_e = t_start
    n_t = t_end - t_start
    correct = 0
    pbar = tqdm(examples, desc=desc, leave=False)
    t0 = time.time()
    for prefix, conts, label in pbar:
        plen = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
        scores = []
        for cont in conts:
            full = prefix + cont
            enc = tokenizer(full, return_tensors="pt", add_special_tokens=False)
            ids = enc["input_ids"].to(device)
            attn = enc.get("attention_mask")
            if attn is not None:
                attn = attn.to(device)
            logits = etd_forward_logits(model, ids, attn, n_e=n_e, n_t=n_t, k=2, alpha=alpha)
            scores.append(loglikelihood_continuation(logits, ids, plen))
        pred = max(range(len(scores)), key=lambda i: scores[i])
        if pred == label:
            correct += 1
    elapsed = time.time() - t0
    return correct / len(examples), elapsed


def main():
    print("=" * 70)
    print("n8_mid (8,16) 候选冠军 — 500 样本完整验证")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"加载模型: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    print(f"  已加载到 {device}\n")

    print("加载评估数据...")
    boolq_ex = load_boolq_examples("validation", N_SAMPLES)
    arc_ex = load_arc_examples("test", N_SAMPLES)
    print(f"  BoolQ: {len(boolq_ex)}, ARC: {len(arc_ex)}\n")

    results = {"baseline": {"boolq": BOOLQ_BASELINE, "arc": ARC_BASELINE}, "configs": {}}

    print(f"{'配置名':<22} {'task':<6} {'acc':>6}   {'vs baseline':>12}  Δ")
    print("-" * 70)

    for name, t_start, t_end, alpha in CONFIGS_TO_VALIDATE:
        n_t = t_end - t_start
        config_key = f"{name}"

        for task, examples, baseline in [("boolq", boolq_ex, BOOLQ_BASELINE),
                                          ("arc", arc_ex, ARC_BASELINE)]:
            acc, elapsed = eval_config(
                model, tokenizer, examples, t_start, t_end, alpha, device,
                desc=f"{name}-{task}"
            )
            delta = acc - baseline
            sym = "✅超baseline" if delta >= 0 else "❌低baseline"
            print(f"{name:<22} {task:<6} {acc:.4f}   {baseline:.4f} ({delta:+.4f})  {sym}  ({elapsed:.0f}s)")

            if config_key not in results["configs"]:
                results["configs"][config_key] = {
                    "t_start": t_start, "t_end": t_end, "n_t": n_t, "alpha": alpha
                }
            results["configs"][config_key][task] = {
                "acc": acc, "baseline": baseline, "delta": delta
            }

    # 汇总分析
    print("\n" + "=" * 70)
    print("汇总：哪个配置最好？")
    print("-" * 40)
    print(f"{'配置':<22} {'BoolQ':>8} {'ARC':>8} {'平均Δ':>8}")
    for name, data in results["configs"].items():
        bq = data.get("boolq", {})
        arc = data.get("arc", {})
        avg_delta = (bq.get("delta", 0) + arc.get("delta", 0)) / 2
        print(f"{name:<22} {bq.get('acc', 0):.4f}   {arc.get('acc', 0):.4f}   {avg_delta:+.4f}")

    out_path = RESULTS_DIR / "n8_validate_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {out_path}")

    # 更新 memory bank
    best_name = max(
        results["configs"].items(),
        key=lambda kv: (kv[1].get("boolq", {}).get("delta", -99) +
                        kv[1].get("arc", {}).get("delta", -99))
    )[0]
    best = results["configs"][best_name]
    print(f"\n🏆 500 样本最佳配置：{best_name}")
    print(f"   BoolQ={best.get('boolq', {}).get('acc', 0):.4f}, ARC={best.get('arc', {}).get('acc', 0):.4f}")

    update_memory_bank(results, best_name, best)


def update_memory_bank(results, best_name, best_data):
    experiments_log = MEMORY_BANK / "experiments-log.md"
    if experiments_log.exists():
        content = experiments_log.read_text()
        entry = f"""
## exp-n8-validate（2026-04-06）

**目标**：验证 n8_mid (8,16) 候选冠军是否超越已知冠军 (10,21) alpha=0.5

**最佳配置（500 样本）**：{best_name}
- BoolQ={best_data.get('boolq', {}).get('acc', 0):.4f}
- ARC={best_data.get('arc', {}).get('acc', 0):.4f}

**完整结果**：见 n8_validate_results.json
"""
        content += entry
        experiments_log.write_text(content)
        print(f"已更新 memory bank: {experiments_log}")


if __name__ == "__main__":
    main()
