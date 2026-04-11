"""Shared data loading and paths for R29 (mirrors exp_round27 patterns)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, "/root/autodl-tmp/loop_layer")
sys.path.insert(0, "/root/autodl-tmp/loop_layer/ETD")

from datasets import load_dataset  # noqa: E402

RESULTS_DIR = "/root/autodl-tmp/loop_layer/experiments/results"
FIGURES_DIR = "/root/autodl-tmp/loop_layer/experiments/figures"
MODEL_PATH = os.environ.get("R29_MODEL_PATH", "/root/autodl-tmp/model_qwen")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def _fmt(prefix: str, conts: list[str], label: int):
    choices_str = [c.strip() for c in conts]
    return {"prompt": prefix, "choices": choices_str, "answer": choices_str[label]}


def load_boolq(n: int):
    ds = load_dataset("aps/super_glue", "boolq")["validation"]
    out = []
    for r in ds:
        if len(out) >= n:
            break
        lab = int(r["label"])
        if lab < 0:
            continue
        prefix = f"{r['passage']}\nQuestion: {r['question']}?\nAnswer:"
        out.append(_fmt(prefix, ["no", "yes"], lab))
    return out


def load_arc(subset: str, n: int):
    ds = load_dataset("allenai/ai2_arc", subset)["test"]
    out = []
    for r in ds:
        if len(out) >= n:
            break
        q = r["question"].strip()
        texts = r["choices"]["text"]
        key = r["answerKey"]
        label = ord(key) - ord("A") if key in "ABCD" else int(key) - 1
        out.append(_fmt(f"Question: {q}\nAnswer:", texts, label))
    return out


def load_csqa(n: int):
    ds = load_dataset("tau/commonsense_qa")["validation"]
    out = []
    lmap = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    for r in ds:
        if len(out) >= n:
            break
        key = r["answerKey"]
        if key not in lmap:
            continue
        out.append(
            _fmt(
                f"Question: {r['question'].strip()}\nAnswer:",
                r["choices"]["text"],
                lmap[key],
            )
        )
    return out


def load_truthfulqa(n: int):
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice")["validation"]
    out = []
    for r in ds:
        if len(out) >= n:
            break
        mc1 = r["mc1_targets"]
        labels = mc1["labels"]
        if 1 not in labels:
            continue
        label = labels.index(1)
        out.append(_fmt(f"Question: {r['question'].strip()}\nAnswer:", mc1["choices"], label))
    return out


def load_benchmark(name: str, n: int):
    if name == "BoolQ":
        return load_boolq(n)
    if name == "ARC-C":
        return load_arc("ARC-Challenge", n)
    if name == "ARC-Easy":
        return load_arc("ARC-Easy", n)
    if name == "CSQA":
        return load_csqa(n)
    if name == "TruthfulQA":
        return load_truthfulqa(n)
    raise ValueError(name)


def load_benchmarks(names: list[str], n: int) -> dict[str, list]:
    data = {}
    for b in names:
        try:
            data[b] = load_benchmark(b, n)
            print(f"  Loaded {b}: {len(data[b])} examples")
        except Exception as e:
            print(f"  SKIP {b}: {e}")
    return data
