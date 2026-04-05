#!/usr/bin/env python3
"""Evaluate BoolQ / ARC test accuracy with ETD (fixed k) or baseline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ARTIFACTS, DEFAULT_EVAL_LIMIT, DEFAULT_MODEL_PATH
from etd_forward import predict_mc_choice


def boolq_items(split: str, limit: int):
    from datasets import load_dataset

    ds = load_dataset("aps/super_glue", "boolq")[split]
    seen = 0
    for i in range(len(ds)):
        if limit > 0 and seen >= limit:
            break
        r = ds[i]
        lab = int(r["label"])
        if lab < 0:
            continue
        prefix = f"{r['passage']}\nQuestion: {r['question']}?\nAnswer:"
        # Match lm_eval doc_to_choice order ["no", "yes"]
        conts = [" no", " yes"]
        seen += 1
        yield prefix, conts, lab


def arc_items(split: str, limit: int):
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")[split]
    n = len(ds) if limit <= 0 else min(limit, len(ds))
    for i in range(n):
        r = ds[i]
        q = r["question"].strip()
        texts = r["choices"]["text"]
        prefix = f"Question: {q}\nAnswer:"
        conts = [f" {t}" for t in texts]
        key = r["answerKey"]
        if isinstance(key, str) and key in "ABCD":
            label = ord(key) - ord("A")
        else:
            label = int(key) - 1
        yield prefix, conts, label


def run_eval(
    model,
    tokenizer,
    dataset: str,
    split: str,
    layer_path: Path,
    ks: list[int | None],
    device: torch.device,
    limit: int,
) -> dict:
    with open(layer_path, encoding="utf-8") as f:
        layer_cfg = json.load(f)
    n_e = int(layer_cfg["n_e"])
    n_t = int(layer_cfg["n_t"])
    n_d = int(layer_cfg["n_d"])

    if dataset == "boolq":
        stream = list(boolq_items(split, limit))
    else:
        stream = list(arc_items(split, limit))

    results: dict[str, float] = {}
    for k in ks:
        name = "baseline" if k is None else f"k={k}"
        correct = 0
        for prefix, conts, label in tqdm(stream, desc=name):
            pred = predict_mc_choice(
                model,
                tokenizer,
                prefix,
                conts,
                n_e,
                n_t,
                k,
                device,
            )
            if pred == label:
                correct += 1
        results[name] = correct / max(len(stream), 1)
    return {
        "dataset": dataset,
        "split": split,
        "n_examples": len(stream),
        "layer_file": str(layer_path),
        "n_e": n_e,
        "n_t": n_t,
        "n_d": n_d,
        "accuracy": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("boolq", "arc"), required=True)
    parser.add_argument(
        "--split",
        default=None,
        help="BoolQ 官方 test 无标签，默认用 validation；ARC 默认 test。",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--layer-json",
        default=None,
        help="JSON from collect_layers.py (default: artifacts/<dataset>_layers.json)",
    )
    parser.add_argument("--k", nargs="*", type=int, default=[2, 3], help="ETD repeat counts; omit for none")
    parser.add_argument("--baseline", action="store_true", help="Also run standard forward (no ETD loop)")
    parser.add_argument("--limit", type=int, default=DEFAULT_EVAL_LIMIT)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    if args.dataset == "boolq" and args.split == "test":
        raise SystemExit(
            "BoolQ 的 test 划分无 gold 标签（label=-1）。请使用 --split validation，"
            "或改用其它带标签的 BoolQ 源。"
        )

    split = args.split
    if split is None:
        split = "validation" if args.dataset == "boolq" else "test"

    layer_path = Path(args.layer_json or (ARTIFACTS / f"{args.dataset}_layers.json"))
    if not layer_path.is_file():
        raise FileNotFoundError(f"Missing {layer_path}; run collect_layers.py first.")

    ks: list[int | None] = [int(x) for x in args.k]
    if args.baseline:
        ks = [None] + ks

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
    )
    device = next(model.parameters()).device

    summary = run_eval(
        model,
        tokenizer,
        args.dataset,
        split,
        layer_path,
        ks,
        device,
        args.limit,
    )
    print(json.dumps(summary, indent=2))

    out_path = Path(args.out) if args.out else (ARTIFACTS / f"eval_{args.dataset}_{split}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
