#!/usr/bin/env python3
"""Offline angle-distance collection + Kneedle layer split for BoolQ or ARC."""

from __future__ import annotations

import argparse
import os
import random

from transformers import AutoModelForCausalLM, AutoTokenizer

from angle_distance import collect_distances, kneedle_split, save_split
from config import ARTIFACTS, DEFAULT_ANGLE_MAX_SAMPLES, DEFAULT_MODEL_PATH


def load_texts_boolq(split: str, max_samples: int, seed: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("aps/super_glue", "boolq")[split]
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    idx = idx[:max_samples]
    rows = [ds[i] for i in idx]
    texts = []
    for r in rows:
        t = f"{r['passage']}\nQuestion: {r['question']}?\nAnswer:"
        texts.append(t)
    return texts


def load_texts_arc(split: str, max_samples: int, seed: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")[split]
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    idx = idx[:max_samples]
    texts = []
    for i in idx:
        r = ds[i]
        q = r["question"].strip()
        texts.append(f"Question: {q}\nAnswer:")
    return texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("boolq", "arc"), required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--split", default="train", help="HF split to sample prompts from")
    parser.add_argument("--max-samples", type=int, default=DEFAULT_ANGLE_MAX_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="Output JSON path")
    args = parser.parse_args()

    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    if args.dataset == "boolq":
        texts = load_texts_boolq(args.split, args.max_samples, args.seed)
    else:
        texts = load_texts_arc(args.split, args.max_samples, args.seed)

    out_path = args.out or (ARTIFACTS / f"{args.dataset}_layers.json")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
    )

    device = next(model.parameters()).device
    distances, meta = collect_distances(
        model,
        tokenizer,
        texts,
        batch_size=args.batch_size,
        device=device,
    )
    n_e, n_t, n_d = kneedle_split(distances, model.config.num_hidden_layers)
    meta.update(
        {
            "dataset": args.dataset,
            "split": args.split,
            "seed": args.seed,
            "max_samples": args.max_samples,
        }
    )
    save_split(out_path, distances, n_e, n_t, n_d, meta)
    print(f"Wrote {out_path}")
    print(f"N_E={n_e}, N_T={n_t}, N_D={n_d} (sum={n_e + n_t + n_d})")


if __name__ == "__main__":
    main()
