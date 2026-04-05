#!/usr/bin/env python3
"""
Sweep T layer ranges (inclusive indices): t_start in [5,15], t_end in [20,28], all t_start <= t_end.
For each cell: evaluate BoolQ + ARC on the SAME fixed example lists; baseline, k=2, k=3.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

from sweep_config import (  # noqa: E402
    ARTIFACTS,
    DEFAULT_ARC_LIMIT,
    DEFAULT_ARC_SPLIT,
    DEFAULT_BOOLQ_LIMIT,
    DEFAULT_BOOLQ_SPLIT,
    DEFAULT_MODEL_PATH,
    NUM_LAYERS,
    T_END_MAX,
    T_END_MIN,
    T_START_MAX,
    T_START_MIN,
)
from data_cache import load_arc_examples, load_boolq_examples  # noqa: E402

# Import ETD forward from sibling folder (after local imports so `config` is not shadowed)
_ETD = Path(__file__).resolve().parent.parent / "ETD"
if str(_ETD) not in sys.path:
    sys.path.insert(0, str(_ETD))

from etd_forward import predict_mc_choice  # noqa: E402


def split_from_t_range(t_start: int, t_end: int) -> tuple[int, int, int]:
    n_e = t_start
    n_t = t_end - t_start + 1
    n_d = NUM_LAYERS - n_e - n_t
    if n_e < 1 or n_t < 1 or n_d < 1:
        raise ValueError(f"Invalid T range [{t_start},{t_end}] -> n_e={n_e}, n_t={n_t}, n_d={n_d}")
    return n_e, n_t, n_d


def iter_t_cells():
    for t_start in range(T_START_MIN, T_START_MAX + 1):
        for t_end in range(T_END_MIN, T_END_MAX + 1):
            if t_start <= t_end:
                yield t_start, t_end


def eval_split_accuracy(
    model,
    tokenizer,
    device: torch.device,
    examples: list[tuple[str, list[str], int]],
    n_e: int,
    n_t: int,
    k_mode: int | None,
    desc: str,
) -> tuple[float, int]:
    correct = 0
    for prefix, conts, label in tqdm(examples, desc=desc, leave=False):
        pred = predict_mc_choice(model, tokenizer, prefix, conts, n_e, n_t, k_mode, device)
        if pred == label:
            correct += 1
    n = len(examples)
    return correct / max(n, 1), correct


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--boolq-split", default=DEFAULT_BOOLQ_SPLIT)
    parser.add_argument("--arc-split", default=DEFAULT_ARC_SPLIT)
    parser.add_argument("--boolq-limit", type=int, default=DEFAULT_BOOLQ_LIMIT)
    parser.add_argument("--arc-limit", type=int, default=DEFAULT_ARC_LIMIT)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--resume", action="store_true", help="Skip (t_start,t_end) rows already in CSV")
    parser.add_argument(
        "--max-cells",
        type=int,
        default=0,
        help="If >0, only first N (t_start,t_end) pairs (debug). Default 0 = all 99 cells.",
    )
    args = parser.parse_args()

    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv) if args.out_csv else ARTIFACTS / "sweep_results.csv"

    print("Loading fixed evaluation sets (same for all sweep cells)...")
    boolq_ex = load_boolq_examples(args.boolq_split, args.boolq_limit)
    arc_ex = load_arc_examples(args.arc_split, args.arc_limit)
    print(f"  BoolQ: {len(boolq_ex)} examples ({args.boolq_split})")
    print(f"  ARC:   {len(arc_ex)} examples ({args.arc_split})")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
    )
    device = next(model.parameters()).device

    done: set[tuple[int, int]] = set()
    if args.resume and out_csv.is_file():
        with open(out_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((int(row["t_start"]), int(row["t_end"])))

    fieldnames = [
        "t_start",
        "t_end",
        "n_e",
        "n_t",
        "n_d",
        "boolq_baseline",
        "boolq_k2",
        "boolq_k3",
        "arc_baseline",
        "arc_k2",
        "arc_k3",
        "boolq_correct_baseline",
        "boolq_correct_k2",
        "boolq_correct_k3",
        "arc_correct_baseline",
        "arc_correct_k2",
        "arc_correct_k3",
        "seconds_total",
    ]

    cells = list(iter_t_cells())
    if args.max_cells > 0:
        cells = cells[: args.max_cells]
    file_exists = out_csv.is_file()
    write_header = not file_exists or not args.resume

    mode = "w" if write_header else "a"
    with open(out_csv, mode, newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for t_start, t_end in tqdm(cells, desc="T-range sweep"):
            if args.resume and (t_start, t_end) in done:
                continue
            n_e, n_t, n_d = split_from_t_range(t_start, t_end)
            t0 = time.perf_counter()

            _, c_bb = eval_split_accuracy(
                model, tokenizer, device, boolq_ex, n_e, n_t, None, f"B-baseline {t_start}-{t_end}"
            )
            _, c_b2 = eval_split_accuracy(
                model, tokenizer, device, boolq_ex, n_e, n_t, 2, f"B-k=2 {t_start}-{t_end}"
            )
            _, c_b3 = eval_split_accuracy(
                model, tokenizer, device, boolq_ex, n_e, n_t, 3, f"B-k=3 {t_start}-{t_end}"
            )

            _, c_ab = eval_split_accuracy(
                model, tokenizer, device, arc_ex, n_e, n_t, None, f"A-baseline {t_start}-{t_end}"
            )
            _, c_a2 = eval_split_accuracy(
                model, tokenizer, device, arc_ex, n_e, n_t, 2, f"A-k=2 {t_start}-{t_end}"
            )
            _, c_a3 = eval_split_accuracy(
                model, tokenizer, device, arc_ex, n_e, n_t, 3, f"A-k=3 {t_start}-{t_end}"
            )

            nb = len(boolq_ex)
            na = len(arc_ex)
            row = {
                "t_start": t_start,
                "t_end": t_end,
                "n_e": n_e,
                "n_t": n_t,
                "n_d": n_d,
                "boolq_baseline": c_bb / nb,
                "boolq_k2": c_b2 / nb,
                "boolq_k3": c_b3 / nb,
                "arc_baseline": c_ab / na,
                "arc_k2": c_a2 / na,
                "arc_k3": c_a3 / na,
                "boolq_correct_baseline": c_bb,
                "boolq_correct_k2": c_b2,
                "boolq_correct_k3": c_b3,
                "arc_correct_baseline": c_ab,
                "arc_correct_k2": c_a2,
                "arc_correct_k3": c_a3,
                "seconds_total": round(time.perf_counter() - t0, 3),
            }
            writer.writerow(row)
            fp.flush()

    print(f"Done. Wrote {out_csv}")


if __name__ == "__main__":
    main()
