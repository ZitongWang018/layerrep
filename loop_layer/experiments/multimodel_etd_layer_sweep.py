#!/usr/bin/env python3
"""
R30-style ETD layer sweep for local Llama / Gemma checkpoints.

Grid: iterate (t_start, t_stop) with T-block [t_start, t_stop) (t_stop exclusive),
same convention as experiments/results/r30_sweep_results.json.

Benchmark profiles:
  r30plus5 — BoolQ, ARC-C, TruthfulQA (50), CSQA, MMLU-HS-Math (R30-style N).
  hard_mc — GPQA-Diamond, AGIEval-Gaokao-MathQA, LogiQA (lm-eval-aligned MC).
  all — r30plus5 + hard_mc (8 benchmarks).

  HF Hub requests default to HF_ENDPOINT=https://hf-mirror.com when unset; override for huggingface.co if needed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# huggingface_hub / datasets read this for Hub API host; mirror helps when huggingface.co is slow or blocked.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path("/root/autodl-tmp/loop_layer")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ETD"))
sys.path.insert(0, str(ROOT / "experiments"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import DownloadConfig, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from etd_forward import baseline_forward_logits, etd_forward_logits, loglikelihood_continuation  # noqa: E402
from hard_mc_benchmark_loaders import (  # noqa: E402
    load_agieval_gaokao_mathqa,
    load_gpqa_diamond,
    load_logiqa,
)
from multimodel_sweep_visualize import generate_all_figures  # noqa: E402
from r29_common import (  # noqa: E402
    load_arc,
    load_boolq,
    load_csqa,
    load_truthfulqa,
)

CHAMP_K = 2

BENCH_PROFILES: dict[str, list[str]] = {
    "r30plus5": ["BoolQ", "ARC-C", "TruthfulQA", "CSQA", "MMLU-HS-Math"],
    "hard_mc": ["GPQA-Diamond", "AGIEval-Gaokao-MathQA", "LogiQA"],
    "all": [
        "BoolQ",
        "ARC-C",
        "TruthfulQA",
        "CSQA",
        "MMLU-HS-Math",
        "GPQA-Diamond",
        "AGIEval-Gaokao-MathQA",
        "LogiQA",
    ],
}

# results/<subdir>/etd_layer_sweep_r30style.json ; figures under same subdir
PROFILE_RESULT_SUBDIR: dict[str, str] = {
    "r30plus5": "",
    "hard_mc": "hard_mc",
    "all": "all8",
}

PRESETS: dict[str, dict] = {
    "llama3-8b": {
        "model_path": "/root/autodl-tmp/Llama3-8B",
        "t_start_min": 7,
        "t_start_max": 24,
        "t_stop_min": 13,
        "t_stop_max": 30,
        "out_subdir": "llama3-8b",
    },
    "gemma2-2b": {
        "model_path": "/root/autodl-tmp/Gemma2-2B",
        "t_start_min": 5,
        "t_start_max": 18,
        "t_stop_min": 10,
        "t_stop_max": 24,
        "out_subdir": "gemma2-2b",
    },
    "qwen3-8b": {
        "model_path": "/root/autodl-tmp/model_qwen",
        "t_start_min": 5,
        "t_start_max": 23,
        "t_stop_min": 18,
        "t_stop_max": 30,
        "out_subdir": "qwen3-8b",
    },
}


def _fmt(prefix: str, conts: list[str], label: int):
    choices_str = [c.strip() for c in conts]
    return {"prompt": prefix, "choices": choices_str, "answer": choices_str[label]}


def load_mmlu_hs_math(n: int) -> list[dict]:
    dc = DownloadConfig(local_files_only=bool(int(os.environ.get("HF_DATASETS_OFFLINE", "0"))))
    ds = load_dataset(
        "cais/mmlu",
        "high_school_mathematics",
        download_config=dc,
    )["test"]
    out = []
    for r in ds:
        if len(out) >= n:
            break
        choices = [str(c) for c in r["choices"]]
        lab = int(r["answer"])
        out.append(_fmt(f"Question: {r['question'].strip()}\nAnswer:", choices, lab))
    return out


def load_benchmark(name: str, n: int) -> list[dict]:
    if name == "BoolQ":
        return load_boolq(n)
    if name == "ARC-C":
        return load_arc("ARC-Challenge", n)
    if name == "CSQA":
        return load_csqa(n)
    if name == "TruthfulQA":
        return load_truthfulqa(n)
    if name == "MMLU-HS-Math":
        return load_mmlu_hs_math(n)
    if name == "GPQA-Diamond":
        return load_gpqa_diamond(n)
    if name == "AGIEval-Gaokao-MathQA":
        return load_agieval_gaokao_mathqa(n)
    if name == "LogiQA":
        return load_logiqa(n)
    raise ValueError(name)


def load_all_benchmarks(
    bench_list: list[str],
    n_default: int,
    n_truthful: int,
) -> dict[str, list[dict]]:
    data: dict[str, list[dict]] = {}
    for b in bench_list:
        n = n_truthful if b == "TruthfulQA" else n_default
        try:
            data[b] = load_benchmark(b, n)
            print(f"  Loaded {b}: {len(data[b])} examples", flush=True)
        except Exception as e:
            print(f"  SKIP {b}: {e}", flush=True)
    return data


def iter_grid(
    t_start_min: int,
    t_start_max: int,
    t_stop_min: int,
    t_stop_max: int,
    num_layers: int,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for ts in range(t_start_min, t_start_max + 1):
        for te in range(t_stop_min, t_stop_max + 1):
            if te <= ts:
                continue
            n_t = te - ts
            n_d = num_layers - ts - n_t
            if n_d < 1 or ts < 1 or n_t < 1:
                continue
            pairs.append((ts, te))
    return pairs


@torch.inference_mode()
def score_mc(
    tok,
    model,
    device: torch.device,
    ex: dict,
    t_start: int | None,
    t_stop: int | None,
    use_etd: bool,
    k: int,
) -> bool:
    prefix = ex["prompt"]
    gold = ex["answer"].strip().lower()
    choices_sp = [" " + c for c in ex["choices"]]
    pref = tok(prefix, return_tensors="pt", add_special_tokens=False)
    prompt_len = pref["input_ids"].shape[1]
    scores = []
    for ch in choices_sp:
        full = tok(prefix + ch, return_tensors="pt", add_special_tokens=False)
        ids = full["input_ids"].to(device)
        am = full.get("attention_mask")
        am = am.to(device) if am is not None else None
        if not use_etd or t_start is None or t_stop is None:
            logits = baseline_forward_logits(model, ids, am)
        else:
            n_t = max(t_stop - t_start, 1)
            logits = etd_forward_logits(model, ids, am, t_start, n_t, k, alpha="auto")
        scores.append(loglikelihood_continuation(logits, ids, prompt_len))
    pred_i = int(np.argmax(scores))
    valid = ex.get("valid_indices")
    if valid is not None:
        return pred_i in set(valid)
    pred = ex["choices"][pred_i].strip().lower()
    return pred == gold


def eval_accuracy(
    tok,
    model,
    device: torch.device,
    examples: list[dict],
    t_start: int | None,
    t_stop: int | None,
    use_etd: bool,
    k: int,
    desc: str,
) -> tuple[float, int]:
    correct = 0
    for ex in tqdm(examples, desc=desc, leave=False):
        if score_mc(tok, model, device, ex, t_start, t_stop, use_etd, k):
            correct += 1
    n = len(examples)
    return correct / max(n, 1), correct


def compute_baseline(
    tok,
    model,
    device: torch.device,
    data: dict[str, list[dict]],
    k: int,
) -> dict[str, float]:
    baselines: dict[str, float] = {}
    for bench, examples in data.items():
        acc, _ = eval_accuracy(tok, model, device, examples, None, None, False, k, f"baseline/{bench}")
        baselines[bench] = float(acc)
    return baselines


def write_top_configs_txt(path: Path, best_rows: dict[str, dict], baselines: dict[str, float]) -> None:
    lines = [
        "# Best (t_start, t_stop) by benchmark — ETD k=2 alpha=auto",
        f"# Generated {datetime.now(timezone.utc).isoformat()}",
        "",
    ]
    for bench, row in best_rows.items():
        if bench == "macro_avg":
            continue
        b = baselines.get(bench, float("nan"))
        lines.append(
            f"{bench}: best_acc={row[bench]:.4f} baseline={b:.4f} "
            f"t_start={row['t_start']} t_stop={row['t_stop']} n_t={row['n_t']}"
        )
    mb = float(np.mean([baselines[b] for b in baselines])) if baselines else float("nan")
    lines.append(f"macro_baseline (mean of bench baselines)={mb:.4f}")
    if "macro_avg" in best_rows:
        r = best_rows["macro_avg"]
        lines.append(
            f"MACRO: best_etd_macro={r['macro_avg']:.4f} delta_vs_macro_baseline={r['macro_avg'] - mb:+.4f} "
            f"t_start={r['t_start']} t_stop={r['t_stop']}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS.keys()), required=True)
    ap.add_argument(
        "--bench-profile",
        choices=sorted(BENCH_PROFILES.keys()),
        default="r30plus5",
        help="Benchmark set: r30plus5 (5 tasks), hard_mc (3 hard MC), or all (8).",
    )
    ap.add_argument("--model-path", default=None, help="Override checkpoint path")
    ap.add_argument("--n-default", type=int, default=100, help="Samples per benchmark (except TruthfulQA)")
    ap.add_argument("--n-truthfulqa", type=int, default=50)
    ap.add_argument("--k", type=int, default=CHAMP_K)
    ap.add_argument("--max-cells", type=int, default=0, help="If >0, only first N grid cells (debug)")
    ap.add_argument("--resume", action="store_true", help="Skip (t_start,t_stop) already in output JSON")
    args = ap.parse_args()

    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    pr = PRESETS[args.preset]
    model_path = args.model_path or pr["model_path"]
    out_base = ROOT / "experiments" / pr["out_subdir"]
    sub = PROFILE_RESULT_SUBDIR[args.bench_profile]
    res_dir = out_base / "results" / sub if sub else out_base / "results"
    fig_dir = (out_base / "figures" / sub) if sub else (out_base / "figures")
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    out_json = res_dir / "etd_layer_sweep_r30style.json"
    out_txt = res_dir / "etd_layer_sweep_top_configs.txt"

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32

    bench_list = list(BENCH_PROFILES[args.bench_profile])
    print(
        json.dumps(
            {
                "preset": args.preset,
                "bench_profile": args.bench_profile,
                "model_path": model_path,
                "use_cuda": use_cuda,
                "result_dir": str(res_dir),
            },
            indent=2,
        )
    )

    data = load_all_benchmarks(bench_list, args.n_default, args.n_truthfulqa)
    if not data:
        raise SystemExit("No benchmark data loaded.")
    active_benches = list(data.keys())

    print("Loading model...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map="auto" if use_cuda else None,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    if not use_cuda:
        model.to(torch.device("cpu"))
    device = next(model.parameters()).device

    num_layers = int(model.config.num_hidden_layers)
    ts0, ts1 = pr["t_start_min"], pr["t_start_max"]
    te0, te1 = pr["t_stop_min"], pr["t_stop_max"]

    grid = iter_grid(ts0, ts1, te0, te1, num_layers)
    print(f"Grid: {len(grid)} valid (t_start, t_stop) pairs (num_layers={num_layers}).", flush=True)
    if args.max_cells > 0:
        grid = grid[: args.max_cells]
        print(f"  --max-cells={args.max_cells}: truncated to {len(grid)} pairs.", flush=True)

    done: set[tuple[int, int]] = set()
    results: list[dict] = []
    baselines: dict[str, float] = {}

    if args.resume and out_json.is_file():
        prev = json.loads(out_json.read_text(encoding="utf-8"))
        baselines = prev.get("baseline", {})
        results = list(prev.get("results", []))
        for row in results:
            done.add((int(row["t_start"]), int(row["t_stop"])))
        print(f"Resume: {len(done)} cells already done.", flush=True)

    if not baselines:
        baselines = compute_baseline(tok, model, device, data, args.k)
        print("Baseline accuracies:", json.dumps(baselines, indent=2), flush=True)

    config_meta = {
        "k": args.k,
        "alpha_fn": "min(1.0, 6.0/n_t)",
        "seed": 42,
        "num_layers": num_layers,
        "grid_bounds": {
            "t_start_min": ts0,
            "t_start_max": ts1,
            "t_stop_min": te0,
            "t_stop_max": te1,
        },
        "n_samples_per_benchmark": {
            "_default": args.n_default,
            "TruthfulQA": args.n_truthfulqa,
        },
        "model_path": model_path,
        "preset": args.preset,
        "bench_profile": args.bench_profile,
        "benchmarks_used": active_benches,
        "t_stop_exclusive": True,
    }

    t_wall = time.time()
    for ts, te in tqdm(grid, desc="grid"):
        if (ts, te) in done:
            continue
        n_t = te - ts
        alpha = min(1.0, 6.0 / max(n_t, 1))
        row: dict = {
            "t_start": ts,
            "t_stop": te,
            "n_t": n_t,
            "alpha": round(alpha, 6),
        }
        macro_sum = 0.0
        macro_n = 0
        for bench, examples in data.items():
            acc, _ = eval_accuracy(
                tok, model, device, examples, ts, te, True, args.k, f"ETD {bench} [{ts},{te})"
            )
            row[bench] = round(float(acc), 6)
            macro_sum += float(acc)
            macro_n += 1
        row["macro_avg"] = round(macro_sum / max(macro_n, 1), 6)
        results.append(row)
        done.add((ts, te))

        payload = {
            "config": config_meta,
            "baseline": baselines,
            "benchmarks_used": active_benches,
            "results": results,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_sec_wall": time.time() - t_wall,
        }
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Best configs
    best_by_bench: dict[str, dict] = {}
    for bench in active_benches:
        best_row = max(results, key=lambda r: r[bench])
        best_by_bench[bench] = best_row
    best_macro = max(results, key=lambda r: r["macro_avg"])
    best_by_bench["macro_avg"] = best_macro

    macro_bl = float(np.mean([baselines[b] for b in active_benches]))
    summary = {
        "best_per_benchmark": {b: {k: best_by_bench[b][k] for k in ("t_start", "t_stop", "n_t", b, "macro_avg")} for b in active_benches},
        "best_macro_avg_cell": {k: best_macro[k] for k in ("t_start", "t_stop", "n_t", "macro_avg") if k in best_macro},
        "macro_baseline": round(macro_bl, 6),
    }

    final_payload = {
        "config": config_meta,
        "baseline": baselines,
        "macro_baseline": round(macro_bl, 6),
        "note_baseline": "Standard 1× forward (no ETD); computed once before the (t_start,t_stop) grid.",
        "benchmarks_used": active_benches,
        "results": results,
        "summary": summary,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec_wall": time.time() - t_wall,
    }
    out_json.write_text(json.dumps(final_payload, indent=2), encoding="utf-8")

    write_top_configs_txt(out_txt, best_by_bench, baselines)

    title_prefix = args.preset.replace("-", " ").upper()
    fig_paths = generate_all_figures(final_payload, fig_dir, title_prefix)
    print(f"Wrote {out_json}", flush=True)
    print(f"Wrote {out_txt}", flush=True)
    for p in fig_paths:
        print(f"Wrote {p}", flush=True)


if __name__ == "__main__":
    main()
