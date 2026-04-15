#!/usr/bin/env python3
"""
R38-Multimodel: 在 Llama3-8B 与 Gemma2-2B 上复现 R38 的 cos_res 信号引导 ETD 验证（全部 8 benchmarks）。

与 Qwen3-8B 的差异：
  - sweep_best：从各模型已有的 R30-style 扫参 JSON 中按 benchmark 取准确率最高的 (t_start, t_stop)
    （llama3-8b/results/etd_layer_sweep_r30style.json + hard_mc/ 合并）
  - 探针层 / min_start / max_start：按 num_hidden_layers 缩放
  - Term1 近似：Llama/Qwen 使用 mlp(post_attention_layernorm(hi))；
    Gemma2 解码层在 FFN 前使用 pre_feedforward_layernorm，故使用 mlp(pre_feedforward_layernorm(hi))

条件（与 R38 一致，无 oracle 命名）：
  baseline, sweep_best, persample_cos8, persample_var, onset_fixed8, calib_onset8, calib_global8

用法：
  python experiments/exp_r38_multimodel_signal.py --preset llama3-8b
  python experiments/exp_r38_multimodel_signal.py --preset gemma2-2b
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
warnings.filterwarnings("ignore")

ROOT = Path("/root/autodl-tmp/loop_layer")
EXP = ROOT / "experiments"
ETD = ROOT / "ETD"
for p in (str(ROOT), str(EXP), str(ETD)):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import DownloadConfig, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from etd_forward import baseline_forward_logits, etd_forward_logits
from hard_mc_benchmark_loaders import load_agieval_gaokao_mathqa, load_gpqa_diamond

# ─── 预设 ────────────────────────────────────────────────────────────────────
N_CALIB = 20
K_ETD = 2
ONSET_THRESHOLD = 0.28
CALIB_ONSET_RATIO = 0.65
NT_VAR = (4, 6, 8)
CALIB_NT = 8

N_SAMPLES = {
    "BoolQ": 100,
    "ARC-C": 100,
    "TruthfulQA": 50,
    "CSQA": 100,
    "MMLU-HS-Math": 100,
    "GPQA-Diamond": 100,
    "AGIEval-Gaokao-MathQA": 100,
    "LogiQA": 100,
}

BENCH_ORDER = [
    "BoolQ",
    "ARC-C",
    "TruthfulQA",
    "CSQA",
    "MMLU-HS-Math",
    "GPQA-Diamond",
    "AGIEval-Gaokao-MathQA",
    "LogiQA",
]

COND_NAMES = [
    "baseline",
    "sweep_best",
    "persample_cos8",
    "persample_var",
    "onset_fixed8",
    "calib_onset8",
    "calib_global8",
]

PRESETS: dict[str, dict] = {
    "llama3-8b": {
        "model_path": "/root/autodl-tmp/Llama3-8B",
        "arch": "llama",
        "sweep_main": EXP / "llama3-8b/results/etd_layer_sweep_r30style.json",
        "sweep_hard": EXP / "llama3-8b/results/hard_mc/etd_layer_sweep_r30style.json",
        "out_json": EXP / "results/r38_multimodel_llama3_signal.json",
        "out_fig": EXP / "figures/r38_multimodel_llama3",
        "probe_layers": list(range(6, 27, 2)),
        "min_start": 8,
        "max_start": 20,
    },
    "gemma2-2b": {
        "model_path": "/root/autodl-tmp/Gemma2-2B",
        "arch": "gemma2",
        "sweep_main": EXP / "gemma2-2b/results/etd_layer_sweep_r30style.json",
        "sweep_hard": EXP / "gemma2-2b/results/hard_mc/etd_layer_sweep_r30style.json",
        "out_json": EXP / "results/r38_multimodel_gemma2_signal.json",
        "out_fig": EXP / "figures/r38_multimodel_gemma2",
        "probe_layers": list(range(4, 23, 2)),
        "min_start": 5,
        "max_start": 16,
    },
}


# ─── 数据加载（与 multimodel sweep 对齐 + LogiQA 离线修复）──────────────────────
def _fmt(prefix: str, conts: list[str], label: int) -> dict:
    choices_str = [c.strip() for c in conts]
    return {"prompt": prefix, "choices": choices_str, "answer": choices_str[label]}


def load_boolq(n: int) -> list[dict]:
    ds = load_dataset("aps/super_glue", "boolq")["validation"]
    out: list[dict] = []
    for r in ds:
        if len(out) >= n:
            break
        lab = int(r["label"])
        if lab < 0:
            continue
        out.append(_fmt(f"{r['passage']}\nQuestion: {r['question']}?\nAnswer:", ["no", "yes"], lab))
    return out


def load_arc_c(n: int) -> list[dict]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")["test"]
    out: list[dict] = []
    for r in ds:
        if len(out) >= n:
            break
        texts = r["choices"]["text"]
        key = r["answerKey"]
        label = ord(key) - ord("A") if key in "ABCD" else int(key) - 1
        out.append(_fmt(f"Question: {r['question'].strip()}\nAnswer:", texts, label))
    return out


def load_csqa(n: int) -> list[dict]:
    ds = load_dataset("tau/commonsense_qa")["validation"]
    lmap = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    out: list[dict] = []
    for r in ds:
        if len(out) >= n:
            break
        key = r["answerKey"]
        if key not in lmap:
            continue
        out.append(_fmt(f"Question: {r['question'].strip()}\nAnswer:", r["choices"]["text"], lmap[key]))
    return out


def load_truthfulqa(n: int) -> list[dict]:
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice")["validation"]
    out: list[dict] = []
    for r in ds:
        if len(out) >= n:
            break
        labels = r["mc1_targets"]["labels"]
        out.append(
            _fmt(
                f"Question: {r['question']}\nAnswer:",
                r["mc1_targets"]["choices"],
                int(np.argmax(labels)),
            )
        )
    return out


def load_mmlu_hs_math(n: int) -> list[dict]:
    dc = DownloadConfig(local_files_only=bool(int(os.environ.get("HF_DATASETS_OFFLINE", "0"))))
    ds = load_dataset("cais/mmlu", "high_school_mathematics", download_config=dc)["test"]
    out: list[dict] = []
    for r in ds:
        if len(out) >= n:
            break
        choices = [str(c) for c in r["choices"]]
        out.append(_fmt(f"Question: {r['question'].strip()}\nAnswer:", choices, int(r["answer"])))
    return out


def _adapt_hard(items: list[dict]) -> list[dict]:
    return [
        {
            "prompt": it["prompt"],
            "choices": it["choices"],
            "label": it.get("valid_indices", [0])[0],
        }
        for it in items
    ]


def load_logiqa_fixed(n: int) -> list[dict]:
    def _strip(opt: str) -> str:
        return re.sub(r"^[ABCDabcd]\.\s*", "", str(opt).strip())

    def _to_letter(r: dict) -> str | None:
        lab = r.get("label") if r.get("label") is not None else r.get("answer")
        if isinstance(lab, (int, float)) and lab == int(lab):
            i = int(lab)
            if 0 <= i < 4:
                return "abcd"[i]
        s = str(lab).strip().lower()
        if s in "abcd":
            return s
        return None

    ds = load_dataset("fireworks-ai/logiqa", split="test")
    out: list[dict] = []
    for r in ds:
        label = _to_letter(r)
        if label is None:
            continue
        opts = r["options"]
        if hasattr(opts, "tolist"):
            opts = opts.tolist()
        choices = ["a", "b", "c", "d"]
        prompt = (
            f"Passage: {r['context']}\nQuestion: {r['question']}\nChoices:\n"
            + "\n".join(f"{x.upper()}. {_strip(o)}" for x, o in zip(choices, opts))
            + "\nAnswer:"
        )
        out.append({"prompt": prompt, "choices": choices, "label": choices.index(label)})
        if len(out) >= n:
            break
    return out


BENCH_LOADERS = {
    "BoolQ": load_boolq,
    "ARC-C": load_arc_c,
    "TruthfulQA": load_truthfulqa,
    "CSQA": load_csqa,
    "MMLU-HS-Math": load_mmlu_hs_math,
    "GPQA-Diamond": lambda n: _adapt_hard(load_gpqa_diamond(n)),
    "AGIEval-Gaokao-MathQA": lambda n: _adapt_hard(load_agieval_gaokao_mathqa(n)),
    "LogiQA": load_logiqa_fixed,
}


def r29_item_to_label_item(ex: dict) -> dict:
    """r29 格式 (answer 字符串) → R38 格式 (label 下标)."""
    choices = ex["choices"]
    gold = str(ex["answer"]).strip().lower()
    label = next(i for i, c in enumerate(choices) if str(c).strip().lower() == gold)
    return {"prompt": ex["prompt"], "choices": choices, "label": label}


def load_benchmark_any(name: str, n: int) -> list[dict]:
    raw = BENCH_LOADERS[name](n)
    out: list[dict] = []
    for ex in raw[:n]:
        if "label" in ex:
            out.append(ex)
        else:
            out.append(r29_item_to_label_item(ex))
    return out


# ─── 从扫参 JSON 提取每 benchmark 最优窗口 ─────────────────────────────────────
def _best_window_for_benchmark(sweep: dict, bench: str) -> tuple[int, int, float]:
    rows = sweep.get("results", [])
    best_acc = -1.0
    best_ts, best_te = 1, 2
    for row in rows:
        if bench not in row:
            continue
        acc = float(row[bench])
        if acc > best_acc:
            best_acc = acc
            best_ts = int(row["t_start"])
            best_te = int(row["t_stop"])
    return best_ts, best_te, best_acc


def merge_sweep_best(preset: dict) -> dict[str, tuple[int, int]]:
    """合并主扫参（5 bench）与 hard_mc（3 bench）的每任务最优 (t_start, t_stop)."""
    out: dict[str, tuple[int, int]] = {}
    for path_key in ("sweep_main", "sweep_hard"):
        p = preset[path_key]
        if not p.exists():
            print(f"  [WARN] 缺少扫参文件: {p}，该文件中的 benchmark 将无 sweep_best")
            continue
        sweep = json.loads(p.read_text(encoding="utf-8"))
        for b in sweep.get("benchmarks_used", []):
            ts, te, _ = _best_window_for_benchmark(sweep, b)
            out[b] = (ts, te)
    return out


# ─── 工具 ────────────────────────────────────────────────────────────────────
def safe_cos(u: torch.Tensor, v: torch.Tensor) -> float:
    u = u.float().reshape(-1).cpu()
    v = v.float().reshape(-1).cpu()
    nu, nv = u.norm(), v.norm()
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    return float((u @ v / (nu * nv)).clamp(-1, 1).item())


def loglikelihood_mc(logits: torch.Tensor, input_ids: torch.Tensor, prompt_len: int) -> float:
    total = 0.0
    for i in range(prompt_len, input_ids.shape[1]):
        logp = F.log_softmax(logits[0, i - 1].float(), dim=-1)
        total += float(logp[input_ids[0, i]].item())
    return total


def mlp_counterfactual(layer, hi: torch.Tensor, arch: str) -> torch.Tensor:
    """Term1 用：对层输入 hi 走「错误顺序」的 FFN 路径（与 R37 一致思想；Gemma2 用 pre_ffn norm）。"""
    if arch == "gemma2":
        h = layer.pre_feedforward_layernorm(hi)
    else:
        h = layer.post_attention_layernorm(hi)
    return layer.mlp(h)


def load_model(path: str):
    print(f"Loading model from {path} …")
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=True,
    )
    mdl.eval()
    return tok, mdl, mdl.config.num_hidden_layers


@torch.no_grad()
def probe_forward_collect_cos_res(
    model,
    input_ids: torch.Tensor,
    attn_mask: torch.Tensor | None,
    n_layers: int,
    probe_layers: list[int],
    arch: str,
) -> dict[int, float]:
    base = model.model
    h_inputs: dict[int, torch.Tensor] = {}
    a_outputs: dict[int, torch.Tensor] = {}
    m_outputs: dict[int, torch.Tensor] = {}
    hooks: list = []

    for li in range(n_layers):
        layer = base.layers[li]

        def make_pre(idx: int):
            def fn(_m, args):
                t = args[0] if isinstance(args, tuple) else args
                h_inputs[idx] = t[:, -1:, :].detach().clone()

            return fn

        def make_attn_post(idx: int):
            def fn(_m, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                a_outputs[idx] = t[:, -1:, :].detach().clone()

            return fn

        def make_mlp_post(idx: int):
            def fn(_m, _inp, out):
                m_outputs[idx] = out[:, -1:, :].detach().clone()

            return fn

        hooks.append(layer.register_forward_pre_hook(make_pre(li)))
        hooks.append(layer.self_attn.register_forward_hook(make_attn_post(li)))
        hooks.append(layer.mlp.register_forward_hook(make_mlp_post(li)))

    model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
    for h in hooks:
        h.remove()

    cos_res: dict[int, float] = {}
    for li in probe_layers:
        hi = h_inputs.get(li)
        al = a_outputs.get(li)
        ml = m_outputs.get(li)
        if hi is None or al is None or ml is None:
            continue
        try:
            layer = base.layers[li]
            m_l0 = mlp_counterfactual(layer, hi, arch)
            term1 = (ml - m_l0).squeeze()
            delta_h = (al + ml).squeeze()
            cos_res[li] = safe_cos(term1, delta_h)
        except Exception:
            pass

    return cos_res


def calibrate_profile(
    items: list[dict],
    model,
    tok,
    n_layers: int,
    probe_layers: list[int],
    arch: str,
    device: str,
) -> dict[int, float]:
    acc: dict[int, list[float]] = defaultdict(list)
    for item in items[:N_CALIB]:
        enc = tok(item["prompt"], return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        amask = enc.get("attention_mask")
        if amask is not None:
            amask = amask.to(device)
        cr = probe_forward_collect_cos_res(model, ids, amask, n_layers, probe_layers, arch)
        for li, v in cr.items():
            acc[li].append(v)
    return {li: float(np.mean(vs)) for li, vs in acc.items() if vs}


def derive_global_window(
    profile: dict[int, float],
    n_t: int,
    min_start: int,
    max_start: int,
) -> tuple[int, int]:
    best_start = min_start
    best_score = -999.0
    for start in range(min_start, max_start + 1):
        stop = start + n_t
        vals = [profile[l] for l in profile if start <= l < stop]
        if len(vals) < 2:
            continue
        score = float(np.mean(vals))
        if score > best_score:
            best_score = score
            best_start = start
    return best_start, best_start + n_t


def derive_onset_adaptive(
    profile: dict[int, float],
    ratio: float,
    n_t: int,
    min_start: int,
    max_start: int,
) -> tuple[int, int]:
    valid = {l: v for l, v in profile.items() if min_start <= l <= max_start}
    if not valid:
        return min_start, min_start + n_t
    max_val = max(valid.values())
    thr = max_val * ratio
    for l in sorted(valid):
        if valid[l] >= thr:
            return l, l + n_t
    t_start = max(valid, key=valid.__getitem__)
    return t_start, t_start + n_t


def select_window_persample(
    cos_res: dict[int, float],
    n_t: int,
    min_start: int,
    max_start: int,
) -> tuple[int, int]:
    best_start = min_start
    best_score = -999.0
    for start in range(min_start, max_start + 1):
        stop = start + n_t
        vals = [cos_res[l] for l in cos_res if start <= l < stop]
        if len(vals) < 2:
            continue
        score = float(np.mean(vals))
        if score > best_score:
            best_score = score
            best_start = start
    return best_start, best_start + n_t


def select_window_variable_nt(
    cos_res: dict[int, float],
    nt_candidates: tuple[int, ...],
    min_start: int,
    max_start: int,
) -> tuple[int, int]:
    best_start = min_start
    best_nt = nt_candidates[0]
    best_score = -999.0
    for n_t in nt_candidates:
        for start in range(min_start, max_start + 1):
            stop = start + n_t
            vals = [cos_res[l] for l in cos_res if start <= l < stop]
            if len(vals) < 2:
                continue
            score = float(np.mean(vals))
            if score > best_score:
                best_score = score
                best_start = start
                best_nt = n_t
    return best_start, best_start + best_nt


def select_window_onset_fixed(
    cos_res: dict[int, float],
    threshold: float,
    n_t: int,
    min_start: int,
    max_start: int,
) -> tuple[int, int]:
    for l in sorted(l for l in cos_res if min_start <= l <= max_start):
        if cos_res[l] >= threshold:
            return l, l + n_t
    return select_window_persample(cos_res, n_t, min_start, max_start)


def mc_predict(
    model,
    tok,
    item: dict,
    device: str,
    n_e: int | None = None,
    n_t: int | None = None,
    k: int = K_ETD,
) -> int:
    prompt = item["prompt"]
    choices = item["choices"]
    scores: list[float] = []
    for cont in choices:
        full = prompt + " " + cont
        enc = tok(full, return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        amask = enc.get("attention_mask")
        if amask is not None:
            amask = amask.to(device)
        plen = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
        if n_e is not None and n_t is not None and n_t > 0:
            alpha = min(1.0, 6.0 / max(n_t, 1))
            lgts = etd_forward_logits(model, ids, amask, n_e=n_e, n_t=n_t, k=k, alpha=alpha)
        else:
            lgts = baseline_forward_logits(model, ids, amask)
        scores.append(loglikelihood_mc(lgts, ids, plen))
    return int(np.argmax(scores))


def evaluate_benchmark(
    bench: str,
    items: list[dict],
    model,
    tok,
    n_layers: int,
    sweep_win: tuple[int, int],
    preset: dict,
    device: str,
) -> dict:
    arch = preset["arch"]
    probe_layers = preset["probe_layers"]
    min_start = preset["min_start"]
    max_start = preset["max_start"]
    n_total = len(items)

    print(f"  [标定] 前 {N_CALIB} 条聚合 cos_res …")
    t0c = time.time()
    mean_profile = calibrate_profile(items, model, tok, n_layers, probe_layers, arch, device)
    calib_g = derive_global_window(mean_profile, CALIB_NT, min_start, max_start)
    calib_o = derive_onset_adaptive(mean_profile, CALIB_ONSET_RATIO, CALIB_NT, min_start, max_start)
    print(f"  [标定] {time.time() - t0c:.1f}s  calib_global8={calib_g} calib_onset8={calib_o} sweep_best={sweep_win}")

    correct = {c: 0 for c in COND_NAMES}
    sel = {c: [] for c in ["persample_cos8", "persample_var", "onset_fixed8"]}
    t0 = time.time()

    for i, item in enumerate(items):
        label = item["label"]
        enc = tok(item["prompt"], return_tensors="pt", add_special_tokens=False)
        probe_ids = enc["input_ids"].to(device)
        probe_mask = enc.get("attention_mask")
        if probe_mask is not None:
            probe_mask = probe_mask.to(device)
        cos_res = probe_forward_collect_cos_res(
            model, probe_ids, probe_mask, n_layers, probe_layers, arch
        )

        ps8 = select_window_persample(cos_res, 8, min_start, max_start)
        var = select_window_variable_nt(cos_res, NT_VAR, min_start, max_start)
        onf = select_window_onset_fixed(cos_res, ONSET_THRESHOLD, 8, min_start, max_start)
        sel["persample_cos8"].append(ps8[0])
        sel["persample_var"].append(var[0])
        sel["onset_fixed8"].append(onf[0])

        wins = {
            "baseline": None,
            "sweep_best": sweep_win,
            "persample_cos8": ps8,
            "persample_var": var,
            "onset_fixed8": onf,
            "calib_onset8": calib_o,
            "calib_global8": calib_g,
        }

        for cname in COND_NAMES:
            win = wins[cname]
            if win is None:
                pred = mc_predict(model, tok, item, device)
            else:
                ts, te = win
                n_e, n_t = ts, te - ts
                if n_layers - te < 1 or n_t < 1:
                    pred = mc_predict(model, tok, item, device)
                else:
                    pred = mc_predict(model, tok, item, device, n_e=n_e, n_t=n_t, k=K_ETD)
            if pred == label:
                correct[cname] += 1

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_total - i - 1)
            line = f"  [{i + 1:3d}/{n_total}] "
            for cn in COND_NAMES:
                line += f"{cn[:4]}={correct[cn] / (i + 1):.3f} "
            line += f"| {elapsed:.0f}s ETA {eta:.0f}s"
            print(line)

        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    accs = {c: correct[c] / n_total for c in COND_NAMES}
    win_stats = {}
    for cn in sel:
        tss = sel[cn]
        win_stats[cn] = {
            "t_start_mean": float(np.mean(tss)) if tss else 0.0,
            "t_start_std": float(np.std(tss)) if tss else 0.0,
        }

    return {
        "benchmark": bench,
        "n": n_total,
        "elapsed_s": elapsed,
        "accuracies": accs,
        "sweep_best_window": list(sweep_win),
        "calib_global8_window": list(calib_g),
        "calib_onset8_window": list(calib_o),
        "mean_profile": {str(k): v for k, v in sorted(mean_profile.items())},
        "window_stats": win_stats,
    }


def plot_summary(all_results: dict[str, dict], preset: dict, title: str):
    fig_dir = Path(preset["out_fig"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    benches = list(all_results.keys())
    conds = COND_NAMES
    n_b = len(benches)
    n_c = len(conds)
    fig, ax = plt.subplots(figsize=(16, 5))
    x = np.arange(n_b)
    w = 0.85 / n_c
    for j, cn in enumerate(conds):
        offs = (j - (n_c - 1) / 2.0) * w
        vals = [all_results[b]["accuracies"].get(cn, 0) for b in benches]
        ax.bar(x + offs, vals, w * 0.92, label=cn)
    ax.set_xticks(x)
    ax.set_xticklabels([b[:14] for b in benches], rotation=22, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend(fontsize=6.5, ncol=2, loc="upper right")
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    p = fig_dir / "summary_multimodel.png"
    plt.savefig(p, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved figure → {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS.keys()), required=True)
    args = ap.parse_args()
    preset = PRESETS[args.preset]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Path(preset["out_json"]).parent.mkdir(parents=True, exist_ok=True)
    Path(preset["out_fig"]).mkdir(parents=True, exist_ok=True)

    sweep_best = merge_sweep_best(preset)
    print("每 benchmark sweep_best（来自扫参 JSON）:")
    for b in BENCH_ORDER:
        if b in sweep_best:
            print(f"  {b}: {sweep_best[b]}")

    tok, model, n_layers = load_model(preset["model_path"])
    print(f"num_layers={n_layers} arch={preset['arch']}")

    all_results: dict[str, dict] = {}
    out_path = Path(preset["out_json"])

    for bench in BENCH_ORDER:
        n = N_SAMPLES[bench]
        if bench not in sweep_best:
            print(f"\n[SKIP] {bench}: 无 sweep_best（缺少扫参结果）")
            continue
        sw = sweep_best[bench]
        print(f"\n{'─' * 50}\n{bench}  N={n}  sweep_best={sw}")
        try:
            items = load_benchmark_any(bench, n)
        except Exception as e:
            print(f"  [ERROR] load {bench}: {e}")
            continue
        if not items:
            continue
        try:
            all_results[bench] = evaluate_benchmark(
                bench, items, model, tok, n_layers, sw, preset, device
            )
        except Exception as e:
            print(f"  [ERROR] evaluate {bench}: {e}")
            import traceback

            traceback.print_exc()
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "preset": args.preset,
                    "model_path": preset["model_path"],
                    "arch": preset["arch"],
                    "n_layers": n_layers,
                    "probe_layers": preset["probe_layers"],
                    "min_start": preset["min_start"],
                    "max_start": preset["max_start"],
                    "sweep_best_source": {k: str(preset[k]) for k in ("sweep_main", "sweep_hard")},
                    "results": all_results,
                },
                f,
                indent=2,
            )
        print(f"  checkpoint → {out_path}")

    if all_results:
        plot_summary(
            all_results,
            preset,
            f"R38-Multimodel {args.preset}: signal vs baseline vs sweep_best",
        )

    print(f"\n完成。结果: {out_path}")


if __name__ == "__main__":
    main()
