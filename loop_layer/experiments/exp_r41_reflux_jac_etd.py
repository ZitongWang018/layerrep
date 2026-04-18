#!/usr/bin/env python3
"""
R41 (Qwen3-8B): 多选题 + 可选 GSM8K（5-shot 贪婪生成，与 R40 gsm8k 任务对齐）—
  baseline、sweep_best（固定扫参窗）、neg_cos_am 标定窗、neg_cos_am×prop_attn（compound）标定窗。
不再评测 emp_logit 与 reflux_rho_gate（保留若干辅助函数供参考）。

子集 CLI：`--benchmarks arc-c mmlu-hs-math gaokao bbh gsm8k`；
输出 JSON、`r41_accuracy_comparison.png`、`neg_cos_am_profiles.png`。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings
from typing import Any
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
for _k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(_k, None)
if "HF_DATASETS_OFFLINE" not in os.environ:
    os.environ["HF_DATASETS_OFFLINE"] = "0"

warnings.filterwarnings("ignore")

ROOT = Path("/root/autodl-tmp/loop_layer")
EXP = ROOT / "experiments"
ETD_DIR = ROOT / "ETD"
LMEVAL_ROOT = Path("/root/autodl-tmp/lm-evaluation-harness")
for p in (str(ROOT), str(EXP), str(ETD_DIR), str(LMEVAL_ROOT)):
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

from lm_eval.tasks import TaskManager

from etd_forward import baseline_forward_logits, etd_forward_logits
from hard_mc_benchmark_loaders import load_agieval_gaokao_mathqa

N_CALIB = 10
K_ETD = 2
MIN_DECODER = 8
CHAMPION_NE = 8
CHAMPION_NT = 14
EPSILON = 1.0
PROBE_LAYERS_R41 = [8, 12, 16, 20, 24, 28]

DEFAULT_BBH_TASKS = [
    "leaderboard_bbh_boolean_expressions",
    "leaderboard_bbh_causal_judgement",
    "leaderboard_bbh_date_understanding",
    "leaderboard_bbh_disambiguation_qa",
    "leaderboard_bbh_logical_deduction_three_objects",
    "leaderboard_bbh_object_counting",
]


def split_bbh_total_across_tasks(total: int, n_tasks: int) -> list[int]:
    """将 total 条样本均分到 n_tasks 个子任务（前 remainder 个多 1 条）。"""
    if n_tasks <= 0:
        return []
    base, rem = divmod(total, n_tasks)
    return [base + (1 if i < rem else 0) for i in range(n_tasks)]

# baseline（无 ETD）| sweep_best（R39C Qwen3 扫参固定窗）| 两路信号标定窗
EVAL_CONDS = [
    "baseline",
    "sweep_best",
    "neg_cos_am_calib",
    "neg_cos_am_prop_attn",
]

# R39C Qwen3-8B 各 MC 任务扫参最优窗；BBH 子任务共用一个默认窗（无单独 sweep 表）
QWEN_SWEEP_BEST: dict[str, tuple[int, int]] = {
    "ARC-C": (14, 20),
    "MMLU-HS-Math": (10, 18),
    "AGIEval-Gaokao-MathQA": (13, 20),
    "BBH": (10, 22),
    # R39C 无 GSM8K 扫参表；与 CSQA 同阶作 generation 参照窗（与 R40 行为一致）
    "GSM8K": (10, 22),
}

# GSM8K：与 exp_r40_bbh_gsm8k_etd 中 gsm8k.yaml 对齐
GSM_MAX_NEW_TOKENS = 256
GSM_STOP_SUBSTRINGS = (
    "Question:",
    "</s>",
    "<|redacte|d_im_end|>",
    "<|redacted_im_end|>",
    "<|eot_id|>",
)
GSM_STRICT_NUM_RE = re.compile(r"####\s*(\-?[0-9.,]+)")

PRESET = {
    "model_path": "/root/autodl-tmp/model_qwen",
    "n_layers": 36,
    "probe_layers": list(range(6, 33, 2)),
    "min_start": 9,
    "max_start": 26,
    "nt_candidates": (2, 4, 6, 8, 12),
    "calib_nt": 8,
}


def _num_fewshot(task) -> int:
    return int(getattr(task.config, "num_fewshot", None) or 0)


def safe_cos(u, v) -> float:
    u = u.float().reshape(-1).cpu()
    v = v.float().reshape(-1).cpu()
    nu, nv = u.norm(), v.norm()
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    return float((u @ v / (nu * nv)).clamp(-1, 1))


def loglikelihood_mc(logits: torch.Tensor, input_ids: torch.Tensor, prompt_len: int) -> float:
    total = 0.0
    for i in range(prompt_len, input_ids.shape[1]):
        logp = F.log_softmax(logits[0, i - 1].float(), dim=-1)
        total += float(logp[input_ids[0, i]])
    return total


def jsd_logits(logits1: torch.Tensor, logits2: torch.Tensor, eps: float = 1e-9) -> float:
    p = F.softmax(logits1.float().view(-1), dim=0).clamp_min(eps)
    q = F.softmax(logits2.float().view(-1), dim=0).clamp_min(eps)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum()
    kl_qm = (q * (q.log() - m.log())).sum()
    return float((0.5 * (kl_pm + kl_qm)).clamp_min(0.0).item())


def load_model(path: str):
    print(f"Loading model: {path}")
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=True,
    )
    mdl.eval()
    return tok, mdl


def _fmt(p, c, l):
    return {"prompt": p, "choices": [x.strip() for x in c], "label": l}


def load_boolq(n: int):
    ds = load_dataset("aps/super_glue", "boolq")["validation"]
    out = []
    for r in ds:
        if len(out) >= n:
            break
        lab = int(r["label"])
        if lab < 0:
            continue
        out.append(_fmt(f"{r['passage']}\nQuestion: {r['question']}?\nAnswer:", ["no", "yes"], lab))
    return out


def load_arc_c(n: int):
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")["test"]
    out = []
    for r in ds:
        if len(out) >= n:
            break
        key = r["answerKey"]
        label = ord(key) - ord("A") if key in "ABCD" else int(key) - 1
        out.append(_fmt(f"Question: {r['question'].strip()}\nAnswer:", r["choices"]["text"], label))
    return out


def load_mmlu_hs_math(n: int):
    dc = DownloadConfig(local_files_only=True)
    ds = load_dataset("cais/mmlu", "high_school_mathematics", download_config=dc)["test"]
    out = []
    for r in ds:
        if len(out) >= n:
            break
        out.append(
            _fmt(
                f"Question: {r['question'].strip()}\nAnswer:",
                [str(c) for c in r["choices"]],
                int(r["answer"]),
            )
        )
    return out


def _adapt(items):
    return [
        {"prompt": it["prompt"], "choices": it["choices"], "label": it.get("valid_indices", [0])[0]}
        for it in items
    ]


def load_gaokao_mc(n: int):
    return _adapt(load_agieval_gaokao_mathqa(n))


def bbh_docs_to_mc_items(task, docs: list) -> list[dict]:
    items = []
    for doc in docs:
        ctx = task.fewshot_context(doc, _num_fewshot(task))
        choices_raw = task.doc_to_choice(doc)
        tgt = str(task.doc_to_target(doc)).strip()
        choices = [str(c).strip() for c in choices_raw]
        if tgt not in choices:
            hit = next((i for i, c in enumerate(choices) if c.lower() == tgt.lower()), None)
            if hit is None:
                continue
            label = hit
        else:
            label = choices.index(tgt)
        items.append({"prompt": ctx, "choices": choices, "label": label})
    return items


def gsm_docs_to_prompt_items(gsm_task, docs: list) -> list[dict]:
    """用于 neg_cos / compound 标定（仅需 prompt）。"""
    items = []
    for doc in docs:
        ctx = gsm_task.fewshot_context(doc, _num_fewshot(gsm_task))
        items.append({"prompt": ctx, "choices": ["0"], "label": 0})
    return items


def extract_gsm_number(text: str) -> str | None:
    m = GSM_STRICT_NUM_RE.search(text)
    return m.group(1).replace(",", "") if m else None


def gsm_exact_match(gen_text: str, doc, gsm_task) -> bool:
    pred = extract_gsm_number(gen_text)
    gold_full = str(gsm_task.doc_to_target(doc))
    gold = extract_gsm_number(gold_full)
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return pred.strip() == gold.strip()


@torch.no_grad()
def greedy_generate_gsm(
    model,
    tok,
    prompt: str,
    device: torch.device,
    n_e: int | None,
    n_t: int | None,
    max_new_tokens: int,
) -> str:
    """与 R40 一致：每步 baseline_forward 或 etd_forward_logits（贪婪）。"""
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask")
    attn = attn.to(device) if attn is not None else torch.ones_like(input_ids)
    eos_id = tok.eos_token_id
    out_ids: list[int] = []
    for _ in range(max_new_tokens):
        if n_e is not None and n_t is not None and n_t > 0:
            logits = etd_forward_logits(
                model,
                input_ids,
                attn,
                n_e=n_e,
                n_t=n_t,
                k=K_ETD,
                alpha=min(1.0, 6.0 / max(n_t, 1)),
            )
        else:
            logits = baseline_forward_logits(model, input_ids, attn)
        next_id = int(logits[0, -1].float().argmax())
        out_ids.append(next_id)
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device, dtype=input_ids.dtype)], dim=1)
        attn = torch.cat([attn, torch.ones(1, 1, device=device, dtype=attn.dtype)], dim=1)
        if eos_id is not None and next_id == eos_id:
            break
        dec = tok.decode(out_ids, skip_special_tokens=True)
        if any(s in dec for s in GSM_STOP_SUBSTRINGS):
            break
    return tok.decode(out_ids, skip_special_tokens=True)


def _gsm_etd_args(
    n_layers: int, cond: str, win_sweep: tuple[int, int], win_calib: tuple[int, int], win_prop: tuple[int, int]
) -> tuple[int | None, int | None]:
    if cond == "baseline":
        return None, None
    if cond == "sweep_best":
        ts, te = win_sweep
        nt = te - ts
        if n_layers - te < 2 or nt < 1:
            return None, None
        return ts, nt
    if cond == "neg_cos_am_calib":
        ts, te = win_calib
        nt = te - ts
        if n_layers - te < 2 or nt < 1:
            return None, None
        return ts, nt
    if cond == "neg_cos_am_prop_attn":
        ts, te = win_prop
        nt = te - ts
        if n_layers - te < 2 or nt < 1:
            return None, None
        return ts, nt
    return None, None


def evaluate_gsm8k_benchmark(
    bench_name: str,
    gsm_task,
    docs: list,
    model,
    tok,
    device: torch.device,
    sweep_lookup_key: str,
) -> dict:
    n_layers = PRESET["n_layers"]
    probe_layers = PRESET["probe_layers"]
    min_s, max_s = PRESET["min_start"], PRESET["max_start"]
    calib_nt = PRESET["calib_nt"]
    items = gsm_docs_to_prompt_items(gsm_task, docs)
    n_total = len(items)
    if n_total < 1:
        raise ValueError(f"{bench_name}: no docs")
    win_sweep = QWEN_SWEEP_BEST[sweep_lookup_key]
    t0 = time.time()
    print(f"  [{bench_name}] neg_cos calib …")
    mean_profile = calibrate_neg_cos_profile(items, model, tok, n_layers, probe_layers, device)
    win_calib = select_calib_global(mean_profile, calib_nt, min_s, max_s)
    print(f"  [{bench_name}] compound (neg_cos × JSD attn) …")
    mean_comp = calibrate_compound_profile(items, model, tok, n_layers, PROBE_LAYERS_R41, device)
    win_prop = select_calib_global(mean_comp, calib_nt, min_s, max_s)
    correct = {c: 0 for c in EVAL_CONDS}
    for i, doc in enumerate(docs):
        ctx = items[i]["prompt"]
        for c in EVAL_CONDS:
            ne, nt = _gsm_etd_args(n_layers, c, win_sweep, win_calib, win_prop)
            gen = greedy_generate_gsm(model, tok, ctx, device, ne, nt, GSM_MAX_NEW_TOKENS)
            if gsm_exact_match(gen, doc, gsm_task):
                correct[c] += 1
        if (i + 1) % 5 == 0 or n_total <= 5:
            accs = {x: correct[x] / (i + 1) for x in EVAL_CONDS}
            print(f"    [{i+1}/{n_total}] " + " ".join(f"{x[:8]}={accs[x]:.2f}" for x in EVAL_CONDS))
        torch.cuda.empty_cache()
    accs = {c: correct[c] / max(n_total, 1) for c in EVAL_CONDS}
    bl = accs["baseline"]
    print(
        f"  [{bench_name}] {time.time() - t0:.0f}s baseline={bl:.3f} "
        f"sweep={accs['sweep_best']:.3f} calib={accs['neg_cos_am_calib']:.3f} prop={accs['neg_cos_am_prop_attn']:.3f}"
    )
    return {
        "benchmark": bench_name,
        "n": n_total,
        "accuracies": accs,
        "delta_vs_baseline": {c: accs[c] - bl for c in EVAL_CONDS},
        "sweep_lookup_key": sweep_lookup_key,
        "sweep_best_window": list(win_sweep),
        "neg_cos_am_calib_win": list(win_calib),
        "neg_cos_am_prop_attn_win": list(win_prop),
        "mean_profile": {str(k): float(v) for k, v in sorted(mean_profile.items())},
        "max_new_tokens": GSM_MAX_NEW_TOKENS,
    }


@torch.no_grad()
def probe_neg_cos_am(
    model, input_ids: torch.Tensor, attn_mask: torch.Tensor | None, n_layers: int, probe_layers: list[int]
) -> dict[int, float]:
    base = model.model
    a_out: dict[int, torch.Tensor] = {}
    m_out: dict[int, torch.Tensor] = {}
    hooks = []
    for li in range(n_layers):

        def _pa(idx):
            def fn(_m, _i, out):
                t = out[0] if isinstance(out, tuple) else out
                a_out[idx] = t[:, -1:, :].detach().clone()

            return fn

        def _pm(idx):
            def fn(_m, _i, out):
                m_out[idx] = out[:, -1:, :].detach().clone()

            return fn

        hooks.append(base.layers[li].self_attn.register_forward_hook(_pa(li)))
        hooks.append(base.layers[li].mlp.register_forward_hook(_pm(li)))
    model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
    for h in hooks:
        h.remove()
    result: dict[int, float] = {}
    for li in probe_layers:
        al, ml = a_out.get(li), m_out.get(li)
        if al is None or ml is None:
            continue
        result[li] = -safe_cos(al.squeeze(), ml.squeeze())
    return result


@torch.no_grad()
def probe_attn_mlp_logits(
    model, input_ids: torch.Tensor, attn_mask: torch.Tensor | None, n_layers: int, probe_layers: list[int]
):
    base = model.model
    a_out: dict[int, torch.Tensor] = {}
    m_out: dict[int, torch.Tensor] = {}
    hooks = []
    for li in range(n_layers):

        def _pa(idx):
            def fn(_m, _i, out):
                t = out[0] if isinstance(out, tuple) else out
                a_out[idx] = t[:, -1:, :].detach().clone()

            return fn

        def _pm(idx):
            def fn(_m, _i, out):
                m_out[idx] = out[:, -1:, :].detach().clone()

            return fn

        hooks.append(base.layers[li].self_attn.register_forward_hook(_pa(li)))
        hooks.append(base.layers[li].mlp.register_forward_hook(_pm(li)))
    out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
    for h in hooks:
        h.remove()
    orig = out.logits[:, -1, :].float()
    negcos: dict[int, float] = {}
    a_cpu: dict[int, torch.Tensor] = {}
    for li in probe_layers:
        al, ml = a_out.get(li), m_out.get(li)
        if al is None or ml is None:
            continue
        negcos[li] = -safe_cos(al.squeeze(), ml.squeeze())
        a_cpu[li] = al.squeeze(0).squeeze(0).float().cpu()
    return negcos, a_cpu, orig


@torch.no_grad()
def perturb_forward_last_logits(
    model, input_ids: torch.Tensor, attn_mask: torch.Tensor | None, probe_layer: int, perturb_1d: torch.Tensor
) -> torch.Tensor:
    base = model.model
    perturb_cpu = perturb_1d.float().cpu()

    def hook_fn(module, args):
        t = args[0] if isinstance(args, tuple) else args
        t_out = t.clone()
        delta = perturb_cpu.to(device=t.device, dtype=t.dtype)
        t_out[:, -1, :] = t_out[:, -1, :] + delta
        if isinstance(args, tuple):
            return (t_out,) + args[1:]
        return t_out

    hook = base.layers[probe_layer].register_forward_pre_hook(hook_fn)
    out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
    hook.remove()
    return out.logits[:, -1, :].float()


def compound_profile_one_item(
    model, input_ids: torch.Tensor, attn_mask: torch.Tensor | None, n_layers: int, probe_layers: list[int]
) -> dict[int, float]:
    negcos, a_cpu, orig = probe_attn_mlp_logits(model, input_ids, attn_mask, n_layers, probe_layers)
    compound: dict[int, float] = {}
    for li in probe_layers:
        if li not in negcos or li not in a_cpu:
            continue
        v = a_cpu[li]
        nv = float(v.norm().item())
        if nv < 1e-12:
            continue
        a_hat = v / nv
        pert = perturb_forward_last_logits(model, input_ids, attn_mask, li, EPSILON * a_hat)
        compound[li] = float(negcos[li]) * jsd_logits(orig, pert)
    return compound


def calibrate_neg_cos_profile(
    items: list[dict], model, tok, n_layers: int, probe_layers: list[int], device: torch.device
) -> dict[int, float]:
    acc: dict[int, list] = defaultdict(list)
    for item in items[:N_CALIB]:
        enc = tok(item["prompt"], return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        am = enc.get("attention_mask")
        am = am.to(device) if am is not None else None
        sig = probe_neg_cos_am(model, ids, am, n_layers, probe_layers)
        for li, v in sig.items():
            acc[li].append(v)
    return {li: float(np.mean(vs)) for li, vs in acc.items() if vs}


def calibrate_compound_profile(
    items: list[dict], model, tok, n_layers: int, probe_layers: list[int], device: torch.device
) -> dict[int, float]:
    acc: dict[int, list] = defaultdict(list)
    for item in items[:N_CALIB]:
        enc = tok(item["prompt"], return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        am = enc.get("attention_mask")
        am = am.to(device) if am is not None else None
        cp = compound_profile_one_item(model, ids, am, n_layers, probe_layers)
        for li, v in cp.items():
            acc[li].append(v)
        torch.cuda.empty_cache()
    return {li: float(np.mean(vs)) for li, vs in acc.items() if vs}


def select_calib_global(profile: dict[int, float], n_t: int, min_start: int, max_start: int) -> tuple[int, int]:
    best_start, best_score = min_start, -999.0
    for start in range(min_start, max_start + 1):
        vals = [profile[l] for l in profile if start <= l < start + n_t]
        if not vals:
            continue
        score = float(np.mean(vals))
        if score > best_score:
            best_score, best_start = score, start
    return best_start, best_start + n_t


def run_empirical_logit(
    items: list[dict],
    model,
    tok,
    n_layers: int,
    min_start: int,
    max_start: int,
    nt_candidates: tuple,
    device: torch.device,
) -> tuple[int, int] | None:
    max_t_stop = n_layers - MIN_DECODER
    stride = max(2, (max_start - min_start) // 5)
    candidates: list[tuple[int, int]] = []
    for n_t in nt_candidates:
        for ts in range(min_start, max_start + 1, stride):
            te = ts + n_t
            if te > max_t_stop:
                continue
            candidates.append((ts, te))
    if not candidates:
        return None
    gain_sum = {c: 0.0 for c in candidates}
    calib_items = items[:N_CALIB]
    for item in calib_items:
        plen = tok(item["prompt"], return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
        cont = item["choices"][item["label"]]
        full = item["prompt"] + " " + cont
        enc = tok(full, return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        am = enc.get("attention_mask")
        am = am.to(device) if am is not None else None
        g_base = loglikelihood_mc(baseline_forward_logits(model, ids, am), ids, plen)
        for ts, te in candidates:
            n_t = te - ts
            try:
                lg = etd_forward_logits(
                    model,
                    ids,
                    am,
                    n_e=ts,
                    n_t=n_t,
                    k=K_ETD,
                    alpha=min(1.0, 6.0 / max(n_t, 1)),
                )
                gain_sum[(ts, te)] += loglikelihood_mc(lg, ids, plen) - g_base
            except Exception:
                pass
        torch.cuda.empty_cache()
    n = max(len(calib_items), 1)
    mean_gain = {c: g / n for c, g in gain_sum.items()}
    best_win, best_gain = max(mean_gain.items(), key=lambda x: x[1])
    if best_gain <= 0:
        return None
    return best_win


@torch.no_grad()
def compute_reflux_rho(model, input_ids: torch.Tensor, attn_mask: torch.Tensor | None, n_e: int, n_t: int) -> float:
    alpha = min(1.0, 6.0 / max(n_t, 1))
    lg1 = etd_forward_logits(model, input_ids, attn_mask, n_e=n_e, n_t=n_t, k=1, alpha=alpha)
    lg2 = etd_forward_logits(model, input_ids, attn_mask, n_e=n_e, n_t=n_t, k=2, alpha=alpha)
    d = lg2[0, -1].float() - lg1[0, -1].float()
    y_hat = int(lg1[0, -1].float().argmax().item())
    dn = float(d.norm().item())
    if dn < 1e-12:
        return 0.0
    return float(d[y_hat].item() / dn)


def calibrate_tau_rho(items: list[dict], model, tok, device: torch.device, n_e: int, n_t: int) -> float:
    rhos: list[float] = []
    for item in items[:N_CALIB]:
        enc = tok(item["prompt"], return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        am = enc.get("attention_mask")
        am = am.to(device) if am is not None else None
        rhos.append(compute_reflux_rho(model, ids, am, n_e, n_t))
        torch.cuda.empty_cache()
    return float(np.median(rhos)) if rhos else 0.0


@torch.no_grad()
def mc_predict_cond(
    model,
    tok,
    item: dict,
    device: torch.device,
    cond: str,
    win_calib: tuple[int, int],
    win_prop: tuple[int, int],
    win_sweep: tuple[int, int],
    n_layers: int,
) -> int:
    prompt, choices = item["prompt"], item["choices"]
    plen = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]

    def score_one(cont: str, n_e: int | None, n_t: int | None, k_etd: int | None) -> float:
        full = prompt + " " + cont
        enc = tok(full, return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        am = enc.get("attention_mask")
        am = am.to(device) if am is not None else None
        if n_e is None or n_t is None or n_t < 1 or k_etd is None:
            lg = baseline_forward_logits(model, ids, am)
        else:
            alpha = min(1.0, 6.0 / max(n_t, 1))
            lg = etd_forward_logits(model, ids, am, n_e=n_e, n_t=n_t, k=k_etd, alpha=alpha)
        return loglikelihood_mc(lg, ids, plen)

    def etd_scores(ts: int, te: int) -> list[float]:
        n_t = te - ts
        if n_layers - te < 2 or n_t < 1:
            return [score_one(c, None, None, None) for c in choices]
        return [score_one(c, ts, n_t, K_ETD) for c in choices]

    if cond == "baseline":
        return int(np.argmax([score_one(c, None, None, None) for c in choices]))

    if cond == "sweep_best":
        ts, te = win_sweep
        return int(np.argmax(etd_scores(ts, te)))

    if cond == "neg_cos_am_calib":
        ts, te = win_calib
        return int(np.argmax(etd_scores(ts, te)))

    if cond == "neg_cos_am_prop_attn":
        ts, te = win_prop
        return int(np.argmax(etd_scores(ts, te)))

    return int(np.argmax([score_one(c, None, None, None) for c in choices]))


def evaluate_mc_benchmark(
    bench_name: str,
    items: list[dict],
    model,
    tok,
    device: torch.device,
    sweep_lookup_key: str,
) -> dict:
    n_layers = PRESET["n_layers"]
    probe_layers = PRESET["probe_layers"]
    min_s, max_s = PRESET["min_start"], PRESET["max_start"]
    calib_nt = PRESET["calib_nt"]
    n_total = len(items)
    if n_total < 1:
        raise ValueError(f"{bench_name}: empty items")
    win_sweep = QWEN_SWEEP_BEST[sweep_lookup_key]
    t0 = time.time()
    print(f"  [{bench_name}] neg_cos calib …")
    mean_profile = calibrate_neg_cos_profile(items, model, tok, n_layers, probe_layers, device)
    win_calib = select_calib_global(mean_profile, calib_nt, min_s, max_s)
    print(f"  [{bench_name}] compound (neg_cos × JSD attn) …")
    mean_comp = calibrate_compound_profile(items, model, tok, n_layers, PROBE_LAYERS_R41, device)
    win_prop = select_calib_global(mean_comp, calib_nt, min_s, max_s)
    correct = {c: 0 for c in EVAL_CONDS}
    for i, item in enumerate(items):
        label = item["label"]
        for c in EVAL_CONDS:
            pred = mc_predict_cond(
                model, tok, item, device, c, win_calib, win_prop, win_sweep, n_layers
            )
            if pred == label:
                correct[c] += 1
        if (i + 1) % 10 == 0 or n_total <= 10:
            accs = {c: correct[c] / (i + 1) for c in EVAL_CONDS}
            print(f"    [{i+1}/{n_total}] " + " ".join(f"{c[:8]}={accs[c]:.2f}" for c in EVAL_CONDS))
        torch.cuda.empty_cache()
    accs = {c: correct[c] / max(n_total, 1) for c in EVAL_CONDS}
    bl = accs["baseline"]
    print(
        f"  [{bench_name}] {time.time() - t0:.0f}s baseline={bl:.3f} "
        f"sweep={accs['sweep_best']:.3f} calib={accs['neg_cos_am_calib']:.3f} prop={accs['neg_cos_am_prop_attn']:.3f}"
    )
    return {
        "benchmark": bench_name,
        "n": n_total,
        "accuracies": accs,
        "delta_vs_baseline": {c: accs[c] - bl for c in EVAL_CONDS},
        "sweep_lookup_key": sweep_lookup_key,
        "sweep_best_window": list(win_sweep),
        "neg_cos_am_calib_win": list(win_calib),
        "neg_cos_am_prop_attn_win": list(win_prop),
        "mean_profile": {str(k): float(v) for k, v in sorted(mean_profile.items())},
    }


# MC loaders: (json key, sweep QWEN key, loader)
MC_BENCHMARK_SPECS: dict[str, tuple[str, str, Any]] = {
    "arc-c": ("ARC-C", "ARC-C", load_arc_c),
    "mmlu-hs-math": ("MMLU-HS-Math", "MMLU-HS-Math", load_mmlu_hs_math),
    "gaokao": ("AGIEval-Gaokao-MathQA", "AGIEval-Gaokao-MathQA", load_gaokao_mc),
}


def plot_r41_neg_cos_profiles(results: dict, fig_path: Path) -> None:
    """R39C-style grid: neg_cos_am calib mean + shaded sweep-best vs calib windows."""
    units: list[tuple[str, dict]] = []
    b = results.get("benchmarks", {})
    mc_order = ["ARC-C", "MMLU-HS-Math", "AGIEval-Gaokao-MathQA", "GSM8K"]
    for k in mc_order:
        if k in b and isinstance(b[k], dict) and b[k].get("mean_profile"):
            units.append((k, b[k]))
    bbh = b.get("bbh") or {}
    for short in sorted(bbh.keys()):
        sub = bbh[short]
        if isinstance(sub, dict) and sub.get("mean_profile"):
            units.append((f"BBH:{short}", sub))
    if not units:
        print("[R41] skip neg_cos_am_profiles: no mean_profile in results")
        return
    n_b = len(units)
    n_cols = 4
    n_rows = (n_b + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for idx, (title, res) in enumerate(units):
        ax = axes_flat[idx]
        prof = {int(k): float(v) for k, v in res.get("mean_profile", {}).items()}
        if not prof:
            ax.set_title(title[:24])
            continue
        layers = sorted(prof.keys())
        vals = [prof[l] for l in layers]
        ax.plot(layers, vals, "r-o", markersize=4, linewidth=1.5, label="neg_cos_am (calib. mean)")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        sw = tuple(res["sweep_best_window"])
        ax.axvspan(sw[0], sw[1], alpha=0.18, color="#1f77b4", label=f"Sweep-best [{sw[0]}, {sw[1]})")
        cw = res.get("neg_cos_am_calib_win")
        if cw:
            ax.axvspan(cw[0], cw[1], alpha=0.15, color="#9467bd", label=f"Calib [{cw[0]}, {cw[1]})")
        ax.set_title(title[:24], fontsize=8)
        ax.set_xlabel("Layer", fontsize=7)
        ax.set_ylabel("neg_cos_am", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, loc="best")
    for idx in range(len(units), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    fig.suptitle("R41 Qwen3-8B: neg_cos_am layer profile (calibration mean) vs windows", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"[R41] figure {fig_path}")


def _mc_result_complete(d: dict | None) -> bool:
    if not d or not isinstance(d, dict) or "accuracies" not in d:
        return False
    return all(c in d["accuracies"] for c in EVAL_CONDS)


def _should_skip_mc_resume(
    args: argparse.Namespace, results: dict, key: str, expected_n: int
) -> bool:
    if not getattr(args, "resume", False):
        return False
    b = results.get("benchmarks", {}).get(key)
    if not _mc_result_complete(b):
        return False
    if int(b.get("n", -1)) != int(expected_n):
        print(f"[resume] {key}: cached n={b.get('n')} != {expected_n}, re-run")
        return False
    print(f"[resume] skip {key} (n={expected_n})")
    return True


def _should_skip_bbh_resume(
    args: argparse.Namespace, results: dict, short: str, expected_n: int
) -> bool:
    if not getattr(args, "resume", False):
        return False
    b = (results.get("benchmarks") or {}).get("bbh") or {}
    sub = b.get(short)
    if not _mc_result_complete(sub):
        return False
    if int(sub.get("n", -1)) != int(expected_n):
        print(f"[resume] BBH:{short}: cached n={sub.get('n')} != {expected_n}, re-run")
        return False
    print(f"[resume] skip BBH:{short} (n={expected_n})")
    return True


def print_accuracy_table(results: dict) -> None:
    col_w = 16
    print("\n" + "=" * (34 + col_w * len(EVAL_CONDS)))
    print("R41 accuracy table (rows = benchmarks)")
    print("=" * (34 + col_w * len(EVAL_CONDS)))
    hdr = f"{'Benchmark':<34}" + "".join(f"{c[: col_w - 1]:>{col_w}}" for c in EVAL_CONDS)
    print(hdr)
    print("-" * len(hdr))
    b = results.get("benchmarks", {})
    for k in ["ARC-C", "MMLU-HS-Math", "AGIEval-Gaokao-MathQA", "GSM8K"]:
        if k not in b:
            continue
        row = f"{k:<34}" + "".join(f"{b[k]['accuracies'][c]:>{col_w}.4f}" for c in EVAL_CONDS)
        print(row)
    bbh = b.get("bbh") or {}
    for short in sorted(bbh.keys()):
        sub = bbh[short]
        name = f"BBH:{short}"
        row = f"{name:<34}" + "".join(f"{sub['accuracies'][c]:>{col_w}.4f}" for c in EVAL_CONDS)
        print(row)
    if "bbh_macro_mean" in results:
        mm = results["bbh_macro_mean"]
        row = f"{'BBH-macro (6 tasks)':<34}" + "".join(f"{mm[c]:>{col_w}.4f}" for c in EVAL_CONDS)
        print(row)
    print("=" * (34 + col_w * len(EVAL_CONDS)) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--benchmarks",
        nargs="*",
        default=["arc-c", "mmlu-hs-math", "gaokao", "bbh", "gsm8k"],
        help="Subset: arc-c, mmlu-hs-math, gaokao, bbh, gsm8k (lm-eval gsm8k, greedy EM)",
    )
    ap.add_argument("--n-samples", type=int, default=50, help="Per MC benchmark (and GSM8K if --gsm-limit unset)")
    ap.add_argument(
        "--gsm-limit",
        type=int,
        default=None,
        help="GSM8K test docs; default = --n-samples when gsm8k is selected",
    )
    ap.add_argument("--bbh-limit", type=int, default=50, help="Docs per BBH subtask (ignored if --bbh-total set)")
    ap.add_argument(
        "--bbh-total",
        type=int,
        default=None,
        help="Total BBH docs across all --bbh-tasks (split evenly); overrides --bbh-limit when set (non-smoke)",
    )
    ap.add_argument("--bbh-tasks", nargs="*", default=DEFAULT_BBH_TASKS)
    ap.add_argument("--output-json", type=str, default="")
    ap.add_argument("--smoke", action="store_true", help="1 sample per unit; quick bug check")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="若 --output-json 已存在：跳过已完成的 MC/BBH 子任务（按 n 与当前参数一致才跳过）",
    )
    args = ap.parse_args()

    n_samples = 1 if args.smoke else args.n_samples
    n_gsm = 1 if args.smoke else (args.gsm_limit if args.gsm_limit is not None else args.n_samples)
    bbh_limit = 1 if args.smoke else args.bbh_limit
    bbh_total_mode = not args.smoke and args.bbh_total is not None
    if bbh_total_mode and args.bbh_total is not None and args.bbh_total < 1:
        raise SystemExit("--bbh-total must be >= 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(
        args.output_json
        or (EXP / "results" / ("r41_qwen3_smoke.json" if args.smoke else "r41_qwen3_arc_mmlu_gaokao_bbh.json"))
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig_dir = EXP / "figures" / "r41_qwen3"
    fig_dir.mkdir(parents=True, exist_ok=True)

    sel = {x.lower().strip() for x in args.benchmarks}
    run_arc = "arc-c" in sel
    run_mmlu = "mmlu-hs-math" in sel
    run_gaokao = "gaokao" in sel
    run_bbh = "bbh" in sel
    run_gsm = "gsm8k" in sel

    tok, model = load_model(PRESET["model_path"])

    if args.resume and not args.smoke and out_path.is_file():
        try:
            results = json.loads(out_path.read_text(encoding="utf-8"))
            results.setdefault("benchmarks", {})
            print(f"[resume] loaded checkpoint: {out_path}")
        except Exception as e:
            print(f"[resume] failed to read {out_path}: {e}; starting fresh")
            results = {}
    else:
        if args.resume and args.smoke:
            print("[resume] ignored with --smoke")
        results = {}

    if not results:
        results = {
            "preset": "qwen3-8b",
            "eval_conds": EVAL_CONDS,
            "n_calib": N_CALIB,
            "probe_layers_compound": PROBE_LAYERS_R41,
            "qwen_sweep_windows": {k: list(v) for k, v in QWEN_SWEEP_BEST.items()},
            "benchmarks": {},
        }
    results["smoke"] = args.smoke
    results["resume"] = bool(args.resume)
    results["n_samples_mc"] = n_samples
    results["bbh_limit"] = bbh_limit
    results["bbh_total"] = args.bbh_total
    results["gsm_limit"] = n_gsm

    # MC benchmarks
    if run_arc:
        spec = MC_BENCHMARK_SPECS["arc-c"]
        key = spec[0]
        if not _should_skip_mc_resume(args, results, key, n_samples):
            results["benchmarks"][key] = evaluate_mc_benchmark(
                key, spec[2](n_samples), model, tok, device, spec[1]
            )
            out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    if run_mmlu:
        spec = MC_BENCHMARK_SPECS["mmlu-hs-math"]
        key = spec[0]
        if not _should_skip_mc_resume(args, results, key, n_samples):
            results["benchmarks"][key] = evaluate_mc_benchmark(
                key, spec[2](n_samples), model, tok, device, spec[1]
            )
            out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    if run_gaokao:
        spec = MC_BENCHMARK_SPECS["gaokao"]
        key = spec[0]
        if not _should_skip_mc_resume(args, results, key, n_samples):
            results["benchmarks"][key] = evaluate_mc_benchmark(
                key, spec[2](n_samples), model, tok, device, spec[1]
            )
            out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    if run_gsm:
        tm_g = TaskManager(include_defaults=True)
        loaded_g = tm_g.load(["gsm8k"])
        gsm_task = loaded_g["tasks"]["gsm8k"]
        gsm_task.set_fewshot_seed(42)
        gsm_docs = list(gsm_task.test_docs())[:n_gsm]
        gkey = "GSM8K"
        if not _should_skip_mc_resume(args, results, gkey, n_gsm):
            results["benchmarks"][gkey] = evaluate_gsm8k_benchmark(
                gkey, gsm_task, gsm_docs, model, tok, device, "GSM8K"
            )
            out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    if run_bbh:
        tm = TaskManager(include_defaults=True)
        results["benchmarks"].setdefault("bbh", {})
        n_bbh_tasks = len(args.bbh_tasks)
        quotas = (
            split_bbh_total_across_tasks(args.bbh_total, n_bbh_tasks)
            if bbh_total_mode
            else None
        )
        if bbh_total_mode and quotas is not None:
            results.setdefault("bbh_per_subtask_n", {})
            print(f"[BBH] total budget={args.bbh_total} split as {quotas}")
        for ti, task_name in enumerate(args.bbh_tasks):
            try:
                loaded = tm.load([task_name])
                task = loaded["tasks"][task_name]
            except Exception as e:
                print(f"[WARN] skip BBH {task_name}: {e}")
                continue
            task.set_fewshot_seed(42)
            if args.smoke:
                this_limit = 1
            elif bbh_total_mode and quotas is not None:
                this_limit = quotas[ti]
            else:
                this_limit = bbh_limit
            if this_limit < 1:
                print(f"[WARN] BBH {task_name}: quota 0, skip")
                continue
            docs = list(task.test_docs())[:this_limit]
            items = bbh_docs_to_mc_items(task, docs)
            if not items:
                print(f"[WARN] BBH {task_name}: no items")
                continue
            short = task_name.replace("leaderboard_bbh_", "")
            if _should_skip_bbh_resume(args, results, short, len(items)):
                continue
            if bbh_total_mode:
                results["bbh_per_subtask_n"][short] = len(items)
            results["benchmarks"]["bbh"][short] = evaluate_mc_benchmark(
                f"BBH:{short}", items, model, tok, device, "BBH"
            )
            out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    bbh = results["benchmarks"].get("bbh") or {}
    if bbh:
        macro = {c: [] for c in EVAL_CONDS}
        for _k, sub in bbh.items():
            for c in EVAL_CONDS:
                macro[c].append(sub["accuracies"][c])
        results["bbh_macro_mean"] = {c: float(np.mean(macro[c])) for c in EVAL_CONDS}
        results["bbh_macro_delta_vs_baseline"] = {
            c: results["bbh_macro_mean"][c] - results["bbh_macro_mean"]["baseline"] for c in EVAL_CONDS
        }

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[R41] wrote {out_path}")
    print_accuracy_table(results)

    plot_accs: dict[str, dict[str, float]] = {}
    for k, v in results["benchmarks"].items():
        if k == "bbh":
            continue
        plot_accs[k] = {c: v["accuracies"][c] for c in EVAL_CONDS}
    if bbh and "bbh_macro_mean" in results:
        plot_accs["BBH-macro"] = {c: results["bbh_macro_mean"][c] for c in EVAL_CONDS}
    if plot_accs:
        fig_labels = {
            "baseline": "baseline",
            "sweep_best": "sweep_best",
            "neg_cos_am_calib": "neg_cos_am_calib",
            "neg_cos_am_prop_attn": "neg_cos_am_prop_attn",
        }
        fig_colors = {
            "baseline": "#37474F",
            "sweep_best": "#1f77b4",
            "neg_cos_am_calib": "#1565C0",
            "neg_cos_am_prop_attn": "#2E7D32",
        }
        n_c = len(EVAL_CONDS)
        fig, ax = plt.subplots(figsize=(max(8, len(plot_accs) * 1.35), 5))
        x = np.arange(len(plot_accs))
        w = 0.78 / max(n_c, 1)
        for j, c in enumerate(EVAL_CONDS):
            ax.bar(
                x + (j - (n_c - 1) / 2) * w,
                [plot_accs[b][c] for b in plot_accs],
                w * 0.92,
                label=fig_labels[c],
                color=fig_colors[c],
                edgecolor="white",
                linewidth=0.35,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(list(plot_accs.keys()), rotation=18, ha="right")
        ax.set_ylabel("Accuracy")
        ax.set_title("R41 Qwen3-8B: baseline vs sweep_best vs neg_cos_calib vs neg_cos×prop_attn")
        ax.legend(fontsize=7, ncol=2)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fp = fig_dir / "r41_accuracy_comparison.png"
        fig.savefig(fp, dpi=140, bbox_inches="tight")
        plt.close()
        print(f"[R41] figure {fp}")

    plot_r41_neg_cos_profiles(results, fig_dir / "neg_cos_am_profiles.png")


if __name__ == "__main__":
    main()
