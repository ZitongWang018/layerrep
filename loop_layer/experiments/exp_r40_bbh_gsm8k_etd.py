#!/usr/bin/env python3
"""
R40: BBH（6 子任务×limit）+ GSM8K（5-shot CoT generate_until），三模型 × 四条件
（baseline + neg_cos_am_calib + emp_logit_fixed + neg_cos_am_ps_nt）。

- BBH：复用 lm-eval Task（SaylorTwift/bbh），multiple_choice + fewshot_context 与官方一致。
- GSM8K：与 gsm8k.yaml 对齐；贪婪解码；每步 baseline_forward / etd_forward_logits。
- Gemma2：min_decoder 放宽；结果中可标记 degraded。

进度：tqdm + 当前 preset/子任务/方法 + 阶段已用时间 + 阶段 ETA + 全程 ETA（warmup 后校准）。
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
# 镜像直连时禁用代理，避免 httpx 经企业代理 TLS 握手超时。
for _proxy_key in (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
):
    os.environ.pop(_proxy_key, None)
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
# BBH 使用 SaylorTwift/bbh，多数环境尚未缓存；默认**不**强制离线，否则 Hub 不可达即报错。
# 若需纯离线：先缓存数据集后 export HF_DATASETS_OFFLINE=1
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

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from etd_forward import baseline_forward_logits, etd_forward_logits

from lm_eval.tasks import TaskManager

# ─────────────────────────────────────────────────────────────────────────────
N_CALIB = 20
K_ETD = 2
DEFAULT_MIN_DECODER = 8

# 默认 6 个 BBH 子任务（lm-eval task 名）
DEFAULT_BBH_TASKS = [
    "leaderboard_bbh_boolean_expressions",
    "leaderboard_bbh_causal_judgement",
    "leaderboard_bbh_date_understanding",
    "leaderboard_bbh_disambiguation_qa",
    "leaderboard_bbh_logical_deduction_three_objects",
    "leaderboard_bbh_object_counting",
]

METHODS = ("baseline", "neg_cos_am_calib", "emp_logit_fixed", "neg_cos_am_ps_nt")

PRESETS: dict[str, dict] = {
    "qwen3-8b": {
        "model_path": "/root/autodl-tmp/model_qwen",
        "n_layers": 36,
        "probe_layers": list(range(6, 33, 2)),
        "min_start": 9,
        "max_start": 26,
        "nt_candidates": (2, 4, 6, 8, 12),
        "calib_nt": 8,
        "min_decoder": DEFAULT_MIN_DECODER,
    },
    "llama3-8b": {
        "model_path": "/root/autodl-tmp/Llama3-8B",
        "n_layers": 32,
        "probe_layers": list(range(6, 27, 2)),
        "min_start": 8,
        "max_start": 20,
        "nt_candidates": (2, 4, 6, 8),
        "calib_nt": 6,
        "min_decoder": DEFAULT_MIN_DECODER,
    },
    "gemma2-2b": {
        "model_path": "/root/autodl-tmp/Gemma2-2B",
        "n_layers": 26,
        "probe_layers": list(range(4, 23, 2)),
        "min_start": 5,
        "max_start": 16,
        "nt_candidates": (2, 4, 8, 14),
        "calib_nt": 8,
        "min_decoder": max(3, 26 // 8),  # 3：避免 max_t_stop 过小
    },
}

# GSM8K 与 gsm8k.yaml generation_kwargs.until 对齐（并加常见 EOS 串）
GSM_MAX_NEW_TOKENS = 256
GSM_STOP_SUBSTRINGS = (
    "Question:",
    "</s>",
    "<|redacte|d_im_end|>",
    "<|im_end|>",
    "<|eot_id|>",
)
GSM_STRICT_NUM_RE = re.compile(r"####\s*(\-?[0-9.,]+)")


def _num_fewshot(task) -> int:
    n = getattr(task.config, "num_fewshot", None) or 0
    return int(n)


# ─────────────────────────────────────────────────────────────────────────────
class EtaState:
    """全程 ETA：用已完成 wall 时间 / 已完成权重 估计剩余。"""

    def __init__(self, total_weight: float):
        self.total_weight = max(total_weight, 1.0)
        self.done_weight = 0.0
        self.t0 = time.time()

    def add(self, w: float) -> tuple[float, float, float]:
        self.done_weight += w
        elapsed = time.time() - self.t0
        if self.done_weight <= 1e-6:
            return elapsed, float("nan"), float("nan")
        rate = self.done_weight / elapsed
        rem_w = max(self.total_weight - self.done_weight, 0.0)
        eta_rem = rem_w / rate if rate > 0 else float("nan")
        eta_total = elapsed + eta_rem if not np.isnan(eta_rem) else float("nan")
        return elapsed, eta_rem, eta_total


def safe_cos(u, v) -> float:
    u = u.float().reshape(-1).cpu()
    v = v.float().reshape(-1).cpu()
    nu, nv = u.norm(), v.norm()
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    return float((u @ v / (nu * nv)).clamp(-1, 1))


def loglikelihood_mc(logits, input_ids, prompt_len: int) -> float:
    total = 0.0
    for i in range(prompt_len, input_ids.shape[1]):
        logp = F.log_softmax(logits[0, i - 1].float(), dim=-1)
        total += float(logp[input_ids[0, i]])
    return total


@torch.no_grad()
def probe_neg_cos_am(model, input_ids, attn_mask, n_layers, probe_layers) -> dict[int, float]:
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


def calibrate_mc_items(
    items: list[dict], model, tok, n_layers, probe_layers, device
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


def select_persample(
    profile: dict[int, float], nt_candidates: tuple, min_start: int, max_start: int
) -> tuple[int, int]:
    best_start, best_nt, best_score = min_start, nt_candidates[0], -999.0
    for n_t in nt_candidates:
        for start in range(min_start, max_start + 1):
            vals = [profile[l] for l in profile if start <= l < start + n_t]
            if not vals:
                continue
            score = float(np.mean(vals))
            if score > best_score:
                best_score, best_start, best_nt = score, start, n_t
    return best_start, best_start + best_nt


def run_empirical_logit_mc(
    items: list[dict],
    model,
    tok,
    n_layers: int,
    min_start: int,
    max_start: int,
    nt_candidates: tuple,
    min_decoder: int,
    device: torch.device,
) -> tuple[int, int] | None:
    max_t_stop = n_layers - min_decoder
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
    calib = items[:N_CALIB]
    for item in calib:
        plen = tok(item["prompt"], return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
        label = item["label"]
        cont = item["choices"][label]
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
    n = max(len(calib), 1)
    mean_gain = {c: g / n for c, g in gain_sum.items()}
    best_win, best_gain = max(mean_gain.items(), key=lambda x: x[1])
    if best_gain <= 0:
        return None
    return best_win


def run_empirical_logit_gsm_proxy(
    gsm_calib_docs: list,
    gsm_task,
    tok,
    model,
    n_layers: int,
    min_start: int,
    max_start: int,
    nt_candidates: tuple,
    min_decoder: int,
    device: torch.device,
) -> tuple[int, int] | None:
    """用 GSM 标定集构造「长 prompt + 金标答案前缀」的伪 MC item，复用 logit-gain 搜窗。"""
    items: list[dict] = []
    for doc in gsm_calib_docs:
        ctx = gsm_task.fewshot_context(doc, _num_fewshot(gsm_task))
        tgt = str(gsm_task.doc_to_target(doc))
        cont = tgt[: min(800, len(tgt))]
        ch = [cont.strip()]
        items.append({"prompt": ctx, "choices": ch, "label": 0})
    return run_empirical_logit_mc(
        items, model, tok, n_layers, min_start, max_start, nt_candidates, min_decoder, device
    )


def mc_predict(
    model,
    tok,
    item: dict,
    device: torch.device,
    n_e: int | None,
    n_t: int | None,
) -> int:
    prompt, choices = item["prompt"], item["choices"]
    plen = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
    scores = []
    for cont in choices:
        full = prompt + " " + cont
        enc = tok(full, return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        am = enc.get("attention_mask")
        am = am.to(device) if am is not None else None
        if n_e is not None and n_t is not None and n_t > 0:
            lg = etd_forward_logits(
                model,
                ids,
                am,
                n_e=n_e,
                n_t=n_t,
                k=K_ETD,
                alpha=min(1.0, 6.0 / max(n_t, 1)),
            )
        else:
            lg = baseline_forward_logits(model, ids, am)
        scores.append(loglikelihood_mc(lg, ids, plen))
    return int(np.argmax(scores))


def load_model(path: str):
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


def bbh_docs_to_mc_items(task, docs: list) -> list[dict]:
    items = []
    for doc in docs:
        ctx = task.fewshot_context(doc, _num_fewshot(task))
        choices_raw = task.doc_to_choice(doc)
        tgt = str(task.doc_to_target(doc)).strip()
        choices = [str(c).strip() for c in choices_raw]
        if tgt not in choices:
            # 宽松匹配
            hit = next((i for i, c in enumerate(choices) if c.lower() == tgt.lower()), None)
            if hit is None:
                continue
            label = hit
        else:
            label = choices.index(tgt)
        items.append({"prompt": ctx, "choices": choices, "label": label})
    return items


def extract_gsm_number(text: str) -> str | None:
    m = GSM_STRICT_NUM_RE.search(text)
    return m.group(1).replace(",", "") if m else None


def greedy_generate(
    model,
    tok,
    prompt: str,
    device: torch.device,
    n_e: int | None,
    n_t: int | None,
    max_new_tokens: int,
) -> str:
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


def estimate_total_weight(
    n_bbh_tasks: int, bbh_limit: int, gsm_limit: int, gsm_avg_tokens: int = 120
) -> float:
    """粗权重：用于 ETA 比例（非秒）。"""
    w_bbh_unit = 4 * 2.5
    w_bbh = n_bbh_tasks * (10 + bbh_limit * w_bbh_unit)
    w_gsm = 10 + gsm_limit * 4 * gsm_avg_tokens * 0.08
    return w_bbh + w_gsm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS.keys()), required=True)
    ap.add_argument("--bbh-limit", type=int, default=50)
    ap.add_argument("--gsm-limit", type=int, default=50)
    ap.add_argument("--bbh-tasks", nargs="*", default=DEFAULT_BBH_TASKS)
    ap.add_argument("--output-json", type=str, default="")
    args = ap.parse_args()

    preset = PRESETS[args.preset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(
        args.output_json
        or (EXP / "results" / f"r40_bbh_gsm8k_{args.preset.replace('-', '_')}.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_layers = preset["n_layers"]
    probe_layers = preset["probe_layers"]
    min_start = preset["min_start"]
    max_start = preset["max_start"]
    nt_cands = preset["nt_candidates"]
    calib_nt = preset["calib_nt"]
    min_decoder = preset["min_decoder"]

    total_w = estimate_total_weight(len(args.bbh_tasks), args.bbh_limit, args.gsm_limit)
    eta = EtaState(total_w * 1.2)

    print(
        f"[R40] preset={args.preset} BBH={len(args.bbh_tasks)}×{args.bbh_limit} "
        f"GSM8K limit={args.gsm_limit} methods={METHODS}"
    )
    print(
        f"[R40] ETA：内部权重≈{total_w * 1.2:.0f}（非秒）；"
        f"tqdm postfix 中 ETA_run=剩余本阶段、ETA_total≈全程剩余墙钟"
    )
    print(f"[R40] min_decoder={min_decoder} calib_nt={calib_nt} nt_cands={nt_cands}")

    tm = TaskManager(include_defaults=True)
    tok, model = load_model(preset["model_path"])

    results: dict = {
        "preset": args.preset,
        "bbh": {},
        "gsm8k": {},
        "meta": {"min_decoder": min_decoder, "n_calib": N_CALIB},
    }
    if args.preset == "gemma2-2b":
        results["meta"]["gemma2_note"] = (
            "min_decoder 已放宽；neg_cos_am 仍为 pre-norm hook，若与 Qwen/Llama 差异大属预期"
        )

    # ─── BBH ───────────────────────────────────────────────────────────────
    for task_name in args.bbh_tasks:
        t0_sub = time.time()
        loaded = tm.load([task_name])
        task = loaded["tasks"][task_name]
        task.set_fewshot_seed(42)
        test_docs = list(task.test_docs())[: args.bbh_limit]
        items = bbh_docs_to_mc_items(task, test_docs)
        if not items:
            print(f"[WARN] {task_name}: no MC items, skip")
            continue

        mean_profile = calibrate_mc_items(items, model, tok, n_layers, probe_layers, device)
        win_calib = select_calib_global(mean_profile, calib_nt, min_start, max_start)
        win_emp = run_empirical_logit_mc(
            items, model, tok, n_layers, min_start, max_start, nt_cands, min_decoder, device
        )

        correct = {m: 0 for m in METHODS}
        n_eval = len(items)
        pbar = tqdm(
            range(n_eval),
            desc=f"{args.preset}|BBH|{task_name.split('_')[-1]}|baseline+3ETD",
            dynamic_ncols=True,
        )
        for i in pbar:
            item = items[i]
            enc = tok(item["prompt"], return_tensors="pt", add_special_tokens=False)
            ids = enc["input_ids"].to(device)
            am = enc.get("attention_mask")
            am = am.to(device) if am is not None else None
            ps_sig = probe_neg_cos_am(model, ids, am, n_layers, probe_layers)
            ps_win = select_persample(ps_sig, nt_cands, min_start, max_start)
            wins = {
                "baseline": None,
                "neg_cos_am_calib": win_calib,
                "emp_logit_fixed": win_emp,
                "neg_cos_am_ps_nt": ps_win,
            }
            label = item["label"]
            for mname in METHODS:
                w = wins[mname]
                if w is None:
                    pred = mc_predict(model, tok, item, device, None, None)
                else:
                    ts, te = w
                    nt = te - ts
                    if n_layers - te < 2 or nt < 1:
                        pred = mc_predict(model, tok, item, device, None, None)
                    else:
                        pred = mc_predict(model, tok, item, device, ts, nt)
                if pred == label:
                    correct[mname] += 1
            torch.cuda.empty_cache()

            elapsed, eta_rem, eta_tot = eta.add(4.0)
            pbar.set_postfix_str(
                f"sub_elapsed={time.time()-t0_sub:.0f}s | "
                f"ETA_run={eta_rem:.0f}s | ETA_total~={eta_tot:.0f}s"
            )

        accs = {m: correct[m] / max(n_eval, 1) for m in METHODS}
        results["bbh"][task_name] = {
            "n": n_eval,
            "accuracies": accs,
            "neg_cos_am_calib_win": list(win_calib),
            "emp_logit_win": list(win_emp) if win_emp else None,
            "elapsed_s": time.time() - t0_sub,
        }
        bl = accs["baseline"]
        print(
            f"  [BBH done] {task_name} acc: baseline={accs['baseline']:.3f} "
            f"calib={accs['neg_cos_am_calib']:.3f} emp={accs['emp_logit_fixed']:.3f} "
            f"ps={accs['neg_cos_am_ps_nt']:.3f}  Δcalib={accs['neg_cos_am_calib']-bl:+.3f}"
        )
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # ─── GSM8K ──────────────────────────────────────────────────────────────
    t0_gsm = time.time()
    loaded_g = tm.load(["gsm8k"])
    gsm_task = loaded_g["tasks"]["gsm8k"]
    gsm_task.set_fewshot_seed(42)
    gsm_docs_all = list(gsm_task.test_docs())[: args.gsm_limit]
    gsm_calib = gsm_docs_all[: min(N_CALIB, len(gsm_docs_all))]

    win_calib_g = (min_start, min_start + calib_nt)
    win_emp_g = None
    if gsm_calib:
        prof_g = defaultdict(list)
        for doc in gsm_calib:
            ctx = gsm_task.fewshot_context(doc, _num_fewshot(gsm_task))
            enc = tok(ctx, return_tensors="pt", add_special_tokens=False)
            ids = enc["input_ids"].to(device)
            am = enc.get("attention_mask")
            am = am.to(device) if am is not None else None
            sig = probe_neg_cos_am(model, ids, am, n_layers, probe_layers)
            for li, v in sig.items():
                prof_g[li].append(v)
        mean_g = {li: float(np.mean(vs)) for li, vs in prof_g.items() if vs}
        if mean_g:
            win_calib_g = select_calib_global(mean_g, calib_nt, min_start, max_start)
        win_emp_g = run_empirical_logit_gsm_proxy(
            gsm_calib, gsm_task, tok, model, n_layers, min_start, max_start, nt_cands, min_decoder, device
        )

    gsm_correct = {m: 0 for m in METHODS}
    n_g = len(gsm_docs_all)
    pbar_g = tqdm(
        range(n_g),
        desc=f"{args.preset}|GSM8K|greedy|baseline+3ETD",
        dynamic_ncols=True,
    )
    for i in pbar_g:
        doc = gsm_docs_all[i]
        ctx = gsm_task.fewshot_context(doc, _num_fewshot(gsm_task))
        enc = tok(ctx, return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        am = enc.get("attention_mask")
        am = am.to(device) if am is not None else None
        ps_sig = probe_neg_cos_am(model, ids, am, n_layers, probe_layers)
        ps_win = select_persample(ps_sig, nt_cands, min_start, max_start)
        wins_g = {
            "baseline": None,
            "neg_cos_am_calib": win_calib_g,
            "emp_logit_fixed": win_emp_g,
            "neg_cos_am_ps_nt": ps_win,
        }
        for mname in METHODS:
            w = wins_g[mname]
            ts, te = (None, None) if w is None else (w[0], w[1])
            nt = 0 if w is None else te - ts
            if w is not None and (n_layers - te < 2 or nt < 1):
                ts, te, nt = None, None, 0
            gen = greedy_generate(
                model,
                tok,
                ctx,
                device,
                ts,
                nt if nt > 0 else None,
                GSM_MAX_NEW_TOKENS,
            )
            if gsm_exact_match(gen, doc, gsm_task):
                gsm_correct[mname] += 1
        torch.cuda.empty_cache()
        elapsed, eta_rem, eta_tot = eta.add(4.0 * GSM_MAX_NEW_TOKENS / 120.0)
        pbar_g.set_postfix_str(
            f"gsm_elapsed={time.time()-t0_gsm:.0f}s | ETA_run~={eta_rem:.0f}s | ETA_total~={eta_tot:.0f}s"
        )

    g_acc = {m: gsm_correct[m] / max(n_g, 1) for m in METHODS}
    results["gsm8k"] = {
        "n": n_g,
        "exact_match": g_acc,
        "neg_cos_am_calib_win": list(win_calib_g),
        "emp_logit_win": list(win_emp_g) if win_emp_g else None,
        "max_new_tokens": GSM_MAX_NEW_TOKENS,
        "elapsed_s": time.time() - t0_gsm,
    }
    blg = g_acc["baseline"]
    print(
        f"  [GSM8K done] EM: baseline={g_acc['baseline']:.3f} calib={g_acc['neg_cos_am_calib']:.3f} "
        f"emp={g_acc['emp_logit_fixed']:.3f} ps={g_acc['neg_cos_am_ps_nt']:.3f}  Δcalib={g_acc['neg_cos_am_calib']-blg:+.3f}"
    )

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[R40] wrote {out_path}")


if __name__ == "__main__":
    main()
