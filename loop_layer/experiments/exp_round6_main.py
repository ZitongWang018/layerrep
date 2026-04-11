#!/usr/bin/env python3
"""
exp_round6_main.py  ─  Round-6 ETD Experiments  (2026-04-07)

Planned tasks + new hypotheses derived from Round-5 findings:

  T6 : Zero-shot step_size layer selection rule validation
       "Can we configure ETD with zero labeled samples, using only the
        model's own layer statistics?"

  T8 : delta_norm vs correction correlation  (ROC / AUC analysis)
       "Is a sample's delta_norm predictive of whether ETD corrects it?"

  H1 : Task Difficulty Mediates ETD Gain
       Hypothesis: Among initially-wrong examples, HIGH baseline confidence
       (large |margin|) → higher ETD correction rate (more 'softening room')
       Source: M1 showed Margin decreases monotonically; ETD = over-confidence reducer

  H2 : Late T-block Layer Dominance
       Hypothesis: Repeating only the LATE half of T-block (layers 15-21)
       matches or beats repeating the early half (layers 8-14)
       Source: M2 heatmap showed layers 18-21 stay active much longer

  H3 : Per-sample cos_sim as Adaptive k-selection Signal
       Hypothesis: Samples with HIGH iteration cos_sim benefit MORE from k=3
       Source: A3+ showed cos_sim ↑ with n_t; BoolQ cos_sim > ARC cos_sim
               and k=3 helps BoolQ but not ARC

Sample sizes: N_EVAL=300, N_T8=300, N_MECH=120
All plots: English labels/titles/legends
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─── Paths ───────────────────────────────────────────────────────────────────
MODEL_PATH  = "/root/autodl-tmp/model_qwen"
ETD_PATH    = Path("/root/autodl-tmp/loop_layer/ETD")
SWEEP_PATH  = Path("/root/autodl-tmp/loop_layer/ETD_layer_sweep")
RESULTS_DIR = Path("/root/autodl-tmp/loop_layer/experiments/results")
FIGURES_DIR = Path("/root/autodl-tmp/loop_layer/experiments/figures")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

for _p in [str(ETD_PATH), str(SWEEP_PATH)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from etd_forward import (  # noqa: E402
    etd_forward_logits,
    baseline_forward_logits,
    loglikelihood_continuation,
    _adaptive_alpha,
    _prepare_position_ids,
)
from data_cache import load_boolq_examples, load_arc_examples  # noqa: E402

# ─── Parameters ──────────────────────────────────────────────────────────────
N_EVAL  = 300   # main accuracy evaluations
N_T8    = 300   # T8 correlation analysis
N_MECH  = 120   # mechanistic per-sample experiments (H1/H3)
N_T6_ZS = 10    # T6 zero-shot: unlabeled samples for step_size computation

# H2 partial T-block configs: (name, n_e, n_t, description)
H2_CONFIGS = [
    ("full_8_22",    8,  14, "Full T-block  (layers 8-21)"),
    ("early_8_15",   8,   7, "Early half    (layers 8-14)"),
    ("mid_11_18",   11,   7, "Middle        (layers 11-17)"),
    ("late_15_22",  15,   7, "Late half     (layers 15-21)"),
    ("very_late_18",18,   4, "Very late     (layers 18-21)"),
]

# ─── Dataset Loaders ─────────────────────────────────────────────────────────
def load_commonsenseqa(limit: int):
    from datasets import load_dataset
    ds = load_dataset("tau/commonsense_qa")["validation"]
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    out = []
    for ex in ds:
        if len(out) >= limit:
            break
        q = ex["question"].strip()
        prefix = f"Question: {q}\nAnswer:"
        conts = [f" {t}" for t in ex["choices"]["text"]]
        key = ex["answerKey"]
        if key not in label_map:
            continue
        out.append((prefix, conts, label_map[key]))
    return out


def load_truthfulqa(limit: int):
    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice")["validation"]
    out = []
    for ex in ds:
        if len(out) >= limit:
            break
        mc1 = ex["mc1_targets"]
        choices, labels = mc1["choices"], mc1["labels"]
        if 1 not in labels:
            continue
        label = labels.index(1)
        prefix = f"Question: {ex['question'].strip()}\nAnswer:"
        conts = [f" {c}" for c in choices]
        out.append((prefix, conts, label))
    return out

# ─── ETD Utilities ───────────────────────────────────────────────────────────

@torch.inference_mode()
def eval_mc(model, tokenizer, examples, n_e, n_t, k, alpha, device, desc="") -> float:
    correct = 0
    for prefix, conts, label in tqdm(examples, desc=desc, leave=False):
        plen = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
        scores = []
        for cont in conts:
            enc = tokenizer(prefix + cont, return_tensors="pt", add_special_tokens=False)
            ids = enc["input_ids"].to(device)
            attn = enc.get("attention_mask")
            if attn is not None:
                attn = attn.to(device)
            logits = etd_forward_logits(model, ids, attn, n_e, n_t, k, alpha=alpha)
            scores.append(loglikelihood_continuation(logits, ids, plen))
        if max(range(len(scores)), key=lambda i: scores[i]) == label:
            correct += 1
    return correct / len(examples)


@torch.inference_mode()
def eval_mc_baseline(model, tokenizer, examples, device, desc="") -> float:
    correct = 0
    for prefix, conts, label in tqdm(examples, desc=desc, leave=False):
        plen = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
        scores = []
        for cont in conts:
            enc = tokenizer(prefix + cont, return_tensors="pt", add_special_tokens=False)
            ids = enc["input_ids"].to(device)
            attn = enc.get("attention_mask")
            if attn is not None:
                attn = attn.to(device)
            logits = baseline_forward_logits(model, ids, attn)
            scores.append(loglikelihood_continuation(logits, ids, plen))
        if max(range(len(scores)), key=lambda i: scores[i]) == label:
            correct += 1
    return correct / len(examples)


@torch.inference_mode()
def run_T_block_track(model, input_ids, attention_mask, n_e, n_t, k_max, alpha):
    """Run E-block then T-block k_max times. Returns list of hidden states
    [h_e, h_after_k1, h_after_k2, ...] for the last token only."""
    from transformers.masking_utils import create_causal_mask
    base = model.model
    cfg  = model.config
    device = input_ids.device

    inputs_embeds = base.embed_tokens(input_ids)
    batch, seq_len = inputs_embeds.shape[:2]
    position_ids = _prepare_position_ids(attention_mask, 0, batch, seq_len, device)

    past_key_values = None
    mask_kwargs = dict(
        config=cfg, inputs_embeds=inputs_embeds,
        attention_mask=attention_mask, past_key_values=past_key_values,
        position_ids=position_ids,
    )
    causal_mask_map = {"full_attention": create_causal_mask(**mask_kwargs)}
    if getattr(base, "has_sliding_layers", False):
        from transformers.masking_utils import create_sliding_window_causal_mask
        causal_mask_map["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    position_embeddings = base.rotary_emb(inputs_embeds, position_ids)
    hidden_states = inputs_embeds

    def _run_layer(idx, hs):
        ltype = cfg.layer_types[idx]
        out = base.layers[idx](
            hs,
            attention_mask=causal_mask_map[ltype],
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=False,
            position_embeddings=position_embeddings,
        )
        return out[0] if isinstance(out, tuple) else out

    for i in range(n_e):
        hidden_states = _run_layer(i, hidden_states)

    # h_e last token
    iter_last_tokens = [hidden_states[0, -1].clone()]

    for _ in range(k_max):
        h_prev = hidden_states
        for i in range(n_e, n_e + n_t):
            hidden_states = _run_layer(i, hidden_states)
        if alpha < 1.0:
            hidden_states = alpha * hidden_states + (1.0 - alpha) * h_prev
        iter_last_tokens.append(hidden_states[0, -1].clone())

    return iter_last_tokens


@torch.inference_mode()
def compute_layer_step_sizes(model, tokenizer, examples, device):
    """Full forward pass with output_hidden_states; return mean step_size per layer."""
    n_layers = model.config.num_hidden_layers
    all_steps = [[] for _ in range(n_layers)]
    for prefix, conts, _ in tqdm(examples, desc="  step_size profile", leave=False):
        full = prefix + conts[0]
        enc = tokenizer(full, return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(device)
        out = model(input_ids=ids, attention_mask=attn,
                    output_hidden_states=True, use_cache=False)
        hs = out.hidden_states
        for l in range(n_layers):
            h_prev = hs[l][0]
            h_curr = hs[l + 1][0]
            step = (h_curr - h_prev).norm(dim=-1).mean().item() / \
                   (h_prev.norm(dim=-1).mean().item() + 1e-8)
            all_steps[l].append(step)
    return [float(np.mean(v)) for v in all_steps]


@torch.inference_mode()
def per_sample_mc_score(model, tokenizer, prefix, conts, device, n_e, n_t, k, alpha):
    """Return (scores, pred_idx) for each continuation under ETD."""
    plen = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
    scores = []
    for cont in conts:
        enc = tokenizer(prefix + cont, return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(device)
        logits = etd_forward_logits(model, ids, attn, n_e, n_t, k, alpha=alpha)
        scores.append(loglikelihood_continuation(logits, ids, plen))
    return scores, max(range(len(scores)), key=lambda i: scores[i])


@torch.inference_mode()
def per_sample_baseline_score(model, tokenizer, prefix, conts, device):
    plen = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
    scores = []
    for cont in conts:
        enc = tokenizer(prefix + cont, return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(device)
        logits = baseline_forward_logits(model, ids, attn)
        scores.append(loglikelihood_continuation(logits, ids, plen))
    return scores, max(range(len(scores)), key=lambda i: scores[i])


def compute_margin(scores, label):
    """log-prob margin: score[label] - max(score[others])"""
    other_max = max(scores[i] for i in range(len(scores)) if i != label) if len(scores) > 1 else scores[0]
    return scores[label] - other_max


def roc_auc_manual(labels, scores):
    """Compute ROC AUC without sklearn."""
    pairs = sorted(zip(scores, labels), reverse=True)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp, fp = 0, 0
    auc = 0.0
    prev_fp = 0
    for _, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
            auc += tp  # rectangular approximation
    return auc / (n_pos * n_neg)


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    t_global = time.time()
    print("=" * 70)
    print("Round-6 ETD Experiments: T6 / T8 / H1 / H2 / H3")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"N_EVAL={N_EVAL}  N_T8={N_T8}  N_MECH={N_MECH}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading model from {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"  Done. device={device}, layers={n_layers}\n")

    results: dict = {
        "meta": {},
        "baselines": {},
        "t6_zero_shot": {},
        "t8_delta_norm_correlation": {},
        "h1_difficulty_mediation": {},
        "h2_late_tblock": {},
        "h3_cosim_k_selection": {},
    }

    # ── Load datasets ─────────────────────────────────────────────────────────
    print("Loading datasets ...")
    boolq_300   = load_boolq_examples("validation", N_EVAL)
    arc_300     = load_arc_examples("test", N_EVAL)
    print(f"  BoolQ: {len(boolq_300)}, ARC: {len(arc_300)}")
    try:
        csqa_300 = load_commonsenseqa(N_EVAL)
        print(f"  CommonsenseQA: {len(csqa_300)}")
    except Exception as e:
        print(f"  CommonsenseQA failed: {e}")
        csqa_300 = []
    try:
        tqa_300 = load_truthfulqa(N_EVAL)
        print(f"  TruthfulQA: {len(tqa_300)}")
    except Exception as e:
        print(f"  TruthfulQA failed: {e}")
        tqa_300 = []

    # ── Baselines ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("BASELINES (300 samples)")
    print("=" * 50)
    baselines = {}
    bench_map = {
        "boolq": boolq_300, "arc": arc_300,
        "commonsenseqa": csqa_300, "truthfulqa": tqa_300,
    }
    for bname, ex in bench_map.items():
        if not ex:
            continue
        acc = eval_mc_baseline(model, tokenizer, ex, device, desc=f"baseline/{bname}")
        baselines[bname] = acc
        print(f"  {bname:<20}: {acc:.4f}")
    results["baselines"] = baselines
    print()

    # ══════════════════════════════════════════════════════════════════════════
    # T6: ZERO-SHOT STEP_SIZE LAYER SELECTION
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 50)
    print(f"T6: Zero-Shot Step_size Layer Selection Rule Validation")
    print(f"  Phase 1: Compute step_size from {N_T6_ZS} UNLABELED samples")
    print(f"  Phase 2: Validate auto-selected config on CommonsenseQA + TruthfulQA (N={N_EVAL})")
    print("=" * 50)

    # Phase 1: zero-shot profile (use CommonsenseQA examples but NO labels)
    zs_examples_unlabeled = [(p, c, -1) for p, c, _ in (csqa_300[:N_T6_ZS] if csqa_300 else boolq_300[:N_T6_ZS])]
    print(f"  Computing step_size profile on {N_T6_ZS} unlabeled samples ...")
    zs_step_sizes = compute_layer_step_sizes(model, tokenizer, zs_examples_unlabeled, device)

    # Rule: safe layers have step_size < threshold
    THRESHOLD = 1.0
    safe_layers = [l for l, s in enumerate(zs_step_sizes) if s < THRESHOLD]
    # Pick t_start as the first safe layer ≥ 8 (hard constraint)
    valid_t_starts = [l for l in safe_layers if l >= 8]
    if valid_t_starts:
        zs_t_start = valid_t_starts[0]
    else:
        zs_t_start = 8  # fallback
    # Choose n_t = 14 (champion block size, energy formula: alpha = 6/14)
    zs_n_e = zs_t_start
    zs_n_t = 14
    zs_alpha = _adaptive_alpha(zs_n_t)
    zs_eff = zs_alpha * zs_n_t
    print(f"  step_size profile (first 24 layers): {[f'{s:.3f}' for s in zs_step_sizes[:24]]}")
    print(f"  Safe layers (step_size < {THRESHOLD}): {safe_layers[:15]}...")
    print(f"  Auto-selected: n_e={zs_n_e}, n_t={zs_n_t}, alpha={zs_alpha:.3f}, eff_step={zs_eff:.2f}")

    # Also try smaller threshold (0.5) to see sensitivity
    safe_layers_strict = [l for l in safe_layers if zs_step_sizes[l] < 0.5 and l >= 8]
    zs_t_start_strict = safe_layers_strict[0] if safe_layers_strict else 8
    zs_n_e_strict = zs_t_start_strict

    # Phase 2: evaluate on CommonsenseQA and TruthfulQA
    t6_results = {
        "step_sizes": zs_step_sizes,
        "safe_layers": safe_layers,
        "threshold": THRESHOLD,
        "zs_t_start": zs_t_start,
        "zs_n_e": zs_n_e,
        "zs_n_t": zs_n_t,
        "zs_alpha": zs_alpha,
        "eff_step": zs_eff,
    }

    print(f"\n  Evaluating zero-shot selected config (n_e={zs_n_e}, n_t={zs_n_t}, k=2) ...")
    t6_eval_benchmarks = {
        "commonsenseqa": csqa_300,
        "truthfulqa": tqa_300,
    }
    for bname, ex in t6_eval_benchmarks.items():
        if not ex:
            continue
        bl = baselines.get(bname, 0)
        # Zero-shot selected
        acc_zs = eval_mc(model, tokenizer, ex, zs_n_e, zs_n_t, 2, zs_alpha,
                         device, desc=f"T6-zs/{bname}")
        # Strict threshold
        acc_strict = eval_mc(model, tokenizer, ex, zs_n_e_strict, zs_n_t, 2, zs_alpha,
                             device, desc=f"T6-strict/{bname}")
        # Oracle champion (n_e=8, n_t=14)
        acc_oracle = eval_mc(model, tokenizer, ex, 8, 14, 2, _adaptive_alpha(14),
                             device, desc=f"T6-oracle/{bname}")
        t6_results[bname] = {
            "baseline": bl,
            "zs_selected": {"n_e": zs_n_e, "acc": acc_zs, "delta": acc_zs - bl},
            "strict_selected": {"n_e": zs_n_e_strict, "acc": acc_strict, "delta": acc_strict - bl},
            "oracle_champion": {"n_e": 8, "acc": acc_oracle, "delta": acc_oracle - bl},
        }
        print(f"  {bname}: baseline={bl:.4f} | zs={acc_zs:.4f}({acc_zs-bl:+.4f}) "
              f"| strict={acc_strict:.4f}({acc_strict-bl:+.4f}) "
              f"| oracle={acc_oracle:.4f}({acc_oracle-bl:+.4f})")

    results["t6_zero_shot"] = t6_results

    # T6 Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"T6: Zero-Shot Layer Selection Rule Validation\n"
        f"Step_size threshold={THRESHOLD:.1f}, selected n_e={zs_n_e}",
        fontsize=12, fontweight="bold"
    )

    ax = axes[0]
    layer_idx = list(range(n_layers))
    ax.plot(layer_idx, zs_step_sizes, "o-", color="#2196F3", lw=1.5, markersize=4,
            label=f"step_size (N={N_T6_ZS} unlabeled)")
    ax.axhline(THRESHOLD, color="#FF5722", ls="--", lw=1.5, label=f"Threshold={THRESHOLD}")
    ax.axhline(0.5, color="#FF9800", ls="--", lw=1.5, alpha=0.7, label="Strict threshold=0.5")
    ax.axvline(zs_t_start, color="#4CAF50", ls="-", lw=2, alpha=0.8,
               label=f"Auto t_start={zs_t_start}")
    ax.axvspan(0, 7.5, color="#FF5722", alpha=0.1, label="Forbidden zone (<8)")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("step_size ||Δh||/||h||")
    ax.set_title("Layer step_size Profile\n(zero-shot signal, no labels used)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, n_layers, 4))
    ax.set_yscale("log")

    ax = axes[1]
    # Show which layers are "safe"
    safe_mask = [1 if s < THRESHOLD and l >= 8 else 0 for l, s in enumerate(zs_step_sizes)]
    ax.bar(layer_idx, safe_mask, color=["#4CAF50" if s else "#FF5722" for s in safe_mask],
           alpha=0.75, width=0.8)
    ax.axvline(zs_t_start, color="#2196F3", ls="-", lw=2.5, label=f"Selected t_start={zs_t_start}")
    ax.axvline(8, color="#9C27B0", ls=":", lw=1.5, alpha=0.7, label="Hard lower bound (8)")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Safe (1=yes, 0=no)")
    ax.set_title("Safe Layer Candidates\n(green = step_size<1.0 and layer>=8)")
    ax.legend(fontsize=8)
    ax.set_xticks(range(0, n_layers, 4))

    ax = axes[2]
    eval_bnames = [b for b in ["commonsenseqa", "truthfulqa"] if b in t6_results and isinstance(t6_results[b], dict)]
    x = np.arange(len(eval_bnames))
    w = 0.22
    bl_vals = [t6_results[b]["baseline"] for b in eval_bnames]
    zs_vals  = [t6_results[b]["zs_selected"]["acc"] for b in eval_bnames]
    st_vals  = [t6_results[b]["strict_selected"]["acc"] for b in eval_bnames]
    or_vals  = [t6_results[b]["oracle_champion"]["acc"] for b in eval_bnames]
    ax.bar(x - w*1.5, bl_vals, w, label="Baseline", color="#9E9E9E", alpha=0.8)
    ax.bar(x - w*0.5, zs_vals,  w, label=f"Zero-shot (n_e={zs_n_e})", color="#2196F3", alpha=0.85)
    ax.bar(x + w*0.5, st_vals,  w, label=f"Strict-ZS (n_e={zs_n_e_strict})", color="#FF9800", alpha=0.85)
    ax.bar(x + w*1.5, or_vals,  w, label="Oracle champion (n_e=8)", color="#4CAF50", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([b[:10] for b in eval_bnames], fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_title("T6 Result: Zero-shot vs Oracle vs Baseline\n(validated on unseen benchmarks)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "r6_t6_zeroshot_selection.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: r6_t6_zeroshot_selection.png\n")

    # ══════════════════════════════════════════════════════════════════════════
    # T8: DELTA_NORM VS CORRECTION CORRELATION (ROC / AUC)
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 50)
    print(f"T8: delta_norm vs Correction Correlation  (N={N_T8} BoolQ samples)")
    print("  Q: Is per-sample delta_norm predictive of whether ETD corrects the error?")
    print("=" * 50)

    n_e_t8, n_t_t8 = 8, 14
    alpha_t8 = _adaptive_alpha(n_t_t8)
    boolq_t8 = load_boolq_examples("validation", N_T8)

    t8_records = []
    for prefix, conts, label in tqdm(boolq_t8, desc="T8: per-sample tracking", leave=True):
        # Baseline scores
        base_scores, base_pred = per_sample_baseline_score(model, tokenizer, prefix, conts, device)
        base_margin = compute_margin(base_scores, label)
        base_correct = (base_pred == label)

        # ETD k=2 scores
        k2_scores, k2_pred = per_sample_mc_score(model, tokenizer, prefix, conts, device,
                                                  n_e_t8, n_t_t8, 2, alpha_t8)
        k2_margin = compute_margin(k2_scores, label)
        k2_correct = (k2_pred == label)

        # delta_norm: compute from T-block tracking
        full = prefix + conts[0]
        enc = tokenizer(full, return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(device)
        iter_last = run_T_block_track(model, ids, attn, n_e_t8, n_t_t8, k_max=2, alpha=alpha_t8)
        h_e_l = iter_last[0]
        h_1_l = iter_last[1]
        h_2_l = iter_last[2]
        dn_k1 = ((h_1_l - h_e_l).norm() / (h_e_l.norm() + 1e-8)).item()
        dn_k2 = ((h_2_l - h_1_l).norm() / (h_1_l.norm() + 1e-8)).item()
        # cos_sim between iteration directions
        dir_k1 = h_1_l - h_e_l
        dir_k2 = h_2_l - h_1_l
        cos_sim = F.cosine_similarity(dir_k2.unsqueeze(0), dir_k1.unsqueeze(0)).item()

        t8_records.append({
            "label": label,
            "base_correct": base_correct,
            "base_margin": base_margin,
            "k2_correct": k2_correct,
            "k2_margin": k2_margin,
            "dn_k1": dn_k1,
            "dn_k2": dn_k2,
            "cos_sim": cos_sim,
            "corrected": (not base_correct) and k2_correct,
            "reversed":  base_correct and (not k2_correct),
            "margin_change": k2_margin - base_margin,
        })

    # --- ROC/AUC Analysis ---
    # For initially-wrong examples, can delta_norm predict correction?
    wrong_records = [r for r in t8_records if not r["base_correct"]]
    correct_records = [r for r in t8_records if r["base_correct"]]

    corrected_labels = [int(r["corrected"]) for r in wrong_records]
    dn_scores_wrong  = [r["dn_k1"] for r in wrong_records]
    cos_scores_wrong = [r["cos_sim"] for r in wrong_records]
    margin_scores_wrong = [abs(r["base_margin"]) for r in wrong_records]

    auc_dn_correction   = roc_auc_manual(corrected_labels, dn_scores_wrong)
    auc_cos_correction  = roc_auc_manual(corrected_labels, cos_scores_wrong)
    # Does higher |margin| at baseline predict correction? (H1 preview)
    auc_margin_correction = roc_auc_manual(corrected_labels, margin_scores_wrong)

    reversed_labels = [int(r["reversed"]) for r in correct_records]
    dn_scores_correct = [r["dn_k1"] for r in correct_records]
    auc_dn_reversal = roc_auc_manual(reversed_labels, dn_scores_correct)

    # Overall stats
    overall_acc_base = sum(r["base_correct"] for r in t8_records) / len(t8_records)
    overall_acc_k2   = sum(r["k2_correct"]  for r in t8_records) / len(t8_records)
    n_corrected = sum(r["corrected"] for r in t8_records)
    n_reversed  = sum(r["reversed"]  for r in t8_records)

    print(f"\n  N={len(t8_records)}, baseline acc={overall_acc_base:.4f}, k=2 acc={overall_acc_k2:.4f}")
    print(f"  Initially wrong: {len(wrong_records)}, corrected: {n_corrected}, reversed: {n_reversed}")
    print(f"\n  ROC/AUC results (predicting whether initially-wrong→corrected):")
    print(f"    AUC(delta_norm_k1 → corrected): {auc_dn_correction:.4f}")
    print(f"    AUC(cos_sim → corrected):        {auc_cos_correction:.4f}")
    print(f"    AUC(|base_margin| → corrected):  {auc_margin_correction:.4f}   [H1 preview]")
    print(f"    AUC(delta_norm_k1 → reversed):   {auc_dn_reversal:.4f}")

    # Bin analysis: split wrong examples by delta_norm quartile
    if len(wrong_records) >= 4:
        sorted_wrong = sorted(wrong_records, key=lambda r: r["dn_k1"])
        q_size = len(sorted_wrong) // 4
        quartile_correction = []
        quartile_dn_means   = []
        for qi in range(4):
            chunk = sorted_wrong[qi*q_size : (qi+1)*q_size]
            quartile_correction.append(sum(r["corrected"] for r in chunk) / len(chunk))
            quartile_dn_means.append(np.mean([r["dn_k1"] for r in chunk]))
        print(f"\n  Correction rate by delta_norm quartile (Q1=smallest, Q4=largest):")
        for qi, (dn_m, corr) in enumerate(zip(quartile_dn_means, quartile_correction)):
            print(f"    Q{qi+1} dn_mean={dn_m:.4f}: correction_rate={corr:.3f}")
    else:
        quartile_correction = []
        quartile_dn_means   = []

    results["t8_delta_norm_correlation"] = {
        "n_total": len(t8_records),
        "n_wrong": len(wrong_records),
        "n_correct": len(correct_records),
        "n_corrected": n_corrected,
        "n_reversed": n_reversed,
        "overall_acc_base": overall_acc_base,
        "overall_acc_k2": overall_acc_k2,
        "auc_dn_correction":  auc_dn_correction,
        "auc_cos_correction": auc_cos_correction,
        "auc_margin_correction": auc_margin_correction,
        "auc_dn_reversal": auc_dn_reversal,
        "quartile_dn_means":  quartile_dn_means,
        "quartile_correction": quartile_correction,
        "stats": {
            "wrong_dn_mean":      float(np.mean(dn_scores_wrong)) if dn_scores_wrong else 0,
            "corrected_dn_mean":  float(np.mean([r["dn_k1"] for r in wrong_records if r["corrected"]])) if n_corrected else 0,
            "uncorrected_dn_mean":float(np.mean([r["dn_k1"] for r in wrong_records if not r["corrected"]])) if (len(wrong_records)-n_corrected) else 0,
        }
    }

    # T8 Plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"T8: delta_norm vs Correction Correlation  (N={N_T8} BoolQ)\n"
        f"Config: n_e={n_e_t8}, n_t={n_t_t8}, alpha={alpha_t8:.3f}  |  "
        f"Baseline acc={overall_acc_base:.3f}, k=2 acc={overall_acc_k2:.3f}",
        fontsize=11, fontweight="bold"
    )

    ax = axes[0][0]
    if wrong_records:
        dn_corr = [r["dn_k1"] for r in wrong_records if r["corrected"]]
        dn_not  = [r["dn_k1"] for r in wrong_records if not r["corrected"]]
        ax.violinplot([dn_not, dn_corr], positions=[0, 1], showmedians=True)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Not corrected", "Corrected"])
        ax.set_ylabel("delta_norm (k=1 vs h_e)")
        ax.set_title(f"delta_norm Distribution\nby Correction Outcome\n"
                     f"AUC={auc_dn_correction:.3f}")
        ax.grid(True, alpha=0.3, axis="y")

    ax = axes[0][1]
    if wrong_records:
        cos_corr = [r["cos_sim"] for r in wrong_records if r["corrected"]]
        cos_not  = [r["cos_sim"] for r in wrong_records if not r["corrected"]]
        ax.violinplot([cos_not, cos_corr], positions=[0, 1], showmedians=True)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Not corrected", "Corrected"])
        ax.set_ylabel("cos_sim (iteration direction)")
        ax.set_title(f"cos_sim Distribution\nby Correction Outcome\n"
                     f"AUC={auc_cos_correction:.3f}")
        ax.grid(True, alpha=0.3, axis="y")

    ax = axes[0][2]
    all_dn = [r["dn_k1"] for r in t8_records]
    all_k2c = [int(r["k2_correct"]) for r in t8_records]
    ax.scatter(all_dn, all_k2c, alpha=0.2, s=20,
               c=["#4CAF50" if c else "#FF5722" for c in all_k2c])
    ax.set_xlabel("delta_norm (k=1 vs h_e)")
    ax.set_ylabel("k=2 Correct (1=yes)")
    ax.set_title("delta_norm vs k=2 Correctness\n(all examples)")
    ax.grid(True, alpha=0.3)

    ax = axes[1][0]
    if quartile_dn_means and quartile_correction:
        bars = ax.bar([f"Q{i+1}\ndn≈{d:.3f}" for i, d in enumerate(quartile_dn_means)],
                      quartile_correction,
                      color=["#2196F3", "#4CAF50", "#FF9800", "#FF5722"], alpha=0.85, width=0.55)
        ax.bar_label(bars, labels=[f"{v:.2%}" for v in quartile_correction], fontsize=9, padding=2)
        ax.set_ylabel("Correction rate")
        ax.set_title("Correction Rate by delta_norm Quartile\n(initially-wrong examples)")
        ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1][1]
    margin_corr = [abs(r["base_margin"]) for r in wrong_records if r["corrected"]]
    margin_not  = [abs(r["base_margin"]) for r in wrong_records if not r["corrected"]]
    if margin_corr and margin_not:
        ax.violinplot([margin_not, margin_corr], positions=[0, 1], showmedians=True)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Not corrected", "Corrected"])
        ax.set_ylabel("|baseline margin|")
        ax.set_title(f"|Baseline Margin| vs Correction\n"
                     f"AUC={auc_margin_correction:.3f}  (H1 preview)")
        ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1][2]
    # Scatter: margin_change vs base_margin for wrong examples
    base_margins = [r["base_margin"] for r in wrong_records]
    margin_changes = [r["margin_change"] for r in wrong_records]
    colors_wrong = ["#4CAF50" if r["corrected"] else "#FF9800" for r in wrong_records]
    ax.scatter(base_margins, margin_changes, c=colors_wrong, alpha=0.5, s=30)
    ax.axhline(0, color="black", lw=0.7)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel("Baseline margin (correct_score - best_wrong)")
    ax.set_ylabel("Margin change (k=2 - baseline)")
    ax.set_title("Margin Evolution for Wrong Examples\n(green=corrected, orange=stayed wrong)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "r6_t8_deltanorm_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: r6_t8_deltanorm_correlation.png\n")

    # ══════════════════════════════════════════════════════════════════════════
    # H1: TASK DIFFICULTY MEDIATES ETD GAIN
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 50)
    print(f"H1: Task Difficulty Mediates ETD Gain  (N={N_MECH} BoolQ + ARC)")
    print("  Hypothesis: HIGH baseline confidence → higher ETD correction rate")
    print("=" * 50)

    n_e_h1, n_t_h1 = 8, 14
    alpha_h1 = _adaptive_alpha(n_t_h1)

    h1_results = {}
    for task_name, examples in [("boolq", load_boolq_examples("validation", N_MECH)),
                                  ("arc", load_arc_examples("test", N_MECH))]:
        records = []
        for prefix, conts, label in tqdm(examples, desc=f"H1/{task_name}", leave=True):
            base_scores, base_pred = per_sample_baseline_score(model, tokenizer, prefix, conts, device)
            k2_scores, k2_pred = per_sample_mc_score(model, tokenizer, prefix, conts, device,
                                                      n_e_h1, n_t_h1, 2, alpha_h1)
            base_margin = compute_margin(base_scores, label)
            k2_margin   = compute_margin(k2_scores, label)
            base_correct = (base_pred == label)
            k2_correct   = (k2_pred == label)
            records.append({
                "base_correct": base_correct,
                "k2_correct": k2_correct,
                "base_margin": base_margin,
                "k2_margin": k2_margin,
                "corrected": not base_correct and k2_correct,
                "reversed":  base_correct and not k2_correct,
                "margin_change": k2_margin - base_margin,
            })

        wrong_recs = [r for r in records if not r["base_correct"]]
        if not wrong_recs:
            h1_results[task_name] = {}
            continue

        # Split by |base_margin|: high confidence (big negative margin = strongly wrong)
        sorted_by_confidence = sorted(wrong_recs, key=lambda r: abs(r["base_margin"]), reverse=True)
        half = len(sorted_by_confidence) // 2
        high_conf_wrong = sorted_by_confidence[:half]  # wrong with high confidence
        low_conf_wrong  = sorted_by_confidence[half:]  # wrong with low confidence

        corr_high = np.mean([r["corrected"] for r in high_conf_wrong])
        corr_low  = np.mean([r["corrected"] for r in low_conf_wrong])
        # Also: margin change analysis
        margin_change_correct = [r["margin_change"] for r in records if r["base_correct"]]
        margin_change_wrong   = [r["margin_change"] for r in wrong_recs]

        h1_results[task_name] = {
            "n_total": len(records),
            "n_wrong": len(wrong_recs),
            "n_high_conf_wrong": len(high_conf_wrong),
            "n_low_conf_wrong":  len(low_conf_wrong),
            "corr_rate_high_conf": float(corr_high),
            "corr_rate_low_conf":  float(corr_low),
            "high_conf_mean_abs_margin": float(np.mean([abs(r["base_margin"]) for r in high_conf_wrong])),
            "low_conf_mean_abs_margin":  float(np.mean([abs(r["base_margin"]) for r in low_conf_wrong])),
            "overall_acc_base": float(np.mean([r["base_correct"] for r in records])),
            "overall_acc_k2":   float(np.mean([r["k2_correct"]   for r in records])),
            "n_corrected": sum(r["corrected"] for r in wrong_recs),
            "margin_change_correct_mean": float(np.mean(margin_change_correct)) if margin_change_correct else 0,
            "margin_change_wrong_mean":   float(np.mean(margin_change_wrong))   if margin_change_wrong   else 0,
        }
        print(f"\n  {task_name}: baseline={h1_results[task_name]['overall_acc_base']:.4f}, "
              f"k=2={h1_results[task_name]['overall_acc_k2']:.4f}")
        print(f"    Correction rate — high-conf wrong: {corr_high:.3f}  vs  "
              f"low-conf wrong: {corr_low:.3f}")
        print(f"    (High confidence = |margin| > {h1_results[task_name]['high_conf_mean_abs_margin']:.3f}, "
              f"Low = {h1_results[task_name]['low_conf_mean_abs_margin']:.3f})")
        print(f"    H1 {'✅ SUPPORTED' if corr_high < corr_low else '❌ CONTRADICTED (low-conf corrected more)'}")

    results["h1_difficulty_mediation"] = h1_results

    # H1 Plots
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        f"H1: Task Difficulty Mediates ETD Gain\n"
        f"Does HIGH baseline confidence mean MORE likely to be corrected by ETD?",
        fontsize=12, fontweight="bold"
    )

    task_colors = {"boolq": "#2196F3", "arc": "#FF5722"}
    ax = axes[0]
    tasks = list(h1_results.keys())
    x = np.arange(len(tasks))
    w = 0.35
    high_corr_vals = [h1_results[t].get("corr_rate_high_conf", 0) for t in tasks]
    low_corr_vals  = [h1_results[t].get("corr_rate_low_conf",  0) for t in tasks]
    bars1 = ax.bar(x - w/2, high_corr_vals, w, label="High-confidence wrong", color="#FF5722", alpha=0.85)
    bars2 = ax.bar(x + w/2, low_corr_vals,  w, label="Low-confidence wrong",  color="#4CAF50", alpha=0.85)
    ax.bar_label(bars1, labels=[f"{v:.2f}" for v in high_corr_vals], fontsize=9, padding=2)
    ax.bar_label(bars2, labels=[f"{v:.2f}" for v in low_corr_vals],  fontsize=9, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([t.upper() for t in tasks])
    ax.set_ylabel("Correction rate")
    ax.set_title("Correction Rate:\nHigh- vs Low-confidence Wrong Examples")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    for col_idx, task_name in enumerate(["boolq", "arc"]):
        ax = axes[col_idx + 1]
        if task_name not in h1_results or not h1_results[task_name]:
            continue
        r_task = h1_results[task_name]
        # Scatter: base_margin vs margin_change, for all examples
        # (Using the pre-computed records stored in h1_results is complex; simplify to bar)
        cats = ["High-conf\nwrong", "Low-conf\nwrong"]
        vals = [r_task.get("corr_rate_high_conf", 0), r_task.get("corr_rate_low_conf", 0)]
        colors = ["#FF5722", "#4CAF50"]
        bars = ax.bar(cats, vals, color=colors, alpha=0.85, width=0.5)
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in vals], fontsize=10, padding=3)
        ax.set_title(f"{task_name.upper()}\n"
                     f"Baseline acc={r_task['overall_acc_base']:.3f} → k=2 acc={r_task['overall_acc_k2']:.3f}\n"
                     f"n_wrong={r_task['n_wrong']}, corrected={r_task['n_corrected']}")
        ax.set_ylabel("Correction rate (initially wrong → k=2 right)")
        ax.grid(True, alpha=0.3, axis="y")
        # Annotate hypothesis verdict
        verdict = "✅ Low-conf corrected more" if vals[1] > vals[0] else "❌ High-conf corrected more"
        ax.annotate(verdict, xy=(0.5, 0.92), xycoords="axes fraction",
                    ha="center", fontsize=9, color="darkblue",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "r6_h1_difficulty_mediation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: r6_h1_difficulty_mediation.png\n")

    # ══════════════════════════════════════════════════════════════════════════
    # H2: LATE T-BLOCK LAYER DOMINANCE
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 50)
    print(f"H2: Late T-block Layer Dominance  (N={N_EVAL} samples)")
    print("  Hypothesis: Repeating late layers (15-21) ≥ early layers (8-14)")
    print(f"  Testing {len(H2_CONFIGS)} configs on BoolQ + ARC")
    print("=" * 50)

    h2_results = {}
    for cfg_name, n_e, n_t, cfg_desc in H2_CONFIGS:
        alpha_cfg = _adaptive_alpha(n_t)
        eff_cfg   = alpha_cfg * n_t
        h2_results[cfg_name] = {
            "desc": cfg_desc, "n_e": n_e, "n_t": n_t,
            "alpha": alpha_cfg, "eff_step": eff_cfg,
            "t_start": n_e, "t_end": n_e + n_t,
        }
        line = f"  {cfg_desc:<35} n_e={n_e} n_t={n_t:2d} alpha={alpha_cfg:.3f} eff={eff_cfg:.1f}"
        for task, examples in [("boolq", boolq_300), ("arc", arc_300)]:
            acc = eval_mc(model, tokenizer, examples, n_e, n_t, 2, alpha_cfg, device,
                          desc=f"H2/{cfg_name}/{task}")
            delta = acc - baselines.get(task, 0)
            h2_results[cfg_name][task] = {"acc": acc, "delta": delta}
            line += f"  {task[:3]}={acc:.4f}({delta:+.4f})"
        print(line)

    results["h2_late_tblock"] = h2_results

    # H2 Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"H2: Late vs Early T-block Layer Dominance  (N={N_EVAL}, k=2, eff_step=6)\n"
        f"Hypothesis from M2: later layers (18-21) stay active longer → repeating them is more effective",
        fontsize=11, fontweight="bold"
    )

    cfg_order = [c[0] for c in H2_CONFIGS]
    cfg_labels = [c[3][:25] for c in H2_CONFIGS]
    cfg_t_starts = [c[1] for c in H2_CONFIGS]

    for col_idx, task in enumerate(["boolq", "arc"]):
        ax = axes[col_idx]
        bl = baselines.get(task, 0)
        acc_vals = [h2_results[cfg_name][task]["acc"] for cfg_name in cfg_order]
        delta_vals = [h2_results[cfg_name][task]["delta"] for cfg_name in cfg_order]
        colors_h2 = ["#2196F3", "#4CAF50", "#FF9800", "#FF5722", "#9C27B0"]
        bars = ax.barh(cfg_labels, acc_vals, color=colors_h2[:len(cfg_order)], alpha=0.85, height=0.6)
        ax.axvline(bl, color="#9E9E9E", ls="--", lw=1.5, label=f"Baseline ({bl:.3f})")
        ax.bar_label(bars,
                     labels=[f"{v:.4f} ({d:+.3f})" for v, d in zip(acc_vals, delta_vals)],
                     fontsize=8, padding=3)
        ax.set_xlabel("Accuracy")
        ax.set_title(f"{task.upper()}: Accuracy by T-block Position\n"
                     f"(Same eff_step=6 for each config)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="x")
        ax.set_xlim(max(0, bl - 0.05), min(1.0, max(acc_vals) + 0.05))

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "r6_h2_late_tblock.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: r6_h2_late_tblock.png\n")

    # ══════════════════════════════════════════════════════════════════════════
    # H3: PER-SAMPLE cos_sim AS ADAPTIVE k-SELECTION SIGNAL
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 50)
    print(f"H3: Per-sample cos_sim as Adaptive k-selection Signal  (N={N_MECH} BoolQ)")
    print("  Hypothesis: High cos_sim samples benefit MORE from k=3 vs k=2")
    print("=" * 50)

    n_e_h3, n_t_h3 = 8, 14
    alpha_h3_k2 = _adaptive_alpha(n_t_h3)         # k=2: eff=6
    alpha_h3_k3 = 12.0 / (3 * n_t_h3)             # k=3: energy conservation

    boolq_h3 = load_boolq_examples("validation", N_MECH)
    arc_h3   = load_arc_examples("test", N_MECH)

    h3_records = {task: [] for task in ["boolq", "arc"]}

    for task_name, examples in [("boolq", boolq_h3), ("arc", arc_h3)]:
        for prefix, conts, label in tqdm(examples, desc=f"H3/{task_name}", leave=True):
            # k=2 score
            k2_scores, k2_pred = per_sample_mc_score(model, tokenizer, prefix, conts, device,
                                                      n_e_h3, n_t_h3, 2, alpha_h3_k2)
            k2_correct = (k2_pred == label)
            # k=3 score
            k3_scores, k3_pred = per_sample_mc_score(model, tokenizer, prefix, conts, device,
                                                      n_e_h3, n_t_h3, 3, alpha_h3_k3)
            k3_correct = (k3_pred == label)
            # cos_sim (computed from T-block tracking at k=2)
            full = prefix + conts[0]
            enc = tokenizer(full, return_tensors="pt", add_special_tokens=False)
            ids = enc["input_ids"].to(device)
            attn = enc.get("attention_mask")
            if attn is not None:
                attn = attn.to(device)
            iter_last = run_T_block_track(model, ids, attn, n_e_h3, n_t_h3, k_max=2, alpha=alpha_h3_k2)
            dir_k1 = iter_last[1] - iter_last[0]
            dir_k2 = iter_last[2] - iter_last[1]
            cos_sim = F.cosine_similarity(dir_k2.unsqueeze(0), dir_k1.unsqueeze(0)).item()

            h3_records[task_name].append({
                "label": label,
                "k2_correct": k2_correct,
                "k3_correct": k3_correct,
                "cos_sim": cos_sim,
                "k3_better": k3_correct and not k2_correct,
                "k2_better": k2_correct and not k3_correct,
                "both_correct": k2_correct and k3_correct,
                "both_wrong": not k2_correct and not k3_correct,
            })

    h3_results = {}
    for task_name, recs in h3_records.items():
        if not recs:
            continue
        # Split by cos_sim (median split)
        med = np.median([r["cos_sim"] for r in recs])
        high_cos = [r for r in recs if r["cos_sim"] >= med]
        low_cos  = [r for r in recs if r["cos_sim"] <  med]

        acc_k2_high = np.mean([r["k2_correct"] for r in high_cos])
        acc_k3_high = np.mean([r["k3_correct"] for r in high_cos])
        acc_k2_low  = np.mean([r["k2_correct"] for r in low_cos])
        acc_k3_low  = np.mean([r["k3_correct"] for r in low_cos])
        k3_benefit_high = acc_k3_high - acc_k2_high
        k3_benefit_low  = acc_k3_low  - acc_k2_low

        overall_k2 = np.mean([r["k2_correct"] for r in recs])
        overall_k3 = np.mean([r["k3_correct"] for r in recs])

        h3_results[task_name] = {
            "n": len(recs),
            "cos_sim_median": float(med),
            "overall_acc_k2": float(overall_k2),
            "overall_acc_k3": float(overall_k3),
            "high_cos_acc_k2": float(acc_k2_high), "high_cos_acc_k3": float(acc_k3_high),
            "low_cos_acc_k2":  float(acc_k2_low),  "low_cos_acc_k3":  float(acc_k3_low),
            "k3_benefit_high_cos": float(k3_benefit_high),
            "k3_benefit_low_cos":  float(k3_benefit_low),
            "cos_sim_mean": float(np.mean([r["cos_sim"] for r in recs])),
        }
        print(f"\n  {task_name}: overall k2={overall_k2:.4f}, k3={overall_k3:.4f}")
        print(f"    cos_sim median={med:.4f}")
        print(f"    High cos_sim half: k2={acc_k2_high:.4f}, k3={acc_k3_high:.4f}, "
              f"k3-benefit={k3_benefit_high:+.4f}")
        print(f"    Low  cos_sim half: k2={acc_k2_low:.4f},  k3={acc_k3_low:.4f},  "
              f"k3-benefit={k3_benefit_low:+.4f}")
        verdict = "✅ SUPPORTED" if k3_benefit_high > k3_benefit_low else "❌ NOT SUPPORTED"
        print(f"    H3 {verdict}: high-cos k3-benefit={k3_benefit_high:+.4f} vs low-cos={k3_benefit_low:+.4f}")

    results["h3_cosim_k_selection"] = h3_results

    # H3 Plots
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        f"H3: Per-sample cos_sim as Adaptive k-selection Signal  (N={N_MECH})\n"
        f"Config: n_e={n_e_h3}, n_t={n_t_h3}  |  alpha_k2={alpha_h3_k2:.3f}, alpha_k3={alpha_h3_k3:.3f}",
        fontsize=11, fontweight="bold"
    )

    task_list = [t for t in ["boolq", "arc"] if t in h3_results]

    ax = axes[0]
    x = np.arange(len(task_list))
    w = 0.35
    k2_acc_all = [h3_results[t]["overall_acc_k2"] for t in task_list]
    k3_acc_all = [h3_results[t]["overall_acc_k3"] for t in task_list]
    bl_vals_h3 = [baselines.get(t, 0) for t in task_list]
    ax.bar(x - w, bl_vals_h3, w*0.8, label="Baseline", color="#9E9E9E", alpha=0.8)
    ax.bar(x,     k2_acc_all, w,     label=f"k=2 (eff=6)", color="#2196F3", alpha=0.85)
    ax.bar(x + w, k3_acc_all, w,     label=f"k=3 (energy conserv.)", color="#FF9800", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([t.upper() for t in task_list])
    ax.set_ylabel("Accuracy")
    ax.set_title("Overall: k=2 vs k=3 Accuracy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    for col_idx, task_name in enumerate(task_list[:2]):
        ax = axes[col_idx + 1]
        hr = h3_results[task_name]
        groups = ["High cos_sim\n(≥median)", "Low cos_sim\n(<median)"]
        k2_vals = [hr["high_cos_acc_k2"], hr["low_cos_acc_k2"]]
        k3_vals = [hr["high_cos_acc_k3"], hr["low_cos_acc_k3"]]
        x2 = np.arange(len(groups))
        bars1 = ax.bar(x2 - 0.2, k2_vals, 0.35, label="k=2", color="#2196F3", alpha=0.85)
        bars2 = ax.bar(x2 + 0.2, k3_vals, 0.35, label="k=3", color="#FF9800", alpha=0.85)
        ax.bar_label(bars1, labels=[f"{v:.3f}" for v in k2_vals], fontsize=9, padding=2)
        ax.bar_label(bars2, labels=[f"{v:.3f}" for v in k3_vals], fontsize=9, padding=2)
        benefit_labels = [f"k3-benefit\n{hr['k3_benefit_high_cos']:+.3f}",
                          f"k3-benefit\n{hr['k3_benefit_low_cos']:+.3f}"]
        ax.set_xticks(x2)
        ax.set_xticklabels(groups, fontsize=9)
        ax.set_ylabel("Accuracy")
        verdict = "✅ High-cos benefits more" if hr["k3_benefit_high_cos"] > hr["k3_benefit_low_cos"] \
                  else "❌ Low-cos benefits more (or equal)"
        ax.set_title(f"{task_name.upper()}: k=2 vs k=3 by cos_sim Group\n"
                     f"cos_sim median={hr['cos_sim_median']:.3f}\n{verdict}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "r6_h3_cosim_k_selection.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: r6_h3_cosim_k_selection.png\n")

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    total_time = time.time() - t_global
    results["meta"] = {
        "timestamp":    time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_s": total_time,
        "n_eval": N_EVAL, "n_t8": N_T8, "n_mech": N_MECH,
        "model": MODEL_PATH,
    }

    out_path = RESULTS_DIR / "round6_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved: {out_path}")

    print("\n" + "=" * 70)
    print("ROUND-6 EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Total runtime: {total_time/60:.1f} min\n")

    print("[T6] Zero-shot layer selection:")
    print(f"  Auto-selected n_e={zs_n_e} (threshold={THRESHOLD})")
    for bname in ["commonsenseqa", "truthfulqa"]:
        if bname in t6_results and isinstance(t6_results[bname], dict):
            r = t6_results[bname]
            print(f"  {bname}: baseline={r['baseline']:.4f} | "
                  f"ZS={r['zs_selected']['acc']:.4f}({r['zs_selected']['delta']:+.4f}) | "
                  f"oracle={r['oracle_champion']['acc']:.4f}({r['oracle_champion']['delta']:+.4f})")

    print(f"\n[T8] delta_norm correlation:")
    t8r = results["t8_delta_norm_correlation"]
    print(f"  AUC(dn→corrected)={t8r['auc_dn_correction']:.4f} | "
          f"AUC(cos→corrected)={t8r['auc_cos_correction']:.4f} | "
          f"AUC(|margin|→corrected)={t8r['auc_margin_correction']:.4f}")
    print(f"  Interpretation: AUC≈0.5 = no individual signal; AUC>0.65 = useful predictor")

    print(f"\n[H1] Difficulty mediation:")
    for tname, r in results["h1_difficulty_mediation"].items():
        if r:
            print(f"  {tname}: high-conf-wrong correction={r.get('corr_rate_high_conf',0):.3f} | "
                  f"low-conf-wrong correction={r.get('corr_rate_low_conf',0):.3f}")

    print(f"\n[H2] Late vs Early T-block:")
    for cfg_name, cfg_data in results["h2_late_tblock"].items():
        boolq_acc = cfg_data.get("boolq", {}).get("acc", 0)
        arc_acc   = cfg_data.get("arc",   {}).get("acc", 0)
        print(f"  {cfg_data['desc']:<35} boolq={boolq_acc:.4f} arc={arc_acc:.4f}")

    print(f"\n[H3] cos_sim k-selection:")
    for tname, r in results["h3_cosim_k_selection"].items():
        print(f"  {tname}: high-cos k3-benefit={r.get('k3_benefit_high_cos',0):+.4f} | "
              f"low-cos k3-benefit={r.get('k3_benefit_low_cos',0):+.4f}")

    print("\nFigures generated:")
    for p in sorted(FIGURES_DIR.glob("r6_*.png")):
        print(f"  {p}")
    print("\nDone!")


if __name__ == "__main__":
    main()
