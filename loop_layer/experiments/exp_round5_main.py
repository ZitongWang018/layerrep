#!/usr/bin/env python3
"""
exp_round5_main.py  ─  Round-5 Comprehensive ETD Experiments (2026-04-07)

Upgrades over Round 4:
  - Larger samples: N_EVAL=200, N_SELECT=80, N_PROFILE=128, N_MECH=60
  - All plot text in English (titles, legends, axis labels)
  - Three new mechanistic analyses:
      M1: Logit Evolution  — how P(correct answer) evolves across ETD iterations
      M2: Per-layer Convergence Heatmap — which T-block layers converge first
      M3: Token-selective delta_norm — answer vs context token behavior

Experiments:
  A3+  : Enhanced autonomous layer selection (80-sample delta_norm scan)
  A4+  : Total Iteration Energy Conservation (k=2/3/4, alpha scan)
  LP+  : Enhanced layer profiling (128 samples, 4 metrics)
  M1   : Logit Evolution Analysis
  M2   : Per-layer Convergence Heatmap
  M3   : Token-selective delta_norm

Benchmarks: BoolQ + ARC-Challenge + CommonsenseQA + TruthfulQA-MC1
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
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─── Paths ──────────────────────────────────────────────────────────────────
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

# ─── Experiment Parameters ───────────────────────────────────────────────────
N_SELECT  = 80   # A3 layer-selection scan
N_EVAL    = 200  # Accuracy evaluation
N_PROFILE = 128  # Layer property profiling
N_MECH    = 60   # Mechanistic analysis (small, per-example tracking)

# A3: fixed n_e=8, scan n_t
A3_N_E = 8
A3_N_T_CANDIDATES = [6, 8, 10, 12, 14, 16, 18]
A3_K_VALUES = [2, 3]

# A4: two canonical configs, test k=2/3/4
A4_CONFIGS = [
    ("n_e8_n_t14", 8, 14),   # champion config
    ("n_e10_n_t11", 10, 11), # secondary config
]
# Energy constant hypothesis: k × alpha × n_t = 12
# → alpha_k = 12 / (k × n_t)
A4_K_VALUES = [2, 3, 4]

# LP: layer profiling t_start groups
LP_BAD_STARTS  = [5, 6]
LP_GOOD_STARTS = [8, 10]
LP_MID_STARTS  = [12, 14]

# M2: Convergence heatmap config
M2_N_E, M2_N_T = 8, 14  # champion config
M2_K_MAX = 4             # iterate up to k=4, track per-layer

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


def load_all_benchmarks(limit: int) -> dict:
    data = {}
    print(f"  Loading BoolQ  (validation, {limit})")
    data["boolq"] = load_boolq_examples("validation", limit)
    print(f"  Loading ARC    (test, {limit})")
    data["arc"] = load_arc_examples("test", limit)
    try:
        print(f"  Loading CommonsenseQA (validation, {limit})")
        data["commonsenseqa"] = load_commonsenseqa(limit)
        print(f"    -> {len(data['commonsenseqa'])} examples")
    except Exception as e:
        print(f"    CommonsenseQA failed: {e}")
    try:
        print(f"  Loading TruthfulQA-MC1 (validation, {limit})")
        data["truthfulqa"] = load_truthfulqa(limit)
        print(f"    -> {len(data['truthfulqa'])} examples")
    except Exception as e:
        print(f"    TruthfulQA failed: {e}")
    return data

# ─── ETD Utilities ──────────────────────────────────────────────────────────

@torch.inference_mode()
def _run_T_block_track(
    model,
    input_ids: torch.Tensor,
    attention_mask,
    n_e: int,
    n_t: int,
    k_max: int,
    alpha: float,
) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
    """
    Run E-block then T-block up to k_max times.
    Returns:
      iter_states: list of hidden states after each full T iteration (length k_max+1, including h_e)
      layer_states_per_iter: for each iteration, list of hidden states after each T-block layer
        shape: [k_max][n_t] each being (1, seq, dim)
    """
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

    # E block
    for i in range(n_e):
        hidden_states = _run_layer(i, hidden_states)

    iter_states = [hidden_states.clone()]          # h_e
    layer_states_per_iter = []                     # [k][layer_in_T]

    use_damping = alpha < 1.0
    for _ in range(k_max):
        h_prev = hidden_states
        layer_hs = []
        for i in range(n_e, n_e + n_t):
            hidden_states = _run_layer(i, hidden_states)
            layer_hs.append(hidden_states.clone())
        if use_damping:
            hidden_states = alpha * hidden_states + (1.0 - alpha) * h_prev
        iter_states.append(hidden_states.clone())
        layer_states_per_iter.append(layer_hs)

    return iter_states, layer_states_per_iter


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


def _logit_for_continuation(model, tokenizer, prefix, conts, device, n_e, n_t, k, alpha):
    """Return index of best continuation (ETD mode)."""
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
    return scores


def _logit_for_continuation_baseline(model, tokenizer, prefix, conts, device):
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
    return scores

# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    t_global = time.time()
    print("=" * 70)
    print("Round-5 ETD Experiments: A3+ / A4+ / LP+ / M1 / M2 / M3")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"N_EVAL={N_EVAL}  N_SELECT={N_SELECT}  N_PROFILE={N_PROFILE}  N_MECH={N_MECH}")
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

    print("Loading datasets ...")
    all_data = load_all_benchmarks(N_EVAL)
    select_data = {k: v[:N_SELECT] for k, v in all_data.items()}
    bench_names = list(all_data.keys())
    print(f"  Benchmarks: {bench_names}\n")

    results: dict = {
        "meta": {},
        "baselines": {},
        "a3_layer_select": {},
        "a4_energy_conservation": {},
        "layer_profile": {},
        "m1_logit_evolution": {},
        "m2_convergence_heatmap": {},
        "m3_token_selective": {},
    }

    # ══════════════════════════════════════════════════════════════════════════
    # BASELINE
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 50)
    print("BASELINE EVALUATION")
    print("=" * 50)
    baselines = {}
    for bname in bench_names:
        ex = all_data[bname][:N_EVAL]
        acc = eval_mc_baseline(model, tokenizer, ex, device, desc=f"baseline/{bname}")
        baselines[bname] = acc
        print(f"  {bname:<20}: {acc:.4f}")
    results["baselines"] = baselines
    print()

    # ══════════════════════════════════════════════════════════════════════════
    # A3+: ENHANCED AUTONOMOUS LAYER SELECTION
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 50)
    print("A3+: Autonomous Layer Selection (delta_norm scan)")
    print(f"  Fixed n_e={A3_N_E}, scanning n_t={A3_N_T_CANDIDATES}")
    print(f"  {N_SELECT} samples from BoolQ + ARC")
    print("=" * 50)

    a3_delta_norms = {bname: {} for bname in ["boolq", "arc"]}
    a3_cos_sims    = {bname: {} for bname in ["boolq", "arc"]}

    for n_t in A3_N_T_CANDIDATES:
        alpha = _adaptive_alpha(n_t)
        for task in ["boolq", "arc"]:
            examples = select_data[task]
            delta_list, cos_list = [], []
            for prefix, conts, _ in examples:
                full = prefix + conts[0]
                enc = tokenizer(full, return_tensors="pt", add_special_tokens=False)
                ids = enc["input_ids"].to(device)
                attn = enc.get("attention_mask")
                if attn is not None:
                    attn = attn.to(device)
                iter_states, _ = _run_T_block_track(
                    model, ids, attn, A3_N_E, n_t, k_max=2, alpha=alpha
                )
                h_e, h_1, h_2 = iter_states[0], iter_states[1], iter_states[2]
                h_e_last = h_e[0, -1]
                h_1_last = h_1[0, -1]
                h_2_last = h_2[0, -1]

                diff_21 = h_2_last - h_1_last
                diff_10 = h_1_last - h_e_last
                dn = diff_21.norm().item() / (h_1_last.norm().item() + 1e-8)
                cs = F.cosine_similarity(diff_21.unsqueeze(0), diff_10.unsqueeze(0)).item()
                delta_list.append(dn)
                cos_list.append(cs)

            a3_delta_norms[task][n_t] = float(np.mean(delta_list))
            a3_cos_sims[task][n_t]    = float(np.mean(cos_list))
            print(f"  n_t={n_t:2d} alpha={alpha:.3f}  [{task}]  "
                  f"delta_norm={a3_delta_norms[task][n_t]:.4f}  "
                  f"cos_sim={a3_cos_sims[task][n_t]:.4f}")

    avg_delta = {
        n_t: (a3_delta_norms["boolq"][n_t] + a3_delta_norms["arc"][n_t]) / 2
        for n_t in A3_N_T_CANDIDATES
    }
    selected_n_t = min(avg_delta, key=avg_delta.get)
    selected_alpha = _adaptive_alpha(selected_n_t)
    eff_step = selected_alpha * selected_n_t
    print(f"\n  >>> Auto-selected: n_t={selected_n_t}, alpha={selected_alpha:.3f}, "
          f"eff_step={eff_step:.2f}")

    # Phase 2: validate selected config
    print(f"\n  Validating selected config (n_e={A3_N_E}, n_t={selected_n_t}) on {N_EVAL} samples ...")
    a3_validation = {}
    for k_val in A3_K_VALUES:
        a3_validation[k_val] = {}
        for bname in bench_names:
            ex = all_data[bname][:N_EVAL]
            acc = eval_mc(model, tokenizer, ex, A3_N_E, selected_n_t, k_val, selected_alpha,
                          device, desc=f"A3-k{k_val}/{bname}")
            delta = acc - baselines.get(bname, 0)
            a3_validation[k_val][bname] = {"acc": acc, "delta": delta}
            print(f"  k={k_val}  {bname:<20}: {acc:.4f}  (delta={delta:+.4f})")

    # Champion comparison
    print(f"\n  Champion reference (n_e=8, n_t=14, k=2):")
    a3_champion = {}
    for bname in bench_names:
        ex = all_data[bname][:N_EVAL]
        acc = eval_mc(model, tokenizer, ex, 8, 14, 2, _adaptive_alpha(14), device,
                      desc=f"champion/{bname}")
        delta = acc - baselines.get(bname, 0)
        a3_champion[bname] = {"acc": acc, "delta": delta}
        print(f"  champion  {bname:<20}: {acc:.4f}  (delta={delta:+.4f})")

    results["a3_layer_select"] = {
        "delta_norms_boolq":  {str(k): v for k, v in a3_delta_norms["boolq"].items()},
        "delta_norms_arc":    {str(k): v for k, v in a3_delta_norms["arc"].items()},
        "cos_sims_boolq":     {str(k): v for k, v in a3_cos_sims["boolq"].items()},
        "cos_sims_arc":       {str(k): v for k, v in a3_cos_sims["arc"].items()},
        "avg_delta":          {str(k): v for k, v in avg_delta.items()},
        "selected_n_t":       selected_n_t,
        "selected_alpha":     selected_alpha,
        "eff_step":           eff_step,
        "validation":         {str(k): v for k, v in a3_validation.items()},
        "champion":           a3_champion,
    }

    # Plot A3
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        f"A3: Autonomous Layer Selection  (N_select={N_SELECT}, N_eval={N_EVAL})",
        fontsize=13, fontweight="bold"
    )
    n_ts = A3_N_T_CANDIDATES

    ax = axes[0]
    ax.plot(n_ts, [a3_delta_norms["boolq"][n] for n in n_ts], "o-", color="#2196F3", label="BoolQ")
    ax.plot(n_ts, [a3_delta_norms["arc"][n]   for n in n_ts], "s-", color="#FF5722", label="ARC")
    ax.plot(n_ts, [avg_delta[n] for n in n_ts], "^--", color="#4CAF50", lw=2, label="Average")
    ax.axvline(selected_n_t, color="gray", ls=":", alpha=0.8, label=f"Selected n_t={selected_n_t}")
    ax.set_xlabel("n_t (T-block size)")
    ax.set_ylabel("delta_norm  (k=2 vs k=1)")
    ax.set_title("delta_norm vs n_t\n(lower = more stable)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_ts)

    ax = axes[1]
    ax.plot(n_ts, [a3_cos_sims["boolq"][n] for n in n_ts], "o-", color="#2196F3", label="BoolQ")
    ax.plot(n_ts, [a3_cos_sims["arc"][n]   for n in n_ts], "s-", color="#FF5722", label="ARC")
    ax.axvline(selected_n_t, color="gray", ls=":", alpha=0.8)
    ax.axhline(0, color="black", lw=0.8, alpha=0.3)
    ax.set_xlabel("n_t (T-block size)")
    ax.set_ylabel("cos_sim (direction consistency)")
    ax.set_title("cos_sim vs n_t\n(+1=same dir, -1=opposite dir)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_ts)

    ax = axes[2]
    x = np.arange(len(bench_names))
    w = 0.18
    bl_vals = [baselines.get(b, 0) for b in bench_names]
    ax.bar(x - w*1.5, bl_vals, w, label="Baseline", color="#9E9E9E", alpha=0.75)
    if 2 in a3_validation:
        vals_k2 = [a3_validation[2][b]["acc"] for b in bench_names]
        ax.bar(x - w*0.5, vals_k2, w, label=f"Auto-select k=2 (n_t={selected_n_t})", color="#2196F3", alpha=0.85)
    if 3 in a3_validation:
        vals_k3 = [a3_validation[3][b]["acc"] for b in bench_names]
        ax.bar(x + w*0.5, vals_k3, w, label=f"Auto-select k=3 (n_t={selected_n_t})", color="#FF9800", alpha=0.85)
    champ_vals = [a3_champion[b]["acc"] for b in bench_names]
    ax.bar(x + w*1.5, champ_vals, w, label="Champion (n_t=14, k=2)", color="#4CAF50", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([b[:8] for b in bench_names], fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title("Auto-selected vs Champion vs Baseline")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "r5_a3_layer_selection.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {FIGURES_DIR}/r5_a3_layer_selection.png")

    # ══════════════════════════════════════════════════════════════════════════
    # A4+: TOTAL ITERATION ENERGY CONSERVATION
    # Hypothesis: k x alpha x n_t = 12 (constant) for optimal performance
    # Test k=2,3,4 and verify the formula alpha_k = 12/(k*n_t)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 50)
    print("A4+: Total Iteration Energy Conservation")
    print("  Hypothesis: k x alpha x n_t = 12 (constant)")
    print(f"  Testing k in {A4_K_VALUES}, {N_EVAL} samples, BoolQ + ARC")
    print("=" * 50)

    ENERGY_CONST = 12.0

    def alpha_energy(k, n_t, C=ENERGY_CONST):
        return min(1.0, C / (k * n_t))

    a4_results = {}
    eval_tasks_a4 = ["boolq", "arc"]

    for cfg_name, n_e, n_t in A4_CONFIGS:
        a4_results[cfg_name] = {}
        print(f"\n  Config {cfg_name}: n_e={n_e}, n_t={n_t}")
        print(f"  {'Hyp':<12} {'k':<3} {'alpha':>6}  {'eff_step':>8}", end="")
        for t in eval_tasks_a4:
            print(f"  {t[:6]:>10}", end="")
        print()
        print("  " + "-" * 60)

        for k_val in A4_K_VALUES:
            a_energy = alpha_energy(k_val, n_t)
            eff_step = a_energy * n_t
            hyp_label = f"k{k_val}_C{ENERGY_CONST:.0f}"
            a4_results[cfg_name][hyp_label] = {
                "k": k_val, "alpha": a_energy, "eff_step": eff_step,
                "total_energy": k_val * eff_step
            }
            line = f"  {hyp_label:<12} {k_val:<3} {a_energy:>6.3f}  {eff_step:>8.2f}"
            for task in eval_tasks_a4:
                ex = all_data[task][:N_EVAL]
                acc = eval_mc(model, tokenizer, ex, n_e, n_t, k_val, a_energy, device,
                              desc=f"A4/{cfg_name}/k{k_val}")
                delta = acc - baselines.get(task, 0)
                a4_results[cfg_name][hyp_label][task] = {"acc": acc, "delta": delta}
                line += f"  {acc:.4f}({delta:+.3f})"
            print(line)

        # Also test k=3 with eff=6 (naive, as control) and eff=4 (energy conserving)
        for label, alpha_val in [("k3_eff6", 6.0 / n_t), ("k3_eff4", 4.0 / n_t)]:
            a4_results[cfg_name][label] = {
                "k": 3, "alpha": alpha_val, "eff_step": alpha_val * n_t,
                "total_energy": 3 * alpha_val * n_t
            }
            line = f"  {label:<12} {'3':<3} {alpha_val:>6.3f}  {alpha_val * n_t:>8.2f}"
            for task in eval_tasks_a4:
                ex = all_data[task][:N_EVAL]
                acc = eval_mc(model, tokenizer, ex, n_e, n_t, 3, alpha_val, device,
                              desc=f"A4/{cfg_name}/{label}")
                delta = acc - baselines.get(task, 0)
                a4_results[cfg_name][label][task] = {"acc": acc, "delta": delta}
                line += f"  {acc:.4f}({delta:+.3f})"
            print(line)

    results["a4_energy_conservation"] = a4_results

    # Plot A4
    fig, axes = plt.subplots(len(A4_CONFIGS), 2, figsize=(13, 5 * len(A4_CONFIGS)))
    fig.suptitle(
        f"A4: Total Iteration Energy Conservation  (C=k×alpha×n_t={ENERGY_CONST:.0f})\n"
        f"N_eval={N_EVAL}, benchmarks: {', '.join(eval_tasks_a4)}",
        fontsize=12, fontweight="bold"
    )

    colors_k = {2: "#2196F3", 3: "#FF9800", 4: "#9C27B0"}
    markers_k = {2: "o", 3: "s", 4: "^"}

    for row_idx, (cfg_name, n_e, n_t) in enumerate(A4_CONFIGS):
        for col_idx, task in enumerate(eval_tasks_a4):
            ax = axes[row_idx][col_idx] if len(A4_CONFIGS) > 1 else axes[col_idx]
            bl = baselines.get(task, 0)
            ax.axhline(bl, color="#9E9E9E", ls="--", lw=1.5, label=f"Baseline ({bl:.3f})")

            for k_val in A4_K_VALUES:
                hyp = f"k{k_val}_C{ENERGY_CONST:.0f}"
                r = a4_results[cfg_name].get(hyp, {})
                acc = r.get(task, {}).get("acc", 0)
                eff = r.get("eff_step", 0)
                ax.scatter(eff, acc, s=120, color=colors_k[k_val],
                           marker=markers_k[k_val], zorder=5,
                           label=f"k={k_val} eff={eff:.1f} (Energy C)")

            # Control points k=3
            for ctrl_label, style in [("k3_eff6", "v"), ("k3_eff4", "D")]:
                r = a4_results[cfg_name].get(ctrl_label, {})
                acc = r.get(task, {}).get("acc", 0)
                eff = r.get("eff_step", 0)
                color = "#FF5722" if ctrl_label == "k3_eff6" else "#4CAF50"
                ax.scatter(eff, acc, s=100, color=color, marker=style, zorder=5,
                           label=f"k=3 eff={eff:.1f} (control)")

            ax.set_xlabel("Effective step size (alpha × n_t)")
            ax.set_ylabel(f"{task} accuracy")
            ax.set_title(f"{cfg_name}  |  {task}\n(n_e={n_e}, n_t={n_t})")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "r5_a4_energy_conservation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {FIGURES_DIR}/r5_a4_energy_conservation.png")

    # ══════════════════════════════════════════════════════════════════════════
    # LP+: ENHANCED LAYER PROPERTY PROFILING
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 50)
    print(f"LP+: Layer Property Profiling ({N_PROFILE} samples, {n_layers} layers)")
    print("=" * 50)

    boolq_prof = load_boolq_examples("validation", N_PROFILE)
    arc_prof   = load_arc_examples("test", N_PROFILE)

    def collect_layer_stats(examples):
        all_step = [[] for _ in range(n_layers)]
        all_norm  = [[] for _ in range(n_layers)]
        all_csim  = [[] for _ in range(n_layers)]
        all_tdiff = [[] for _ in range(n_layers)]
        for prefix, conts, _ in tqdm(examples, desc="  layer-profile", leave=False):
            full = prefix + conts[0]
            enc = tokenizer(full, return_tensors="pt", add_special_tokens=False)
            ids = enc["input_ids"].to(device)
            attn = enc.get("attention_mask")
            if attn is not None:
                attn = attn.to(device)
            with torch.inference_mode():
                out = model(input_ids=ids, attention_mask=attn,
                            output_hidden_states=True, use_cache=False)
            hs = out.hidden_states
            for l in range(n_layers):
                h_prev = hs[l][0]
                h_curr = hs[l + 1][0]
                diff = h_curr - h_prev
                step = diff.norm(dim=-1).mean().item() / (h_prev.norm(dim=-1).mean().item() + 1e-8)
                all_step[l].append(step)
                all_norm[l].append(h_curr.norm(dim=-1).mean().item())
                all_csim[l].append(F.cosine_similarity(h_curr, h_prev, dim=-1).mean().item())
                if h_curr.shape[0] > 1:
                    td = 1.0 - F.cosine_similarity(h_curr[:-1], h_curr[1:], dim=-1).mean().item()
                else:
                    td = 0.0
                all_tdiff[l].append(td)
        return {
            "step_size":  [float(np.mean(v)) for v in all_step],
            "norm":       [float(np.mean(v)) for v in all_norm],
            "cosine_sim": [float(np.mean(v)) for v in all_csim],
            "token_diff": [float(np.mean(v)) for v in all_tdiff],
        }

    print("  Profiling BoolQ ...")
    lp_boolq = collect_layer_stats(boolq_prof)
    print("  Profiling ARC ...")
    lp_arc   = collect_layer_stats(arc_prof)
    results["layer_profile"]["boolq"] = lp_boolq
    results["layer_profile"]["arc"]   = lp_arc

    # Decision signal comparison
    t_start_groups = {
        "Bad (5-6)":  LP_BAD_STARTS,
        "Good (8-10)": LP_GOOD_STARTS,
        "Mid (12-14)": LP_MID_STARTS,
    }
    signal_results = {}
    metrics = ["step_size", "norm", "cosine_sim", "token_diff"]
    metric_labels = [
        "Step Size  ||Δh||/||h||",
        "Hidden State Norm  ||h||",
        "Layer Cosine Sim  cos(h_l, h_{l-1})",
        "Token Differentiation  1-cos(h_i, h_{i+1})"
    ]
    for metric in metrics:
        signal_results[metric] = {}
        for gname, t_starts in t_start_groups.items():
            vals = [lp_boolq[metric][l] for l in t_starts] + [lp_arc[metric][l] for l in t_starts]
            signal_results[metric][gname] = float(np.mean(vals))
        print(f"  {metric:<18}:", {g: f"{v:.4f}" for g, v in signal_results[metric].items()})
    results["layer_profile"]["decision_signal"] = signal_results

    # LP Plot — 4-panel layer curves
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"LP: Layer Property Profile  (Baseline forward, N={N_PROFILE})\n"
        f"Green span = good t_start (8-10), Red span = bad t_start (5-6)",
        fontsize=12, fontweight="bold"
    )
    layer_idx = list(range(n_layers))
    for idx, (metric, ylabel) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx // 2][idx % 2]
        ax.plot(layer_idx, lp_boolq[metric], "-", color="#2196F3", lw=1.5, label="BoolQ")
        ax.plot(layer_idx, lp_arc[metric],   "-", color="#FF5722", lw=1.5, label="ARC")
        ax.axvspan(LP_BAD_STARTS[0] - 0.5, LP_BAD_STARTS[-1] + 0.5, color="#FF5722", alpha=0.15, label="Bad t_start (5-6)")
        ax.axvspan(LP_GOOD_STARTS[0] - 0.5, LP_GOOD_STARTS[-1] + 0.5, color="#4CAF50", alpha=0.15, label="Good t_start (8-10)")
        ax.axvspan(8 - 0.5, 22 + 0.5, color="#9C27B0", alpha=0.07, label="Optimal T-block (8-22)")
        ax.set_xlabel("Layer Index (0-35)")
        ax.set_ylabel(ylabel)
        ax.set_title(metric)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, n_layers, 4))
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "r5_lp_layer_profile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # LP signal comparison bar chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        "LP: Zero-Shot Layer Selection Signal\n"
        "Comparison of layer statistics at bad/good/mid t_start positions",
        fontsize=12, fontweight="bold"
    )
    group_colors = {"Bad (5-6)": "#FF5722", "Good (8-10)": "#4CAF50", "Mid (12-14)": "#2196F3"}
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2][idx % 2]
        groups = list(signal_results[metric].keys())
        vals = [signal_results[metric][g] for g in groups]
        bars = ax.bar(groups, vals, color=[group_colors.get(g, "#9E9E9E") for g in groups], alpha=0.82, width=0.45)
        ax.bar_label(bars, labels=[f"{v:.4f}" for v in vals], fontsize=9, padding=2)
        ax.set_title(f"{metric}\n({metric_labels[idx]})")
        ax.set_ylabel("Average value")
        ax.grid(True, alpha=0.3, axis="y")
        if metric == "step_size":
            ratio = signal_results[metric]["Bad (5-6)"] / (signal_results[metric]["Good (8-10)"] + 1e-8)
            ax.annotate(f"Bad/Good ratio: {ratio:.1f}x", xy=(0.02, 0.92),
                        xycoords="axes fraction", fontsize=9, color="darkred")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "r5_lp_decision_signal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plots saved: r5_lp_layer_profile.png, r5_lp_decision_signal.png")

    # ══════════════════════════════════════════════════════════════════════════
    # M1: LOGIT EVOLUTION ANALYSIS
    # Track how P(correct answer) changes across ETD iterations k=1..4
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 50)
    print(f"M1: Logit Evolution Analysis ({N_MECH} BoolQ samples)")
    print("  Tracking P(correct answer) across k=0,1,2,3,4 iterations")
    print("=" * 50)

    m1_n_e, m1_n_t = 8, 14
    m1_alpha = _adaptive_alpha(m1_n_t)   # k=2 optimal; we use same alpha for k tracking
    # For k tracking, use energy-conserving alpha for k=3+ = 4/n_t
    m1_alpha_ktrack = 4.0 / m1_n_t       # energy-conserving alpha (k=3, eff=4)

    m1_examples = load_boolq_examples("validation", N_MECH)
    # Track: scores for correct cont and best cont, at each k
    m1_records = []  # list of {k0_correct, k0_pred, k1_correct, ..., label}

    for prefix, conts, label in tqdm(m1_examples, desc="M1 logit tracking", leave=True):
        plen = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
        enc = tokenizer(prefix + conts[0], return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(device)

        record = {"label": label, "n_conts": len(conts)}

        # Baseline (k=0)
        scores_base = []
        for cont in conts:
            enc2 = tokenizer(prefix + cont, return_tensors="pt", add_special_tokens=False)
            ids2 = enc2["input_ids"].to(device)
            attn2 = enc2.get("attention_mask")
            if attn2 is not None:
                attn2 = attn2.to(device)
            logits = baseline_forward_logits(model, ids2, attn2)
            scores_base.append(loglikelihood_continuation(logits, ids2, plen))
        pred_base = max(range(len(scores_base)), key=lambda i: scores_base[i])
        record["k0"] = {"pred": pred_base, "correct": (pred_base == label),
                        "correct_score": scores_base[label], "margin": scores_base[label] - max(scores_base[i] for i in range(len(scores_base)) if i != label)}

        # ETD iterations k=1,2,3,4 (using energy-conserving alpha)
        for k_val in [1, 2, 3, 4]:
            alpha_use = m1_alpha_ktrack if k_val >= 3 else m1_alpha
            scores_k = []
            for cont in conts:
                enc2 = tokenizer(prefix + cont, return_tensors="pt", add_special_tokens=False)
                ids2 = enc2["input_ids"].to(device)
                attn2 = enc2.get("attention_mask")
                if attn2 is not None:
                    attn2 = attn2.to(device)
                logits = etd_forward_logits(model, ids2, attn2, m1_n_e, m1_n_t, k_val, alpha=alpha_use)
                scores_k.append(loglikelihood_continuation(logits, ids2, plen))
            pred_k = max(range(len(scores_k)), key=lambda i: scores_k[i])
            other_max = max(scores_k[i] for i in range(len(scores_k)) if i != label) if len(scores_k) > 1 else scores_k[0]
            record[f"k{k_val}"] = {
                "pred": pred_k, "correct": (pred_k == label),
                "correct_score": scores_k[label],
                "margin": scores_k[label] - other_max,
                "alpha": alpha_use
            }
        m1_records.append(record)

    # Analyze trajectories
    k_tags = ["k0", "k1", "k2", "k3", "k4"]
    acc_by_k = [np.mean([r[kt]["correct"] for r in m1_records]) for kt in k_tags]
    margin_by_k = [np.mean([r[kt]["margin"] for r in m1_records]) for kt in k_tags]

    # Transition matrix: initially-wrong (k0) split by what happens at each k
    initially_wrong = [r for r in m1_records if not r["k0"]["correct"]]
    initially_correct = [r for r in m1_records if r["k0"]["correct"]]

    corrections = {}  # {k_tag: fraction of initially-wrong that become correct}
    for kt in k_tags[1:]:
        corrections[kt] = np.mean([r[kt]["correct"] for r in initially_wrong]) if initially_wrong else 0

    reversals = {}  # {k_tag: fraction of initially-correct that become wrong}
    for kt in k_tags[1:]:
        reversals[kt] = np.mean([not r[kt]["correct"] for r in initially_correct]) if initially_correct else 0

    print(f"\n  Accuracy by iteration: {[f'{a:.3f}' for a in acc_by_k]}")
    print(f"  Margin by iteration:   {[f'{m:.4f}' for m in margin_by_k]}")
    print(f"  Correction rate (initially-wrong->correct): {corrections}")
    print(f"  Reversal rate   (initially-correct->wrong): {reversals}")

    results["m1_logit_evolution"] = {
        "n_examples": len(m1_records),
        "config": {"n_e": m1_n_e, "n_t": m1_n_t, "alpha_k12": m1_alpha, "alpha_k34": m1_alpha_ktrack},
        "acc_by_k": acc_by_k,
        "margin_by_k": margin_by_k,
        "corrections": corrections,
        "reversals": reversals,
        "initially_wrong_count": len(initially_wrong),
        "initially_correct_count": len(initially_correct),
    }

    # Plot M1
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"M1: Logit Evolution — How ETD Iterations Change Predictions\n"
        f"Config: n_e={m1_n_e}, n_t={m1_n_t}  "
        f"alpha(k<=2)={m1_alpha:.3f}, alpha(k>=3)={m1_alpha_ktrack:.3f}  "
        f"N={N_MECH} BoolQ examples",
        fontsize=11, fontweight="bold"
    )

    k_labels = ["k=0\n(Baseline)", "k=1", "k=2", "k=3", "k=4"]
    ax = axes[0]
    ax.plot(range(5), acc_by_k, "o-", color="#2196F3", lw=2, markersize=8)
    for i, (kl, a) in enumerate(zip(k_labels, acc_by_k)):
        ax.annotate(f"{a:.3f}", (i, a), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xticks(range(5))
    ax.set_xticklabels(k_labels, fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy per Iteration")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(max(0, min(acc_by_k) - 0.05), min(1.0, max(acc_by_k) + 0.05))

    ax = axes[1]
    ax.plot(range(5), margin_by_k, "s-", color="#FF5722", lw=2, markersize=8)
    for i, (kl, m) in enumerate(zip(k_labels, margin_by_k)):
        ax.annotate(f"{m:.3f}", (i, m), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xticks(range(5))
    ax.set_xticklabels(k_labels, fontsize=9)
    ax.set_ylabel("Avg log-prob margin (correct - best-wrong)")
    ax.set_title("Prediction Confidence Margin per Iteration")
    ax.axhline(0, color="black", lw=0.7, alpha=0.4)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    k_plot = ["k1", "k2", "k3", "k4"]
    k_display = ["k=1", "k=2", "k=3", "k=4"]
    corr_vals = [corrections[k] for k in k_plot]
    rev_vals  = [reversals[k]   for k in k_plot]
    x = np.arange(len(k_plot))
    w = 0.35
    bars1 = ax.bar(x - w/2, corr_vals, w, label=f"Correction rate\n(initially wrong -> correct)", color="#4CAF50", alpha=0.85)
    bars2 = ax.bar(x + w/2, rev_vals,  w, label=f"Reversal rate\n(initially correct -> wrong)",  color="#FF5722", alpha=0.85)
    ax.bar_label(bars1, labels=[f"{v:.2f}" for v in corr_vals], fontsize=8, padding=2)
    ax.bar_label(bars2, labels=[f"{v:.2f}" for v in rev_vals],  fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(k_display)
    ax.set_ylabel("Fraction of examples")
    ax.set_title(f"Correction vs Reversal\n"
                 f"(initially wrong: {len(initially_wrong)}, correct: {len(initially_correct)})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "r5_m1_logit_evolution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {FIGURES_DIR}/r5_m1_logit_evolution.png")

    # ══════════════════════════════════════════════════════════════════════════
    # M2: PER-LAYER CONVERGENCE HEATMAP
    # For the champion config, track step_size at each T-block layer x iteration
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 50)
    print(f"M2: Per-layer Convergence Heatmap ({N_MECH} samples)")
    print(f"  Config: n_e={M2_N_E}, n_t={M2_N_T}, k_max={M2_K_MAX}")
    print("=" * 50)

    m2_alpha = 4.0 / M2_N_T   # energy-conserving
    m2_examples_boolq = load_boolq_examples("validation", N_MECH)
    m2_examples_arc   = load_arc_examples("test", N_MECH)

    # heatmap[layer][iter] = mean step_size across examples
    heatmap_boolq = np.zeros((M2_N_T, M2_K_MAX))  # (layer_in_T, iter)
    heatmap_arc   = np.zeros((M2_N_T, M2_K_MAX))

    def compute_heatmap(examples, heatmap):
        layer_iter_vals = [[[] for _ in range(M2_K_MAX)] for _ in range(M2_N_T)]
        for prefix, conts, _ in tqdm(examples, desc="  M2 heatmap", leave=False):
            full = prefix + conts[0]
            enc = tokenizer(full, return_tensors="pt", add_special_tokens=False)
            ids = enc["input_ids"].to(device)
            attn = enc.get("attention_mask")
            if attn is not None:
                attn = attn.to(device)
            iter_states, layer_states_per_iter = _run_T_block_track(
                model, ids, attn, M2_N_E, M2_N_T, M2_K_MAX, m2_alpha
            )
            h_prev_iter = iter_states[0]   # h_e
            for it_idx in range(M2_K_MAX):
                layer_hs = layer_states_per_iter[it_idx]
                h_in = h_prev_iter  # input to this iteration's T block
                for l_idx, h_out in enumerate(layer_hs):
                    # step_size at layer l in iteration it
                    diff = h_out[0, -1] - h_in[0, -1]  # last token
                    s = diff.norm().item() / (h_in[0, -1].norm().item() + 1e-8)
                    layer_iter_vals[l_idx][it_idx].append(s)
                    h_in = h_out
                # After damping, the "output" h_prev changes
                h_prev_iter = iter_states[it_idx + 1]

        for l in range(M2_N_T):
            for it in range(M2_K_MAX):
                heatmap[l, it] = float(np.mean(layer_iter_vals[l][it])) if layer_iter_vals[l][it] else 0

    print("  Computing BoolQ convergence heatmap ...")
    compute_heatmap(m2_examples_boolq, heatmap_boolq)
    print("  Computing ARC convergence heatmap ...")
    compute_heatmap(m2_examples_arc, heatmap_arc)

    results["m2_convergence_heatmap"] = {
        "config": {"n_e": M2_N_E, "n_t": M2_N_T, "k_max": M2_K_MAX, "alpha": m2_alpha},
        "heatmap_boolq": heatmap_boolq.tolist(),
        "heatmap_arc":   heatmap_arc.tolist(),
    }

    # Plot M2
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"M2: Per-layer Convergence Heatmap  (n_e={M2_N_E}, n_t={M2_N_T}, alpha={m2_alpha:.3f})\n"
        f"x-axis = T-block layer index (0 to n_t-1), y-axis = iteration index",
        fontsize=11, fontweight="bold"
    )

    layer_labels = [f"L{M2_N_E + i}" for i in range(M2_N_T)]
    iter_labels  = [f"iter {i+1}" for i in range(M2_K_MAX)]

    for ax, heatmap, title in [
        (axes[0], heatmap_boolq, "BoolQ"),
        (axes[1], heatmap_arc,   "ARC"),
        (axes[2], (heatmap_boolq + heatmap_arc) / 2, "Average (BoolQ+ARC)"),
    ]:
        im = ax.imshow(heatmap.T, aspect="auto", cmap="YlOrRd", origin="upper")
        plt.colorbar(im, ax=ax, label="Step Size ||Δh_last||/||h_in||")
        ax.set_xticks(range(M2_N_T))
        ax.set_xticklabels([f"{M2_N_E+i}" for i in range(M2_N_T)], fontsize=6, rotation=45)
        ax.set_yticks(range(M2_K_MAX))
        ax.set_yticklabels(iter_labels)
        ax.set_xlabel("T-block Layer Index (absolute)")
        ax.set_ylabel("ETD Iteration")
        ax.set_title(f"{title}\nStep size of last token per layer per iteration")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "r5_m2_convergence_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {FIGURES_DIR}/r5_m2_convergence_heatmap.png")

    # ══════════════════════════════════════════════════════════════════════════
    # M3: TOKEN-SELECTIVE DELTA_NORM
    # Does the T-block affect answer tokens vs context tokens differently?
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 50)
    print(f"M3: Token-selective delta_norm ({N_MECH} samples)")
    print("  Comparing: last token vs first-half tokens vs second-half tokens")
    print("=" * 50)

    m3_n_e, m3_n_t = 8, 14
    m3_alpha = _adaptive_alpha(m3_n_t)
    m3_examples = load_boolq_examples("validation", N_MECH)

    m3_records = {"last": [], "first_half": [], "second_half": [], "all": []}

    for prefix, conts, _ in tqdm(m3_examples, desc="M3 token analysis", leave=True):
        full = prefix + conts[0]
        enc = tokenizer(full, return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(device)

        iter_states, _ = _run_T_block_track(
            model, ids, attn, m3_n_e, m3_n_t, k_max=2, alpha=m3_alpha
        )
        h_e, h_1, h_2 = iter_states[0][0], iter_states[1][0], iter_states[2][0]
        # h_e, h_1, h_2 : (seq, dim)
        seq_len = h_1.shape[0]
        mid = seq_len // 2

        def dn(h_a, h_b, idx_slice):
            diff = h_b[idx_slice] - h_a[idx_slice]
            return (diff.norm(dim=-1) / (h_a[idx_slice].norm(dim=-1) + 1e-8)).mean().item()

        m3_records["last"].append(dn(h_e, h_1, [-1]))
        m3_records["first_half"].append(dn(h_e, h_1, slice(0, mid)))
        m3_records["second_half"].append(dn(h_e, h_1, slice(mid, None)))
        m3_records["all"].append(dn(h_e, h_1, slice(None)))

    m3_means = {k: float(np.mean(v)) for k, v in m3_records.items()}
    m3_stds  = {k: float(np.std(v))  for k, v in m3_records.items()}
    print(f"  delta_norm (k=1 vs h_e): {m3_means}")

    results["m3_token_selective"] = {
        "config": {"n_e": m3_n_e, "n_t": m3_n_t, "alpha": m3_alpha},
        "means": m3_means,
        "stds":  m3_stds,
        "raw":   {k: v[:20] for k, v in m3_records.items()},  # first 20 for inspection
    }

    # Also compute: does delta_norm correlate with accuracy gain?
    # Compare delta_norm of examples where k=2 corrects vs doesn't correct baseline errors
    m3_corr_records = []
    for prefix, conts, label in tqdm(m3_examples, desc="M3 correlation", leave=False):
        plen = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
        # Baseline score
        scores_base = _logit_for_continuation_baseline(model, tokenizer, prefix, conts, device)
        pred_base = max(range(len(scores_base)), key=lambda i: scores_base[i])
        # ETD k=2 score
        scores_k2 = _logit_for_continuation(model, tokenizer, prefix, conts, device,
                                             m3_n_e, m3_n_t, 2, m3_alpha)
        pred_k2 = max(range(len(scores_k2)), key=lambda i: scores_k2[i])

        # delta_norm for last token
        full = prefix + conts[0]
        enc = tokenizer(full, return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(device)
        iter_states, _ = _run_T_block_track(
            model, ids, attn, m3_n_e, m3_n_t, k_max=2, alpha=m3_alpha
        )
        h_e_l, h_1_l = iter_states[0][0, -1], iter_states[1][0, -1]
        dn_last = ((h_1_l - h_e_l).norm() / (h_e_l.norm() + 1e-8)).item()

        m3_corr_records.append({
            "base_correct": pred_base == label,
            "k2_correct":   pred_k2 == label,
            "corrected":    (not (pred_base == label)) and (pred_k2 == label),
            "reversed":     (pred_base == label) and not (pred_k2 == label),
            "dn_last":      dn_last,
        })

    corrected_dn = [r["dn_last"] for r in m3_corr_records if r["corrected"]]
    reversed_dn  = [r["dn_last"] for r in m3_corr_records if r["reversed"]]
    neutral_dn   = [r["dn_last"] for r in m3_corr_records if not r["corrected"] and not r["reversed"]]
    print(f"  delta_norm by outcome:")
    print(f"    Corrected examples (base-wrong, k2-right): n={len(corrected_dn)}, "
          f"mean_dn={np.mean(corrected_dn):.4f}" if corrected_dn else "    Corrected: n=0")
    print(f"    Reversed  (base-right, k2-wrong):          n={len(reversed_dn)}, "
          f"mean_dn={np.mean(reversed_dn):.4f}"  if reversed_dn  else "    Reversed:  n=0")
    print(f"    Neutral   (unchanged):                     n={len(neutral_dn)}, "
          f"mean_dn={np.mean(neutral_dn):.4f}"   if neutral_dn   else "    Neutral:   n=0")

    results["m3_token_selective"]["correlation"] = {
        "corrected": {"n": len(corrected_dn), "mean_dn": float(np.mean(corrected_dn)) if corrected_dn else 0},
        "reversed":  {"n": len(reversed_dn),  "mean_dn": float(np.mean(reversed_dn))  if reversed_dn  else 0},
        "neutral":   {"n": len(neutral_dn),   "mean_dn": float(np.mean(neutral_dn))   if neutral_dn   else 0},
    }

    # Plot M3
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        f"M3: Token-selective delta_norm Analysis  "
        f"(n_e={m3_n_e}, n_t={m3_n_t}, alpha={m3_alpha:.3f}, N={N_MECH} BoolQ)",
        fontsize=11, fontweight="bold"
    )

    ax = axes[0]
    token_groups = ["last", "second_half", "first_half", "all"]
    token_labels = ["Last token\n(answer)", "2nd half\n(context)", "1st half\n(context)", "All tokens"]
    means_plot = [m3_means[g] for g in token_groups]
    stds_plot  = [m3_stds[g]  for g in token_groups]
    colors_m3  = ["#FF5722", "#FF9800", "#9E9E9E", "#2196F3"]
    bars = ax.barh(token_labels, means_plot, xerr=stds_plot, color=colors_m3, alpha=0.85,
                   capsize=4, height=0.5)
    ax.bar_label(bars, labels=[f"{v:.4f}" for v in means_plot], padding=4, fontsize=9)
    ax.set_xlabel("delta_norm (k=1 vs h_e)")
    ax.set_title("delta_norm by Token Position\n(k=1 vs encoder output)")
    ax.grid(True, alpha=0.3, axis="x")
    ax.axvline(0, color="black", lw=0.8)

    ax = axes[1]
    ax.scatter(
        [r["dn_last"] for r in m3_corr_records],
        [int(r["k2_correct"]) for r in m3_corr_records],
        alpha=0.4, s=30,
        c=["#4CAF50" if r["k2_correct"] else "#FF5722" for r in m3_corr_records]
    )
    ax.set_xlabel("delta_norm (last token, k=1 vs h_e)")
    ax.set_ylabel("k=2 correct (1=yes, 0=no)")
    ax.set_title("delta_norm vs Correctness at k=2\n(scatter per example)")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    outcome_labels = ["Corrected\n(base wrong→k2 right)", "Reversed\n(base right→k2 wrong)", "Neutral\n(unchanged)"]
    outcome_means  = [
        np.mean(corrected_dn) if corrected_dn else 0,
        np.mean(reversed_dn)  if reversed_dn  else 0,
        np.mean(neutral_dn)   if neutral_dn   else 0,
    ]
    outcome_counts = [len(corrected_dn), len(reversed_dn), len(neutral_dn)]
    outcome_colors = ["#4CAF50", "#FF5722", "#9E9E9E"]
    bars2 = ax.bar(outcome_labels, outcome_means, color=outcome_colors, alpha=0.85, width=0.5)
    ax.bar_label(bars2, labels=[f"{v:.4f}\n(n={c})" for v, c in zip(outcome_means, outcome_counts)],
                 fontsize=8, padding=3)
    ax.set_ylabel("Mean delta_norm (last token)")
    ax.set_title("delta_norm by Prediction Outcome\n(Does larger delta = more likely correction?)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "r5_m3_token_selective.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {FIGURES_DIR}/r5_m3_token_selective.png")

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    total_time = time.time() - t_global
    results["meta"] = {
        "timestamp":    time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_s": total_time,
        "n_select":     N_SELECT,
        "n_eval":       N_EVAL,
        "n_profile":    N_PROFILE,
        "n_mech":       N_MECH,
        "benchmarks":   bench_names,
        "model":        MODEL_PATH,
    }

    out_path = RESULTS_DIR / "round5_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved: {out_path}")

    print("\n" + "=" * 70)
    print("ROUND-5 EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Total runtime: {total_time/60:.1f} min")
    print(f"\n[A3+] Auto-selected n_t={selected_n_t}, eff_step={eff_step:.2f}")
    for bname in bench_names:
        r2 = a3_validation.get(2, {}).get(bname, {})
        bl = baselines.get(bname, 0)
        print(f"  k=2  {bname:<20}: {r2.get('acc', 0):.4f} (delta={r2.get('delta', 0):+.4f} vs baseline {bl:.4f})")

    print(f"\n[A4+] Energy Conservation (k x eff = {ENERGY_CONST:.0f}):")
    for cfg_name, n_e, n_t in A4_CONFIGS:
        for k_val in A4_K_VALUES:
            hyp = f"k{k_val}_C{ENERGY_CONST:.0f}"
            r = a4_results[cfg_name].get(hyp, {})
            eff = r.get("eff_step", 0)
            line = f"  {cfg_name} k={k_val} eff={eff:.1f}:"
            for task in eval_tasks_a4:
                acc = r.get(task, {}).get("acc", 0)
                line += f"  {task[:6]}={acc:.4f}"
            print(line)

    print(f"\n[LP+] Zero-shot selection signal:")
    for metric in metrics:
        sig = signal_results[metric]
        print(f"  {metric:<18}: bad={sig.get('Bad (5-6)', 0):.4f}  "
              f"good={sig.get('Good (8-10)', 0):.4f}  "
              f"mid={sig.get('Mid (12-14)', 0):.4f}")

    print(f"\n[M1] Logit evolution: acc={[f'{a:.3f}' for a in acc_by_k]}")
    print(f"     Correction rate (initially-wrong→correct): {corrections}")
    print(f"     Reversal rate   (initially-correct→wrong):  {reversals}")

    print(f"\n[M2] Convergence heatmap:")
    print(f"     Saved to r5_m2_convergence_heatmap.png")

    print(f"\n[M3] Token delta_norm: {m3_means}")
    if corrected_dn and reversed_dn:
        print(f"     Corrected examples dn_last={np.mean(corrected_dn):.4f}  "
              f"Reversed dn_last={np.mean(reversed_dn):.4f}")

    print("\nFigures generated:")
    for p in sorted(FIGURES_DIR.glob("r5_*.png")):
        print(f"  {p}")

    print("\nDone!")


if __name__ == "__main__":
    main()
