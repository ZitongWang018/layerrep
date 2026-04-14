"""
R32 Phase 2: Skip Gate 实验（N=500/benchmark）
==============================================
基于 Phase 1 的 Spearman ρ 分析结果，设计基于二阶信号的 Skip Gate 规则，
并在 N=500/benchmark 上验证 macro accuracy 是否超过 Champion。

Skip Gate 逻辑：
  对每个样本运行轻量 2-Pass Probe，提取信号 S
  if S 满足"ETD 有害"条件 → 跳过 ETD，使用 Baseline
  else → 使用 Champion ETD (t_start=8, t_stop=22, k=2)

本脚本会在 Phase 1 统计结果基础上自动选择最有希望的信号设计门控，
同时测试一系列预设阈值，报告最优配置。

设计的 Skip Gate 变体（基于 Phase 1 假设检验）：
  SG1_rc_high:     r_c > threshold → skip（高收缩率 = T-block 扩张 = 有害）
  SG2_theta_neg:   θ < threshold → skip（方向反转 = 迭代振荡 = 有害）
  SG3_la_neg:      logit_align < threshold → skip（Logit 方向不一致 = 有害）
  SG4_combined:    rc > thr_rc AND theta < thr_th → skip
  SG5_hessian:     hessian_proxy > threshold → skip（高二阶效应 = 难预测 = 有害）

输出：
  results/r32_phase2_results.json
  figures/r32_phase2_accuracy_comparison.png
  figures/r32_phase2_skip_rates.png
"""

from __future__ import annotations

import sys, os, json, time, warnings
sys.path.insert(0, "/root/autodl-tmp/loop_layer")
sys.path.insert(0, "/root/autodl-tmp/loop_layer/ETD")
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from etd_forward import (
    etd_forward_logits,
    baseline_forward_logits,
    loglikelihood_continuation,
)
from transformers.masking_utils import create_causal_mask

# ─── 配置 ──────────────────────────────────────────────────────────────────────
MODEL_PATH  = "/root/autodl-tmp/model_qwen"
RESULTS_DIR = "/root/autodl-tmp/loop_layer/experiments/results"
FIGURES_DIR = "/root/autodl-tmp/loop_layer/experiments/figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

N_PHASE2  = 500
N_LAYERS  = 36
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE     = torch.bfloat16
T_START   = 8
T_STOP    = 22
CHAMP_K   = 2


# ─── 模型加载 ──────────────────────────────────────────────────────────────────
def load_model():
    print("Loading model …")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, device_map="auto", trust_remote_code=True
    )
    model.eval()
    return tok, model


# ─── 数据集加载 ────────────────────────────────────────────────────────────────
def load_boolq(n):
    ds = load_dataset("aps/super_glue", "boolq")["validation"]
    samples = []
    for x in ds:
        lab = int(x["label"])
        if lab < 0:
            continue
        prompt = f"{x['passage']}\nQuestion: {x['question']}?\nAnswer:"
        samples.append({"prompt": prompt, "choices": ["no", "yes"], "label": lab})
        if len(samples) >= n:
            break
    return samples

def load_arc_c(n):
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")["test"]
    samples = []
    for x in ds:
        labels_map = {k: i for i, k in enumerate(x["choices"]["label"])}
        choices    = x["choices"]["text"]
        label      = labels_map.get(x["answerKey"], 0)
        prompt     = f"Question: {x['question']}\nAnswer:"
        samples.append({"prompt": prompt, "choices": choices, "label": label})
        if len(samples) >= n:
            break
    return samples

def load_csqa(n):
    ds = load_dataset("tau/commonsense_qa")["validation"]
    samples = []
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    for x in ds:
        choices = x["choices"]["text"]
        label   = label_map.get(x["answerKey"], 0)
        prompt  = f"Question: {x['question']}\nAnswer:"
        samples.append({"prompt": prompt, "choices": choices, "label": label})
        if len(samples) >= n:
            break
    return samples

def load_truthfulqa(n):
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice")["validation"]
    samples = []
    for x in ds:
        choices = x["mc1_targets"]["choices"]
        labels  = x["mc1_targets"]["labels"]
        label   = int(np.argmax(labels))
        prompt  = f"Question: {x['question']}\nAnswer:"
        samples.append({"prompt": prompt, "choices": choices, "label": label})
        if len(samples) >= n:
            break
    return samples

BENCHMARKS = {
    "BoolQ":      load_boolq,
    "ARC-C":      load_arc_c,
    "CSQA":       load_csqa,
    "TruthfulQA": load_truthfulqa,
}


# ─── 轻量 2-Pass Probe（仅计算全局信号，不保存逐层数据）────────────────────────
@torch.no_grad()
def lite_probe(model, input_ids: torch.Tensor, attention_mask):
    """轻量版 2-Pass Probe：只计算全局 rc, theta, logit_align, hessian_proxy"""
    base = model.model
    cfg  = model.config
    device = input_ids.device
    batch, seq_len = input_ids.shape

    inputs_embeds = base.embed_tokens(input_ids)
    position_ids  = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)

    mask_kwargs = {
        "config": cfg, "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask, "past_key_values": None,
        "position_ids": position_ids,
    }
    causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
    if getattr(base, "has_sliding_layers", False):
        from transformers.masking_utils import create_sliding_window_causal_mask
        causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    position_embeddings = base.rotary_emb(inputs_embeds, position_ids)

    def run_layer(layer_idx, hs):
        layer_type = cfg.layer_types[layer_idx]
        out = base.layers[layer_idx](
            hs, attention_mask=causal_mask_mapping[layer_type],
            position_ids=position_ids, past_key_values=None,
            use_cache=False, position_embeddings=position_embeddings,
        )
        return out[0] if isinstance(out, tuple) else out

    # Pass 0（保留必要层）
    h = inputs_embeds.clone()
    h_tstart_minus1 = None
    h_tstop_minus1  = None

    for l in range(N_LAYERS):
        h = run_layer(l, h)
        if l == T_START - 1:
            h_tstart_minus1 = h.detach()
        if l == T_STOP - 1:
            h_tstop_minus1 = h.detach()

    h_in   = h_tstart_minus1
    h_out1 = h_tstop_minus1

    # Pass 1（仅跑 T-block）
    h_2nd = h_out1.clone()
    for l in range(T_START, T_STOP):
        h_2nd = run_layer(l, h_2nd)

    # 全局信号
    d0 = (h_out1[:, -1, :] - h_in[:, -1, :]).float()
    d1 = (h_2nd[:, -1, :] - h_out1[:, -1, :]).float()
    rc_global    = float(d1.norm() / (d0.norm() + 1e-9))
    theta_global = float(F.cosine_similarity(d0.flatten().unsqueeze(0), d1.flatten().unsqueeze(0)).item())

    ln_f   = base.norm
    lm_head = model.lm_head
    def llens(hv):
        return lm_head(ln_f(hv[:, -1:, :])).float().squeeze()
    ld0 = llens(h_out1) - llens(h_in)
    ld1 = llens(h_2nd) - llens(h_out1)
    logit_align = float(F.cosine_similarity(ld0.unsqueeze(0), ld1.unsqueeze(0)).item())

    # Hessian 代理量
    dot_  = float(torch.dot(d0.flatten(), d1.flatten()).item())
    proj_ = dot_ / (float(d0.norm().item()) ** 2 + 1e-9)
    d1_orth = d1 - proj_ * d0
    hessian_proxy = float(d1_orth.norm() / (d1.norm() + 1e-9))

    return {
        "rc_global":     rc_global,
        "theta_global":  theta_global,
        "logit_align":   logit_align,
        "hessian_proxy": hessian_proxy,
    }


# ─── 样本评估（baseline + champion）─────────────────────────────────────────
@torch.no_grad()
def eval_sample_both(tok, model, sample) -> tuple[bool, bool]:
    prompt  = sample["prompt"]
    choices = sample["choices"]
    label   = sample["label"]
    pref_ids   = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    prompt_len = pref_ids.shape[1]

    def score(use_etd: bool):
        scores = []
        for ch in choices:
            full_str = prompt + " " + ch
            enc   = tok(full_str, return_tensors="pt", add_special_tokens=False)
            iids  = enc["input_ids"].to(DEVICE)
            amask = enc.get("attention_mask")
            amask = amask.to(DEVICE) if amask is not None else None
            if use_etd:
                n_e, n_t = T_START, T_STOP - T_START
                logits = etd_forward_logits(
                    model, iids, amask, n_e=n_e, n_t=n_t, k=CHAMP_K,
                    alpha=min(1.0, 6.0 / max(n_t, 1))
                )
            else:
                logits = baseline_forward_logits(model, iids, amask)
            scores.append(loglikelihood_continuation(logits, iids, prompt_len))
        return int(np.argmax(scores)) == label

    return score(False), score(True)


# ─── Skip Gate 决策函数 ────────────────────────────────────────────────────────
def apply_skip_gate(probe_signals: dict, strategy: str, params: dict) -> bool:
    """
    返回 True = 跳过 ETD（使用 baseline），False = 使用 Champion ETD
    """
    rc    = probe_signals["rc_global"]
    theta = probe_signals["theta_global"]
    la    = probe_signals["logit_align"]
    hp    = probe_signals["hessian_proxy"]

    if strategy == "champion":
        return False  # 永远使用 ETD
    elif strategy == "baseline":
        return True   # 永远跳过
    elif strategy == "SG1_rc_high":
        return rc > params["thr_rc"]
    elif strategy == "SG2_theta_neg":
        return theta < params["thr_theta"]
    elif strategy == "SG3_la_neg":
        return la < params["thr_la"]
    elif strategy == "SG4_combined":
        return (rc > params["thr_rc"]) and (theta < params["thr_theta"])
    elif strategy == "SG5_hessian":
        return hp > params["thr_hp"]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ─── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    # 读取 Phase 1 结果来决定阈值设定
    phase1_path = os.path.join(RESULTS_DIR, "r32_phase1_stats.json")
    if os.path.exists(phase1_path):
        with open(phase1_path) as f:
            phase1_stats = json.load(f)
        print(f"Loaded Phase 1 stats from {phase1_path}")
    else:
        phase1_stats = None
        print("Phase 1 stats not found, using default thresholds")

    tok, model = load_model()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    # 测试的策略集合
    strategies = {
        "champion":    {},
        "baseline":    {},
        "SG1_rc_high_0.72": {"thr_rc": 0.72},
        "SG1_rc_high_0.70": {"thr_rc": 0.70},
        "SG1_rc_high_0.68": {"thr_rc": 0.68},
        "SG2_theta_neg_0.30": {"thr_theta": 0.30},
        "SG2_theta_neg_0.35": {"thr_theta": 0.35},
        "SG2_theta_neg_0.40": {"thr_theta": 0.40},
        "SG3_la_neg_0.1":  {"thr_la": 0.1},
        "SG3_la_neg_0.2":  {"thr_la": 0.2},
        "SG4_combined_rc70_th35": {"thr_rc": 0.70, "thr_theta": 0.35},
        "SG5_hessian_0.9": {"thr_hp": 0.9},
        "SG5_hessian_0.95": {"thr_hp": 0.95},
    }

    # 分解策略名到函数名
    def get_strategy_fn(sname):
        if sname.startswith("SG1"):
            return "SG1_rc_high"
        elif sname.startswith("SG2"):
            return "SG2_theta_neg"
        elif sname.startswith("SG3"):
            return "SG3_la_neg"
        elif sname.startswith("SG4"):
            return "SG4_combined"
        elif sname.startswith("SG5"):
            return "SG5_hessian"
        return sname

    all_results = {}  # bench -> {strategy -> {"acc": float, "skip_rate": float}}

    for bench_name, loader_fn in BENCHMARKS.items():
        t_bench = time.time()
        print(f"\n{'='*60}\nBenchmark: {bench_name} (N={N_PHASE2})")
        samples = loader_fn(N_PHASE2)

        # 初始化计数器
        strat_correct = {sn: 0 for sn in strategies}
        strat_skipped = {sn: 0 for sn in strategies}
        n_bl_correct = 0
        n_etd_correct = 0

        for i, samp in enumerate(samples):
            enc   = tok(samp["prompt"], return_tensors="pt", add_special_tokens=False)
            iids  = enc["input_ids"].to(DEVICE)
            amask = enc.get("attention_mask")
            amask = amask.to(DEVICE) if amask is not None else None

            probe = lite_probe(model, iids, amask)
            bl_ok, etd_ok = eval_sample_both(tok, model, samp)
            n_bl_correct  += int(bl_ok)
            n_etd_correct += int(etd_ok)

            for sname, sparams in strategies.items():
                sfn = get_strategy_fn(sname)
                skip = apply_skip_gate(probe, sfn, sparams)
                if skip:
                    strat_skipped[sname] += 1
                    strat_correct[sname] += int(bl_ok)
                else:
                    strat_correct[sname] += int(etd_ok)

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t_bench
                print(f"  [{i+1:3d}/{N_PHASE2}]  bl={n_bl_correct/(i+1):.3f}  "
                      f"etd={n_etd_correct/(i+1):.3f}  elapsed={elapsed:.0f}s")

        n = len(samples)
        bench_res = {}
        for sname in strategies:
            bench_res[sname] = {
                "acc":       strat_correct[sname] / n,
                "skip_rate": strat_skipped[sname] / n,
            }

        bench_res["_baseline"] = {"acc": n_bl_correct / n, "skip_rate": 1.0}
        bench_res["_champion"] = {"acc": n_etd_correct / n, "skip_rate": 0.0}
        all_results[bench_name] = bench_res

        print(f"\n  === {bench_name} Results ===")
        print(f"  Baseline: {n_bl_correct/n:.4f}  Champion: {n_etd_correct/n:.4f}")
        for sname in strategies:
            if sname in ("champion", "baseline"):
                continue
            acc    = bench_res[sname]["acc"]
            srate  = bench_res[sname]["skip_rate"]
            delta  = acc - n_etd_correct / n
            print(f"  {sname:<35} acc={acc:.4f} ({delta:+.4f} vs champ)  skip={srate:.2f}")

    # ── 计算宏平均 ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Macro Average（4 benchmark）")
    print("-" * 80)

    macro = {}
    for sname in list(strategies.keys()) + ["_baseline", "_champion"]:
        accs = [all_results[b].get(sname, {}).get("acc", float("nan")) for b in BENCHMARKS]
        skip_rates = [all_results[b].get(sname, {}).get("skip_rate", 0) for b in BENCHMARKS]
        macro[sname] = {
            "macro_acc": float(np.nanmean(accs)),
            "per_bench": {b: a for b, a in zip(BENCHMARKS, accs)},
            "mean_skip_rate": float(np.nanmean(skip_rates)),
        }

    champ_macro = macro["_champion"]["macro_acc"]
    bl_macro    = macro["_baseline"]["macro_acc"]

    print(f"{'Strategy':<38}  macro_acc  vs_champ  skip_rate")
    print("-" * 80)
    print(f"{'Baseline':<38}  {bl_macro:.4f}   {bl_macro-champ_macro:+.4f}   1.00")
    print(f"{'Champion':<38}  {champ_macro:.4f}   {0.00:+.4f}   0.00")
    for sname in strategies:
        if sname in ("champion", "baseline"):
            continue
        m = macro[sname]
        delta = m["macro_acc"] - champ_macro
        print(f"  {sname:<36}  {m['macro_acc']:.4f}   {delta:+.4f}   {m['mean_skip_rate']:.2f}")

    # ── 保存结果 ─────────────────────────────────────────────────────────────
    save_out = {
        "per_bench": all_results,
        "macro":     macro,
        "config":    {"N": N_PHASE2, "T_START": T_START, "T_STOP": T_STOP, "K": CHAMP_K},
    }
    result_path = os.path.join(RESULTS_DIR, "r32_phase2_results.json")
    with open(result_path, "w") as f:
        json.dump(save_out, f, indent=2)
    print(f"\nSaved → {result_path}")

    # ── 绘图 1：宏平均柱状图 ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7))
    plot_names = ["_baseline", "_champion"] + [sn for sn in strategies if sn not in ("champion", "baseline")]
    plot_labels = ["Baseline", "Champion"] + [sn[3:] if sn.startswith("SG") else sn for sn in plot_names[2:]]
    plot_accs   = [macro[sn]["macro_acc"] for sn in plot_names]
    plot_colors = []
    for sn in plot_names:
        if sn == "_baseline":
            plot_colors.append("#9E9E9E")
        elif sn == "_champion":
            plot_colors.append("#2196F3")
        elif macro[sn]["macro_acc"] > champ_macro:
            plot_colors.append("#4CAF50")
        else:
            plot_colors.append("#FF7043")

    bars = ax.bar(range(len(plot_names)), plot_accs, color=plot_colors, edgecolor="white", linewidth=0.5)
    ax.axhline(champ_macro, color="#2196F3", linestyle="--", linewidth=1.5, label=f"Champion={champ_macro:.4f}")
    ax.axhline(bl_macro,    color="#9E9E9E", linestyle="--", linewidth=1.2, label=f"Baseline={bl_macro:.4f}")
    for bar, acc in zip(bars, plot_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.001, f"{acc:.3f}",
                ha="center", va="bottom", fontsize=7, rotation=45)
    ax.set_xticks(range(len(plot_names)))
    ax.set_xticklabels(plot_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Macro Accuracy (4 bench)", fontsize=12)
    ax.set_title(f"R32 Phase 2: Skip Gate 策略对比（N={N_PHASE2}/bench）\n"
                 f"绿色=超过 Champion，红色=不如 Champion", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "r32_phase2_accuracy_comparison.png"),
                dpi=150, bbox_inches="tight")
    print(f"Saved → {FIGURES_DIR}/r32_phase2_accuracy_comparison.png")

    # ── 绘图 2：per-benchmark 分组柱状图 ─────────────────────────────────────
    bench_list = list(BENCHMARKS.keys())
    # 只显示最优几个策略 + baseline + champion
    top_strategies = sorted(
        [sn for sn in strategies if sn not in ("champion", "baseline")],
        key=lambda sn: macro[sn]["macro_acc"],
        reverse=True
    )[:5]

    show_strategies = ["_baseline", "_champion"] + top_strategies
    show_labels     = ["Baseline", "Champion"] + [sn[3:] for sn in top_strategies]
    x = np.arange(len(bench_list))
    width = 0.15

    fig2, ax2 = plt.subplots(figsize=(14, 7))
    for j, (sn, sl) in enumerate(zip(show_strategies, show_labels)):
        accs_bench = [all_results[b].get(sn, {}).get("acc", float("nan")) for b in bench_list]
        offset = (j - len(show_strategies) / 2) * width + width / 2
        bars2 = ax2.bar(x + offset, accs_bench, width, label=sl,
                        edgecolor="white", linewidth=0.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels(bench_list, fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title(f"R32 Phase 2: Per-Benchmark 分组对比（N={N_PHASE2}/bench）", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")
    fig2.tight_layout()
    fig2.savefig(os.path.join(FIGURES_DIR, "r32_phase2_per_benchmark.png"),
                 dpi=150, bbox_inches="tight")
    print(f"Saved → {FIGURES_DIR}/r32_phase2_per_benchmark.png")

    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
