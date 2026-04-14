"""
R32 Phase 1: 大规模假设验证（N=200/benchmark）
==============================================
基于 Phase 0 的可视化结论，扩大样本量到 N=200/benchmark，
正式检验 H_contract、H_align、H_critical_layer、H_logit_align 四个假设。

核心洞察（Phase 0 发现）：
  - rc_global ≈ 0.65-0.68：T-block 确实是压缩映射（rc < 1），但 benchmark 间差异小
  - theta_global ≈ 0.35-0.42：方向对齐度为正，迭代自洽
  - oracle_gain=0 占 95%+：需要更多样本来捕捉有效差异

Phase 1 新增内容：
  1. 对每个样本同时计算多种变体信号（不只是全局，还有各层统计量）
  2. 对所有 N=200 样本计算 Spearman ρ（含 gain=0 样本，视为三值 label）
  3. 计算 argmax(rc_per_layer) 与 R30 oracle t_start 的 MAE（H_critical_layer）
  4. 绘制 Spearman ρ 热图（信号 × benchmark）
  5. 设计简单阈值门控规则，在验证集上测试准确率

输出：
  results/r32_phase1_data.json
  results/r32_phase1_stats.json
  figures/r32_phase1_spearman_heatmap.png
  figures/r32_phase1_critical_layer_dist.png
  figures/r32_phase1_signal_gain_violin.png
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
import matplotlib.colors as mcolors
from scipy import stats

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

N_PHASE1  = 200   # 样本数/benchmark
N_LAYERS  = 36
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE     = torch.bfloat16
T_START   = 8
T_STOP    = 22
CHAMP_K   = 2

# R30 oracle t_start（用于 H_critical_layer 验证）
ORACLE_TSTART = {"ARC-C": 14, "TruthfulQA": 16, "CSQA": 10, "BoolQ": 8}


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


# ─── 2-Pass Probe（复用 Phase 0 逻辑，返回更丰富统计量）────────────────────────
@torch.no_grad()
def two_pass_probe(model, input_ids: torch.Tensor, attention_mask):
    """
    运行两次 T-block，返回丰富的二阶统计量。
    包含：全局 rc/theta/logit_align，逐层 rc/theta，argmax_rc 等。
    """
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

    # Pass 0：全前向
    h = inputs_embeds.clone()
    h0 = {}
    for l in range(N_LAYERS):
        h = run_layer(l, h)
        h0[l] = h.detach()

    h_in   = h0[T_START - 1]   # T-block 入口（第一次）
    h_out1 = h0[T_STOP - 1]    # T-block 出口（第一次）

    # Pass 1：从 T-block 出口重跑 T-block（ETD 第二次迭代）
    h_2nd = h_out1.clone()
    h1 = {}
    for l in range(T_START, T_STOP):
        h_2nd = run_layer(l, h_2nd)
        h1[l] = h_2nd.detach()

    # 逐层统计
    rc_layers, theta_layers = [], []
    for l in range(T_START, T_STOP):
        h_prev0 = h0[l - 1] if l > T_START else h0[T_START - 1]
        d0 = (h0[l][:, -1, :] - h_prev0[:, -1, :]).float()
        d1 = (h1[l][:, -1, :] - h0[l][:, -1, :]).float()
        n0 = d0.norm().item()
        n1 = d1.norm().item()
        rc_layers.append(n1 / (n0 + 1e-9))
        cos_ = F.cosine_similarity(d0.flatten().unsqueeze(0), d1.flatten().unsqueeze(0)).item()
        theta_layers.append(float(cos_))

    # 全局（T-block 整体）
    d0_tot = (h_out1[:, -1, :] - h_in[:, -1, :]).float()
    d1_tot = (h_2nd[:, -1, :] - h_out1[:, -1, :]).float()
    rc_global = float(d1_tot.norm() / (d0_tot.norm() + 1e-9))
    theta_global = float(F.cosine_similarity(d0_tot.flatten().unsqueeze(0), d1_tot.flatten().unsqueeze(0)).item())

    # Logit 空间对齐
    ln_f   = base.norm
    lm_head = model.lm_head
    def llens(h_vec):
        return lm_head(ln_f(h_vec[:, -1:, :])).float().squeeze()
    ld0 = llens(h_out1) - llens(h_in)
    ld1 = llens(h_2nd) - llens(h_out1)
    logit_align = float(F.cosine_similarity(ld0.unsqueeze(0), ld1.unsqueeze(0)).item())

    # argmax(rc_per_layer)：H_critical_layer
    rc_arr = np.array(rc_layers)
    argmax_rc = int(T_START + np.argmax(rc_arr)) if rc_arr.size > 0 else T_START

    # 额外统计量
    # 最大 rc 层
    max_rc_val = float(np.max(rc_arr)) if rc_arr.size > 0 else float("nan")
    # rc > 1 的层数
    n_expanding = int(np.sum(rc_arr > 1.0))
    # theta 均值（逐层）
    theta_arr = np.array(theta_layers)
    mean_theta_layers = float(np.mean(theta_arr))

    # Hessian 代理量：‖δ - proj_Δ₀(δ)‖ / ‖δ‖（δ 垂直于 Δ₀ 的成分比例）
    dot_  = float(torch.dot(d0_tot.flatten(), d1_tot.flatten()).item())
    proj_ = dot_ / (float(d0_tot.norm().item()) ** 2 + 1e-9)
    d1_orth = d1_tot - proj_ * d0_tot
    hessian_proxy = float(d1_orth.norm() / (d1_tot.norm() + 1e-9))

    return {
        "rc_global":        rc_global,
        "theta_global":     theta_global,
        "logit_align":      logit_align,
        "argmax_rc":        argmax_rc,
        "max_rc":           max_rc_val,
        "n_expanding":      n_expanding,
        "mean_theta_layers":mean_theta_layers,
        "hessian_proxy":    hessian_proxy,
        "rc_per_layer":     rc_layers,
        "theta_per_layer":  theta_layers,
    }


# ─── 样本评估（baseline vs champion）─────────────────────────────────────────
@torch.no_grad()
def eval_sample(tok, model, sample) -> tuple[bool, bool]:
    prompt  = sample["prompt"]
    choices = sample["choices"]
    label   = sample["label"]
    pref_ids   = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    prompt_len = pref_ids.shape[1]

    def score_choices(use_etd: bool):
        scores = []
        for ch in choices:
            full_str = prompt + " " + ch
            enc      = tok(full_str, return_tensors="pt", add_special_tokens=False)
            iids     = enc["input_ids"].to(DEVICE)
            amask    = enc.get("attention_mask")
            amask    = amask.to(DEVICE) if amask is not None else None
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

    return score_choices(False), score_choices(True)


# ─── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    tok, model = load_model()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    all_data = {}

    for bench_name, loader_fn in BENCHMARKS.items():
        t_bench = time.time()
        print(f"\n{'='*60}\nBenchmark: {bench_name} (N={N_PHASE1})")
        samples  = loader_fn(N_PHASE1)
        records  = []
        n_bl_ok = n_etd_ok = 0

        for i, samp in enumerate(samples):
            enc   = tok(samp["prompt"], return_tensors="pt", add_special_tokens=False)
            iids  = enc["input_ids"].to(DEVICE)
            amask = enc.get("attention_mask")
            amask = amask.to(DEVICE) if amask is not None else None

            probe = two_pass_probe(model, iids, amask)
            bl_ok, etd_ok = eval_sample(tok, model, samp)
            oracle_gain = int(etd_ok) - int(bl_ok)

            n_bl_ok  += int(bl_ok)
            n_etd_ok += int(etd_ok)

            rec = {"oracle_gain": oracle_gain}
            rec.update({k: probe[k] for k in [
                "rc_global", "theta_global", "logit_align",
                "argmax_rc", "max_rc", "n_expanding",
                "mean_theta_layers", "hessian_proxy",
            ]})
            records.append(rec)

            if (i + 1) % 20 == 0:
                elapsed = time.time() - t_bench
                n_pos = sum(r["oracle_gain"] > 0 for r in records)
                n_neg = sum(r["oracle_gain"] < 0 for r in records)
                print(f"  [{i+1:3d}/{N_PHASE1}]  bl={n_bl_ok/(i+1):.3f} etd={n_etd_ok/(i+1):.3f}"
                      f"  +{n_pos}/-{n_neg}  elapsed={elapsed:.0f}s")

        all_data[bench_name] = records
        print(f"  Done: {len(records)} samples in {time.time()-t_bench:.1f}s")

    # ── 保存原始数据 ──────────────────────────────────────────────────────────
    raw_path = os.path.join(RESULTS_DIR, "r32_phase1_data.json")
    with open(raw_path, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\nSaved raw data → {raw_path}")

    # ── Spearman 相关分析 ──────────────────────────────────────────────────────
    SIGNALS = ["rc_global", "theta_global", "logit_align", "max_rc",
               "n_expanding", "mean_theta_layers", "hessian_proxy"]
    rho_matrix = {}  # signal -> {bench: rho}
    p_matrix   = {}

    print("\n" + "="*60)
    print("Spearman ρ 分析（信号 vs oracle_gain）")
    print(f"{'Signal':<22}  " + "  ".join(f"{b:>12}" for b in BENCHMARKS))
    print("-" * 80)

    summary = {}
    for sig in SIGNALS:
        rho_row = {}
        p_row   = {}
        for bench_name, records in all_data.items():
            sig_vals  = [r[sig] for r in records]
            gain_vals = [r["oracle_gain"] for r in records]
            # 过滤 nan
            pairs = [(s, g) for s, g in zip(sig_vals, gain_vals) if np.isfinite(s)]
            if len(pairs) < 5 or len(set(g for _, g in pairs)) < 2:
                rho_row[bench_name] = float("nan")
                p_row[bench_name]   = float("nan")
            else:
                sv, gv = zip(*pairs)
                rho, p = stats.spearmanr(sv, gv)
                rho_row[bench_name] = float(rho)
                p_row[bench_name]   = float(p)
        rho_matrix[sig] = rho_row
        p_matrix[sig]   = p_row

        vals = [f"{rho_row[b]:+.3f}{'*' if p_row[b] < 0.05 else ' '}" for b in BENCHMARKS]
        print(f"  {sig:<20}  " + "  ".join(f"{v:>12}" for v in vals))

    # H_critical_layer：argmax(rc_per_layer) 与 oracle t_start 的 MAE
    print("\n" + "="*60)
    print("H_critical_layer 分析：argmax(rc_per_layer) vs oracle t_start")
    print(f"{'Benchmark':<14}  oracle_ts  argmax_rc_mean  argmax_rc_std  MAE")
    print("-" * 65)

    critical_layer_stats = {}
    for bench_name, records in all_data.items():
        oracle_ts = ORACLE_TSTART.get(bench_name, T_START)
        argmax_rc_vals = [r["argmax_rc"] for r in records]
        mae = float(np.mean(np.abs(np.array(argmax_rc_vals) - oracle_ts)))
        critical_layer_stats[bench_name] = {
            "oracle_t_start": oracle_ts,
            "argmax_rc_mean": float(np.mean(argmax_rc_vals)),
            "argmax_rc_std":  float(np.std(argmax_rc_vals)),
            "mae":            mae,
        }
        print(f"  {bench_name:<12}  {oracle_ts:>9}  {np.mean(argmax_rc_vals):>14.2f}"
              f"  {np.std(argmax_rc_vals):>13.2f}  {mae:.2f}")

    # ── 保存统计结果 ──────────────────────────────────────────────────────────
    stats_out = {
        "spearman_rho": rho_matrix,
        "spearman_p":   p_matrix,
        "critical_layer": critical_layer_stats,
        "per_bench": {}
    }
    for bench_name, records in all_data.items():
        n_pos = sum(r["oracle_gain"] > 0 for r in records)
        n_neg = sum(r["oracle_gain"] < 0 for r in records)
        stats_out["per_bench"][bench_name] = {
            "n": len(records),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_zero": len(records) - n_pos - n_neg,
            "bl_acc": float(sum(
                (r["oracle_gain"] >= 0) for r in records
            ) / len(records)),  # approx
        }

    stats_path = os.path.join(RESULTS_DIR, "r32_phase1_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats_out, f, indent=2)
    print(f"\nSaved stats → {stats_path}")

    # ── 绘图 1：Spearman ρ 热图 ───────────────────────────────────────────────
    bench_list = list(BENCHMARKS.keys())
    rho_arr = np.array([[rho_matrix[sig].get(b, float("nan")) for b in bench_list]
                         for sig in SIGNALS])
    p_arr   = np.array([[p_matrix[sig].get(b, 1.0) for b in bench_list]
                         for sig in SIGNALS])

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rw", ["#1565C0", "white", "#B71C1C"]
    )
    im = ax.imshow(rho_arr, cmap=cmap, vmin=-0.4, vmax=0.4, aspect="auto")
    ax.set_xticks(range(len(bench_list)))
    ax.set_xticklabels(bench_list, fontsize=12)
    ax.set_yticks(range(len(SIGNALS)))
    ax.set_yticklabels(SIGNALS, fontsize=11)
    for i, sig in enumerate(SIGNALS):
        for j, b in enumerate(bench_list):
            rho = rho_arr[i, j]
            p   = p_arr[i, j]
            if np.isfinite(rho):
                star = "**" if p < 0.01 else ("*" if p < 0.05 else "")
                ax.text(j, i, f"{rho:+.2f}{star}", ha="center", va="center",
                        fontsize=10, color="black")
    plt.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_title(
        f"R32 Phase 1: Spearman ρ（二阶信号 vs oracle_gain）N={N_PHASE1}/bench\n"
        f"*p<0.05  **p<0.01  一阶信号历史天花板: |ρ|≤0.14",
        fontsize=11
    )
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "r32_phase1_spearman_heatmap.png"),
                dpi=150, bbox_inches="tight")
    print(f"Saved → {FIGURES_DIR}/r32_phase1_spearman_heatmap.png")

    # ── 绘图 2：argmax(rc_per_layer) 分布 ─────────────────────────────────────
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    colors_bench = {"BoolQ": "#2196F3", "ARC-C": "#F44336",
                    "CSQA": "#4CAF50", "TruthfulQA": "#FF9800"}

    for ax, (bench_name, records) in zip(axes2.flatten(), all_data.items()):
        oracle_ts = ORACLE_TSTART.get(bench_name, T_START)
        argmax_vals = [r["argmax_rc"] for r in records]
        layer_range = list(range(T_START, T_STOP))
        ax.hist(argmax_vals, bins=range(T_START, T_STOP + 1),
                color=colors_bench[bench_name], edgecolor="white", alpha=0.8)
        ax.axvline(oracle_ts, color="red", linewidth=2, linestyle="--",
                   label=f"Oracle t_start={oracle_ts}")
        ax.axvline(np.mean(argmax_vals), color="navy", linewidth=1.5, linestyle="-.",
                   label=f"Signal mean={np.mean(argmax_vals):.1f}")
        mae = float(np.mean(np.abs(np.array(argmax_vals) - oracle_ts)))
        ax.set_title(f"{bench_name}  |  MAE={mae:.1f} 层  (vs R31 MAE=7-8 层)", fontsize=11)
        ax.set_xlabel("argmax(r_c(l))", fontsize=10)
        ax.set_ylabel("样本数", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    fig2.suptitle("H_critical_layer: argmax(r_c) vs oracle t_start (N=200/bench)", fontsize=13)
    fig2.tight_layout()
    fig2.savefig(os.path.join(FIGURES_DIR, "r32_phase1_critical_layer_dist.png"),
                 dpi=150, bbox_inches="tight")
    print(f"Saved → {FIGURES_DIR}/r32_phase1_critical_layer_dist.png")

    # ── 绘图 3：violin/box 图（信号按 oracle_gain 分组）────────────────────────
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 6))
    PLOT_SIGNALS = ["rc_global", "theta_global", "logit_align"]
    SIGNAL_LABELS = ["r_c (收缩率)", "θ (方向对齐度)", "logit 空间对齐"]

    for ax, sig, slabel in zip(axes3, PLOT_SIGNALS, SIGNAL_LABELS):
        gain_groups = {-1: [], 0: [], 1: []}
        for bench_name, records in all_data.items():
            for r in records:
                v = r[sig]
                if np.isfinite(v):
                    gain_groups[r["oracle_gain"]].append(v)

        data_parts  = [gain_groups[-1], gain_groups[0], gain_groups[1]]
        n_parts     = [len(g) for g in data_parts]
        labels_part = [f"ETD害\n(n={n_parts[0]})", f"无影响\n(n={n_parts[1]})",
                       f"ETD益\n(n={n_parts[2]})"]
        colors_part = ["#F44336", "#9E9E9E", "#4CAF50"]

        # 过滤空组
        non_empty = [(d, l, c) for d, l, c in zip(data_parts, labels_part, colors_part) if len(d) > 0]
        if len(non_empty) >= 2:
            vp = ax.violinplot([d for d, _, _ in non_empty], showmedians=True)
            for vc, (_, _, c) in zip(vp["bodies"], non_empty):
                vc.set_facecolor(c)
                vc.set_alpha(0.6)
            for part in ["cbars", "cmins", "cmaxes", "cmedians"]:
                if part in vp:
                    vp[part].set_color("black")
            ax.set_xticks(range(1, len(non_empty) + 1))
            ax.set_xticklabels([l for _, l, _ in non_empty], fontsize=10)

        ax.set_ylabel(slabel, fontsize=11)
        ax.set_title(slabel, fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")

    fig3.suptitle("R32 Phase 1: 二阶信号按 oracle_gain 分组分布（所有 benchmark 合并）",
                  fontsize=13)
    fig3.tight_layout()
    fig3.savefig(os.path.join(FIGURES_DIR, "r32_phase1_signal_gain_violin.png"),
                 dpi=150, bbox_inches="tight")
    print(f"Saved → {FIGURES_DIR}/r32_phase1_signal_gain_violin.png")

    # ── 最终摘要打印 ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("假设验证总结（Phase 1）")
    print("-" * 60)
    for sig in SIGNALS:
        mean_rho = np.nanmean([rho_matrix[sig][b] for b in bench_list])
        max_rho  = np.nanmax([abs(rho_matrix[sig][b]) for b in bench_list])
        n_sig    = sum(
            p_matrix[sig].get(b, 1.0) < 0.05 for b in bench_list
            if np.isfinite(rho_matrix[sig].get(b, float("nan")))
        )
        print(f"  {sig:<22}  mean|ρ|={abs(mean_rho):.3f}  max|ρ|={max_rho:.3f}  "
              f"n_significant={n_sig}/{len(bench_list)}")

    print("\nH_critical_layer MAE（vs R31 benchmark: MAE=7-8 层）：")
    for bench_name, d in critical_layer_stats.items():
        print(f"  {bench_name:<14} MAE={d['mae']:.2f}  "
              f"(argmax_mean={d['argmax_rc_mean']:.1f} vs oracle={d['oracle_t_start']})")

    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
