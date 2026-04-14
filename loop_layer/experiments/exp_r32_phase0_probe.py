"""
R32 Phase 0: 二次前向探针（2-Pass Probe）可视化诊断
=====================================================
理论背景（Taylor-Hessian）：
  ETD 的净扰动 δ_ETD = F(h_1) - F(h_0) = J_F(h_0)·Δ_0 + O(‖Δ_0‖²)
  包含雅可比矩阵信息，而一阶信号只测量 ‖Δ_0‖ 的标量属性。

本脚本：
  1. 对 N=20 样本/benchmark，运行 2-Pass Probe：
     Pass 0（Baseline）：h_0 → h_1 = h_0 + F(h_0)，记录 Δ_0(l)
     Pass 1（ETD 第一步）：h_1 → h_2 = h_1 + F(h_1)，记录 δ(l) = F(h_1) - F(h_0)
  2. 逐层计算：
     - 收缩率 r_c(l) = ‖δ(l)‖ / ‖Δ_0(l)‖
     - 方向对齐度 θ(l) = cos(Δ_0(l), δ(l))
  3. 绘制剖面图（按 benchmark 分组）
  4. 绘制 oracle_gain 分组的分布图（检验假设可区分性）

输出：
  figures/r32_phase0_rc_profile.png
  figures/r32_phase0_theta_profile.png
  figures/r32_phase0_oracle_gain_scatter.png
  results/r32_phase0_stats.json
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
from scipy import stats

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from etd_forward import (
    etd_forward_logits,
    baseline_forward_logits,
    loglikelihood_continuation,
    _prepare_position_ids,
)
from transformers.masking_utils import create_causal_mask

# ─── 配置 ──────────────────────────────────────────────────────────────────────
MODEL_PATH  = "/root/autodl-tmp/model_qwen"
RESULTS_DIR = "/root/autodl-tmp/loop_layer/experiments/results"
FIGURES_DIR = "/root/autodl-tmp/loop_layer/experiments/figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

N_PHASE0    = 20         # 样本数/benchmark（可视化阶段）
N_LAYERS    = 36
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE       = torch.bfloat16

# Champion 配置
T_START = 8
T_STOP  = 22
CHAMP_K = 2


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
    "BoolQ":       load_boolq,
    "ARC-C":       load_arc_c,
    "CSQA":        load_csqa,
    "TruthfulQA":  load_truthfulqa,
}


# ─── 2-Pass Probe 核心函数 ─────────────────────────────────────────────────────
@torch.no_grad()
def two_pass_probe(model, input_ids: torch.Tensor, attention_mask: torch.Tensor | None):
    """
    运行两次 T-block，提取逐层二阶信息。

    Pass 0（Baseline）：
      运行全部 36 层，收集每层隐层状态 h_0[l]
      T-block 残差 Δ_0[l] = h_0[l] - h_0[l-1]（层 t_start..t_stop 内的残差变化）

    Pass 1（ETD 第一步）：
      从 h_0[t_start-1]（T-block 入口状态）重跑 T-block，但输入为 h_1 = h_0[t_stop]（T-block 出口）
      即：模拟 ETD 第二次迭代的 T-block，收集 h_1[l]
      δ[l] = h_1[l] - h_0[l]（第二次 T-block 各层的输出变化）

    返回：
      delta0[l]: T-block 第一次的残差向量（最后 token）
      delta1[l]: T-block 第二次与第一次的差（二阶扰动近似）
      rc[l]:     收缩率 ‖delta1[l]‖ / ‖delta0[l]‖
      theta[l]:  方向对齐度 cos(delta0[l], delta1[l])
    """
    base = model.model
    cfg  = model.config
    device = input_ids.device

    batch, seq_len = input_ids.shape
    inputs_embeds = base.embed_tokens(input_ids)
    position_ids  = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)

    mask_kwargs = {
        "config": cfg,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
    if getattr(base, "has_sliding_layers", False):
        from transformers.masking_utils import create_sliding_window_causal_mask
        causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    position_embeddings = base.rotary_emb(inputs_embeds, position_ids)

    def run_layer(layer_idx: int, hs: torch.Tensor) -> torch.Tensor:
        layer_type = cfg.layer_types[layer_idx]
        attn_mask  = causal_mask_mapping[layer_type]
        out = base.layers[layer_idx](
            hs,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=position_embeddings,
        )
        return out[0] if isinstance(out, tuple) else out

    # ── Pass 0：完整基线前向，记录每层隐状态 ──────────────────────────────────
    h = inputs_embeds.clone()
    h0_layers = {}  # h0_layers[l] = 第 l 层输出（全序列）

    for l in range(N_LAYERS):
        h = run_layer(l, h)
        h0_layers[l] = h.detach()

    # ── Pass 1：从 T-block 入口状态重新跑 T-block ──────────────────────────────
    # 第一次 T-block 的"入口"是 h0_layers[T_START - 1]（即层 T_START 的输入）
    # 第一次 T-block 的"出口"是 h0_layers[T_STOP - 1]
    # ETD 第二次 T-block 的输入 = 第一次 T-block 出口 = h0_layers[T_STOP - 1]
    # 但注意 ETD 带阻尼：h_etd = alpha * T(h_prev) + (1-alpha)*h_prev
    # 这里我们只分析无阻尼的"纯二阶信号"，α=1.0
    # 即：h_in = h0_layers[T_STOP - 1]，再跑一次 T-block

    h_in = h0_layers[T_START - 1].clone()  # T-block 的初始输入（第一次）
    h_out_1st = h0_layers[T_STOP - 1].clone()  # T-block 第一次输出

    # Δ_0[l] = h0_layers[l][:,-1,:] - h0_layers[l-1][:,-1,:]  for l in T_START..T_STOP-1
    # 但更准确地定义 T-block 整体残差：
    #   Δ_0_total = h_out_1st - h_in  （T-block 整体变化量）

    # 第二次 T-block：输入 = h_out_1st（ETD 的第二次迭代输入）
    h_2nd = h_out_1st.clone()
    h1_layers = {}  # T-block 内部各层的第二次输出

    for l in range(T_START, T_STOP):
        h_2nd = run_layer(l, h_2nd)
        h1_layers[l] = h_2nd.detach()

    # ── 提取最后 token 的向量，计算二阶统计量 ──────────────────────────────────
    rc_list    = []   # 收缩率
    theta_list = []   # 方向对齐度
    delta0_norms = [] # ‖Δ_0(l)‖
    delta1_norms = [] # ‖δ(l)‖

    for l in range(T_START, T_STOP):
        if l == T_START:
            h_prev_0 = h0_layers[T_START - 1]
        else:
            h_prev_0 = h0_layers[l - 1]

        d0 = (h0_layers[l][:, -1, :] - h_prev_0[:, -1, :]).float()  # Δ_0(l)
        d1 = (h1_layers[l][:, -1, :] - h0_layers[l][:, -1, :]).float()  # δ(l)

        norm_d0 = d0.norm().item()
        norm_d1 = d1.norm().item()

        if norm_d0 < 1e-9:
            rc = float("nan")
            cos = float("nan")
        else:
            rc = norm_d1 / norm_d0
            cos = float(F.cosine_similarity(d0.flatten().unsqueeze(0),
                                             d1.flatten().unsqueeze(0)).item())

        rc_list.append(rc)
        theta_list.append(cos)
        delta0_norms.append(norm_d0)
        delta1_norms.append(norm_d1)

    # 全局 T-block 统计（整体入口/出口）
    delta0_total = (h_out_1st[:, -1, :] - h_in[:, -1, :]).float()
    delta1_total = (h_2nd[:, -1, :] - h_out_1st[:, -1, :]).float()

    rc_global = float(delta1_total.norm() / (delta0_total.norm() + 1e-9))
    theta_global = float(F.cosine_similarity(
        delta0_total.flatten().unsqueeze(0),
        delta1_total.flatten().unsqueeze(0)
    ).item())

    # Logit 空间对齐
    ln_f   = base.norm
    lm_head = model.lm_head
    ld0 = lm_head(ln_f(h_out_1st[:, -1:, :])).float().squeeze() - \
          lm_head(ln_f(h_in[:, -1:, :])).float().squeeze()
    ld1 = lm_head(ln_f(h_2nd[:, -1:, :])).float().squeeze() - \
          lm_head(ln_f(h_out_1st[:, -1:, :])).float().squeeze()
    logit_align = float(F.cosine_similarity(
        ld0.flatten().unsqueeze(0),
        ld1.flatten().unsqueeze(0)
    ).item())

    return {
        "rc_per_layer":    rc_list,       # len = T_STOP - T_START
        "theta_per_layer": theta_list,
        "d0_norm_per_layer": delta0_norms,
        "d1_norm_per_layer": delta1_norms,
        "rc_global":       rc_global,
        "theta_global":    theta_global,
        "logit_align":     logit_align,
        "layer_range":     list(range(T_START, T_STOP)),
    }


# ─── 评估单样本（baseline vs champion）──────────────────────────────────────────
@torch.no_grad()
def eval_sample(tok, model, sample) -> tuple[bool, bool]:
    """返回 (baseline_correct, champion_correct)"""
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
            attn     = enc.get("attention_mask")
            amask    = attn.to(DEVICE) if attn is not None else None

            if use_etd:
                n_e = T_START
                n_t = T_STOP - T_START
                alpha = min(1.0, 6.0 / max(n_t, 1))
                logits = etd_forward_logits(
                    model, iids, amask, n_e=n_e, n_t=n_t, k=CHAMP_K, alpha=alpha
                )
            else:
                logits = baseline_forward_logits(model, iids, amask)

            ll = loglikelihood_continuation(logits, iids, prompt_len)
            scores.append(ll)
        return int(np.argmax(scores)) == label

    bl_ok   = score_choices(use_etd=False)
    etd_ok  = score_choices(use_etd=True)
    return bl_ok, etd_ok


# ─── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    tok, model = load_model()
    print(f"Model loaded in {time.time()-t0:.1f}s, device={DEVICE}")

    all_results = {}  # bench_name -> list of sample records

    for bench_name, loader_fn in BENCHMARKS.items():
        print(f"\n{'='*60}")
        print(f"Benchmark: {bench_name} (N={N_PHASE0})")
        samples = loader_fn(N_PHASE0)
        records = []

        for i, sample in enumerate(samples):
            prompt  = sample["prompt"]
            choices = sample["choices"]

            # 只对第一个 choice 做 2-Pass Probe（取 prefix token）
            # 用 prompt 本身（不加 choice）进行 probe
            enc      = tok(prompt, return_tensors="pt", add_special_tokens=False)
            iids     = enc["input_ids"].to(DEVICE)
            attn     = enc.get("attention_mask")
            amask    = attn.to(DEVICE) if attn is not None else None

            probe_out = two_pass_probe(model, iids, amask)

            bl_ok, etd_ok = eval_sample(tok, model, sample)
            oracle_gain = int(etd_ok) - int(bl_ok)

            rec = {
                "idx":          i,
                "baseline_ok":  bl_ok,
                "etd_ok":       etd_ok,
                "oracle_gain":  oracle_gain,
                "rc_global":    probe_out["rc_global"],
                "theta_global": probe_out["theta_global"],
                "logit_align":  probe_out["logit_align"],
                "rc_per_layer": probe_out["rc_per_layer"],
                "theta_per_layer": probe_out["theta_per_layer"],
            }
            records.append(rec)

            flag = {-1: "↓", 0: "=", 1: "↑"}[oracle_gain]
            print(f"  [{i+1:2d}/{N_PHASE0}] gain={flag}  rc={probe_out['rc_global']:.3f}  "
                  f"θ={probe_out['theta_global']:.3f}  logit_align={probe_out['logit_align']:.3f}")

        all_results[bench_name] = records
        n_pos = sum(r["oracle_gain"] > 0 for r in records)
        n_neg = sum(r["oracle_gain"] < 0 for r in records)
        rcs   = [r["rc_global"] for r in records]
        thetas= [r["theta_global"] for r in records]
        print(f"  oracle_gain: +{n_pos} / ={N_PHASE0-n_pos-n_neg} / -{n_neg}")
        print(f"  rc_global:   mean={np.mean(rcs):.3f}  std={np.std(rcs):.3f}")
        print(f"  theta_global:mean={np.mean(thetas):.3f}  std={np.std(thetas):.3f}")

    # ── 保存原始数据 ───────────────────────────────────────────────────────────
    save_path = os.path.join(RESULTS_DIR, "r32_phase0_data.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved raw data → {save_path}")

    # ── 绘图 1：r_c(l) 逐层剖面 ────────────────────────────────────────────────
    layer_range = list(range(T_START, T_STOP))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {"BoolQ": "#2196F3", "ARC-C": "#F44336", "CSQA": "#4CAF50", "TruthfulQA": "#FF9800"}

    ax = axes[0, 0]
    for bench_name, records in all_results.items():
        rc_matrix = np.array([r["rc_per_layer"] for r in records])
        mean_rc   = np.nanmean(rc_matrix, axis=0)
        std_rc    = np.nanstd(rc_matrix, axis=0)
        ax.plot(layer_range, mean_rc, label=bench_name, color=colors[bench_name], linewidth=2)
        ax.fill_between(layer_range, mean_rc - std_rc, mean_rc + std_rc,
                        color=colors[bench_name], alpha=0.15)
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1, label="r_c=1 (临界)")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Contraction Rate r_c(l) = ‖δ‖/‖Δ₀‖", fontsize=11)
    ax.set_title("Per-Layer Contraction Rate (mean ± std)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── 绘图 2：θ(l) 逐层剖面 ──────────────────────────────────────────────────
    ax = axes[0, 1]
    for bench_name, records in all_results.items():
        th_matrix = np.array([r["theta_per_layer"] for r in records])
        mean_th   = np.nanmean(th_matrix, axis=0)
        std_th    = np.nanstd(th_matrix, axis=0)
        ax.plot(layer_range, mean_th, label=bench_name, color=colors[bench_name], linewidth=2)
        ax.fill_between(layer_range, mean_th - std_th, mean_th + std_th,
                        color=colors[bench_name], alpha=0.15)
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1, label="θ=0 (临界)")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Direction Alignment θ(l) = cos(Δ₀, δ)", fontsize=11)
    ax.set_title("Per-Layer Direction Alignment (mean ± std)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── 绘图 3：r_c_global 按 oracle_gain 分组箱线图 ───────────────────────────
    ax = axes[1, 0]
    gain_groups = {-1: [], 0: [], 1: []}
    for bench_name, records in all_results.items():
        for r in records:
            gain_groups[r["oracle_gain"]].append(r["rc_global"])

    box_data  = [gain_groups[-1], gain_groups[0], gain_groups[1]]
    box_labels= [f"ETD害({len(gain_groups[-1])})", f"无影响({len(gain_groups[0])})", f"ETD益({len(gain_groups[1])})"]
    bplot = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch, color in zip(bplot["boxes"], ["#F44336", "#9E9E9E", "#4CAF50"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("r_c (全局收缩率)", fontsize=12)
    ax.set_title("H_contract: r_c vs Oracle ETD Gain", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    # ── 绘图 4：θ_global 按 oracle_gain 分组箱线图 ─────────────────────────────
    ax = axes[1, 1]
    th_groups = {-1: [], 0: [], 1: []}
    for bench_name, records in all_results.items():
        for r in records:
            th_groups[r["oracle_gain"]].append(r["theta_global"])

    box_data2  = [th_groups[-1], th_groups[0], th_groups[1]]
    bplot2 = ax.boxplot(box_data2, labels=box_labels, patch_artist=True)
    for patch, color in zip(bplot2["boxes"], ["#F44336", "#9E9E9E", "#4CAF50"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("θ (全局方向对齐度)", fontsize=12)
    ax.set_title("H_align: θ vs Oracle ETD Gain", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("R32 Phase 0: 二阶探针信号可视化（N=20/benchmark）", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "r32_phase0_probe_profiles.png"),
                dpi=150, bbox_inches="tight")
    print(f"Saved → {FIGURES_DIR}/r32_phase0_probe_profiles.png")

    # ── 绘图 5：散点图 (rc_global, theta_global) 标注 oracle_gain ──────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
    marker_map = {-1: "v", 0: "o", 1: "^"}
    color_map  = {-1: "#F44336", 0: "#9E9E9E", 1: "#4CAF50"}

    ax = axes2[0]
    for bench_name, records in all_results.items():
        for r in records:
            g = r["oracle_gain"]
            ax.scatter(r["rc_global"], r["theta_global"],
                       c=color_map[g], marker=marker_map[g],
                       s=80, alpha=0.7,
                       label=f"{bench_name} gain={g}" if r == records[0] else "")
    ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
    ax.axvline(1, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel("r_c (收缩率)", fontsize=12)
    ax.set_ylabel("θ (方向对齐度)", fontsize=12)
    ax.set_title("(r_c, θ) 空间 — 颜色=oracle_gain", fontsize=12)

    # 只加简单图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#F44336", label="ETD 有害 (gain=-1)"),
        Patch(facecolor="#9E9E9E", label="无影响 (gain=0)"),
        Patch(facecolor="#4CAF50", label="ETD 有益 (gain=+1)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    ax.grid(True, alpha=0.3)

    # 第二子图：argmax(r_c(l)) 分布（按 benchmark）
    ax = axes2[1]
    for bench_name, records in all_results.items():
        argmax_rc = []
        for r in records:
            rc_arr = np.array(r["rc_per_layer"])
            valid  = np.isfinite(rc_arr)
            if valid.any():
                argmax_rc.append(layer_range[int(np.argmax(rc_arr[valid]))])
        ax.hist(argmax_rc, bins=range(T_START, T_STOP + 1), alpha=0.5,
                label=bench_name, color=colors[bench_name], edgecolor="white")
    ax.set_xlabel("argmax(r_c(l)) — 最大扩张层", fontsize=12)
    ax.set_ylabel("样本数", fontsize=12)
    ax.set_title("H_critical_layer: argmax(r_c) 分布\n(R30 oracle: ARC-C=14, TQA=16, CSQA=10)",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    fig2.suptitle("R32 Phase 0: 二阶信号散点图", fontsize=14)
    fig2.tight_layout()
    fig2.savefig(os.path.join(FIGURES_DIR, "r32_phase0_scatter.png"),
                 dpi=150, bbox_inches="tight")
    print(f"Saved → {FIGURES_DIR}/r32_phase0_scatter.png")

    # ── 计算初步统计 ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("初步相关性分析（N=20/bench，仅供参考）")
    summary = {}
    for bench_name, records in all_results.items():
        rc_vals    = [r["rc_global"] for r in records]
        th_vals    = [r["theta_global"] for r in records]
        la_vals    = [r["logit_align"] for r in records]
        gain_vals  = [r["oracle_gain"] for r in records]

        if len(set(gain_vals)) > 1:
            rho_rc, p_rc = stats.spearmanr(rc_vals, gain_vals)
            rho_th, p_th = stats.spearmanr(th_vals, gain_vals)
            rho_la, p_la = stats.spearmanr(la_vals, gain_vals)
        else:
            rho_rc = p_rc = rho_th = p_th = rho_la = p_la = float("nan")

        summary[bench_name] = {
            "n": len(records),
            "n_pos": sum(r["oracle_gain"] > 0 for r in records),
            "n_neg": sum(r["oracle_gain"] < 0 for r in records),
            "rc_mean": float(np.mean(rc_vals)),
            "theta_mean": float(np.mean(th_vals)),
            "logit_align_mean": float(np.mean(la_vals)),
            "spearman_rc_gain": float(rho_rc), "p_rc": float(p_rc),
            "spearman_th_gain": float(rho_th), "p_th": float(p_th),
            "spearman_la_gain": float(rho_la), "p_la": float(p_la),
        }
        print(f"\n{bench_name}:")
        print(f"  rc_global:    mean={np.mean(rc_vals):.3f}  ρ(gain)={rho_rc:.3f}(p={p_rc:.3f})")
        print(f"  theta_global: mean={np.mean(th_vals):.3f}  ρ(gain)={rho_th:.3f}(p={p_th:.3f})")
        print(f"  logit_align:  mean={np.mean(la_vals):.3f}  ρ(gain)={rho_la:.3f}(p={p_la:.3f})")

    stats_path = os.path.join(RESULTS_DIR, "r32_phase0_stats.json")
    with open(stats_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved stats → {stats_path}")
    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
