"""
R33 Phase 0: FFN 慢权重激活相变 × Attention 快权重谱隙 可视化诊断
================================================================
核心理论（FFN 相变 + KV 幂迭代）：
  - FFN Gini：Ω(x) 集中度，低=多方向弱激活（塑性态），高=少数神经元主导（稳态）
  - Attn Spectral Gap：W_fast(x) 谱隙，高=有清晰吸引子，低=多竞争方向
  - 关键验证：这些信号的样本间 std 是否远超 R32 的 rc_global（std≈0.02）

本脚本（N=20/benchmark）：
  1. 提取 T-block 区间逐层 FFN Gini / Act Entropy / Boundary Frac
  2. 提取逐层 Attn Spectral Gap / Head Consensus
  3. 绘制剖面图（按 benchmark 分组）
  4. 绘制 (FFN_Gini@8, Attn_Gap@8) 2D 散点图，颜色=oracle_gain
  5. 打印 std 对比（vs R32 rc_global std=0.02）

输出：
  figures/r33_phase0_ffn_profile.png / r33_phase0_attn_profile.png（多 bench 叠图）
  figures/r33_by_layer_r30/<bench>_ffn.png / <bench>_attn.png（每 bench 均值±std + R30 最优 t_start/t_stop）
  figures/r33_phase0_2d_signal_space.png
  results/r33_phase0_stats.json
  results/r33_phase0_data_full.json（含 per_layer，便于离线重画）
"""
from __future__ import annotations
import sys, os, json, time, warnings

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

sys.path.insert(0, "/root/autodl-tmp/loop_layer")
sys.path.insert(0, "/root/autodl-tmp/loop_layer/ETD")
warnings.filterwarnings("ignore")

import torch
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

# ── 新增信号函数（直接内联，不依赖 r29 包）─────────────────────────────────────
from r29.signal_funcs import (
    ffn_gini_last_token,
    ffn_activation_entropy_last_token,
    ffn_boundary_frac_last_token,
    attn_spectral_gap_last_token,
    attn_head_consensus_last_token,
    attn_top2_mass_last_token,
)

# ─── 配置 ──────────────────────────────────────────────────────────────────────
MODEL_PATH  = "/root/autodl-tmp/model_qwen"
RESULTS_DIR = "/root/autodl-tmp/loop_layer/experiments/results"
FIGURES_DIR = "/root/autodl-tmp/loop_layer/experiments/figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

N_PHASE0  = 20
N_LAYERS  = 36
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE     = torch.bfloat16
T_START, T_STOP, CHAMP_K = 8, 22, 2

BENCH_COLORS = {
    "BoolQ": "#2196F3", "ARC-C": "#F44336",
    "CSQA":  "#4CAF50", "TruthfulQA": "#FF9800",
    "MMLU-HS": "#9C27B0",
}

# R30 网格最优 T-block（与 r30_top_configs.txt / plot_proposed_signals_by_layer 一致）
# BoolQ 未列入 R30 单任务 Top；此处用文档中的 Champion 宏平均块 [8,22) 作参照。
R30_OPTIMAL = {
    "BoolQ": {"t_start": 8, "t_stop": 22},
    "ARC-C": {"t_start": 14, "t_stop": 20},
    "TruthfulQA": {"t_start": 16, "t_stop": 19},
    "CSQA": {"t_start": 10, "t_stop": 22},
    "MMLU-HS": {"t_start": 10, "t_stop": 18},
}


def add_r30_tblock_marks(ax, bench_name: str, zorder: int = 4) -> None:
    """在横轴为 layer 的图上标注该 benchmark 的 R30 最优 t_start / t_stop（t_stop 为右开区间）。"""
    opt = R30_OPTIMAL.get(bench_name)
    if opt is None:
        return
    t0, t1 = opt["t_start"], opt["t_stop"]
    ax.axvspan(t0, t1, alpha=0.12, color="gold", zorder=0)
    ax.axvline(t0, color="#2ca02c", linestyle="--", linewidth=2.0, zorder=zorder, label=f"R30 t_start={t0}")
    ax.axvline(t1, color="#d62728", linestyle="--", linewidth=2.0, zorder=zorder, label=f"R30 t_stop={t1}")


# ─── 模型加载 ──────────────────────────────────────────────────────────────────
def load_model():
    print("Loading model …")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, device_map="auto",
        attn_implementation="eager",    # ← 必须 eager 才能拿到 attn_weights
        trust_remote_code=True,
    )
    model.eval()
    return tok, model


# ─── 数据集加载 ────────────────────────────────────────────────────────────────
def load_boolq(n):
    ds = load_dataset("aps/super_glue", "boolq")["validation"]
    out = []
    for x in ds:
        if int(x["label"]) < 0: continue
        out.append({"prompt": f"{x['passage']}\nQuestion: {x['question']}?\nAnswer:",
                    "choices": ["no", "yes"], "label": int(x["label"])})
        if len(out) >= n: break
    return out

def load_arc_c(n):
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")["test"]
    out = []
    for x in ds:
        lmap = {k: i for i, k in enumerate(x["choices"]["label"])}
        out.append({"prompt": f"Question: {x['question']}\nAnswer:",
                    "choices": x["choices"]["text"],
                    "label": lmap.get(x["answerKey"], 0)})
        if len(out) >= n: break
    return out

def load_csqa(n):
    ds = load_dataset("tau/commonsense_qa")["validation"]
    lmap = {"A":0,"B":1,"C":2,"D":3,"E":4}
    out = []
    for x in ds:
        out.append({"prompt": f"Question: {x['question']}\nAnswer:",
                    "choices": x["choices"]["text"],
                    "label": lmap.get(x["answerKey"], 0)})
        if len(out) >= n: break
    return out

def load_truthfulqa(n):
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice")["validation"]
    out = []
    for x in ds:
        labels = x["mc1_targets"]["labels"]
        out.append({"prompt": f"Question: {x['question']}\nAnswer:",
                    "choices": x["mc1_targets"]["choices"],
                    "label": int(np.argmax(labels))})
        if len(out) >= n: break
    return out

def load_mmlu_hs(n):
    try:
        ds = load_dataset("cais/mmlu", "high_school_mathematics")["test"]
        out = []
        for x in ds:
            out.append({"prompt": f"Question: {x['question']}\nAnswer:",
                        "choices": x["choices"], "label": int(x["answer"])})
            if len(out) >= n: break
        return out
    except Exception as e:
        print(f"  [WARN] MMLU-HS unavailable offline: {e}")
        return None

BENCHMARK_LOADERS = [
    ("BoolQ",       load_boolq),
    ("ARC-C",       load_arc_c),
    ("CSQA",        load_csqa),
    ("TruthfulQA",  load_truthfulqa),
    ("MMLU-HS",     load_mmlu_hs),
]


# ─── 核心探针：单次前向，采集逐层 FFN + Attn 信号 ────────────────────────────
@torch.no_grad()
def collect_r33_signals(model, input_ids: torch.Tensor, attention_mask):
    """
    单次前向，钩取：
      - mlp.act_fn 输出（SiLU gate）→ ffn_gini / act_entropy / boundary_frac
      - self_attn 输出（eager attn_weights）→ attn_spectral_gap / head_consensus
    全层 0..N_LAYERS-1 采集，便于与 R30 各 bench 最优层对齐作图。
    """
    base   = model.model
    n_layers = N_LAYERS

    gate_acts = {}
    attn_wts  = {}
    hooks     = []

    for li in range(n_layers):
        def make_gate_hook(idx):
            def fn(_m, _inp, out):
                gate_acts[idx] = out.detach()
            return fn
        def make_attn_hook(idx):
            def fn(_m, _inp, out):
                # eager mode: out = (attn_output, attn_weights, ...)
                if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                    attn_wts[idx] = out[1].detach()
            return fn
        hooks.append(base.layers[li].mlp.act_fn.register_forward_hook(make_gate_hook(li)))
        hooks.append(base.layers[li].self_attn.register_forward_hook(make_attn_hook(li)))

    _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    for h in hooks:
        h.remove()

    layer_range = list(range(n_layers))
    signals = {
        "ffn_gini":        [],
        "ffn_act_entropy": [],
        "ffn_boundary_frac": [],
        "attn_spectral_gap": [],
        "attn_head_consensus": [],
        "attn_top2_mass":  [],
    }

    for li in layer_range:
        ga = gate_acts.get(li)
        if ga is not None:
            signals["ffn_gini"].append(ffn_gini_last_token(ga))
            signals["ffn_act_entropy"].append(ffn_activation_entropy_last_token(ga))
            signals["ffn_boundary_frac"].append(ffn_boundary_frac_last_token(ga))
        else:
            for k in ["ffn_gini","ffn_act_entropy","ffn_boundary_frac"]:
                signals[k].append(float("nan"))

        aw = attn_wts.get(li)
        if aw is not None:
            signals["attn_spectral_gap"].append(attn_spectral_gap_last_token(aw))
            signals["attn_head_consensus"].append(attn_head_consensus_last_token(aw))
            signals["attn_top2_mass"].append(attn_top2_mass_last_token(aw))
        else:
            for k in ["attn_spectral_gap","attn_head_consensus","attn_top2_mass"]:
                signals[k].append(float("nan"))

    # 标量摘要（沿用 Champion T-block 入口层 T_START=8，便于与 Phase1 列名一致）
    scalars = {}
    for sig in signals:
        arr = signals[sig]
        scalars[f"{sig}_at8"] = arr[T_START] if len(arr) > T_START else float("nan")
        block = arr[T_START:T_STOP] if len(arr) >= T_STOP else []
        valid_b = [v for v in block if np.isfinite(v)]
        scalars[f"{sig}_mean"] = float(np.mean(valid_b)) if valid_b else float("nan")

    # 联合信号
    g8 = scalars["ffn_gini_at8"]
    s8 = scalars["attn_spectral_gap_at8"]
    scalars["plasticity_score"] = ((1 - g8) * s8) if np.isfinite(g8) and np.isfinite(s8) else float("nan")

    return {"per_layer": signals, "scalars": scalars, "layer_range": layer_range}


# ─── 样本评估 ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_sample(tok, model, sample) -> tuple[bool, bool]:
    prompt, choices, label = sample["prompt"], sample["choices"], sample["label"]
    pref_ids   = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    prompt_len = pref_ids.shape[1]

    def score(use_etd):
        scores = []
        for ch in choices:
            full  = prompt + " " + ch
            enc   = tok(full, return_tensors="pt", add_special_tokens=False)
            iids  = enc["input_ids"].to(DEVICE)
            amask = enc.get("attention_mask")
            amask = amask.to(DEVICE) if amask is not None else None
            if use_etd:
                nt = T_STOP - T_START
                logits = etd_forward_logits(
                    model, iids, amask, n_e=T_START, n_t=nt, k=CHAMP_K,
                    alpha=min(1.0, 6.0/max(nt,1))
                )
            else:
                logits = baseline_forward_logits(model, iids, amask)
            scores.append(loglikelihood_continuation(logits, iids, prompt_len))
        return int(np.argmax(scores)) == label

    return score(False), score(True)


def plot_per_benchmark_r30_overlay(all_data: dict, out_dir: str) -> None:
    """每个 benchmark 单独一张图：均值±std vs 全层，叠加该 bench 的 R30 最优 t_start/t_stop。"""
    os.makedirs(out_dir, exist_ok=True)
    layer_range = list(range(N_LAYERS))

    ffn_specs = [
        ("ffn_gini", "FFN Gini（↓ 塑性）"),
        ("ffn_act_entropy", "FFN 激活熵"),
        ("ffn_boundary_frac", "FFN 边界邻近比例"),
    ]
    attn_specs = [
        ("attn_spectral_gap", "Attn spectral gap (max/2nd)"),
        ("attn_head_consensus", "Head consensus (1−mean JSD)"),
        ("attn_top2_mass", "Top-2 attention mass"),
    ]

    for bench_name, records in all_data.items():
        if not records:
            continue
        c = BENCH_COLORS.get(bench_name, "gray")

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        for ax, (sig, ylab) in zip(axes[0], ffn_specs):
            mat = np.array([r["per_layer"][sig] for r in records])
            mean_ = np.nanmean(mat, axis=0)
            std_ = np.nanstd(mat, axis=0)
            add_r30_tblock_marks(ax, bench_name)
            ax.plot(layer_range, mean_, color=c, linewidth=2.2, label="mean")
            ax.fill_between(layer_range, mean_ - std_, mean_ + std_, color=c, alpha=0.2)
            ax.set_xlabel("Layer")
            ax.set_ylabel(ylab)
            ax.set_title(sig)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.5, N_LAYERS - 0.5)

        for ax, (sig, ylab) in zip(axes[1], attn_specs):
            mat = np.array([r["per_layer"][sig] for r in records])
            mean_ = np.nanmean(mat, axis=0)
            std_ = np.nanstd(mat, axis=0)
            add_r30_tblock_marks(ax, bench_name)
            ax.plot(layer_range, mean_, color=c, linewidth=2.2, label="mean")
            ax.fill_between(layer_range, mean_ - std_, mean_ + std_, color=c, alpha=0.2)
            ax.set_xlabel("Layer")
            ax.set_ylabel(ylab)
            ax.set_title(sig)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.5, N_LAYERS - 0.5)

        opt = R30_OPTIMAL.get(bench_name, {})
        tnote = f"R30 T-block [{opt.get('t_start','?')}, {opt.get('t_stop','?')})"
        fig.suptitle(
            f"R33 信号逐层曲线 + {tnote}  |  {bench_name}  N={len(records)}",
            fontsize=13,
        )
        fig.tight_layout()
        safe = bench_name.replace("/", "-")
        fig.savefig(os.path.join(out_dir, f"{safe}_r33_signals_vs_layer.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {out_dir}/{safe}_r33_signals_vs_layer.png")


# ─── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    tok, model = load_model()
    print(f"Model loaded ({time.time()-t0:.1f}s)  device={DEVICE}")

    all_data = {}  # bench -> list[record]

    for bench_name, loader_fn in BENCHMARK_LOADERS:
        samples = loader_fn(N_PHASE0)
        if samples is None:
            print(f"  [{bench_name}] 跳过（数据不可用）")
            continue
        print(f"\n{'='*55}\n{bench_name} N={len(samples)}")

        records = []
        for i, samp in enumerate(samples):
            enc   = tok(samp["prompt"], return_tensors="pt", add_special_tokens=False)
            iids  = enc["input_ids"].to(DEVICE)
            amask = enc.get("attention_mask")
            amask = amask.to(DEVICE) if amask is not None else None

            probe    = collect_r33_signals(model, iids, amask)
            bl_ok, etd_ok = eval_sample(tok, model, samp)
            oracle_gain   = int(etd_ok) - int(bl_ok)

            rec = {"oracle_gain": oracle_gain, **probe["scalars"]}
            rec["per_layer"] = probe["per_layer"]
            records.append(rec)

            flag = {-1:"↓",0:"=",1:"↑"}[oracle_gain]
            print(f"  [{i+1:2d}] {flag}  "
                  f"gini@8={probe['scalars']['ffn_gini_at8']:.3f}  "
                  f"gap@8={probe['scalars']['attn_spectral_gap_at8']:.2f}  "
                  f"plasticity={probe['scalars']['plasticity_score']:.3f}")

        all_data[bench_name] = records
        ginis = [r["ffn_gini_at8"] for r in records if np.isfinite(r["ffn_gini_at8"])]
        gaps  = [r["attn_spectral_gap_at8"] for r in records if np.isfinite(r["attn_spectral_gap_at8"])]
        print(f"  ffn_gini@8:  mean={np.mean(ginis):.4f}  std={np.std(ginis):.4f}  "
              f"[R32 rc_global std=0.020 for comparison]")
        print(f"  attn_gap@8:  mean={np.mean(gaps):.4f}  std={np.std(gaps):.4f}")

    # ── 保存原始数据 ──────────────────────────────────────────────────────────
    save_path = os.path.join(RESULTS_DIR, "r33_phase0_data.json")
    with open(save_path, "w") as f:
        json.dump({b: [{k: v for k, v in r.items() if k != "per_layer"} for r in recs]
                   for b, recs in all_data.items()}, f, indent=2)
    print(f"\nSaved → {save_path}")

    full_path = os.path.join(RESULTS_DIR, "r33_phase0_data_full.json")
    with open(full_path, "w") as f:
        json.dump(
            {
                "n_layers": N_LAYERS,
                "r30_optimal": R30_OPTIMAL,
                "benches": {
                    b: [{k: v for k, v in r.items()} for r in recs]
                    for b, recs in all_data.items()
                },
            },
            f,
            indent=2,
        )
    print(f"Saved (含 per_layer) → {full_path}")

    layer_axis = list(range(N_LAYERS))

    # ── 绘图 1：FFN Gini 逐层剖面 ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    signal_names = ["ffn_gini", "ffn_act_entropy", "ffn_boundary_frac"]
    y_labels = ["FFN Gini（越低越塑性）", "FFN 激活熵", "FFN 边界邻近比例"]

    for ax, sig, ylabel in zip(axes, signal_names, y_labels):
        for bench_name, records in all_data.items():
            if not records: continue
            mat = np.array([r["per_layer"][sig] for r in records])
            layer_range = layer_axis
            mean_ = np.nanmean(mat, axis=0)
            std_  = np.nanstd(mat, axis=0)
            c = BENCH_COLORS.get(bench_name, "gray")
            ax.plot(layer_range, mean_, label=bench_name, color=c, linewidth=2)
            ax.fill_between(layer_range, mean_-std_, mean_+std_, color=c, alpha=0.15)
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(sig, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, N_LAYERS - 0.5)

    fig.suptitle("R33 Phase 0: FFN 信号逐层剖面（全层 0–35，多 bench 叠图）", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "r33_phase0_ffn_profile.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → r33_phase0_ffn_profile.png")

    # ── 绘图 2：Attention 谱隙剖面 ───────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    attn_sigs  = ["attn_spectral_gap", "attn_head_consensus", "attn_top2_mass"]
    attn_ylabs = ["Attn Spectral Gap（幂迭代收敛速度）", "Head Consensus（方向一致性）", "Top-2 Token 注意力质量"]

    for ax, sig, ylabel in zip(axes2, attn_sigs, attn_ylabs):
        for bench_name, records in all_data.items():
            if not records: continue
            mat = np.array([r["per_layer"][sig] for r in records])
            layer_range = layer_axis
            mean_ = np.nanmean(mat, axis=0)
            std_  = np.nanstd(mat, axis=0)
            c = BENCH_COLORS.get(bench_name, "gray")
            ax.plot(layer_range, mean_, label=bench_name, color=c, linewidth=2)
            ax.fill_between(layer_range, mean_-std_, mean_+std_, color=c, alpha=0.15)
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(sig, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, N_LAYERS - 0.5)

    fig2.suptitle("R33 Phase 0: Attention 信号逐层剖面（全层 0–35，多 bench 叠图）", fontsize=13)
    fig2.tight_layout()
    fig2.savefig(os.path.join(FIGURES_DIR, "r33_phase0_attn_profile.png"), dpi=150, bbox_inches="tight")
    print(f"Saved → r33_phase0_attn_profile.png")

    plt.close(fig2)

    r30_overlay_dir = os.path.join(FIGURES_DIR, "r33_by_layer_r30")
    plot_per_benchmark_r30_overlay(all_data, r30_overlay_dir)

    fig3, ax3 = plt.subplots(figsize=(9, 7))
    gain_markers = {-1: "v", 0: "o", 1: "^"}
    gain_colors  = {-1: "#F44336", 0: "#BDBDBD", 1: "#4CAF50"}
    gain_labels  = {-1: "ETD有害", 0: "无影响", 1: "ETD有益"}
    plotted_gains = set()

    for bench_name, records in all_data.items():
        for r in records:
            g  = r["oracle_gain"]
            gx = r["ffn_gini_at8"]
            sy = r["attn_spectral_gap_at8"]
            if not (np.isfinite(gx) and np.isfinite(sy)): continue
            label = gain_labels[g] if g not in plotted_gains else ""
            ax3.scatter(gx, sy, c=gain_colors[g], marker=gain_markers[g],
                        s=90, alpha=0.75, edgecolors=BENCH_COLORS.get(bench_name,"gray"),
                        linewidths=1.2, label=label)
            plotted_gains.add(g)

    ax3.set_xlabel("FFN Gini@Layer8（↓ 塑性态）", fontsize=12)
    ax3.set_ylabel("Attn Spectral Gap@Layer8（↑ 有吸引子）", fontsize=12)
    ax3.set_title("R33 Phase 0: 2D 信号空间\n预测：左上象限（低Gini+高Gap）= ETD有益", fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    # 参考象限分割线
    all_ginis = [r["ffn_gini_at8"] for recs in all_data.values() for r in recs if np.isfinite(r["ffn_gini_at8"])]
    all_gaps  = [r["attn_spectral_gap_at8"] for recs in all_data.values() for r in recs if np.isfinite(r["attn_spectral_gap_at8"])]
    if all_ginis: ax3.axvline(np.median(all_ginis), color="gray", linestyle="--", alpha=0.5, label="median Gini")
    if all_gaps:  ax3.axhline(np.median(all_gaps),  color="gray", linestyle=":",  alpha=0.5, label="median Gap")

    fig3.tight_layout()
    fig3.savefig(os.path.join(FIGURES_DIR, "r33_phase0_2d_signal_space.png"), dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved → r33_phase0_2d_signal_space.png")

    # ── 统计摘要 ─────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("信号 std 对比（R32 rc_global std ≈ 0.020）：")
    summary = {}
    for bench_name, records in all_data.items():
        ginis = [r["ffn_gini_at8"] for r in records if np.isfinite(r.get("ffn_gini_at8", float("nan")))]
        gaps  = [r["attn_spectral_gap_at8"] for r in records if np.isfinite(r.get("attn_spectral_gap_at8", float("nan")))]
        plast = [r["plasticity_score"] for r in records if np.isfinite(r.get("plasticity_score", float("nan")))]
        n_pos = sum(r["oracle_gain"] > 0 for r in records)
        n_neg = sum(r["oracle_gain"] < 0 for r in records)
        print(f"\n{bench_name}:  +{n_pos}/-{n_neg}")
        print(f"  ffn_gini@8:         mean={np.mean(ginis):.4f}  std={np.std(ginis):.4f}")
        print(f"  attn_spectral_gap@8:mean={np.mean(gaps):.4f}  std={np.std(gaps):.4f}")
        print(f"  plasticity_score:   mean={np.mean(plast):.4f}  std={np.std(plast):.4f}")
        summary[bench_name] = {
            "n": len(records), "n_pos": n_pos, "n_neg": n_neg,
            "ffn_gini_mean": float(np.mean(ginis)), "ffn_gini_std": float(np.std(ginis)),
            "attn_gap_mean": float(np.mean(gaps)),  "attn_gap_std":  float(np.std(gaps)),
        }

    stats_path = os.path.join(RESULTS_DIR, "r33_phase0_stats.json")
    with open(stats_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved stats → {stats_path}")
    print(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
