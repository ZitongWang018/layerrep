"""
R33 Phase 1: FFN/Attn 信号与 oracle_gain 的 Spearman 相关分析（N=100/benchmark）
================================================================================
新增信号（较 R32 的 rc_global/theta_global 精准对应理论输入依赖性）：

慢权重（FFN 激活相变）:
  ffn_gini@8        - Ω(x) 集中度（低=塑性态）
  ffn_act_entropy@8 - 多方向竞争激活程度
  ffn_boundary_frac@8 - 临界神经元比例（高=相变易发）

快权重（KV Cache 幂迭代）:
  attn_spectral_gap@8 - W_fast(x) 主特征值主导度（高=有吸引子）
  attn_head_consensus@8 - 各头语义方向一致性

2-Pass 神经元翻转（直接测量相变）:
  neuron_flip_rate  - 两次前向中改变激活符号的 FFN 神经元比例（直接相变证据）
  rc_global, theta_global - R32 的基准信号（作为对照）

核心目标：
  H1: ρ(ffn_gini@8, oracle_gain) < -0.15（低Gini → ETD有益）
  H2: ρ(attn_spectral_gap@8, oracle_gain) > 0.15（高Gap → ETD有益）
  H3: ρ(neuron_flip_rate, oracle_gain) > 0.15（高翻转率 → ETD效果强）
  H4: ρ(ffn_boundary_frac@8, oracle_gain) > 0.15（高临界比 → ETD效果强）

输出：
  results/r33_phase1_spearman.json
  figures/r33_phase1_spearman_heatmap.png
  figures/r33_phase1_scatter_grid.png
"""
from __future__ import annotations
import sys, os, json, time, warnings

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

sys.path.insert(0, "/root/autodl-tmp/loop_layer")
sys.path.insert(0, "/root/autodl-tmp/loop_layer/ETD")
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from etd_forward import (
    etd_forward_logits,
    baseline_forward_logits,
    loglikelihood_continuation,
)
from r29.signal_funcs import (
    ffn_gini_last_token,
    ffn_activation_entropy_last_token,
    ffn_boundary_frac_last_token,
    ffn_active_frac_last_token,
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

N_PHASE1     = 100
N_LAYERS     = 36
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE        = torch.bfloat16
T_START, T_STOP, CHAMP_K = 8, 22, 2
ALPHA        = min(1.0, 6.0 / max(T_STOP - T_START, 1))

BENCH_CFG = {
    "BoolQ": "#2196F3", "ARC-C": "#F44336",
    "CSQA": "#4CAF50",  "TruthfulQA": "#FF9800",
    "MMLU-HS": "#9C27B0",
}

SIGNAL_KEYS = [
    "ffn_gini_at8",
    "ffn_act_entropy_at8",
    "ffn_boundary_frac_at8",
    "ffn_active_frac_at8",
    "attn_spectral_gap_at8",
    "attn_head_consensus_at8",
    "attn_top2_mass_at8",
    "neuron_flip_rate",
    "rc_global",
    "theta_global",
    "plasticity_score",
]


# ─── 模型加载 ──────────────────────────────────────────────────────────────────
def load_model():
    print("Loading model …")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, device_map="auto",
        attn_implementation="eager",
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
                    "choices": ["no","yes"], "label": int(x["label"])})
        if len(out) >= n: break
    return out

def load_arc_c(n):
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")["test"]
    out = []
    for x in ds:
        lmap = {k:i for i,k in enumerate(x["choices"]["label"])}
        out.append({"prompt": f"Question: {x['question']}\nAnswer:",
                    "choices": x["choices"]["text"], "label": lmap.get(x["answerKey"],0)})
        if len(out) >= n: break
    return out

def load_csqa(n):
    ds = load_dataset("tau/commonsense_qa")["validation"]
    lmap = {"A":0,"B":1,"C":2,"D":3,"E":4}
    out = []
    for x in ds:
        out.append({"prompt": f"Question: {x['question']}\nAnswer:",
                    "choices": x["choices"]["text"], "label": lmap.get(x["answerKey"],0)})
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
        print(f"  [WARN] MMLU-HS unavailable: {e}")
        return None

BENCHMARK_LOADERS = [
    ("BoolQ",      load_boolq),
    ("ARC-C",      load_arc_c),
    ("CSQA",       load_csqa),
    ("TruthfulQA", load_truthfulqa),
    ("MMLU-HS",    load_mmlu_hs),
]


# ─── 2-Pass 探针：ETD k=2 + hook计数 获取两次前向的激活数据 ─────────────────
@torch.no_grad()
def collect_probe_signals(model, input_ids: torch.Tensor, attention_mask) -> dict:
    """
    策略：用 ETD forward (k=2) 作为 2-pass probe。
      - T-block 层的 hook 在 ETD 中会触发两次（第1次=pass1，第2次=pass2）
      - 第 T_STOP-1 层的输出 hook 同样触发两次 → 得到 h1 / h2
      - 第 T_START-1 层的输出 hook 触发一次 → 得到 h0

    这避免了手动逐层调用 Qwen3 层时缺少 position_embeddings 的问题。
    """
    base = model.model

    gate_p1, gate_p2 = {}, {}
    attn_p1 = {}
    pass_counts_gate = {li: 0 for li in range(T_START, T_STOP)}

    # 用于捕获 h0 / h1 / h2
    hidden_captures: list[torch.Tensor] = []
    hooks = []

    # ── hook: T_START-1 层输出 → h0 ─────────────────────────────────────────
    if T_START > 0:
        def h0_hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            if len(hidden_captures) == 0:
                hidden_captures.append(h.detach().clone())
        hooks.append(base.layers[T_START - 1].register_forward_hook(h0_hook))

    # ── hook: T_STOP-1 层输出 → h1 (first call) / h2 (second call) ──────────
    last_t_count = [0]
    def last_t_hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        if last_t_count[0] == 0:
            hidden_captures.append(h.detach().clone())   # h1
        elif last_t_count[0] == 1:
            hidden_captures.append(h.detach().clone())   # h2
        last_t_count[0] += 1
    hooks.append(base.layers[T_STOP - 1].register_forward_hook(last_t_hook))

    # ── hook: T-block 中各层的 gate + attn（pass1/pass2 计数） ───────────────
    for li in range(T_START, T_STOP):
        def mk_gate(idx):
            def fn(_m, _i, out):
                cnt = pass_counts_gate[idx]
                if cnt == 0:
                    gate_p1[idx] = out.detach()
                elif cnt == 1:
                    gate_p2[idx] = out.detach()
                pass_counts_gate[idx] += 1
            return fn
        def mk_attn(idx):
            def fn(_m, _i, out):
                if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                    if idx not in attn_p1:      # 只记录第一次（pass1）
                        attn_p1[idx] = out[1].detach()
            return fn
        hooks.append(base.layers[li].mlp.act_fn.register_forward_hook(mk_gate(li)))
        hooks.append(base.layers[li].self_attn.register_forward_hook(mk_attn(li)))

    # ── ETD k=2 forward：T-block 走两次 ─────────────────────────────────────
    nt = T_STOP - T_START
    etd_forward_logits(model, input_ids, attention_mask,
                       n_e=T_START, n_t=nt, k=2, alpha=ALPHA)

    for h in hooks:
        h.remove()

    # ── 解析捕获的 hidden states ──────────────────────────────────────────────
    # hidden_captures: [h0(可能), h1, h2]
    if len(hidden_captures) == 3:
        h0, h1, h2 = hidden_captures
    elif len(hidden_captures) == 2:
        h1, h2 = hidden_captures
        h0 = h1  # fallback（T_START==0 极少发生）
    else:
        h0 = h1 = h2 = None

    # rc_global / theta_global
    if h0 is not None and h1 is not None and h2 is not None:
        delta0 = (h1 - h0).reshape(-1).float()
        delta1 = (h2 - h1).reshape(-1).float()
        rc    = float((delta1.norm() / (delta0.norm() + 1e-9)).item())
        theta = float(F.cosine_similarity(delta0.unsqueeze(0), delta1.unsqueeze(0)).item())
    else:
        rc = theta = float("nan")

    # ── neuron_flip_rate ──────────────────────────────────────────────────────
    flip_rates = []
    for li in range(T_START, T_STOP):
        ga1, ga2 = gate_p1.get(li), gate_p2.get(li)
        if ga1 is None or ga2 is None:
            continue
        act1 = ga1[0, -1, :].float()
        act2 = ga2[0, -1, :].float()
        flipped = ((act1 > 0) != (act2 > 0)).float().mean().item()
        flip_rates.append(flipped)
    neuron_flip_rate = float(np.mean(flip_rates)) if flip_rates else float("nan")

    # ── 组装标量信号 ──────────────────────────────────────────────────────────
    li0 = T_START
    ga0 = gate_p1.get(li0)
    aw0 = attn_p1.get(li0)

    def s(fn, arr, fb=float("nan")): return float(fn(arr)) if arr is not None else fb

    scalars = {
        "ffn_gini_at8":           s(ffn_gini_last_token, ga0),
        "ffn_act_entropy_at8":    s(ffn_activation_entropy_last_token, ga0),
        "ffn_boundary_frac_at8":  s(ffn_boundary_frac_last_token, ga0),
        "ffn_active_frac_at8":    s(ffn_active_frac_last_token, ga0),
        "attn_spectral_gap_at8":  s(attn_spectral_gap_last_token, aw0),
        "attn_head_consensus_at8":s(attn_head_consensus_last_token, aw0),
        "attn_top2_mass_at8":     s(attn_top2_mass_last_token, aw0),
        "neuron_flip_rate":        neuron_flip_rate,
        "rc_global":               rc,
        "theta_global":            theta,
    }
    g8  = scalars["ffn_gini_at8"]
    sg8 = scalars["attn_spectral_gap_at8"]
    scalars["plasticity_score"] = ((1 - g8) * sg8) if np.isfinite(g8) and np.isfinite(sg8) else float("nan")

    return scalars


# ─── 样本评估 ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_sample(tok, model, sample) -> tuple[bool, bool]:
    prompt, choices, label = sample["prompt"], sample["choices"], sample["label"]
    pref_ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
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
                logits = etd_forward_logits(model, iids, amask, n_e=T_START, n_t=nt, k=CHAMP_K, alpha=ALPHA)
            else:
                logits = baseline_forward_logits(model, iids, amask)
            scores.append(loglikelihood_continuation(logits, iids, prompt_len))
        return int(np.argmax(scores)) == label

    return score(False), score(True)


# ─── Spearman 报告 ────────────────────────────────────────────────────────────
def compute_spearman(records: list[dict]) -> dict:
    result = {}
    og = [r["oracle_gain"] for r in records]
    for sig in SIGNAL_KEYS:
        vals = [r.get(sig, float("nan")) for r in records]
        pairs = [(v, g) for v, g in zip(vals, og) if np.isfinite(v)]
        if len(pairs) < 5:
            result[sig] = {"rho": float("nan"), "n": len(pairs)}
            continue
        vs, gs = zip(*pairs)
        rho, pval = spearmanr(vs, gs)
        result[sig] = {"rho": float(rho), "pval": float(pval), "n": len(pairs)}
    return result


# ─── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    tok, model = load_model()
    print(f"Model loaded ({time.time()-t0:.1f}s)")

    bench_results = {}

    for bench_name, loader_fn in BENCHMARK_LOADERS:
        samples = loader_fn(N_PHASE1)
        if samples is None:
            print(f"\n[{bench_name}] 跳过（数据不可用）")
            continue
        print(f"\n{'='*58}\n{bench_name}  N={len(samples)}")
        records = []
        for i, samp in enumerate(samples):
            enc   = tok(samp["prompt"], return_tensors="pt", add_special_tokens=False)
            iids  = enc["input_ids"].to(DEVICE)
            amask = enc.get("attention_mask")
            amask = amask.to(DEVICE) if amask is not None else None

            sigs = collect_probe_signals(model, iids, amask)
            bl_ok, etd_ok = eval_sample(tok, model, samp)
            og = int(etd_ok) - int(bl_ok)
            rec = {"oracle_gain": og, **sigs}
            records.append(rec)

            flag = {-1:"↓",0:"=",1:"↑"}[og]
            if (i+1) % 10 == 0 or og != 0:
                print(f"  [{i+1:3d}] {flag}  "
                      f"gini={sigs['ffn_gini_at8']:.3f}  "
                      f"gap={sigs['attn_spectral_gap_at8']:.2f}  "
                      f"flip={sigs['neuron_flip_rate']:.3f}")

        sp = compute_spearman(records)
        bench_results[bench_name] = {"spearman": sp, "records": records}

        n_pos = sum(r["oracle_gain"]>0 for r in records)
        n_neg = sum(r["oracle_gain"]<0 for r in records)
        print(f"\n  oracle_gain 分布：+{n_pos}/-{n_neg}/={len(records)-n_pos-n_neg}")
        print(f"  {'信号名':<30} {'ρ':>8}  {'p':>8}  {'n':>5}")
        print(f"  {'-'*55}")
        for sig, v in sp.items():
            rho  = v.get('rho',float('nan'))
            pval = v.get('pval',float('nan'))
            n_   = v.get('n', 0)
            sig_mark = " ←★" if abs(rho)>0.15 else ""
            print(f"  {sig:<30} {rho:>+8.4f}  {pval:>8.4f}  {n_:>5}{sig_mark}")

    # ── 保存原始结果 ──────────────────────────────────────────────────────────
    save_obj = {
        b: {"spearman": r["spearman"],
            "records": [{k:v for k,v in rec.items()} for rec in r["records"]]}
        for b, r in bench_results.items()
    }
    res_path = os.path.join(RESULTS_DIR, "r33_phase1_spearman.json")
    with open(res_path, "w") as f:
        json.dump(save_obj, f, indent=2)
    print(f"\nSaved → {res_path}")

    # ── 热图：各 bench × 各信号 的 ρ 值 ─────────────────────────────────────
    benches = list(bench_results.keys())
    rho_mat = np.full((len(SIGNAL_KEYS), len(benches)), float("nan"))
    for bi, bn in enumerate(benches):
        sp = bench_results[bn]["spearman"]
        for si, sig in enumerate(SIGNAL_KEYS):
            rho_mat[si, bi] = sp.get(sig, {}).get("rho", float("nan"))

    fig, ax = plt.subplots(figsize=(max(8, len(benches)*1.8), max(7, len(SIGNAL_KEYS)*0.65)))
    cmap = plt.cm.RdYlGn
    im = ax.imshow(rho_mat, cmap=cmap, vmin=-0.4, vmax=0.4, aspect="auto")
    ax.set_xticks(range(len(benches)));  ax.set_xticklabels(benches, fontsize=11)
    ax.set_yticks(range(len(SIGNAL_KEYS))); ax.set_yticklabels(SIGNAL_KEYS, fontsize=10)
    for si in range(len(SIGNAL_KEYS)):
        for bi in range(len(benches)):
            v = rho_mat[si, bi]
            if np.isfinite(v):
                ax.text(bi, si, f"{v:+.3f}", ha="center", va="center",
                        fontsize=9, color="black" if abs(v)<0.25 else "white",
                        fontweight="bold" if abs(v)>0.15 else "normal")
    plt.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_title(f"R33 Phase 1: Spearman ρ (信号 vs oracle_gain)\nN={N_PHASE1}/benchmark  T-block=[{T_START},{T_STOP})", fontsize=12)
    fig.tight_layout()
    heatmap_path = os.path.join(FIGURES_DIR, "r33_phase1_spearman_heatmap.png")
    fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    print(f"Saved → r33_phase1_spearman_heatmap.png")

    # ── 散点图：最优信号 vs oracle_gain（前 3 高 |ρ|）─────────────────────────
    # 找全局平均 |ρ| 最高的信号
    mean_rho = {}
    for sig in SIGNAL_KEYS:
        rhos = [bench_results[bn]["spearman"].get(sig,{}).get("rho",float("nan"))
                for bn in benches]
        valid = [abs(r) for r in rhos if np.isfinite(r)]
        mean_rho[sig] = float(np.mean(valid)) if valid else 0.0
    top3 = sorted(mean_rho, key=mean_rho.get, reverse=True)[:3]

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    for ax, sig in zip(axes2, top3):
        for bn, color in BENCH_CFG.items():
            if bn not in bench_results: continue
            for rec in bench_results[bn]["records"]:
                v = rec.get(sig, float("nan"))
                og = rec["oracle_gain"]
                if np.isfinite(v):
                    ax.scatter(v, og + (np.random.rand()-0.5)*0.1,
                               c=color, alpha=0.55, s=30, edgecolors="none")
        rho_avg = mean_rho[sig]
        ax.set_xlabel(sig, fontsize=11)
        ax.set_ylabel("oracle_gain (jitter)", fontsize=11)
        ax.set_title(f"{sig}\nmean|ρ|={rho_avg:.4f}", fontsize=11)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3)

    bench_labels = [plt.Line2D([0],[0], marker='o', color=c, linestyle='', label=b, markersize=8)
                    for b,c in BENCH_CFG.items() if b in benches]
    fig2.legend(handles=bench_labels, loc="lower right", fontsize=10)
    fig2.suptitle(f"R33 Phase 1: Top-3 信号散点图（N={N_PHASE1}/benchmark）", fontsize=13)
    fig2.tight_layout()
    scatter_path = os.path.join(FIGURES_DIR, "r33_phase1_scatter_grid.png")
    fig2.savefig(scatter_path, dpi=150, bbox_inches="tight")
    print(f"Saved → r33_phase1_scatter_grid.png")

    # ── 总结报告 ─────────────────────────────────────────────────────────────
    print("\n" + "="*58)
    print("Phase 1 摘要（H1~H4 假设检验）：")
    hypotheses = [
        ("H1", "ffn_gini_at8",        "<",  -0.15, "低Gini→ETD有益"),
        ("H2", "attn_spectral_gap_at8",">" ,  0.15, "高Gap→ETD有益"),
        ("H3", "neuron_flip_rate",     ">",   0.15, "高翻转率→ETD强"),
        ("H4", "ffn_boundary_frac_at8",">",  0.15, "高临界比→ETD强"),
    ]
    n_sig_hyp = 0
    for hid, sig, direction, threshold, desc in hypotheses:
        rhos = [bench_results[bn]["spearman"].get(sig,{}).get("rho",float("nan"))
                for bn in benches if bn in bench_results]
        valid = [r for r in rhos if np.isfinite(r)]
        if not valid:
            print(f"  {hid} [{desc}]: 无数据")
            continue
        mean_r = np.mean(valid)
        passed = (mean_r < threshold) if direction=="<" else (mean_r > threshold)
        status = "✓ 支持" if passed else "✗ 不支持"
        print(f"  {hid} [{desc}]: mean_ρ={mean_r:+.4f}  {status}")
        if passed: n_sig_hyp += 1

    print(f"\n  总计 {n_sig_hyp}/4 个假设得到初步支持")
    if n_sig_hyp > 0:
        print("  → 建议触发 Phase 2 Skip Gate 实验")
    else:
        print("  → 信号预测能力不足，需继续理论探索")

    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
