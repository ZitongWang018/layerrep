"""
R26: Selective Skip Gate — top1_prob@8 + causal_norm_ratio

核心假设:
  TruthfulQA 熵斜率≈0（既不上升也不下降），S1_slope0.05_e53 过度循环（91.2%触发率）。
  当模型在第8层已经"自信但停滞"时（top1_prob@8 高 + slope≈0），
  继续循环反而干扰收敛，降低准确率。

引入两个新因果信号:
  1. top1_prob@8: logit-lens 在第8层的 top-1 概率（绝对置信度）
  2. causal_norm_ratio: mean(norm_delta[1:8]) / norm_delta[8]
     — 若早期层变化大于第8层，说明编码在前期已完成，T-block 收益有限

测试策略(均在 S1_slope0.05_e53 基础上添加跳过条件):
  S3_conf07_flat003: top1_prob@8 > 0.7 AND |slope_8| < 0.03 → skip ETD
  S3_conf08_flat003: top1_prob@8 > 0.8 AND |slope_8| < 0.03 → skip ETD
  S3_conf07_flat005: top1_prob@8 > 0.7 AND |slope_8| < 0.05 → skip ETD
  S4_norm12: causal_norm_ratio > 1.2 → skip ETD
  S4_norm15: causal_norm_ratio > 1.5 → skip ETD
  S5_compound: S3_conf07_flat003 OR S4_norm12 → skip

Phase 1: N=200 × 5 benchmarks（快速筛选，两个bench落后就淘汰）
Phase 2: N=500 × 5 benchmarks（验证最优策略）
"""

import sys, os, json, time, math, warnings
sys.path.insert(0, "/root/autodl-tmp/loop_layer")
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# 使用项目已有的 ETD 前向传播（正确处理 Qwen3 的 position_embeddings）
_ETD_PATH = "/root/autodl-tmp/loop_layer/ETD"
if _ETD_PATH not in sys.path:
    sys.path.insert(0, _ETD_PATH)
from etd_forward import etd_forward_logits  # noqa: E402

# ─── 路径配置 ───────────────────────────────────────────────────────────────
MODEL_PATH = "/root/autodl-tmp/model_qwen"
RESULTS_DIR = "/root/autodl-tmp/loop_layer/experiments/results"
FIGURES_DIR = "/root/autodl-tmp/loop_layer/experiments/figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─── ETD 配置 ──────────────────────────────────────────────────────────────
CHAMP_T_START = 8
CHAMP_T_STOP  = 22
CHAMP_K       = 2
# S1 门控参数
S1_ENT_THRESH   = 5.3
S1_SLOPE_THRESH = 0.05
# 新信号阈值（Phase 1 扫描）
CONF_THRESHS    = [0.7, 0.8]
FLAT_THRESHS    = [0.03, 0.05]
NORM_RATIO_THRESHS = [1.2, 1.5]

N_LAYERS   = 36
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE      = torch.bfloat16


# ─── 模型加载 ──────────────────────────────────────────────────────────────
def load_model():
    print("Loading model …")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE,
        device_map="auto", trust_remote_code=True
    )
    model.eval()
    try:
        lm_head = model.lm_head
        ln_f    = model.model.norm
    except AttributeError:
        lm_head = model.model.embed_out
        ln_f    = model.model.final_layer_norm
    return tok, model, lm_head, ln_f


# ─── 信号采集（含新信号）───────────────────────────────────────────────────
@torch.no_grad()
def collect_signals(model, lm_head, ln_f, input_ids):
    """
    因果信号（仅用 [0..8] 层信息）:
      entropy_arr[l], top1_prob_arr[l]: logit-lens 指标 (l=0..8)
      norm_delta[l]: 相对激活变化 ‖h[l]-h[l-1]‖/‖h[l-1]‖ (l=1..8)
      entropy_8, entropy_slope_8, top1_prob_8
      causal_norm_ratio: mean(norm_delta[1:8]) / norm_delta[8]
      encode_bias: 全程 EBF 信号（非因果，仅供对比）
    """
    hidden_all = []  # 收集全部36层，用于 encode_bias

    hooks = []
    for li in range(N_LAYERS):
        def make_hook(idx):
            def hook_fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                hidden_all.append((idx, h.detach().cpu().float()))
            return hook_fn
        hooks.append(model.model.layers[li].register_forward_hook(make_hook(li)))

    _ = model(input_ids, use_cache=False)
    for h in hooks:
        h.remove()
    hidden_all.sort(key=lambda x: x[0])

    # 整理 hidden states
    hiddens = [t for _, t in hidden_all]  # list[36], each (1, seq, dim)

    # Logit-lens 计算（只计算层 0..8，保持因果）
    entropy_arr   = []
    top1_prob_arr = []
    model_dtype = next(model.parameters()).dtype
    for li in range(9):
        h_l = hiddens[li][0, -1, :].unsqueeze(0).to(DEVICE).to(model_dtype)
        logits = lm_head(ln_f(h_l)).float()
        probs  = F.softmax(logits, dim=-1)[0]
        ent    = -(probs * (probs + 1e-12).log()).sum().item()
        top1   = probs.max().item()
        entropy_arr.append(ent)
        top1_prob_arr.append(top1)

    # norm_delta (l=1..8)
    norm_delta = []
    for li in range(1, 9):
        h_cur  = hiddens[li][0, -1, :]
        h_prev = hiddens[li-1][0, -1, :]
        nd = (h_cur - h_prev).norm() / (h_prev.norm() + 1e-8)
        norm_delta.append(nd.item())

    entropy_8      = entropy_arr[8]
    entropy_6      = entropy_arr[6]
    entropy_slope_8 = (entropy_8 - entropy_6) / 2.0
    top1_prob_8    = top1_prob_arr[8]

    # causal_norm_ratio = mean(norm_delta[0:7]) / norm_delta[7]
    # norm_delta[0] = layers[1] relative to layers[0], ..., norm_delta[7] = layers[8] vs layers[7]
    early_mean = float(np.mean(norm_delta[0:7]))  # layers 1..7
    nd_8       = norm_delta[7]                    # layers 8 vs 7
    causal_norm_ratio = early_mean / (nd_8 + 1e-8)

    # encode_bias（非因果，全 36 层，仅供对比）
    delta_gaps = []
    for li in range(N_LAYERS - 1):
        h_cur  = hiddens[li+1][0, -1, :]
        h_prev = hiddens[li][0, -1, :]
        nd = (h_cur - h_prev).norm() / (h_prev.norm() + 1e-8)
        delta_gaps.append(nd.item())
    early_part = float(np.mean(delta_gaps[0:7]))
    think_part = float(np.mean(delta_gaps[7:22]))
    encode_bias = early_part / (think_part + 1e-8)

    return {
        "entropy_arr":       entropy_arr,
        "top1_prob_arr":     top1_prob_arr,
        "norm_delta":        norm_delta,
        "entropy_8":         entropy_8,
        "entropy_slope_8":   entropy_slope_8,
        "top1_prob_8":       top1_prob_8,
        "causal_norm_ratio": causal_norm_ratio,
        "encode_bias":       encode_bias,
    }


# ─── ETD / Baseline 前向传播 ───────────────────────────────────────────────
def run_etd(model, input_ids, t_start, t_stop, k=2):
    """调用项目已有的 etd_forward_logits（正确处理 Qwen3 position_embeddings）。"""
    n_t   = t_stop - t_start
    if n_t < 1:
        out = model(input_ids, use_cache=False)
        return out.logits[:, -1, :]
    alpha = min(1.0, 6.0 / n_t)
    logits = etd_forward_logits(
        model, input_ids,
        attention_mask=None,
        n_e=t_start, n_t=n_t, k=k,
        alpha=alpha
    )
    return logits[:, -1, :]


def run_baseline(model, input_ids):
    out = model(input_ids, use_cache=False)
    return out.logits[:, -1, :]


# ─── t_stop 早停（S4 causal / S1 基础）────────────────────────────────────
@torch.no_grad()
def s4_capped_tstop(model, lm_head, ln_f, input_ids, entropy_8, t_start=8):
    """
    早停搜索：从 t_start+4 到 CHAMP_T_STOP，entropy[l] < 0.5*entropy_8 时停。
    不超过 CHAMP_T_STOP=22。
    """
    hidden_think = {}
    hooks = []
    for li in range(t_start, CHAMP_T_STOP + 1):
        def make_h(idx):
            def fn(m, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                hidden_think[idx] = h.detach().cpu().float()
            return fn
        hooks.append(model.model.layers[li].register_forward_hook(make_h(li)))

    _ = model(input_ids, use_cache=False)
    for h in hooks:
        h.remove()

    model_dtype = next(model.parameters()).dtype
    threshold = 0.5 * entropy_8
    for li in range(t_start + 4, CHAMP_T_STOP + 1):
        if li not in hidden_think:
            continue
        h_l   = hidden_think[li][0, -1, :].unsqueeze(0).to(DEVICE).to(model_dtype)
        logits = lm_head(ln_f(h_l)).float()
        probs  = F.softmax(logits, dim=-1)[0]
        ent    = -(probs * (probs + 1e-12).log()).sum().item()
        if ent < threshold:
            return li
    return CHAMP_T_STOP


# ─── 策略决策 ─────────────────────────────────────────────────────────────
def decide_strategy(signals, strategy_name, model=None, lm_head=None, ln_f=None, input_ids=None):
    """
    返回: (use_etd: bool, t_stop: int)
    use_etd=False → 使用 Baseline
    """
    ent8   = signals["entropy_8"]
    slope8 = signals["entropy_slope_8"]
    top1p8 = signals["top1_prob_8"]
    cnr    = signals["causal_norm_ratio"]

    # S1 baseline 门控
    s1_fires = (ent8 > S1_ENT_THRESH) or (slope8 > S1_SLOPE_THRESH)

    if strategy_name == "Baseline":
        return False, CHAMP_T_STOP

    elif strategy_name == "Champion":
        return True, CHAMP_T_STOP

    elif strategy_name == "EncBias-Filter":
        eb = signals["encode_bias"]
        return (eb <= 1.0), CHAMP_T_STOP

    elif strategy_name == "S1_slope0.05_e53":
        if s1_fires:
            return True, CHAMP_T_STOP
        else:
            t_stop = s4_capped_tstop(model, lm_head, ln_f, input_ids, ent8)
            return True, t_stop

    # ── 新策略：在 S1 基础上加 skip 条件 ──
    elif strategy_name.startswith("S3_"):
        # 解析参数: S3_conf{X}_flat{Y}
        parts = strategy_name[3:].split("_")
        conf_thresh = float(parts[0].replace("conf", "0.")) if len(parts) > 0 else 0.7
        # 用更精确的解析
        try:
            conf_thresh = float(strategy_name.split("conf")[1].split("_")[0]) / 10.0
            flat_thresh = float(strategy_name.split("flat")[1]) / 100.0
        except Exception:
            conf_thresh = 0.7
            flat_thresh = 0.03

        # skip 条件：高置信度 + 斜率平坦
        should_skip = (top1p8 > conf_thresh) and (abs(slope8) < flat_thresh)

        if should_skip:
            return False, CHAMP_T_STOP  # 跳过 ETD

        # 否则沿用 S1 逻辑
        if s1_fires:
            return True, CHAMP_T_STOP
        else:
            t_stop = s4_capped_tstop(model, lm_head, ln_f, input_ids, ent8)
            return True, t_stop

    elif strategy_name.startswith("S4_norm"):
        try:
            thresh = float(strategy_name.split("norm")[1].replace("_", "."))
        except Exception:
            thresh = 1.2

        should_skip = (cnr > thresh)

        if should_skip:
            return False, CHAMP_T_STOP

        if s1_fires:
            return True, CHAMP_T_STOP
        else:
            t_stop = s4_capped_tstop(model, lm_head, ln_f, input_ids, ent8)
            return True, t_stop

    elif strategy_name == "S5_compound":
        # S3_conf07_flat003 OR S4_norm12
        skip_conf = (top1p8 > 0.7) and (abs(slope8) < 0.03)
        skip_norm = (cnr > 1.2)
        should_skip = skip_conf or skip_norm

        if should_skip:
            return False, CHAMP_T_STOP

        if s1_fires:
            return True, CHAMP_T_STOP
        else:
            t_stop = s4_capped_tstop(model, lm_head, ln_f, input_ids, ent8)
            return True, t_stop

    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


# ─── 单样本评估 ────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_sample(tok, model, lm_head, ln_f, example, strategies):
    """
    返回: {strategy_name: correct (bool), ...}, signals
    """
    prompt     = example["prompt"]
    answer_key = example["answer"]

    # Tokenize
    inputs   = tok(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]

    # 采集信号
    signals = collect_signals(model, lm_head, ln_f, input_ids)

    results = {}
    for strat in strategies:
        use_etd, t_stop = decide_strategy(
            signals, strat, model, lm_head, ln_f, input_ids
        )
        # 多选题打分（对每个选项扩展输入，取最后 token logit）
        choices = example.get("choices", [])
        if not choices:
            results[strat] = False
            continue
        choice_scores = []
        for ch in choices:
            ch_ids    = tok.encode(" " + ch, add_special_tokens=False)
            ch_tensor = torch.tensor([ch_ids], device=DEVICE)
            inp_full  = torch.cat([input_ids, ch_tensor], dim=1)
            if use_etd:
                full_logits = run_etd(model, inp_full, CHAMP_T_START, t_stop, CHAMP_K)
            else:
                full_logits = run_baseline(model, inp_full)
            score = full_logits[0, ch_ids[-1]].item()
            choice_scores.append(score)
        pred    = choices[int(np.argmax(choice_scores))]
        correct = (pred.strip().lower() == answer_key.strip().lower())
        results[strat] = correct

    return results, signals


# ─── 数据加载 ──────────────────────────────────────────────────────────────
# 统一返回格式: {"prompt": str, "choices": [str, ...], "answer": str}

def _fmt(prefix, conts, label):
    """将 (prefix, conts, label) 三元组转为 R26 期望的字典格式。"""
    choices_str = [c.strip() for c in conts]
    return {
        "prompt": prefix,
        "choices": choices_str,
        "answer": choices_str[label],
    }


def load_boolq(n):
    ds  = load_dataset("aps/super_glue", "boolq")["validation"]
    out = []
    for r in ds:
        if len(out) >= n:
            break
        lab = int(r["label"])
        if lab < 0:
            continue
        prefix = f"{r['passage']}\nQuestion: {r['question']}?\nAnswer:"
        conts  = ["no", "yes"]
        out.append(_fmt(prefix, conts, lab))
    return out


def load_arc(subset, n):
    ds  = load_dataset("allenai/ai2_arc", subset)["test"]
    out = []
    for r in ds:
        if len(out) >= n:
            break
        q      = r["question"].strip()
        texts  = r["choices"]["text"]
        prefix = f"Question: {q}\nAnswer:"
        key    = r["answerKey"]
        label  = ord(key) - ord("A") if key in "ABCD" else int(key) - 1
        out.append(_fmt(prefix, texts, label))
    return out


def load_csqa(n):
    ds  = load_dataset("tau/commonsense_qa")["validation"]
    out = []
    lmap = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    for r in ds:
        if len(out) >= n:
            break
        key = r["answerKey"]
        if key not in lmap:
            continue
        prefix = f"Question: {r['question'].strip()}\nAnswer:"
        texts  = r["choices"]["text"]
        out.append(_fmt(prefix, texts, lmap[key]))
    return out


def load_truthfulqa(n):
    ds  = load_dataset("truthfulqa/truthful_qa", "multiple_choice")["validation"]
    out = []
    for r in ds:
        if len(out) >= n:
            break
        mc1    = r["mc1_targets"]
        labels = mc1["labels"]
        if 1 not in labels:
            continue
        label  = labels.index(1)
        prefix = f"Question: {r['question'].strip()}\nAnswer:"
        out.append(_fmt(prefix, mc1["choices"], label))
    return out


def load_benchmark(name, n):
    if name == "BoolQ":
        return load_boolq(n)
    elif name == "ARC-C":
        return load_arc("ARC-Challenge", n)
    elif name == "ARC-Easy":
        return load_arc("ARC-Easy", n)
    elif name == "CSQA":
        return load_csqa(n)
    elif name == "TruthfulQA":
        return load_truthfulqa(n)
    else:
        raise ValueError(name)


# ─── 相位 1：快速筛选 ──────────────────────────────────────────────────────
def run_phase(tok, model, lm_head, ln_f, benchmarks, strategies, n_per_bench, phase_name):
    results = {}
    signal_records = {}

    for bench in benchmarks:
        print(f"\n[{phase_name}] Benchmark: {bench} (N={n_per_bench})")
        examples = load_benchmark(bench, n_per_bench)
        correct  = {s: 0 for s in strategies}
        total    = 0
        bench_signals = []

        for i, ex in enumerate(examples):
            try:
                res, sig = eval_sample(tok, model, lm_head, ln_f, ex, strategies)
            except Exception as e:
                print(f"  [WARN] sample {i} error: {e}")
                continue
            for s in strategies:
                correct[s] += int(res.get(s, False))
            bench_signals.append(sig)
            total += 1

            if (i + 1) % 50 == 0:
                accs = {s: correct[s] / max(total, 1) for s in strategies}
                bl_acc = accs["Baseline"]
                print(f"  [{i+1}/{n_per_bench}] ", end="")
                for s in strategies:
                    print(f"{s[:20]}={accs[s]:.3f}({accs[s]-bl_acc:+.3f}) ", end="")
                print()

        accs = {s: correct[s] / max(total, 1) for s in strategies}
        results[bench] = {"accuracies": accs, "n": total}
        signal_records[bench] = bench_signals
        print(f"  Final | ", end="")
        for s in strategies:
            print(f"{s[:25]}={accs[s]:.3f} ", end="")
        print()

    return results, signal_records


# ─── 可视化 ───────────────────────────────────────────────────────────────
def plot_results(results, strategies, phase_name):
    benches = list(results.keys())
    baseline_accs = [results[b]["accuracies"]["Baseline"] for b in benches]

    fig, axes = plt.subplots(1, len(benches), figsize=(4 * len(benches), 5))
    if len(benches) == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, len(strategies)))

    for ax, bench, bl in zip(axes, benches, baseline_accs):
        for ci, s in enumerate(strategies):
            if s == "Baseline":
                continue
            acc  = results[bench]["accuracies"].get(s, 0)
            diff = acc - bl
            bar  = ax.bar(s[:15], diff, color=colors[ci], alpha=0.8)
            ax.text(bar[0].get_x() + bar[0].get_width() / 2,
                    diff + 0.001, f"{acc:.3f}", ha="center", fontsize=7)

        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set_title(bench, fontsize=11)
        ax.set_ylabel("Δ vs Baseline", fontsize=9)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.set_ylim(-0.05, 0.07)

    plt.suptitle(f"R26 {phase_name}: New Skip Strategies vs Baseline", fontsize=12, y=1.02)
    plt.tight_layout()
    path = f"{FIGURES_DIR}/r26_{phase_name.lower().replace(' ', '_')}_results.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_signal_distributions(signal_records, phase_name):
    benches = list(signal_records.keys())
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()

    metrics = [
        ("entropy_8", "Entropy@8"),
        ("entropy_slope_8", "Slope@8"),
        ("top1_prob_8", "top1_prob@8"),
        ("causal_norm_ratio", "causal_norm_ratio"),
        ("encode_bias", "encode_bias (ref)"),
    ]

    colors = plt.cm.tab10(np.linspace(0, 1, len(benches)))

    for mi, (key, label) in enumerate(metrics):
        ax = axes[mi]
        for ci, bench in enumerate(benches):
            vals = [s[key] for s in signal_records[bench] if key in s]
            if not vals:
                continue
            ax.hist(vals, bins=30, alpha=0.5, color=colors[ci],
                    label=bench, density=True)
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Density", fontsize=8)
        ax.set_title(label, fontsize=10)
        if mi == 0:
            ax.legend(fontsize=7)

    axes[-1].axis("off")
    plt.suptitle(f"R26 {phase_name}: Signal Distributions by Benchmark", fontsize=13)
    plt.tight_layout()
    path = f"{FIGURES_DIR}/r26_{phase_name.lower().replace(' ', '_')}_signals.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_skip_analysis(signal_records, results, strategies, phase_name):
    """
    散点图：top1_prob@8 vs entropy_slope@8，颜色区分各策略跳过/循环决策。
    仅用 TruthfulQA 数据。
    """
    bench = "TruthfulQA"
    if bench not in signal_records:
        return

    sigs  = signal_records[bench]
    top1  = [s["top1_prob_8"] for s in sigs]
    slope = [s["entropy_slope_8"] for s in sigs]
    eb    = [s["encode_bias"] for s in sigs]

    # EBF 跳过/应用
    ebf_skip = [1 if e > 1.0 else 0 for e in eb]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左：top1_prob vs slope，颜色 = EBF 决策
    ax = axes[0]
    colors_ebf = ["#e74c3c" if sk else "#2ecc71" for sk in ebf_skip]
    ax.scatter(slope, top1, c=colors_ebf, alpha=0.5, s=20)
    ax.axvline(-0.03, color="gray", linestyle="--", linewidth=0.8, label="flat_thresh=0.03")
    ax.axvline(0.03, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(0.7, color="orange", linestyle="--", linewidth=0.8, label="conf_thresh=0.7")
    ax.set_xlabel("entropy_slope@8", fontsize=10)
    ax.set_ylabel("top1_prob@8", fontsize=10)
    ax.set_title("TruthfulQA: EBF Decision\n(red=skip, green=loop)", fontsize=10)
    ax.legend(fontsize=8)

    # 右：causal_norm_ratio 分布
    ax = axes[1]
    cnr = [s["causal_norm_ratio"] for s in sigs]
    ax.hist(cnr, bins=40, color="#3498db", alpha=0.7, edgecolor="white")
    ax.axvline(1.2, color="red", linestyle="--", linewidth=1.5, label="norm_thresh=1.2")
    ax.axvline(1.5, color="orange", linestyle="--", linewidth=1.5, label="norm_thresh=1.5")
    ax.set_xlabel("causal_norm_ratio", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("TruthfulQA: causal_norm_ratio Dist.", fontsize=10)
    ax.legend(fontsize=8)

    plt.suptitle(f"R26 {phase_name}: Skip Signal Analysis — TruthfulQA", fontsize=12)
    plt.tight_layout()
    path = f"{FIGURES_DIR}/r26_{phase_name.lower().replace(' ', '_')}_skip_analysis.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── 主函数 ────────────────────────────────────────────────────────────────
def main():
    tok, model, lm_head, ln_f = load_model()

    benchmarks = ["BoolQ", "ARC-C", "ARC-Easy", "CSQA", "TruthfulQA"]

    # ────── Phase 1：N=200 快速筛选 ──────
    strategies_p1 = [
        "Baseline",
        "Champion",
        "EncBias-Filter",
        "S1_slope0.05_e53",
        "S3_conf07_flat003",
        "S3_conf08_flat003",
        "S3_conf07_flat005",
        "S4_norm12",
        "S4_norm15",
        "S5_compound",
    ]

    print("\n" + "=" * 60)
    print("Phase 1: N=200 Screening")
    print("=" * 60)
    t0 = time.time()
    p1_results, p1_signals = run_phase(
        tok, model, lm_head, ln_f,
        benchmarks, strategies_p1, n_per_bench=200,
        phase_name="Phase1"
    )
    print(f"\nPhase 1 done in {(time.time()-t0)/60:.1f} min")

    # 打印汇总表
    print("\n[Phase 1 Summary]")
    header = f"{'Strategy':<30}" + "".join(f"{b:>12}" for b in benchmarks) + f"{'Avg':>10}"
    print(header)
    print("-" * len(header))
    for s in strategies_p1:
        accs = [p1_results[b]["accuracies"].get(s, 0) for b in benchmarks]
        avg  = np.mean(accs)
        bl_accs = [p1_results[b]["accuracies"]["Baseline"] for b in benchmarks]
        diffs = [a - bl for a, bl in zip(accs, bl_accs)]
        cols = "".join(f"{a:.3f}({d:+.3f})"[:11].rjust(12) for a, d in zip(accs, diffs))
        print(f"{s:<30}{cols}{avg:>10.3f}")

    # 可视化
    plot_results(p1_results, strategies_p1, "Phase1")
    plot_signal_distributions(p1_signals, "Phase1")
    plot_skip_analysis(p1_signals, p1_results, strategies_p1, "Phase1")

    # 保存 Phase 1 结果
    p1_save = {
        "phase": 1, "N": 200,
        "results": {b: {"accuracies": p1_results[b]["accuracies"]} for b in benchmarks}
    }
    json.dump(p1_save, open(f"{RESULTS_DIR}/round26_phase1.json", "w"), indent=2)

    # ── 筛选 Phase 2 候选策略 ──
    bl_avgs = np.mean([p1_results[b]["accuracies"]["Baseline"] for b in benchmarks])
    s1_avg  = np.mean([p1_results[b]["accuracies"]["S1_slope0.05_e53"] for b in benchmarks])

    candidates_p2 = ["Baseline", "Champion", "EncBias-Filter", "S1_slope0.05_e53"]
    for s in strategies_p1:
        if s in candidates_p2:
            continue
        avg = np.mean([p1_results[b]["accuracies"].get(s, 0) for b in benchmarks])
        # 候选条件：平均不低于 S1 - 0.003
        if avg >= s1_avg - 0.003:
            candidates_p2.append(s)
            print(f"  ✓ {s} qualified for Phase 2 (avg={avg:.4f})")
        else:
            print(f"  ✗ {s} eliminated (avg={avg:.4f} < S1-0.003={s1_avg-0.003:.4f})")

    if len(candidates_p2) == 4:
        print("  [Info] No new strategies qualified; Phase 2 will run reference set only")

    # ────── Phase 2：N=500 大规模验证 ──────
    print("\n" + "=" * 60)
    print(f"Phase 2: N=500 Validation  |  strategies: {candidates_p2}")
    print("=" * 60)
    t1 = time.time()
    p2_results, p2_signals = run_phase(
        tok, model, lm_head, ln_f,
        benchmarks, candidates_p2, n_per_bench=500,
        phase_name="Phase2"
    )
    print(f"\nPhase 2 done in {(time.time()-t1)/60:.1f} min")

    print("\n[Phase 2 Summary]")
    for s in candidates_p2:
        accs = [p2_results[b]["accuracies"].get(s, 0) for b in benchmarks]
        avg  = np.mean(accs)
        bl_accs = [p2_results[b]["accuracies"]["Baseline"] for b in benchmarks]
        diffs = [a - bl for a, bl in zip(accs, bl_accs)]
        cols2 = "".join(f"{a:.3f}({d:+.3f})"[:11].rjust(12) for a, d in zip(accs, diffs))
        print(f"{s:<30}{cols2}{avg:>10.3f}")

    # 可视化
    plot_results(p2_results, candidates_p2, "Phase2")
    plot_signal_distributions(p2_signals, "Phase2")
    plot_skip_analysis(p2_signals, p2_results, candidates_p2, "Phase2")

    # 保存 Phase 2 结果
    p2_save = {
        "phase": 2, "N": 500,
        "results": {b: {
            "accuracies": p2_results[b]["accuracies"],
            "entropy_8_mean":  float(np.mean([s["entropy_8"] for s in p2_signals[b]])),
            "entropy_8_std":   float(np.std([s["entropy_8"] for s in p2_signals[b]])),
            "slope_8_mean":    float(np.mean([s["entropy_slope_8"] for s in p2_signals[b]])),
            "top1_prob_8_mean": float(np.mean([s["top1_prob_8"] for s in p2_signals[b]])),
            "causal_norm_ratio_mean": float(np.mean([s["causal_norm_ratio"] for s in p2_signals[b]])),
        } for b in benchmarks}
    }
    json.dump(p2_save, open(f"{RESULTS_DIR}/round26_results.json", "w"), indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/round26_results.json")

    total_min = (time.time() - t0) / 60
    print(f"Total time: {total_min:.1f} min")


if __name__ == "__main__":
    main()
