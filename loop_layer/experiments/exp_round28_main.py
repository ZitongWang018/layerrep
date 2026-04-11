"""
R28: 早期熵动态信号诊断 + 新 Skip 策略

─── R27 教训 ──────────────────────────────────────────────────────────────────
1. S6/S7/S8 skip_rate = 0.0（全 benchmark）
   根因：rank_flip_streak 与 entropy@8 高度相关（均测量收敛程度），无正交信息。

2. MMLU-HS-Math：Champion = 0.396 < Baseline = 0.407（ETD 有轻微危害）
   MMLU-Col-Math：Champion = 0.370 > Baseline = 0.340（ETD 有益）
   暗示：ETD 对"推理型"任务有效，对部分"程序性计算"任务有害。

─── R28 核心假设 ──────────────────────────────────────────────────────────────
H-R28a（早期熵动态）：
  对不同任务，entropy 从 layer4 → layer8 的变化轨迹不同：
  - 推理型（BoolQ/ARC-C）: entropy@4 高 → entropy@8 也高（信息未汇聚）→ ETD 有效
  - 计算型（MMLU-HS-Math）: entropy@4 中等 → entropy@8 上升（模型"重新搜索"）→ ETD 可能有害

  定义 entropy_drop = entropy@4 - entropy@8：
    正值 = 从 4 到 8 熵减少（收敛方向，模型锁定答案）
    负值 = 从 4 到 8 熵增加（发散方向，模型"重新考虑"）

H-R28b（浅层低熵）：
  若 entropy@4 < 4.5，模型在浅层就已快速锁定答案，ETD 没有改进空间。

H-R28c（斜率方向分叉）：
  定义 slope_46 = (entropy@6 - entropy@4) / 2（熵在 4→6 层的斜率）
       slope_68 = (entropy@8 - entropy@6) / 2（熵在 6→8 层的斜率）
  若 slope_46 > 0.1 AND slope_68 < -0.1（先升后降，∩形轨迹），
  说明模型已经"过了峰值"→ 预测已经在稳定，ETD 在 8 层后循环可能过早。

─── 新策略 ────────────────────────────────────────────────────────────────────
  S9_drop0    : entropy_drop < 0（entropy@8 > entropy@4）→ skip ETD
  S9_drop05   : entropy_drop < -0.5 → skip ETD（强发散）
  S10_e4low45 : entropy@4 < 4.5 → skip ETD（浅层已收敛）
  S10_e4low50 : entropy@4 < 5.0 → skip ETD（更保守）
  S11_vshape  : slope_46 > 0.1 AND slope_68 < -0.1 → skip（∩形）
  S12_comp    : S9_drop0 OR S10_e4low45（联合）

─── Benchmark ────────────────────────────────────────────────────────────────
  7 个：BoolQ, ARC-C, ARC-Easy, CSQA, TruthfulQA, MMLU-HS-Math, MMLU-Col-Math

Phase 1: N=200 × 7bench（快速筛选）
Phase 2: N=500 × 7bench（最终验证）
"""

import sys, os, json, time, math, warnings
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

# ─── 路径配置 ─────────────────────────────────────────────────────────────────
MODEL_PATH  = "/root/autodl-tmp/model_qwen"
RESULTS_DIR = "/root/autodl-tmp/loop_layer/experiments/results"
FIGURES_DIR = "/root/autodl-tmp/loop_layer/experiments/figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─── ETD 配置 ─────────────────────────────────────────────────────────────────
CHAMP_T_START = 8
CHAMP_T_STOP  = 22
CHAMP_K       = 2
S1_ENT_THRESH   = 5.3
S1_SLOPE_THRESH = 0.05

N_LAYERS = 36
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE    = torch.bfloat16


# ─── 模型加载 ─────────────────────────────────────────────────────────────────
def load_model():
    print("Loading model …")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=DTYPE,
        device_map="auto", trust_remote_code=True
    )
    model.eval()
    lm_head = model.lm_head
    ln_f    = model.model.norm
    return tok, model, lm_head, ln_f


# ─── 信号采集（扩展到 entropy@0-8）────────────────────────────────────────────
@torch.no_grad()
def collect_signals(model, lm_head, ln_f, input_ids):
    """
    新增 R28 信号（基于 entropy@4 的早期动态）：
      entropy_4    : logit-lens 熵@层4
      entropy_drop : entropy@4 - entropy@8
        正值 = 收敛方向（4→8 熵下降）
        负值 = 发散方向（4→8 熵上升，模型重新搜索）
      slope_46     : (entropy@6 - entropy@4) / 2
      slope_68     : (entropy@8 - entropy@6) / 2
    R27 沿用信号：entropy@8, entropy_slope@8, rank_flip_streak_8
    """
    model_dtype = next(model.parameters()).dtype

    hidden_all = []
    hooks = []
    for li in range(N_LAYERS):
        def make_hook(idx):
            def hook_fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                hidden_all.append((idx, h[:, -1, :].detach().cpu()))
            return hook_fn
        hooks.append(model.model.layers[li].register_forward_hook(make_hook(li)))

    with torch.no_grad():
        _ = model(input_ids, use_cache=False)
    for h in hooks:
        h.remove()
    hidden_all.sort(key=lambda x: x[0])
    hiddens = [h for _, h in hidden_all]

    # logit-lens 层 0..8
    entropy_arr   = []
    top1_prob_arr = []
    argmax_arr    = []
    for li in range(9):
        h_l = hiddens[li].to(DEVICE).to(model_dtype)
        with torch.no_grad():
            logits = lm_head(ln_f(h_l)).float()
        probs = F.softmax(logits, dim=-1)[0]
        ent   = -(probs * (probs + 1e-12).log()).sum().item()
        top1  = probs.max().item()
        am    = int(probs.argmax().item())
        entropy_arr.append(ent)
        top1_prob_arr.append(top1)
        argmax_arr.append(am)

    entropy_8   = entropy_arr[8]
    entropy_6   = entropy_arr[6]
    entropy_4   = entropy_arr[4]

    entropy_slope_8 = (entropy_8 - entropy_6) / 2.0   # 层 6→8 的斜率
    slope_46        = (entropy_6 - entropy_4) / 2.0   # 层 4→6 的斜率（新）
    slope_68        = (entropy_8 - entropy_6) / 2.0   # 层 6→8 的斜率（=entropy_slope_8）
    entropy_drop    = entropy_4 - entropy_8            # 正=收敛，负=发散（新）

    # rank_flip_streak_8
    streak = 0
    ref_am = argmax_arr[8]
    for li in range(7, -1, -1):
        if argmax_arr[li] == ref_am:
            streak += 1
        else:
            break

    return {
        "entropy_arr":         entropy_arr,
        "top1_prob_arr":       top1_prob_arr,
        "entropy_8":           entropy_8,
        "entropy_6":           entropy_6,
        "entropy_4":           entropy_4,
        "entropy_slope_8":     entropy_slope_8,
        "slope_46":            slope_46,
        "slope_68":            slope_68,
        "entropy_drop":        entropy_drop,
        "top1_prob_8":         top1_prob_arr[8],
        "rank_flip_streak_8":  streak,
    }


# ─── 早停 t_stop ──────────────────────────────────────────────────────────────
@torch.no_grad()
def s4_capped_tstop(model, lm_head, ln_f, input_ids, entropy_8):
    model_dtype = next(model.parameters()).dtype
    hidden_think = {}
    hooks = []
    for li in range(CHAMP_T_START, CHAMP_T_STOP + 1):
        def make_h(idx):
            def fn(m, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                hidden_think[idx] = h[:, -1, :].detach().cpu()
            return fn
        hooks.append(model.model.layers[li].register_forward_hook(make_h(li)))

    with torch.no_grad():
        _ = model(input_ids, use_cache=False)
    for h in hooks:
        h.remove()

    threshold = 0.5 * entropy_8
    for li in range(CHAMP_T_START + 4, CHAMP_T_STOP + 1):
        if li not in hidden_think:
            continue
        h_l = hidden_think[li].to(DEVICE).to(model_dtype)
        with torch.no_grad():
            logits = lm_head(ln_f(h_l)).float()
        probs = F.softmax(logits, dim=-1)[0]
        ent   = -(probs * (probs + 1e-12).log()).sum().item()
        if ent < threshold:
            return li
    return CHAMP_T_STOP


# ─── 策略决策 ─────────────────────────────────────────────────────────────────
def decide_strategy(signals, strategy_name, model=None, lm_head=None, ln_f=None, input_ids=None):
    """返回 (use_etd: bool, t_stop: int)"""
    ent8   = signals["entropy_8"]
    ent6   = signals["entropy_6"]
    ent4   = signals["entropy_4"]
    slope8 = signals["entropy_slope_8"]
    edrop  = signals["entropy_drop"]      # entropy@4 - entropy@8
    s46    = signals["slope_46"]
    s68    = signals["slope_68"]

    s1_fires = (ent8 > S1_ENT_THRESH) or (slope8 > S1_SLOPE_THRESH)

    def s1_tstop():
        if s1_fires:
            return True, CHAMP_T_STOP
        else:
            t = s4_capped_tstop(model, lm_head, ln_f, input_ids, ent8)
            return True, t

    if strategy_name == "Baseline":
        return False, CHAMP_T_STOP

    elif strategy_name == "Champion":
        return True, CHAMP_T_STOP

    elif strategy_name == "S1_slope0.05_e53":
        return s1_tstop()

    # ── R28 新策略 ─────────────────────────────────────────────────────────────
    elif strategy_name == "S9_drop0":
        # 若 entropy@8 > entropy@4（发散）→ skip ETD
        skip = (edrop < 0.0)
        if skip:
            return False, CHAMP_T_STOP
        return s1_tstop()

    elif strategy_name == "S9_drop05":
        # 强发散：entropy@8 > entropy@4 + 0.5
        skip = (edrop < -0.5)
        if skip:
            return False, CHAMP_T_STOP
        return s1_tstop()

    elif strategy_name == "S10_e4low45":
        # 浅层已收敛：entropy@4 < 4.5
        skip = (ent4 < 4.5)
        if skip:
            return False, CHAMP_T_STOP
        return s1_tstop()

    elif strategy_name == "S10_e4low50":
        # 保守版：entropy@4 < 5.0
        skip = (ent4 < 5.0)
        if skip:
            return False, CHAMP_T_STOP
        return s1_tstop()

    elif strategy_name == "S11_vshape":
        # ∩ 形轨迹：先升后降（slope_46 > 0.1 AND slope_68 < -0.1）
        skip = (s46 > 0.1) and (s68 < -0.1)
        if skip:
            return False, CHAMP_T_STOP
        return s1_tstop()

    elif strategy_name == "S12_comp":
        # S9_drop0 OR S10_e4low45
        skip = (edrop < 0.0) or (ent4 < 4.5)
        if skip:
            return False, CHAMP_T_STOP
        return s1_tstop()

    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


# ─── 数据集加载 ───────────────────────────────────────────────────────────────
def load_boolq(n):
    ds = load_dataset("aps/super_glue", "boolq")["validation"]
    samples = []
    for x in ds:
        lab = int(x["label"])
        if lab < 0:
            continue
        # 与 R27 对齐：使用 ["no", "yes"]（tokenizer 对 no/yes 有更好的边界处理）
        prompt = f"{x['passage']}\nQuestion: {x['question']}?\nAnswer:"
        choices = ["no", "yes"]
        samples.append({"prompt": prompt, "choices": choices, "label": lab})
        if len(samples) >= n:
            break
    return samples

def load_arc(split_name, n):
    ds = load_dataset("allenai/ai2_arc", split_name)["test"]
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
    label_map = {"A":0, "B":1, "C":2, "D":3, "E":4}
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

def load_mmlu_math(subset, n):
    try:
        ds = load_dataset("cais/mmlu", subset)["test"]
        samples = []
        for x in ds:
            choices = x["choices"]
            label   = int(x["answer"])
            prompt  = f"Question: {x['question']}\nAnswer:"
            samples.append({"prompt": prompt, "choices": choices, "label": label})
            if len(samples) >= n:
                break
        return samples
    except Exception as e:
        print(f"  [WARN] MMLU/{subset} unavailable: {e}")
        return None


# ─── 样本评估 ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_mc_choice(tok, prompt_str, model, choices, use_etd, t_stop):
    """
    正确评分：对每个 choice 拼接 prompt + " " + choice 为完整序列，
    用 loglikelihood_continuation(logits, full_ids, prompt_len) 计算对数似然。
    
    关键：选项前必须加空格（" " + ch），否则 Qwen3 tokenizer 会将冒号与选项
    合并成同一 token，导致续写 token 数为 0，所有选项 LL = 0。
    
    etd_forward_logits API：(model, input_ids, attn_mask, n_e, n_t, k, alpha)
    input_ids = full prompt + " " + choice token ids
    """
    n_e   = CHAMP_T_START
    n_t   = t_stop - n_e
    alpha = min(1.0, 6.0 / max(n_t, 1))

    # 预先计算 prompt 长度（prompt + " " 前缀的 token 数）
    pref_ids = tok(prompt_str, return_tensors="pt", add_special_tokens=False)["input_ids"]
    prompt_len = pref_ids.shape[1]

    scores = []
    for ch in choices:
        # 选项前加空格，确保 tokenizer 正确分割（对齐语言模型风格）
        full_str = prompt_str + " " + ch
        enc = tok(full_str, return_tensors="pt", add_special_tokens=False)
        full_ids = enc["input_ids"].to(DEVICE)
        attn = enc.get("attention_mask")
        attn_mask = attn.to(DEVICE) if attn is not None else None

        if use_etd and n_t >= 1:
            logits = etd_forward_logits(
                model, full_ids, attn_mask,
                n_e=n_e, n_t=n_t, k=CHAMP_K, alpha=alpha
            )
        else:
            logits = baseline_forward_logits(model, full_ids, attn_mask)

        ll = loglikelihood_continuation(logits, full_ids, prompt_len)
        scores.append(ll)
    return int(np.argmax(scores))


@torch.no_grad()
def eval_sample(tok, model, lm_head, ln_f, sample, strategies, signal_store):
    """
    对一个样本跑所有策略，返回 {strategy: correct (0/1)}
    同时将 signals 追加到 signal_store[bench_name] 列表中（若提供）
    """
    input_ids = tok.encode(sample["prompt"], return_tensors="pt").to(DEVICE)
    signals   = collect_signals(model, lm_head, ln_f, input_ids)

    # 收集信号（用于后续分析）
    if signal_store is not None:
        signal_store.append({
            "entropy_8":    signals["entropy_8"],
            "entropy_6":    signals["entropy_6"],
            "entropy_4":    signals["entropy_4"],
            "entropy_drop": signals["entropy_drop"],
            "slope_46":     signals["slope_46"],
            "slope_68":     signals["slope_68"],
            "entropy_slope_8": signals["entropy_slope_8"],
            "streak":       signals["rank_flip_streak_8"],
        })

    results = {}
    for strat in strategies:
        use_etd, t_stop = decide_strategy(
            signals, strat, model, lm_head, ln_f, input_ids
        )
        # predict_mc_choice 使用 prompt_str（字符串），内部拼接 prompt+choice
        pred = predict_mc_choice(
            tok, sample["prompt"], model,
            sample["choices"], use_etd, t_stop
        )
        results[strat] = int(pred == sample["label"])
    return results


# ─── 主评估循环 ───────────────────────────────────────────────────────────────
def run_benchmark(tok, model, lm_head, ln_f, bench_name, samples, strategies,
                  print_interval=50, signal_store=None):
    n = len(samples)
    accs = {s: 0.0 for s in strategies}
    counts = {s: 0 for s in strategies}

    t0 = time.time()
    for i, sample in enumerate(samples):
        res = eval_sample(tok, model, lm_head, ln_f, sample, strategies, signal_store)
        for s in strategies:
            counts[s] += res[s]

        if (i + 1) % print_interval == 0 or (i + 1) == n:
            parts = []
            for s in strategies:
                a = counts[s] / (i + 1)
                d = a - counts["Baseline"] / (i + 1)
                parts.append(f"{s.replace('S1_slope0.05_e53', '1_slope0.05_e53')}={a:.3f}({d:+.3f})")
            print(f"  [{i+1}/{n}] " + " ".join(parts))

    final = {s: counts[s] / n for s in strategies}
    final_str = " ".join(
        f"{s.replace('S1_slope0.05_e53', '1_slope0.05_e53')}={final[s]:.3f}"
        for s in strategies
    )
    print(f"  Final | {final_str}")
    elapsed = (time.time() - t0) / 60
    return final, elapsed


# ─── 可视化 ───────────────────────────────────────────────────────────────────
def plot_results(results_dict, phase, strategies, benchmarks, fname):
    """绘制各策略在各 benchmark 上的精度条形图，以 Baseline 为参考线"""
    n_bench = len(benchmarks)
    x = np.arange(n_bench)
    width = 0.8 / max(len(strategies) - 1, 1)

    fig, ax = plt.subplots(figsize=(max(12, n_bench * 2), 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))

    bl = [results_dict[b]["Baseline"] for b in benchmarks]
    ax.bar(x, bl, width=0.8, alpha=0.15, color="gray", label="Baseline (ref)")

    non_bl = [s for s in strategies if s != "Baseline"]
    for si, strat in enumerate(non_bl):
        vals = [results_dict[b].get(strat, 0) for b in benchmarks]
        offs = (si - len(non_bl) / 2 + 0.5) * width
        ax.bar(x + offs, vals, width=width * 0.85,
               alpha=0.8, color=colors[si], label=strat[:18])

    ax.set_xticks(x)
    ax.set_xticklabels([b.replace("MMLU-", "M-") for b in benchmarks], fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"R28 Phase {phase} Results")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"  Saved: {fname}")


def plot_signal_distributions(all_signals, benchmarks, fname):
    """绘制各 benchmark 的 entropy_4, entropy_8, entropy_drop 分布"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    keys  = ["entropy_4", "entropy_8", "entropy_drop"]
    titles = ["Entropy@layer4", "Entropy@layer8", "Entropy Drop (4-8)"]
    colors = plt.cm.Set2(np.linspace(0, 1, len(benchmarks)))

    for ax, key, title in zip(axes, keys, titles):
        for bi, bench in enumerate(benchmarks):
            if bench not in all_signals or len(all_signals[bench]) == 0:
                continue
            vals = [s[key] for s in all_signals[bench]]
            ax.hist(vals, bins=30, alpha=0.5, label=bench[:10],
                    color=colors[bi], density=True)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Value")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"  Saved: {fname}")


def plot_skip_rates(skip_info, benchmarks, strategies, fname):
    """绘制各策略在各 benchmark 的 skip 率"""
    x = np.arange(len(benchmarks))
    width = 0.8 / max(len(strategies), 1)
    fig, ax = plt.subplots(figsize=(max(10, len(benchmarks) * 2), 4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
    for si, strat in enumerate(strategies):
        rates = [skip_info.get(strat, {}).get(b, 0.0) for b in benchmarks]
        offs  = (si - len(strategies) / 2 + 0.5) * width
        ax.bar(x + offs, rates, width=width * 0.85,
               alpha=0.8, color=colors[si], label=strat[:16])
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace("MMLU-", "M-") for b in benchmarks], fontsize=9)
    ax.set_ylabel("Skip Rate (using Baseline)")
    ax.set_title("Skip Rate by Strategy and Benchmark")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"  Saved: {fname}")


def plot_entropy_scatter(all_signals, bench_results_bl, bench_results_champ, benchmarks, fname):
    """
    散点图：x=entropy_drop, y=entropy@8，颜色=Champion-Baseline per sample
    (仅用 Phase1 的样本，需要 per-sample correctness - 这里用 benchmark 级别总结)
    对每个 benchmark 画 entropy_4 vs entropy_8 的分布，附上该 benchmark 的 Champion-Baseline 差值
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for bi, bench in enumerate(benchmarks):
        ax = axes[bi]
        if bench not in all_signals or len(all_signals[bench]) == 0:
            ax.set_visible(False)
            continue
        e4   = [s["entropy_4"] for s in all_signals[bench]]
        e8   = [s["entropy_8"] for s in all_signals[bench]]
        drop = [s["entropy_drop"] for s in all_signals[bench]]
        diff = bench_results_champ.get(bench, 0) - bench_results_bl.get(bench, 0)
        color = "green" if diff >= 0 else "red"
        ax.scatter(drop, e8, alpha=0.3, s=8, color=color)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("entropy_drop (pos=converge)", fontsize=8)
        ax.set_ylabel("entropy@8", fontsize=8)
        ax.set_title(f"{bench[:12]}\nΔ={diff:+.3f}", fontsize=9)
    if len(benchmarks) < len(axes):
        for ax in axes[len(benchmarks):]:
            ax.set_visible(False)
    plt.suptitle("Entropy Drop vs Entropy@8 per Benchmark", fontsize=11)
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"  Saved: {fname}")


# ─── 主流程 ───────────────────────────────────────────────────────────────────
def main():
    tok, model, lm_head, ln_f = load_model()

    # 所有策略（Phase1 全测，Phase2 全测）
    ALL_STRATEGIES = [
        "Baseline", "Champion", "S1_slope0.05_e53",
        "S9_drop0", "S9_drop05",
        "S10_e4low45", "S10_e4low50",
        "S11_vshape", "S12_comp",
    ]

    BENCHMARKS = [
        "BoolQ", "ARC-C", "ARC-Easy", "CSQA",
        "TruthfulQA", "MMLU-HS-Math", "MMLU-Col-Math",
    ]

    # ── Phase 1: N=200 ────────────────────────────────────────────────────────
    N1 = 200
    print(f"\n{'='*65}")
    print(f"Phase 1: N={N1} Screening")
    print(f"{'='*65}")

    def load_all(n):
        d = {}
        d["BoolQ"]        = load_boolq(n)
        d["ARC-C"]        = load_arc("ARC-Challenge", n)
        d["ARC-Easy"]     = load_arc("ARC-Easy", n)
        d["CSQA"]         = load_csqa(n)
        d["TruthfulQA"]   = load_truthfulqa(n)
        d["MMLU-HS-Math"] = load_mmlu_math("high_school_mathematics", n)
        d["MMLU-Col-Math"]= load_mmlu_math("college_mathematics", n)
        return d

    data1 = load_all(N1)
    for b, v in data1.items():
        if v is not None:
            print(f"  Loaded {b}: {len(v)} examples")
        else:
            print(f"  Skipped {b} (unavailable)")

    p1_results  = {}
    p1_signals  = {b: [] for b in BENCHMARKS}
    skip_counts = {s: {b: 0 for b in BENCHMARKS} for s in ALL_STRATEGIES}
    total_counts = {b: 0 for b in BENCHMARKS}
    t_p1_start  = time.time()

    for bench in BENCHMARKS:
        samples = data1.get(bench)
        if samples is None:
            continue
        print(f"\n[Phase1] {bench} (N={len(samples)})")
        final, _ = run_benchmark(
            tok, model, lm_head, ln_f,
            bench, samples, ALL_STRATEGIES,
            signal_store=p1_signals[bench]
        )
        p1_results[bench] = final
        total_counts[bench] = len(samples)

    t_p1_min = (time.time() - t_p1_start) / 60
    print(f"\nPhase 1 done in {t_p1_min:.1f} min")

    # Phase 1 汇总
    print(f"\n[Phase1 Summary]")
    header  = f"{'Strategy':<30}"
    for b in BENCHMARKS:
        if b in p1_results:
            header += f"{b[:12]:>13}"
    header += f"{'Avg':>10}"
    print(header)
    print("-" * len(header))

    p1_avgs = {}
    for strat in ALL_STRATEGIES:
        row = f"{strat:<30}"
        vals = []
        for b in BENCHMARKS:
            if b not in p1_results:
                continue
            a = p1_results[b].get(strat, 0)
            d = a - p1_results[b].get("Baseline", 0)
            s = f"{a:.3f}({d:+.3f})"
            row += f"{s[:11]:>13}"
            vals.append(a)
        avg = np.mean(vals) if vals else 0.0
        p1_avgs[strat] = avg
        row += f"{avg:>10.3f}"
        print(row)

    # 保存 phase1 结果
    valid_benches = [b for b in BENCHMARKS if b in p1_results]
    json.dump({
        "phase": 1, "N": N1,
        "results": {b: p1_results[b] for b in valid_benches},
        "signal_stats": {
            b: {
                "entropy_4_mean":   float(np.mean([s["entropy_4"]    for s in p1_signals[b]])) if p1_signals[b] else 0,
                "entropy_8_mean":   float(np.mean([s["entropy_8"]    for s in p1_signals[b]])) if p1_signals[b] else 0,
                "entropy_drop_mean":float(np.mean([s["entropy_drop"] for s in p1_signals[b]])) if p1_signals[b] else 0,
                "slope_46_mean":    float(np.mean([s["slope_46"]     for s in p1_signals[b]])) if p1_signals[b] else 0,
                "slope_68_mean":    float(np.mean([s["slope_68"]     for s in p1_signals[b]])) if p1_signals[b] else 0,
                "n":                len(p1_signals[b]),
            }
            for b in BENCHMARKS
        },
    }, open(f"{RESULTS_DIR}/round28_phase1.json", "w"), indent=2)

    # 图：Phase1 结果
    plot_results(
        {b: p1_results[b] for b in valid_benches},
        "1", ALL_STRATEGIES, valid_benches,
        f"{FIGURES_DIR}/r28_phase1_results.png"
    )
    # 图：信号分布
    plot_signal_distributions(
        p1_signals, valid_benches,
        f"{FIGURES_DIR}/r28_phase1_signal_dist.png"
    )
    # 图：entropy scatter per bench
    bl_vals  = {b: p1_results[b].get("Baseline", 0) for b in valid_benches}
    champ_vals = {b: p1_results[b].get("Champion", 0) for b in valid_benches}
    plot_entropy_scatter(
        p1_signals, bl_vals, champ_vals, valid_benches,
        f"{FIGURES_DIR}/r28_phase1_entropy_scatter.png"
    )

    # 打印信号统计（帮助理解 skip 行为）
    print("\n[Signal Statistics per Benchmark (Phase1)]")
    print(f"{'Benchmark':<16}{'ent@4':>8}{'ent@8':>8}{'ent_drop':>10}{'slope_46':>10}{'slope_68':>10}")
    print("-" * 62)
    for b in valid_benches:
        if not p1_signals[b]:
            continue
        e4   = np.mean([s["entropy_4"]    for s in p1_signals[b]])
        e8   = np.mean([s["entropy_8"]    for s in p1_signals[b]])
        drop = np.mean([s["entropy_drop"] for s in p1_signals[b]])
        s46  = np.mean([s["slope_46"]     for s in p1_signals[b]])
        s68  = np.mean([s["slope_68"]     for s in p1_signals[b]])
        print(f"{b:<16}{e4:>8.3f}{e8:>8.3f}{drop:>10.3f}{s46:>10.3f}{s68:>10.3f}")

    # ── Phase 2: N=500 ────────────────────────────────────────────────────────
    N2 = 500
    print(f"\n{'='*65}")
    print(f"Phase 2: N={N2} | strategies: {ALL_STRATEGIES}")
    print(f"{'='*65}")

    data2 = load_all(N2)
    for b, v in data2.items():
        if v is not None:
            print(f"  Loaded {b}: {len(v)} examples")

    p2_results = {}
    p2_signals = {b: [] for b in BENCHMARKS}
    t_p2_start = time.time()

    for bench in BENCHMARKS:
        samples = data2.get(bench)
        if samples is None:
            continue
        print(f"\n[Phase2] {bench} (N={len(samples)})")
        final, _ = run_benchmark(
            tok, model, lm_head, ln_f,
            bench, samples, ALL_STRATEGIES,
            signal_store=p2_signals[bench]
        )
        p2_results[bench] = final

    t_p2_min = (time.time() - t_p2_start) / 60
    print(f"\nPhase 2 done in {t_p2_min:.1f} min")

    # Phase 2 汇总
    print(f"\n[Phase2 Summary]")
    valid_b2 = [b for b in BENCHMARKS if b in p2_results]
    header2  = f"{'Strategy':<30}"
    for b in valid_b2:
        header2 += f"{b[:12]:>13}"
    header2 += f"{'Avg':>10}"
    print(header2)
    print("-" * len(header2))

    p2_avgs = {}
    best_strat, best_avg = None, -1
    for strat in ALL_STRATEGIES:
        row  = f"{strat:<30}"
        vals = []
        for b in valid_b2:
            a = p2_results[b].get(strat, 0)
            d = a - p2_results[b].get("Baseline", 0)
            s = f"{a:.3f}({d:+.3f})"
            row += f"{s[:11]:>13}"
            vals.append(a)
        avg = np.mean(vals) if vals else 0.0
        p2_avgs[strat] = avg
        row += f"{avg:>10.3f}"
        print(row)
        if strat != "Baseline" and avg > best_avg:
            best_avg, best_strat = avg, strat

    print(f"\n  Best strategy: {best_strat} (avg={best_avg:.4f})")

    # 保存最终结果
    json.dump({
        "phase": 2, "N": N2,
        "results": {b: p2_results[b] for b in valid_b2},
        "final_avgs": p2_avgs,
        "signal_stats": {
            b: {
                "entropy_4_mean":    float(np.mean([s["entropy_4"]    for s in p2_signals[b]])) if p2_signals[b] else 0,
                "entropy_8_mean":    float(np.mean([s["entropy_8"]    for s in p2_signals[b]])) if p2_signals[b] else 0,
                "entropy_drop_mean": float(np.mean([s["entropy_drop"] for s in p2_signals[b]])) if p2_signals[b] else 0,
                "entropy_drop_std":  float(np.std( [s["entropy_drop"] for s in p2_signals[b]])) if p2_signals[b] else 0,
                "pct_diverging":     float(np.mean([1 if s["entropy_drop"] < 0 else 0 for s in p2_signals[b]])) if p2_signals[b] else 0,
                "slope_46_mean":     float(np.mean([s["slope_46"]     for s in p2_signals[b]])) if p2_signals[b] else 0,
                "slope_68_mean":     float(np.mean([s["slope_68"]     for s in p2_signals[b]])) if p2_signals[b] else 0,
                "n":                 len(p2_signals[b]),
            }
            for b in BENCHMARKS
        },
        "best_strategy": best_strat,
        "best_avg": best_avg,
    }, open(f"{RESULTS_DIR}/round28_results.json", "w"), indent=2)

    # 图：Phase2 结果
    plot_results(
        {b: p2_results[b] for b in valid_b2},
        "2", ALL_STRATEGIES, valid_b2,
        f"{FIGURES_DIR}/r28_phase2_results.png"
    )
    plot_signal_distributions(
        p2_signals, valid_b2,
        f"{FIGURES_DIR}/r28_phase2_signal_dist.png"
    )
    bl2_vals    = {b: p2_results[b].get("Baseline", 0) for b in valid_b2}
    champ2_vals = {b: p2_results[b].get("Champion", 0) for b in valid_b2}
    plot_entropy_scatter(
        p2_signals, bl2_vals, champ2_vals, valid_b2,
        f"{FIGURES_DIR}/r28_phase2_entropy_scatter.png"
    )

    total_min = (time.time() - t_p1_start) / 60
    print(f"\nTotal: {total_min:.1f} min")
    print(f"Results saved to {RESULTS_DIR}/round28_results.json")


if __name__ == "__main__":
    main()
