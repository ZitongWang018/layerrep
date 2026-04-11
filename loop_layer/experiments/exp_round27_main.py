"""
R27: 修正评分方法 + MMLU 数学子集 + 新Skip信号

─── R26 教训 ─────────────────────────────────────────────
1. 评分 Bug：R26 对 [prompt + choice] 取末尾 logit，实际应对
   [prompt] 末尾 logit 评选项（单 token），或用完整 log-likelihood
   ∑ log P(choice_token_i | prompt + choice_tokens[:i])。
   修复：直接使用项目已有的 predict_mc_choice / loglikelihood_continuation。

2. encode_bias Bug：R26 把 norm_delta 比值当作 encode_bias，导致
   EncBias-Filter = Baseline。修复：encode_bias 仍非因果（需两次前向），
   仅保留原始 ETD 模块的 predict_mc_choice 路径对齐历史结果。

3. top1_prob@8 从不超过 0.7：层8的 logit-lens 熵均值约 5.4，对应
   top1_prob ≈ 0.004-0.05，远低于阈值。S3 策略不可行。

─── R27 新方向 ────────────────────────────────────────────
核心问题：S1_slope0.05_e53 在 TruthfulQA (0.292) 上比 EBF (0.296) 低。
EBF 通过 encode_bias 正确跳过部分 TruthfulQA 样本，而 S1 过度循环（91.2%触发）。

R27 假设：TruthfulQA 独特特征 = "平坦斜率 + 中等熵 + 低 rank_flip_streak"
（模型在 8 层后不再改变预测，不需要更多循环）

新信号（层 6/8 可因果获取）：
  rank_flip_streak_8: 在 logit-lens 中，层0→8 连续 argmax 不变的层数。
    若 streak_8 >= 4（模型从层 4 起未改变预测），ETD 可能无效。
  entropy_delta_68: entropy@8 - entropy@6（已有，即 2*slope）
  entropy_at_6: 较早时刻的熵

策略（均在 S1_slope0.05_e53 基础上添加 skip）：
  S6_streak4: rank_flip_streak_8 >= 4 → skip ETD
  S6_streak3: rank_flip_streak_8 >= 3 → skip ETD
  S7_e6low05: entropy@6 < 5.0 AND |slope_8| < 0.05 → skip
              (如果熵早已很低且不上升，说明模型早收敛)
  S8_compound: S6_streak4 OR S7_e6low05 → skip

新 Benchmark：
  MMLU-HS-Math (cais/mmlu, high_school_mathematics): 270题 4选1
  MMLU-College-Math (cais/mmlu, college_mathematics): 100题 4选1
  注意：若数据集未缓存则跳过，不影响其他 benchmark。

─── 评分修正 ───────────────────────────────────────────────
使用 ETD 模块的 predict_mc_choice + loglikelihood_continuation：
  score(choice) = ∑_i log P(choice_token_i | prompt + choice_tokens[:i])

Phase 1: N=200 × 5+2 benchmarks（快速筛选）
Phase 2: N=500 × 所有 benchmark（最优策略）
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
from etd_forward import (  # noqa: E402
    etd_forward_logits,
    baseline_forward_logits,
    loglikelihood_continuation,
)

# ─── 路径配置 ───────────────────────────────────────────────────────────────
MODEL_PATH  = "/root/autodl-tmp/model_qwen"
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

N_LAYERS = 36
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE    = torch.bfloat16


# ─── 模型加载 ──────────────────────────────────────────────────────────────
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


# ─── 信号采集 ──────────────────────────────────────────────────────────────
@torch.no_grad()
def collect_signals(model, lm_head, ln_f, input_ids):
    """
    因果信号（仅用层 0..8 信息）：
      entropy_arr[l], top1_prob_arr[l]: logit-lens 指标 (l=0..8)
      rank_flip_streak_8: 从第 8 层往前数，argmax 连续不变的层数
      entropy_8, entropy_slope_8 (= (entropy@8 - entropy@6)/2)
    """
    model_dtype = next(model.parameters()).dtype

    hidden_all = []
    hooks = []
    for li in range(N_LAYERS):
        def make_hook(idx):
            def hook_fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                hidden_all.append((idx, h[:, -1, :].detach().cpu()))  # 只取最后token
            return hook_fn
        hooks.append(model.model.layers[li].register_forward_hook(make_hook(li)))

    with torch.no_grad():
        _ = model(input_ids, use_cache=False)
    for h in hooks:
        h.remove()
    hidden_all.sort(key=lambda x: x[0])
    hiddens = [h for _, h in hidden_all]  # list[N_LAYERS], each (1, dim)

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

    entropy_8     = entropy_arr[8]
    entropy_6     = entropy_arr[6]
    entropy_slope_8 = (entropy_8 - entropy_6) / 2.0
    top1_prob_8   = top1_prob_arr[8]

    # rank_flip_streak_8: 从第 8 层往前数，argmax 与 layer 8 相同的连续层数
    # 表示模型"预测未变"持续了多少层（>= 1 意味着 7 和 8 一致）
    streak = 0
    ref_am = argmax_arr[8]
    for li in range(7, -1, -1):  # 7, 6, 5, ..., 0
        if argmax_arr[li] == ref_am:
            streak += 1
        else:
            break
    rank_flip_streak_8 = streak  # 0..8，数值越大说明预测越早稳定

    return {
        "entropy_arr":         entropy_arr,
        "top1_prob_arr":       top1_prob_arr,
        "entropy_8":           entropy_8,
        "entropy_6":           entropy_6,
        "entropy_slope_8":     entropy_slope_8,
        "top1_prob_8":         top1_prob_8,
        "rank_flip_streak_8":  rank_flip_streak_8,
    }


# ─── 早停 t_stop 搜索 ─────────────────────────────────────────────────────
@torch.no_grad()
def s4_capped_tstop(model, lm_head, ln_f, input_ids, entropy_8):
    """
    早停搜索：从 t_start+4 到 CHAMP_T_STOP，
    entropy[l] < 0.5*entropy_8 时停止（不超过 CHAMP_T_STOP=22）。
    使用 hook + 全量前向（正确获取每层 hidden state）。
    """
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
        h_l   = hidden_think[li].to(DEVICE).to(model_dtype)
        with torch.no_grad():
            logits = lm_head(ln_f(h_l)).float()
        probs = F.softmax(logits, dim=-1)[0]
        ent   = -(probs * (probs + 1e-12).log()).sum().item()
        if ent < threshold:
            return li
    return CHAMP_T_STOP


# ─── 策略决策 ─────────────────────────────────────────────────────────────
def decide_strategy(signals, strategy_name, model=None, lm_head=None, ln_f=None, input_ids=None):
    """
    返回 (use_etd: bool, t_stop: int)
    use_etd=False → 使用 Baseline（不循环）
    """
    ent8   = signals["entropy_8"]
    ent6   = signals["entropy_6"]
    slope8 = signals["entropy_slope_8"]
    streak = signals["rank_flip_streak_8"]

    # S1 基础门控
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

    # ── 新策略（均在 S1 基础上增加 skip 条件）──
    elif strategy_name == "S6_streak4":
        # 若层 8 的 argmax 已连续 >=4 层不变，说明预测已稳定，skip ETD
        if streak >= 4:
            return False, CHAMP_T_STOP
        return s1_tstop()

    elif strategy_name == "S6_streak3":
        if streak >= 3:
            return False, CHAMP_T_STOP
        return s1_tstop()

    elif strategy_name == "S7_e6low05":
        # 如果熵在层6就已经 < 5.0 且斜率不上升，模型在第6层已基本收敛
        if (ent6 < 5.0) and (abs(slope8) < 0.05):
            return False, CHAMP_T_STOP
        return s1_tstop()

    elif strategy_name == "S8_compound":
        # S6_streak4 OR S7_e6low05
        skip = (streak >= 4) or ((ent6 < 5.0) and (abs(slope8) < 0.05))
        if skip:
            return False, CHAMP_T_STOP
        return s1_tstop()

    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


# ─── 正确的 MCQ 评分 ─────────────────────────────────────────────────────
@torch.no_grad()
def score_choices_mc(tok, model, lm_head, ln_f, prefix_str, choices, strategy_name, signals, model_dtype):
    """
    用 loglikelihood_continuation 正确评分：
      score(choice) = ∑_i log P(choice_token_i | prompt + choice_tokens[:i])
    """
    # 确定 ETD 参数
    use_etd, t_stop = decide_strategy(signals, strategy_name, model, lm_head, ln_f, None)
    n_e   = CHAMP_T_START
    n_t   = t_stop - n_e
    alpha = min(1.0, 6.0 / max(n_t, 1))

    scores = []
    pref_enc   = tok(prefix_str, return_tensors="pt", add_special_tokens=False)
    prompt_len = pref_enc["input_ids"].shape[1]

    for ch in choices:
        full_str   = prefix_str + ch
        full_enc   = tok(full_str, return_tensors="pt", add_special_tokens=False)
        input_ids  = full_enc["input_ids"].to(DEVICE)
        attn_mask  = full_enc.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(DEVICE)

        if not use_etd or n_t < 1:
            logits = baseline_forward_logits(model, input_ids, attn_mask)
        else:
            logits = etd_forward_logits(
                model, input_ids, attn_mask,
                n_e=n_e, n_t=n_t, k=CHAMP_K, alpha=alpha
            )
        # logit shape: (1, seq_len, vocab_size)
        scores.append(loglikelihood_continuation(logits, input_ids, prompt_len))

    return scores


# ─── 单样本评估 ────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_sample(tok, model, lm_head, ln_f, example, strategies):
    """
    返回 {strategy_name: correct(bool), ...}, signals
    """
    prefix_str = example["prompt"]
    choices    = example["choices"]         # 裸文本（不带前缀空格）
    answer_key = example["answer"].strip()

    # 采集因果信号
    pref_enc  = tok(prefix_str, return_tensors="pt", add_special_tokens=False)
    input_ids = pref_enc["input_ids"].to(DEVICE)
    signals   = collect_signals(model, lm_head, ln_f, input_ids)

    model_dtype = next(model.parameters()).dtype

    # 构造带空格前缀的选项（对齐语言模型风格）
    choices_with_space = [(" " + ch) for ch in choices]

    results = {}
    for strat in strategies:
        sc = score_choices_mc(
            tok, model, lm_head, ln_f,
            prefix_str, choices_with_space, strat, signals, model_dtype
        )
        pred    = choices[int(np.argmax(sc))]
        correct = (pred.strip().lower() == answer_key.strip().lower())
        results[strat] = correct

    return results, signals


# ─── 数据加载 ──────────────────────────────────────────────────────────────
def _fmt(prefix, conts, label):
    """三元组 → R27 字典格式"""
    choices_str = [c.strip() for c in conts]
    return {"prompt": prefix, "choices": choices_str, "answer": choices_str[label]}


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
        out.append(_fmt(prefix, ["no", "yes"], lab))
    return out


def load_arc(subset, n):
    ds  = load_dataset("allenai/ai2_arc", subset)["test"]
    out = []
    for r in ds:
        if len(out) >= n:
            break
        q     = r["question"].strip()
        texts = r["choices"]["text"]
        key   = r["answerKey"]
        label = ord(key) - ord("A") if key in "ABCD" else int(key) - 1
        out.append(_fmt(f"Question: {q}\nAnswer:", texts, label))
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
        out.append(_fmt(
            f"Question: {r['question'].strip()}\nAnswer:",
            r["choices"]["text"], lmap[key]
        ))
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
        out.append(_fmt(
            f"Question: {r['question'].strip()}\nAnswer:",
            mc1["choices"], label
        ))
    return out


def load_mmlu_math(subset, n):
    """
    MMLU 数学子集（需 cais/mmlu 已缓存）。
    subset: 'high_school_mathematics' / 'college_mathematics' / 'elementary_mathematics'
    格式: question + 4 choices ABCD，answer 是整数 0-3
    数据集现已缓存（通过 hf-mirror.com 下载）。
    """
    ds = load_dataset("cais/mmlu", subset)["test"]
    out = []
    lmap = {0: "A", 1: "B", 2: "C", 3: "D"}
    for r in ds:
        if len(out) >= n:
            break
        q       = r["question"].strip()
        choices = r["choices"]  # list of 4 strings
        label   = int(r["answer"])  # 0-3
        # 格式同 ARC
        out.append(_fmt(f"Question: {q}\nAnswer:", choices, label))
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
    elif name == "MMLU-HS-Math":
        return load_mmlu_math("high_school_mathematics", n)
    elif name == "MMLU-Col-Math":
        return load_mmlu_math("college_mathematics", n)
    else:
        raise ValueError(f"Unknown benchmark: {name}")


# ─── 多 Benchmark 加载（跳过不可用的）─────────────────────────────────────
def load_all_benchmarks(benchmark_list, n):
    """返回 {bench_name: [examples], ...}（不可用的跳过）"""
    data = {}
    for bench in benchmark_list:
        try:
            examples = load_benchmark(bench, n)
            data[bench] = examples
            print(f"  Loaded {bench}: {len(examples)} examples")
        except Exception as e:
            print(f"  SKIP {bench}: {str(e)[:80]}")
    return data


# ─── 运行一轮评估 ─────────────────────────────────────────────────────────
def run_phase(tok, model, lm_head, ln_f, benchmark_data, strategies, phase_name):
    results      = {}
    signal_recs  = {}

    for bench, examples in benchmark_data.items():
        n_per_bench = len(examples)
        print(f"\n[{phase_name}] {bench} (N={n_per_bench})")
        correct = {s: 0 for s in strategies}
        total   = 0
        sigs    = []

        for i, ex in enumerate(examples):
            try:
                res, sig = eval_sample(tok, model, lm_head, ln_f, ex, strategies)
            except Exception as e:
                print(f"  [WARN] sample {i} error: {e}")
                continue
            for s in strategies:
                correct[s] += int(res.get(s, False))
            sigs.append(sig)
            total += 1

            if (i + 1) % 50 == 0:
                accs = {s: correct[s] / max(total, 1) for s in strategies}
                bl   = accs["Baseline"]
                print(f"  [{i+1}/{n_per_bench}] ", end="")
                for s in strategies[:6]:  # 只打前6个
                    print(f"{s[-15:]}={accs[s]:.3f}({accs[s]-bl:+.3f}) ", end="")
                print()

        accs = {s: correct[s] / max(total, 1) for s in strategies}
        results[bench]     = {"accuracies": accs, "n": total}
        signal_recs[bench] = sigs
        print(f"  Final | ", end="")
        for s in strategies:
            print(f"{s[-15:]}={accs[s]:.3f} ", end="")
        print()

    return results, signal_recs


# ─── 可视化 ───────────────────────────────────────────────────────────────
def plot_results(results, strategies, phase_name):
    benches = list(results.keys())
    fig, axes = plt.subplots(1, len(benches), figsize=(4 * len(benches), 5))
    if len(benches) == 1:
        axes = [axes]
    colors = plt.cm.Set2(np.linspace(0, 1, len(strategies)))

    for ax, bench in zip(axes, benches):
        bl = results[bench]["accuracies"]["Baseline"]
        for ci, s in enumerate(strategies):
            if s == "Baseline":
                continue
            acc  = results[bench]["accuracies"].get(s, 0)
            diff = acc - bl
            bar  = ax.bar(s[-15:], diff, color=colors[ci], alpha=0.8)
            ax.text(bar[0].get_x() + bar[0].get_width() / 2,
                    diff + 0.001, f"{acc:.3f}", ha="center", fontsize=7)
        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set_title(bench, fontsize=10)
        ax.set_ylabel("Δ vs Baseline", fontsize=9)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.set_ylim(-0.07, 0.09)

    plt.suptitle(f"R27 {phase_name}: Skip Signal Strategies", fontsize=12, y=1.02)
    plt.tight_layout()
    path = f"{FIGURES_DIR}/r27_{phase_name.lower()}_results.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_streak_distributions(signal_recs, phase_name):
    benches = list(signal_recs.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    colors = plt.cm.tab10(np.linspace(0, 1, len(benches)))

    for mi, (key, label, xlim) in enumerate([
        ("rank_flip_streak_8", "rank_flip_streak@8", (0, 9)),
        ("entropy_8",          "Entropy@8",           (4.5, 7.0)),
        ("entropy_6",          "Entropy@6",           (4.0, 7.0)),
    ]):
        ax = axes[mi]
        for ci, bench in enumerate(benches):
            vals = [s[key] for s in signal_recs[bench] if key in s]
            if not vals:
                continue
            ax.hist(vals, bins=20, alpha=0.55, color=colors[ci], label=bench, density=True)
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.set_xlim(*xlim)
        if mi == 0:
            ax.legend(fontsize=8)

    plt.suptitle(f"R27 {phase_name}: Signal Distributions", fontsize=13)
    plt.tight_layout()
    path = f"{FIGURES_DIR}/r27_{phase_name.lower()}_signals.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_streak_skip_rate(results, signal_recs, strategies, phase_name):
    """
    条形图：各 benchmark 下 S6_streak4 实际跳过率（vs 总样本）
    """
    bench_names = list(results.keys())
    skip_rates  = []
    for bench in bench_names:
        sigs  = signal_recs[bench]
        skips = sum(1 for s in sigs if s["rank_flip_streak_8"] >= 4)
        skip_rates.append(skips / max(len(sigs), 1))

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(bench_names, skip_rates, color=plt.cm.Blues(np.linspace(0.4, 0.9, len(bench_names))))
    for bar, rate in zip(bars, skip_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.005,
                f"{rate:.1%}", ha="center", fontsize=9)
    ax.set_ylabel("Fraction of samples skipped", fontsize=10)
    ax.set_title(f"R27 {phase_name}: S6_streak4 Skip Rate by Benchmark", fontsize=11)
    ax.set_ylim(0, max(skip_rates) * 1.3 + 0.05)
    plt.tight_layout()
    path = f"{FIGURES_DIR}/r27_{phase_name.lower()}_skip_rate.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── 汇总打印 ──────────────────────────────────────────────────────────────
def print_summary(results, strategies, phase_name):
    benches = list(results.keys())
    print(f"\n[{phase_name} Summary]")
    header = f"{'Strategy':<28}" + "".join(f"{b:>14}" for b in benches) + f"{'Avg':>10}"
    print(header)
    print("-" * len(header))
    for s in strategies:
        accs    = [results[b]["accuracies"].get(s, 0) for b in benches]
        bl_accs = [results[b]["accuracies"]["Baseline"] for b in benches]
        diffs   = [a - bl for a, bl in zip(accs, bl_accs)]
        avg     = float(np.mean(accs))
        cols    = "".join(f"{a:.3f}({d:+.3f})"[:11].rjust(14) for a, d in zip(accs, diffs))
        print(f"{s:<28}{cols}{avg:>10.3f}")


# ─── 主函数 ────────────────────────────────────────────────────────────────
def main():
    tok, model, lm_head, ln_f = load_model()

    desired_benchmarks = [
        "BoolQ", "ARC-C", "ARC-Easy", "CSQA", "TruthfulQA",
        "MMLU-HS-Math", "MMLU-Col-Math",
    ]
    strategies = [
        "Baseline",
        "Champion",
        "S1_slope0.05_e53",
        "S6_streak4",
        "S6_streak3",
        "S7_e6low05",
        "S8_compound",
    ]

    # ── Phase 1: N=200 ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Phase 1: N=200 Screening")
    print("=" * 65)
    t0 = time.time()
    bench_data_p1 = load_all_benchmarks(desired_benchmarks, 200)
    p1_results, p1_sigs = run_phase(
        tok, model, lm_head, ln_f,
        bench_data_p1, strategies, "Phase1"
    )
    print(f"\nPhase 1 done in {(time.time()-t0)/60:.1f} min")
    print_summary(p1_results, strategies, "Phase1")

    # 可视化
    plot_results(p1_results, strategies, "Phase1")
    plot_streak_distributions(p1_sigs, "Phase1")
    plot_streak_skip_rate(p1_results, p1_sigs, strategies, "Phase1")

    # 保存 Phase 1
    json.dump(
        {"phase": 1, "N": 200,
         "results": {b: {"accuracies": p1_results[b]["accuracies"]}
                     for b in p1_results},
         "streak_skip_rates": {
             b: sum(1 for s in p1_sigs[b] if s["rank_flip_streak_8"] >= 4) / max(len(p1_sigs[b]), 1)
             for b in p1_sigs
         }},
        open(f"{RESULTS_DIR}/round27_phase1.json", "w"), indent=2
    )

    # ── 筛选 Phase 2 策略 ──
    s1_avg = float(np.mean([p1_results[b]["accuracies"].get("S1_slope0.05_e53", 0)
                            for b in p1_results]))
    candidates_p2 = ["Baseline", "Champion", "S1_slope0.05_e53"]
    for s in strategies:
        if s in candidates_p2:
            continue
        avg = float(np.mean([p1_results[b]["accuracies"].get(s, 0) for b in p1_results]))
        if avg >= s1_avg - 0.005:
            candidates_p2.append(s)
            print(f"  ✓ {s} qualified for Phase 2 (avg={avg:.4f})")
        else:
            print(f"  ✗ {s} eliminated (avg={avg:.4f})")

    # ── Phase 2: N=500 ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"Phase 2: N=500 | strategies: {candidates_p2}")
    print("=" * 65)
    t1 = time.time()
    bench_data_p2 = load_all_benchmarks(desired_benchmarks, 500)
    p2_results, p2_sigs = run_phase(
        tok, model, lm_head, ln_f,
        bench_data_p2, candidates_p2, "Phase2"
    )
    print(f"\nPhase 2 done in {(time.time()-t1)/60:.1f} min")
    print_summary(p2_results, candidates_p2, "Phase2")

    # 可视化
    plot_results(p2_results, candidates_p2, "Phase2")
    plot_streak_distributions(p2_sigs, "Phase2")
    plot_streak_skip_rate(p2_results, p2_sigs, candidates_p2, "Phase2")

    # 保存 Phase 2
    final = {
        "phase": 2, "N": 500,
        "results": {b: {
            "accuracies": p2_results[b]["accuracies"],
            "entropy_8_mean":  float(np.mean([s["entropy_8"] for s in p2_sigs[b]])),
            "entropy_8_std":   float(np.std([s["entropy_8"] for s in p2_sigs[b]])),
            "slope_8_mean":    float(np.mean([s["entropy_slope_8"] for s in p2_sigs[b]])),
            "streak_8_mean":   float(np.mean([s["rank_flip_streak_8"] for s in p2_sigs[b]])),
            "streak_8_skip_rate": sum(1 for s in p2_sigs[b] if s["rank_flip_streak_8"] >= 4) / max(len(p2_sigs[b]), 1),
        } for b in p2_results},
        "final_avgs": {s: float(np.mean([p2_results[b]["accuracies"].get(s, 0)
                                          for b in p2_results]))
                       for s in candidates_p2}
    }
    json.dump(final, open(f"{RESULTS_DIR}/round27_results.json", "w"), indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/round27_results.json")
    print(f"Total: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
