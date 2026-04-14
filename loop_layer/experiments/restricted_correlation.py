#!/usr/bin/env python3
import json, sys
import numpy as np
from scipy.stats import spearmanr
from collections import defaultdict

with open('results/signal_raw_all_per_sample.json') as f:
    raw = json.load(f)
with open('results/r30_sweep_results.json') as f:
    sweep = json.load(f)

BENCHMARKS = ['ARC-C','TruthfulQA','CSQA','MMLU-HS-Math']
SIGNALS = [
    'attn_entropy','ffn_gate_norm','layer_sim','head_specialization',
    'logit_lens_KL','attention_locality','residual_write_norm',
    'participation_ratio','prediction_flip_rate','attn_sink_ratio',
    'residual_delta_l2','contraction_ratio','logit_lens_jsd_vel',
    'logit_lens_jsd_curv','erank','delta_erank','attn_consensus',
    'logit_top1_margin',
]

best_ts = defaultdict(lambda: defaultdict(float))
for row in sweep['results']:
    ts = row['t_start']
    for bench in BENCHMARKS:
        if bench in row and row[bench] > best_ts[bench][ts]:
            best_ts[bench][ts] = row[bench]

mean_curves = {}
for bench in BENCHMARKS:
    samples = raw[bench]
    means = {}
    for k in SIGNALS:
        stack = np.array([s[k] for s in samples], dtype=np.float64)
        means[k] = np.nanmean(stack, axis=0)
    mean_curves[bench] = means

print("="*90, flush=True)
print("Spearman rho: signal[L] vs best_acc(t_start=L)  [t_start >= 7 ONLY]", flush=True)
print("Removes confound from early-layer danger zone", flush=True)
print("="*90, flush=True)
sys.stdout.flush()

header = f"{'Signal':25s}"
for b in BENCHMARKS:
    header += f" | {b:>13s}"
header += " | Mean|rho|"
print(header, flush=True)
print("-"*100, flush=True)

for sk in SIGNALS:
    abs_rhos = []
    row = f"{sk:25s}"
    for bench in BENCHMARKS:
        sig = mean_curves[bench][sk]
        tmap = best_ts[bench]
        layers = sorted([l for l in tmap.keys() if l >= 7])
        x = np.array([sig[l] for l in layers if l < len(sig)], dtype=np.float64)
        y = np.array([tmap[l] for l in layers if l < len(sig)], dtype=np.float64)
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() >= 5:
            rho, p = spearmanr(x[mask], y[mask])
            star = ""
            if p < 0.05: star = "*"
            if p < 0.01: star = "**"
            row += f" | {rho:+.3f}{star:2s}     "
            abs_rhos.append(abs(rho))
        else:
            row += " |           N/A"
    mrho = np.mean(abs_rhos) if abs_rhos else 0
    row += f" | {mrho:.3f}"
    print(row, flush=True)

print(flush=True)
print("="*90, flush=True)
print("Pooled cross-benchmark Spearman (t_start >= 7)", flush=True)
print("="*90, flush=True)

for sk in SIGNALS:
    all_x, all_y = [], []
    for bench in BENCHMARKS:
        sig = mean_curves[bench][sk]
        tmap = best_ts[bench]
        layers = sorted([l for l in tmap.keys() if l >= 7])
        for l in layers:
            if l < len(sig) and not np.isnan(sig[l]):
                all_x.append(sig[l])
                all_y.append(tmap[l])
    if len(all_x) >= 10:
        rho, p = spearmanr(all_x, all_y)
        star = ""
        if p < 0.05: star = "*"
        if p < 0.01: star = "**"
        print(f"  {sk:25s}: rho={rho:+.3f} p={p:.4f} {star}", flush=True)

print(flush=True)
print("="*90, flush=True)
print("Inter-sample CV at optimal t_start vs mean CV (layers 7-35)", flush=True)
print("="*90, flush=True)

opt_map = {'ARC-C':14,'TruthfulQA':16,'CSQA':10,'MMLU-HS-Math':10}
top_sigs = ['attn_entropy','erank','attention_locality','attn_consensus',
            'logit_lens_jsd_vel','contraction_ratio','layer_sim']
for bench in BENCHMARKS:
    t0 = opt_map[bench]
    print(f"\n{bench} (optimal t_start={t0}):", flush=True)
    for sk in top_sigs:
        samples = raw[bench]
        vals_at_t0 = [s[sk][t0] for s in samples]
        mean_t0 = np.nanmean(vals_at_t0)
        std_t0 = np.nanstd(vals_at_t0)
        cv_t0 = std_t0 / (abs(mean_t0) + 1e-12)
        cvs = []
        for l in range(7, 36):
            vals = [s[sk][l] for s in samples]
            m = np.nanmean(vals)
            s_ = np.nanstd(vals)
            cvs.append(s_ / (abs(m) + 1e-12))
        mean_cv = np.mean(cvs)
        print(f"  {sk:25s}: CV@t0={cv_t0:.4f}, meanCV={mean_cv:.4f}, ratio={cv_t0/(mean_cv+1e-12):.2f}", flush=True)

print("\nDone.", flush=True)
