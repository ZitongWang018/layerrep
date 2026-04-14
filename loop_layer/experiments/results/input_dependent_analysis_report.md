# Input-Dependent Signal Analysis Report

## 1. Input Sensitivity Depth

For each signal, the deepest layer where inter-sample std > 10% of max std:

| Signal | ARC-C | TruthfulQA | CSQA | MMLU-HS-Math |
|--------|--------|--------|--------|--------|
| attn_entropy | 35 | 35 | 35 | 35 |
| ffn_gate_norm | 35 | 35 | 35 | 35 |
| layer_sim | 35 | 35 | 35 | 35 |
| head_specialization | 35 | 35 | 35 | 35 |
| logit_lens_KL | 34 | 34 | 34 | 34 |
| attention_locality | 35 | 35 | 35 | 35 |
| residual_write_norm | 6 | 6 | 6 | 6 |
| participation_ratio | 5 | 5 | 5 | 5 |
| prediction_flip_rate | 35 | 35 | 35 | 35 |
| attn_sink_ratio | 35 | 35 | 35 | 35 |
| residual_delta_l2 | 35 | 35 | 35 | 35 |
| contraction_ratio | 6 | 6 | 6 | 6 |
| logit_lens_jsd_vel | 35 | 35 | 35 | 35 |
| logit_lens_jsd_curv | 35 | 35 | 35 | 35 |
| erank | 35 | 35 | 35 | 35 |
| delta_erank | 34 | 35 | 35 | 34 |
| attn_consensus | 35 | 35 | 35 | 35 |
| logit_top1_margin | 35 | 35 | 35 | 35 |

Optimal t_start for reference: ARC-C=14, TruthfulQA=16, CSQA=10, MMLU-HS-Math=10

## 2. Benchmark Separability (F-ratio)

Top-3 layers per signal where benchmarks are most distinguishable:

| Signal | Peak F-ratio Layers | Peak F values |
|--------|--------------------|--------------|
| attn_entropy | 34, 6, 35 | 2.146, 1.956, 1.953 |
| ffn_gate_norm | 8, 6, 9 | 2.118, 2.009, 1.895 |
| layer_sim | 6, 13, 1 | 1.932, 0.893, 0.867 |
| head_specialization | 12, 11, 17 | 1.715, 1.694, 1.509 |
| logit_lens_KL | 22, 25, 23 | 0.793, 0.613, 0.585 |
| attention_locality | 34, 32, 35 | 3.334, 2.375, 2.295 |
| residual_write_norm | 6, 33, 34 | 1.675, 1.237, 1.171 |
| participation_ratio | 28, 27, 29 | 0.872, 0.868, 0.861 |
| prediction_flip_rate | 21, 11, 22 | 0.426, 0.366, 0.359 |
| attn_sink_ratio | 4, 3, 35 | 2.312, 2.187, 2.159 |
| residual_delta_l2 | 8, 5, 16 | 1.983, 1.792, 1.689 |
| contraction_ratio | 0, 13, 17 | nan, 1.854, 1.799 |
| logit_lens_jsd_vel | 0, 7, 2 | nan, 2.317, 1.940 |
| logit_lens_jsd_curv | 1, 0, 6 | nan, nan, 1.854 |
| erank | 35, 5, 0 | 1.466, 1.425, 1.424 |
| delta_erank | 0, 34, 33 | nan, 1.984, 1.780 |
| attn_consensus | 1, 7, 0 | 2.337, 1.816, 1.480 |
| logit_top1_margin | 19, 16, 22 | 1.050, 0.885, 0.836 |

## 3. Input-Dependent Features vs Optimal t_start

Spearman correlation of per-sample deviation features with optimal t_start:

### Top 30 features by |ρ|

| Feature | ρ | p-value | Significant? |
|---------|-----|---------|-------------|
| logit_lens_jsd_vel_val_L10 | -0.586 | 0.0000 | ** |
| contraction_ratio_val_L20 | -0.564 | 0.0000 | ** |
| logit_lens_jsd_curv_val_L10 | -0.550 | 0.0000 | ** |
| logit_lens_KL_val_L20 | -0.512 | 0.0000 | ** |
| residual_write_norm_val_L20 | -0.499 | 0.0000 | ** |
| layer_sim_val_L20 | +0.482 | 0.0000 | ** |
| residual_delta_l2_val_L20 | -0.447 | 0.0000 | ** |
| logit_lens_jsd_curv_early_dev_rms | -0.379 | 0.0005 | ** |
| logit_lens_jsd_vel_val_L8 | +0.366 | 0.0008 | ** |
| attn_consensus_val_L14 | -0.366 | 0.0009 | ** |
| logit_lens_KL_val_L14 | -0.359 | 0.0011 | ** |
| ffn_gate_norm_val_L8 | +0.357 | 0.0012 | ** |
| prediction_flip_rate_val_L10 | -0.354 | 0.0013 | ** |
| logit_lens_KL_val_L5 | -0.342 | 0.0019 | ** |
| attn_consensus_mid_dev_rms | -0.341 | 0.0019 | ** |
| logit_lens_KL_val_L16 | -0.338 | 0.0021 | ** |
| logit_top1_margin_mid_dev_rms | -0.337 | 0.0022 | ** |
| residual_write_norm_val_L14 | -0.327 | 0.0031 | ** |
| logit_lens_jsd_vel_early_dev_rms | -0.327 | 0.0031 | ** |
| logit_lens_KL_val_L8 | -0.326 | 0.0032 | ** |
| logit_lens_KL_val_L10 | -0.325 | 0.0033 | ** |
| logit_top1_margin_dev_decay | -0.324 | 0.0033 | ** |
| attn_consensus_val_L8 | -0.322 | 0.0036 | ** |
| layer_sim_early_dev_rms | -0.319 | 0.0039 | ** |
| attn_consensus_peak_dev_layer | +0.319 | 0.0040 | ** |
| layer_sim_val_L14 | +0.318 | 0.0040 | ** |
| logit_lens_KL_dev_decay | -0.314 | 0.0045 | ** |
| residual_delta_l2_val_L5 | +0.310 | 0.0052 | ** |
| residual_delta_l2_val_L8 | +0.309 | 0.0052 | ** |
| layer_sim_val_L16 | +0.305 | 0.0060 | ** |

## 4. Key Observations

- Total features tested: 198
- Features with p < 0.05: 46
- Features with p < 0.01: 33
- Expected false positives at p<0.05 by chance: ~10

