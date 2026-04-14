# Signal Predictability Analysis Report
Generated: 2026-04-11 16:47
N samples per benchmark: 20
Total signals analyzed: 18

## R30 Optimal Configurations
- **ARC-C**: t_start=14, t_stop=20
- **TruthfulQA**: t_start=16, t_stop=19
- **CSQA**: t_start=10, t_stop=22
- **MMLU-HS-Math**: t_start=10, t_stop=18

## Spearman Correlation: signal[layer] vs best_acc(t_start=layer)

| Signal | ARC-C | TruthfulQA | CSQA | MMLU-HS-Math | Mean |ρ| |
|--------|--------|--------|--------|--------|--------|
| attn_entropy | -0.482* | -0.029 | -0.717** | -0.557* | 0.446 |
| ffn_gate_norm | +0.180 | +0.730** | -0.179 | +0.021 | 0.278 |
| layer_sim | -0.123 | +0.786** | -0.158 | -0.326 | 0.348 |
| head_specialization | +0.212 | +0.417 | +0.138 | +0.317 | 0.271 |
| logit_lens_KL | +0.028 | -0.703** | +0.231 | +0.312 | 0.319 |
| attention_locality | +0.325 | +0.136 | +0.611** | +0.420 | 0.373 |
| residual_write_norm | -0.194 | -0.816** | -0.148 | -0.161 | 0.330 |
| participation_ratio | +0.137 | -0.301 | -0.028 | -0.310 | 0.194 |
| prediction_flip_rate | -0.016 | -0.726** | +0.458* | -0.025 | 0.306 |
| attn_sink_ratio | +0.341 | +0.111 | +0.539* | +0.407 | 0.349 |
| residual_delta_l2 | +0.072 | +0.353 | -0.363 | -0.198 | 0.247 |
| contraction_ratio | -0.249 | -0.190 | -0.537* | -0.411 | 0.347 |
| logit_lens_jsd_vel | +0.053 | -0.498* | +0.375 | +0.236 | 0.291 |
| logit_lens_jsd_curv | -0.049 | -0.140 | -0.251 | -0.165 | 0.151 |
| erank | -0.297 | +0.257 | -0.638** | -0.491* | 0.421 |
| delta_erank | -0.187 | -0.141 | -0.359 | -0.544* | 0.308 |
| attn_consensus | +0.325 | +0.279 | +0.573** | +0.290 | 0.367 |
| logit_top1_margin | +0.031 | +0.358 | +0.060 | +0.058 | 0.127 |

*p<0.05, **p<0.01

## Top Signals by Mean |ρ| (t_start correlation)

1. **attn_entropy**: mean |ρ| = 0.446
2. **erank**: mean |ρ| = 0.421
3. **attention_locality**: mean |ρ| = 0.373
4. **attn_consensus**: mean |ρ| = 0.367
5. **attn_sink_ratio**: mean |ρ| = 0.349

## Spearman Correlation: signal[layer] vs best_acc(t_stop=layer)

| Signal | ARC-C | TruthfulQA | CSQA | MMLU-HS-Math | Mean |ρ| |
|--------|--------|--------|--------|--------|--------|
| attn_entropy | +0.355 | -0.161 | +0.037 | +0.520* | 0.268 |
| ffn_gate_norm | -0.068 | +0.292 | +0.189 | -0.221 | 0.192 |
| layer_sim | +0.200 | +0.514 | +0.261 | +0.253 | 0.307 |
| head_specialization | +0.200 | +0.002 | -0.031 | +0.363 | 0.149 |
| logit_lens_KL | +0.007 | -0.288 | -0.209 | +0.248 | 0.188 |
| attention_locality | -0.193 | +0.369 | -0.009 | -0.365 | 0.234 |
| residual_write_norm | -0.457 | -0.453 | -0.352 | -0.423 | 0.421 |
| participation_ratio | -0.334 | -0.011 | -0.106 | -0.494 | 0.236 |
| prediction_flip_rate | -0.354 | -0.459 | -0.471 | -0.750** | 0.509 |
| attn_sink_ratio | -0.220 | +0.363 | +0.026 | -0.370 | 0.245 |
| residual_delta_l2 | -0.083 | +0.303 | +0.185 | -0.228 | 0.200 |
| contraction_ratio | +0.197 | +0.326 | +0.104 | +0.328 | 0.239 |
| logit_lens_jsd_vel | -0.334 | -0.372 | -0.347 | -0.649** | 0.426 |
| logit_lens_jsd_curv | -0.009 | +0.424 | -0.130 | +0.109 | 0.168 |
| erank | -0.070 | +0.308 | +0.191 | -0.224 | 0.198 |
| delta_erank | -0.068 | +0.293 | +0.093 | -0.232 | 0.171 |
| attn_consensus | -0.333 | +0.264 | -0.152 | -0.268 | 0.254 |
| logit_top1_margin | +0.391 | +0.160 | +0.193 | +0.616* | 0.340 |

## Feature Alignment: Derivative Peaks vs Optimal Boundaries

| Benchmark | Signal | Optimal t_start | Closest Peak | Distance |
|-----------|--------|----------------|-------------|----------|
| ARC-C | attn_entropy | 14 | 14 | 0 |
| ARC-C | ffn_gate_norm | 14 | 12 | 2 |
| ARC-C | layer_sim | 14 | 13 | 1 |
| ARC-C | head_specialization | 14 | 15 | 1 |
| ARC-C | logit_lens_KL | 14 | 14 | 0 |
| ARC-C | attention_locality | 14 | 14 | 0 |
| ARC-C | residual_write_norm | 14 | 13 | 1 |
| ARC-C | participation_ratio | 14 | 14 | 0 |
| ARC-C | prediction_flip_rate | 14 | 14 | 0 |
| ARC-C | attn_sink_ratio | 14 | 14 | 0 |
| ARC-C | residual_delta_l2 | 14 | 15 | 1 |
| ARC-C | contraction_ratio | 14 | 15 | 1 |
| ARC-C | logit_lens_jsd_vel | 14 | 15 | 1 |
| ARC-C | logit_lens_jsd_curv | 14 | 13 | 1 |
| ARC-C | erank | 14 | 12 | 2 |
| ARC-C | delta_erank | 14 | 13 | 1 |
| ARC-C | attn_consensus | 14 | 14 | 0 |
| ARC-C | logit_top1_margin | 14 | 13 | 1 |
| TruthfulQA | attn_entropy | 16 | 17 | 1 |
| TruthfulQA | ffn_gate_norm | 16 | 15 | 1 |
| TruthfulQA | layer_sim | 16 | 15 | 1 |
| TruthfulQA | head_specialization | 16 | 15 | 1 |
| TruthfulQA | logit_lens_KL | 16 | 15 | 1 |
| TruthfulQA | attention_locality | 16 | 15 | 1 |
| TruthfulQA | residual_write_norm | 16 | 15 | 1 |
| TruthfulQA | participation_ratio | 16 | 15 | 1 |
| TruthfulQA | prediction_flip_rate | 16 | 16 | 0 |
| TruthfulQA | attn_sink_ratio | 16 | 15 | 1 |
| TruthfulQA | residual_delta_l2 | 16 | 15 | 1 |
| TruthfulQA | contraction_ratio | 16 | 15 | 1 |
| TruthfulQA | logit_lens_jsd_vel | 16 | 17 | 1 |
| TruthfulQA | logit_lens_jsd_curv | 16 | 16 | 0 |
| TruthfulQA | erank | 16 | 15 | 1 |
| TruthfulQA | delta_erank | 16 | 15 | 1 |
| TruthfulQA | attn_consensus | 16 | 16 | 0 |
| TruthfulQA | logit_top1_margin | 16 | 15 | 1 |
| CSQA | attn_entropy | 10 | 10 | 0 |
| CSQA | ffn_gate_norm | 10 | 10 | 0 |
| CSQA | layer_sim | 10 | 10 | 0 |
| CSQA | head_specialization | 10 | 10 | 0 |
| CSQA | logit_lens_KL | 10 | 10 | 0 |
| CSQA | attention_locality | 10 | 10 | 0 |
| CSQA | residual_write_norm | 10 | 9 | 1 |
| CSQA | participation_ratio | 10 | 11 | 1 |
| CSQA | prediction_flip_rate | 10 | 11 | 1 |
| CSQA | attn_sink_ratio | 10 | 10 | 0 |
| CSQA | residual_delta_l2 | 10 | 10 | 0 |
| CSQA | contraction_ratio | 10 | 11 | 1 |
| CSQA | logit_lens_jsd_vel | 10 | 11 | 1 |
| CSQA | logit_lens_jsd_curv | 10 | 10 | 0 |
| CSQA | erank | 10 | 11 | 1 |
| CSQA | delta_erank | 10 | 11 | 1 |
| CSQA | attn_consensus | 10 | 9 | 1 |
| CSQA | logit_top1_margin | 10 | 10 | 0 |
| MMLU-HS-Math | attn_entropy | 10 | 10 | 0 |
| MMLU-HS-Math | ffn_gate_norm | 10 | 10 | 0 |
| MMLU-HS-Math | layer_sim | 10 | 10 | 0 |
| MMLU-HS-Math | head_specialization | 10 | 10 | 0 |
| MMLU-HS-Math | logit_lens_KL | 10 | 10 | 0 |
| MMLU-HS-Math | attention_locality | 10 | 10 | 0 |
| MMLU-HS-Math | residual_write_norm | 10 | 9 | 1 |
| MMLU-HS-Math | participation_ratio | 10 | 11 | 1 |
| MMLU-HS-Math | prediction_flip_rate | 10 | 9 | 1 |
| MMLU-HS-Math | attn_sink_ratio | 10 | 10 | 0 |
| MMLU-HS-Math | residual_delta_l2 | 10 | 11 | 1 |
| MMLU-HS-Math | contraction_ratio | 10 | 11 | 1 |
| MMLU-HS-Math | logit_lens_jsd_vel | 10 | 9 | 1 |
| MMLU-HS-Math | logit_lens_jsd_curv | 10 | 10 | 0 |
| MMLU-HS-Math | erank | 10 | 12 | 2 |
| MMLU-HS-Math | delta_erank | 10 | 11 | 1 |
| MMLU-HS-Math | attn_consensus | 10 | 9 | 1 |
| MMLU-HS-Math | logit_top1_margin | 10 | 11 | 1 |

(Only showing signals with derivative peak within 3 layers of optimal t_start)

## Conclusion

See correlation heatmap and overlay plots for detailed visual analysis.
