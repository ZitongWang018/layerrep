"""
Block-level metrics for the five-signal ETD visualization plan.

Uses per-layer residual L2 writes δ[ℓ] = mean ||h_ℓ - h_{ℓ-1}||_2 (same indexing as probe).
T-block is [t_start, t_stop) with t_stop exclusive (R30 convention).
"""
from __future__ import annotations

import math
from typing import Sequence


def residual_delta_series(per_layer: dict[int, dict[str, float]], n_layers: int) -> list[float]:
    return [float(per_layer.get(li, {}).get("residual_delta_l2", float("nan"))) for li in range(n_layers)]


def compute_cr_block(delta: Sequence[float], t_start: int, t_stop: int, eps: float = 1e-12) -> float:
    """
    Geometric mean of layer-wise contraction ratios inside the T-block:
      ∏_{ℓ=t_start+1}^{t_stop-1} (δ[ℓ]/δ[ℓ-1]) = δ[t_stop-1]/δ[t_start]
    CR_block = (δ[t_stop-1]/δ[t_start])^(1/(t_stop - t_start - 1))
    """
    if t_stop <= t_start + 1:
        return float("nan")
    n = t_stop - t_start - 1
    if t_start < 0 or t_stop - 1 >= len(delta):
        return float("nan")
    d0 = delta[t_start]
    d1 = delta[t_stop - 1]
    if math.isnan(d0) or math.isnan(d1) or d0 <= eps:
        return float("nan")
    ratio = d1 / d0
    if ratio <= 0:
        return float("nan")
    return float(ratio ** (1.0 / n))


def compute_fpr_simple(delta: Sequence[float], t_start: int, t_stop: int, eps: float = 1e-12) -> float:
    """
    FPR_simple = δ[t_stop-1] / δ[t_start] (exit write vs block-entry write).
    """
    if t_start < 0 or t_stop - 1 >= len(delta) or t_stop <= t_start:
        return float("nan")
    d0 = delta[t_start]
    d1 = delta[t_stop - 1]
    if math.isnan(d0) or math.isnan(d1) or d0 <= eps:
        return float("nan")
    return float(d1 / d0)
