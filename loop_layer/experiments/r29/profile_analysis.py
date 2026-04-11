"""Profile-based t_start / t_stop detection (per-sample normalized)."""
from __future__ import annotations

import math
from typing import Literal

import numpy as np

Mode = Literal["peak", "valley"]


def find_contiguous_regions(binary: np.ndarray) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    n = len(binary)
    i = 0
    while i < n:
        if binary[i]:
            j = i
            while j < n and binary[j]:
                j += 1
            regions.append((i, j - 1))
            i = j
        else:
            i += 1
    return regions


def apply_boundary_constraints(
    t_start: int,
    t_stop: int,
    min_start: int = 8,
    max_start: int = 18,
    min_block: int = 4,
    max_block: int = 28,
    max_stop: int | None = None,
) -> tuple[int, int]:
    if max_stop is None:
        max_stop = 35
    t_start = int(max(min_start, min(t_start, max_start)))
    t_stop = int(max(t_start + min_block, min(t_stop, max_stop)))
    if t_stop - t_start > max_block:
        t_stop = t_start + max_block
    return t_start, t_stop


def _series(
    signals: dict[int, dict[str, float]], key: str, l_min: int, l_max: int, n_layers: int
) -> np.ndarray:
    arr = np.array(
        [signals.get(l, {}).get(key, float("nan")) for l in range(n_layers)], dtype=np.float64
    )
    return arr[l_min : l_max + 1]


def pa1_single_signal(
    signals: dict[int, dict[str, float]],
    signal_name: str,
    mode: Mode,
    n_layers: int,
    score_fraction: float = 0.5,
    l_min: int = 8,
    l_max: int = 32,
    fallback: tuple[int, int] = (8, 22),
) -> tuple[int, int, bool]:
    """
    Returns (t_start, t_stop, used_fallback).
    """
    sub = _series(signals, signal_name, l_min, l_max, n_layers)
    valid = np.isfinite(sub)
    if valid.sum() < 4:
        return (*fallback, True)
    sub_v = sub[valid]
    lo, hi = float(np.nanmin(sub_v)), float(np.nanmax(sub_v))
    if hi - lo < 1e-6:
        return (*fallback, True)
    full = np.array(
        [signals.get(l, {}).get(signal_name, float("nan")) for l in range(n_layers)], dtype=np.float64
    )
    seg = full[l_min : l_max + 1].astype(np.float64)
    norm = (seg - lo) / (hi - lo)
    norm = np.where(np.isfinite(norm), norm, 0.0)
    score = 1.0 - norm if mode == "valley" else norm
    if np.nanmax(score) < 1e-9:
        return (*fallback, True)
    thr = score_fraction * float(np.nanmax(score))
    active = (score > thr).astype(np.int32)
    regions = find_contiguous_regions(active)
    if not regions:
        return (*fallback, True)
    best = max(regions, key=lambda r: r[1] - r[0])
    t_start = l_min + best[0]
    t_stop = l_min + best[1]
    t_start, t_stop = apply_boundary_constraints(t_start, t_stop, max_stop=n_layers - 2)
    return t_start, t_stop, False


SIGNAL_MODES: dict[str, Mode] = {
    "layer_sim": "valley",
    "attn_entropy": "peak",
    "ffn_gate_norm": "peak",
    "head_specialization": "peak",
    "logit_lens_KL": "peak",
    "attention_locality": "peak",
    "residual_write_norm": "peak",
    "participation_ratio": "peak",
    "prediction_flip_rate": "peak",
    "attn_sink_ratio": "peak",
}


def b1_weighted_combo(
    signals: dict[int, dict[str, float]],
    n_layers: int,
    signal_modes: dict[str, Mode],
    weights: dict[str, float] | None,
    score_fraction: float = 0.5,
    l_min: int = 8,
    l_max: int = 32,
    fallback: tuple[int, int] = (8, 22),
) -> tuple[int, int, bool]:
    if weights is None:
        n = len(signal_modes)
        weights = {k: 1.0 / n for k in signal_modes}
    seg_len = l_max - l_min + 1
    combined = np.zeros(seg_len, dtype=np.float64)
    wsum = 0.0
    for name, mode in signal_modes.items():
        sub = _series(signals, name, l_min, l_max, n_layers)
        if not np.any(np.isfinite(sub)):
            continue
        lo, hi = np.nanmin(sub), np.nanmax(sub)
        if not math.isfinite(lo) or hi - lo < 1e-6:
            continue
        norm = (sub - lo) / (hi - lo)
        norm = np.where(np.isfinite(norm), norm, 0.0)
        score = 1.0 - norm if mode == "valley" else norm
        w = weights.get(name, 0.0)
        combined += w * score
        wsum += w
    if wsum < 1e-9:
        return (*fallback, True)
    combined /= wsum
    thr = score_fraction * float(np.max(combined))
    active = (combined > thr).astype(np.int32)
    regions = find_contiguous_regions(active)
    if not regions:
        return (*fallback, True)
    best = max(regions, key=lambda r: r[1] - r[0])
    t_start = l_min + best[0]
    t_stop = l_min + best[1]
    t_start, t_stop = apply_boundary_constraints(t_start, t_stop, max_stop=n_layers - 2)
    return t_start, t_stop, False


def b3_median_consensus(
    signals: dict[int, dict[str, float]],
    n_layers: int,
    names: list[str],
    l_min: int = 8,
    l_max: int = 32,
    fallback: tuple[int, int] = (8, 22),
) -> tuple[int, int, bool]:
    starts: list[int] = []
    stops: list[int] = []
    for name in names:
        mode = SIGNAL_MODES.get(name, "peak")
        ts, te, fb = pa1_single_signal(
            signals, name, mode, n_layers, l_min=l_min, l_max=l_max, fallback=fallback
        )
        if not fb:
            starts.append(ts)
            stops.append(te)
    if len(starts) < 2:
        return (*fallback, True)
    t_start = int(np.median(starts))
    t_stop = int(np.median(stops))
    t_start, t_stop = apply_boundary_constraints(t_start, t_stop, max_stop=n_layers - 2)
    return t_start, t_stop, False
