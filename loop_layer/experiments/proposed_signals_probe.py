"""
Extended probe signals (theory-motivated): per-layer scalars for plotting vs layer index.

Independent of r29.probe_forward to avoid changing R29 code paths.
Requires attn_implementation='eager' for attention weights.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


def _jsd_probs(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """JSD between two distributions on last dim. p,q: [..., V], sum to 1."""
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def attn_entropy_from_weights(attn_weights: torch.Tensor) -> float:
    """attn_weights: [B, H, S, S] — entropy over key dim, mean all."""
    w = attn_weights.clamp_min(1e-9)
    ent = -(w * w.log()).sum(dim=-1)
    return float(ent.mean().item())


def residual_delta_l2_mean(h_l: torch.Tensor, h_prev: torch.Tensor) -> float:
    """Mean L2 ||h_l - h_prev|| over all token positions. [B,S,D]."""
    a = h_l.reshape(-1, h_l.shape[-1]).float()
    b = h_prev.reshape(-1, h_prev.shape[-1]).float()
    d = (a - b).norm(dim=-1)
    return float(d.mean().item())


def layer_cosine_sim_mean(h_l: torch.Tensor, h_prev: torch.Tensor) -> float:
    a = h_l.reshape(-1, h_l.shape[-1])
    b = h_prev.reshape(-1, h_prev.shape[-1])
    cos = F.cosine_similarity(a, b, dim=-1)
    return float(cos.mean().item())


def logit_probs_last_token(
    h: torch.Tensor, ln_f: torch.nn.Module, lm_head: torch.nn.Module
) -> torch.Tensor:
    """[B, V] float32 probabilities at last token (CPU, for stable JSD across layers)."""
    dev = next(ln_f.parameters()).device
    h_last = h[:, -1:, :].to(dev)
    with torch.no_grad():
        logits = lm_head(ln_f(h_last)).float()
    return F.softmax(logits, dim=-1).squeeze(1).cpu()


def effective_rank_svd_rows(h_seq: torch.Tensor, max_tokens: int = 64) -> float:
    """
    h_seq: [S, D] — one sequence slice (one batch item).
    Subsample up to max_tokens rows, center, SVD singular values -> erank.
    SVD on CPU avoids CPU/CUDA mix when device_map='auto' places tensors on different devices.
    """
    h = h_seq.float().detach().cpu().contiguous()
    s_len = h.shape[0]
    if s_len == 0:
        return float("nan")
    if s_len > max_tokens:
        idx = torch.linspace(0, s_len - 1, max_tokens).long()
        h = h[idx]
    h = h - h.mean(dim=0, keepdim=True)
    # svdvals: min(S,D) singular values
    sig = torch.linalg.svdvals(h)
    sig = sig[sig > 1e-6]
    if sig.numel() == 0:
        return 0.0
    p = sig / sig.sum()
    ent = -(p * (p + 1e-20).log()).sum()
    return float(ent.exp().item())


def attention_consensus_index(aw: torch.Tensor) -> float:
    """
    aw: [B, H, S, S] attention probs.
    Last query position: for each head, distribution over keys [B,H,S].
    Mean JSD over unordered pairs of heads, then map to [0,1] consensus.
    """
    b, h, s, _ = aw.shape
    if h < 2:
        return float("nan")
    # [B, H, S]
    dist = aw[:, :, -1, :].clamp_min(1e-9)
    dist = dist / dist.sum(dim=-1, keepdim=True)
    # average over batch -> representative [H, S]
    p = dist.float().mean(dim=0)
    jsds = []
    for i in range(h):
        for j in range(i + 1, h):
            jsds.append(_jsd_probs(p[i : i + 1], p[j : j + 1]).item())
    mean_jsd = sum(jsds) / len(jsds)
    # normalize by ln(2) (max JSD for binary support); clip to [0,1]
    cons = 1.0 - mean_jsd / math.log(2.0)
    return float(max(0.0, min(1.0, cons)))


def logit_top1_margin_last(h: torch.Tensor, ln_f: torch.nn.Module, lm_head: torch.nn.Module) -> float:
    probs = logit_probs_last_token(h, ln_f, lm_head)
    top2 = torch.topk(probs, k=2, dim=-1).values
    if top2.shape[-1] < 2:
        return float("nan")
    return float((top2[:, 0] - top2[:, 1]).mean().item())


def collect_proposed_signals(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    n_layers: int,
    erank_max_tokens: int = 64,
) -> dict[int, dict[str, float]]:
    """
    Returns signals[l] for l in 0..n_layers-1 (after decoder layer l).
    """
    base = model.model
    ln_f = base.norm
    lm_head = model.lm_head

    emb_dev = next(base.embed_tokens.parameters()).device
    input_ids = input_ids.to(emb_dev)
    if attention_mask is not None:
        attention_mask = attention_mask.to(emb_dev)

    hidden_per_layer: list[torch.Tensor | None] = [None] * n_layers
    attn_weights_per_layer: list[torch.Tensor | None] = [None] * n_layers
    hooks: list[Any] = []

    def hid_hook(li: int):
        def fn(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            hidden_per_layer[li] = h.detach()

        return fn

    def attn_hook(li: int):
        def fn(_m, _inp, out):
            if isinstance(out, tuple) and len(out) > 1:
                attn_weights_per_layer[li] = out[1].detach() if out[1] is not None else None

        return fn

    for li in range(n_layers):
        hooks.append(base.layers[li].register_forward_hook(hid_hook(li)))
        hooks.append(base.layers[li].self_attn.register_forward_hook(attn_hook(li)))

    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    for h in hooks:
        h.remove()

    if hidden_per_layer[n_layers - 1] is None:
        raise RuntimeError("Proposed probe: no final hidden state")

    with torch.no_grad():
        inputs_embeds = base.embed_tokens(input_ids)

    out: dict[int, dict[str, float]] = {}
    probs_cache: list[torch.Tensor | None] = [None] * n_layers
    delta_l2_list: list[float] = []

    for li in range(n_layers):
        h = hidden_per_layer[li]
        if h is None:
            continue
        h_prev = inputs_embeds if li == 0 else hidden_per_layer[li - 1]
        if h_prev is None:
            h_prev = inputs_embeds
        if h_prev.device != h.device:
            h_prev = h_prev.to(h.device)

        rec: dict[str, float] = {}
        d2 = residual_delta_l2_mean(h, h_prev)
        delta_l2_list.append(d2)
        rec["residual_delta_l2"] = d2
        rec["layer_cos_sim"] = layer_cosine_sim_mean(h, h_prev)

        if li >= 1:
            prev_d = delta_l2_list[li - 1]
            rec["contraction_ratio"] = d2 / (prev_d + 1e-12)
        else:
            rec["contraction_ratio"] = float("nan")

        probs_cache[li] = logit_probs_last_token(h, ln_f, lm_head)
        rec["logit_top1_margin"] = logit_top1_margin_last(h, ln_f, lm_head)

        if li >= 1 and probs_cache[li - 1] is not None:
            jsd = _jsd_probs(probs_cache[li], probs_cache[li - 1]).mean().item()
            rec["logit_lens_jsd_vel"] = float(jsd)
        else:
            rec["logit_lens_jsd_vel"] = float("nan")

        # erank on first batch row only (typical B=1); multi-B: use mean erank over batch
        hb = h.shape[0]
        eranks = []
        for bi in range(hb):
            eranks.append(effective_rank_svd_rows(h[bi], max_tokens=erank_max_tokens))
        rec["erank"] = float(sum(eranks) / max(len(eranks), 1))

        aw = attn_weights_per_layer[li]
        if aw is not None:
            rec["attn_entropy"] = attn_entropy_from_weights(aw)
            rec["attn_consensus"] = attention_consensus_index(aw)
        else:
            rec["attn_entropy"] = float("nan")
            rec["attn_consensus"] = float("nan")

        out[li] = rec

    # Second pass: delta_erank, jsd_curv need previous layer fields
    prev_erank = float("nan")
    prev_jsd_vel = float("nan")
    for li in range(n_layers):
        rec = out[li]
        e = rec["erank"]
        if math.isnan(prev_erank):
            rec["delta_erank"] = float("nan")
        else:
            rec["delta_erank"] = e - prev_erank
        prev_erank = e

        jv = rec["logit_lens_jsd_vel"]
        if math.isnan(prev_jsd_vel):
            rec["logit_lens_jsd_curv"] = float("nan")
        else:
            rec["logit_lens_jsd_curv"] = jv - prev_jsd_vel
        prev_jsd_vel = jv

    return out


def add_delta_norm_to_tstart(
    per_layer: dict[int, dict[str, float]], n_layers: int, t_start: int
) -> None:
    """In-place: delta_norm_to_tstart[l] = residual_delta_l2[l] / (residual_delta_l2[t_start] + eps)."""
    if t_start < 0 or t_start >= n_layers:
        for li in range(n_layers):
            if li in per_layer:
                per_layer[li]["delta_norm_to_tstart"] = float("nan")
        return
    denom_layer = per_layer.get(t_start, {})
    denom = float(denom_layer.get("residual_delta_l2", 0.0)) + 1e-12
    for li in range(n_layers):
        if li not in per_layer:
            continue
        num = float(per_layer[li]["residual_delta_l2"])
        per_layer[li]["delta_norm_to_tstart"] = num / denom


def proposed_signals_to_lists(
    per_layer: dict[int, dict[str, float]], n_layers: int, keys: list[str]
) -> dict[str, list[float]]:
    rows: dict[str, list[float]] = {k: [] for k in keys}
    for li in range(n_layers):
        d = per_layer.get(li, {})
        for k in keys:
            rows[k].append(float(d.get(k, float("nan"))))
    return rows
