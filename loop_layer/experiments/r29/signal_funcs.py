"""Per-layer signal computations for R29 (torch tensors on device)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def attn_entropy_from_weights(attn_weights: torch.Tensor) -> float:
    """attn_weights: [B, H, S, S] — entropy over key dim."""
    w = attn_weights.clamp_min(1e-9)
    ent = -(w * w.log()).sum(dim=-1)  # [B, H, S]
    return float(ent.mean().item())


def head_specialization_from_weights(attn_weights: torch.Tensor) -> float:
    """Std over heads of mean entropy per head."""
    w = attn_weights.clamp_min(1e-9)
    ent = -(w * w.log()).sum(dim=-1)  # [B, H, S]
    ent_per_head = ent.mean(dim=(0, 2))  # [H]
    return float(ent_per_head.std().item())


def attention_locality_from_weights(attn_weights: torch.Tensor) -> float:
    """Normalized expected |q-k| distance in [0, 1]."""
    b, h, s, _ = attn_weights.shape
    device = attn_weights.device
    positions = torch.arange(s, device=device, dtype=torch.float32)
    dist = (positions.view(1, 1, s, 1) - positions.view(1, 1, 1, s)).abs()
    dist = dist / max(s - 1, 1)
    expected = (attn_weights.float() * dist).sum(dim=-1)  # [B, H, S]
    return float(expected.mean().item())


def attn_sink_ratio(attn_weights: torch.Tensor, sink_idx: int = 0) -> float:
    """Mean attention mass on key index sink_idx (e.g. BOS). attn [B,H,Q,K]."""
    sink_mass = attn_weights[:, :, :, sink_idx]  # [B, H, Q]
    return float(sink_mass.mean().item())


def layer_cosine_sim(h_l: torch.Tensor, h_prev: torch.Tensor) -> float:
    """h_*: [B, S, D]"""
    a = h_l.reshape(-1, h_l.shape[-1])
    b = h_prev.reshape(-1, h_prev.shape[-1])
    cos = F.cosine_similarity(a, b, dim=-1)
    return float(cos.mean().item())


def residual_write_norm(h_l: torch.Tensor, h_prev: torch.Tensor) -> float:
    """Relative L2 change."""
    a = h_l.reshape(-1, h_l.shape[-1]).float()
    b = h_prev.reshape(-1, h_prev.shape[-1]).float()
    delta = a - b
    rel = delta.norm(dim=-1) / (b.norm(dim=-1) + 1e-9)
    return float(rel.mean().item())


def participation_ratio(h: torch.Tensor) -> float:
    """Diagonal participation ratio on token×dim matrix, normalized by D."""
    h2 = h.reshape(-1, h.shape[-1]).float()
    h2 = h2 - h2.mean(dim=0, keepdim=True)
    var = (h2**2).mean(dim=0) + 1e-10
    pr = (var.sum() ** 2) / (var**2).sum()
    d = float(h.shape[-1])
    return float((pr / d).item())


def logit_lens_kl_last_token(
    h_l: torch.Tensor,
    h_final: torch.Tensor,
    ln_f: torch.nn.Module,
    lm_head: torch.nn.Module,
) -> float:
    """
    KL(P_l || P_final) at last token, using final RMSNorm + lm_head (logit lens).
    h_l, h_final: [B, S, D] in model dtype
    """
    h_l_last = h_l[:, -1:, :]
    h_f_last = h_final[:, -1:, :]
    with torch.no_grad():
        logits_l = lm_head(ln_f(h_l_last)).float()  # [B, 1, V]
        logits_f = lm_head(ln_f(h_f_last)).float()
    log_p_f = F.log_softmax(logits_f, dim=-1)
    p_l = F.softmax(logits_l, dim=-1)
    # KL(P_l || P_f) = sum p_l * (log p_l - log p_f)
    kl = F.kl_div(log_p_f, p_l, reduction="batchmean", log_target=False)
    return float(kl.item())


def prediction_flip_rate_last_token(
    h_l: torch.Tensor,
    h_prev: torch.Tensor,
    ln_f: torch.nn.Module,
    lm_head: torch.nn.Module,
) -> float:
    """Fraction of batch where argmax logit lens changes from prev to l."""
    with torch.no_grad():
        t1 = lm_head(ln_f(h_l[:, -1:, :])).argmax(dim=-1).view(-1)
        t0 = lm_head(ln_f(h_prev[:, -1:, :])).argmax(dim=-1).view(-1)
    return float((t1 != t0).float().mean().item())


def ffn_gate_norm(gate_act: torch.Tensor) -> float:
    """gate_act: silu(gate_proj(x)), [B, S, I]"""
    return float(gate_act.norm(dim=-1).mean().item())
