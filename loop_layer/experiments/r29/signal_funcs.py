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


# ─── R33 新增：FFN 慢权重激活相变信号 ──────────────────────────────────────────

def ffn_gini_last_token(gate_act: torch.Tensor) -> float:
    """
    Gini coefficient of |gate activation| at the last token position.
    gate_act: [B, S, I]  (silu(gate_proj(h)))

    Gini = 0: 完全均匀激活（最大塑性态）
    Gini = 1: 单一神经元独大（最大稀疏稳态）

    低 Gini → FFN 处于"塑性态"，多个 key 方向被均匀弱激活，ETD 微扰更容易触发相变。
    """
    acts = gate_act[0, -1, :].float().abs()
    sorted_acts = acts.sort().values  # ascending
    n = sorted_acts.shape[0]
    total = sorted_acts.sum()
    if total < 1e-9:
        return float("nan")
    idx = torch.arange(1, n + 1, dtype=torch.float32, device=acts.device)
    # Gini = (2 * Σ(i * x_i) / (n * Σx_i)) - (n+1)/n
    gini = (2.0 * (idx * sorted_acts).sum() / (n * total) - (n + 1.0) / n)
    return float(gini.clamp(0.0, 1.0).item())


def ffn_activation_entropy_last_token(gate_act: torch.Tensor) -> float:
    """
    Shannon entropy of normalized |gate activation| at the last token.
    gate_act: [B, S, I]

    高熵 → 多个语义方向被均等激活（概念竞争态），ETD 可能帮助完成收敛。
    低熵 → 单一方向占主导（稳定态）。
    """
    acts = gate_act[0, -1, :].float().abs()
    total = acts.sum()
    if total < 1e-9:
        return float("nan")
    p = acts / total
    ent = -(p * (p + 1e-12).log()).sum()
    return float(ent.item())


def ffn_boundary_frac_last_token(gate_act: torch.Tensor, eps: float = 0.5) -> float:
    """
    Fraction of FFN neurons within ε of the activation boundary (|gate| < eps).
    gate_act: [B, S, I]

    高比例 → 大量神经元处于激活临界区，ETD 的微扰更容易使其越过/落下阈值（相变概率高）。
    """
    acts = gate_act[0, -1, :].float().abs()
    return float((acts < eps).float().mean().item())


def ffn_active_frac_last_token(gate_act: torch.Tensor, thr: float = 1.0) -> float:
    """
    Fraction of neurons with |gate activation| > thr (proxy for activation density).
    gate_act: [B, S, I]
    """
    acts = gate_act[0, -1, :].float().abs()
    return float((acts > thr).float().mean().item())


# ─── R33 新增：Attention 快权重幂迭代信号 ──────────────────────────────────────

def attn_spectral_gap_last_token(attn_weights: torch.Tensor) -> float:
    """
    Spectral gap of attention distribution at last query position.
    attn_weights: [B, H, S, S]

    = mean_over_heads(max_weight / second_max_weight) at last query position

    高谱隙 → W_fast(x) 有强主导特征向量（清晰吸引子），ETD 的幂迭代快速收敛到正确上下文方向。
    低谱隙 → 多个竞争方向，幂迭代可能振荡。
    """
    if attn_weights.shape[2] < 1:
        return float("nan")
    w = attn_weights[0, :, -1, :].float()  # [H, S]
    if w.shape[1] < 2:
        return float("nan")
    top2 = w.topk(2, dim=-1).values  # [H, 2]
    gap = top2[:, 0] / (top2[:, 1] + 1e-9)  # [H]
    return float(gap.mean().item())


def attn_head_consensus_last_token(attn_weights: torch.Tensor) -> float:
    """
    Consensus among attention heads at last query position.
    attn_weights: [B, H, S, S]

    = 1 - mean_JSD_between_heads

    高共识 → 所有头都同意关注相同的位置，W_fast(x) 的主方向清晰，ETD 幂迭代方向一致。
    低共识 → 各头分歧，W_fast(x) 无主导特征向量。
    """
    if attn_weights.shape[2] < 1:
        return float("nan")
    w = attn_weights[0, :, -1, :].float() + 1e-9  # [H, S]
    p = w / w.sum(dim=-1, keepdim=True)           # normalize each head [H, S]
    m = p.mean(dim=0, keepdim=True)               # mixture distribution [1, S]
    m = m / m.sum()
    # JSD(P_h || M) = KL(P_h || M)  (simplified Jensen-Shannon)
    kl = F.kl_div(m.log().expand_as(p), p, reduction="none").sum(-1)  # [H]
    mean_jsd = kl.mean().clamp(max=1.0)
    return float(1.0 - mean_jsd.item())


def attn_top2_mass_last_token(attn_weights: torch.Tensor) -> float:
    """
    Mean fraction of attention mass on top-2 tokens per head at last query position.
    attn_weights: [B, H, S, S]

    高值 → 注意力集中在少数 token（强吸引子）。
    """
    if attn_weights.shape[2] < 1:
        return float("nan")
    w = attn_weights[0, :, -1, :].float()  # [H, S]
    top2_mass = w.topk(min(2, w.shape[1]), dim=-1).values.sum(dim=-1)  # [H]
    return float(top2_mass.mean().item())


# ─── R34: 方向与交叉交互信号 ─────────────────────────────────────────────────
#
# 设计原则（吸取 R33 Gini/spectral gap 失败教训）：
#   - 方向 > 幅度分布形状（cos 是输入依赖的，Gini 不是）
#   - 跨模块关系 > 单模块统计
#   - 有限差分 > 间接推断


def _last_token_vec(t: torch.Tensor) -> torch.Tensor:
    """Extract last-token vector from [B, S, D] or [1, S, D], return [D] float32."""
    return t[0, -1, :].float()


def attn_write_norm_last(a_l: torch.Tensor, h_input: torch.Tensor) -> float:
    """||a_l|| / ||h_input|| at last token. a_l, h_input: [B, S, D]."""
    a = _last_token_vec(a_l)
    h = _last_token_vec(h_input)
    denom = h.norm() + 1e-9
    return float((a.norm() / denom).item())


def ffn_write_norm_last(m_l: torch.Tensor, h_input: torch.Tensor) -> float:
    """||m_l|| / ||h_input|| at last token. m_l, h_input: [B, S, D]."""
    m = _last_token_vec(m_l)
    h = _last_token_vec(h_input)
    denom = h.norm() + 1e-9
    return float((m.norm() / denom).item())


def ffn_direction_drift_last(m_l: torch.Tensor, m_prev: torch.Tensor) -> float:
    """1 - cos(m_l, m_{l-1}) at last token. High = FFN still exploring different knowledge."""
    a = _last_token_vec(m_l)
    b = _last_token_vec(m_prev)
    cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1)
    return float((1.0 - cos).item())


def attn_direction_drift_last(a_l: torch.Tensor, a_prev: torch.Tensor) -> float:
    """1 - cos(a_l, a_{l-1}) at last token. High = attention still searching context."""
    a = _last_token_vec(a_l)
    b = _last_token_vec(a_prev)
    cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1)
    return float((1.0 - cos).item())


def hidden_rotation_rate_last(h_out: torch.Tensor, h_in: torch.Tensor) -> float:
    """1 - cos(h_output, h_input) at last token."""
    a = _last_token_vec(h_out)
    b = _last_token_vec(h_in)
    cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1)
    return float((1.0 - cos).item())


def cross_cos_a_m_last(a_l: torch.Tensor, m_l: torch.Tensor) -> float:
    """cos(a_l, m_l) at last token (signed). Measures attn-FFN direction alignment."""
    a = _last_token_vec(a_l)
    m = _last_token_vec(m_l)
    cos = F.cosine_similarity(a.unsqueeze(0), m.unsqueeze(0), dim=-1)
    return float(cos.item())


def attn_ffn_balance_last(a_l: torch.Tensor, m_l: torch.Tensor) -> float:
    """||a_l|| / (||a_l|| + ||m_l||) at last token. 0.5 = balanced contributions."""
    a_norm = _last_token_vec(a_l).norm()
    m_norm = _last_token_vec(m_l).norm()
    denom = a_norm + m_norm + 1e-9
    return float((a_norm / denom).item())


@torch.no_grad()
def cross_attn_to_ffn_sensitivity(
    mlp: torch.nn.Module,
    post_attn_ln: torch.nn.Module,
    h_post_attn: torch.Tensor,
    h_input: torch.Tensor,
) -> float:
    """
    Finite-difference: how much does attention's contribution change FFN output?
      sensitivity = ||MLP(LN(h_input + a)) - MLP(LN(h_input))|| / ||MLP(LN(h_input + a))||
    where a = h_post_attn - h_input.
    h_post_attn, h_input: [B, S, D]. Uses last token only for efficiency.
    """
    hp = h_post_attn[:, -1:, :]
    hi = h_input[:, -1:, :]
    m_actual = mlp(post_attn_ln(hp)).float()
    m_counterfactual = mlp(post_attn_ln(hi)).float()
    diff = (m_actual - m_counterfactual).norm()
    denom = m_actual.norm() + 1e-9
    return float((diff / denom).item())


@torch.no_grad()
def cross_attn_to_ffn_dir_shift(
    mlp: torch.nn.Module,
    post_attn_ln: torch.nn.Module,
    h_post_attn: torch.Tensor,
    h_input: torch.Tensor,
) -> float:
    """
    Direction version: 1 - cos(MLP(LN(h_post_attn)), MLP(LN(h_input))) at last token.
    Measures whether attention changes the *direction* of FFN output.
    """
    hp = h_post_attn[:, -1:, :]
    hi = h_input[:, -1:, :]
    m_actual = mlp(post_attn_ln(hp)).float().view(1, -1)
    m_cf = mlp(post_attn_ln(hi)).float().view(1, -1)
    cos = F.cosine_similarity(m_actual, m_cf, dim=-1)
    return float((1.0 - cos).item())
