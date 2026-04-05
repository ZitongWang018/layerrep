"""ETD forward for Qwen3: E -> T^k -> D using the same masks/RoPE as Qwen3Model."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.masking_utils import create_causal_mask


def _prepare_position_ids(
    attention_mask: torch.Tensor | None,
    past_seen_tokens: int,
    batch: int,
    seq_len: int,
    device: torch.device,
) -> torch.LongTensor:
    if attention_mask is not None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_seen_tokens > 0:
            position_ids = position_ids + past_seen_tokens
    else:
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
        if past_seen_tokens > 0:
            position_ids = position_ids + past_seen_tokens
    return position_ids


@torch.inference_mode()
def etd_forward_logits(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    n_e: int,
    n_t: int,
    k: int,
) -> torch.Tensor:
    """
    Run Qwen3 with encoder (first n_e layers), thinking block (next n_t layers, repeated k times),
    decoder (remaining layers), then final norm + lm_head.

    n_e + n_t + n_d must equal config.num_hidden_layers.
    """
    if k < 1:
        raise ValueError("k must be >= 1 for ETD (use baseline_forward_logits for standard one pass).")

    base = model.model
    cfg = model.config
    l_total = cfg.num_hidden_layers
    n_d = l_total - n_e - n_t
    if n_d < 1:
        raise ValueError(f"Invalid split: n_e={n_e}, n_t={n_t} => n_d={n_d} < 1")
    if n_e < 1 or n_t < 1:
        raise ValueError("n_e and n_t must be >= 1")

    device = input_ids.device
    dtype = model.dtype
    inputs_embeds = base.embed_tokens(input_ids)

    past_seen_tokens = 0
    batch, seq_len = inputs_embeds.shape[:2]
    position_ids = _prepare_position_ids(attention_mask, past_seen_tokens, batch, seq_len, device)

    past_key_values = None
    use_cache = False

    mask_kwargs = {
        "config": cfg,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
    }
    causal_mask_mapping: dict[str, torch.Tensor] = {
        "full_attention": create_causal_mask(**mask_kwargs),
    }
    if getattr(base, "has_sliding_layers", False):
        from transformers.masking_utils import create_sliding_window_causal_mask

        causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    position_embeddings = base.rotary_emb(inputs_embeds, position_ids)
    hidden_states = inputs_embeds

    def run_layer(layer_idx: int, hs: torch.Tensor) -> torch.Tensor:
        layer_type = cfg.layer_types[layer_idx]
        attn_mask = causal_mask_mapping[layer_type]
        return base.layers[layer_idx](
            hs,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
        )

    for i in range(n_e):
        hidden_states = run_layer(i, hidden_states)

    for _ in range(k):
        for i in range(n_e, n_e + n_t):
            hidden_states = run_layer(i, hidden_states)

    for i in range(n_e + n_t, l_total):
        hidden_states = run_layer(i, hidden_states)

    hidden_states = base.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return logits


@torch.inference_mode()
def baseline_forward_logits(model, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """Standard single-pass forward (for sanity check / baseline)."""
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return out.logits


def loglikelihood_continuation(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_len: int,
) -> float:
    """Sum log p(token[i] | prefix) for i in [prompt_len, seq) using logits from a single forward."""
    total = 0.0
    seq_len = input_ids.shape[1]
    for i in range(prompt_len, seq_len):
        prev = logits[0, i - 1].float()
        tgt = int(input_ids[0, i].item())
        logp = F.log_softmax(prev, dim=-1)[tgt]
        total += float(logp.item())
    return total


def predict_mc_choice(
    model,
    tokenizer,
    prefix: str,
    continuations: list[str],
    n_e: int,
    n_t: int,
    k: int | None,
    device: torch.device,
) -> int:
    """
    Multiple-choice by summed token log-likelihood of each continuation after `prefix`.
    Returns index of the winning continuation.
    """
    scores: list[float] = []
    for cont in continuations:
        full = prefix + cont
        enc = tokenizer(full, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask")
        attention_mask = attn.to(device) if attn is not None else None
        pref = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)
        prompt_len = pref["input_ids"].shape[1]

        if k is None:
            logits = baseline_forward_logits(model, input_ids, attention_mask)
        else:
            logits = etd_forward_logits(model, input_ids, attention_mask, n_e, n_t, k)
        scores.append(loglikelihood_continuation(logits, input_ids, prompt_len))

    best = max(range(len(scores)), key=lambda i: scores[i])
    return best
