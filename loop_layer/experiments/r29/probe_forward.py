"""
Single probe forward: collect per-layer signals (hidden states, attention weights, gate activations).
Requires attn_implementation='eager' so attention weights are returned.
"""
from __future__ import annotations

from typing import Any

import torch

from r29.signal_funcs import (
    attn_entropy_from_weights,
    attention_locality_from_weights,
    attn_sink_ratio,
    ffn_gate_norm,
    head_specialization_from_weights,
    layer_cosine_sim,
    logit_lens_kl_last_token,
    participation_ratio,
    prediction_flip_rate_last_token,
    residual_write_norm,
)


def collect_probe_signals(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    n_layers: int,
) -> dict[int, dict[str, float]]:
    """
    Returns signals[l] for l in 0..n_layers-1 (layer index after decoder layer l).
    Layer 0 uses embed vs layer0 output for sim/write/flip where needed.
    """
    base = model.model
    ln_f = base.norm
    lm_head = model.lm_head

    hidden_per_layer: list[torch.Tensor | None] = [None] * n_layers
    attn_weights_per_layer: list[torch.Tensor | None] = [None] * n_layers
    gate_act_per_layer: list[torch.Tensor | None] = [None] * n_layers

    hooks = []

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

    def gate_hook(li: int):
        def fn(_m, _inp, out):
            gate_act_per_layer[li] = out.detach()

        return fn

    for li in range(n_layers):
        hooks.append(base.layers[li].register_forward_hook(hid_hook(li)))
        hooks.append(base.layers[li].self_attn.register_forward_hook(attn_hook(li)))
        hooks.append(base.layers[li].mlp.act_fn.register_forward_hook(gate_hook(li)))

    with torch.no_grad():
        _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

    for h in hooks:
        h.remove()

    h_final = hidden_per_layer[n_layers - 1]
    if h_final is None:
        raise RuntimeError("Probe failed: no final hidden state")

    # Pre-layer-0 "previous" = embeddings
    with torch.no_grad():
        inputs_embeds = base.embed_tokens(input_ids)
    seq_len = input_ids.shape[1]

    out: dict[int, dict[str, float]] = {}

    for li in range(n_layers):
        h = hidden_per_layer[li]
        if h is None:
            continue
        rec: dict[str, float] = {}

        if li == 0:
            h_prev = inputs_embeds
        else:
            h_prev = hidden_per_layer[li - 1]
            if h_prev is None:
                h_prev = inputs_embeds

        rec["layer_sim"] = layer_cosine_sim(h, h_prev)
        rec["residual_write_norm"] = residual_write_norm(h, h_prev)
        rec["participation_ratio"] = participation_ratio(h)
        rec["logit_lens_KL"] = logit_lens_kl_last_token(h, h_final, ln_f, lm_head)
        rec["prediction_flip_rate"] = prediction_flip_rate_last_token(h, h_prev, ln_f, lm_head)

        aw = attn_weights_per_layer[li]
        if aw is not None:
            rec["attn_entropy"] = attn_entropy_from_weights(aw)
            rec["head_specialization"] = head_specialization_from_weights(aw)
            rec["attention_locality"] = attention_locality_from_weights(aw)
            rec["attn_sink_ratio"] = attn_sink_ratio(aw, sink_idx=0)
        else:
            rec["attn_entropy"] = float("nan")
            rec["head_specialization"] = float("nan")
            rec["attention_locality"] = float("nan")
            rec["attn_sink_ratio"] = float("nan")

        ga = gate_act_per_layer[li]
        if ga is not None:
            rec["ffn_gate_norm"] = ffn_gate_norm(ga)
        else:
            rec["ffn_gate_norm"] = float("nan")

        out[li] = rec

    return out


def signals_dict_to_lists(
    per_layer: dict[int, dict[str, float]], n_layers: int, keys: list[str]
) -> dict[str, list[float]]:
    rows: dict[str, list[float]] = {k: [] for k in keys}
    for li in range(n_layers):
        d = per_layer.get(li, {})
        for k in keys:
            rows[k].append(float(d.get(k, float("nan"))))
    return rows
