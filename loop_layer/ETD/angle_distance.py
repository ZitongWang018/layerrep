"""Collect mean angular distances between adjacent layer hidden states (last token) and Kneedle splits."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from kneed import KneeLocator
from tqdm import tqdm


def _last_token_hidden(hidden_states: tuple, attention_mask: torch.Tensor) -> list[torch.Tensor]:
    """Per-layer last *non-pad* token vector, shape [batch, hidden]."""
    idx = attention_mask.long().sum(dim=1) - 1  # [batch]
    batch_size = attention_mask.shape[0]
    out: list[torch.Tensor] = []
    for h in hidden_states:
        # h: [batch, seq, dim]
        rows = []
        for b in range(batch_size):
            pos = int(idx[b].item())
            rows.append(h[b, pos])
        out.append(torch.stack(rows, dim=0))
    return out


def angular_distance_pair(h0: torch.Tensor, h1: torch.Tensor) -> torch.Tensor:
    """Mean over batch of (1/pi) arccos(cos sim) in [0,1]."""
    h0n = F.normalize(h0.float(), dim=-1)
    h1n = F.normalize(h1.float(), dim=-1)
    cos = (h0n * h1n).sum(dim=-1).clamp(-1.0, 1.0)
    return (torch.acos(cos) / math.pi).mean()


def collect_distances(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = 2,
    device: torch.device | str = "cuda",
) -> tuple[list[float], dict[str, Any]]:
    """
    Returns:
      distances: list of length L (num_layers), mean d(l,l+1) between hidden_states[l] and hidden_states[l+1].
      meta: batch count etc.
    """
    model.eval()
    device = torch.device(device) if isinstance(device, str) else device

    num_layers = model.config.num_hidden_layers
    # hidden_states: embed + each layer => num_layers + 1 states, num_layers gaps
    acc = torch.zeros(num_layers, device=device, dtype=torch.float64)
    total_weight = 0

    for start in tqdm(range(0, len(texts), batch_size), desc="angle batches"):
        batch_texts = texts[start : start + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        w = int(input_ids.shape[0])

        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        hs = out.hidden_states
        assert hs is not None and len(hs) == num_layers + 1
        lasts = _last_token_hidden(hs, attention_mask)

        for l in range(num_layers):
            d = angular_distance_pair(lasts[l], lasts[l + 1])
            acc[l] += d.double() * w
        total_weight += w

    mean_d = (acc / max(total_weight, 1)).cpu().numpy().tolist()
    meta = {"total_weight": total_weight, "num_texts": len(texts), "num_layers": num_layers}
    return mean_d, meta


def kneedle_split(distances: list[float], num_layers: int) -> tuple[int, int, int]:
    """
    Use Kneedle on forward d and reversed d to get N_E, N_T, N_D (counts of layers).
    distances length must be num_layers (gaps between adjacent hidden states).
    """
    L = num_layers
    d = np.array(distances, dtype=np.float64)
    if d.shape[0] != L:
        raise ValueError(f"Expected {L} distances, got {d.shape[0]}")

    x = np.arange(L, dtype=np.float64)

    def elbow(y: np.ndarray) -> int | None:
        kl = KneeLocator(x, y, curve="convex", direction="decreasing")
        return kl.elbow

    e0 = elbow(d)
    e1 = elbow(d[::-1])

    if e0 is None:
        e0 = L // 3
    if e1 is None:
        e1 = L // 3

    n_e = int(max(1, min(e0 + 1, L - 2)))
    n_d = int(max(1, min(e1 + 1, L - 2)))
    n_t = L - n_e - n_d
    if n_t < 1:
        if n_e >= n_d:
            n_e = max(1, L - n_d - 1)
        else:
            n_d = max(1, L - n_e - 1)
        n_t = L - n_e - n_d
    return n_e, n_t, n_d


def save_split(
    path: Path | str,
    distances: list[float],
    n_e: int,
    n_t: int,
    n_d: int,
    meta: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_e": n_e,
        "n_t": n_t,
        "n_d": n_d,
        "distances": distances,
        "meta": meta,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_split(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
