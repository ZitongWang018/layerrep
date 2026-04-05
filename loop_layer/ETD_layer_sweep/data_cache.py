"""Load fixed BoolQ / ARC evaluation examples (identical across all sweep runs)."""

from __future__ import annotations


def load_boolq_examples(split: str, limit: int) -> list[tuple[str, list[str], int]]:
    from datasets import load_dataset

    ds = load_dataset("aps/super_glue", "boolq")[split]
    out: list[tuple[str, list[str], int]] = []
    for i in range(len(ds)):
        if len(out) >= limit:
            break
        r = ds[i]
        lab = int(r["label"])
        if lab < 0:
            continue
        prefix = f"{r['passage']}\nQuestion: {r['question']}?\nAnswer:"
        conts = [" no", " yes"]
        out.append((prefix, conts, lab))
    return out


def load_arc_examples(split: str, limit: int) -> list[tuple[str, list[str], int]]:
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")[split]
    n = min(limit, len(ds))
    out: list[tuple[str, list[str], int]] = []
    for i in range(n):
        r = ds[i]
        q = r["question"].strip()
        texts = r["choices"]["text"]
        prefix = f"Question: {q}\nAnswer:"
        conts = [f" {t}" for t in texts]
        key = r["answerKey"]
        if isinstance(key, str) and key in "ABCD":
            label = ord(key) - ord("A")
        else:
            label = int(key) - 1
        out.append((prefix, conts, label))
    return out
