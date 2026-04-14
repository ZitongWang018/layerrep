"""
Hard multiple-choice benchmarks aligned with EleutherAI lm-evaluation-harness prompts.

- GPQA-Diamond: lm_eval/tasks/gpqa/zeroshot/_gpqa_zeroshot_yaml + gpqa_diamond_zeroshot.yaml
  Dataset gated: https://huggingface.co/datasets/Idavidrein/gpqa — accept terms + huggingface-cli login.
- AGIEval Gaokao MathQA: lm_eval/tasks/agieval/gaokao-mathqa.yaml (includes aqua-rat.yaml template)
  Dataset: hails/agieval-gaokao-mathqa
- LogiQA: lm_eval/tasks/logiqa/logiqa.yaml + utils_logiqa.py
  Dataset: EleutherAI/logiqa (config logiqa). On datasets>=3 (no loading_dataset scripts), falls back to
  fireworks-ai/logiqa (parquet; same passage/question/options structure, answers A–D).

Hub: unset HF_ENDPOINT defaults to https://hf-mirror.com (same as other experiments/ scripts). Use
  export HF_ENDPOINT=https://huggingface.co for the official API if the mirror is stale.

Shuffle policy: GPQA uses per-row deterministic shuffle (lm-eval uses unseeded random); we use random.Random(sha256).
"""
from __future__ import annotations

import hashlib
import os
import random
import re
from typing import Any

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from datasets import DownloadConfig, load_dataset


def _dc() -> DownloadConfig | None:
    if bool(int(os.environ.get("HF_DATASETS_OFFLINE", "0"))):
        return DownloadConfig(local_files_only=True)
    return None


def _stable_seed(doc_key: str) -> int:
    h = hashlib.sha256(doc_key.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") % (2**31)


def _gpqa_preprocess(text: str | None) -> str:
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def load_gpqa_diamond(n: int) -> list[dict]:
    """Multiple choice (A)-(D); prompt matches lm-eval GPQA zeroshot."""
    ds = load_dataset(
        "Idavidrein/gpqa",
        "gpqa_diamond",
        split="train",
        download_config=_dc(),
    )
    out: list[dict] = []
    for idx, r in enumerate(ds):
        if len(out) >= n:
            break
        q = str(r["Question"])
        choices_raw = [
            _gpqa_preprocess(r["Incorrect Answer 1"]),
            _gpqa_preprocess(r["Incorrect Answer 2"]),
            _gpqa_preprocess(r["Incorrect Answer 3"]),
            _gpqa_preprocess(r["Correct Answer"]),
        ]
        correct = _gpqa_preprocess(r["Correct Answer"])
        rng = random.Random(_stable_seed(q + correct))
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [choices_raw[i] for i in order]
        correct_idx = shuffled.index(correct)
        letters = ["(A)", "(B)", "(C)", "(D)"]
        # doc_to_text / doc_to_choice in lm-eval
        prompt = (
            f"What is the correct answer to this question:{q}\nChoices:\n"
            f"(A) {shuffled[0]}\n(B) {shuffled[1]}\n(C) {shuffled[2]}\n(D) {shuffled[3]}\nAnswer:"
        )
        choice_tokens = [letters[i] for i in range(4)]
        gold = letters[correct_idx].lower()
        out.append(
            {
                "prompt": prompt,
                "choices": choice_tokens,
                "answer": gold,
                "valid_indices": [correct_idx],
            }
        )
    return out


def load_agieval_gaokao_mathqa(n: int) -> list[dict]:
    """AGIEval Gaokao Math MC; doc_to_text = query (lm-eval aqua-rat template)."""
    dc = _dc()
    ds = load_dataset(
        "hails/agieval-gaokao-mathqa",
        split="test",
        download_config=dc,
    )
    out: list[dict] = []
    for r in ds:
        if len(out) >= n:
            break
        query = str(r["query"])
        ch = r["choices"]
        if hasattr(ch, "tolist"):
            ch = ch.tolist()
        choice_strs = [str(c).strip() for c in ch]
        gold = r["gold"]
        if hasattr(gold, "tolist"):
            gold = gold.tolist()
        valid = [int(x) for x in gold]
        if not choice_strs or not valid:
            continue
        # Primary label for logging; scoring uses valid_indices
        lab0 = valid[0]
        if lab0 < 0 or lab0 >= len(choice_strs):
            continue
        out.append(
            {
                "prompt": query,
                "choices": choice_strs,
                "answer": choice_strs[lab0].strip().lower(),
                "valid_indices": valid,
            }
        )
    return out


def _logiqa_label_to_letter(r: dict[str, Any]) -> str | None:
    """Normalize gold label to a single letter a–d (EleutherAI uses label index or letter; fireworks uses A–D)."""
    if "label" in r and r["label"] is not None:
        lab = r["label"]
    else:
        lab = r.get("answer")
    if isinstance(lab, (int, float)) and lab == int(lab):
        i = int(lab)
        if 0 <= i < 4:
            return "abcd"[i]
        return None
    s = str(lab).strip()
    if len(s) == 1:
        s = s.lower()
        if s in ("a", "b", "c", "d"):
            return s
        u = s.upper()
        if u in ("A", "B", "C", "D"):
            return u.lower()
    return None


def _strip_option_letter_prefix(opt: str) -> str:
    return re.sub(r"^[ABCDabcd]\.\s*", "", str(opt).strip())


def _logiqa_doc_to_text(doc: dict[str, Any]) -> str:
    choice_letters = ["a", "b", "c", "d"]
    prompt = "Passage: " + doc["context"] + "\n"
    prompt += "Question: " + doc["question"] + "\nChoices:\n"
    for letter, option in zip(choice_letters, doc["options"]):
        prompt += f"{letter.upper()}. {option}\n"
    prompt += "Answer:"
    return prompt


def load_logiqa(n: int) -> list[dict]:
    """LogiQA test split; continuations are single letters a–d (lm-eval multiple_choice)."""
    dc = _dc()
    try:
        ds = load_dataset(
            "EleutherAI/logiqa",
            "logiqa",
            split="test",
            download_config=dc,
        )
    except RuntimeError as e:
        err = str(e)
        if "no longer supported" in err or "logiqa.py" in err:
            ds = load_dataset(
                "fireworks-ai/logiqa",
                split="test",
                download_config=dc,
            )
        else:
            raise

    out: list[dict] = []
    for r in ds:
        if len(out) >= n:
            break
        label = _logiqa_label_to_letter(r)
        if label is None:
            continue
        opts = r["options"]
        if hasattr(opts, "tolist"):
            opts = opts.tolist()
        doc = {
            "context": str(r["context"]),
            "question": str(r["question"]),
            "options": [_strip_option_letter_prefix(o) for o in opts],
        }
        prompt = _logiqa_doc_to_text(doc)
        choices = ["a", "b", "c", "d"]
        out.append(
            {
                "prompt": prompt,
                "choices": choices,
                "answer": label,
                "valid_indices": [choices.index(label)],
            }
        )
    return out
