#!/usr/bin/env python3
"""
Import R30 Qwen3 layer sweep from experiments/results/r30_sweep_results.json into this folder,
normalize to etd_layer_sweep_r30style.json, and regenerate the same figure set as Llama/Gemma.
No GPU; idempotent.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/root/autodl-tmp/loop_layer")
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np

from multimodel_sweep_visualize import generate_all_figures

SRC = ROOT / "experiments" / "results" / "r30_sweep_results.json"
OUT_JSON = ROOT / "experiments" / "qwen3-8b" / "results" / "etd_layer_sweep_r30style.json"
OUT_TXT = ROOT / "experiments" / "qwen3-8b" / "results" / "etd_layer_sweep_top_configs.txt"
FIG_DIR = ROOT / "experiments" / "qwen3-8b" / "figures"


def write_top_configs_txt(path: Path, results: list[dict], baseline: dict[str, float], benches: list[str]) -> None:
    best_by_bench: dict[str, dict] = {}
    for b in benches:
        best_by_bench[b] = max(results, key=lambda r: r[b])
    best_macro = max(results, key=lambda r: r["macro_avg"])
    mb = float(np.mean([baseline[b] for b in benches]))
    lines = [
        "# Imported from R30 sweep — Qwen3-8B",
        f"# {datetime.now(timezone.utc).isoformat()}",
        "",
    ]
    for b in benches:
        row = best_by_bench[b]
        lines.append(
            f"{b}: best_acc={row[b]:.4f} baseline={baseline[b]:.4f} "
            f"t_start={row['t_start']} t_stop={row['t_stop']} n_t={row['n_t']}"
        )
    lines.append(f"macro_baseline (mean of bench baselines)={mb:.4f}")
    r = best_macro
    lines.append(
        f"MACRO: best_etd_macro={r['macro_avg']:.4f} delta_vs_macro_baseline={r['macro_avg'] - mb:+.4f} "
        f"t_start={r['t_start']} t_stop={r['t_stop']}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not SRC.is_file():
        raise SystemExit(f"Missing source file: {SRC}")

    raw = json.loads(SRC.read_text(encoding="utf-8"))
    cfg = dict(raw["config"])
    cfg.setdefault("preset", "qwen3-8b")
    cfg["t_stop_exclusive"] = True

    benches = list(raw["benchmarks_used"])
    baseline = dict(raw["baseline"])
    results = list(raw["results"])
    mb = float(np.mean([baseline[b] for b in benches]))

    best_macro = max(results, key=lambda r: r["macro_avg"])
    best_per: dict[str, dict] = {}
    for b in benches:
        br = max(results, key=lambda r: r[b])
        best_per[b] = {k: br[k] for k in ("t_start", "t_stop", "n_t", b, "macro_avg")}
    summary = {
        "best_per_benchmark": best_per,
        "best_macro_avg_cell": {
            k: best_macro[k] for k in ("t_start", "t_stop", "n_t", "macro_avg") if k in best_macro
        },
        "macro_baseline": round(mb, 6),
    }

    payload = {
        "config": cfg,
        "baseline": baseline,
        "macro_baseline": round(mb, 6),
        "note_baseline": (
            "Standard 1× forward (no ETD), from R30 sweep file. "
            "Compared against ETD (k=2, alpha=auto) per grid cell."
        ),
        "benchmarks_used": benches,
        "source_import": str(SRC.relative_to(ROOT)),
        "imported_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "summary": summary,
    }
    for k in ("mmlu_note",):
        if k in raw:
            payload[k] = raw[k]

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")

    write_top_configs_txt(OUT_TXT, results, baseline, benches)
    print(f"Wrote {OUT_TXT}")

    paths = generate_all_figures(payload, FIG_DIR, "QWEN3 8B")
    for p in paths:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
