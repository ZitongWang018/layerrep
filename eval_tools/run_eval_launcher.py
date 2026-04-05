#!/usr/bin/env python3
"""
以编程方式启动与 run_eval_qwen.sh 相同的评测（便于在 IDE 里改参）。
依赖: 已安装 lm-evaluation-harness[hf,math] 与 wandb。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HARNESS_ROOT = Path(os.environ.get("HARNESS_ROOT", "/root/autodl-tmp/lm-evaluation-harness"))
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = os.environ.get("MODEL_PATH", "/root/autodl-tmp/model_qwen")
RUN_NAME = os.environ.get("RUN_NAME", "")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "lm-eval-qwen3-8b")
WANDB_GROUP = os.environ.get("WANDB_GROUP", "default")
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", str(SCRIPT_DIR / "results")))
APPLY_CHAT = os.environ.get("APPLY_CHAT", "true").lower() == "true"

TASKS = [
    "triviaqa",
    "gsm8k",
    "hendrycks_math",
    "commonsense_qa",
    "boolq",
    "arc_easy",
    "arc_challenge",
    "humaneval",
    "truthfulqa_mc1",
]


def main() -> int:
    from datetime import datetime

    run_name = RUN_NAME or f"qwen3-8b-eval-{datetime.now():%Y%m%d-%H%M%S}"
    out_dir = OUTPUT_ROOT / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        sys.executable,
        "-m",
        "lm_eval",
        "run",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={MODEL_PATH},dtype=auto,trust_remote_code=true",
        "--tasks",
        *TASKS,
        "--num_fewshot",
        "0",
        "--batch_size",
        "auto",
        "--device",
        "cuda",
        "--output_path",
        str(out_dir),
        "--trust_remote_code",
        "--confirm_run_unsafe_code",
    ]
    wandb_kv = [
        f"project={WANDB_PROJECT}",
        f"name={run_name}",
        f"group={WANDB_GROUP}",
        "job_type=lm-eval",
    ]
    entity = os.environ.get("WANDB_ENTITY")
    if entity:
        wandb_kv.append(f"entity={entity}")
    cmd.extend(["--wandb_args", *wandb_kv])
    cmd.extend(
        [
            "--wandb_config_args",
            f"model_path={MODEL_PATH}",
            "model_name=Qwen3-8B",
            "eval_suite=core9",
            f"run_name={run_name}",
        ]
    )

    if APPLY_CHAT:
        cmd.append("--apply_chat_template")

    env = os.environ.copy()
    # HumanEval / code_eval：与 HF 评测库要求一致，须在子进程中启用
    env.setdefault("HF_ALLOW_CODE_EVAL", "1")

    print("Command:", " ".join(cmd))
    print("Output:", out_dir)
    r = subprocess.run(cmd, cwd=str(HARNESS_ROOT), env=env)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
