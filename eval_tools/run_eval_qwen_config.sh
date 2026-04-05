#!/usr/bin/env bash
# 使用 eval_qwen_core.yaml，并用环境变量覆盖模型路径、W&B 名称与输出目录（无需 PyYAML）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/hf_download_env.sh"

HARNESS_ROOT="${HARNESS_ROOT:-/root/autodl-tmp/lm-evaluation-harness}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/model_qwen}"
RUN_NAME="${RUN_NAME:-qwen3-8b-eval-$(date +%Y%m%d-%H%M%S)}"
WANDB_PROJECT="${WANDB_PROJECT:-lm-eval-qwen3-8b}"
WANDB_GROUP="${WANDB_GROUP:-default}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/results}"

OUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "${OUT_DIR}"

ENTITY_ARGS=()
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  ENTITY_ARGS=("entity=${WANDB_ENTITY}")
fi

cd "${HARNESS_ROOT}"

python -m lm_eval run \
  --config "${SCRIPT_DIR}/eval_qwen_core.yaml" \
  --model_args "pretrained=${MODEL_PATH},dtype=auto,trust_remote_code=true" \
  --output_path "${OUT_DIR}" \
  --wandb_args "project=${WANDB_PROJECT}" "name=${RUN_NAME}" "group=${WANDB_GROUP}" "job_type=lm-eval" "${ENTITY_ARGS[@]}" \
  --wandb_config_args \
    "model_path=${MODEL_PATH}" \
    "model_name=Qwen3-8B" \
    "run_name=${RUN_NAME}" \
    "eval_suite=core9" \
  --confirm_run_unsafe_code

echo "==> Results: ${OUT_DIR}"
