#!/usr/bin/env bash
# Qwen3-8B 本地权重 + 9 项 benchmark，结果同步 Weights & Biases（个人账号：wandb login 即可，不必设 ENTITY）
# 依赖: pip install -e "/path/to/lm-evaluation-harness[hf]" wandb
#
# 环境变量（可选）:
#   MODEL_PATH      模型目录，默认 /root/autodl-tmp/model_qwen
#   RUN_NAME        本次 run 名称（W&B 与输出目录），默认 qwen3-8b-eval-时间戳
#   WANDB_PROJECT   W&B 项目名，默认 lm-eval-qwen3-8b（个人主页下可见）
#   WANDB_GROUP     实验组（多轮实验用同一组名便于对比），默认 default
#   WANDB_ENTITY    仅团队/组织项目需要；个人账号不要设
#   APPLY_CHAT      true/false，是否对 instruct 模型用 chat template，默认 true
#   BATCH_SIZE      默认 auto
#   OUTPUT_ROOT     结果根目录，默认本目录下 results/
#   USE_HF_MIRROR  默认 1：使用 HF 国内镜像下载评测数据；设为 0 走官方源
#   HF_HUB_DOWNLOAD_TIMEOUT  数据集单文件下载超时（秒），默认 600
#   HF_ALLOW_CODE_EVAL     HumanEval 必需，默认 1（同意在沙箱中执行模型生成代码做 pass@k）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 缓解 datasets 从 Hub 拉 parquet 时的 SSL 握手超时（ConnectTimeout）
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/hf_download_env.sh"

HARNESS_ROOT="${HARNESS_ROOT:-/root/autodl-tmp/lm-evaluation-harness}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/model_qwen}"
RUN_NAME="${RUN_NAME:-qwen3-8b-eval-$(date +%Y%m%d-%H%M%S)}"
WANDB_PROJECT="${WANDB_PROJECT:-lm-eval-qwen3-8b}"
WANDB_GROUP="${WANDB_GROUP:-default}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/results}"
APPLY_CHAT="${APPLY_CHAT:-true}"

OUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "${OUT_DIR}"

export WANDB_PROJECT

if [[ ! -d "${HARNESS_ROOT}/lm_eval" ]]; then
  echo "ERROR: lm-evaluation-harness not found at ${HARNESS_ROOT}"
  exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "ERROR: model not found or missing config.json: ${MODEL_PATH}"
  exit 1
fi

cd "${HARNESS_ROOT}"

# 任务说明:
#   hendrycks_math = MATH 数据集（多子域聚合）
#   truthfulqa_mc1 = TruthfulQA 多选题设置
# HumanEval 需确认执行模型生成代码（沙箱执行单测）
CHAT_FLAG=()
if [[ "${APPLY_CHAT}" == "true" ]]; then
  CHAT_FLAG=(--apply_chat_template)
fi

ENTITY_ARGS=()
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  ENTITY_ARGS=("entity=${WANDB_ENTITY}")
fi

echo "==> Model:     ${MODEL_PATH}"
echo "==> Run name:  ${RUN_NAME}"
echo "==> W&B:       project=${WANDB_PROJECT} group=${WANDB_GROUP}"
echo "==> Results:   ${OUT_DIR}"
echo

python -m lm_eval run \
  --model hf \
  --model_args "pretrained=${MODEL_PATH},dtype=auto,trust_remote_code=true" \
  --tasks triviaqa gsm8k hendrycks_math commonsense_qa boolq arc_easy arc_challenge humaneval truthfulqa_mc1 \
  --num_fewshot 0 \
  --batch_size "${BATCH_SIZE}" \
  --device cuda \
  "${CHAT_FLAG[@]}" \
  --trust_remote_code \
  --confirm_run_unsafe_code \
  --output_path "${OUT_DIR}" \
  --wandb_args "project=${WANDB_PROJECT}" "name=${RUN_NAME}" "group=${WANDB_GROUP}" "job_type=lm-eval" "${ENTITY_ARGS[@]}" \
  --wandb_config_args \
    "model_path=${MODEL_PATH}" \
    "model_name=Qwen3-8B" \
    "eval_suite=core9" \
    "run_name=${RUN_NAME}"
# 注意：不要在 --wandb_config_args 里写 tasks=a,b,c（逗号会被 lm-eval CLI 拆成多段，导致解析报错）。任务列表已由 --tasks 指定。

echo
echo "==> Done. Metrics JSON under: ${OUT_DIR}"
echo "==> Open W&B project '${WANDB_PROJECT}' to see charts and compare runs in group '${WANDB_GROUP}'."
