# 在运行评测前 source 此文件，减轻 Hugging Face 数据集下载超时、SSL 握手失败等问题。
# 用法: source /root/autodl-tmp/eval_tools/hf_download_env.sh
#
# USE_HF_MIRROR=0  — 不使用镜像，走官方 huggingface.co（海外机器可用）

if [[ "${USE_HF_MIRROR:-1}" == "1" ]]; then
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  echo "[hf] HF_ENDPOINT=${HF_ENDPOINT}"
else
  export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
  echo "[hf] HF_ENDPOINT=${HF_ENDPOINT} (official)"
fi

# 单次下载超时（秒），大文件可再加大
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"

# 避免 libgomp 对 OMP_NUM_THREADS 报错
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# HumanEval：HuggingFace evaluate 的 code_eval 要求显式同意执行模型生成代码（见官方 WARNING）
export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"

# 若仍出现 httpx SSL/代理握手超时，可临时取消代理再试：
#   unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
