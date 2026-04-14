#!/usr/bin/env bash
# Source from experiments/*.sh (after `cd` to loop_layer repo root).
# - If hf-mirror is reachable without HTTP(S) proxy: clear proxies, default HF_ENDPOINT to mirror.
# - Else: source /etc/network_turbo (academic proxy), then prefer mirror, else huggingface.co.
#
# Skip entirely: HF_SKIP_NETWORK_PROBE=1
# Force-keep your own proxy: set vars after sourcing this file.

if [[ -n "${HF_SKIP_NETWORK_PROBE:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

_hf_mirror_ok() {
  curl -sf --connect-timeout 6 --max-time 25 "https://hf-mirror.com/api/datasets/" >/dev/null 2>&1
}

if _hf_mirror_ok; then
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  echo "[hf_hub_network_env] hf-mirror OK without proxy -> HF_ENDPOINT=$HF_ENDPOINT"
  return 0 2>/dev/null || exit 0
fi

if [[ -f /etc/network_turbo ]]; then
  # shellcheck source=/dev/null
  source /etc/network_turbo
fi

if _hf_mirror_ok; then
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  echo "[hf_hub_network_env] hf-mirror OK via proxy -> HF_ENDPOINT=$HF_ENDPOINT"
else
  export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
  echo "[hf_hub_network_env] mirror still unreachable -> HF_ENDPOINT=$HF_ENDPOINT (official hub, proxy active)"
fi

return 0 2>/dev/null || exit 0
