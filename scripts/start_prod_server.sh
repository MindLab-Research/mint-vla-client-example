#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

set -a
. ./configs/prod_volcano.env.sh
. ./.secrets.env
set +a

if [ -n "${TINKER_GATEWAY_GLM51_BASE_URL:-}" ]; then
  if [ "${TINKER_GATEWAY_GLM51_AUTH_MODE:-static_api_key}" != "static_api_key" ]; then
    echo "unsupported GLM5.1 gateway auth mode: ${TINKER_GATEWAY_GLM51_AUTH_MODE}" >&2
    exit 1
  fi
  if [ -z "${TINKER_API_KEY:-}" ]; then
    echo "missing TINKER_API_KEY for GLM5.1 static gateway auth" >&2
    exit 1
  fi
  export TINKER_GATEWAY_CONFIG_JSON="$(python - <<'PY'
import json
import os

raw = os.environ.get("TINKER_GATEWAY_CONFIG_JSON", "").strip()
cfg = json.loads(raw) if raw else {}
model_to_upstream = dict(cfg.get("model_to_upstream") or {})
upstreams = dict(cfg.get("upstreams") or {})
alias = os.environ["TINKER_GATEWAY_GLM51_ALIAS"].strip()
model = os.environ["TINKER_GATEWAY_GLM51_MODEL"].strip()
base_url = os.environ["TINKER_GATEWAY_GLM51_BASE_URL"].strip().rstrip("/")
if not alias or not model or not base_url:
    raise SystemExit("GLM5.1 gateway config is incomplete")
model_to_upstream[model] = alias
upstreams[alias] = {
    "base_url": base_url,
    "auth_mode": "static_api_key",
    "api_key": os.environ["TINKER_API_KEY"].strip(),
}
cfg["model_to_upstream"] = model_to_upstream
cfg["upstreams"] = upstreams
print(json.dumps(cfg, separators=(",", ":")))
PY
)"
fi

api_tmp_root="${MINT_TMP_ROOT}/api/${USER:-unknown}"
api_tmp_link="/tmp/mpa"
mkdir -p "$api_tmp_root"
if [ -L "$api_tmp_link" ] || [ -f "$api_tmp_link" ]; then
  rm -f "$api_tmp_link"
elif [ -e "$api_tmp_link" ]; then
  echo "refusing to replace non-link temp path: $api_tmp_link" >&2
  exit 1
fi
ln -s "$api_tmp_root" "$api_tmp_link"

export TMPDIR="${api_tmp_link}/t"
export XDG_CACHE_HOME="${api_tmp_link}/c"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${TINKER_RUNTIME_CHECKPOINT_DIR}"

exec "${PFS_RUNTIME_ENV_ROOT}/host-venv/bin/python" scripts/run_server.py
