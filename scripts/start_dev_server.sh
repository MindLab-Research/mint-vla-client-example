#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

set -a
dev_config_env="${MINT_DEV_CONFIG_ENV:-/share/mint/dev/config/common.env}"
if [ ! -r "${dev_config_env}" ]; then
  echo "missing dev config: ${dev_config_env}" >&2
  exit 1
fi
. "${dev_config_env}"
dev_secrets_env="${MINT_DEV_SECRETS_ENV:-/share/mint/dev/config/secrets.env}"
if [ -r "${dev_secrets_env}" ]; then
  . "${dev_secrets_env}"
fi
set +a

api_tmp_root="${MINT_TMP_ROOT}/api/${USER:-unknown}"
api_tmp_link="/tmp/mda"
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
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${MINT_RUNTIME_CHECKPOINT_DIR}"

ray_head_ip_path="${MINT_RAY_HEAD_ADDRESS_PATH:-/vePFS-Mindverse/share/code/tinker-server/ray_head_ip.txt}"
ray_head_ip=""
if [ -r "$ray_head_ip_path" ]; then
  ray_head_ip=$(tr -d '[:space:]' < "$ray_head_ip_path")
fi
if [ -n "$ray_head_ip" ]; then
  case "${RAY_ADDRESS:-}" in
    ray://*) ;;
    *:6379|"") export RAY_ADDRESS="ray://${ray_head_ip}:10001" ;;
  esac
fi
if [ -z "${MINT_RAY_CLIENT_ADDRESS:-}" ]; then
  case "${RAY_ADDRESS:-}" in
    ray://*) export MINT_RAY_CLIENT_ADDRESS="${RAY_ADDRESS}" ;;
  esac
fi

exec "${PFS_RUNTIME_ENV_ROOT}/host-venv/bin/python" scripts/run_server.py
