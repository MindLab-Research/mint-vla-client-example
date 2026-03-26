#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

set -a
. ./configs/dev_volcano.env.sh
[ -f .secrets.env ] && . ./.secrets.env
export PFS_TINKER_PATH="${PFS_TINKER_PATH:-$repo_root}"
export PYTHONPATH="${PFS_EXTRA_PYTHONPATH:+$PFS_EXTRA_PYTHONPATH:}${PFS_VLLM_PATH:+$PFS_VLLM_PATH:}${PFS_MEGATRON_BRIDGE_HOLLOWMAN_PATH:+$PFS_MEGATRON_BRIDGE_HOLLOWMAN_PATH:}${PFS_VERL_PATH:+$PFS_VERL_PATH:}$PFS_TINKER_PATH${PFS_HF_MODULES_PATH:+:$PFS_HF_MODULES_PATH}"
set +a
exec "$PYTHON_BIN" scripts/run_server.py
