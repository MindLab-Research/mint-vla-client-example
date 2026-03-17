#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

set -a
. ./configs/prod_volcano.env.sh
. ./.secrets.env
set +a

exec "${PFS_RUNTIME_ENV_ROOT}/host-venv/bin/python" scripts/run_server.py
