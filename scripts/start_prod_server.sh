#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

set -a
. ./configs/prod_volcano.env.sh
. ./.secrets.env
set +a

exec /root/tinker_project/tinker-server-auth/.venv31213/bin/python scripts/run_server.py
