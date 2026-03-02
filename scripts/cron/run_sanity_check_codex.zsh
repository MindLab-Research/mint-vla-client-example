#!/bin/zsh
set -eo pipefail

cd /home/yiwen/tinker_project/tinker-server-prod

# Cron-safe environment: do not source interactive shell configs.
export PATH="$HOME/.npm-global/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

secrets_env="/home/yiwen/tinker_project/tinker-server-prod/.secrets.env"
if [ ! -f "$secrets_env" ]; then
  print -u2 "missing $secrets_env"
  exit 1
fi

set +u
source "$secrets_env"
set -u

: "${CRS_OAI_KEY:?CRS_OAI_KEY is not set after sourcing $secrets_env}"

if ! command -v codex >/dev/null 2>&1; then
  print -u2 "codex not found in PATH"
  exit 1
fi

codex --no-alt-screen exec --color=never 'Run the pipeline in the sanity-check skill'
