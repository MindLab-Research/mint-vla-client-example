#!/bin/zsh
set -eo pipefail

cd /home/yiwen/tinker_project/tinker-server-prod

# Load CRS_OAI_KEY (and PATH) as defined in your interactive shell config.
# Cron does not load ~/.zshrc automatically.
if [ -f "$HOME/.zshrc" ]; then
  set +u
  source "$HOME/.zshrc"
  set -u
fi

: "${CRS_OAI_KEY:?CRS_OAI_KEY is not set after sourcing ~/.zshrc}"

if ! command -v codex >/dev/null 2>&1; then
  print -u2 "codex not found in PATH"
  exit 1
fi

codex exec 'Run the pipeline in the sanity-check skill'

