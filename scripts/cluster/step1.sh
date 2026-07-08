#!/usr/bin/env bash
# step1.sh — rsync 本地仓库根 -> server 代码根。
# 脚本已移到 scripts/cluster/,显式解析仓库根(脚本目录上两级),
# 避免 rsync 误同步子目录。
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
rsync -a --exclude '.git' --exclude '__pycache__' \
  "${REPO_ROOT}/" /vePFS-Mindverse/share/code/wenxi/dev_vla_wenxi/