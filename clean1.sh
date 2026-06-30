#!/usr/bin/env bash
# clean1.sh — 杀掉本机的 dev server 进程（本地化，无 ssh）
#
# 替代远程 `ssh driver 'kill ... run_server.py'`。单机部署下 server 就跑在本台机，
# 直接按命令行特征杀掉即可。start_dev_server.sh 最终拉起的是 scripts/run_server.py。
set -euo pipefail

PIDS=$(pgrep -f "scripts/run_server.py" || true)
if [ -z "${PIDS}" ]; then
  echo "no run_server.py process found"
  exit 0
fi

echo "killing run_server.py: ${PIDS}"
# shellcheck disable=SC2086
kill ${PIDS} 2>/dev/null || true
sleep 2

# 仍在的话强杀
PIDS=$(pgrep -f "scripts/run_server.py" || true)
if [ -n "${PIDS}" ]; then
  echo "still alive, SIGKILL: ${PIDS}"
  # shellcheck disable=SC2086
  kill -9 ${PIDS} 2>/dev/null || true
fi
echo "done"
