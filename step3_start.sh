#!/usr/bin/env bash
# step3_start.sh — 本机化起 mint-server（替代三机版 step3.sh 的 ssh driver 包装）
#
# 关键差异 vs 三机版：
#   - 无 ssh driver：server 直接在本机起。
#   - MINT_RAY_GCS_ADDRESS=127.0.0.1:6379：显式直连本机 GCS（不读 ray_head_ip.txt，
#     不用 ray:// client 模式）。
#   - PFS_RUNTIME_ENV_ROOT=/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime：用固定后的本地 runtime（脱离 share）。
#   - MINT_QWEN36_DEPS_PATH / MINT_QWEN36_VLLM_DEPS_PATH：指向本地 qwen36 隔离栈
#     副本（gpu_rl tier 内的 symlink 已重指 /vePFS-Mindverse/user/intern/wenxi/mint_env/runtime/qwen36-stack）。
#
# code mirror 仍用 share（worker import mint_server 需所有 Ray 节点可见；单机下
# worker 在本机，PFS 挂载着 → 可见）。改完代码记得先跑 step1 的 rsync 再重启。
set -euo pipefail
PFS_RUNTIME_ENV_ROOT=/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime

RUNTIME_ROOT="${PFS_RUNTIME_ENV_ROOT:-/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime}"
CODE_ROOT="${MINT_CODE_ROOT:-/vePFS-Mindverse/share/code/wenxi/dev_vla_wenxi}"

MINT_CODE_ROOT="${CODE_ROOT}" \
MINT_DEV_USER=wenxi \
MINT_RAY_NAMESPACE=mint_wenxi_dev \
MINT_RAY_GCS_ADDRESS=127.0.0.1:6379 \
MINT_TASK_STATE_STORE_DB_PATH=/vePFS-Mindverse/share/mint/dev/data/wenxi/task-state/task_state.sqlite3 \
MINT_LOG_FILE=/vePFS-Mindverse/share/mint/dev/logs/mint-dev-server.log \
MINT_DISABLE_MINT_ROUTE=0 \
MINT_UVICORN_WORKERS=1 \
MINT_SUPERVISOR_STATE_BACKEND=memory \
MINT_SUPPORTED_MODELS="Qwen/Qwen3.6-27B,Qwen/Qwen3-0.6B,Qwen/Qwen3-4B-Instruct-2507,Qwen/Qwen3-30B-A3B-Instruct-2507" \
MINT_DEV_RUN_ENV=/tmp/mint_dev_run.env \
MINT_QWEN36_DEPS_PATH="${RUNTIME_ROOT}/gpu_rl/qwen36-deps" \
MINT_QWEN36_VLLM_DEPS_PATH="${RUNTIME_ROOT}/gpu_rl/qwen36-vllm-deps" \
PFS_RUNTIME_ENV_ROOT="${RUNTIME_ROOT}" \
nohup "${CODE_ROOT}/scripts/start_dev_server.sh" \
  >> /tmp/mint_dev_launch_wenxi.log 2>&1 &

echo "server 启动中，日志：/tmp/mint_dev_launch_wenxi.log"
echo "等 launcher 打印 MINT_PORT（namespace hash 派生，应为 30496），然后 step4 健康检查。"
