#!/usr/bin/env bash
# step2_placement.sh — 本机化的 placement 生成（替代三机版 step2.sh）
#
# 三机版：读 ray_head_ip.txt → 查远程 dashboard → scp 到 driver。
# 单机版：head dashboard 在 127.0.0.1:8265；worker 自报 192.168.* 真实 IP。
#   --head-ip 127.0.0.1 同时满足：
#     (a) 查 dashboard http://127.0.0.1:8265
#     (b) 过滤 node_ip==head_ip 时不会误杀 192.168.* 的 worker
#   无 scp（server 就在本机）。
set -euo pipefail

PFS_RUNTIME_ENV_ROOT=/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime

RUNTIME_ROOT="${PFS_RUNTIME_ENV_ROOT:-/vePFS-Mindverse/user/intern/wenxi/mint_env/runtime}"
CPU_PY="${RUNTIME_ROOT}/cpu/base-python/bin/python3.13"
CPU_SP="${RUNTIME_ROOT}/cpu/site-packages"
CODE_ROOT="${MINT_CODE_ROOT:-/vePFS-Mindverse/share/code/wenxi/dev_vla_wenxi}"

rm -f /tmp/mint_dev_run.env

PYTHONPATH="${CPU_SP}:${CODE_ROOT}" "${CPU_PY}" "${CODE_ROOT}/scripts/tools/gen_dev_placement.py" \
  --head-ip 127.0.0.1 \
  --model Qwen/Qwen3.6-27B --gpu-count 4 \
  --output /tmp/mint_dev_run.env

echo "=== /tmp/mint_dev_run.env（确认 node_ip 是本机 worker 192.168.* IP）==="
cat /tmp/mint_dev_run.env
