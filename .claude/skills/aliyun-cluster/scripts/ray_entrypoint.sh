#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_OFFLINE=1
export HF_HOME=/vePFS-Mindverse/share/huggingface
export PYTHONDONTWRITEBYTECODE=1

if ! test -d /vePFS-Mindverse/share; then
  echo "ERROR: /vePFS-Mindverse/share missing (CPFS not mounted?)" >&2
  sleep 300
  exit 1
fi

echo "CPFS_MOUNT $(mount | grep -F ' /vePFS-Mindverse ' || true)" >&2

ckpt_root="/vePFS-Mindverse/share/mint_checkpoints"
if ! mkdir -p "${ckpt_root}" 2>/dev/null; then
  echo "ERROR: CPFS is not writable (mkdir failed): ${ckpt_root}" >&2
  exit 1
fi

probe_file="${ckpt_root}/.rw_probe_${HOSTNAME:-unknown}_$(date +%s)"
if ! (echo "probe" >"${probe_file}") 2>/dev/null; then
  echo "ERROR: CPFS is not writable (write failed): ${probe_file}" >&2
  exit 1
fi
rm -f "${probe_file}" 2>/dev/null || true

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"
ray_client_pkg="/vePFS-Mindverse/share/code/ray-client-pkg"
if test -d "${ray_client_pkg}"; then
  export PYTHONPATH="${ray_client_pkg}:${repo_root}:${PYTHONPATH:-}"
else
  export PYTHONPATH="${repo_root}:${PYTHONPATH:-}"
fi

role="worker"

# DLC injects role env vars (RANK/PAI_TASK_ROLE) that cannot be reliably overridden by `dlc submit ... --envs`.
# Use an explicit override that takes precedence over injected values.
if [ -n "${MINT_RAY_ROLE:-}" ]; then
  case "${MINT_RAY_ROLE}" in
    head|worker) role="${MINT_RAY_ROLE}" ;;
    *)
      echo "ERROR: MINT_RAY_ROLE must be 'head' or 'worker', got ${MINT_RAY_ROLE}" >&2
      exit 2
      ;;
  esac
elif [ "${RANK:-}" = "0" ]; then
  role="head"
elif [ -n "${PAI_TASK_ROLE:-}" ]; then
  case "${PAI_TASK_ROLE}" in
    head|master) role="head" ;;
    *) role="worker" ;;
  esac
fi

echo "ray_entrypoint role=${role} hostname=${HOSTNAME:-}" >&2

if [ "${role}" = "head" ]; then
  if [ "${EXTERNAL_RAY_HEAD:-}" = "1" ]; then
    echo "ray_entrypoint external_head=1 role=head: sleeping" >&2
    while true; do sleep 3600; done
  fi
  python3 - <<'PY'
import time

import ray
from ray._private.node import Node
from ray._private.parameter import RayParams

ip = ray.util.get_node_ip_address()
print("RAY_VERSION", ray.__version__, flush=True)
print("RAY_HEAD_IP", ip, flush=True)

ray_params = RayParams(
    num_cpus=4,
    num_gpus=0,
    include_dashboard=False,
    gcs_server_port=6379,
    ray_client_server_port=10001,
)
Node(ray_params, head=True, shutdown_at_exit=False, spawn_reaper=False)

while True:
    time.sleep(3600)
PY
else
  head_host="${HEAD_IP:-${MASTER_ADDR:-}}"
  if [ -z "${head_host}" ]; then
    echo "ERROR: HEAD_IP and MASTER_ADDR are empty (cannot find Ray head host)" >&2
    sleep 300
    exit 1
  fi
  export HEAD_IP="${head_host}"

  num_gpus="${RAY_NUM_GPUS:-8}"
  export RAY_NUM_GPUS="${num_gpus}"

  num_cpus="${RAY_NUM_CPUS:-}"
  export RAY_NUM_CPUS="${num_cpus}"

  python3 - <<'PY'
import os
import socket
import time

import ray
from ray._private.node import Node
from ray._private.parameter import RayParams

head = os.environ["HEAD_IP"]
num_gpus = int(os.environ.get("RAY_NUM_GPUS", "8"))
raw_num_cpus = os.environ.get("RAY_NUM_CPUS", "").strip()
num_cpus = int(raw_num_cpus) if raw_num_cpus else None

ip = ray.util.get_node_ip_address()
print("RAY_VERSION", ray.__version__, flush=True)
print("RAY_WORKER_IP", ip, "head", head, "num_gpus", num_gpus, "num_cpus", num_cpus, flush=True)

for _ in range(240):
    try:
        with socket.create_connection((head, 6379), timeout=2):
            break
    except OSError:
        print("waiting_for_ray_head", head, flush=True)
        time.sleep(2)

ray_params = RayParams(
    gcs_address=f"{head}:6379",
    num_gpus=num_gpus,
    num_cpus=num_cpus,
)
Node(ray_params, head=False, shutdown_at_exit=False, spawn_reaper=False)
while True:
    time.sleep(3600)
PY
fi
