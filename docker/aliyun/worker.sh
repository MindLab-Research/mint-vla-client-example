#!/usr/bin/env bash
# Aliyun PAI-DLC GPU worker job template
#
# Prerequisites:
#   - Head DLC job is running (docker/aliyun/head.sh)
#   - HEAD_IP is set to the head pod's IP
#   - dlc CLI installed with valid credentials
#   - Resource queue determined (run: source docker/aliyun/dlc-lib.sh && dlc-list-quotas)
#
# Usage:
#   # Submit COUNT workers (1..N), atomic batch
#   DLC_RESOURCE_ID=<quota_id> HEAD_IP=10.x.x.x NAME_PREFIX=mint-qwen COUNT=4 docker/aliyun/worker.sh
#
#   # Submit a single worker with a specific ID (for scaling out)
#   DLC_RESOURCE_ID=<quota_id> HEAD_IP=10.x.x.x NAME_PREFIX=mint-qwen WORKER_ID=5 docker/aliyun/worker.sh
#
#   # Cross-queue: workers from a different queue can join the same head
#   DLC_RESOURCE_ID=<other_quota_id> HEAD_IP=<head_ip> NAME_PREFIX=mint-qwen WORKER_ID=6 docker/aliyun/worker.sh
#
# Environment variables:
#   DLC_RESOURCE_ID   (REQUIRED — run dlc-list-quotas to see options)
#   HEAD_IP           (REQUIRED — Ray head pod IP)
#   NAME_PREFIX       (REQUIRED — e.g. mint-qwen)
#   DLC_WORKSPACE_ID  (default: 341495)
#   DLC_IMAGE          (default: mint:latest-sm90)
#   COUNT              (default: 1, submit workers 1..N)
#   WORKER_ID          (optional, submit a single worker with this ID; overrides COUNT)
#   WORKER_CPU         (default: 160)
#   WORKER_MEMORY      (default: 1600Gi)
#   WORKER_GPU         (default: 8)
#   WORKER_SHARED_MEM  (default: 600)
#   RAY_NUM_CPUS       (default: 16)

# ─── Data source: dataset mount (NOT storage mount) ────────────────
# We use --data_sources (dataset ID + version + mount path) instead of
# --data_source_uris (raw CPFS/NAS URI). Dataset mounts are versioned,
# access-controlled, and isolated per workspace. Raw URI mounts bypass
# dataset access policies and can expose unrelated paths — do not use
# --data_source_uris unless you have a specific reason and understand
# the isolation trade-offs.
#
# Mounted dataset:
#   d-t3o24m34nmm1oksycx:v2 → /vePFS-Mindverse/share/  (RW, shared code + runtime)
#
# Not mounted (intentionally):
#   d-tnvczkoow0apjilnsn (user/nolanho) — personal dir, not needed by DLC jobs

set -euo pipefail

DLC_WORKSPACE_ID="${DLC_WORKSPACE_ID:-341495}"
DLC_RESOURCE_ID="${DLC_RESOURCE_ID:?DLC_RESOURCE_ID is required — run 'source docker/aliyun/dlc-lib.sh && dlc-list-quotas' to see options}"
DLC_IMAGE="${DLC_IMAGE:-acr-qhxx-registry.cn-beijing.cr.aliyuncs.com/mindverse/mint:latest-sm90}"
DLC_DATASETS="${DLC_DATASETS:-d-t3o24m34nmm1oksycx:v2:/vePFS-Mindverse/share/}"
NAME_PREFIX="${NAME_PREFIX:?NAME_PREFIX is required — e.g. mint-qwen}"
COUNT="${COUNT:-1}"
WORKER_CPU="${WORKER_CPU:-160}"
WORKER_MEMORY="${WORKER_MEMORY:-1600Gi}"
WORKER_SHARED_MEMORY="${WORKER_SHARED_MEMORY:-600}"
WORKER_GPU="${WORKER_GPU:-8}"
WORKER_PRIORITY="${WORKER_PRIORITY:-1}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-16}"
HEAD_IP="${HEAD_IP:?HEAD_IP is required — set it to the Ray head pod IP}"

# Determine which worker IDs to submit.
if [ -n "${WORKER_ID:-}" ]; then
  worker_ids=("${WORKER_ID}")
else
  worker_ids=()
  for i in $(seq 1 "${COUNT}"); do
    worker_ids+=("${i}")
  done
fi

# Pre-check: ensure no worker name collides with an existing DLC job.
existing_names=""
existing_names="$(dlc get job \
  --workspace_id "${DLC_WORKSPACE_ID}" \
  --page_num 1 --page_size 200 \
  2>/dev/null | awk '{print $2}' || true)"

collision=""
for id in "${worker_ids[@]}"; do
  name="${NAME_PREFIX}-${id}"
  if echo "${existing_names}" | grep -qx "${name}"; then
    collision="${collision} ${name}"
  fi
done

if [ -n "${collision}" ]; then
  echo "ERROR: name collision detected — aborting (no jobs submitted)" >&2
  echo "Already exist:${collision}" >&2
  echo "Cancel them first or use a different NAME_PREFIX." >&2
  exit 1
fi

# All names are clear — submit all workers.
total="${#worker_ids[@]}"
idx=0
for id in "${worker_ids[@]}"; do
  idx=$((idx + 1))
  name="${NAME_PREFIX}-${id}"
  echo "--- Submitting ${name} (${idx}/${total}) on queue ${DLC_RESOURCE_ID} ---"
  dlc submit pytorchjob \
    --name "${name}" \
    --workspace_id "${DLC_WORKSPACE_ID}" \
    --resource_id "${DLC_RESOURCE_ID}" \
    --masters 1 \
    --workers 0 \
    --master_image "${DLC_IMAGE}" \
    --master_cpu "${WORKER_CPU}" \
    --master_memory "${WORKER_MEMORY}" \
    --master_shared_memory "${WORKER_SHARED_MEMORY}Gi" \
    --master_gpu "${WORKER_GPU}" \
    --priority "${WORKER_PRIORITY}" \
    --data_sources "${DLC_DATASETS}" \
    --envs "MINT_RAY_ROLE=worker,HEAD_IP=${HEAD_IP},RAY_NUM_GPUS=${WORKER_GPU},RAY_NUM_CPUS=${RAY_NUM_CPUS}" \
    --command "bash /vePFS-Mindverse/share/code/tinker-server-aliyun/.claude/skills/aliyun-cluster/scripts/ray_entrypoint.sh"
done

echo "--- Submitted ${total} worker(s) on queue ${DLC_RESOURCE_ID} ---"
