#!/usr/bin/env bash
# Aliyun PAI-DLC Ray Head job template — CPU node with large memory
#
# Ray head runs GCS + dashboard. Needs more memory than a typical CPU node.
# This is a DLC PyTorchJob, NOT a local process.
#
# Submit AFTER the DSW driver is up, BEFORE workers (workers need HEAD_IP).
#
# Prerequisites:
#   - dlc CLI installed with valid credentials (ALI_ACCESSKEY_ID / ALI_ACCESSKEY_SECRET)
#   - Resource queue determined (run: source docker/aliyun/dlc-lib.sh && dlc-list-quotas)
#
# Usage:
#   DLC_RESOURCE_ID=<quota_id> docker/aliyun/head.sh
#
# Environment variables:
#   DLC_RESOURCE_ID   (REQUIRED — run dlc-list-quotas to see options)
#   DLC_WORKSPACE_ID  (default: 341495)
#   DLC_IMAGE          (default: mint:latest-sm90)
#   NAME               (default: ray-head)
#   HEAD_CPU           (default: 8)
#   HEAD_MEMORY        (default: 32Gi)

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
NAME="${NAME:-ray-head}"
HEAD_CPU="${HEAD_CPU:-8}"
HEAD_MEMORY="${HEAD_MEMORY:-32Gi}"

dlc submit pytorchjob \
  --name "${NAME}" \
  --workspace_id "${DLC_WORKSPACE_ID}" \
  --resource_id "${DLC_RESOURCE_ID}" \
  --masters 1 \
  --workers 0 \
  --master_image "${DLC_IMAGE}" \
  --master_cpu "${HEAD_CPU}" \
  --master_memory "${HEAD_MEMORY}" \
  --master_gpu 0 \
  --priority 1 \
  --data_sources "${DLC_DATASETS}" \
  --envs "MINT_RAY_ROLE=head" \
  --command "bash /vePFS-Mindverse/share/code/tinker-server-aliyun/.claude/skills/aliyun-cluster/scripts/ray_entrypoint.sh"
