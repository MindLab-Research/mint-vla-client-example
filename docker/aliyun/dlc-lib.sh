#!/usr/bin/env bash
# Aliyun PAI-DLC helper functions for Ray cluster management.
# Source this file: source docker/aliyun/dlc-lib.sh
#
# Commands:
#   dlc-list-quotas               List available resource queues
#   dlc-get-head-ip <job_id>      Get head pod IP from a DLC job
#   dlc-list-jobs [name_filter]   List running DLC jobs
#   dlc-stop-jobs <name_prefix>   Stop all jobs matching name prefix

dlc-list-quotas() {
  echo "Available resource queues (workspace ${DLC_WORKSPACE_ID:-341495}):"
  echo ""
  dlc get quota \
    --workspace_id "${DLC_WORKSPACE_ID:-341495}" \
    --page_num 1 --page_size 50 \
    2>/dev/null || aliyun aiworkspace ListQuotas --WorkspaceId "${DLC_WORKSPACE_ID:-341495}" 2>/dev/null
  echo ""
  echo "Usage: DLC_RESOURCE_ID=<quota_id> docker/aliyun/head.sh"
  echo "       DLC_RESOURCE_ID=<quota_id> HEAD_IP=<ip> NAME_PREFIX=xxx docker/aliyun/worker.sh"
}

dlc-get-head-ip() {
  local job_id="${1:?Usage: dlc-get-head-ip <job_id>}"
  local ip
  ip="$(aliyun pai-dlc GetJob --JobId "${job_id}" --NeedDetail true 2>/dev/null \
    | python3 -c "
import sys, json
d = json.load(sys.stdin)
def find_ips(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and v.startswith('10.') and '.' in v:
                return v
            r = find_ips(v)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_ips(v)
            if r:
                return r
    return None
ip = find_ips(d)
print(ip or '')
" 2>/dev/null)"

  if [ -z "${ip}" ]; then
    echo "ERROR: could not find pod IP for job ${job_id}" >&2
    return 1
  fi
  echo "${ip}"
}

dlc-list-jobs() {
  local filter="${1:-}"
  local result
  result="$(dlc get job \
    --workspace_id "${DLC_WORKSPACE_ID:-341495}" \
    --page_num 1 --page_size 200 \
    2>/dev/null | awk '{print $1, $2, $8}')"

  if [ -n "${filter}" ]; then
    echo "${result}" | grep "${filter}"
  else
    echo "${result}"
  fi
}

dlc-stop-jobs() {
  local prefix="${1:?Usage: dlc-stop-jobs <name_prefix>}"
  local job_ids
  job_ids="$(dlc get job \
    --workspace_id "${DLC_WORKSPACE_ID:-341495}" \
    --status Running \
    --page_num 1 --page_size 200 \
    2>/dev/null | awk -v p="${prefix}" '$1 ~ p {print $2}')"

  if [ -z "${job_ids}" ]; then
    echo "No running jobs matching '${prefix}'"
    return 0
  fi

  while read -r job_id; do
    [ -n "${job_id}" ] || continue
    echo "Stopping ${job_id}..."
    dlc stop job "${job_id}" --force 2>&1
  done <<< "${job_ids}"
}
