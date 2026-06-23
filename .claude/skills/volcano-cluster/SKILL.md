---
name: volcano-cluster
description: |
  Ray cluster lifecycle management on Volcano ML platform.

  Use for: create cluster, tear down cluster, list tasks, cancel tasks, check Ray dashboard, GPU allocation, stale actor cleanup.

  Triggers: "create cluster", "tear down", "list tasks", "cancel tasks", "Ray dashboard", "stale actor", "GPU", "volcano", "volc"

  This skill covers generic Volcano/Ray operations. For environment-specific server operations, use mint-dev or mint-prod.

  Procedure contract: read this SKILL.md end-to-end before acting. Do not slice it on demand or use it as a lookup table mid-run.
---

# Volcano Cluster Management

Procedure contract:
- Read this SKILL.md end-to-end before taking any action.
- Do not sample sections opportunistically while already in motion.
- If the procedure is missing something important, update the skill. Do not improvise around the gap.

## Core Rule

Mint node lifecycle is managed through the Volcano Engine Python SDK, not by shelling out to the Volcano CLI.

Use one of these two paths:
- Preferred service path: start `mint_model_actor_supervisor` with `MINT_TOPOLOGY_CONFIG_PATH`; it reconciles desired `mint-worker-{idx}` nodes through `VolcanoTopologyProvider`.
- Operator path: run `scripts/tools/volcano_sdk_jobs.py` on the trusted driver/bastion host to list, submit, inspect, or stop Volcano jobs.

Do not print AK/SK, session tokens, credential files, signed headers, or process environments. Credential checks may report only whether a credential source exists.

## Quick Reference

Run SDK commands on the matching driver host with the canonical runtime Python:

```bash
cd /vePFS-Mindverse/share/mint/<env>/mint-server
PY=/vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python

# List Mint worker jobs
$PY scripts/tools/volcano_sdk_jobs.py --region cn-beijing list --name-contains mint-<env>-worker- --limit 200

# Inspect job instances and node IPs
$PY scripts/tools/volcano_sdk_jobs.py --region cn-beijing instances --job-id <job_id>

# Submit one desired topology alias
$PY scripts/tools/volcano_sdk_jobs.py --region cn-beijing submit-topology-node \
  --config /vePFS-Mindverse/share/mint/<env>/runtime/topology.yaml \
  --alias mint-worker-0

# Stop a job
$PY scripts/tools/volcano_sdk_jobs.py --region cn-beijing stop --job-id <job_id>
```

`volcano_sdk_jobs.py` uses the SDK default credential chain and never prints credentials. If read calls hang or fail, first verify SSH connectivity and then verify that the driver host exposes a supported credential source.

## Credential Model

Only the trusted driver/control-plane host should hold Volcano credentials. API workers, ConfigActor, model actors, metrics daemon actors, and Ray runtime actors must not receive provider credentials.

Supported credential sources depend on the installed `volcengine-python-sdk` version. Prefer:
- `VOLCENGINE_ACCESS_KEY` / `VOLCENGINE_SECRET_KEY`, with optional `VOLCENGINE_SESSION_TOKEN`
- `VOLCENGINE_CLI_CONFIG_FILE`
- `~/.volcengine/config.json`
- OIDC or ECS role metadata when intentionally configured on the driver host

Older CLI locations such as `~/.volc/config`, `~/.volc/credentials`, `VOLC_ACCESSKEY`, or `VOLC_SECRETKEY` are not the target contract. Treat them as compatibility only after verifying the installed SDK reads them.

Safe credential probe:

```bash
python3 - <<'PY'
import os
from pathlib import Path
for p in (Path.home()/".volcengine/config.json", Path.home()/".volc/config", Path.home()/".volc/credentials"):
    print(str(p), "exists", p.exists())
for key in ("VOLCENGINE_ACCESS_KEY", "VOLCENGINE_SECRET_KEY", "VOLCENGINE_SESSION_TOKEN", "VOLCENGINE_CLI_CONFIG_FILE", "VOLCENGINE_PROFILE"):
    print(key, "set", key in os.environ)
PY
```

Do not dump credential file contents or process environments.

## Instance Flavors And Queues

| Flavor | GPUs | Use Case |
|--------|------|----------|
| `ml.r3i.4xlarge` | 0 | Ray head node |
| `ml.hpcpni2l.28xlarge` | 8x A800 80GB | GPU workers |

| Queue Name | Queue ID | Type | Use Case |
|------------|----------|------|----------|
| `cpu-mindverse` | `q-20251225183621-m2297` | CPU | Ray head node |
| `a800-mindverse-C1` | `q-20251126180002-26lwz` | GPU | Default/prod A800 workers |
| `a800-mindverse-C2` | `q-20260203101340-www2h` | GPU | Reserved specialist pool |

CPU-only instances must use CPU queues. GPU instances must use GPU queues. Queue capacity still comes from the Volcano console; the SDK job list is an estimate of current demand, not an authoritative free-capacity API.

## Topology-Managed Workers

Workers are named by alias:
- Desired alias: `mint-worker-{idx}`
- Provider job name: `mint-{deployment_env}-worker-{idx+1}` (`mint-worker-0` becomes `mint-<env>-worker-1`)
- Runtime debug state: `/vePFS-Mindverse/share/mint/<env>/runtime/topology_state.yaml`

`topology_state.yaml` is output-only. On restart, the supervisor reads the static topology config and provider/Ray live state, then rewrites the state file. Do not edit `topology_state.yaml` as input.

Minimal topology shape:

```yaml
version: 1
deployment_env: dev
cluster_id: volcano
state_path: /vePFS-Mindverse/share/mint/dev/runtime/topology_state.yaml
ray:
  head_ip_path: /vePFS-Mindverse/share/mint/dev/ray/head-address/ray_head_ip.txt
providers:
  volcano:
    region: cn-beijing
    credentials:
      mode: default_chain
    templates:
      a800-8gpu-c1:
        template_path: /vePFS-Mindverse/share/<owner>/mint-server/.claude/skills/volcano-cluster/configs/mint-dev-worker.yaml
        resource_queue_id: q-20251126180002-26lwz
        gpu_count: 8
nodes:
  desired:
    - alias: mint-worker-0
      provider: volcano
      template: a800-8gpu-c1
      role: gpu
      enabled: true
```

The supervisor submits workers in ascending alias order. If `mint-worker-0` is missing, `mint-worker-1` waits and `topology_state.yaml` explains the missing lower alias. If a live provider job and alive Ray node already satisfy an alias, the supervisor reuses it instead of submitting another job.

## Head Nodes

Head-node lifecycle is still environment-owned bootstrap. Worker lifecycle is topology/SDK-owned.

Head templates:
- `.claude/skills/volcano-cluster/configs/mint-dev-head.yaml`
- `.claude/skills/volcano-cluster/configs/mint-prod-head.yaml`

Head nodes publish their IP to:
- Dev: `/vePFS-Mindverse/share/mint/dev/ray/head-address/ray_head_ip.txt`
- Prod: `/vePFS-Mindverse/share/mint/prod/ray/head-address/ray_head_ip.txt`

Workers read the head IP from that PFS file. Do not patch head IPs into worker templates.

## Ray Driver Rules

- Do not run `ray start` on `mint-dev`.
- Do not start a schedulable Ray node on API/bastion hosts.
- On dev API hosts, attach Python directly to GCS from the dev driver:
  `<RAY_HEAD_IP>:6379`. Do not use Ray Client mode for Mint dev.
- For CLI-style status checks, use the canonical Ray wrapper and connect to the head address.

Example:

```bash
PY=/vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python
$PY - <<'PY'
import ray
ray.init(address="<RAY_HEAD_IP>:6379")
print(ray.cluster_resources())
ray.shutdown()
PY
```

## Stale Actor Cleanup

Use API surfaces first. Do not restart Ray as a first response to actor placement issues.

```bash
# Dev
curl -s "http://localhost:${MINT_PORT}/internal/actors?type=megatron" | jq
curl -s -X POST -H "Content-Type: application/json" -d '{"actor_type":"megatron"}' http://localhost:${MINT_PORT}/internal/actors/kill

# Prod
curl -s -H "X-API-Key: $MINT_API_KEY" "http://localhost:18000/internal/actors?type=megatron" | jq
curl -s -X POST -H "X-API-Key: $MINT_API_KEY" -H "Content-Type: application/json" \
  -d '{"actor_type":"megatron"}' http://localhost:18000/internal/actors/kill
```

Actor naming reference:

| Actor | Name Pattern |
|-------|--------------|
| Supervisor | `mint_model_actor_supervisor` |
| Config | `mint_config` |
| Task state | `mint_task_state_store` |
| Scheduler | `mint_model_work_scheduler` |
| Maintenance cron | `mint_maintenance_cron` |
| Megatron | `mint_megatron_{model_slug}` |
| vLLM per model | `mint_vllm_{model_slug}` |
| Dense trainer | `mint_dense_{model_slug}` |
| Runtime session | `mint_model_runtime_*` |
| Node metrics daemon | `mint_daemon_node_metrics_{worker_alias}` |

`vllm_server` is not a valid default actor for new topology-managed model runtime.

## Monitoring

Ray dashboard:

```text
http://<RAY_HEAD_IP>:8265
```

Worker IPs:

```bash
PY=/vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python
$PY scripts/tools/volcano_sdk_jobs.py --region cn-beijing instances --job-id <job_id>
cat /vePFS-Mindverse/share/mint/<env>/runtime/topology_state.yaml
```

## Troubleshooting

| Symptom | Action |
|---------|--------|
| SDK read call times out | Verify SSH connectivity first, then credential source existence. Do not fall back to CLI lifecycle commands. |
| Worker job exists but alias is not ready | Inspect `instances --job-id` and `topology_state.yaml`; confirm Ray node joined. |
| Higher alias waits | Fix the lower missing alias first; creation is ordered by index. |
| Task stuck in Queue | Check Volcano console capacity. If capacity is sufficient but the job remains queued, split desired nodes into one-job-per-node aliases. |
| Worker fails to join Ray | Confirm head IP file, port reachability, image/runtime path, and PFS mounts. |
| Stale actors block placement | Kill via Mint API or supervisor-owned actor cleanup before canceling provider jobs. |
| Ray reports temp/spill filesystem over 95% full | Treat this as a scratch hygiene warning, not proof that spilling is disabled. First inspect the `mint ray path check` startup lines and confirm `object_spilling_directory` resolves under `/mnt/tmp` with the expected Vepfs mount. If the backing mount is correct, clean or rotate stale vePFS scratch in a separate owner/age-scoped procedure; do not add startup fail-fast behavior by default. |

## Config Files

| Environment | Head Config | Worker Config |
|-------------|-------------|---------------|
| Dev | `mint-dev-head.yaml` | `mint-dev-worker.yaml` |
| Prod | `mint-prod-head.yaml` | `mint-prod-worker.yaml` |

All configs live in `.claude/skills/volcano-cluster/configs/`.

## Shared Node Runtime Assets

The dev/prod templates depend on shared runtime assets under
`/vePFS-Mindverse/share/mint/runtime`:

- `supervisor/current/bin/mint-ray-node`: starts Ray through Python internals
  with `/mnt/tmp` for Ray temp/spill/cache paths. On startup it emits
  warning-only `mint ray path check` diagnostics for the configured path,
  resolved path, writability, free/total bytes, and backing mount. Low free
  space is logged as a warning; it is not a startup failure.
- `services/{dev,prod}-{head,worker}`: runit service directories for Ray and
  sshd.
- `ssh/sshd_config`: shared sshd config using
  `/vePFS-Mindverse/share/mint/runtime/ssh/authorized_keys`.

The version-controlled source for these scripts is under
`.claude/skills/volcano-cluster/runtime/`. After changing these files, sync them
to `/vePFS-Mindverse/share/mint/runtime` before creating or replacing nodes.
