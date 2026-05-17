---
name: mint-ops
description: |
  Mint internal operations: use MinT `/internal/*` control-plane APIs for
  actor inventory/kill, scheduler diagnostics, deep health, Ray diagnostics,
  and deploy/operate the standalone Mint ops console.

  Use for: internal actor admin, scheduler/admission/debug snapshots, deep health,
  Ray cluster diagnostics, maintenance actor status, or building/deploying the `ops/` console.

  Triggers: "mint ops", "internal ops", "internal actors", "actor admin",
  "scheduler debug", "admission stats", "deep health", "deploy ops", "ops console",
  "sync mint-ops", "start mint-ops"

  Procedure contract: read this SKILL.md end-to-end before acting. Do not slice it on demand or use it as a lookup table mid-run.
---

# Mint Ops

Procedure contract:
- Read this SKILL.md end-to-end before taking any action.
- Do not sample sections opportunistically while already in motion.
- If the procedure is missing something important, update the skill. Do not improvise around the gap.

Use this skill for MinT internal control-plane operations and for the standalone ops console under `ops/`.

## Internal Control Plane

Use `/internal/*` for operator-only control-plane state. Do not use removed legacy public paths such as `/api/v1/actors`, `/api/v1/actors/kill`, `/api/v1/kill_vllm`, `/api/v1/kill_megatron`, `/api/v1/vllm_status`, or `/api/v1/megatron_status`.

Environment-specific skills provide host, port, auth, and restart rules:
- For development, read `mint-dev` first; default local base URL is `http://localhost:8000` and auth is usually disabled.
- For production, read `mint-prod` first; default local base URL is `http://localhost:18000` and every internal call requires `X-API-Key` except public health checks.

### Hard Rules

- Default to read-only endpoints.
- Mutating calls, especially `/internal/actors/kill`, require an explicit actor family, model, actor name, or reason.
- Do not dump secrets. In prod, source `.secrets.env` without printing it.
- Public `/api/v1/healthz` is cheap API-worker health. Use `/internal/healthz/deep` only when Ray / placement-group diagnostics are needed.
- `/internal/actors` is an inventory/admin view, not the scheduling source of truth. Scheduler state is under `/internal/model_work_scheduler`; desired runtime state is under `/internal/model_actor_supervisor`.
- Do not run local `ray` or `volc` commands through this skill. Use the environment skill or `volcano-cluster`.

### Request Pattern

Development:

```bash
BASE=http://localhost:8000
curl -s "$BASE/internal/actors?type=vllm" | jq
```

Production:

```bash
source /vePFS-Mindverse/share/code/tinker-server-auth/.secrets.env
BASE=http://localhost:18000
curl -s -H "X-API-Key: $TINKER_API_KEY" "$BASE/internal/actors?type=vllm" | jq
```

### Read-only Endpoints

| Endpoint | Meaning |
|----------|---------|
| `/internal/health` | Internal API + usage store health |
| `/internal/healthz/deep` | Costly Ray / placement-group health observation |
| `/internal/admission_stats` | Combined scheduler, actor, store, process, Ray summary |
| `/internal/metrics` | Prometheus-style metrics |
| `/internal/model_work_scheduler` | Scheduler health and hot subqueue state |
| `/internal/model_work_scheduler/debug_state` | Scheduler debug snapshot |
| `/internal/debug/scheduler_decisions` | Scheduler decision history; supports filters |
| `/internal/model_actor_supervisor` | Desired runtime reconciliation state |
| `/internal/actors` | Live actor inventory; supports `type`, `model_name`, `refresh_metadata` |
| `/internal/ray_cluster_health` | Ray cluster health snapshot |
| `/internal/ray_gcs_metrics` | Ray GCS metrics snapshot |
| `/internal/maintenance_cron_actor` | Maintenance cron actor health |

Useful examples:

```bash
curl -s "$BASE/internal/actors?type=vllm&refresh_metadata=false" | jq '.actors[] | {actor_name,actor_type,base_model,current_session,num_gpus,idle}'
curl -s "$BASE/internal/model_actor_supervisor" | jq
curl -s "$BASE/internal/model_work_scheduler/debug_state" | jq
curl -s "$BASE/internal/admission_stats?include_actor_rss=false" | jq
```

For prod, add `-H "X-API-Key: $TINKER_API_KEY"` to the curl commands above.

### Actor Kill

`POST /internal/actors/kill` is the single actor-admin mutation endpoint.

Payload fields:
- `actor_type`: `vllm`, `megatron`, `dense`, `openpi`, or `all`
- optional `model_name`
- optional `actor_name`
- optional `force`
- optional `reason`

Examples:

```bash
# Kill one actor family.
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"actor_type":"vllm","reason":"operator_requested"}' \
  "$BASE/internal/actors/kill" | jq

# Kill one model's Megatron actor.
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"actor_type":"megatron","model_name":"Qwen/Qwen3-30B-A3B-Instruct-2507","reason":"reload_actor_code"}' \
  "$BASE/internal/actors/kill" | jq

# Kill all tracked GPU actors. This does not clear detached stores or scheduler actors.
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"actor_type":"all","reason":"gpu_actor_code_reload"}' \
  "$BASE/internal/actors/kill" | jq
```

For prod, add `-H "X-API-Key: $TINKER_API_KEY"`.

### Evidence To Report

For read-only diagnostics, report endpoint, status, and the key fields that explain the state. For mutations, report endpoint, request body without secrets, response summary, and whether a server restart or actor recreation is still required.

## Ops Console

Use this section when the task is specifically about deploying or operating the standalone ops console under `ops/`.

### Scope

- Target host: `mint-prod-volcano`
- Remote app root: `/vePFS-Mindverse/share/mint-ops`
- Python env: `/vePFS-Mindverse/share/code/tinker-server-auth/.venv31213/bin/python`
- Mint API: `http://127.0.0.1:18000`
- Ray: deploy on the driver node, so default to `auto`
- Supervisor config: `/mlplatform/supervisord/supervisord.conf`
- Single exposed URL: `http://<internal-ip>:8787/`

### Hard rules

- Use `ssh mint-prod-volcano`, not a guessed host or raw IP, for server-side changes.
- Do not use `rsync --delete`.
- Do not print secrets from `.secrets.env`.
- Production should use one service only: build the frontend with relative `/api` paths and let `ops.backend` serve `ops/frontend/dist/`.
- Keep backend pointed at local Mint (`http://127.0.0.1:18000`) and source the API key from remote `.secrets.env`.

### Deploy SOP

#### 1. Build frontend locally

Build with the default relative API path:

```bash
cd /vePFS-Mindverse/user/intern/nolanho/code/mint-feat-mint-ops/ops/frontend
pnpm build
```

#### 2. Sync only the `ops` app

Sync the directory itself so the remote layout becomes `/vePFS-Mindverse/share/mint-ops/ops/...`:

```bash
cd /vePFS-Mindverse/user/intern/nolanho/code/mint-feat-mint-ops
rsync -avz ops \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='node_modules/' \
  --exclude='*.tsbuildinfo' \
  mint-prod-volcano:/vePFS-Mindverse/share/mint-ops/
```

#### 3. Add the supervisor program

Append one program to `/mlplatform/supervisord/supervisord.conf`:

```ini
[program:mint-ops-backend]
command=/usr/bin/bash -lc 'cd /vePFS-Mindverse/share/mint-ops && set -a && source /vePFS-Mindverse/share/code/tinker-server-auth/.secrets.env && set +a && export PYTHONPATH=/vePFS-Mindverse/share/mint-ops:/vePFS-Mindverse/share/code/tinker-server-auth && export MINT_OPS_MINT_BASE_URL=http://127.0.0.1:18000 && exec /vePFS-Mindverse/share/code/tinker-server-auth/.venv31213/bin/python -m ops.backend --bind 0.0.0.0 --backend-port 8787'
directory=/vePFS-Mindverse/share/mint-ops
autostart=true
autorestart=true
priority=15
startsecs=5
startretries=3
stopasgroup=true
killasgroup=true
user=root
stdout_logfile=/tmp/mint_ops_backend.log
stderr_logfile=/tmp/mint_ops_backend.err.log
stdout_logfile_maxbytes=20MB
stdout_logfile_backups=5
environment=PYTHONUNBUFFERED="1"
```

Reload supervisor:

```bash
ssh mint-prod-volcano 'supervisorctl reread && supervisorctl update && supervisorctl restart mint-ops-backend'
```

#### 4. Verify

```bash
ssh mint-prod-volcano 'supervisorctl status mint-ops-backend'
ssh mint-prod-volcano 'curl -s http://127.0.0.1:8787/api/health'
curl -I http://<internal-ip>:8787/
```

Use `http://<internal-ip>:8787/` as the single entrypoint. The same port serves both the UI and `/api/*`, so SSH port forwarding to `8787` also works cleanly.
