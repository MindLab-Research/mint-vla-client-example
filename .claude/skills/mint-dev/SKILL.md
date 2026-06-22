---
name: mint-dev
description: |
  Development environment operations for the Mint server on the dev driver node.

  Use for: starting/restarting the dev API server, reading dev logs, updating the
  dev server checkout, and validating local changes against the dev deployment.

  Triggers: "dev server", "start dev", "restart dev", "dev logs", "sync to dev",
  "dev vLLM", "mint-dev"

  Do not use this skill for production. Use mint-prod instead.
  For cluster lifecycle work, use volcano-cluster.

  Procedure contract: read this SKILL.md end-to-end before acting.
---

# Mint Dev

Read this whole file before touching the dev environment.

## Architecture

```
Local Machine          mint-dev (driver)              mint-dev-head (Volcano pod)
─────────────          ────────────────               ──────────────────────────
tinker-cookbook ──HTTP──> mint-server API ──Ray──>   GCS + raylet + dashboard
                    (port from namespace)  direct attach   GPU Workers (Volcano pods)
                       ↑
                  SSH tunnel
```

| Item | Value |
|------|-------|
| Driver node | `mint-dev` (SSH alias) |
| Ray head | Volcano pod (`mint-dev-head`), managed by volcano-cluster skill |
| Head IP | `cat /vePFS-Mindverse/share/mint/dev/ray/head-address/ray_head_ip.txt` |
| Connection mode | Direct attach (`ray.init(address="auto")`) |
| API port | Derived from namespace hash (`[30000, 40000)`); override with `MINT_PORT` |
| Runtime root | `/vePFS-Mindverse/share/mint/dev/runtime` |
| Ray namespace | `mint_<user>_dev` (per-user) |
| Auth mode | `no-auth` by default |
| Log file | `/vePFS-Mindverse/share/mint/dev/logs/mint-dev-server.log` |

## Quick Start

### 1. Sync code to PFS

```bash
rsync -a --exclude '.git' --exclude '__pycache__' \
  <your-checkout>/ /vePFS-Mindverse/share/code/<you>/<branch>/
```

**Do not use `--delete`** — it can remove files that other users or the
runtime depend on. Only sync your code, not the entire directory tree.

`MINT_CODE_ROOT` must be under `/vePFS-Mindverse/share/` and visible to all
Ray nodes. Do not use local paths or `/vePFS-Mindverse/user/...`.

### 2. Generate placement config

Worker IPs change when the cluster is recreated. Generate a run env file:

```bash
HEAD_IP=$(cat /vePFS-Mindverse/share/mint/dev/ray/head-address/ray_head_ip.txt)
python scripts/tools/gen_dev_placement.py --head-ip $HEAD_IP \
  --model Qwen/Qwen3-0.6B --gpu-count 1 \
  --output /tmp/mint_dev_run.env
scp /tmp/mint_dev_run.env mint-dev:/tmp/mint_dev_run.env
```

### 3. Start dev server

```bash
ssh mint-dev 'MINT_CODE_ROOT=/vePFS-Mindverse/share/code/<you>/<branch> \
  MINT_DEV_USER=<you> \
  MINT_RAY_NAMESPACE=mint_<you>_dev \
  MINT_LOG_FILE=/vePFS-Mindverse/share/mint/dev/logs/mint-dev-server.log \
  MINT_DISABLE_MINT_ROUTE=1 \
  MINT_UVICORN_WORKERS=1 \
  MINT_SUPERVISOR_STATE_BACKEND=memory \
  MINT_DEV_RUN_ENV=/tmp/mint_dev_run.env \
  nohup /vePFS-Mindverse/share/code/<you>/<branch>/scripts/start_dev_server.sh \
  >> /tmp/mint_dev_launch.log 2>&1 &'
```

The launcher auto-loads deployment defaults (model lists, vLLM flags, OTel
endpoint, logging) and `secrets.env` (OTLP API keys). Port is derived from
the namespace; override with `MINT_PORT=8000` if needed.

The reconcile loop is **enabled by default** — the supervisor will
automatically create runtime/training/vLLM replicas. To disable for
debugging, set `MINT_SUPERVISOR_RECONCILE_LOOP=0`.

### 4. Wait for healthz

```bash
ssh mint-dev 'curl -s http://localhost:8000/api/v1/healthz'
# Expected: {"status":"ready"}
```

### 5. SSH tunnel for local access

```bash
ssh -f -N -L 8000:localhost:8000 mint-dev
```

## Configuration Reference

### Required inputs

| Variable | Purpose |
|----------|---------|
| `MINT_CODE_ROOT` | mint-server checkout (PFS path visible to all Ray nodes) |
| `MINT_DEV_USER` | Username for namespace derivation (must be non-root) |

### Recommended dev settings

| Variable | Value | Why |
|----------|-------|-----|
| `MINT_SUPERVISOR_STATE_BACKEND` | `memory` | Avoids RocksDB lock conflicts from stale actors |
| `MINT_DEV_RUN_ENV` | `/tmp/mint_dev_run.env` | Per-run placement config (IPs change when cluster recreated) |
| `MINT_UVICORN_WORKERS` | `1` | Single worker for dev |

### Auto-loaded config

The launcher (`start_dev_server.sh`) automatically loads:
- **Deployment defaults**: model lists, vLLM flags, OTel endpoint, logging, resource queues
- **secrets.env**: `/vePFS-Mindverse/share/mint/dev/config/secrets.env` (OTLP API keys)
- **`RAY_ENABLE_AUTO_CONNECT=0`**: Prevents zombie GCS processes in worker actors

### Placement config

See `scripts/tools/gen_dev_placement.py` — generates a run env file with
placement JSON from the current cluster's Ray dashboard. Run it whenever
the cluster is recreated.

### `MINT_DISABLE_MINT_ROUTE`

When set to `1`, skips loading the `/api/v1/mint/*` route module (`routes/mint.py`).
These routes provide Mint-only API extensions: VLA action sessions, VLA train_step,
checkpoint interpolation, and reverse KL forward/backward. Setting this flag avoids
loading their dependencies (action session manager, interpolation utils) during
startup, which is useful for standard RL training/inference that doesn't use these
features. Leave unset if you need VLA or checkpoint interpolation endpoints.

## Restart After Code Changes

Python servers do not hot-reload. After any code change:

```bash
# 1. Kill old server
ssh mint-dev 'kill $(pgrep -f "scripts/run_server.py" | head -1) 2>/dev/null; sleep 2'

# 2. Clean stale control-plane actors (if fingerprint mismatch)
ssh mint-dev 'PY=/vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python
HEAD_IP=$(cat /vePFS-Mindverse/share/mint/dev/ray/head-address/ray_head_ip.txt)
$PY -c "
import ray
ray.init(address=\"$HEAD_IP:6379\", namespace=\"mint_<you>_dev\", ignore_reinit_error=True, log_to_driver=False)
for name in [\"mint_config\", \"mint_task_state_store\", \"mint_model_work_scheduler\", \"mint_maintenance_cron\", \"mint_model_actor_supervisor\"]:
    try:
        a = ray.get_actor(name, namespace=\"mint_<you>_dev\")
        ray.kill(a, no_restart=True)
        print(f\"killed {name}\")
    except Exception:
        pass
ray.shutdown()
"'

# 3. Sync updated code (no --delete!)
rsync -a --exclude '.git' --exclude '__pycache__' \
  <your-checkout>/ /vePFS-Mindverse/share/code/<you>/<branch>/

# 4. Restart (same command as Quick Start step 3)
```

## Health And Logs

```bash
# API health
ssh mint-dev 'curl -s http://localhost:8000/api/v1/healthz'

# Server info
ssh mint-dev 'curl -s http://localhost:8000/api/v1/server_info'

# Admission stats (scheduler + supervisor + ray cluster)
ssh mint-dev 'curl -s http://localhost:8000/internal/admission_stats'

# Server logs
ssh mint-dev 'tail -50 /vePFS-Mindverse/share/mint/dev/logs/mint-dev-server.log'

# Ray cluster status
ssh mint-dev 'curl -s http://<HEAD_IP>:8265/api/cluster_status'
```

## RL Sanity Check

After starting the dev server, run the RL check:

```bash
ssh -f -N -L 8000:localhost:8000 mint-dev

MINT_BASE_URL=http://localhost:8000 \
TINKER_API_KEY=dummy \
MINT_API_KEY=dummy \
python scripts/tools/rl_check.py \
  --model Qwen/Qwen3-0.6B \
  --steps 10 \
  --group-size 4 \
  --timeout-s 600
```

Pass criteria: all 10 steps complete, `num_datums > 0` per step, status `pass`.

## Issue-Scoped Servers

Isolation comes from the Ray namespace. Use a scoped namespace, port, and log:

```bash
MINT_CODE_ROOT=/path/to/checkout \
MINT_RAY_NAMESPACE=mint_<you>_issue_<n> \
MINT_PORT=10416 \
MINT_LOG_FILE=/tmp/mint_server_issue_<n>.log \
MINT_UVICORN_WORKERS=1 \
MINT_SUPERVISOR_STATE_BACKEND=memory \
MINT_DEV_RUN_ENV=/tmp/mint_dev_run.env \
scripts/start_dev_server.sh
```

## Cleanup

After finishing dev validation, clean up **all** actors in your namespace.
This is a required closeout step, not optional.

```bash
ssh mint-dev 'PY=/vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python
HEAD_IP=$(cat /vePFS-Mindverse/share/mint/dev/ray/head-address/ray_head_ip.txt)
NS=mint_<you>_dev
$PY -c "
import ray
ray.init(address=\"$HEAD_IP:6379\", namespace=\"$NS\", ignore_reinit_error=True, log_to_driver=False)
# Kill all named actors in your namespace
for actor in ray.util.list_named_actors(all_namespaces=True):
    ns = str(actor.get(\"namespace\") or \"\")
    name = str(actor.get(\"name\") or \"\")
    if ns == \"$NS\" and name:
        try:
            a = ray.get_actor(name, namespace=ns)
            ray.kill(a, no_restart=True)
            print(f\"killed {name}\")
        except Exception as e:
            print(f\"skip {name}: {e}\")
# Also clean issue-scoped namespaces
for actor in ray.util.list_named_actors(all_namespaces=True):
    ns = str(actor.get(\"namespace\") or \"\")
    name = str(actor.get(\"name\") or \"\")
    if ns.startswith(\"mint_<you>_issue_\") and name:
        try:
            a = ray.get_actor(name, namespace=ns)
            ray.kill(a, no_restart=True)
            print(f\"killed {name} in {ns}\")
        except Exception as e:
            print(f\"skip {name} in {ns}: {e}\")
ray.shutdown()
"'
```

Also stop your API server process and verify the port is no longer listening.
Do not leave dev tasks marked complete while known actors remain in your namespace.

## Internal Ops

For `/internal/*` actor inventory, actor kill, scheduler diagnostics, and Ray
diagnostics, use the `mint-ops` skill. Internal routes on the dev server use
pass-through mode (no auth headers needed).

## Worker Node Lifecycle

Dev GPU workers are Volcano pods managed by the `volcano-cluster` skill.
Do not run local `ray` or `volc` CLI commands. Use the volcano-cluster skill
for cluster lifecycle (create/teardown workers, list jobs, inspect instances).

## Hard Rules

- Do not perform production operations from this skill.
- Do not run local `ray` or `volc` CLI commands. Use project Python with Ray
  for inspection, and use the `volcano-cluster` skill for lifecycle work.
- Do not source or print private config unless the task requires it.
- Do not switch ports to hide a failed restart; fix the listener or process.
- Do not default `MINT_CODE_ROOT` to the shared dev checkout. It is a required,
  explicit input; ask the user which checkout to run.
- Do not invent a Ray namespace. Derive `mint_<user>_dev`; if that resolves to
  root, ask the user for `MINT_DEV_USER` or `MINT_RAY_NAMESPACE`.
- Do not install packages until the runtime root (`PFS_RUNTIME_ENV_ROOT`) and
  the resolved `PYTHONPATH` have been verified.
- Do not finish dev validation without cleaning owned zombie namespaces, actors,
  and placement groups, or recording why cleanup was blocked.
- **NEVER use `rsync --delete`** to sync code or runtime assets. It can remove
  files that other users or the runtime depend on. Use `rsync -a` without
  `--delete`, or copy individual files explicitly.
