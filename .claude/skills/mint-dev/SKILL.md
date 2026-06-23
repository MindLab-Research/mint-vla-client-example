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
Local Machine          mint-dev-driver (driver)       mint-dev-head (Volcano pod)
─────────────          ────────────────               ──────────────────────────
tinker-cookbook ──HTTP──> mint-server API ──Ray──>   GCS + raylet + dashboard
                    (port from namespace)  direct attach   GPU Workers (Volcano pods)
                       ↑
                  SSH tunnel
```

| Item | Value |
|------|-------|
| Driver node | `mint-dev-driver`, `192.168.42.106` (SSH alias: `mint-dev`) |
| Ray head | Volcano pod (`mint-dev-head`), managed by volcano-cluster skill |
| Head IP | `cat /vePFS-Mindverse/share/mint/dev/ray/head-address/ray_head_ip.txt` |
| Connection mode | Direct GCS attach (`MINT_RAY_GCS_ADDRESS=<head_ip>:6379`) |
| API port | Derived from namespace hash (`[30000, 40000)`); override with `MINT_PORT` |
| Runtime root | `/vePFS-Mindverse/share/mint/dev/runtime` |
| Ray namespace | Defaults to `mint_<user>`; override with `MINT_RAY_NAMESPACE` |
| Auth mode | `no-auth` by default |
| Log file | `/vePFS-Mindverse/share/mint/dev/logs/mint-dev-server.log` |

## Access

All dev API-server and driver-side debugging runs on `mint-dev` /
`mint-dev-driver`. Do not start Mint dev servers on the Ray head pod.

To SSH into Mint dev machines, append only your public key to:

```bash
/vePFS-Mindverse/share/mint/runtime/ssh/authorized_keys
```

Never write private keys there, and do not remove or rewrite other users' keys.

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

### 2. Placement config

Worker IPs change when the cluster is recreated. Do not hand-edit placement
for normal dev startup: `scripts/start_dev_server.sh` auto-generates a
run-local placement env from the current Ray dashboard when no placement env
vars are already set.

Manual placement remains an override for special debugging:

```bash
HEAD_IP=$(cat /vePFS-Mindverse/share/mint/dev/ray/head-address/ray_head_ip.txt)
python scripts/tools/gen_dev_placement.py --head-ip $HEAD_IP \
  --model Qwen/Qwen3-0.6B --gpu-count 1 \
  --output /vePFS-Mindverse/share/mint/dev/tmp/mint_dev_run.env
```

### 3. Start dev server

```bash
ssh mint-dev 'MINT_CODE_ROOT=/vePFS-Mindverse/share/code/<you>/<branch> \
  MINT_DEV_USER=<you> \
  MINT_LOG_FILE=/vePFS-Mindverse/share/mint/dev/logs/mint-dev-server.log \
  MINT_DISABLE_MINT_ROUTE=1 \
  MINT_UVICORN_WORKERS=1 \
  MINT_SUPERVISOR_STATE_BACKEND=memory \
  MINT_TASK_STATE_STORE_DB_PATH=:memory: \
  nohup /vePFS-Mindverse/share/code/<you>/<branch>/scripts/start_dev_server.sh \
  >> /tmp/mint_dev_launch.log 2>&1 &'
```

The launcher auto-loads deployment defaults (model lists, vLLM flags, OTel
endpoint, logging) and `secrets.env` (OTLP API keys), derives the port from
the namespace, and auto-generates placement if needed. The launcher prints
the resolved `MINT_PORT` during startup; use that value for health checks,
tunnels, and local clients. Override with `MINT_PORT=<port>` or
`MINT_DEV_RUN_ENV=<path>` if needed.

After launch, set the printed port in your local shell before running the
following commands:

```bash
export MINT_PORT=<printed-port>
```

The reconcile loop is **enabled by default** — the supervisor will
automatically create runtime/training/vLLM replicas. To disable for
debugging, set `MINT_SUPERVISOR_RECONCILE_LOOP=0`.

### 4. Wait for healthz

```bash
ssh mint-dev "curl -s http://localhost:${MINT_PORT}/api/v1/healthz"
# Expected: {"status":"ready"}
```

### 5. SSH tunnel for local access

```bash
ssh -f -N -L ${MINT_PORT}:localhost:${MINT_PORT} mint-dev
```

## Configuration Reference

### Required inputs

| Variable | Purpose |
|----------|---------|
| `MINT_CODE_ROOT` | mint-server checkout (PFS path visible to all Ray nodes) |
| `MINT_DEV_USER` | Username for namespace derivation (must be non-root) |
| `MINT_RAY_NAMESPACE` | Optional explicit namespace; defaults to `mint_<user>` |
| `MINT_PORT` | Optional explicit API port; defaults to a stable namespace hash |

### Recommended dev settings

| Variable | Value | Why |
|----------|-------|-----|
| `MINT_SUPERVISOR_STATE_BACKEND` | `memory` | Avoids RocksDB lock conflicts from stale actors |
| `MINT_TASK_STATE_STORE_DB_PATH` | `:memory:` | Keeps TaskStateStore non-persistent for disposable dev servers |
| `MINT_DEV_RUN_ENV` | optional | Per-run overrides; if it sets placement vars, auto placement is skipped |
| `MINT_UVICORN_WORKERS` | `1` | Single worker for dev |

### Auto-loaded config

The launcher (`start_dev_server.sh`) automatically loads:
- **Deployment defaults**: model lists, vLLM flags, OTel endpoint, logging, resource queues
- **Optional deployment env**: only when `MINT_DEV_DEPLOYMENT_ENV=<path>` is set explicitly
- **Auto placement env**: generated under `MINT_TMP_ROOT/auto-placement/` unless placement vars already exist

### Placement config

See `scripts/tools/gen_dev_placement.py` — it generates placement JSON from
the current cluster's Ray dashboard. The launcher runs it automatically when
no placement env var is set. To force manual placement, set one of
`MINT_MODEL_PLACEMENT_JSON`, `MINT_DENSE_MODEL_PLACEMENT_JSON`,
`MINT_VLLM_MODEL_PLACEMENT_JSON`, or `MINT_MEGATRON_MODEL_PLACEMENT_JSON`
directly or through `MINT_DEV_RUN_ENV`.

If you already have placement JSON, provide it through the canonical runtime
variables read by the supervisor and backends:

- `MINT_MODEL_PLACEMENT_JSON`
- `MINT_DENSE_MODEL_PLACEMENT_JSON`
- `MINT_VLLM_MODEL_PLACEMENT_JSON`
- `MINT_MEGATRON_MODEL_PLACEMENT_JSON`

`scripts/tools/gen_dev_placement.py` writes those variables into the run env
file consumed by `MINT_DEV_RUN_ENV`.

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
ray.init(address=\"$HEAD_IP:6379\", namespace=\"mint_<you>\", ignore_reinit_error=True, log_to_driver=False)
for name in [\"mint_config\", \"mint_task_state_store\", \"mint_model_work_scheduler\", \"mint_maintenance_cron\", \"mint_model_actor_supervisor\"]:
    try:
        a = ray.get_actor(name, namespace=\"mint_<you>\")
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
ssh mint-dev "curl -s http://localhost:${MINT_PORT}/api/v1/healthz"

# Server info
ssh mint-dev "curl -s http://localhost:${MINT_PORT}/api/v1/server_info"

# Admission stats (scheduler + supervisor + ray cluster)
ssh mint-dev "curl -s http://localhost:${MINT_PORT}/internal/admission_stats"

# Server logs
ssh mint-dev 'tail -50 /vePFS-Mindverse/share/mint/dev/logs/mint-dev-server.log'

# Ray cluster status
ssh mint-dev 'curl -s http://<HEAD_IP>:8265/api/cluster_status'
```

## RL Sanity Check

After starting the dev server, run the RL check:

```bash
ssh -f -N -L ${MINT_PORT}:localhost:${MINT_PORT} mint-dev

MINT_BASE_URL=http://localhost:${MINT_PORT} \
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
MINT_LOG_FILE=/vePFS-Mindverse/share/mint/dev/logs/mint-server-issue-<n>.log \
MINT_UVICORN_WORKERS=1 \
MINT_SUPERVISOR_STATE_BACKEND=memory \
MINT_TASK_STATE_STORE_DB_PATH=:memory: \
scripts/start_dev_server.sh
```

## CI And Bug Workflow

Current CI gates cover type checking and the Scheduler component. Run the
relevant local checks for the files you touch, and treat CI failures as blockers
unless you can identify and record an unrelated pre-existing failure.

When you find a bug, open a GitHub issue and put it into MinT task management.
Do not rely on chat context or a local todo as the only tracking mechanism.

## Cleanup

After finishing dev validation, clean up **all** actors in your namespace.
This is a required closeout step, not optional.

```bash
ssh mint-dev 'PY=/vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python
HEAD_IP=$(cat /vePFS-Mindverse/share/mint/dev/ray/head-address/ray_head_ip.txt)
NS=mint_<you>
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
- Do not use Ray Client mode (`ray://...:10001`) for dev. Mint dev API servers
  and driver-side scripts must direct-attach through `MINT_RAY_GCS_ADDRESS`.
- Do not start dev API servers on the Ray head pod; use the dev driver
  (`mint-dev` / `mint-dev-driver`) only.
- Do not run local `ray` or `volc` CLI commands. Use project Python with Ray
  for inspection, and use the `volcano-cluster` skill for lifecycle work.
- Do not source or print private config unless the task requires it.
- Do not switch ports to hide a failed restart; fix the listener or process.
- Do not default `MINT_CODE_ROOT` to the shared dev checkout. It is a required,
  explicit input; ask the user which checkout to run.
- Do not invent a Ray namespace. Derive `mint_<user>`; if that resolves to
  root, ask the user for `MINT_DEV_USER` or `MINT_RAY_NAMESPACE`.
- Do not install packages until the runtime root (`PFS_RUNTIME_ENV_ROOT`) and
  the resolved `PYTHONPATH` have been verified.
- Do not finish dev validation without cleaning owned zombie namespaces, actors,
  and placement groups, or recording why cleanup was blocked.
- **NEVER use `rsync --delete`** to sync code or runtime assets. It can remove
  files that other users or the runtime depend on. Use `rsync -a` without
  `--delete`, or copy individual files explicitly.
