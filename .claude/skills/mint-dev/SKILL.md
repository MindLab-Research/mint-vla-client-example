---
name: mint-dev
description: |
  Development environment operations for the Mint server on Volcano cluster.

  Use for: code sync, server start/stop, vLLM management, logs - all in DEV environment.

  Triggers: "dev server", "start dev", "restart dev", "dev logs", "sync to dev", "dev vLLM"

  **Do NOT invoke this skill for production deployment. Use mint-prod instead.**

  For cluster lifecycle (create/teardown tasks), invoke the volcano-cluster skill.
---

# Mint Development Environment

> **STOP. USE THESE COMMANDS EXACTLY.**
>
> Do NOT guess SSH hosts, log locations, or process names. Everything is documented below.
>
> | Task | Command |
> |------|---------|
> | SSH to server | `ssh mint-dev` (NOT direct IP) |
> | Server logs | `ssh mint-dev "tail -50 /tmp/tinker_server.log"` |
> | Health check | `curl http://localhost:8000/api/v1/healthz` |
> | Restart server | See "Start Server" section below |
> | Kill vLLM | `curl -X POST http://localhost:8000/api/v1/kill_vllm` |
>
> If you find yourself guessing or trial-and-error debugging basic infrastructure, **STOP and re-read this skill**.
>
> Note: `GET /api/v1/healthz` is the cheap public API-worker health endpoint. For costly Ray / placement-group diagnostics, use the internal deep health surface instead of expecting `healthz` to reflect cluster capacity.

> **CRITICAL: RESTART SERVER AFTER CODE CHANGES**
>
> Python servers do NOT hot-reload. After ANY code change:
> 1. Verify code synced: `ssh mint-dev 'grep "your_change" /path/to/file'`
> 2. **RESTART SERVER** (see section 2 below)
> 3. Verify new process: `ssh mint-dev 'ps aux | grep run_server'`
>
> **Server running old code = your fix does not exist.** This has wasted hours of debugging.

---

## NEVER Do These (Production Belongs to mint-prod-volcano / mint-prod-aliyun)

- **NEVER** `ssh mint-prod-volcano` - that's production (router)
- **NEVER** `ssh mint-prod-aliyun` - that's production (upstream)
- **NEVER** use port `18000` - that's production
- **NEVER** use `volcano-tinker-auth` unison profile - that's production
- **NEVER** use `mint-prod-*.yaml` Ray configs - that's production
- **NEVER** use `tinker-server-auth` directory - that's production
- **NEVER** sync to `/vePFS-Mindverse/share/code/tinker-server` (shared; causes devs to clobber each other)
- **NEVER** set `TINKER_PORT` - not needed for dev (uses default 8000)

If user asks for production operations, **stop and invoke mint-prod skill instead**.

---

## Environment Config

| Property | Value |
|----------|-------|
| SSH Host | `mint-dev` |
| Port | 8000 |
| Code Directory | `tinker-server` |
| PFS Path | Required: `/vePFS-Mindverse/share/code/$USER/tinker-server` |
| Unison Profile | Required: `volcano-tinker-$USER` |
| Ray Configs | `mint-dev-head.yaml`, `mint-dev-worker.yaml` |
| Dev GPU Queue | Do not assume a fixed queue. Confirm availability in Volcano console. If only prod GPU queues are available, get explicit user approval. |
| API Key | Not required (auth disabled when `TINKER_API_KEY` unset) |
| Log File | `/tmp/tinker_server.log` |

---

## Python And PYTHONPATH Invariants

For mint-dev operator work, use the canonical runtime-env host interpreter:

```bash
/vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python
```

The canonical runtime root also provides a matching Ray CLI wrapper:

```bash
/vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/ray --version
```

Do not use system Python for Ray inspection, actor probes, or server startup.
The dev Ray cluster is running Python 3.12.13, and host-side `ray.init(...)`
with the wrong interpreter will fail with version mismatch or import-path
errors.

For API-server startup, prefer a built runtime-env root plus its host interpreter:

```bash
python scripts/build_runtime_env.py --env-root /vePFS-Mindverse/share/code/mint-runtime-py31213
export PFS_RUNTIME_ENV_ROOT=/vePFS-Mindverse/share/code/mint-runtime-py31213
export PFS_TINKER_PATH=/vePFS-Mindverse/share/code/$USER/tinker-server
export PFS_HF_MODULES_PATH=/vePFS-Mindverse/share/huggingface/modules
export LD_LIBRARY_PATH=/vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/lib/python3.12/site-packages/torch/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64
/vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python scripts/run_server.py
```

Reason:
- actor `runtime_env` and API-host bootstrap now share the same canonical dependency root
- `scripts/run_server.py` bootstraps `PYTHONPATH` from `PFS_RUNTIME_ENV_ROOT`
- `LD_LIBRARY_PATH` must be injected before process start so the host Python 3.12 torch libs win over any inherited Python 3.10 host path
- repo-root-only startup still creates fake import failures

Do not pip-install packages until you have first verified that the API-host
runtime env root matches the intended PFS environment.

## Canonical Runtime Baseline

For long-running dev validation, merge-gate work, or any dev server bring-up:

- Do **not** assemble the server environment from scratch.
- Start from the authoritative prod runtime baseline:
  [configs/prod_volcano.env.sh](/home/yiwen/tinker_project/tinker-server/configs/prod_volcano.env.sh)
- Then override **only** the dev-specific values that must differ.

Hard rule:
- If you are typing a long list of `export ...` lines by hand, you are probably doing it wrong.
- The default move is:
  1. `cd /root/tinker_project/tinker-server`
  2. `. ./configs/prod_volcano.env.sh`
  3. override the minimum required dev values
  4. run `scripts/run_server.py`

Required dev overrides after sourcing `configs/prod_volcano.env.sh`:

- `TINKER_PORT=8000`
- `TINKER_LOG_FILE=/tmp/tinker_server.log`
- `TINKER_USAGE_LOG_DIR` to a dev-safe path
- `RAY_ADDRESS` for the current dev head
- `PFS_TINKER_PATH=/vePFS-Mindverse/share/code/$USER/tinker-server`
- `MINT_VLLM_CHILD_PYTHON_EXECUTABLE=/vePFS-Mindverse/share/code/$USER/tinker-server/scripts/vllm_worker_python.py`
- `TINKER_RAY_NAMESPACE` and `MINT_RAY_NAMESPACE` to a fresh per-run namespace
- `MINT_PERSISTENT_MODELS` if you need a reduced dev prewarm set
- `USE_MBRIDGE_LORA_EXPORT=1` when validating Megatron LoRA sampler export / vLLM hot-load behavior

## Pin Override Discipline

If you source `configs/prod_volcano.env.sh` and then target a different worker slice, you must override **all** relevant pinning variables together.

Do not override only one of them.

When moving from prod topology to a dev slice, update all of:

- `MINT_DENSE_MODEL_NODE_IPS_JSON`
- `MINT_MODEL_NODE_IPS_JSON`
- `MINT_VLLM_MODEL_NODE_IPS_JSON`
- `MINT_MEGATRON_MODEL_NODE_IPS_JSON`
- `MINT_VLLM_PINNED_NODE_IP_JSON`

If one of these still points at prod IPs, the run is contaminated even if the others are correct.

For exact-split 30B dev validation:

- pin Megatron training to one assigned worker
- pin 30B vLLM to a different assigned worker
- do not let both compete for the same 8-GPU node unless that is explicitly the scenario under test

## Readiness Gate

Before starting any real test run:

- `run_server.py` must still be alive
- `curl http://localhost:8000/api/v1/healthz` must return a valid response body
- do not treat a TCP accept followed by `connection reset by peer` as healthy
- do not start merge-gate items while startup prewarm is still in flight

If startup prewarm is enabled:

- wait for prewarm completion or prewarm failure
- only then treat the server as ready for validation

## Retry Gate

The placement-group check is not one-time setup. It is a gate before **every** retry.

Before each retry:

1. List all non-REMOVED placement groups cluster-wide.
2. Filter to the assigned node slice.
3. Remove **all** owned stale PGs on that slice by PG id taken directly from `placement_group_table()`.
4. Re-check the PG table.
5. Only then check physical GPU occupancy.
6. Only then retry actor creation or server restart.

Hard rule:

- If a retry happens without a fresh PG-table check, that retry is invalid.
- Actor kills and `nvidia-smi` do **not** replace the PG check.
- A node can look free physically and still be blocked logically by stale PGs.
- Stale PG removal must use the PG id from `placement_group_table()`. Do not bounce to other removal methods.
- Do **not** scope PG cleanup to the model you currently care about. The cleanup scope is the entire assigned slice.
- A server restart is itself a placement event because startup prewarm can place actors. Clear stale PGs for startup surfaces too, not just for the scenario you intend to run next.

## Placement Group Hygiene Is Mandatory

Before any new actor placement attempt on mint-dev:

1. List all non-REMOVED placement groups cluster-wide.
2. If any owned stale or pending PG can reserve GPUs on the assigned slice, remove it first.
3. Only after that, check physical GPU occupancy on the target nodes.
4. Only after both checks pass, start the server or actor.

Hard rule:
- Do not treat physically idle GPUs as sufficient evidence.
- A stale PG is a real blocker even when every GPU shows `2 MiB`.
- Do not retry placement until the stale PG is gone.
- The placement-group table is the oracle. Do not override it with weaker signals.
- The cleanup target is the whole assigned slice, not the currently investigated model.
- If you are about to restart the server, you must assume startup prewarm may place dense trainers, Megatron actors, vLLM actors, and control-plane actors. Clear stale PGs for all of them first.

Oracle precedence for placement debugging:

1. `placement_group_table()` filtered to the target node slice
2. actor state / named actor lookups
3. physical GPU occupancy (`nvidia-smi`)

If these disagree, the higher item wins.

In particular:

- `get_placement_group(name)` failing is **not** proof that the stale reservation is gone.
- `ray.get_actor(...)` failing is **not** proof that the stale reservation is gone.
- `nvidia-smi` showing `2 MiB` is **not** proof that the stale reservation is gone.
- If `placement_group_table()` still shows a non-REMOVED PG on the target nodes, the node is still blocked. Stop there and remove the PG before doing anything else.

Removal method:

1. Dump `placement_group_table()`.
2. Copy the exact PG id for the stale entry from that table.
3. Remove that PG by id.
4. Dump `placement_group_table()` again and verify the entry is gone.

Hard bans:

- Do **not** use `get_placement_group(name)` as the cleanup path.
- Do **not** treat actor kills as PG cleanup.
- Do **not** retry placement because a name lookup failed.
- Do **not** invent alternate cleanup methods while the PG table still shows the stale entry.
- Do **not** clear only the PGs for the item you plan to run next.
- Do **not** restart the server while any owned stale PG remains anywhere on the assigned slice.

Exact check pattern:

```bash
ssh mint-dev 'RAY_ADDRESS="${RAY_ADDRESS:?set explicit validated head:port first}" /vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python - <<'\''PY'\'''
import json
import os
import ray
from ray.util.placement_group import placement_group_table
ray.init(address=os.environ["RAY_ADDRESS"], ignore_reinit_error=True)
rows = []
for pgid, info in placement_group_table().items():
    if info.get("state") != "REMOVED":
        rows.append({
            "id": pgid,
            "name": info.get("name"),
            "state": info.get("state"),
            "stats": info.get("stats"),
        })
print(json.dumps(rows, indent=2))
PY'
```

Exact removal pattern:

```bash
ssh mint-dev 'RAY_ADDRESS="${RAY_ADDRESS:?set explicit validated head:port first}" /vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python - <<'\''PY'\'''
import os
import ray
from ray.util.placement_group import PlacementGroup, placement_group_table, remove_placement_group

TARGET_PG_ID = "replace_with_pgid_from_table"

ray.init(address=os.environ["RAY_ADDRESS"], ignore_reinit_error=True)
pg = PlacementGroup(ray._raylet.PlacementGroupID.from_hex(TARGET_PG_ID))
remove_placement_group(pg)
print(f"remove_requested {TARGET_PG_ID}")
print(placement_group_table().get(TARGET_PG_ID, {}))
PY'
```

If a stale PG is yours, remove it before any retry.

Additional hard rule:

- Repeat this check before **every** retry, not just once at the beginning.
- If a previous attempt failed, assume stale PGs may have been left behind until you prove otherwise.
- After **any** failed retry, go back to step 1 and re-enumerate the PG table before interpreting the failure.
- Never perform two consecutive large-model retries without a fresh PG-table dump in between.
- If you find yourself reasoning from memory about whether a PG is gone, stop and dump the table again.

---

**Worker queue selection:** `.claude/skills/volcano-cluster/configs/mint-dev-worker.yaml` uses a `<GPU_QUEUE_ID>` placeholder. Set it explicitly before submitting any new dev worker tasks.

## Concurrent Dev Runs (Issue #83)

Goal: isolate code + detached Ray actor state across developers sharing the same dev Ray cluster.

Required env vars:
- `TINKER_RAY_NAMESPACE`: Ray namespace for all server-owned actors (default `tinker`)
- `PFS_TINKER_PATH`: PFS code root used in Ray worker `runtime_env` `PYTHONPATH`
Hard rule: never create/get/kill Ray actors outside `TINKER_RAY_NAMESPACE` unless the user explicitly requests cross-namespace action.

### Unison Profile (Per-Developer)

Create a per-developer profile (no shared PFS root):

```bash
mkdir -p ~/.unison
sed "s/__PFS_USER__/$USER/g" .claude/skills/mint-dev/configs/volcano-tinker.prf > ~/.unison/volcano-tinker-$USER.prf

# Start unison as a persistent daemon (systemd --user)
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/unison@.service <<'EOF'
[Unit]
Description=Unison (%i) watch

[Service]
Type=simple
ExecStart=/usr/bin/unison %i -repeat watch -ui text
Restart=always
RestartSec=2
StandardOutput=append:/tmp/unison-%i.log
StandardError=append:/tmp/unison-%i.log

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
loginctl enable-linger "$USER" || true
systemctl --user enable --now "unison@volcano-tinker-$USER.service"
```

### Volcano Symlink (Per-Developer)

Point the server working tree at the same per-developer PFS directory:

```bash
ssh mint-dev "rm -rf /root/tinker_project/tinker-server && \
  ln -s /vePFS-Mindverse/share/code/$USER/tinker-server /root/tinker_project/tinker-server"
```

---

## Finding the Server Process

**Always verify the actual log file location before tailing logs:**

```bash
# Find server process
ssh mint-dev 'ps aux | grep run_server | grep -v grep'

# Check where stdout goes (actual log file)
ssh mint-dev 'ls -la /proc/<PID>/fd/1'

# Example output: /proc/12345/fd/1 -> /tmp/tinker_server.log
```

The log file is typically `/tmp/tinker_server.log`, but verify with the above if logs seem stale.

---

## Quick Reference

```bash
# SSH tunnel
ssh -f -N -L 8000:localhost:8000 mint-dev

# Health check
curl http://localhost:8000/api/v1/healthz

# Server logs
ssh mint-dev "tail -50 /tmp/tinker_server.log"

# vLLM status
curl http://localhost:8000/api/v1/vllm_status

# Kill vLLM
curl -X POST http://localhost:8000/api/v1/kill_vllm
```

---

## 1. Code Synchronization

> **CRITICAL: ALWAYS USE DAEMON MODE (`-repeat watch`)**
>
> **NEVER** run one-off `unison volcano-tinker-$USER -batch` commands. This causes stale code on workers.

> **PRE-FLIGHT CHECK:** Before ANY dev work, verify unison daemon is running:
> ```bash
> systemctl --user is-active --quiet "unison@volcano-tinker-$USER.service" || echo "WARNING: unison not running - server has outdated code!"
> ```
> If not running: `systemctl --user enable --now "unison@volcano-tinker-$USER.service"`

```bash
# Start daemon (keep running)
systemctl --user enable --now "unison@volcano-tinker-$USER.service"

# Check status
systemctl --user status "unison@volcano-tinker-$USER.service" --no-pager

# Logs
tail -n 200 "/tmp/unison-volcano-tinker-$USER.log"

# Stop daemon
systemctl --user stop "unison@volcano-tinker-$USER.service"
```

**First-time setup:**
```bash
mkdir -p ~/.unison
sed "s/__PFS_USER__/$USER/g" .claude/skills/mint-dev/configs/volcano-tinker.prf > ~/.unison/volcano-tinker-$USER.prf
```

**SSH server symlink setup** (one-time):
```bash
ssh mint-dev "rm -rf /root/tinker_project/tinker-server && \
  ln -s /vePFS-Mindverse/share/code/$USER/tinker-server /root/tinker_project/tinker-server"
```

---

## 2. Server Management

### Environment Variables

```bash
export HF_HUB_OFFLINE=1
export HF_HOME=/vePFS-Mindverse/share/huggingface
export PFS_HF_MODULES_PATH=/vePFS-Mindverse/share/huggingface/modules
export PYTHONDONTWRITEBYTECODE=1
export PFS_TINKER_PATH=/vePFS-Mindverse/share/code/$USER/tinker-server
export PFS_RUNTIME_ENV_ROOT=/vePFS-Mindverse/share/code/mint-runtime-py31213
export TINKER_HOST_PYTHON=/vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python
export TINKER_HOST_TORCH_LIB=/vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/lib/python3.12/site-packages/torch/lib
export LD_LIBRARY_PATH=$TINKER_HOST_TORCH_LIB:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64
# For concurrent dev runs, set this to a unique value (example: tinker_$USER).
# export TINKER_RAY_NAMESPACE=tinker
# Also set this to the same value (used by detached metadata stores):
# export MINT_RAY_NAMESPACE=$TINKER_RAY_NAMESPACE

# MoE vLLM placement mode:
# - Default: MINT_MOE_MULTINODE_MIN_GPUS=4, so Qwen3-30B (TP=4) uses MultiNodeInferenceEngine
#   (Ray distributed executor, can spread TP across nodes; slower but schedules under GPU fragmentation).
# - Set to 16 to force single-node MultiLoRAInferenceEngine for Qwen3-30B (requires 4 GPUs on one node).
# export MINT_MOE_MULTINODE_MIN_GPUS=16
```

**Note:** No default model is configured. Clients specify models per-request. Model paths are resolved via `_resolve_model_path()` in `multi_lora_engine.py`.

### Start Server

```bash
ssh mint-dev "cd /root/tinker_project/tinker-server && nohup bash -c \
  \"PFS_RUNTIME_ENV_ROOT=/vePFS-Mindverse/share/code/mint-runtime-py31213 \
   PFS_HF_MODULES_PATH=/vePFS-Mindverse/share/huggingface/modules \
   HF_HUB_OFFLINE=1 HF_HOME=/vePFS-Mindverse/share/huggingface \
   PYTHONDONTWRITEBYTECODE=1 \
   PFS_TINKER_PATH=/vePFS-Mindverse/share/code/$USER/tinker-server \
   LD_LIBRARY_PATH=/vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/lib/python3.12/site-packages/torch/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
   TINKER_RAY_NAMESPACE=${TINKER_RAY_NAMESPACE:-tinker_$USER} \
   MINT_RAY_NAMESPACE=${TINKER_RAY_NAMESPACE:-tinker_$USER} \
   /vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python scripts/run_server.py\" >> /tmp/tinker_server.log 2>&1 &"
```

### Stop Server

```bash
ssh mint-dev 'pkill -f "[p]ython scripts/run_server.py" 2>/dev/null || true'

# If multiple server processes remain, force kill:
ssh mint-dev 'pkill -9 -f "python scripts/run_server.py" 2>/dev/null || true'

# Verify:
ssh mint-dev 'ps aux | grep run_server | grep -v grep'
```

### Check Status

```bash
ssh mint-dev "ps aux | grep run_server | grep -v grep"
```

---

## 3. vLLM Actor

| Operation | Time | When to use |
|-----------|------|-------------|
| Reconnect (existing) | ~2s | Server restart, vLLM actor still alive |
| Kill + restart | ~80s | Base model changed, OOM, vLLM code changed |

### Kill vLLM Actor

```bash
# Via API (admin only when auth is enabled; do not kill random processes)
curl -X POST http://localhost:8000/api/v1/kill_vllm

# Kill specific model's vLLM actor
curl -X POST -H "Content-Type: application/json" \
  -d '{"model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507"}' \
  http://localhost:8000/api/v1/kill_vllm
```

### Kill Megatron Actor

```bash
curl -X POST http://localhost:8000/api/v1/kill_megatron

# Kill specific model's Megatron actor
curl -X POST -H "Content-Type: application/json" \
  -d '{"base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507"}' \
  http://localhost:8000/api/v1/kill_megatron
```

### Check Actor Status

```bash
# vLLM status
curl -s http://localhost:8000/api/v1/vllm_status | jq

# Megatron status
curl -s http://localhost:8000/api/v1/megatron_status | jq

# Kill all actors (nuclear option; admin only when auth is enabled)
curl -X POST http://localhost:8000/api/v1/kill_all_actors
```

---

## 4. Code Update SOP

### Rule 0: server restart does not reload detached actors

The API server is a Python process. Ray actors are separate Python processes.

Detached actors (vLLM, Megatron, DenseTrainerPool, stores) survive server restarts and keep running old code until killed.

### Kill criteria after code changes

Always restart the server after code changes.

Kill detached actors only if the change can be imported/executed inside that actor process:
- vLLM: `tinker_server/backend/verl_inference.py`, `tinker_server/backend/multi_lora_engine.py`, `tinker_server/backend/multinode_inference.py`, `tinker_server/backend/vllm_*.py`
- Megatron: `tinker_server/backend/megatron_distributed.py`, `tinker_server/backend/megatron_training.py`, `tinker_server/backend/verl_patches.py`
- Dense training pool: `tinker_server/backend/verl_training.py`
- Detached stores: `tinker_server/backend/future_store.py`, `tinker_server/backend/training_session_store.py`, `tinker_server/backend/gateway_session_store.py`
- Shared (kills required for all GPU actor types): `tinker_server/config.py`, `tinker_server/ray_utils.py`, `tinker_server/backend/ray_kill.py`, `tinker_server/backend/model_registry.py`

If none of the above changed: restart server only.

### Kill Actors

> **Actor naming convention:**
> - vLLM: `tinker_vllm_{model_name}` (e.g., `tinker_vllm_kimi-k2-thinking`)
> - Megatron: `megatron_{model_name}` (e.g., `megatron_kimi_k2_thinking`; model name is lowercased and `-`/`.` become `_`)
> - Dense training pool: `dense_trainer_pool_{model_name}_maxr{rank}` (e.g., `dense_trainer_pool_qwen3_4b_instruct_2507_maxr64`)
> - Stores: `tinker_future_store`, `tinker_training_session_store`, `tinker_gateway_session_store`
> - Namespace: `TINKER_RAY_NAMESPACE` (default `tinker`)
>
> Hard rule: never create/get/kill actors outside `TINKER_RAY_NAMESPACE` unless the user explicitly requests it.
>
> **When to kill actors:**
> - Implementation code changed (actors cache old code)
> - OOM or stuck state
> - Switching to different model

```bash
# Kill vLLM actor for K2
ssh mint-dev "RAY_ADDRESS='${RAY_ADDRESS:?set explicit validated head:port first}' TINKER_RAY_NAMESPACE='${TINKER_RAY_NAMESPACE:?unset}' /vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python -c \"
import os
import ray
ray.init(address=os.environ[\"RAY_ADDRESS\"], ignore_reinit_error=True)
try:
    ns = os.environ[\"TINKER_RAY_NAMESPACE\"]
    actor = ray.get_actor(\"tinker_vllm_kimi-k2-thinking\", namespace=ns)
    ray.kill(actor)
    print(\"Killed vLLM actor\")
except ValueError as e:
    print(f\"Actor not found: {e}\")
\""

# Kill Megatron actor for K2
ssh mint-dev "RAY_ADDRESS='${RAY_ADDRESS:?set explicit validated head:port first}' TINKER_RAY_NAMESPACE='${TINKER_RAY_NAMESPACE:?unset}' /vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python -c \"
import os
import ray
ray.init(address=os.environ[\"RAY_ADDRESS\"], ignore_reinit_error=True)
try:
    ns = os.environ[\"TINKER_RAY_NAMESPACE\"]
    actor = ray.get_actor(\"megatron_kimi_k2_thinking\", namespace=ns)
    ray.kill(actor)
    print(\"Killed Megatron actor\")
except ValueError as e:
    print(f\"Actor not found: {e}\")
\""

# List all actors in current namespace (to find actor names)
ssh mint-dev "RAY_ADDRESS='${RAY_ADDRESS:?set explicit validated head:port first}' TINKER_RAY_NAMESPACE='${TINKER_RAY_NAMESPACE:?unset}' MINT_RAY_NAMESPACE='${TINKER_RAY_NAMESPACE:?unset}' /vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python -c \"
import os
import ray
ray.init(address=os.environ[\"RAY_ADDRESS\"], ignore_reinit_error=True)
ns = os.environ[\"TINKER_RAY_NAMESPACE\"]
actors = ray.util.list_named_actors(all_namespaces=True)
for a in actors:
    if a.get(\"namespace\") == ns:
        print(a)
\""

# Kill all dense trainer pool actors in current namespace (prefix match)
ssh mint-dev "RAY_ADDRESS='${RAY_ADDRESS:?set explicit validated head:port first}' TINKER_RAY_NAMESPACE='${TINKER_RAY_NAMESPACE:?unset}' MINT_RAY_NAMESPACE='${TINKER_RAY_NAMESPACE:?unset}' /vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python -c \"
import os
import ray
ray.init(address=os.environ['RAY_ADDRESS'], ignore_reinit_error=True)
ns = os.environ['TINKER_RAY_NAMESPACE']
actors = ray.util.list_named_actors(all_namespaces=True)
killed = 0
for a in actors:
    if a.get('namespace') != ns:
        continue
    name = a.get('name') or ''
    if not name.startswith('dense_trainer_pool_'):
        continue
    try:
        ray.kill(ray.get_actor(name, namespace=ns))
        killed += 1
    except Exception as e:
        print(f\"kill_failed name={name!r} namespace={ns!r} err={e!r}\")
print(f\"killed={killed} prefix='dense_trainer_pool_' namespace={ns}\")
\""

# Kill detached store actors in current namespace (name match)
ssh mint-dev "RAY_ADDRESS='${RAY_ADDRESS:?set explicit validated head:port first}' TINKER_RAY_NAMESPACE='${TINKER_RAY_NAMESPACE:?unset}' MINT_RAY_NAMESPACE='${TINKER_RAY_NAMESPACE:?unset}' /vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python -c \"
import os
import ray
ray.init(address=os.environ['RAY_ADDRESS'], ignore_reinit_error=True)
ns = os.environ['TINKER_RAY_NAMESPACE']
names = ['tinker_future_store', 'tinker_training_session_store', 'tinker_gateway_session_store']
killed = 0
for name in names:
    try:
        ray.kill(ray.get_actor(name, namespace=ns))
        killed += 1
    except ValueError:
        pass
    except Exception as e:
        print(f\"kill_failed name={name!r} namespace={ns!r} err={e!r}\")
print(f\"killed={killed} kind='stores' namespace={ns}\")
\""
```

### Restart Server

Use this after server-only code changes. If you killed any actors, restart the server after the kill so in-process caches are cleared.

```bash
ssh mint-dev 'pkill -f "[p]ython scripts/run_server.py" 2>/dev/null || true'
ssh mint-dev "cd /root/tinker_project/tinker-server && nohup bash -c \
  \"PFS_RUNTIME_ENV_ROOT=/vePFS-Mindverse/share/code/mint-runtime-py31213 \
   PFS_HF_MODULES_PATH=/vePFS-Mindverse/share/huggingface/modules \
   HF_HUB_OFFLINE=1 HF_HOME=/vePFS-Mindverse/share/huggingface \
   PYTHONDONTWRITEBYTECODE=1 \
   PFS_TINKER_PATH=/vePFS-Mindverse/share/code/$USER/tinker-server \
   LD_LIBRARY_PATH=/vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/lib/python3.12/site-packages/torch/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
   TINKER_RAY_NAMESPACE=${TINKER_RAY_NAMESPACE:-tinker_$USER} \
   MINT_RAY_NAMESPACE=${TINKER_RAY_NAMESPACE:-tinker_$USER} \
   /vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python scripts/run_server.py\" >> /tmp/tinker_server.log 2>&1 &"
```

### Restart after killing vLLM

Use this after vLLM actor code changes, OOM, or switching base model.

```bash
curl -X POST http://localhost:8000/api/v1/kill_vllm
ssh mint-dev 'pkill -f "[p]ython scripts/run_server.py" 2>/dev/null || true'
ssh mint-dev "cd /root/tinker_project/tinker-server && nohup bash -c \
  \"PFS_RUNTIME_ENV_ROOT=/vePFS-Mindverse/share/code/mint-runtime-py31213 \
   PFS_HF_MODULES_PATH=/vePFS-Mindverse/share/huggingface/modules \
   HF_HUB_OFFLINE=1 HF_HOME=/vePFS-Mindverse/share/huggingface \
   PYTHONDONTWRITEBYTECODE=1 \
   PFS_TINKER_PATH=/vePFS-Mindverse/share/code/$USER/tinker-server \
   LD_LIBRARY_PATH=/vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/lib/python3.12/site-packages/torch/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
   TINKER_RAY_NAMESPACE=${TINKER_RAY_NAMESPACE:-tinker_$USER} \
   MINT_RAY_NAMESPACE=${TINKER_RAY_NAMESPACE:-tinker_$USER} \
   /vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python scripts/run_server.py\" >> /tmp/tinker_server.log 2>&1 &"
sleep 80 && curl -s http://localhost:8000/api/v1/healthz
```

### Restart after killing Megatron

Use this after Megatron actor code changes, OOM, or switching base model.

```bash
curl -X POST http://localhost:8000/api/v1/kill_megatron
ssh mint-dev 'pkill -f "[p]ython scripts/run_server.py" 2>/dev/null || true'
ssh mint-dev "cd /root/tinker_project/tinker-server && nohup bash -c \
  \"PFS_RUNTIME_ENV_ROOT=/vePFS-Mindverse/share/code/mint-runtime-py31213 \
   PFS_HF_MODULES_PATH=/vePFS-Mindverse/share/huggingface/modules \
   HF_HUB_OFFLINE=1 HF_HOME=/vePFS-Mindverse/share/huggingface \
   PYTHONDONTWRITEBYTECODE=1 \
   PFS_TINKER_PATH=/vePFS-Mindverse/share/code/$USER/tinker-server \
   LD_LIBRARY_PATH=/vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/lib/python3.12/site-packages/torch/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
   TINKER_RAY_NAMESPACE=${TINKER_RAY_NAMESPACE:-tinker_$USER} \
   MINT_RAY_NAMESPACE=${TINKER_RAY_NAMESPACE:-tinker_$USER} \
   /vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python scripts/run_server.py\" >> /tmp/tinker_server.log 2>&1 &"
curl -s http://localhost:8000/api/v1/healthz
```

### Restart after killing all tracked GPU actors

Use this after shared actor code changes (for example `tinker_server/backend/model_registry.py`) or when GPUs are exhausted.

Note: `/api/v1/kill_all_actors` kills ResourcePool-tracked GPU actors (vLLM, Megatron, dense trainer pool). It does not kill detached store actors.

```bash
curl -X POST http://localhost:8000/api/v1/kill_all_actors
ssh mint-dev 'pkill -f "[p]ython scripts/run_server.py" 2>/dev/null || true'
ssh mint-dev "cd /root/tinker_project/tinker-server && nohup bash -c \
  \"PFS_RUNTIME_ENV_ROOT=/vePFS-Mindverse/share/code/mint-runtime-py31213 \
   PFS_HF_MODULES_PATH=/vePFS-Mindverse/share/huggingface/modules \
   HF_HUB_OFFLINE=1 HF_HOME=/vePFS-Mindverse/share/huggingface \
   PYTHONDONTWRITEBYTECODE=1 \
   PFS_TINKER_PATH=/vePFS-Mindverse/share/code/$USER/tinker-server \
   LD_LIBRARY_PATH=/vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/lib/python3.12/site-packages/torch/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
   TINKER_RAY_NAMESPACE=${TINKER_RAY_NAMESPACE:-tinker_$USER} \
   MINT_RAY_NAMESPACE=${TINKER_RAY_NAMESPACE:-tinker_$USER} \
   /vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python scripts/run_server.py\" >> /tmp/tinker_server.log 2>&1 &"
sleep 80 && curl -s http://localhost:8000/api/v1/healthz
```

---

## 5. Ray Cluster

**Find Ray head task (if task ID unknown):**
```bash
ssh mint-dev '/root/.volc/bin/volc ml_task list --output json --limit 200' | jq '.[] | select(.Name | startswith("ray-head")) | {Id, Name, Status}'
```

**Get Ray head IP from task logs:**
```bash
ssh mint-dev '/root/.volc/bin/volc ml_task logs -t <head_task_id> -i worker_0' | grep "Local node IP"
```

**After a fresh dev-cluster rebuild, join `mint-dev` to the head as a zero-resource Ray node before any `ray.init(...)` or `scripts/run_server.py`:**
- Direct `ray.init(address=...)` from `mint-dev` can fail with local `node_ip_address.json` / missing-local-raylet errors even when the head is healthy.
- The working attach pattern is a local Ray node with `--num-cpus=0 --num-gpus=0`, so `mint-dev` remains a driver/API host and does not contribute schedulable compute.
- Do not use `hostname -I | awk '{print $1}'` for `MINT_RAY_NODE_IP_ADDRESS`. Derive the API-host IP from the route to the head.

**Placement-group hygiene before retrying a large actor:**
- If exact nodes are physically idle but `healthz` reports pending placement groups, inspect the global placement-group table before any retry.
- Remove only placement groups you own, by exact actor-name namespace match.
- Do not treat idle GPUs as proof that Ray has no logical reservations.
- If you have not listed non-REMOVED PGs yet, you are not ready to start a new actor.

**Correct attach and connectivity check:**
```bash
ssh mint-dev "python3 - <<'PY'\nimport socket\ns=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\ns.connect(('<RAY_HEAD_IP>', 6379))\nprint(s.getsockname()[0])\ns.close()\nPY"
ssh mint-dev "/vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/ray stop --force || true"
ssh mint-dev "/vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/ray start --address='<RAY_HEAD_IP>:6379' --num-cpus=0 --num-gpus=0 --disable-usage-stats"
ssh mint-dev "/vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/ray status --address='<RAY_HEAD_IP>:6379'"
ssh mint-dev "export MINT_RAY_NODE_IP_ADDRESS='<API_IP_FROM_ROUTE_TO_HEAD>'; export MINT_RAY_TEMP_DIR=/tmp/ray_api_driver; cd /vePFS-Mindverse/share/code/$USER/tinker-server; /vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python - <<'PY'\nfrom tinker_server.ray_utils import init_ray\nimport ray\ninit_ray(address='<RAY_HEAD_IP>:6379', ignore_reinit_error=True)\nprint(ray.cluster_resources())\nray.shutdown()\nPY"
```

**Canonical dev server bring-up after cluster rebuild:**
- 1. Attach `mint-dev` to the head with zero resources using `ray start --address=... --num-cpus=0 --num-gpus=0`.
- 2. Export `MINT_RAY_NODE_IP_ADDRESS` from the route-derived API IP and a fresh `MINT_RAY_TEMP_DIR`.
- 3. Then start `scripts/run_server.py`.
- 4. If `ray.init` fails before startup completes, stop and fix this attach step first. Do not thrash on server env, healthz, or training logic before the attach is correct.

**For cluster create/teardown, invoke the `volcano-cluster` skill.**

Dev-specific values:
- Ray head config: `.claude/skills/volcano-cluster/configs/mint-dev-head.yaml`
- Ray worker config: `.claude/skills/volcano-cluster/configs/mint-dev-worker.yaml`
- Task names: no "prod" prefix

---

## 6. GPU Requirements for MoE Models

> **CRITICAL: ALWAYS verify cluster has enough GPUs before starting MoE actors.**
>
> Insufficient GPUs cause pending placement groups that block the cluster.

### GPU Requirements by Model

| Model | vLLM (Inference) | Megatron (Training) | Total (Simultaneous) |
|-------|------------------|---------------------|----------------------|
| **Qwen3-30B-A3B** | TP=4 → **4 GPUs** | TP=4, EP=1 → **4 GPUs** | **8 GPUs** |
| **Moonlight-16B-A3B** | TP=8 → **8 GPUs** | TP=8, EP=8 → **8 GPUs** | **16 GPUs** |
| Dense models (7B-14B) | **1 GPU** | **1 GPU** | **2 GPUs** |

### Pre-flight Check (MANDATORY)

Before starting any MoE test, run:

```bash
# Quick status command (MANDATORY before any work)
ssh mint-dev '/vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python << "PYEOF"
import ray
ray.init(address="<RAY_HEAD_IP>:6379", ignore_reinit_error=True)
r = ray.available_resources()
t = ray.cluster_resources()
gpu_avail = r.get("GPU", 0)
gpu_total = t.get("GPU", 0)
print(f"GPUs: {gpu_avail:.0f} / {gpu_total:.0f}")
# List actors by prefix (vLLM actors are named tinker_vllm_{model_name})
actors = ray.util.list_named_actors(all_namespaces=True)
for a in actors:
    name = a["name"]
    if name.startswith("tinker_vllm_") or name.startswith("megatron_"):
        print(f"{name}: ALIVE")
PYEOF'

# Check pending placement groups (MUST be empty)
ssh mint-dev "/vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/ray status --address='<RAY_HEAD_IP>:6379' 2>/dev/null | grep -A5 'Pending Demands'"
```

**Required for Qwen3-30B-A3B tests:** At least 8 available GPUs and no pending placement groups.

### Parallelism Configuration

**vLLM (Inference)** - configured in `model_registry.py`:
- TP (tensor_parallel): Shards model weights across GPUs
- DP (data_parallel): Runs multiple model replicas
- MoE uses expert parallelism: EP = TP × DP

**Megatron (Training)** - configured in `verl_training.py`:
- TP=4: Tensor parallelism (shards attention/FFN)
- EP=2: Expert parallelism (distributes MoE experts)
- world_size = TP × PP × EP × CP = 4 × 1 × 2 × 1 = 8 GPUs

### Clearing Stuck Resources

If placement groups are pending (blocking GPUs):

```bash
# Kill Megatron actor (see "Kill Actors" section above for commands)
# Kill vLLM actor (see "Kill Actors" section above for commands)

# Verify resources freed
ssh mint-dev "/vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/ray status --address='<RAY_HEAD_IP>:6379' 2>/dev/null | head -20"
```

---

## 7. Debugging

```bash
# Error search
ssh mint-dev "grep -i 'error\\|exception\\|traceback' /tmp/tinker_server.log | tail -20"

# Training worker logs
ssh mint-dev "grep 'TrainingWorker' /tmp/tinker_server.log | tail -20"

# Forward/backward issues
ssh mint-dev "grep 'loss_fn_inputs\\|Missing' /tmp/tinker_server.log | tail -10"
```

---

## 8. Running Test Scripts

> **CRITICAL: Test Scripts Run LOCALLY, Not on Server**
>
> Test scripts that use HTTP API (pytest, `scripts/tools/smoke.py service`, etc.) run on your LOCAL machine.
> Local machine has internet access for downloading tokenizers from HuggingFace Hub.
>
> **Do NOT:**
> - Run test scripts on the server (no internet for tokenizer downloads)
> - Set `HF_HUB_OFFLINE=1` or `HF_HOME=/vePFS-...` for test scripts
>
> **Server commands** (ssh mint-dev '...') need HF_HUB_OFFLINE because the server has no internet.
> **Test commands** run locally and download tokenizers automatically.

```bash
# Ensure SSH tunnel is active
ssh -f -N -L 8000:localhost:8000 mint-dev

# Run test script LOCALLY (downloads tokenizer from HuggingFace)
# CRITICAL: Always set TINKER_TELEMETRY=0 to prevent log flooding
TINKER_BASE_URL=http://localhost:8000 TINKER_TELEMETRY=0 python scripts/tools/smoke.py service

# Run merge gate tests LOCALLY
TINKER_BASE_URL=http://localhost:8000 TINKER_TELEMETRY=0 python -m pytest .claude/skills/merge-gate/tests/ -v

# For training scripts (e.g., tinker_cookbook)
TINKER_BASE_URL=http://localhost:8000 TINKER_TELEMETRY=0 python -m tinker_cookbook.recipes.math_rl.train ...
```

---

## 9. Ray Actor Status and Logs

> **CRITICAL: NEVER assume actor state without verifying.**
>
> A failed `ray.get_actor()` lookup could mean: wrong name, wrong namespace, or actor actually dead.
> **ALWAYS list actors first** to see what exists before concluding anything.

### List All Actors (DO THIS FIRST)

```bash
# List all actors - this shows actual names and states
ssh mint-dev 'RAY_ADDRESS="${RAY_ADDRESS:?set explicit validated head:port first}" /vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/ray list actors --address "$RAY_ADDRESS" 2>&1 | grep -E "(vllm|megatron|Extended)" | head -20'

# Or list with full details
ssh mint-dev 'RAY_ADDRESS="${RAY_ADDRESS:?set explicit validated head:port first}" /vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/ray list actors --address "$RAY_ADDRESS" --filter "state=ALIVE" 2>&1 | head -30'
```

### Check Specific Actor Status

```bash
# WRONG: Guessing actor name and concluding "DEAD" if not found
# RIGHT: List first, then check with exact name from list

ssh mint-dev 'RAY_ADDRESS="${RAY_ADDRESS:?set explicit validated head:port first}" /vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python -c "
import os
import ray
ray.init(address=os.environ[\"RAY_ADDRESS\"], ignore_reinit_error=True)

# List actors first to get exact names
actors = ray.util.list_named_actors(all_namespaces=True)
print(\"Named actors:\")
for a in actors:
    print(f\"  {a}\")
"'
```

### Get Actor Logs

```bash
# Get actor ID from ray list actors output, then:
ssh mint-dev 'RAY_ADDRESS="${RAY_ADDRESS:?set explicit validated head:port first}" /vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/ray logs actor --address "$RAY_ADDRESS" --id <ACTOR_ID> --tail 100 2>&1'

# Example with actual ID:
ssh mint-dev 'RAY_ADDRESS="${RAY_ADDRESS:?set explicit validated head:port first}" /vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/ray logs actor --address "$RAY_ADDRESS" --id 618fd2b45b4f8ac797dafdbd1e000000 --tail 100 2>&1'
```

### List Dead Actors (for crash investigation)

```bash
ssh mint-dev 'RAY_ADDRESS="${RAY_ADDRESS:?set explicit validated head:port first}" /vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/ray list actors --address "$RAY_ADDRESS" --filter "state=DEAD" 2>&1 | head -30'
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Unison not running | `pgrep -af "unison.*volcano-tinker-$USER"` then restart daemon |
| Symlink broken | Re-run symlink setup command |
| Server won't start | Check logs: `tail -100 /tmp/tinker_server.log` |
| Can't connect | Check SSH tunnel, Ray cluster connection |
| vLLM OOM | Kill vLLM actor, restart server |
| Pending placement groups | Not enough GPUs. Kill stale actors (see section 6) |
| MoE test hangs on startup | Check GPU availability first. Need 8 GPUs for 30B MoE |
| Tokenizer download fails | Run test script locally, not on server (server has no internet) |
| Actor lookup fails | **LIST actors first** (`ray list actors`), don't assume dead |

---

## 10. Scripts Directory Structure

```
scripts/
├── run_server.py      # Server entry point (core)
├── tools/             # Reusable debug utilities (tracked in git)
└── wip/               # Work-in-progress investigations (gitignored)
```

**Workflow:**
- Active investigation scripts → `scripts/wip/` (not tracked)
- Scripts worth sharing/collaborating → promote to `scripts/tools/`
- Throwaway scripts → delete after use

**Do NOT** accumulate investigation scripts in `scripts/` root.
