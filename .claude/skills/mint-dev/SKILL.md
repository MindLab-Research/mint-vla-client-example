---
name: mint-dev
description: |
  Development environment operations for the Mint server on Volcano cluster.

  Use for: code sync, server start/stop, vLLM management, logs - all in DEV environment.

  Triggers: "dev server", "start dev", "restart dev", "dev logs", "sync to dev", "dev vLLM"

  **Do NOT invoke this skill for production deployment. Use mint-prod instead.**

  For cluster lifecycle (create/teardown tasks), invoke the volcano-cluster skill.

  Procedure contract: read this SKILL.md end-to-end before acting. Do not slice it on demand or use it as a lookup table mid-run.
---

# Mint Development Environment

Procedure contract:
- Read this SKILL.md end-to-end before taking any action.
- Do not sample sections opportunistically while already in motion.
- If the procedure is missing something important, update the skill. Do not improvise around the gap.

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
> | Kill vLLM | `curl -X POST -H "Content-Type: application/json" -d '{"actor_type":"vllm"}' http://localhost:8000/api/v1/actors/kill` |
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
- Start from the checked-in dev template:
  [configs/dev_volcano.env.sh](../../../configs/dev_volcano.env.sh)
- Then override **only** the dev-specific values that must differ.

Hard rule:
- If you are typing a long list of `export ...` lines by hand, you are probably doing it wrong.
- The default move is:
  1. `cd /root/tinker_project/tinker-server`
  2. `. ./configs/dev_volcano.env.sh`
  3. override the minimum required dev values
  4. run `scripts/run_server.py`
- Any ad hoc startup command that does not follow that pattern is invalid evidence.
- Do not debug failures from a hand-built startup command. Stop, throw it away, and return to the runbook.
- If the startup needs many overrides, write them in a small script file on the server and execute that file after sourcing `configs/dev_volcano.env.sh`. Do not keep nesting shell quotes until the env becomes unverifiable.

Required overrides after sourcing `configs/dev_volcano.env.sh`:

- `RAY_ADDRESS` and `MINT_RAY_CLIENT_ADDRESS` for the current dev head (`ray://<RAY_HEAD_IP>:10001` for Python attach)
- `PFS_TINKER_PATH=/vePFS-Mindverse/share/code/$USER/tinker-server`
- `MINT_VLLM_CHILD_PYTHON_EXECUTABLE=/vePFS-Mindverse/share/code/$USER/tinker-server/scripts/vllm_worker_python.py`
- `TINKER_RAY_NAMESPACE` and `MINT_RAY_NAMESPACE` to a fresh per-run namespace
- `MINT_SUPPORTED_MODELS` / `MINT_PERSISTENT_MODELS` if you need a reduced dev prewarm set
- any pinning JSONs and control-plane pin vars needed for your assigned node slice
- `USE_MBRIDGE_LORA_EXPORT=1` when validating Megatron LoRA sampler export / vLLM hot-load behavior

Why these overrides are still mandatory:

- `configs/dev_volcano.env.sh` is the right dev baseline, but it contains checked-in example values for a specific prior dev setup.
- In particular, do not trust its checked-in `RAY_ADDRESS`, namespace, or `PFS_TINKER_PATH` for your run.
- Source it first, then overwrite those run-specific values explicitly.

## Ray Attach Mode On The API Host

Hard rule:

- On `mint-dev`, do not point Python `ray.init(...)` or `scripts/run_server.py` at raw GCS `192.168.39.31:6379`.
- From the API host, use the Ray client endpoint for Python attach: `ray://<RAY_HEAD_IP>:10001`.
- Symptom of getting this wrong: startup stalls around Ray attach and logs `Can't find a node_ip_address.json`.

Use this split:

- CLI health checks: `ray status --address=<RAY_HEAD_IP>:6379`
- Python attach on the API host: `ray.init(address="ray://<RAY_HEAD_IP>:10001")`
- Isolated API server startup: set both `RAY_ADDRESS` and `MINT_RAY_CLIENT_ADDRESS` to `ray://<RAY_HEAD_IP>:10001`

If a Python attach on the API host still uses `:6379`, stop and fix that first.

## Isolated Debug Server For Path-Based Checkpoints

Use this when you need a private dev server for checkpoint loading or issue-specific evaluation.

Hard rules:

- Use a fresh `TINKER_RAY_NAMESPACE` and the same `MINT_RAY_NAMESPACE`.
- Use a fresh `MINT_STARTUP_LEASE_ACTOR_NAME`, otherwise the server may come up as a follower and `/asample` can fail because detached stores belong to some other run.
- Set `MINT_UVICORN_WORKERS=1` for isolated debug bring-up. Multi-worker startup can thrash on the Ray init lock and hide the real issue.
- If requests will pass absolute checkpoint directories in `model_path` or `state_path`, enable auth with a known admin key (for example `TINKER_API_KEY=dummy`) and send the same key in client requests. Absolute paths are rejected for non-admin requests.
- If you do not need absolute paths, prefer `mint://...` or `ckpt_...` identifiers.

Minimal isolated bring-up pattern:

```bash
ssh -f -N -L 8010:localhost:8010 mint-dev

ssh mint-dev 'cat > /tmp/start_tinker_issue.sh <<'\''SH'\'''
#!/bin/bash
set -euo pipefail
cd /root/tinker_project/tinker-server
. ./configs/dev_volcano.env.sh
export RAY_ADDRESS=ray://<RAY_HEAD_IP>:10001
export MINT_RAY_CLIENT_ADDRESS=ray://<RAY_HEAD_IP>:10001
export PFS_TINKER_PATH=/vePFS-Mindverse/share/code/$USER/tinker-server
export MINT_VLLM_CHILD_PYTHON_EXECUTABLE=/vePFS-Mindverse/share/code/$USER/tinker-server/scripts/vllm_worker_python.py
export TINKER_API_KEY=dummy
export TINKER_PORT=8010
export MINT_LOG_FILE=/tmp/tinker_server_issue.log
export TINKER_USAGE_LOG_DIR=/tmp/tinker_usage_issue
export TINKER_RAY_NAMESPACE=tinker_<issue>
export MINT_RAY_NAMESPACE=tinker_<issue>
export MINT_STARTUP_LEASE_ACTOR_NAME=tinker_startup_lease_<issue>
export MINT_UVICORN_WORKERS=1
exec /vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python scripts/run_server.py
SH
chmod +x /tmp/start_tinker_issue.sh
nohup /tmp/start_tinker_issue.sh >> /tmp/tinker_server_issue.log 2>&1 &'
```

Preflight gate before **any** isolated startup or retry:

1. API-host import probe must pass from the intended issue PFS root.
2. Ray `runtime_env` import probe must pass using the same `actor_runtime_env(PFS_PYTHONPATH)` path that detached actors will use.
3. The exact intended API port must be free on the API host.
4. If any probe fails, do **not** start `run_server.py`. Fix that resource first.

Port/listener preflight is a hard gate:

- Treat the API port as an explicit resource, independent of Ray actors, placement groups, and GPU state.
- Before every isolated startup or retry, check the exact intended port on `mint-dev`.
- If the port is occupied by your previous issue server, kill that exact listener and re-check the same port.
- Do not switch to another port to work around a conflict unless the user explicitly changes the port.
- Do not declare a retry valid unless the logs later show that `run_server.py` bound the intended port.

Exact port check. Use Python and `/proc`; `ss`, `fuser`, and `lsof` are not guaranteed on the API host.

```bash
ssh mint-dev 'PORT=8010 /usr/bin/python3 - <<'"'"'PY'"'"'
import os
import socket

port = int(os.environ["PORT"])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("0.0.0.0", port))
except OSError as exc:
    raise SystemExit(f"PORT_BLOCKED port={port} errno={exc.errno} message={exc.strerror}")
finally:
    sock.close()
print(f"PORT_FREE port={port}")
PY'
```

Exact listener cleanup for an owned issue server:

```bash
ssh mint-dev 'PORT=8010 ISSUE_ROOT=/vePFS-Mindverse/share/code/$USER/tinker-server-issue-<ISSUE> ISSUE_LOG=/tmp/tinker_server_issue.log /usr/bin/python3 - <<'"'"'PY'"'"'
import os
import signal
import socket
import time
from pathlib import Path

port = int(os.environ["PORT"])
issue_root = os.environ["ISSUE_ROOT"]
issue_log = os.environ["ISSUE_LOG"]

def socket_inodes(path):
    try:
        rows = Path(path).read_text().splitlines()[1:]
    except FileNotFoundError:
        return set()
    inodes = set()
    for row in rows:
        cols = row.split()
        if cols[3] == "0A" and int(cols[1].rsplit(":", 1)[1], 16) == port:
            inodes.add(cols[9])
    return inodes

inodes = socket_inodes("/proc/net/tcp") | socket_inodes("/proc/net/tcp6")
killed = []
for proc in Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    try:
        owns_port = any(
            os.readlink(fd).startswith("socket:[") and os.readlink(fd)[8:-1] in inodes
            for fd in (proc / "fd").iterdir()
        )
    except Exception:
        continue
    if not owns_port:
        continue
    cwd = os.path.realpath(proc / "cwd")
    out = os.path.realpath(proc / "fd" / "1")
    if cwd == issue_root or out == issue_log:
        os.kill(int(proc.name), signal.SIGTERM)
        killed.append({"pid": int(proc.name), "cwd": cwd, "stdout": out})
time.sleep(2)
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("0.0.0.0", port))
except OSError as exc:
    print({"killed": killed, "port_free": False, "errno": exc.errno, "message": exc.strerror})
    raise SystemExit(1)
finally:
    sock.close()
print({"killed": killed, "port_free": True})
PY'
```

API-host import probe:

```bash
ssh mint-dev 'cd /root/tinker_project/tinker-server-issue-<ISSUE> && \
  PFS_RUNTIME_ENV_ROOT=/vePFS-Mindverse/share/code/mint-runtime-py31213 \
  PFS_TINKER_PATH=/vePFS-Mindverse/share/code/$USER/tinker-server-issue-<ISSUE> \
  PFS_HF_MODULES_PATH=/vePFS-Mindverse/share/huggingface/modules \
  PYTHONPATH=/vePFS-Mindverse/share/code/$USER/tinker-server-issue-<ISSUE> \
  /vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python - <<'\''PY'\'''
import json
import os
import tinker_server
from tinker_server.runtime_env import build_runtime_pythonpath

print(json.dumps({
    "tinker_server_file": tinker_server.__file__,
    "runtime_pythonpath": build_runtime_pythonpath(
        env_root=os.environ["PFS_RUNTIME_ENV_ROOT"],
        pfs_tinker_path=os.environ["PFS_TINKER_PATH"],
        pfs_hf_modules_path=os.environ["PFS_HF_MODULES_PATH"],
    ),
}, indent=2))
PY'
```

Ray `runtime_env` import probe:

```bash
ssh mint-dev 'cd /root/tinker_project/tinker-server-issue-<ISSUE> && \
  PFS_RUNTIME_ENV_ROOT=/vePFS-Mindverse/share/code/mint-runtime-py31213 \
  PFS_TINKER_PATH=/vePFS-Mindverse/share/code/$USER/tinker-server-issue-<ISSUE> \
  PFS_HF_MODULES_PATH=/vePFS-Mindverse/share/huggingface/modules \
  RAY_ADDRESS=ray://<RAY_HEAD_IP>:10001 \
  MINT_RAY_CLIENT_ADDRESS=ray://<RAY_HEAD_IP>:10001 \
  TINKER_RAY_NAMESPACE=tinker_<issue> \
  MINT_RAY_NAMESPACE=tinker_<issue> \
  MINT_DETACHED_ACTOR_NODE_IP=<CONTROL_PLANE_IP> \
  PYTHONPATH=/vePFS-Mindverse/share/code/$USER/tinker-server-issue-<ISSUE> \
  /vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python - <<'\''PY'\'''
import json
import os
import ray
from tinker_server.config import PFS_PYTHONPATH, actor_runtime_env

ray.init(address=os.environ["RAY_ADDRESS"], ignore_reinit_error=True)

@ray.remote(num_cpus=0, runtime_env=actor_runtime_env(pythonpath=PFS_PYTHONPATH))
def probe():
    import os
    import sys
    import tinker_server
    return {
        "tinker_server_file": tinker_server.__file__,
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "sys_path_head": sys.path[:8],
    }

print(json.dumps(ray.get(probe.remote()), indent=2))
PY'
```

Ray client `working_dir` packaging:

- In Ray client mode, detached actor creation may deserialize actor classes before
  the `PYTHONPATH` runtime env is applied. If a detached actor fails with
  `ModuleNotFoundError: No module named 'tinker_server'` even though the
  `runtime_env` import probe passes, package the issue checkout into a PFS zip
  and set `MINT_RAY_JOB_WORKING_DIR=file:///...zip` before starting the server.
- Build the zip from the synced issue checkout, not from a stale shared tree.
  Include the repo packages and runtime config needed for class import, for
  example `tinker_server`, `scripts`, and `configs`.
- Make the zip filename or URI versioned for every code change that affects
  detached actors. Ray caches `working_dir` by URI; reusing the same URI can run
  old actor code after a server restart.
- This package is only for Ray client class distribution. It does not replace
  unison; unison remains the source of truth for syncing the issue checkout to
  PFS.

Example:

```bash
ssh mint-dev 'cd /vePFS-Mindverse/share/code/$USER/tinker-server-issue-<ISSUE> && \
  ZIP=/vePFS-Mindverse/share/code/$USER/tinker_server_issue<ISSUE>_working_dir_$(date +%s).zip && \
  /usr/bin/python3 -m zipfile -c "$ZIP" tinker_server scripts configs && \
  echo "export MINT_RAY_JOB_WORKING_DIR=file://$ZIP"'
```

Hard bans for isolated startup:

- Do **not** invent a new startup command when one retry fails.
- Do **not** debug `run_server.py` startup until both import probes pass.
- Do **not** treat `connection reset by peer` as a model bug or server bug before the import probes pass.
- Do **not** use system `python3` for these probes.
- Do **not** “quickly test” a modified env inline if you cannot print and verify the exact values first.

Before debugging model behavior, verify this private server can:

1. return `200` on `/api/v1/healthz`;
2. create a sampling session against the intended checkpoint;
3. accept one `/api/v1/asample` request.

If one of those fails, fix that first. Do not pretend the model logic is under test yet.

## Pin Override Discipline

If you source `configs/dev_volcano.env.sh` and then target a different worker slice, you must override **all** relevant pinning variables together.

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
curl "http://localhost:8000/api/v1/actors?type=vllm"

# Kill vLLM
curl -X POST -H "Content-Type: application/json" \
  -d '{"actor_type":"vllm"}' \
  http://localhost:8000/api/v1/actors/kill
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
  \". ./configs/dev_volcano.env.sh && \
   export RAY_ADDRESS=\${RAY_ADDRESS:?set to ray://<head>:10001} && \
   export MINT_RAY_CLIENT_ADDRESS=\${MINT_RAY_CLIENT_ADDRESS:-\$RAY_ADDRESS} && \
   export PFS_TINKER_PATH=/vePFS-Mindverse/share/code/$USER/tinker-server && \
   export MINT_VLLM_CHILD_PYTHON_EXECUTABLE=/vePFS-Mindverse/share/code/$USER/tinker-server/scripts/vllm_worker_python.py && \
   export TINKER_RAY_NAMESPACE=\${TINKER_RAY_NAMESPACE:-tinker_$USER} && \
   export MINT_RAY_NAMESPACE=\${TINKER_RAY_NAMESPACE:-tinker_$USER} && \
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
curl -X POST -H "Content-Type: application/json" \
  -d '{"actor_type":"vllm"}' \
  http://localhost:8000/api/v1/actors/kill

# Kill specific model's vLLM actor
curl -X POST -H "Content-Type: application/json" \
  -d '{"actor_type":"vllm","model_name":"Qwen/Qwen3-30B-A3B-Instruct-2507"}' \
  http://localhost:8000/api/v1/actors/kill
```

### Kill Megatron Actor

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"actor_type":"megatron"}' \
  http://localhost:8000/api/v1/actors/kill

# Kill specific model's Megatron actor
curl -X POST -H "Content-Type: application/json" \
  -d '{"actor_type":"megatron","model_name":"Qwen/Qwen3-30B-A3B-Instruct-2507"}' \
  http://localhost:8000/api/v1/actors/kill
```

### Check Actor Status

```bash
# vLLM status
curl -s "http://localhost:8000/api/v1/actors?type=vllm" | jq

# Megatron status
curl -s "http://localhost:8000/api/v1/actors?type=megatron" | jq

# Kill all tracked GPU actors (admin only when auth is enabled)
curl -X POST -H "Content-Type: application/json" \
  -d '{"actor_type":"all"}' \
  http://localhost:8000/api/v1/actors/kill
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
- Detached stores/schedulers: `tinker_server/backend/task_state_store.py`, `tinker_server/backend/model_work_scheduler.py`, `tinker_server/backend/model_runtime_actor.py`, `tinker_server/backend/maintenance_cron_actor.py`, `tinker_server/backend/training_session_store.py`, `tinker_server/backend/gateway_session_store.py`
- Shared (kills required for all GPU actor types): `tinker_server/config.py`, `tinker_server/ray_utils.py`, `tinker_server/backend/ray_kill.py`, `tinker_server/backend/model_registry.py`

If none of the above changed: restart server only.

### Kill Actors

> **Actor naming convention:**
> - vLLM: `tinker_vllm_{model_name}` (e.g., `tinker_vllm_kimi-k2-thinking`)
> - Megatron: `megatron_{model_name}` (e.g., `megatron_kimi_k2_thinking`; model name is lowercased and `-`/`.` become `_`)
> - Dense training pool: `dense_trainer_pool_{model_name}_maxr{rank}` (e.g., `dense_trainer_pool_qwen3_4b_instruct_2507_maxr64`)
> - Stores/schedulers: `mint_task_state_store`, `mint_model_work_scheduler`, `mint_model_runtime_*`, `mint_maintenance_cron`, `tinker_training_session_store`, `tinker_gateway_session_store`
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

# Kill detached store/scheduler actors in current namespace (name/prefix match)
ssh mint-dev "RAY_ADDRESS='${RAY_ADDRESS:?set explicit validated head:port first}' TINKER_RAY_NAMESPACE='${TINKER_RAY_NAMESPACE:?unset}' MINT_RAY_NAMESPACE='${TINKER_RAY_NAMESPACE:?unset}' /vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python -c \"
import os
import ray
ray.init(address=os.environ['RAY_ADDRESS'], ignore_reinit_error=True)
ns = os.environ['TINKER_RAY_NAMESPACE']
names = ['mint_task_state_store', 'mint_model_work_scheduler', 'mint_maintenance_cron', 'tinker_training_session_store', 'tinker_gateway_session_store']
prefixes = ('mint_model_runtime_',)
killed = 0
for row in ray.util.list_named_actors(all_namespaces=True):
    if row.get('namespace') != ns:
        continue
    name = str(row.get('name') or '')
    if name not in names and not any(name.startswith(prefix) for prefix in prefixes):
        continue
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
  \". ./configs/dev_volcano.env.sh && \
   export RAY_ADDRESS=\${RAY_ADDRESS:?set to ray://<head>:10001} && \
   export MINT_RAY_CLIENT_ADDRESS=\${MINT_RAY_CLIENT_ADDRESS:-\$RAY_ADDRESS} && \
   export PFS_TINKER_PATH=/vePFS-Mindverse/share/code/$USER/tinker-server && \
   export MINT_VLLM_CHILD_PYTHON_EXECUTABLE=/vePFS-Mindverse/share/code/$USER/tinker-server/scripts/vllm_worker_python.py && \
   export TINKER_RAY_NAMESPACE=\${TINKER_RAY_NAMESPACE:-tinker_$USER} && \
   export MINT_RAY_NAMESPACE=\${TINKER_RAY_NAMESPACE:-tinker_$USER} && \
   /vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python scripts/run_server.py\" >> /tmp/tinker_server.log 2>&1 &"
```

### Restart after killing vLLM

Use this after vLLM actor code changes, OOM, or switching base model.

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"actor_type":"vllm"}' \
  http://localhost:8000/api/v1/actors/kill
ssh mint-dev 'pkill -f "[p]ython scripts/run_server.py" 2>/dev/null || true'
ssh mint-dev "cd /root/tinker_project/tinker-server && nohup bash -c \
  \". ./configs/dev_volcano.env.sh && \
   export RAY_ADDRESS=\${RAY_ADDRESS:?set to ray://<head>:10001} && \
   export MINT_RAY_CLIENT_ADDRESS=\${MINT_RAY_CLIENT_ADDRESS:-\$RAY_ADDRESS} && \
   export PFS_TINKER_PATH=/vePFS-Mindverse/share/code/$USER/tinker-server && \
   export MINT_VLLM_CHILD_PYTHON_EXECUTABLE=/vePFS-Mindverse/share/code/$USER/tinker-server/scripts/vllm_worker_python.py && \
   export TINKER_RAY_NAMESPACE=\${TINKER_RAY_NAMESPACE:-tinker_$USER} && \
   export MINT_RAY_NAMESPACE=\${TINKER_RAY_NAMESPACE:-tinker_$USER} && \
   /vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python scripts/run_server.py\" >> /tmp/tinker_server.log 2>&1 &"
sleep 80 && curl -s http://localhost:8000/api/v1/healthz
```

### Restart after killing Megatron

Use this after Megatron actor code changes, OOM, or switching base model.

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"actor_type":"megatron"}' \
  http://localhost:8000/api/v1/actors/kill
ssh mint-dev 'pkill -f "[p]ython scripts/run_server.py" 2>/dev/null || true'
ssh mint-dev "cd /root/tinker_project/tinker-server && nohup bash -c \
  \". ./configs/dev_volcano.env.sh && \
   export RAY_ADDRESS=\${RAY_ADDRESS:?set to ray://<head>:10001} && \
   export MINT_RAY_CLIENT_ADDRESS=\${MINT_RAY_CLIENT_ADDRESS:-\$RAY_ADDRESS} && \
   export PFS_TINKER_PATH=/vePFS-Mindverse/share/code/$USER/tinker-server && \
   export MINT_VLLM_CHILD_PYTHON_EXECUTABLE=/vePFS-Mindverse/share/code/$USER/tinker-server/scripts/vllm_worker_python.py && \
   export TINKER_RAY_NAMESPACE=\${TINKER_RAY_NAMESPACE:-tinker_$USER} && \
   export MINT_RAY_NAMESPACE=\${TINKER_RAY_NAMESPACE:-tinker_$USER} && \
   /vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python scripts/run_server.py\" >> /tmp/tinker_server.log 2>&1 &"
curl -s http://localhost:8000/api/v1/healthz
```

### Restart after killing all tracked GPU actors

Use this after shared actor code changes (for example `tinker_server/backend/model_registry.py`) or when GPUs are exhausted.

Note: `/api/v1/actors/kill` with `{"actor_type":"all"}` kills ModelActorRegistry-tracked GPU actors (vLLM, Megatron, dense trainer pool). It does not kill detached store actors.

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"actor_type":"all"}' \
  http://localhost:8000/api/v1/actors/kill
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

**Do not run `ray start` on `mint-dev`:**
- `mint-dev` is a driver/API host. Starting a local raylet makes it schedulable and can steal actor placement.
- Dev uses Ray client mode, so the local bastion does not need to join the cluster as a zero-resource driver node.
- Use `ray.init(address=...)` in Python or Ray CLI commands that connect to the head directly.

**Placement-group hygiene before retrying a large actor:**
- If exact nodes are physically idle but `healthz` reports pending placement groups, inspect the global placement-group table before any retry.
- Remove only placement groups you own, by exact actor-name namespace match.
- Do not treat idle GPUs as proof that Ray has no logical reservations.
- If you have not listed non-REMOVED PGs yet, you are not ready to start a new actor.

**Safe connectivity check with Ray client mode:**
```bash
ssh mint-dev "/vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/ray status --address='<RAY_HEAD_IP>:6379'"
ssh mint-dev "/vePFS-Mindverse/share/code/mint-runtime-py31213/host-venv/bin/python - <<'PY'\nimport ray\nray.init(address='ray://<RAY_HEAD_IP>:10001')\nprint(ray.cluster_resources())\nray.shutdown()\nPY"
```

**Canonical dev server bring-up after cluster rebuild:**
- 1. Verify the head is healthy with `ray status --address=...`.
- 2. Verify Python connectivity from the API host with `ray.init(address='ray://<RAY_HEAD_IP>:10001')`.
- 3. Then start `scripts/run_server.py`.
- 4. If `ray.init` fails before startup completes, fix head connectivity first. Do not thrash on server env, healthz, or training logic before the client connection is correct.
- 5. If you are starting a private issue server, use a fresh namespace, a fresh startup-lease actor name, and `MINT_UVICORN_WORKERS=1`.

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
ray.init(address="ray://<RAY_HEAD_IP>:10001", ignore_reinit_error=True)
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
