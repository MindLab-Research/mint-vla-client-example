---
name: mint-dev
description: |
  Development environment operations for the Mint server on the Volcano dev host.

  Use for: starting/restarting the dev API server, reading dev logs, updating the
  dev server checkout, and validating local changes against the dev deployment.

  Triggers: "dev server", "start dev", "restart dev", "dev logs", "sync to dev",
  "dev vLLM", "mint-dev"

  Do not use this skill for production. Use mint-prod instead.
  For cluster lifecycle work, use volcano-cluster.

  Procedure contract: read this SKILL.md end-to-end before acting.
---

# Mint Dev

Read this whole file before touching the dev environment. If a step is missing,
update the skill instead of reviving old deployment paths.

## Environment

| Item | Value |
|------|-------|
| Ray head task | `mint-dev-head` |
| Ray head IP | Read from `/vePFS-Mindverse/share/mint/dev/ray/head-address/ray_head_ip.txt` |
| Shell access | No SSH access to the dev head; attach only through Ray Client |
| API port | `8000` (code default) |
| Runtime root | `/vePFS-Mindverse/share/mint/dev/runtime` (launcher default) |
| Ray namespace | `mint_${USER}` with a non-root user name |
| Private config | `/vePFS-Mindverse/share/mint/dev/config/secrets.env` |
| Log file | `/vePFS-Mindverse/share/mint/dev/logs/mint_server_auth.log` |

The dev Ray head is reachable only as a Ray Client endpoint. Do not use
historical `ssh mint-dev` commands.

### Launch contract: minimal inputs

`scripts/start_dev_server.sh` is the dev launcher. It takes the smallest
possible input set and lets everything else fall back to code defaults
(host/port, `LD_LIBRARY_PATH`, vLLM child python, HF modules, supported-model
list). It does NOT source a shared `common.env`; that historical file hardcodes
a fixed code checkout and a shared Ray namespace, which the per-launch contract
forbids.

| Variable | Role | Source |
|----------|------|--------|
| `MINT_CODE_ROOT` | mint-server checkout to run | **required, no default** |
| `MINT_RAY_NAMESPACE` | actor namespace | derived `mint_<user>`, refuses root/empty |
| `PFS_RUNTIME_ENV_ROOT` | prebuilt host-venv (interpreter + torch/vllm), not business code | defaults to dev runtime |
| `MINT_RAY_HEAD_ADDRESS_PATH` | head-address file; server reads live head IP | defaults to canonical dev path |
| `MINT_TMP_ROOT` | scratch root | defaults to dev tmp |
| `MINT_DEV_DEPLOYMENT_ENV` | optional deployment policy (models, placement, prewarm, OTEL) | optional; must not set code root or namespace |
| `MINT_RAY_JOB_WORKING_DIR` | (not recommended) Ray working_dir override | leave unset |

Do **not** set `MINT_RAY_JOB_WORKING_DIR`. Setting it causes Ray to package and
upload the entire directory (~100-240 MB) over the Ray Client connection on
every job. Workers access code via `PYTHONPATH` which already points at the PFS
path in `MINT_CODE_ROOT`.

`MINT_CODE_ROOT` has no default on purpose. It **must** be a path under
`/vePFS-Mindverse/share/` that is visible to all Ray nodes (head + workers).
Each user should maintain their own personal checkout — the exact path is up to
you, as long as it is under `/vePFS-Mindverse/share/` and not the shared dev
checkout. Do **not** use:

- `/root/code/mint` or any other local path — Ray head nodes cannot see it
- `/vePFS-Mindverse/user/<you>/...` — the `/user` subtree is not mounted on Ray nodes
- `/vePFS-Mindverse/share/mint/dev/mint-server` — shared checkout, changes here affect everyone

Sync your local checkout to your chosen share path before starting:

```bash
rsync -a --delete <your-local-checkout>/ /vePFS-Mindverse/share/<your-path>/
MINT_CODE_ROOT=/vePFS-Mindverse/share/<your-path> MINT_DEV_USER=<you> \
  MINT_DEV_DEPLOYMENT_ENV=/share/mint/dev/config/common.env \
  nohup scripts/start_dev_server.sh >> /tmp/mint_dev.log 2>&1 &
```

If it is unset, the launcher refuses. **Ask the user which checkout to run**
before launching; do not guess.

`MINT_RAY_NAMESPACE` is user-scoped. The launcher derives `mint_<user>` from the
effective non-root user and refuses `mint`, `root`, `mint_root`, or empty. **If
the namespace cannot be derived (for example you run as root), ask the user**
for `MINT_RAY_NAMESPACE` or `MINT_DEV_USER` instead of inventing one. Do not
hard-code shared namespaces such as `tinker_leixiang`.

`secrets.env` holds private values (API keys, credentials). Source it only when
needed; never print it, commit it, or paste its contents into logs.

## Code Versioning

The dev server code is whatever checkout you pass as `MINT_CODE_ROOT`. Manage it
as a git checkout; do not use file sync tools. Record the branch and commit SHA
before testing.

```bash
cd "${MINT_CODE_ROOT}"
git fetch origin
git status --short --branch
git rev-parse HEAD
```

For issue branches, check out the requested branch in your `MINT_CODE_ROOT` and
record the commit SHA before testing. There is no canonical shared checkout the
launcher defaults to; the checkout is always an explicit input.

## Start Or Restart

The launcher re-execs into the runtime interpreter under
`PFS_RUNTIME_ENV_ROOT/host-venv`; do not use system Python for server startup or
Ray inspection.

The current dev head cannot be entered with SSH, so the old `pkill`/`nohup`
restart path is invalid. Do not restart the shared dev API process until the
API process owner/launcher for the Ray-Client-only topology is identified.

For a local or issue-scoped dev server on a machine with the PFS mounted, the
one-shot minimal launch is:

```bash
MINT_CODE_ROOT=/path/to/your/mint-server-checkout \
MINT_DEV_USER=<you> \
scripts/start_dev_server.sh
```

That is the whole required input: the checkout to run, plus a non-root user for
the namespace. Runtime root, head address, port, model list, and library paths
all use defaults. The launcher prints the resolved contract to stderr before
starting; review it. Override a default only when needed, for example:

```bash
MINT_CODE_ROOT=/path/to/checkout MINT_DEV_USER=<you> \
PFS_RUNTIME_ENV_ROOT=/path/to/other/runtime \
MINT_DEV_DEPLOYMENT_ENV=/path/to/deployment.env \
scripts/start_dev_server.sh
```

If `MINT_CODE_ROOT` is missing the launcher refuses; ask the user which checkout
to run. If the namespace resolves to root the launcher refuses; ask the user for
`MINT_RAY_NAMESPACE` or `MINT_DEV_USER`.

After any code change, restart the server before validating behavior. Python
servers do not hot-reload. In the Ray-Client-only dev topology, treat shared
server restart as blocked unless the current API launcher is known.

### Issue-scoped servers

There is no separate issue launcher. Isolation between concurrent dev servers
comes entirely from the Ray namespace: every control-plane actor is looked up
with `namespace=`, so a per-launch `mint_<user>` (or issue-scoped) namespace
already gives the server its own scheduler, task-store, and cron actors. Run an
isolated issue server with `start_dev_server.sh` plus a scoped namespace, port,
and log, and any issue-specific tuning as plain env vars:

```bash
MINT_CODE_ROOT=/path/to/issue/checkout \
MINT_RAY_NAMESPACE=mint_<you>_issue_<n> \
MINT_PORT=10416 \
MINT_LOG_FILE=/tmp/mint_server_issue_<n>.log \
MINT_DISABLE_MINT_ROUTE=1 MINT_UVICORN_WORKERS=1 \
scripts/start_dev_server.sh
```

## Issue-Scoped Cleanup

Clean up detached Ray actors you created for issue-scoped dev/API validation
when they are no longer needed. Only clean namespaces that are clearly yours,
for example a prefix containing the issue/PR and your user name. Never clean
shared namespaces such as `tinker_leixiang` or another user's namespace.

Also stop local issue-scoped API servers or TermDeck-backed tasks that you
started once the validation is finished or blocked. Before stopping a process,
verify all of the following:

- the process cwd, listening port, log path, or TermDeck session name matches
  your issue-scoped run
- `/api/v1/server_info` or the startup log identifies the expected local
  process, port, branch/SHA, and namespace
- no paired validation command is still running against that port

Use `SIGTERM` or TermDeck control input first, then confirm the port is no
longer listening. Do not kill shared dev API processes or another user's
server just because they are old.

Before killing anything, list exact targets and verify the namespace prefix:

```bash
HEAD_IP="$(cat /vePFS-Mindverse/share/mint/dev/ray/head-address/ray_head_ip.txt)"
export CLEANUP_NAMESPACE_PREFIX="mint_pr668_"
RAY_ADDRESS="ray://${HEAD_IP}:10001" \
/vePFS-Mindverse/share/mint/dev/runtime/host-venv/bin/python - <<'PY'
import os
import ray

prefix = os.environ["CLEANUP_NAMESPACE_PREFIX"]
ray.init(address=os.environ["RAY_ADDRESS"], namespace="mint_cleanup", ignore_reinit_error=True, log_to_driver=False)
targets = []
for actor in ray.util.list_named_actors(all_namespaces=True):
    namespace = str(actor.get("namespace") or actor.get("ray_namespace") or "")
    name = str(actor.get("name") or actor.get("actor_name") or "")
    if namespace.startswith(prefix) and name:
        targets.append((namespace, name))
print(targets)
ray.shutdown()
PY
```

If every target is from the issue-scoped namespace you created and the task is
finished, kill those actors with `ray.kill(..., no_restart=True)` and re-list to
confirm `remaining=[]`. Remove only placement groups that are named for the same
issue-scoped actors/namespace. Do not use local `ray` CLI commands.

## Health And Logs

```bash
curl http://localhost:8000/api/v1/healthz
```

If validating Ray connectivity directly, use Ray Client:

```bash
PY=/vePFS-Mindverse/share/mint/dev/runtime/host-venv/bin/python
export HEAD_IP="$(cat /vePFS-Mindverse/share/mint/dev/ray/head-address/ray_head_ip.txt)"
$PY - <<'PY'
import ray
import os
ray.init(address=f"ray://{os.environ['HEAD_IP']}:10001")
print(ray.cluster_resources())
ray.shutdown()
PY
```

For local or issue-scoped dev servers with `MINT_INTERNAL_API_TOKEN` configured,
`/internal/*` also requires platform-forwarded identity headers. A plain
`X-API-Key` request can return `401 {"error":"Missing platform auth headers"}`.
When auth is enabled, source the dev secrets without printing values and send
the internal token plus a synthetic operator identity:

```bash
set -a
. /vePFS-Mindverse/share/mint/dev/config/secrets.env
set +a
/usr/bin/curl -s \
  -H "X-Internal-Token: ${MINT_INTERNAL_API_TOKEN:-}" \
  -H "X-MinT-User-Id: 000000000000000000000001" \
  -H "X-MinT-User-Role: admin" \
  -H "X-MinT-Account-Id: 000000000000000000000001" \
  -H "X-MinT-Apikey-Id: 000000000000000000000002" \
  -H "X-MinT-Request-Id: dev-operator-check" \
  http://localhost:8000/internal/model_actor_supervisor
```

If the dev server is intentionally running without `MINT_INTERNAL_API_TOKEN`,
internal routes use local/dev pass-through and do not need these headers. Check
which mode the target server uses before concluding an endpoint is broken.

## Internal Ops

For `/internal/*` actor inventory, actor kill, scheduler diagnostics, deep
health, and Ray diagnostics, read and use the `mint-ops` skill after this one.
Public `/api/v1/healthz` is only a cheap API-worker health check.

## Control-Plane Actor Refresh

Dev and prod share the same detached control-plane actor pattern. After a code
or config update, startup can fail because a stale namespace-local control-plane
actor has an old config fingerprint or code identity. Common messages include
`ConfigActorSnapshotMismatchError` for `mint_config` and
`model_actor_supervisor_code_mismatch` for `mint_model_actor_supervisor`.

Do not restart Ray or worker nodes for these failures. In dev, first confirm the
namespace is yours (`MINT_RAY_NAMESPACE="mint_${USER}"` or an issue-scoped
namespace). Then kill only the stale control-plane actor in that namespace with
the project runtime Python and Ray Client:

```bash
PY=/vePFS-Mindverse/share/mint/dev/runtime/host-venv/bin/python
HEAD_IP="$(cat /vePFS-Mindverse/share/mint/dev/ray/head-address/ray_head_ip.txt)"
export RAY_ADDRESS="ray://${HEAD_IP}:10001"
export MINT_RAY_NAMESPACE="mint_${USER}"
$PY - <<'PY'
import os
import ray

namespace = os.environ["MINT_RAY_NAMESPACE"]
ray.init(address=os.environ["RAY_ADDRESS"], namespace=namespace, ignore_reinit_error=True, log_to_driver=False)
actor = ray.get_actor("mint_config", namespace=namespace)
ray.kill(actor, no_restart=True)
print(f"killed mint_config namespace={namespace}")
ray.shutdown()
PY
```

Restart the issue-scoped API server after the control-plane actor is removed.
If the namespace is shared or belongs to another user, stop and ask before
killing anything. `/internal/model_actor_supervisor` is the source of truth for
topology/runtime health after a supervisor control-plane rebuild; `/internal/actors`
is a backend publication inventory and may be empty until backend actors publish
again.

## Worker Node Lifecycle

Dev GPU worker lifecycle is topology/Supervisor owned. Do not use historical
Volcano CLI commands to list, submit, cancel, or inspect worker jobs.

Use the `volcano-cluster` skill. The dev head is `mint-dev-head`, and Python
attaches through Ray Client using the current head-address file:

```bash
HEAD_IP="$(cat /vePFS-Mindverse/share/mint/dev/ray/head-address/ray_head_ip.txt)"
RAY_ADDRESS="ray://${HEAD_IP}:10001" \
MINT_RAY_CLIENT_ADDRESS="ray://${HEAD_IP}:10001" \
/vePFS-Mindverse/share/mint/dev/runtime/host-venv/bin/python scripts/tools/volcano_sdk_jobs.py --region cn-beijing list --name-contains mint-dev-worker- --limit 200
```

Credential checks may report only source existence. Never print secret values,
credential files, signed requests, or process environments.

## Hard Rules

- Do not perform production operations from this skill.
- Do not run local `ray` or `volc` CLI commands. Use project Python with Ray
  Client for inspection, and use the `volcano-cluster` skill for lifecycle work.
- Do not use `ssh mint-dev`; the dev head is Ray-Client-only.
- Do not source or print private config unless the task requires it.
- Do not switch ports to hide a failed restart; fix the listener or process that
  owns port `8000`.
- Do not default `MINT_CODE_ROOT` to the shared dev checkout. It is a required,
  explicit input; ask the user which checkout to run.
- Do not invent a Ray namespace. Derive `mint_<user>`; if that resolves to root
  or is otherwise unavailable, ask the user.
- Do not install packages until the runtime root (`PFS_RUNTIME_ENV_ROOT`) and the
  resolved `PYTHONPATH` have been verified.
