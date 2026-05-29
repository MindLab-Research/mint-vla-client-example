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
| Ray Client address | `ray://$(cat /vePFS-Mindverse/share/mint/dev/ray/head-address/ray_head_ip.txt):10001` |
| Shell access | No SSH access to the dev head; attach only through Ray Client |
| API port | `8000` |
| Code checkout | `/vePFS-Mindverse/share/mint/dev/mint-server` |
| Runtime root | `/vePFS-Mindverse/share/mint/dev/runtime` |
| Ray namespace | `mint_${USER}` with a non-root user name |
| Public config | `/vePFS-Mindverse/share/mint/dev/config/common.env` |
| Private config | `/vePFS-Mindverse/share/mint/dev/config/secrets.env` |
| Log file | `/vePFS-Mindverse/share/mint/dev/logs/mint_server_auth.log` |

Dev config is split deliberately:
- `common.env`: non-secret deployment config such as port, Ray address, runtime
  root, code root, namespace, log path, model lists, and feature flags.
- `secrets.env`: private values such as API keys or credentials. Source it only
  when needed; never print it, commit it, or paste its contents into logs.

Use `/vePFS-Mindverse/share/mint/dev/config/common.env` as the dev server startup contract.
The dev Ray head is reachable only as a Ray Client endpoint. Do not use
historical `ssh mint-dev` commands.

Dev namespaces are user-scoped. Derive them from the effective non-root user:

```bash
MINT_DEV_USER="${MINT_DEV_USER:-${USER:-$(id -un)}}"
if [ -z "${MINT_DEV_USER}" ] || [ "${MINT_DEV_USER}" = "root" ]; then
  echo "error: dev Ray namespace requires a non-root user name" >&2
  exit 1
fi
export MINT_RAY_NAMESPACE="mint_${MINT_DEV_USER}"
```

Do not hard-code shared namespaces such as `tinker_leixiang` in dev commands.

## Code Versioning

The dev server code is managed as a PFS git checkout. Do not use file sync tools.
Because the current dev head has no SSH shell access, update the checkout from a
machine that has the PFS mounted, then record the branch and commit SHA before
testing.

```bash
cd /vePFS-Mindverse/share/mint/dev/mint-server
git fetch origin
git status --short --branch
git rev-parse HEAD
```

For issue branches, checkout the requested remote branch in
`/vePFS-Mindverse/share/mint/dev/mint-server` and record the commit SHA before testing.

## Start Or Restart

Use the runtime interpreter from `/vePFS-Mindverse/share/mint/dev/runtime`; do not use system
Python for server startup or Ray inspection.

The current dev head cannot be entered with SSH, so the old `pkill`/`nohup`
restart path is invalid. Do not restart the shared dev API process until the
API process owner/launcher for the Ray-Client-only topology is identified.

For local or issue-scoped dev servers running on a machine with the PFS mounted,
use the project launcher and point it at the dev Ray head:

```bash
cd /vePFS-Mindverse/share/mint/dev/mint-server
HEAD_IP="$(cat /vePFS-Mindverse/share/mint/dev/ray/head-address/ray_head_ip.txt)"
RAY_ADDRESS="ray://${HEAD_IP}:10001" \
MINT_RAY_CLIENT_ADDRESS="ray://${HEAD_IP}:10001" \
scripts/start_dev_server.sh
```

After any code change, restart the server before validating behavior. Python
servers do not hot-reload. In the Ray-Client-only dev topology, treat shared
server restart as blocked unless the current API launcher is known.

## Issue-Scoped Cleanup

Clean up detached Ray actors you created for issue-scoped dev/API validation
when they are no longer needed. Only clean namespaces that are clearly yours,
for example a prefix containing the issue/PR and your user name. Never clean
shared namespaces such as `tinker_leixiang` or another user's namespace.

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

## Internal Ops

For `/internal/*` actor inventory, actor kill, scheduler diagnostics, deep
health, and Ray diagnostics, read and use the `mint-ops` skill after this one.
Public `/api/v1/healthz` is only a cheap API-worker health check.

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
- Do not install packages until the runtime root and `PYTHONPATH` from
  `/vePFS-Mindverse/share/mint/dev/config/common.env` have been verified.
