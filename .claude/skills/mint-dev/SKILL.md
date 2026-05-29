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
| SSH host | `mint-dev` |
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

The dev server code is managed as a git checkout on the server host. Do not use
file sync tools.

```bash
ssh mint-dev 'cd /vePFS-Mindverse/share/mint/dev/mint-server && git fetch origin && git status --short --branch'
ssh mint-dev 'cd /vePFS-Mindverse/share/mint/dev/mint-server && git checkout refactor && git pull --ff-only origin refactor'
```

For issue branches, checkout the requested remote branch in
`/vePFS-Mindverse/share/mint/dev/mint-server` and record the commit SHA before testing.

## Start Or Restart

Use the runtime interpreter from `/vePFS-Mindverse/share/mint/dev/runtime`; do not use system
Python for server startup or Ray inspection.

```bash
ssh mint-dev 'pkill -f "[p]ython scripts/run_server.py" 2>/dev/null || true'

ssh mint-dev 'cat > /vePFS-Mindverse/share/mint/dev/tmp/start_mint_dev.sh <<'"'"'SH'"'"'
#!/usr/bin/env bash
set -euo pipefail
cd /vePFS-Mindverse/share/mint/dev/mint-server
set -a
. /vePFS-Mindverse/share/mint/dev/config/common.env
if [ -f /vePFS-Mindverse/share/mint/dev/config/secrets.env ]; then
  . /vePFS-Mindverse/share/mint/dev/config/secrets.env
fi
set +a
MINT_DEV_USER="${MINT_DEV_USER:-${USER:-$(id -un)}}"
if [ -z "${MINT_DEV_USER}" ] || [ "${MINT_DEV_USER}" = "root" ]; then
  echo "error: dev Ray namespace requires a non-root user name" >&2
  exit 1
fi
export MINT_RAY_NAMESPACE="${MINT_RAY_NAMESPACE:-mint_${MINT_DEV_USER}}"
exec /vePFS-Mindverse/share/mint/dev/runtime/host-venv/bin/python scripts/run_server.py
SH
chmod +x /vePFS-Mindverse/share/mint/dev/tmp/start_mint_dev.sh
nohup /vePFS-Mindverse/share/mint/dev/tmp/start_mint_dev.sh >> /vePFS-Mindverse/share/mint/dev/logs/mint_server_auth.log 2>&1 &'
```

After any code change, restart the server before validating behavior. Python
servers do not hot-reload.

## Health And Logs

```bash
curl http://localhost:8000/api/v1/healthz
ssh mint-dev 'tail -n 200 /vePFS-Mindverse/share/mint/dev/logs/mint_server_auth.log'
ssh mint-dev 'ps aux | grep "[s]cripts/run_server.py"'
```

If using a local tunnel:

```bash
ssh -f -N -L 8000:localhost:8000 mint-dev
MINT_BASE_URL=http://localhost:8000 MINT_API_KEY=dummy python scripts/tools/smoke.py service
```

## Internal Ops

For `/internal/*` actor inventory, actor kill, scheduler diagnostics, deep
health, and Ray diagnostics, read and use the `mint-ops` skill after this one.
Public `/api/v1/healthz` is only a cheap API-worker health check.

## Worker Node Lifecycle

Dev GPU worker lifecycle is topology/Supervisor owned. Do not use historical
Volcano CLI commands to list, submit, cancel, or inspect worker jobs.

Use the `volcano-cluster` skill. The operator entrypoint on `mint-dev` is:

```bash
ssh mint-dev 'cd /vePFS-Mindverse/share/mint/dev/mint-server && /vePFS-Mindverse/share/mint/dev/runtime/host-venv/bin/python scripts/tools/volcano_sdk_jobs.py --region cn-beijing list --name-contains mint-dev-worker- --limit 200'
ssh mint-dev 'cd /vePFS-Mindverse/share/mint/dev/mint-server && /vePFS-Mindverse/share/mint/dev/runtime/host-venv/bin/python scripts/tools/volcano_sdk_jobs.py --region cn-beijing submit-topology-node --config /vePFS-Mindverse/share/mint/dev/runtime/topology.yaml --alias mint-worker-0'
```

Credential checks may report only source existence. Never print secret values,
credential files, signed requests, or process environments.

## Hard Rules

- Do not perform production operations from this skill.
- Do not run local `ray` or `volc` commands. Use the environment host or the
  `volcano-cluster` skill.
- Do not source or print private config unless the task requires it.
- Do not switch ports to hide a failed restart; fix the listener or process that
  owns port `8000`.
- Do not install packages until the runtime root and `PYTHONPATH` from
  `/vePFS-Mindverse/share/mint/dev/config/common.env` have been verified.
