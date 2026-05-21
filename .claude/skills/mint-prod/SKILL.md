---
name: mint-prod
description: |
  Production environment operations for the Mint server on the Volcano production host.

  Use for: production server restart, production logs, updating the production
  server checkout, production health checks, and production internal ops setup.

  Triggers: "prod server", "start prod", "restart prod", "prod logs",
  "sync to prod", "production", "mint-prod"

  Do not use this skill for development work. Use mint-dev instead.
  For cluster lifecycle work, use volcano-cluster.

  Procedure contract: read this SKILL.md end-to-end before acting.
---

# Mint Prod

Read this whole file before touching production. Production is shared and
auth-required; do not guess process names, paths, or credentials.

## Environment

| Item | Value |
|------|-------|
| SSH host | `mint-prod-volcano` |
| API port | `18000` |
| Public URLs | `https://mint.macaron.im`, `https://mint.macaron.xin` |
| Code checkout | `/vePFS-Mindverse/share/mint/prod/mint-server` |
| Runtime root | `/vePFS-Mindverse/share/mint/prod/runtime` |
| Public config | `/vePFS-Mindverse/share/mint/prod/config/prod.env` |
| Private config | `/vePFS-Mindverse/share/mint/prod/config/secrets.env` |
| Log file | `/vePFS-Mindverse/share/mint/prod/logs/mint_server_auth.log` |

Production config is split deliberately:
- `prod.env`: non-secret deployment config such as port, Ray address, runtime
  root, code root, namespace, log path, model lists, and feature flags.
- `secrets.env`: private values such as API keys or credentials. Source it only
  when needed; never print it, commit it, or paste its contents into logs.

Use `/vePFS-Mindverse/share/mint/prod/config/prod.env` as the production server startup contract.

## Code Versioning

Production server code is managed as a git checkout on the production host. Do
not deploy with file sync tools.

```bash
ssh mint-prod-volcano 'cd /vePFS-Mindverse/share/mint/prod/mint-server && git fetch origin && git status --short --branch'
ssh mint-prod-volcano 'cd /vePFS-Mindverse/share/mint/prod/mint-server && git checkout refactor && git pull --ff-only origin refactor'
```

Record the commit SHA before and after a production deploy:

```bash
ssh mint-prod-volcano 'cd /vePFS-Mindverse/share/mint/prod/mint-server && git rev-parse HEAD'
```

## Start Or Restart

Prefer the configured process supervisor if one is present on the host. Verify
the program first instead of assuming a historical name.

```bash
ssh mint-prod-volcano 'supervisorctl status | grep -Ei "mint|tinker|server"'
```

If the process is not supervisor-managed, start it with the shared runtime and
config:

```bash
ssh mint-prod-volcano 'cat > /vePFS-Mindverse/share/mint/prod/tmp/start_mint_prod.sh <<'"'"'SH'"'"'
#!/usr/bin/env bash
set -euo pipefail
cd /vePFS-Mindverse/share/mint/prod/mint-server
set -a
. /vePFS-Mindverse/share/mint/prod/config/prod.env
. /vePFS-Mindverse/share/mint/prod/config/secrets.env
set +a
exec /vePFS-Mindverse/share/mint/prod/runtime/host-venv/bin/python scripts/run_server.py
SH
chmod +x /vePFS-Mindverse/share/mint/prod/tmp/start_mint_prod.sh'
```

Use the exact configured supervisor program or an explicit listener/process
target for restart. Do not use broad `pkill` patterns in production.

## Health And Logs

```bash
curl http://localhost:18000/api/v1/healthz
ssh mint-prod-volcano 'tail -n 200 /vePFS-Mindverse/share/mint/prod/logs/mint_server_auth.log'
ssh mint-prod-volcano 'ps aux | grep "[s]cripts/run_server.py"'
```

Authenticated calls must include `X-API-Key`:

```bash
set -a
. /vePFS-Mindverse/share/mint/prod/config/secrets.env
set +a
curl -H "X-API-Key: $MINT_API_KEY" http://localhost:18000/internal/actors
```

## Internal Ops

For `/internal/*` actor inventory, actor kill, scheduler diagnostics, deep
health, and Ray diagnostics, read and use the `mint-ops` skill after this one.
Public `/api/v1/healthz` is only a cheap API-worker health check.

## Production Topology Notes

- Volcano production is the router/master deployment for the local production
  models.
- Additional upstream deployments can be routed through gateway config in the
  shared production config.
- GPU worker lifecycle belongs to `volcano-cluster` or the relevant cluster
  skill, not this server operation skill.
- Volcano production worker nodes are topology/Supervisor owned. Do not use
  historical Volcano CLI commands to list, submit, cancel, or inspect worker
  jobs.
- Use the Volcano SDK operator helper from the production checkout:

```bash
ssh mint-prod-volcano 'cd /vePFS-Mindverse/share/mint/prod/mint-server && /vePFS-Mindverse/share/mint/prod/runtime/host-venv/bin/python scripts/tools/volcano_sdk_jobs.py --region cn-beijing list --name-contains mint-prod-worker- --limit 200'
ssh mint-prod-volcano 'cd /vePFS-Mindverse/share/mint/prod/mint-server && /vePFS-Mindverse/share/mint/prod/runtime/host-venv/bin/python scripts/tools/volcano_sdk_jobs.py --region cn-beijing instances --job-id <job_id>'
```

Credential checks may report only source existence. Never print secret values,
credential files, signed requests, or process environments.

## Hard Rules

- Do not perform development operations from this skill.
- Do not print private config or process environments.
- Do not deploy with file sync tools.
- Use shared production config for server startup.
- Do not run local `ray` or `volc` commands. Use the environment host or the
  appropriate cluster skill.
- Do not restart Ray or worker nodes to fix an API process problem.
