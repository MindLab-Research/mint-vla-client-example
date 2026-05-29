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

Production server code is managed as a git checkout under the shared vePFS
deployment directory. The local development machine and the production driver
see the same `/vePFS-Mindverse/share/mint/prod/mint-server` path, so update the
checkout locally first. This avoids relying on GitHub credentials on the driver.
Do not deploy with file sync tools.

```bash
cd /vePFS-Mindverse/share/mint/prod/mint-server
git fetch origin
git status --short --branch
git checkout develop
git pull --ff-only origin develop
```

If `develop` is already checked out in another local worktree, do not touch
that other worktree. Keep the production checkout detached and point it at the
intended remote commit:

```bash
cd /vePFS-Mindverse/share/mint/prod/mint-server
git fetch origin develop
git checkout --detach origin/develop
git rev-parse HEAD
```

If the local shared path is unavailable or the local git operation fails for an
environmental reason, fall back to running the same git commands on the driver:

```bash
ssh mint-prod-volcano 'cd /vePFS-Mindverse/share/mint/prod/mint-server && git fetch origin && git status --short --branch'
ssh mint-prod-volcano 'cd /vePFS-Mindverse/share/mint/prod/mint-server && git checkout --detach origin/develop && git rev-parse HEAD'
```

Record the commit SHA before and after a production deploy. Prefer local checks
on the shared checkout, and use the driver only to confirm what the running host
will see:

```bash
cd /vePFS-Mindverse/share/mint/prod/mint-server && git rev-parse HEAD
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

### Control-Plane Actor Refresh After Code Updates

Production startup bootstraps detached CPU/control-plane actors before the API
listener starts. After code or config changes, an old detached actor can block
startup with a fingerprint or code-identity mismatch. Typical messages:

- `ConfigActorSnapshotMismatchError: Existing ConfigActor snapshot fingerprint mismatch`
- `[model_actor_supervisor] killing detached actor reason=model_actor_supervisor_code_mismatch`

Do not restart Ray, do not restart worker nodes, and do not kill GPU training or
inference actors for this class of failure. First inspect the supervisor log:

```bash
ssh mint-prod-volcano 'supervisorctl status mint-server-auth; tail -n 120 /tmp/mint_server_auth_supervisor.log'
```

If the only blocker is `mint_config` snapshot mismatch, rebuild just that
namespace-local ConfigActor:

```bash
ssh mint-prod-volcano 'cd /vePFS-Mindverse/share/mint/prod/mint-server && set -a && . /vePFS-Mindverse/share/mint/prod/config/prod.env && set +a && /vePFS-Mindverse/share/mint/prod/runtime/host-venv/bin/python - <<'"'"'PY'"'"'
import os
import ray

namespace = os.environ.get("RAY_NAMESPACE") or os.environ.get("MINT_RAY_NAMESPACE") or "mint"
address = os.environ.get("RAY_ADDRESS") or "auto"
ray.init(address=address, namespace=namespace, ignore_reinit_error=True, log_to_driver=False)
actor = ray.get_actor("mint_config", namespace=namespace)
ray.kill(actor, no_restart=True)
print(f"killed mint_config namespace={namespace}")
ray.shutdown()
PY'
```

Then start `mint-server-auth` again. If the bootstrap log reports
`model_actor_supervisor_code_mismatch`, the startup path may rebuild the
detached `mint_model_actor_supervisor` control-plane actor. That is expected for
code identity changes and is still not a GPU node restart. Validate the runtime
source of truth with `/internal/model_actor_supervisor`; `/internal/actors` is a
backend publication inventory and can be empty immediately after a supervisor
control-plane rebuild.

## Health And Logs

```bash
curl http://localhost:18000/api/v1/healthz
ssh mint-prod-volcano 'tail -n 200 /vePFS-Mindverse/share/mint/prod/logs/mint_server_auth.log'
ssh mint-prod-volcano 'ps aux | grep "[s]cripts/run_server.py"'
```

Production `/internal/*` calls require platform-forwarded identity headers,
not only `X-API-Key`. A plain `X-API-Key` call can return
`401 {"error":"Missing platform auth headers"}` even when the key is valid.
Source secrets without printing them, send `X-Internal-Token`, and include a
synthetic admin identity for operator-only localhost checks:

```bash
ssh mint-prod-volcano 'set -a && . /vePFS-Mindverse/share/mint/prod/config/secrets.env && set +a && /usr/bin/curl -s \
  -H "X-Internal-Token: ${MINT_INTERNAL_API_TOKEN:-}" \
  -H "X-MinT-User-Id: 000000000000000000000001" \
  -H "X-MinT-User-Role: admin" \
  -H "X-MinT-Account-Id: 000000000000000000000001" \
  -H "X-MinT-Apikey-Id: 000000000000000000000002" \
  -H "X-MinT-Request-Id: operator-check" \
  http://localhost:18000/internal/model_actor_supervisor'
```

If `prod.env` was sourced in the current shell and simple commands like `curl`
or `head` unexpectedly disappear, use absolute tool paths such as `/usr/bin/curl`
or run a fresh shell. Do not print the token value while debugging auth.

## Internal Ops

For `/internal/*` actor inventory, actor kill, scheduler diagnostics, deep
health, and Ray diagnostics, read and use the `mint-ops` skill after this one.
Public `/api/v1/healthz` is only a cheap API-worker health check.

## Production Topology Notes

- Volcano production is the router/master deployment for the local production
  models.
- Volcano production hosts Qwen3 0.6B, Qwen3 4B Instruct, Qwen3 4B Thinking,
  Qwen3 30B A3B Instruct, Qwen3 235B A22B Instruct, and OpenPI runtime models.
- 235B is deployed locally on Volcano with 16 GPUs for vLLM and 16 GPUs for
  Megatron. This requires the production model override to set the 235B
  Megatron world size to 16 GPUs.
- Topology aliases are the stable placement contract:
  - `mint-worker-0`: 30B vLLM plus 30B Megatron.
  - `mint-worker-1`: 0.6B/4B vLLM, dense training, and OpenPI runtimes.
  - `mint-worker-2` and `mint-worker-3`: 235B vLLM, 8 GPUs each.
  - `mint-worker-4` and `mint-worker-5`: 235B Megatron, 8 GPUs each.
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
