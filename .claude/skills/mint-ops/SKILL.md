---
name: mint-ops
description: |
  Deploy and operate the Mint ops console on the production Volcano driver host.

  Use for: build the `ops/` frontend, rsync it to `/vePFS-Mindverse/share/mint-ops`,
  run the backend under supervisor, and expose one URL where the Python backend serves both UI and API.

  Triggers: "mint ops", "deploy ops", "ops console", "sync mint-ops", "start mint-ops"

  Procedure contract: read this SKILL.md end-to-end before acting. Do not slice it on demand or use it as a lookup table mid-run.
---

# Mint Ops

Procedure contract:
- Read this SKILL.md end-to-end before taking any action.
- Do not sample sections opportunistically while already in motion.
- If the procedure is missing something important, update the skill. Do not improvise around the gap.

Use this skill when the task is specifically about the standalone ops console under `ops/`.

## Scope

- Target host: `mint-prod-volcano`
- Remote app root: `/vePFS-Mindverse/share/mint-ops`
- Python env: `/vePFS-Mindverse/share/code/tinker-server-auth/.venv31213/bin/python`
- Mint API: `http://127.0.0.1:18000`
- Ray: deploy on the driver node, so default to `auto`
- Supervisor config: `/mlplatform/supervisord/supervisord.conf`
- Single exposed URL: `http://<internal-ip>:8787/`

## Hard rules

- Use `ssh mint-prod-volcano`, not a guessed host or raw IP, for server-side changes.
- Do not use `rsync --delete`.
- Do not print secrets from `.secrets.env`.
- Production should use one service only: build the frontend with relative `/api` paths and let `ops.backend` serve `ops/frontend/dist/`.
- Keep backend pointed at local Mint (`http://127.0.0.1:18000`) and source the API key from remote `.secrets.env`.

## Deploy SOP

### 1. Build frontend locally

Build with the default relative API path:

```bash
cd /vePFS-Mindverse/user/intern/nolanho/code/mint-feat-mint-ops/ops/frontend
pnpm build
```

### 2. Sync only the `ops` app

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

### 3. Add the supervisor program

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

### 4. Verify

```bash
ssh mint-prod-volcano 'supervisorctl status mint-ops-backend'
ssh mint-prod-volcano 'curl -s http://127.0.0.1:8787/api/health'
curl -I http://<internal-ip>:8787/
```

Use `http://<internal-ip>:8787/` as the single entrypoint. The same port serves both the UI and `/api/*`, so SSH port forwarding to `8787` also works cleanly.
