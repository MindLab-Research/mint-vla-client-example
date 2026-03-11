# `scripts/ops` Unified Ops CLI

Single entrypoint for Mint/tinker-server operations.

## Script

- `scripts/ops/mint_ops.py`

## Scope

- Server lifecycle via supervisord
- Actor probe / kill / rebuild
- Placement-group topology and cleanup
- Integrated status report (GPU topology + actor status + machine probe)
- Verification checks (healthz/capabilities/actors + optional sampling smoke)

## Quick Start

```bash
# 1) Cluster status (markdown + json)
python scripts/ops/mint_ops.py --host mint-prod-volcano status \
  --md-out /tmp/mint_ops_status.md \
  --json-out /tmp/mint_ops_status.json

# 1b) HTML status report
python scripts/ops/mint_ops.py --host mint-prod-volcano status \
  --html-out /tmp/mint_ops_status.html

# 1c) Live serve mode (with Refresh button in HTML)
python scripts/ops/mint_ops.py --host mint-prod-volcano status \
  --html-out /tmp/mint_ops_status.html \
  --serve --serve-port 8765

# 1d) Live serve mode (no html file written; in-memory report + API)
python scripts/ops/mint_ops.py status --serve --serve-port 8765

# 1e) Direct remote serve via SSH tunnel (no remote /tmp html + no scp)
python scripts/ops/mint_ops.py --host mint-prod-volcano status \
  --serve --direct --serve-port 8765
# open http://127.0.0.1:8765/status.html
# default auto-cleans stale mint_ops on same port before bind (disable with --no-kill-stale-ops)

# 1f) Dedicated ops HTTP server (status html/json/md + refresh endpoint)
python scripts/ops/mint_ops.py ops-server --server-port 8765
# curl http://127.0.0.1:8765/api/v1/status?format=json
# curl http://127.0.0.1:8765/api/v1/status?format=md
# curl http://127.0.0.1:8765/api/v1/status?format=html
# open http://127.0.0.1:8765/status.html

# 2) Restart server only (optionally clean dirty non-supervisor run_server pids)
python scripts/ops/mint_ops.py --host mint-prod-volcano server-restart --clean-dirty

# 3) Actor operations
python scripts/ops/mint_ops.py --host mint-prod-volcano actor-list
python scripts/ops/mint_ops.py --host mint-prod-volcano actor-kill --actor-type vllm --model-name Qwen/Qwen3-30B-A3B-Instruct-2507
python scripts/ops/mint_ops.py --host mint-prod-volcano actor-rebuild --kind vllm --model Qwen/Qwen3-30B-A3B-Instruct-2507 --sample-ping

# 4) Placement groups
python scripts/ops/mint_ops.py --host mint-prod-volcano pg-list
python scripts/ops/mint_ops.py --host mint-prod-volcano pg-remove --state PENDING --only-gpu      # dry-run
python scripts/ops/mint_ops.py --host mint-prod-volcano pg-remove --state PENDING --only-gpu --apply

# 5) Verify
python scripts/ops/mint_ops.py --host mint-prod-volcano verify
python scripts/ops/mint_ops.py --host mint-prod-volcano verify --sampling-model Qwen/Qwen3-4B-Instruct-2507
```

## UI Tabs

`/status.html` now provides a deployment console with left-side tab navigation:

1. `Dashboard`
   - Node count / per-node GPU capacity / node state / estimated GPU reservation
   - Placement group distribution across nodes
   - Actor distribution and status (including `creating/pending`)
   - `Actor Readiness (Not Ready)`: shows expected persistent actors (from `MINT_PERSISTENT_MODELS` + prewarm flags) that are `creating` / `missing`
   - Gateway-routed models (`TINKER_GATEWAY_CONFIG_JSON.model_to_upstream`) are excluded from local readiness expectation
2. `Deploy`
   - `2.1 Node Scale`: TODO placeholder
   - `2.2 Placement Group`: preview/remove PG + restart server to flush config
   - `2.3 Actor`: kill by actor type/model and quick-kill from actor table
   - `Rebuild Actor` model selector is a dropdown from repo model registry + current managed actors (no manual free-text needed)
3. `Cronjob`
   - TODO placeholder

## HTTP APIs (for curl / integration)

```bash
# status
curl http://127.0.0.1:8765/api/v1/status?format=json
curl http://127.0.0.1:8765/api/v1/status?format=md
curl http://127.0.0.1:8765/api/v1/status?format=html
curl -X POST http://127.0.0.1:8765/api/v1/status/refresh

# deploy actions
curl -X POST http://127.0.0.1:8765/api/v1/deploy/actor/kill \
  -H 'Content-Type: application/json' \
  -d '{"actor_type":"vllm","model_name":"Qwen/Qwen3-30B-A3B-Instruct-2507"}'

curl -X POST http://127.0.0.1:8765/api/v1/deploy/actor/rebuild \
  -H 'Content-Type: application/json' \
  -d '{"kind":"training","model":"Qwen/Qwen3-235B-A22B-Instruct-2507","lora_rank":16,"poll_timeout_s":1800}'

curl -X POST http://127.0.0.1:8765/api/v1/deploy/pg/remove \
  -H 'Content-Type: application/json' \
  -d '{"state":"PENDING","only_gpu":true,"apply":false}'

curl -X POST http://127.0.0.1:8765/api/v1/deploy/server/restart \
  -H 'Content-Type: application/json' \
  -d '{"clean_dirty":false,"wait_healthz_s":60}'
```

## Notes

- `--host` runs the same script remotely over SSH using the remote Python path.
- In `--host` mode, `--md-out` / `--json-out` / `--html-out` are treated as **local output paths**.
  The CLI writes to a remote temp file and automatically downloads it back.
- `status --serve` without `--html-out` serves status from memory and does not write html files.
- `status --serve --direct --host <host>` creates an SSH local-forward and directly serves remote status.
  This avoids remote temp html files and avoids `scp` latency.
- To avoid `Ctrl+C` leftovers causing `Address already in use`, serve modes default to
  `--kill-stale-ops` (can disable with `--no-kill-stale-ops`).
- Destructive PG actions are dry-run by default; add `--apply` to execute.
- API key resolution order:
  - `--api-key`
  - local env `TINKER_API_KEY`/`MINT_API_KEY`
  - running `run_server.py` process environment
