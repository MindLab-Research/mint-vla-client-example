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

## Notes

- `--host` runs the same script remotely over SSH using the remote Python path.
- In `--host` mode, `--md-out` / `--json-out` / `--html-out` are treated as **local output paths**.
  The CLI writes to a remote temp file and automatically downloads it back.
- Destructive PG actions are dry-run by default; add `--apply` to execute.
- API key resolution order:
  - `--api-key`
  - local env `TINKER_API_KEY`/`MINT_API_KEY`
  - running `run_server.py` process environment
