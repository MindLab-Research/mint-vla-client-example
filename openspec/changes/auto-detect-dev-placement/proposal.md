## Why

Mint dev server startup still requires developers to manually regenerate worker
placement JSON whenever the Ray cluster is recreated or worker IPs change. This
is brittle and wastes operator time because the dev launcher already knows the
current head-address file and can inspect the live Ray dashboard.

## What Changes

- Add a launcher-owned automatic placement env generation path for Mint dev.
- Keep explicit placement env (`MINT_DEV_RUN_ENV` or `MINT_*_PLACEMENT_JSON`)
  as the highest-precedence override.
- Generate placement from the current Ray head/dashboard and alive GPU workers
  when no explicit placement is supplied.
- Emit clear launch logs for the detected head, selected workers, and generated
  env file.
- Fail fast when no alive GPU workers can be discovered.

## Capabilities

### New Capabilities

- `dev-auto-placement`: Mint dev startup can auto-detect current Ray GPU
  workers and provide placement env without a manual per-cluster edit.

### Modified Capabilities

- None.

## Impact

- Affected scripts: `scripts/start_dev_server.sh`,
  `scripts/tools/gen_dev_placement.py`.
- Affected docs: `.claude/skills/mint-dev/SKILL.md`.
- Affected tests: startup config tests and new placement-generation unit tests.
- No production startup behavior changes.
