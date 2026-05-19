# VLA Deterministic Startup Runbook

Scope: cold-start the dedicated PR422 VLA server on `mint-dev` from zero in a reproducible way.

## Preconditions

- Dedicated code root: `/vePFS-Mindverse/share/code/root/mint-server-pr422-vla-20260402`
- Dedicated runtime root: `/vePFS-Mindverse/share/code/root/mint-runtime-py31213-openpi-pr422-20260402`
- Assigned Ray worker: `192.168.38.176`
- Ray head: `192.168.38.184`
- Server-side launcher script: [openpi_vla_start_server.sh](/home/yiwen/mint_project/mint-server-pr422-vla-wt/scripts/wip/openpi_vla_start_server.sh)

## Exact Start Command

Run on `mint-dev`:

```bash
cd /vePFS-Mindverse/share/code/root/mint-server-pr422-vla-20260402
nohup bash scripts/wip/openpi_vla_start_server.sh mint_root_vla_pr422_20260404c 18125 /tmp/mint_server_vla_pr422_startdet3.log >/tmp/mint_server_vla_pr422_startdet3.nohup 2>&1 &
```

## Required Invariants

- Source `configs/prod_volcano.env.sh`, then override only the dedicated VLA values.
- Keep `MINT_UVICORN_WORKERS=1`.
- Use a fresh Ray namespace for each cold-start validation.
- Use a namespace-specific scheduler actor name via `MINT_MODEL_WORK_SCHEDULER_ACTOR_NAME`.
- Pin the scheduler actor with `MINT_MODEL_WORK_SCHEDULER_PINNED_NODE_IP=192.168.38.176`.
- Pin the detached control-plane actors with `MINT_CONTROL_PLANE_PINNED_NODE_IP=192.168.38.176`.
- Set `MINT_OPENPI_FAST_WEIGHTS_PATH` to the shared params directory: `/vePFS-Mindverse/share/models/openpi/pi0_fast_base/params`.
- Set `MINT_OPENPI_PI05_WEIGHTS_PATH` to the shared params directory: `/vePFS-Mindverse/share/models/openpi/pi05_base/params`.
- Set `MINT_OPENPI_FAST_ASSETS_BASE_DIR` to a source assets tree that actually contains `assets/**/norm_stats.json`:
  `/vePFS-Mindverse/share/models/openpi/pi0_fast_base_official_20260428/assets`.
- Set `MINT_OPENPI_PI05_ASSETS_BASE_DIR` to the shared pi0.5 source assets tree:
  `/vePFS-Mindverse/share/models/openpi/pi05_base/assets`.
- Optional: set `MINT_VLA_FAST_WEIGHTS_PATH`, `MINT_VLA_PI05_WEIGHTS_PATH`, `MINT_VLA_FAST_ASSETS_PATH`, or `MINT_VLA_PI05_ASSETS_PATH` before running the startup script if you need a different known-good bundle.
- Do not point `MINT_OPENPI_*_ASSETS_BASE_DIR` at a run output directory such as `results/.../assets`; that is not a valid source of OpenPI `norm_stats`.
- Do not rely on inherited shell env. Explicitly unset stray actor-name overrides before starting.

## Root Cause Fixed

Fresh cold-starts were failing serially on detached control-plane actors because:

1. Many detached startup actors were hard-pinned to `node:__internal_head__`.
2. Pinning the first actor only was insufficient.
3. Nested detached actors created by other actors still fell back to head pinning because `MINT_CONTROL_PLANE_PINNED_NODE_IP` was not forwarded through `actor_runtime_env_vars()`.

The deterministic fix is:

- `preferred_control_plane_resources()` selects the explicit worker pin when `MINT_CONTROL_PLANE_PINNED_NODE_IP` is set.
- `actor_runtime_env_vars()` now forwards:
  - `MINT_CONTROL_PLANE_PINNED_NODE_IP`
  - `MINT_MODEL_WORK_SCHEDULER_PINNED_NODE_IP`
  - `MINT_STARTUP_LEASE_PINNED_NODE_IP`

That makes both top-level and nested detached actors inherit the same cold-start placement policy.

## Verification

Expected success signals:

```bash
grep -n 'Application startup complete' /tmp/mint_server_vla_pr422_startdet3.log
curl -s http://localhost:18125/api/v1/healthz
curl -s http://localhost:18125/internal/actors
```

Observed successful cold-start reference:

- namespace: `mint_root_vla_pr422_20260404c`
- port: `18125`
- healthz: `{\"status\":\"ready\"}`
- actor inventory immediately after startup: no model actors yet, `total_gpus_used=0`

## Failure Pattern To Watch

If cold-start regresses, inspect in this order:

1. `mint_config`
2. `mint_model_actor_supervisor`
3. `task_state_store` / `TaskStateStore` session, index, heartbeat, gateway, and future methods
4. `maintenance_cron_actor`
5. `model_work_scheduler`
6. `mint_model_runtime_*`

If a detached actor is stuck in `PENDING_CREATION`, inspect its `required_resources` and its serialized `runtime_env.env_vars` before changing any code.
