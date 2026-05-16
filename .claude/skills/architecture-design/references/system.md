# System boundary and code map

## System boundary

tinker-server ("MinT") is a FastAPI service that implements a Tinker-compatible REST API and brokers training/inference to Ray GPU actors.

- Clients: `tinker` SDK (ServiceClient/TrainingClient/SamplingClient) and `tinker-cookbook` recipes.
- API server: `tinker_server/app.py` (FastAPI). Owns HTTP, auth, request validation, and lightweight in-process state.
- Compute: Ray actors on GPU nodes (training workers, vLLM inference servers). Own model weights and do the heavy compute.
- Storage/transfer:
  - Ray object store for tensor transfer (small/medium payloads).
  - Shared filesystem (PFS) for artifacts that are too large to serialize through Ray.

High-level flow:

client -> HTTP -> tinker-server (FastAPI) -> Ray -> GPU actors

## Ray persistent (detached) actors

Some control-plane and GPU actors are created as detached Ray actors in a fixed namespace so they can be reused across API server restarts.

Control-plane detached actors:
- `mint_task_state_store`: durable task state and future polling metadata backed by SQLite.
- `mint_model_work_scheduler`: hot scheduling actor for model work backlog, per-replica subqueues, and leases.
- `mint_maintenance_cron`: periodic maintenance actor for cleanup and reconciliation.
- `tinker_session_index_store`: minimal session and sampler metadata for REST enumeration after API restart.
- `tinker_training_session_store`: minimal training-session metadata used to recover routing to detached trainer actors.
- `tinker_gateway_session_store`: routing metadata for upstream-created sampling sessions and training models when this server acts as a gateway.

Implications:
- A server restart loses in-process mappings (live session registries, engine bindings, LoRA id mappings), but detached actors may still exist and hold GPU memory.
- Some REST-visible metadata survives restart in detached control-plane stores even though the live engine/session objects do not.
- Startup reconciliation is required to rediscover or kill these actors (`tinker_server/app.py:_cleanup_stale_actors`).
- Changes to actor code require recreating the actor. Detached actors do not hot-reload.
- Request paths should use async control-plane helpers and fail fast when Ray is unavailable instead of calling `init_ray()` from inside a route.

## Code map (where to look)

- `tinker_server/app.py`
  - FastAPI startup/shutdown (lifespan), auth middleware, route registration.
  - Startup actor reconciliation: `_cleanup_stale_actors()` kills dead actors and registers alive detached actors with `ModelActorInventory`.
  - Warms detached control-plane actors used on request paths (`TaskStateStore`, metadata stores, scheduler actors).

- `tinker_server/routes/*`
  - HTTP endpoints. Most heavy work is delegated to backend modules.
  - `sampling.py`: async sampling + backpressure; uses `SessionManager`.
  - `training.py`: training control plane; uses `TrainingSessionManager` + training engine.
  - `weights.py`: save/load weights and checkpoints; bridges training to inference.
  - `futures.py`: `request_id` polling.

- `tinker_server/backend/async_ray_control.py`
  - Async wrappers for Ray control-plane operations that would otherwise block the FastAPI event loop.
  - Used for actor lookup/kill and placement-group inspection on request paths.

- `tinker_server/models/types.py`
  - Pydantic request/response models intended to match the Tinker API.

- `tinker_server/backend/model_actor_inventory.py`
  - Process-local GPU actor inventory, inflight counts, metadata cache, and actor observability and inflight helpers.
  - Durable scheduling state does not live here; use `TaskStateStore`, `ModelWorkScheduler`, and `ModelActorSupervisor`.

- `tinker_server/backend/model_actor_supervisor.py`
  - Process-local desired-state reconciler for model runtime actors.
  - Desired specs come from local config/env; the supervisor itself is not a durable store.

- `tinker_server/backend/session_manager.py`
  - Sampling session bookkeeping and cleanup.
  - Supports legacy per-session engines and a shared-engine mode for faster ephemeral weight sync.

- `tinker_server/backend/multi_lora_engine.py`
  - Multi-tenant inference: one detached vLLM actor per base model, with many LoRA adapters loaded and selected by `lora_int_id`.
  - Selects between single-node vLLM and `MultiNodeInferenceEngine` (Ray distributed vLLM) based on model config.

- `tinker_server/backend/verl_inference.py`
  - vLLM-backed inference actor implementation and weight loading.

- `tinker_server/backend/verl_training.py`
  - Dense LoRA training backend via Ray actors.

- `tinker_server/backend/megatron_training.py`, `tinker_server/backend/megatron_distributed.py`
  - MoE LoRA training backend via Megatron (distributed actors + placement groups).

- `tinker_server/backend/model_registry.py`
  - Supported model allowlist and per-model parallelism/memory knobs (inference and training are specified separately).

- `tinker_server/config.py`
  - Server config + Ray runtime `PYTHONPATH` for actors.
  - Canonical mode assembles `PFS_PYTHONPATH` from `PFS_RUNTIME_ENV_ROOT`.
  - Legacy mode assembles it from per-package overlay paths.

- `tinker_server/runtime_env.py`
  - Pure-stdlib runtime-env layout and bootstrap helpers shared by config and startup code.

- `scripts/build_runtime_env.py`
  - Materializes the PFS runtime-env root from `pyproject.toml`.
