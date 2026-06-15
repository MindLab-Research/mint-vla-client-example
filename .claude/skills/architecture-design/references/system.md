# System boundary and code map

## System boundary

mint-server ("MinT") is a FastAPI service that implements a Tinker-compatible REST API and brokers training/inference to Ray GPU actors.

- Clients: `tinker` SDK (ServiceClient/TrainingClient/SamplingClient) and `tinker-cookbook` recipes.
- API server: `mint_server/app.py` (FastAPI). Owns HTTP, auth, request validation, and lightweight per-process clients/caches. It should not own actor reconciliation or durable session/execution state.
- Compute: Ray actors on GPU nodes (training workers, vLLM inference servers). Own model weights and do the heavy compute.
- Storage/transfer:
  - Ray object store for tensor transfer (small/medium payloads).
  - Shared filesystem (PFS) for artifacts that are too large to serialize through Ray.

High-level flow:

client -> HTTP -> mint-server (FastAPI) -> Ray -> GPU actors

## Ray persistent (detached) actors

Some control-plane and GPU actors are created as detached Ray actors in a fixed namespace so they can be reused across API server restarts.

Control-plane detached actors:
- `mint_config`: read-only config snapshot actor used to hydrate Ray actor runtime configuration. It must be started before `mint_model_actor_supervisor`.
- `mint_task_state_store`: durable task state and future polling metadata backed by SQLite.
- `mint_model_work_scheduler`: hot scheduling actor for model work backlog, per-replica subqueues, and leases. It must not use `mint_task_state_store` as scheduling authority or lifecycle owner, but it may depend on it for durable task admission, lease persistence, indexes, and recovery.
- `mint_model_actor_supervisor`: highest control-plane actor. It reconciles long-lived model runtime actors, durable OpenPI runtime actors, CPU control-plane actors it owns, and topology daemon actors.
- `mint_maintenance_cron`: periodic maintenance actor for cron jobs such as cleanup/reapers. It does not call the supervisor and does not own model actor reconciliation.

Implications:
- External operations bootstrap only `mint_config`, then `mint_model_actor_supervisor`, then API workers. The supervisor is responsible for ensuring the remaining CPU control-plane actors and all desired runtime/daemon actors.
- A server restart loses only per-process clients/caches. Detached control-plane actors and runtime actors remain the recovery boundary.
- `TaskStateStore` owns durable task state, active indexes, session/index/heartbeat metadata, and gateway routing metadata. Do not reintroduce separate session metadata-store actors.
- `ModelWorkScheduler` can keep accepting work for desired domains while replicas are pending; work waits until the supervisor registers a healthy replica, bounded by task/request TTL. The durable task/future reaper must expire pending/queued/assigned tasks after 24h, retain terminal result payloads for 24h after `done_at` or `failed_at`, and retain tombstones for 7d.
- `ModelWorkScheduler` and `ModelActorSupervisor` derive rebuildable projections from `ConfigActor`, config, provider/Ray state, and runtime actor health. Supervisor state is operational/live state; user and business metadata belongs in `TaskStateStore`.
- `ModelActorSupervisor` state storage starts with two modes: memory-only and SQLite. The SQLite DB path defaults to `/vePFS-Mindverse/share/mint/<env>/runtime/supervisor_state.sqlite3` and is overrideable by `MINT_SUPERVISOR_STATE_DB_PATH`. It is internal supervisor state for ownership/reconcile continuity, not a dependency from scheduler to `TaskStateStore`. Use a conservative SQLite journal mode on vePFS; do not enable WAL unless explicitly configured and validated.
- Actor reconciliation is owned by `mint_model_actor_supervisor`, not by FastAPI startup and not by `mint_maintenance_cron`.
- External operations own first bootstrap and high availability for `mint_config` and `mint_model_actor_supervisor`. API workers must not ensure or recreate them.
- Changes to actor code require recreating the actor. Detached actors do not hot-reload. Rebuildable control-plane actors that publish `code_identity` (`mint_model_actor_supervisor`, `mint_model_work_scheduler`, and `mint_maintenance_cron`) may be recreated by their owner/ensure path when the identity is stale. API request paths only observe identity mismatch and fail fast; they do not kill or recreate detached actors.
- Request paths should use async control-plane helpers and fail fast when Ray is unavailable instead of calling `init_ray()` from inside a route.
- API health endpoints are split by audience. `/api/v1/healthz` is external business health and only checks cached `mint_model_work_scheduler` + `mint_task_state_store` availability. It never exposes degraded/internal reasons. The external cache TTL is 30s; dirty refresh is single-flight and has a 5s total wait/refresh budget. `/api/v1/internal/healthz` is internal operations health and reads the current supervisor summary plus process-local cron/startup degraded markers without per-request runtime actor fanout. Supervisor summary timestamps older than 60s are degraded.

## Code map (where to look)

- `mint_server/app.py`
  - FastAPI startup/shutdown (lifespan), auth middleware, route registration.
  - Stateless API composition layer. Lifespan checks lightweight clients to detached control-plane actors; it must not ensure actors, own reconciliation loops, or own authoritative actor inventory.

- `mint_server/routes/*`
  - HTTP endpoints. Most heavy work is delegated to backend modules.
  - `sampling.py`: async sampling + backpressure; HTTP paths read detached
    sampling metadata and enqueue through `ModelWorkScheduler`; runtime
    execution may use actor-local `SessionManager` caches.
  - `training.py`: training control plane; HTTP paths read detached
    training metadata and enqueue through `ModelWorkScheduler`; runtime
    execution may use actor-local `TrainingSessionManager` + training engine.
  - `weights.py`: save/load weights and checkpoints; HTTP paths read detached
    training metadata before enqueue and bridge training to inference through
    runtime actor execution.
  - `futures.py`: `request_id` polling.

- `mint_server/backend/execution_context.py`
  - Runtime-actor-local execution context for queued model work. `ModelEngineHost`
    binds manager/engine handles with a contextvar while executing dispatcher
    work items. API workers must leave route module execution globals unbound;
    runtime dispatch must not temporarily write route globals.

- `mint_server/backend/async_ray_control.py`
  - Async wrappers for Ray control-plane operations that would otherwise block the FastAPI event loop.
  - Used for actor lookup/kill and placement-group inspection on request paths.

- `mint_server/models/types.py`
  - Pydantic request/response models intended to match the Tinker API.

- `mint_server/backend/model_actor_inventory.py`
  - GPU actor inventory state machine, inflight counts, metadata cache, and actor observability helpers.
  - The authoritative instance is owned by the detached `ModelActorSupervisor`; API-process caches must not be treated as source of truth.

- `mint_server/backend/model_actor_supervisor.py`
  - Detached desired-state reconciler for model runtime actors, OpenPI runtime actors, and topology daemon actors.
  - Desired specs come from config/topology/provider state; runtime state is rebuildable from Ray and persisted control-plane state.
  - Owns the inventory/launcher publication contract used by backend-specific vLLM, Megatron, dense, and OpenPI launchers.

- `mint_server/backend/model_actor_launchers.py`
  - Launcher registry used by `ModelActorSupervisor` when reconciling desired model runtime actors.
  - Owns runtime actor launch-time placement environment construction.

- `mint_server/backend/model_actor_publication.py`
  - Shared launch-publication helper for backend-created GPU actors.
  - Backend-specific launchers still own backend Ray actor and placement-group creation, but they must publish a `BackendModelActorLaunch` through this helper. Lifecycle registration, ready marking, and observability metadata merge all go through `ModelActorSupervisor`.

- `mint_server/backend/session_manager.py`
  - Sampling session bookkeeping and cleanup.
  - Routes named sessions and ephemeral weight sync through the current shared-engine sampling path.

- `mint_server/backend/multi_lora_engine.py`
  - Multi-tenant inference: one detached vLLM actor per base model, with many LoRA adapters loaded and selected by `lora_int_id`.
  - Selects between single-node vLLM and `MultiNodeInferenceEngine` (Ray distributed vLLM) based on model config.

- `mint_server/backend/verl_inference.py`
  - vLLM-backed inference actor implementation and weight loading.

- `mint_server/backend/verl_training.py`
  - Dense LoRA training backend via Ray actors.

- `mint_server/backend/megatron_training.py`, `mint_server/backend/megatron_distributed.py`
  - MoE LoRA training backend via Megatron (distributed actors + placement groups).

- `mint_server/backend/model_registry.py`
  - Supported model allowlist and per-model parallelism/memory knobs (inference and training are specified separately).

- `mint_server/config.py`
  - Server config + Ray runtime `PYTHONPATH` for actors.
  - Canonical mode assembles `PFS_PYTHONPATH` from `PFS_RUNTIME_ENV_ROOT`.
  - Reads server-owned deployment knobs through the runtime-env helper; `MINT_*` is canonical.

- `mint_server/runtime_env.py`
  - Pure-stdlib runtime-env layout and bootstrap helpers shared by config and startup code.

- `scripts/build_runtime_env.py`
  - Materializes the PFS runtime-env root from `pyproject.toml`.
