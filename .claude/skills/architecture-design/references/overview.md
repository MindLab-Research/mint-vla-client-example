# Mint architecture overview

mint-server is a FastAPI service that implements the Mint REST contract and brokers training and inference to Ray GPU actors. The server is a control plane plus request validation, not a compute engine. The design is shaped by two goals: keep the SDK-compatible HTTP contract stable, and multiplex GPU resources across many LoRA sessions for both inference and training.

Dependency management follows the same control-plane versus compute-engine split:
- the worker image owns ABI-bound GPU packages
- a PFS runtime-env root owns the shared Python dependency graph and pinned source overlays for the API host and Ray actors

## Mint API contract as the boundary

The API surface follows the Mint SDK expectations while preserving the externally compatible HTTP contract: create sessions and models, submit work, poll for completion, and manage weights. The server owns HTTP, auth, and request validation. The GPU actors own model weights and do the compute. That split is the primary boundary.

Contract implications:
- Long-running work returns a `request_id` and is polled via `/api/v1/retrieve_future` (408 pending). This is the Mint async future protocol, not a server-specific choice.
- `session_id`, `model_id`, and `sampling_session_id` are control-plane identifiers. Some live routing and engine bindings are in process, but minimal recovery metadata is persisted through `TaskStateStore`.
- The server can restart while detached Ray actors remain alive. Fast in-process registries are still lost, but detached control-plane actors keep enough metadata for reconciliation and selected REST reads.

## Multi-LoRA inference: one base model, many adapters

Inference uses a MultiLoRA design: one detached vLLM actor per base model, many LoRA adapters loaded into that actor, and requests select the adapter by `lora_int_id`.

Flow:
- `POST /api/v1/create_sampling_session` validates access and selects a base model.
- Request handling records durable session/task metadata and submits work through
  the scheduler. Existing healthy vLLM actors can serve immediately; missing
  desired actors are created by `ModelActorSupervisor`, not by the API worker.
- Adapter weights are loaded on demand, then requests call `generate_with_lora` or `generate_base`.

Boundaries and tradeoffs:
- The mapping `sampling_session_id` to `lora_int_id` is in server memory. After a server restart, the vLLM actor may still have LoRAs loaded, but the server no longer knows which session maps to which adapter without additional reconciliation.
- Small and medium adapters go through the Ray object store. Very large MoE adapters use path-based loading on a shared filesystem to avoid serializing thousands of tensors.
- Detached actors reduce warmup cost for repeated use but keep holding GPU memory until evicted.
- Inference engine selection is per-model. Models that require Ray-distributed vLLM execution use `MultiNodeInferenceEngine` and add a CPU-only controller actor (no extra GPU reservation) (see `inference.md` and `placement-groups.md`).

## Async request-path control plane

Hot HTTP paths must not block the API event loop on synchronous Ray control-plane calls.

- Request routes use async APIs on detached control-plane actors through the API control-plane client. `TaskStateStore` owns futures and business indexes, `ModelWorkScheduler` owns hot scheduling, and `ModelActorSupervisor` owns live actor/node reconciliation. `TaskFutureService` is the in-process compatibility facade over `TaskStateStore`.
- API startup may establish Ray connectivity and check required detached
  control-plane actors. It must not ensure, create, or reconcile detached
  actors, and handle warming is not a bootstrap responsibility.
- Request paths fail fast when Ray or required detached control-plane actors are unavailable; they must not call `init_ray()` or silently reconnect from inside a route.
- Cached handles can be reacquired by name through the control-plane client if they die, but request paths still do not create new Ray clients or hide hard Ray outages.

## Multi-tenant training: time-sliced state swap

Training supports many concurrent sessions on fewer GPU trainers by swapping per-session state into the active trainer, while keeping session isolation of:
- LoRA weights
- accumulated gradients (for gradient accumulation across calls)
- optimizer state (momentum/variance)

Two distinct backends implement this:

1. Dense models: pooled detached TrainingWorker per base model, many sessions per actor
- `DenseTrainerPool` reuses a detached `TrainingWorker` keyed by `base_model` and configured with a `max_lora_rank`.
- Each call passes `session_id`, and the actor swaps LoRA weights, optimizer state, and gradients via disk-backed state on the trainer node.

2. MoE models: one MegatronWorkerGroup, many sessions
- A worker group owns a placement group and N rank workers (see `placement-groups.md`).
- On session switch, each rank swaps optimizer and gradient state in memory, while LoRA weights are saved and loaded from a shared filesystem path.

Boundaries and tradeoffs:
- The swap mechanism isolates sessions for time-slicing, but it is not a durable resume system. If an actor dies, in-memory optimizer and gradient state is lost.
- Dense swap state is stored on the trainer node (default `/tmp`). If the actor moves to another node, that state does not follow.
- MoE LoRA weights persist on shared storage, but optimizer state does not. After restart, a session resumes with a fresh optimizer unless restored externally.

## Weight formats and transfer constraints

Inference consumes LoRA adapters in PEFT format:
- `adapter_model.safetensors`
- `adapter_config.json`

This is a hard constraint. vLLM multi-LoRA expects adapter matrices separate from base weights. Any export path that merges LoRA into the base model is unusable for multi-LoRA inference.

Two transfer mechanisms exist because of size and serialization limits:
- Ray object store for smaller adapters.
- Path-based loading on PFS for large or highly sharded adapters (MoE).

MoE training uses Megatron and must export PEFT adapters by reconstructing full tensors across TP and EP sharding. The preferred path is a newer Megatron-Bridge adapter export API that returns adapter weights without merging.

## ModelActorSupervisor, ModelWorkScheduler, and inventory

`ConfigActor` must already exist before `ModelActorSupervisor` starts. External operations bootstrap only `mint_config`, then `mint_model_actor_supervisor`, then API workers. `ModelActorSupervisor` starts its own periodic reconcile loop and is the only component allowed to reconcile long-lived model runtime actors, control-plane actors that it owns, and topology daemon actors. It ensures the remaining CPU control-plane actors, including `ModelWorkScheduler`, `TaskStateStore`, and `MaintenanceCronActor`, plus all desired GPU/CPU runtime actors. Backend-specific vLLM, Megatron, dense, and OpenPI launchers still create their backend Ray actors, but they publish those actors through the supervisor launch-publication contract. Daemon actors are reconciled separately from model replicas and are never synced to the scheduler.

`MaintenanceCronActor` remains a detached cron runner only. It may run periodic cleanup/reaper jobs, but it must not call `ModelActorSupervisor`, own model actor reconciliation state, trigger reconcile, or make reconcile decisions. `ModelActorSupervisor` may manage the cron actor lifecycle; that dependency is one-way.

`ModelWorkScheduler` owns hot task scheduling, replica subqueues, and leases. It must not use `TaskStateStore` as scheduling authority or lifecycle owner; `TaskStateStore` remains the durable task/result/index/session source and may be required for durable task admission and lease persistence. Scheduler and Supervisor keep rebuildable projections and must recover from `ConfigActor`, live Ray actor state, topology config, and provider state rather than from FastAPI process memory. `ModelActorInventory` is owned by the detached supervisor and is not a per-API-process authority. Supervisor live/operational metadata belongs to Supervisor; user and business metadata belongs to `TaskStateStore`.

Detached rebuildable control-plane actors publish a `code_identity` in their health or stats snapshots. `ModelActorSupervisor`, `ModelWorkScheduler`, and `MaintenanceCronActor` must be recreated when a control-plane ensure path observes a stale identity. For `ModelWorkScheduler`, that recreate permission belongs to the supervisor-owned dependency ensure path (`stats(create_if_missing=True)`). API request paths and external health checks only validate the scheduler identity and fail fast on mismatch; they must not kill or recreate scheduler actors from inside normal traffic.

The FastAPI application is a stateless API boundary. It should assemble HTTP middleware and routes, connect to detached control-plane clients, and expose unhealthy status when the highest control plane is unavailable. It may read/check detached actors and submit work to `ModelWorkScheduler`, but it must not ensure, create, or reconcile control-plane actors. If a request observes that a desired model runtime is not ready, it may send a fire-and-forget supervisor nudge that asks the supervisor to run a fast ensure pass. The request must not wait for supervisor ensure/reconcile before enqueuing or returning. If the supervisor is already reconciling, the nudge should be acknowledged without starting a second reconcile.

`ModelWorkScheduler` accepts work even when a desired replica is not yet registered. That work remains pending until the supervisor registers a healthy replica, subject to the request/task TTL. This preserves the Mint async future contract: submit returns a `request_id`, `retrieve_future` may hold the HTTP request open for a bounded wait and then returns HTTP 408 while still pending, and the request is only terminal when it succeeds, fails, expires, is cancelled, or is forgotten after retention. This lets existing healthy actors continue serving when the supervisor is temporarily unavailable, while new or missing actors will not be recreated until the supervisor recovers.

Retention policy: local `retrieve_future` pending long-poll waits up to 20s, retrieve hot-cache entries live for 300s, retrieve replay grace is 600s, pending/queued/assigned tasks expire after 24h, terminal result payloads are retained for 24h after `done_at` or `failed_at`, and tombstones are retained for 7d. `TaskFutureService.async_reap()` enforces this policy through `TaskStateStore` and the in-process `TaskPayloadStore` filesystem helper.

Clients do not explicitly end all sessions, so idle timeouts still affect training and inference:
- Detached inference actors can remain alive across server restarts and keep CUDA memory until evicted.
- Training actors can be evicted if idle, which can discard in-memory session state.

Eviction is a resource policy. It is not a fault-tolerance mechanism.

Mint can optionally pre-create and protect ("never evict") a set of persistent actors through the detached supervisor. This is a capacity planning knob, not a correctness requirement.

## API deployment mode

The target API process is stateless and runs as an external FastAPI/Uvicorn service that connects to Ray through a control-plane client. It is not deployed through Ray Serve.

API workers may access Ray for read/check and scheduling operations through the client, but they must not ensure or recreate detached control-plane actors. `mint_config` and `mint_model_actor_supervisor` are bootstrapped and kept highly available by external operations.

Health policy:
- `/api/v1/healthz` is the external business health endpoint. It is lightweight and answers only whether normal traffic can be served. It checks `ModelWorkScheduler` and `TaskStateStore` through a per-API-worker cache with a 30s TTL. A dirty cache is refreshed synchronously with single-flight behavior; concurrent callers wait up to 5s total. The underlying Scheduler and TaskStateStore pings each get a shorter timeout, leaving room inside that 5s budget. If refresh fails, the dirty previous value is ignored and the response is HTTP 503 with `{"status": "unhealthy"}`. A ready response is HTTP 200 with `{"status": "ready"}`. The response must not expose degraded state, actor names, supervisor availability, cron status, Ray cluster details, or internal reasons. The initial no-cache request follows the same 5s limit and returns 503 on failure.
- `/api/v1/internal/healthz` is the internal operations health endpoint. It is still lightweight: it reads the current `ModelActorSupervisor` summary snapshot plus process-local maintenance-cron/startup degraded markers, rather than fanning out to every runtime actor on each request. It reports `ready`, `degraded`, or `unhealthy` with supervisor reachability, supervisor summary counters, maintenance cron degraded state, and startup control-plane check state. `ModelActorSupervisor` unavailable is unhealthy for the internal endpoint. A supervisor snapshot whose `snapshot_generated_at`, `observed_at`, or topology `observed_at` timestamp is older than 60s is degraded; do not use `last_reconcile_at` for snapshot staleness. `MaintenanceCronActor` unavailable is degraded unless it also prevents required control-plane state from being read.
- Component endpoints such as scheduler stats, supervisor snapshot, cron health, Ray cluster health, and admission stats are debug/diagnostic endpoints, not external health probes.

## Auth and access boundaries

Auth is enforced by middleware when `MINT_INTERNAL_API_TOKEN` is configured. In
production the platform authenticates callers and forwards trusted `X-MinT-*`
identity headers plus `X-Internal-Token`.

Model access control is centralized and applied on session creation. The server hides detailed exception text unless the request is privileged.

## Non-goals

Mint does not aim to:
- reconstruct in-process sampling adapter bindings across server restarts without explicit reconciliation
- migrate training state across GPU nodes automatically
- store full training state in the FastAPI process
- support LoRA formats that merge adapter weights into base weights
