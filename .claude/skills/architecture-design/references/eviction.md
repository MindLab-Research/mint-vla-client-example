# GPU actor inventory

Mint separates desired actor reconciliation from local actor inventory:

- `ModelActorSupervisor` is the desired-state reconciler. It owns node-pin based model-runtime actor planning and talks to `ModelWorkScheduler` about replica availability.
- `ModelActorInventory` (`mint_server/backend/actors/model_actor_inventory.py`) is an internal helper owned by `ModelActorSupervisor`. It tracks local metadata, inflight counts, session bindings, protection flags, RSS samples, and actor handles for admin list/kill and observability. Legacy direct GPU launch paths must publish lifecycle through the explicit `ModelActorSupervisor` inventory contract (`register`, `mark_ready`, `unregister`, inflight/session/protection methods), not by reaching around the supervisor.
- Durable task/lease/result state lives in `TaskStateStore`; hot scheduling state lives in `ModelWorkScheduler`.

## No LRU capacity manager

The old demand-driven GPU capacity path is removed. Direct actor creation no longer calls local `ensure_gpus_available`, pending GPU reservations, or opportunistic LRU eviction.

GPU placement is now expected to be deterministic:

- scheduler-owned work should flow through `ModelWorkScheduler` and runtime subqueues;
- model runtime actors should be reconciled from `ModelActorSupervisor` desired state and node pins;
- direct creation paths should rely on explicit placement or Ray placement failure, not a process-local best-effort eviction loop.

## Inventory responsibilities

`ModelActorSupervisor` inventory tracks GPU-using Ray actors observed by the API process:

- `ActorType.VLLM`: vLLM inference actors, including multi-LoRA vLLM actors
- `ActorType.DENSE`: dense training actors
- `ActorType.OPENPI`: OpenPI action/training actors
- `ActorType.MEGATRON`: MegatronWorkerGroup actors

The inventory:

- records actor metadata such as type, GPU count, session association, node id, and backend observability fields;
- updates access timestamps on use;
- tracks inflight counts so admin/metrics can distinguish busy actors;
- keeps optional protection/session fields used by lifecycle code;
- powers `/internal/actors`, `/internal/actors/kill`, `/internal/admission_stats`, and OTel-pushed actor inventory metrics.

It does not own desired placement and is not durable scheduling state.

## Idle state

Each actor has:

- `creating`: true while the actor is initializing.
- `protected`: true for actors that admin tooling should not casually remove.
- `last_accessed`: updated via `touch()`/`get()`.
- `current_session`: optional marker for training-like actors.

`ActorEntry.is_idle(session_idle_timeout)` is observability-only:

- vLLM: idle if `idle_time() > session_idle_timeout`
- training actors: idle if `current_session is None` or `idle_time() > session_idle_timeout`

The only config knob left for this surface is `[model_actor_inventory].session_idle_timeout_s`, mirrored by `MINT_MODEL_ACTOR_INVENTORY_SESSION_IDLE_TIMEOUT_S`.

## Admin kill

Admin kill remains explicit. `ModelActorSupervisor.clear(kill_actors=True)` and actor-family kill routes:

- look up actor handles by name and namespace;
- try `shutdown.remote()` when present;
- call `ray_kill.kill(...)`;
- unregister supervisor-owned inventory state.

This is an operator action, not automatic admission control.

## Startup reconciliation

Many actors are detached and can survive API server restart. Startup reconciliation:

- lists named actors in `mint_server.config.RAY_NAMESPACE`;
- health-checks them via `__ray_ready__`;
- registers alive actors into `ModelActorSupervisor` inventory and marks them ready;
- unregisters or kills dead/unresponsive actors.

This rebuilds the process-local inventory projection from live Ray named actors.

## Architecture guidance

- New scheduler-owned GPU runtime actors should be modeled as `ModelActorSupervisor` desired state plus `ModelWorkScheduler` replica registrations.
- New direct actor paths should register live actors in `ModelActorSupervisor` inventory for admin and metrics, but must not add local capacity reservation or LRU eviction.
- Detached GPU actors should participate in startup reconciliation.
