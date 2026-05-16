# GPU actor inventory and eviction

Mint now separates desired actor reconciliation from local actor inventory:

- `ModelActorSupervisor` is the desired-state reconciler. It owns node-pin based model-runtime actor planning and talks to `ModelWorkScheduler` about replica availability.
- `ModelActorRegistry` (`tinker_server/backend/model_actor_registry.py`) is a process-local projection of currently known GPU actors. It tracks metadata, inflight counts, session bindings, LRU timestamps, and best-effort eviction helpers. It is not a detached store and is not durable scheduling state.
- Durable task/lease/result state lives in `TaskStateStore`; hot scheduling state lives in `ModelWorkScheduler`.

## What gets evicted

`ModelActorRegistry` tracks GPU-using Ray actors observed by the API process:
- `ActorType.VLLM`: vLLM inference actors (including multi-LoRA vLLM actors)
- `ActorType.DENSE`: dense training actors (TrainingWorker) managed by DenseTrainerPool
- `ActorType.MEGATRON`: MegatronWorkerGroup actors (MoE training)

The registry does not create actors and does not decide desired placement. It only:
- records actor metadata (type, GPU count, session association, node_id)
- updates LRU timestamps on use
- kills idle actors when a legacy direct actor creation path explicitly asks for capacity

## When eviction happens in V1

Eviction is still demand-driven on direct actor creation paths. Callers must explicitly request capacity:
- `ModelActorRegistry.ensure_gpus_available(needed_gpus)` checks Ray GPU availability and triggers eviction if needed.
- Call sites include:
  - vLLM actor creation (`tinker_server/backend/multi_lora_engine.py`)
  - Dense trainer actor creation (`tinker_server/backend/verl_training.py` DenseTrainerPool)
  - Megatron actor creation (`tinker_server/backend/megatron_distributed.py`)

If no code calls `ensure_gpus_available`, the registry does not evict anything. Scheduler-driven runtime placement should prefer deterministic node pins and supervisor reconciliation instead of opportunistic "find any GPU" logic.

## DenseTrainerPool interaction

Dense training has two layers of caching/lifecycle:
- `DenseTrainerPool` reuses a detached `TrainingWorker` per `base_model` and can also kill idle pool entries.
- `ModelActorRegistry` is the local cross-subsystem inventory and best-effort eviction helper.

In practice, `DenseTrainerPool.get_or_create(...)` calls `ModelActorRegistry.ensure_gpus_available(1)` before creating a new actor and registers the actor into `ModelActorRegistry` for LRU accounting.

## Idle and evictable definitions

Each actor has:
- `creating`: True while the actor is initializing. Creating actors are never evicted.
- `protected`: True means the actor is never evicted by `ModelActorRegistry` LRU (used for "persistent" actors).
- `last_accessed`: updated via `touch()`/`get()` to implement LRU.
- `current_session`: optional marker for training actors.

`ActorEntry.is_idle(session_idle_timeout)`:
- vLLM: idle if `idle_time() > session_idle_timeout`
- training actors: idle if `current_session is None` OR `idle_time() > session_idle_timeout`

Evictability is stricter than idleness:
- An actor is evictable if it is idle AND `idle_time() > MIN_ACTOR_AGE`.
- Env vars:
  - `MINT_SESSION_IDLE_TIMEOUT` (default 300s)
  - `MINT_MIN_ACTOR_AGE` (default 300s)

LRU ordering:
- `ModelActorRegistry._get_evictable_actors_lru()` sorts evictable actors by `last_accessed` ascending (oldest first).

## Pending reservations (concurrency safety)

To avoid races where multiple concurrent requests all observe the same "available GPUs" and over-allocate:
- `ModelActorRegistry.reserve_gpus(n)` increments `_pending_gpus`.
- `ModelActorRegistry.get_effective_available_gpus()` returns `ray.available_resources()["GPU"] - _pending_gpus`.
- `ModelActorRegistry.release_pending_gpus(n)` decrements `_pending_gpus` after the actor is created (or the create failed).

If a caller ignores pending reservations and uses raw Ray availability, the pool cannot prevent oversubscription.

## Killing an actor

`ModelActorRegistry._kill_actor(entry)`:
- gets actor handle (cached handle or `ray.get_actor(name, namespace)`)
- tries `actor.shutdown.remote()` if present (best-effort)
- then `ray_kill.kill(...)` and removes it from `_entries`

This frees GPUs at Ray scheduling level. It does not guarantee CUDA memory defragmentation on the node.

`ray_kill.kill(...)`:
- logs structured context (reason, actor_name, namespace, GPU footprint, etc.)
- optionally logs a call stack when `MINT_LOG_KILL_STACK=1`
- best-effort removes a detached placement group named `{actor_name}_pg` (a common leak source)

## Clearing stale session pins

Session deletion can race with in-flight actor creation. Two helpers exist to avoid permanently pinning an actor as "non-idle" due to stale `current_session` values:
- `ModelActorRegistry.clear_session(session_id)` clears `current_session` fields that still point at a deleted session.
- Dense training has a parallel mechanism: `DenseTrainerPool.clear_session(session_id)` (see `tinker_server/backend/verl_training.py`).

## Startup reconciliation

Many actors are created as detached so they can survive an API server restart.

Problem: after restart, server memory is empty, but actors still exist.

Startup hook `tinker_server/app.py:_cleanup_stale_actors()`:
- lists named actors in `tinker_server.config.RAY_NAMESPACE` (from `TINKER_RAY_NAMESPACE`)
- health-checks them (`__ray_ready__`)
- registers alive actors into `ModelActorRegistry` and marks them ready
- kills dead/unresponsive ones

This rebuilds the process-local `ModelActorRegistry` projection from live Ray named actors.

## Persistent actors (prewarm + eviction protection)

At API server startup, Mint can optionally pre-create long-lived training/inference actors and mark them as `protected` in `ModelActorRegistry` so direct LRU eviction paths do not kill them.

Implementation: `tinker_server/app.py:_prewarm_persistent_models(...)`.

Controls:
- `MINT_PERSISTENT_MODELS`: comma-separated HF model names (enables prewarm)
- `MINT_PERSISTENT_TRAIN_LORA_RANK` (default 16)
- `MINT_PERSISTENT_TRAIN_LR` (default 5e-5)
- `MINT_PERSISTENT_MEGATRON_READY_TIMEOUT_S` (default 3600)
- `MINT_PERSISTENT_INFER_TIMEOUT_S` (default 1800)

## Implications for architecture changes

- If you introduce a new GPU-using actor type, first decide whether it should be reconciled by `ModelActorSupervisor` and claimed through `ModelWorkScheduler`.
- Register live actors in `ModelActorRegistry` so local observability and eviction safeguards stay correct. Decide how it should set:
  - `creating` (protect during init)
  - `protected` (if you need a never-evict policy)
  - `current_session` (if training-like)
  - `node_id` (if placement decisions need it)
- For direct, non-scheduler actor creation paths, ensure the code path calls `ensure_gpus_available` before `ray.remote(...).remote(...)`.
- If the actor is detached, add it to startup reconciliation in `tinker_server/app.py`.
