# Auto eviction and GPU allocation (ResourcePool)

The auto eviction mechanism is centralized in `tinker_server/backend/resource_pool.py` and is used by both inference and training code paths to make progress when the cluster has no free GPUs.

## What gets evicted

`ResourcePool` tracks GPU-using Ray actors:
- `ActorType.VLLM`: vLLM inference actors (including multi-LoRA vLLM actors)
- `ActorType.DENSE`: dense training actors (TrainingWorker) managed by DenseTrainerPool
- `ActorType.MEGATRON`: MegatronWorkerGroup actors (MoE training)

The pool does not create actors. It only:
- records actor metadata (type, GPU count, session association, node_id)
- updates LRU timestamps on use
- kills idle actors when code asks for GPUs

## When eviction happens

Eviction is demand-driven. Callers must explicitly request capacity:
- `ResourcePool.ensure_gpus_available(needed_gpus)` checks Ray GPU availability and triggers eviction if needed.
- Call sites include:
  - vLLM actor creation (`tinker_server/backend/multi_lora_engine.py`)
  - Dense trainer actor creation (`tinker_server/backend/verl_training.py` DenseTrainerPool)
  - Megatron actor creation (`tinker_server/backend/megatron_distributed.py`)

If no code calls `ensure_gpus_available`, nothing gets evicted.

## DenseTrainerPool interaction

Dense training has two layers of caching/lifecycle:
- `DenseTrainerPool` reuses a detached `TrainingWorker` per `base_model` and can also kill idle pool entries.
- `ResourcePool` is the cross-subsystem eviction mechanism and is used by both training and inference to reclaim GPUs under pressure.

In practice, `DenseTrainerPool.get_or_create(...)` calls `ResourcePool.ensure_gpus_available(1)` before creating a new actor and registers the actor into `ResourcePool` for LRU accounting.

## Idle and evictable definitions

Each actor has:
- `creating`: True while the actor is initializing. Creating actors are never evicted.
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
- `ResourcePool._get_evictable_actors_lru()` sorts evictable actors by `last_accessed` ascending (oldest first).

## Pending reservations (concurrency safety)

To avoid races where multiple concurrent requests all observe the same "available GPUs" and over-allocate:
- `ResourcePool.reserve_gpus(n)` increments `_pending_gpus`.
- `ResourcePool.get_effective_available_gpus()` returns `ray.available_resources()["GPU"] - _pending_gpus`.
- `ResourcePool.release_pending_gpus(n)` decrements `_pending_gpus` after the actor is created (or the create failed).

If a caller ignores pending reservations and uses raw Ray availability, the pool cannot prevent oversubscription.

## Killing an actor

`ResourcePool._kill_actor(entry)`:
- gets actor handle (cached handle or `ray.get_actor(name, namespace)`)
- tries `actor.shutdown.remote()` if present (best-effort)
- then `ray.kill(actor)` and removes it from `_entries`

This frees GPUs at Ray scheduling level. It does not guarantee CUDA memory defragmentation on the node.

## Startup reconciliation (detached actors)

Many actors are created as detached so they can survive an API server restart.

Problem: after restart, server memory is empty, but actors still exist.

Startup hook `tinker_server/app.py:_cleanup_stale_actors()`:
- lists named actors in the `tinker` namespace
- health-checks them (`__ray_ready__`)
- registers alive actors into `ResourcePool` and marks them ready
- kills dead/unresponsive ones

This keeps `ResourcePool` aligned with the set of detached actors that still exist.

## Implications for architecture changes

- If you introduce a new GPU-using actor type, register it in `ResourcePool` and decide how it should set:
  - `creating` (protect during init)
  - `current_session` (if training-like)
  - `node_id` (if placement decisions need it)
- Ensure the code path that creates the actor calls `ensure_gpus_available` before `ray.remote(...).remote(...)`.
- If the actor is detached, add it to startup reconciliation in `tinker_server/app.py`.
