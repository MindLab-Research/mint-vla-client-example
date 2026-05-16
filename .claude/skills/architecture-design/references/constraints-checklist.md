# Design constraints and change checklist

## Design constraints that affect architecture

- Ray actors keep the code they started with. If you change actor code, you must recreate the actor to observe the change.
- Ray "GPU availability" is scheduling-level, not CUDA-memory-level. Mint relies on `ModelActorSupervisor` node-pin reconciliation, `ModelWorkScheduler` leases, `ModelActorRegistry` inventory, and placement groups (Megatron + multi-node vLLM) to keep large GPU reservations schedulable.
- Detached actors survive API restarts. Startup must reconcile: kill dead actors, register alive ones, and accept that in-process mappings (sessions, LoRA registries) may be lost.

## Architecture change checklist (project-specific)

- Adding a new endpoint
  - Add/extend request/response types in `tinker_server/models/types.py`.
  - Implement route in `tinker_server/routes/*`.
  - If work is async, return `request_id` and use `TaskStateFutures` + `/retrieve_future` semantics.

- Adding or changing a Ray actor type
  - Decide: is it detached? If yes, add startup reconciliation logic in `tinker_server/app.py:_cleanup_stale_actors()`.
  - If it is GPU-using, register live actors in `tinker_server/backend/model_actor_registry.py` so local inventory, inflight marking, and eviction safeguards stay correct.
  - Audit filesystem assumptions: can the API server see the same paths as the actor?

- Adding a new model or changing parallelism
  - Update `tinker_server/backend/model_registry.py` (inference and training parallelism are specified separately).
  - Check inference-vs-training constraints independently (vLLM LoRA limitations differ from Megatron LoRA limitations).

- Changing weight transfer
  - Choose Ray object store vs shared-path load based on tensor count and payload size.
  - Preserve Tinker semantics: sampling sessions expect frozen weights per `sampling_session_id`.
