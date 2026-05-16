# Training architecture

Training is controlled via `/api/v1/*` routes in `tinker_server/routes/training.py`. The API surface follows the Tinker SDK expectations (create model, forward/backward, optimizer step, checkpointing).

For a request-by-request walkthrough of queueing, execution serialization, backend session switching, and checkpoint/export interactions, see `training-session-switch-flow.md`.

## Why time-slicing exists

Training is designed to support multiple sessions with limited GPUs by swapping session-local state (LoRA weights, gradients, optimizer state) into a smaller number of trainer actors. This keeps per-session isolation while reusing expensive base model initialization.

See `training-multitenancy.md` for the dense vs Megatron swap mechanisms.

## State ownership

- `TrainingSessionManager` stores per-`model_id` metadata and lifecycle in server memory.
- The actual training state (weights, optimizer, step counters) lives inside Ray actors.
- `ModelActorSupervisor` and `ModelWorkScheduler` own runtime reconciliation and scheduling. `ModelActorSupervisorInventory` is the local inventory and best-effort eviction helper for GPU actors.

## Backends

- Dense models: pooled `TrainingWorker` actors (created by `DenseTrainerPool`) that time-slice many sessions by swapping per-session state onto the shared actor.
- MoE models: distributed Megatron workers (`megatron_distributed.py`) with explicit parallelism.

## Session lifecycle and idle cleanup

Training sessions (`model_id`) have a bounded lifecycle:

- **Explicit deletion**: Client calls `DELETE /api/v1/models/{model_id}`.
- **Idle cleanup**: `TrainingSessionManager` runs a background task (every 60s) that evicts sessions inactive for longer than `MINT_TRAINING_INACTIVITY_TIMEOUT` (default 3600s / 1 hour).
- **Server shutdown**: `shutdown_all()` cleans up all sessions.

The idle cleanup mirrors the inference `SessionManager._cleanup_loop` pattern, including `inflight_ops` protection (analogous to `SessionInfo.inflight_requests`):

- **Queued HTTP handlers** call `mark_inflight(+1)` before enqueue so queue delay cannot race idle cleanup.
- **Background workers** for queued existing-session operations release that claim with `mark_inflight(-1)` in `finally`, which also refreshes `last_activity` on completion.
- **`_do_create_model`** / **`_do_create_model_from_state`** call `mark_inflight(+1)` right after `create_session()` to protect during slow actor creation.
- **Session activity persistence**: `touch_session()` / `mark_inflight()` write `last_activity` into the detached training-session store so API restarts restore the real idle deadline instead of falling back to `created_at`.
- **Read-only lookups** (`GET /models/{model_id}`, `GET /training_runs`, existence checks) do NOT extend the idle deadline.
- **`_restore_training_session`** restores persisted `last_activity` when present, and falls back to `created_at` only for older store entries that predate that field.

When cleanup fires, it skips sessions with `inflight_ops > 0`, then performs the full deletion flow:
1. `engine.shutdown_session` (release GPU actor reference)
2. `delete_session` (remove from in-memory manager)
3. `delete_training_session` (remove from detached Ray store)
4. `model_actor_supervisor_inventory.clear_session` (clear stale session pins)

This prevents unbounded session accumulation when clients disconnect without calling DELETE.
