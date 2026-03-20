# Training architecture

Training is controlled via `/api/v1/*` routes in `tinker_server/routes/training.py`. The API surface follows the Tinker SDK expectations (create model, forward/backward, optimizer step, checkpointing).

## Why time-slicing exists

Training is designed to support multiple sessions with limited GPUs by swapping session-local state (LoRA weights, gradients, optimizer state) into a smaller number of trainer actors. This keeps per-session isolation while reusing expensive base model initialization.

See `training-multitenancy.md` for the dense vs Megatron swap mechanisms.

## State ownership

- `TrainingSessionManager` stores per-`model_id` metadata and lifecycle in server memory.
- The actual training state (weights, optimizer, step counters) lives inside Ray actors.
- `ResourcePool` is the global control for reclaiming GPUs by evicting idle actors.

## Backends

- Dense models: pooled `TrainingWorker` actors (created by `DenseTrainerPool`) that time-slice many sessions by swapping per-session state onto the shared actor.
- MoE models: distributed Megatron workers (`megatron_distributed.py`) with explicit parallelism.

## Session lifecycle and idle cleanup

Training sessions (`model_id`) have a bounded lifecycle:

- **Explicit deletion**: Client calls `DELETE /api/v1/models/{model_id}`.
- **Idle cleanup**: `TrainingSessionManager` runs a background task (every 60s) that evicts sessions inactive for longer than `MINT_TRAINING_INACTIVITY_TIMEOUT` (default 3600s / 1 hour).
- **Server shutdown**: `shutdown_all()` cleans up all sessions.

The idle cleanup mirrors the inference `SessionManager._cleanup_loop` pattern. Training operation routes (`_do_train_step`, `_do_forward_backward`, `_do_forward`, `_do_optim_step`, `_do_save_weights_for_sampler`, `_do_save_state`, `_do_save_weights`, `_do_load_state`) explicitly call `touch_session()` to update the session's `last_activity` timestamp. Read-only lookups (`GET /models/{model_id}`, `GET /training_runs`, existence checks) do NOT extend the idle deadline. When cleanup fires, it performs the full deletion flow:
1. `engine.shutdown_session` (release GPU actor reference)
2. `delete_session` (remove from in-memory manager)
3. `delete_training_session` (remove from detached Ray store)
4. `resource_pool.clear_session` (clear stale session pins)

This prevents unbounded session accumulation when clients disconnect without calling DELETE.
