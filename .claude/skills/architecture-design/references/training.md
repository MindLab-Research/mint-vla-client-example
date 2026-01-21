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
