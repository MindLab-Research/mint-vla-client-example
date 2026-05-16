# Multi-tenant concurrent training (session state offload/swap)

"Multi-tenant concurrent training" here means multiple independent training sessions share a smaller number of GPU trainers via time-slicing, while preserving per-session isolation of:
- LoRA weights (theta)
- accumulated gradients (grad theta)
- optimizer state (momentum/variance)

The system implements two distinct mechanisms:
1. Dense models (PEFT TrainingWorker): disk-backed state per session.
2. MoE models (MegatronWorkerGroup): disk-backed LoRA weights plus in-memory optimizer/gradient swap on each rank.

## Dense training: pooled detached trainer actor, many sessions

Actors:
- Dense training uses a detached `TrainingWorker` (`tinker_server/backend/verl_training.py`).
- Actors are created and reused via `DenseTrainerPool` (same file).

Pool key and rank policy:
- The pool is keyed by `base_model`.
- Each pooled actor is initialized with a `max_lora_rank` (default 64) and can serve any session with `lora_rank <= max_lora_rank`.
- If a request needs a larger rank than the existing actor supports, the caller must kill/recreate the actor with a higher `max_lora_rank`.

State swap implementation:
- Each training RPC passes `session_id=session.model_id` into the actor.
- Inside the actor, `_ensure_session_loaded(session_id)` enforces the session boundary:
  - If switching away from another session:
    - saves outgoing session state via `SessionStateManager.save_state(...)`
    - this writes:
      - `adapter_model.safetensors` (LoRA weights)
      - `optimizer.pt` (optimizer state_dict)
      - `gradients.pt` (parameter gradients, if present)
      - `training_meta.json` (step count, lr)
  - If switching to an existing session:
    - loads the above files and restores gradients back onto parameters
  - If switching to a new session:
    - reinitializes LoRA weights (`reinit_lora_weights()`)
    - clears optimizer momentum and zeros gradients

What makes this "multi-tenant":
- The trainer actor holds exactly one model instance on GPU.
- Session identity chooses which session state is loaded into that model before each forward/backward/optim step.

Where state is stored:
- `SessionStateManager` defaults to `base_path="/tmp/mint_sessions"` on the trainer's node.
- This is local to the node/process environment, not a shared filesystem contract.

Failure and restart behavior:
- If the trainer actor is killed (eviction/crash) and recreated on a different node, `/tmp/mint_sessions` state is not guaranteed to follow.
- The disk-backed session swap is an isolation mechanism for time-slicing, not a durable "resume training anywhere" mechanism.

Idle behavior:
- `TrainingWorker` is created with `idle_timeout=0` by default (no self-termination); scheduler-owned runtimes are reconciled by `ModelActorSupervisor`, while direct legacy eviction paths use `ModelActorSupervisorInventory`.

## MoE training: shared MegatronWorkerGroup with per-session isolation

Actors:
- MoE training uses a `MegatronWorkerGroup` (detached controller actor, `num_gpus=0`) that owns:
  - a Ray placement group
  - N `MegatronRankWorker` actors, each with `num_gpus=1`

Session switching call path:
- `MegatronWorkerGroup._ensure_session_loaded(session_id)` is invoked at the start of training ops (forward/backward/forward/optim_step).
- It performs two separate swaps:

1. Optimizer and gradient swap (in-memory, per-rank)
   - `MegatronWorkerGroup._swap_session_on_workers(session_id)` calls `MegatronRankWorker.swap_session_state(new_session_id)` on every rank.
   - Each rank worker:
     - captures outgoing gradients to CPU tensors (`_capture_gradients`) unless already captured by forward_backward
     - captures outgoing optimizer state to CPU (`_capture_optimizer_state`)
     - restores incoming gradients (or zeros if new/consumed)
     - restores incoming optimizer state (or resets if new)
   - The cached state is stored in per-rank Python dicts keyed by session_id:
     - `_session_gradients`
     - `_session_optimizer_states`

2. LoRA weight swap (disk-backed adapter checkpoints)
   - `MegatronSessionStateManager` maps `session_id` to a checkpoint directory on a shared filesystem:
     - default: `${PFS_TINKER_PATH}/checkpoints/megatron_sessions/{session_id}_checkpoint/` (override: `MINT_MEGATRON_SESSIONS_BASE_PATH`)
   - On session switch:
     - saves outgoing adapter state via `save_adapter_state(...)` into that directory
     - loads incoming adapter state via `load_adapter_state(...)` if it exists
     - otherwise reinitializes LoRA weights for a new session

What makes this "multi-tenant":
- One MegatronWorkerGroup can serve multiple sessions by swapping session-local state on demand.
- Optimizer momentum and gradient accumulation remain session-local despite time-slicing.

Failure and restart behavior:
- LoRA weights can persist (saved on PFS).
- Optimizer and gradient caches live in actor memory. If the actor restarts, those are lost and new sessions start with fresh optimizer/zero gradients unless explicitly restored by some external mechanism.
- `MegatronSessionStateManager` metadata dict (step, lr, actual_rank) is in-process; adapter files can exist even if metadata is lost after restart.

## Interaction with eviction

Eviction kills whole actors. Session state caches in memory are lost.

To reduce accidental eviction during active training:
- MoE actors: `VerlTrainingEngine._touch_actor()` calls `ModelActorSupervisorInventory.touch(actor_name)` and sets `current_session`.
- Dense pool entries and registry entries are also touched on reuse.

Eviction is still possible when a session stops making requests long enough to be considered idle.

## Relationship to checkpoint APIs

Time-slicing swap (this document) is an internal mechanism to multiplex a limited number of GPU trainers.

Durable resume is a separate concern handled by `/save_state` and `/load_state` in `tinker_server/routes/weights.py`.
For Megatron, public training checkpoints include per-rank optimizer shards, so optimizer resume can be backed by a checkpoint. Actor-only snapshots also include LR scheduler state, but public Megatron checkpoints do not currently write scheduler state. Treat optimizer state and scheduler state as separate sources in docs and code.

`load_state(..., optimizer=True)` loads checkpoint weights and optimizer into the live Megatron actor, then primes the session cache from that exact checkpoint. The cache metadata records the checkpoint path and identity, while `actor_only_state.json` records that optimizer authority is actor-local until a durable save or actor snapshot changes that source. Public Megatron checkpoints do not currently restore gradient or scheduler state.
