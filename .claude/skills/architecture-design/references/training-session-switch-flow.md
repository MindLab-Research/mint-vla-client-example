# Training request flow and session switching

This document explains the end-to-end control flow for training requests, and how a single shared trainer switches between multiple training sessions without mixing state.

It is written against the fix for issue 193/194 on branch `codex/issue193-194-session-serialization`.

## Why this document exists

The training path has several different concepts that are easy to conflate:

- client `session_id`
- server `model_id`
- async `request_id`
- queue scheduling metadata
- the "currently loaded session" inside a Ray training actor

The bug behind issue 193/194 came from exactly this gap: dequeue order was being preserved better, but same-session operations could still reach the backend actor out of order under load.

## The four identifiers

- `session_id`
  - Created by `POST /api/v1/create_session`
  - Client-facing grouping/metadata identifier
  - Stored in server memory
  - Does not directly identify mutable trainer state

- `model_id`
  - Created by `POST /api/v1/create_model`
  - The real server-side training session identity
  - Keys `TrainingSessionManager`
  - Used as the session identity passed into backend training actors

- `request_id`
  - Created per async API request
  - Keys `FutureStore`
  - Used for polling via `retrieve_future`

- backend current session
  - The session currently loaded into a shared trainer actor
  - Dense backend stores it as `TrainingWorker._current_session_id`
  - Megatron stores it as `MegatronWorkerGroup._current_session`

The important rule is:

- `model_id` is the identity that owns mutable training state
- `session_id` is not

See also `state.md`.

## End-to-end request path

For asynchronous training operations such as `forward_backward`, `optim_step`, and `save_weights_for_sampler`, the control flow is:

1. Route validates request and looks up the `TrainingSession` by `model_id`.
2. Route creates `request_id` and a pending future in `FutureStore`.
3. Route enqueues work into `ApiWorkQueue` with extra metadata.
4. Queue actor chooses a dequeue candidate.
5. Queue actor assigns a per-session execution sequence number if the item carries `execution_serial_key`.
6. A local queue worker receives the item.
7. The worker waits in `_execution_serialized(...)` until this item is the next allowed sequence for that session.
8. The app-level executor deserializes the JSON and calls the `_do_*` route helper.
9. The route helper calls `training_engine.*(...)`.
10. The backend actor ensures the correct session is loaded before doing any forward/backward/save/load work.
11. The result resolves or fails the `request_id` in `FutureStore`.

Relevant code:

- route enqueue metadata: `tinker_server/routes/training.py`
- executor registration: `tinker_server/app.py`
- dequeue and execution gate: `tinker_server/backend/api_work_queue.py`

## Route metadata: scheduler vs execution serialization

Training routes attach extra metadata using `_build_training_scheduler_extra(...)`:

- `scheduler_enabled`
- `scheduler_domain`
- `scheduler_session_key`
- `execution_serial_key`
- `training_op`
- optional `seq_id`

The distinction matters:

- `scheduler_*` influences how the queue chooses which item to dequeue next
- `execution_serial_key` forces same-session execution order after dequeue

For training routes, `execution_serial_key` is:

```text
training_session:{model_id}
```

This same key is also attached to checkpoint/state routes in `tinker_server/routes/weights.py`:

- `/save_weights`
- `/save_state`
- `/load_state`

And the remaining same-session mutating sync routes now enqueue internal work onto that same serialized lane before returning:

- `/reset_expert_bias`
- `DELETE /models/{model_id}`

That means the following operations now share one per-session serialized lane:

- `training.create_model`
- `training.create_model_from_state`
- `training.forward_backward`
- `training.optim_step`
- `training.save_weights_for_sampler`
- `training.reset_expert_bias`
- `training.delete_model`
- `weights.save_weights`
- `weights.save_state`
- `weights.load_state`

## Why scheduler alone was not enough

Issue 194 improved dequeue ordering, but dequeue order is not the same thing as backend submission order.

Under load, multiple queue workers may process items for the same `model_id` concurrently. If one operation does more local preprocessing before it calls `.remote()`, then a later-dequeued operation can still reach the backend actor first.

That is exactly what could happen here:

- `forward_backward` does more request shaping and data serialization before its remote call
- `optim_step` is lighter
- with multiple queue workers, a later `optim_step` could overtake an earlier `forward_backward`

So the fix added a second mechanism:

1. On dequeue, the queue actor assigns a monotonic `execution_serial_seq` per `execution_serial_key`.
2. Before running the executor, the local worker waits until that sequence is next.

This is implemented in:

- dequeue sequence assignment: `ApiWorkQueueActor.dequeue()`
- execution gate: `ApiWorkQueueClient._execution_serialized(...)`

This is the key difference:

- scheduler controls selection
- execution serialization controls actual submission order

Both are needed.

The same `execution_serial_key` also now covers lifecycle creation for a concrete `model_id`:

- `create_model`
- `create_model_from_state`

That prevents a follow-up op from running against a half-initialized session after
`TrainingSessionManager` metadata has been published but before actor creation or
checkpoint load has finished.

## App executor layer

`tinker_server/app.py` registers one executor per queue op. Examples:

- `training.forward_backward` -> `training._do_forward_backward(...)`
- `training.optim_step` -> `training._do_optim_step(...)`
- `training.save_weights_for_sampler` -> `training._do_save_weights_for_sampler(...)`
- `weights.save_state` -> `weights._do_save_state(...)`
- `weights.load_state` -> `weights._do_load_state(...)`

The execution gate happens before these executors run, so by the time an executor begins, same-session FIFO order is already enforced.

## Server-side training session metadata

`TrainingSessionManager` stores one `TrainingSession` per `model_id`.

`TrainingSession` contains:

- `model_id`
- original client `session_id`
- `base_model`
- `current_step`
- `accumulated_gradients`
- `backend`
- optional per-session inference engine

This is control-plane state in API server memory. It is not the full mutable training state.

The real mutable state lives in backend actors:

- LoRA weights
- gradients
- optimizer state
- backend-local current step

## Dense backend session switching

Dense training uses a pooled `TrainingWorker` in `tinker_server/backend/verl_training.py`.

Every training RPC passes `session.model_id` into the worker. The worker calls:

- `TrainingWorker._ensure_session_loaded(session_id)`

The switch logic is:

1. If `target == current`, do nothing.
2. If another session is currently loaded:
   - save outgoing session state with gradients
3. If the new session already exists on disk:
   - load LoRA weights
   - load optimizer state
   - load gradients
   - restore step and learning rate
4. Otherwise:
   - reinitialize LoRA weights
   - zero gradients
   - reset step state
5. Mark `self._current_session_id = session_id`

Dense session state is managed by `SessionStateManager`, which writes:

- `adapter_model.safetensors`
- `optimizer.pt`
- `gradients.pt`
- `training_meta.json`

Important detail:

- `forward_backward(...)` increments server-side `session.accumulated_gradients`
- `optim_step(...)` resets that counter after the backend applies the gradients

So if `optim_step` races ahead of the matching `forward_backward`, the trainer can apply the wrong state. The execution serialization fix prevents that.

## Megatron backend session switching

Megatron uses a detached `MegatronWorkerGroup` plus per-rank `MegatronRankWorker` actors.

At the start of each training operation, the group calls:

- `MegatronWorkerGroup._ensure_session_loaded(session_id, ...)`

This switch has two parts.

### 1. In-memory swap of optimizer and gradients

The group calls:

- `MegatronWorkerGroup._swap_session_on_workers(new_session_id)`

Each rank then runs:

- `MegatronRankWorker.swap_session_state(new_session_id)`

Per-rank cached state is stored in Python dicts:

- `_session_gradients`
- `_session_optimizer_states`

This isolates:

- accumulated gradients
- optimizer momentum / variance

### 2. Adapter save/load

The group also saves and loads LoRA adapter state through `MegatronSessionStateManager`.

That means Megatron session switching is hybrid:

- LoRA weights: disk-backed
- optimizer and gradients: actor-memory-backed

## Sticky train mode in Megatron

Megatron also has a separate optimization: sticky train mode.

Key methods:

- `_ensure_sticky_train_mode(...)`
- `_release_sticky_train_mode(...)`

Purpose:

- keep the expensive `train_mode()` context open across multiple chunks of the same session
- avoid repeated CPU<->GPU model offload churn when `param_offload=True`

This is not the same thing as session isolation. It is a performance optimization layered on top of session isolation.

The interaction with session switching is:

- if the same session continues, sticky mode may be reused
- if the session changes, sticky mode is released first, optionally snapshotting gradients
- then session swap runs
- then the new session can open or reuse its own sticky context

This is why session ordering bugs were so damaging under high load:

- they could corrupt not only logical session order
- but also the sticky reuse assumptions inside Megatron

## Save/export/checkpoint operations

Three families of state-related operations matter here.

### `save_weights_for_sampler`

Purpose:

- export inference-ready LoRA weights
- must not include optimizer artifacts

Flow:

- route enqueues with `execution_serial_key`
- executor calls `training._do_save_weights_for_sampler(...)`
- backend ensures the correct session is loaded
- dense and Megatron each export sampler-only artifacts

### `save_state`

Purpose:

- export a training checkpoint for resume-like usage
- must include optimizer artifacts

Flow:

- route enqueues with the same `execution_serial_key`
- executor calls `weights._do_save_state(...)`
- backend ensures the correct session is loaded
- checkpoint metadata is written and mirror is started

### `load_state`

Purpose:

- load a training checkpoint back into a `model_id`

This also uses the same serialized lane, so it cannot interleave with forward/backward or optimizer step for the same session.

## Example timeline

Suppose one session `A` emits:

1. `forward_backward(seq_id=10)`
2. `optim_step(seq_id=10)`
3. `save_weights_for_sampler(seq_id=10)`

And another session `B` is also active.

What now happens is:

1. The queue may interleave dequeue across `A` and `B` depending on scheduling policy.
2. But every `A` item gets the same `execution_serial_key=training_session:A`.
3. The queue actor stamps those `A` items with monotonically increasing `execution_serial_seq`.
4. Local workers must execute `A` in that sequence order, even if different workers picked the items.
5. Inside the backend actor:
   - first `A` loads or reuses session `A`
   - second `A` sees `A` already loaded, so no session switch
   - third `A` also stays on `A`
6. When a `B` item arrives:
   - backend saves `A` state if needed
   - backend loads or initializes `B`

This is the intended layering:

- queue-level fairness/scheduling between sessions
- queue-worker serialization within one session
- backend actor session swap only when the active session changes

## Persistence and restart boundaries

Not all state survives the same failures.

Dense backend:

- session swap files are local trainer-node files under `/tmp/mint_sessions`
- good for time-slicing isolation
- not a durable cross-node checkpoint contract

Megatron backend:

- adapter files can persist on PFS
- optimizer and gradient caches are in actor memory
- if actors die, those in-memory caches are lost

API server:

- `TrainingSessionManager`
- `FutureStore`
- in-process sampling mappings

These are all process-memory state and are lost on API restart.

So "session isolation" and "durable resume" are related but separate concerns.

## Mental model

When reading the training code, use this model:

- `model_id` chooses the mutable training session
- queue scheduling decides which request to consider next
- execution serialization decides the per-session execution order
- backend `_ensure_session_loaded(...)` decides whether a real session swap is needed
- sticky train mode is only a performance optimization inside the Megatron backend

If any of those layers is missing or confused, high-load multi-step training can mix state and produce sudden loss spikes.
