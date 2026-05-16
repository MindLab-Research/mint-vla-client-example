# Async futures (Tinker polling protocol)

Many endpoints return `{"request_id": "<uuid>"}` immediately and complete the work asynchronously. The polling surface is the Tinker contract and must be preserved.

## Retrieve semantics (`POST /api/v1/retrieve_future`)

Implementation: `tinker_server/routes/futures.py` backed by `TaskStateFutures` in `tinker_server/backend/task_state_store.py`.

- HTTP 408: `PENDING` (client should retry)
- HTTP 200 with `{"error": ...}`: terminal non-success (for example `FAILED`, `EXPIRED`, `RETRIEVED`)
- HTTP 200 with result payload: `DONE`
- HTTP 404: unknown `request_id` (server has forgotten it)

Important detail: `TaskStateStore` is the single durable index for terminal futures. After returning `DONE` or `FAILED`, the server marks the `request_id` as `RETRIEVED` while retaining terminal metadata and payload pointers. A second `retrieve_future` is served idempotently from `TaskStateStore` when the payload or error is still retained; if the payload has been removed but the terminal task record remains, the route returns `{"error": "Known terminal future evicted", ...}` rather than treating the request as unknown.

## Why futures exist in Mint

Most work runs on Ray GPU actors and can exceed typical HTTP request lifetimes. The futures protocol keeps the HTTP surface stable and matches the Tinker client contract. Do not silently change status codes (for example, 408 to 202) or switch to streaming without updating the client contract.

## Where futures live

`TaskStateFutures` is an in-process facade. Durable task state lives in the detached `TaskStateStore` actor (`mint_task_state_store` by default), and result payloads are written through `TaskPayloadStore`.

The facade preserves the old Tinker future methods (`async_resolve`, `async_fail`, `async_get_status`, etc.) while routing all persistent state through `TaskStateStore`. Completed result payloads are written to the payload store, and the task record stores the payload path, checksum, size, status, error, and metadata.

There is no separate future replay index. Retrieve hot-cache entries are process-local accelerators only; restart recovery, terminal replay, and payload-evicted detection all use `TaskStateStore`.

`TaskStateStore` uses active-task indexes (`pending`, `queued`, `assigned`, `leased`, `running`, `finalizing`) for scheduler hydration and metadata-based failure/cleanup. Full table scans are not part of the hot path.

## Admission and scheduling

Async endpoints that require model-runtime scheduling go through `ModelWorkScheduler`:
- API routes first create or ensure the task in `TaskStateStore` via `TaskStateFutures`.
- The route appends a `ModelWorkItem` to the detached `ModelWorkScheduler` actor (`mint_model_work_scheduler` by default).
- `ModelWorkScheduler` keeps the hot domain backlog, per-replica subqueues, leases, and fairness state in memory.
- Runtime actors claim from their scheduler-owned subqueue. Claiming is independent of `retrieve_future`; result polling reads `TaskStateStore`.
- For scheduler leases with `attempt_id` and `scheduler_epoch`, `ModelRuntimeActor` owns terminal commit to `TaskStateStore` and lease completion/failure. Route-level `_do_*` functions may still use `TaskStateFutures.async_resolve/async_fail` as an executor-local completion signal, but those calls are buffered while running under a model-work execution context and do not write terminal state directly.

On admission failure, the API must return HTTP 429 with a structured overload reason. V1 does not enforce a hard active-task cap; add one only if the active-task index becomes a measured bottleneck.

## Request-path async rules

The request path uses native async Ray integration on hot control-plane operations:

- Routes await Ray refs directly through async helpers instead of calling blocking `ray.get(...)`.
- Request paths do not call `init_ray()` or attempt reconnection. Startup owns Ray initialization.
- Startup warms cached detached-actor handles for the request-path stores.
- If a cached detached-actor handle dies, the async helper may reacquire the actor by name once. This is a stale-handle recovery path, not permission for routes to bootstrap a new Ray client or hide a missing actor.

## Model work scheduling

For training-session-bound requests, the training routes tag scheduler metadata by default (`MINT_SCHEDULER_ENABLE` defaults to `1` unless explicitly disabled). The tagged route set includes:

- `training.create_model`
- `training.create_model_from_state`
- `training.forward`
- `training.forward_backward`
- `training.optim_step`
- `training.train_step`
- `training.reset_expert_bias`
- `training.save_weights_for_sampler`
- `training.delete_model`

The training route tags these requests with `extra` metadata:

- `scheduler_enabled`
- `scheduler_domain` (typically `"{backend}:{base_model}"`)
- `scheduler_session_key` (uses server-side `model_id`)

Tagged requests are grouped by scheduler domain and assigned into runtime-owned subqueues. Scheduling semantics:

- Same session preserves FIFO order.
- Each scheduler domain is single-flight: only one scheduled request from that domain can be leased to a worker at a time. This is intentional for shared training actors where overlapping requests against the same actor would violate session-state invariants.
- Across sessions, selection is fairness-based (`MINT_SCHEDULER_FAIRNESS=oldest|rr`) with starvation guard (`MINT_SCHEDULER_STARVATION_S`).
- Sticky bursts are bounded by `MINT_SCHEDULER_MAX_CONSECUTIVE`.
- Optional coalescing window (`MINT_SCHEDULER_COALESCE_MS`) briefly waits for another chunk from the previous session before switching.
- There is no follow-up hold window anymore. A session only keeps the domain lease while its claimed request is still live; stale consumer generations release those leases during restart reconciliation.

This is a deliberate tradeoff: global strict FIFO across sessions is relaxed for these tagged training ops to reduce cross-session thrash, while preserving per-session ordering and bounded fairness.

## Reaping and cleanup

`MaintenanceCronActor` owns periodic cleanup. `TaskStateFutures.async_reap()` is currently a compatibility facade over `TaskStateStore`; task/result retention policy belongs in `TaskStateStore` and payload-store cleanup.

## Detached actor hygiene

Detached actors do not hot-reload. Changes to `TaskStateStore`, `ModelWorkScheduler`, `ModelRuntimeActor`, or `MaintenanceCronActor` require killing the matching detached actors in the target namespace before restart.
