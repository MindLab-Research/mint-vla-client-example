# Async futures (Tinker polling protocol)

Many endpoints return `{"request_id": "<uuid>"}` immediately and complete the work asynchronously. The polling surface is the Tinker contract and must be preserved.

## Retrieve semantics (`POST /api/v1/retrieve_future`)

Implementation: `tinker_server/routes/futures.py` backed by `tinker_server/backend/future_store.py`.

- HTTP 408: `PENDING` (client should retry)
- HTTP 200 with `{"error": ...}`: terminal non-success (for example `FAILED`, `EXPIRED`, `RETRIEVED`)
- HTTP 200 with result payload: `DONE`
- HTTP 404: unknown `request_id` (server has forgotten it)

Important detail: `request_id` is single-use for results. After returning `DONE` or `FAILED`, the server marks the `request_id` as `RETRIEVED` and releases associated reservations. A second `retrieve_future` returns HTTP 200 with `{"error": "Future already retrieved", ...}` (not 404).

## Why futures exist in Mint

Most work runs on Ray GPU actors and can exceed typical HTTP request lifetimes. The futures protocol keeps the HTTP surface stable and matches the Tinker client contract. Do not silently change status codes (for example, 408 to 202) or switch to streaming without updating the client contract.

## Where futures live (bounded API heap)

`FutureStore` is a detached Ray actor (`tinker_future_store`) in the configured Ray namespace. This avoids in-process retention when:
- many requests are queued faster than they can execute
- clients do not retrieve futures promptly

Completed results are stored as Ray object store references (`ray.put(...)`), and `FutureStore` only holds request_id to ref mappings and small metadata.

`FutureStore` TTLs:
- `future_store_ttl_s`: execution timeout (applies only after `RUNNING`)
- `future_store_queue_ttl_s`: queue timeout (applies after `QUEUED`, before `RUNNING`)
- `future_store_done_ttl_s`: retention window for `DONE`/`FAILED` before transitioning to `EXPIRED`
- `future_store_tombstone_ttl_s`: retention window for `EXPIRED`/`RETRIEVED` tombstones before forgetting the request_id

## Admission control (no OOM under overload)

Async endpoints must use the admission layer before creating futures:
- `tinker_server/backend/capacity_manager.py` (detached Ray actor `tinker_capacity_manager`)
  - reserves `queue_bytes` (request JSON enqueued in Ray) against a fixed budget
  - reserves `object_store_bytes` (expected result size) against Ray's `available_resources()["object_store_memory"]`
- `tinker_server/backend/api_work_queue.py` (detached Ray actor `tinker_api_work_queue`)
  - stores request JSON in Ray, and local API workers pull items and execute them

On admission failure, the API must return HTTP 429 with a structured overload reason (for example `queue_bytes_budget_exceeded` or `object_store_budget_exceeded`). Overload is explicit; the server must not allow unbounded backlog to grow until OOM.

## Training queue scheduling (session-aware mode)

The detached API work queue is still FIFO for untagged work. For training-session-bound requests, the training routes now tag scheduler metadata by default (`MINT_SCHEDULER_ENABLE` defaults to `1` unless explicitly disabled). The tagged route set includes:

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

When enabled, tagged requests are grouped into per-domain/per-session subqueues. Scheduling semantics:

- Same session preserves FIFO order.
- Each scheduler domain is single-flight: only one scheduled request from that domain can be leased to a worker at a time. This is intentional for shared training actors where overlapping requests against the same actor would violate session-state invariants.
- Across sessions, selection is fairness-based (`MINT_SCHEDULER_FAIRNESS=oldest|rr`) with starvation guard (`MINT_SCHEDULER_STARVATION_S`).
- Sticky bursts are bounded by `MINT_SCHEDULER_MAX_CONSECUTIVE`.
- Optional coalescing window (`MINT_SCHEDULER_COALESCE_MS`) briefly waits for another chunk from the previous session before switching.
- There is no follow-up hold window anymore. A session only keeps the domain lease while its dequeued request is still live; stale consumer generations release those leases during restart reconciliation.

This is a deliberate tradeoff: global strict FIFO across sessions is relaxed for these tagged training ops to reduce cross-session thrash, while preserving per-session ordering and bounded fairness.

## Reaping and reservation release

`tinker_server/app.py` runs a reaper loop that calls `FutureStore.reap()` and releases any external reservations for request_ids that transitioned to terminal tombstones (expired or timed out). This is what prevents reservation leaks when clients do not retrieve futures.

## Detached actor hygiene

Detached actors do not hot-reload. Changes to `FutureStore` or admission/work queue logic require killing the detached actors in the target namespace before restart.
