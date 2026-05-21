# Async futures (Mint polling protocol)

Many endpoints return `{"request_id": "<uuid>"}` immediately and complete the work asynchronously. The polling surface is the Tinker contract and must be preserved.

## Retrieve semantics (`POST /api/v1/retrieve_future`)

Implementation: `mint_server/routes/futures.py` backed by `TaskFutureService` in `mint_server/backend/task_state_store.py`.

- HTTP 408: `PENDING` (client should retry)
- HTTP 200 with `{"error": ...}`: terminal non-success (for example `FAILED`, `EXPIRED`, `RETRIEVED`)
- HTTP 200 with result payload: `DONE`
- HTTP 404: unknown `request_id` (server has forgotten it)

Important detail: `TaskStateStore` is the single durable source of truth for task state, result metadata, and terminal future indexes. After returning `DONE` or `FAILED`, the server marks the `request_id` as `RETRIEVED` while retaining terminal metadata and payload pointers. A second `retrieve_future` is served idempotently from `TaskStateStore` when the payload or error is still retained; if the payload has been removed but the terminal task record remains, the route returns `{"error": "Known terminal future evicted", ...}` rather than treating the request as unknown.

## Why futures exist in Mint

Most work runs on Ray GPU actors and can exceed typical HTTP request lifetimes. The futures protocol keeps the HTTP surface stable and matches the Mint SDK client contract. Do not silently change status codes (for example, 408 to 202) or switch to streaming without updating the client contract.

## Where futures live

`TaskFutureService` is an in-process facade. Durable task state lives in the detached `TaskStateStore` actor (`mint_task_state_store` by default), and result payloads are written through the in-process `TaskPayloadStore` filesystem helper. `TaskPayloadStore` is not a Ray actor and has no lifecycle to reconcile.

The facade preserves the old Tinker future methods (`async_resolve`, `async_fail`, `async_get_status`, etc.) while routing all persistent state through `TaskStateStore`. Completed result payloads use a staged commit protocol: first SQLite records the expected payload path, then the payload store helper atomically publishes the vePFS JSON file, then SQLite commits the terminal status, checksum, size, and result pointer. Model-work finalization records the lease identity and `finalizing_until`; direct in-process future resolution records the staged path while leaving the task pending until terminal commit. This is primarily for GC correctness: every non-temporary payload path is attributable to a task row even if the process crashes between file publish and terminal metadata commit.

Direct in-process future resolution always uses a fresh `future__<uuid>` staged path, even if request metadata contains a model-work attempt id. Model-work attempt ids are valid only on the scheduler lease finalization path.

There is no separate future replay index. Retrieve hot-cache entries are process-local accelerators only; restart recovery, terminal replay, and payload-evicted detection all use `TaskStateStore`.

`TaskStateStore` uses its SQLite-backed active-task indexes (`pending`, `queued`, `assigned`, `leased`, `running`, `finalizing`) for scheduler hydration, retrieve projection, and metadata-based failure/cleanup. `ModelWorkScheduler` must rebuild its in-memory projection from those indexes on startup; it is not a second durable indexer. Full table scans are not part of the hot path.

## Admission and scheduling

Async endpoints that require model-runtime scheduling go through `ModelWorkScheduler`:
- API routes first create or ensure the task in `TaskStateStore` via `TaskFutureService`.
- The route appends a `ModelWorkItem` to the detached `ModelWorkScheduler` actor (`mint_model_work_scheduler` by default).
- `ModelWorkScheduler` keeps the hot domain backlog, per-replica subqueues, leases, and fairness state in memory.
- `ModelActorSupervisor` observes active scheduler domains and reconciles the matching desired runtime actors from config and placement JSON. A queued training domain can therefore create the runtime needed to claim it.
- Runtime actors claim from their scheduler-owned subqueue. Claiming is independent of `retrieve_future`; result polling reads `TaskStateStore`.
- Scheduler leases must include `attempt_id` and `scheduler_epoch`. `ModelRuntimeActor` owns terminal commit to `TaskStateStore` and lease completion/failure. Route-level `_do_*` functions may still use `TaskFutureService.async_resolve/async_fail` as an executor-local completion signal, but those calls are buffered while running under a model-work execution context and do not write terminal state directly. There is no scheduler-work fallback that writes terminal state through the facade.

On admission failure, the API must return HTTP 429 with a structured overload reason. V1 does not enforce a hard active-task cap; add one only if the active-task index becomes a measured bottleneck.

Pending tasks may wait for `ModelActorSupervisor` to create/register the desired
runtime actor. This must keep Mint async semantics: the client receives a
`request_id`, `retrieve_future` returns HTTP 408 while the task is pending, and
the task eventually becomes `DONE`, `FAILED`, `EXPIRED`, `CANCELLED`, or
forgotten after retention.

TTL facts:

- `MINT_RETRIEVE_FUTURE_HOT_TTL_S` defaults to 300s and only controls the
  process-local retrieve hot cache.
- `MINT_RETRIEVE_FUTURE_GRACE_S` defaults to 600s for cached retrieve
  replay.
- Scheduler owner/lease TTLs default to 30s and only protect scheduler
  ownership/claim recovery.
- Durable pending/task result/tombstone TTLs are enforced by the async future
  reaper and default to `MINT_TASK_PENDING_TTL_S=86400`,
  `MINT_TASK_RESULT_TTL_S=86400`, and
  `MINT_TASK_TOMBSTONE_TTL_S=604800`.

## Request-path async rules

The request path uses native async Ray integration on hot control-plane operations:

- Routes await Ray refs directly through async helpers instead of calling blocking `ray.get(...)`.
- Request paths do not call `init_ray()` or attempt reconnection. Startup owns Ray initialization.
- Startup may check detached-actor availability, but it must not create
  request-path stores or treat handle warming as a bootstrap responsibility.
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

`MaintenanceCronActor` owns periodic cleanup. `TaskFutureService.async_reap()`
is the facade, but task/result retention policy belongs in `TaskStateStore` and
payload deletion is performed through the in-process `TaskPayloadStore` helper.

Reaper contract:

- Each reaper tick runs in this fixed order: expire active pending work, evict
  terminal result payloads, then delete old tombstones. Each phase has its own
  batch limit; the default is 1000 rows per phase per loop.
- Expire only `pending`, `queued`, and `assigned` tasks older than
  `MINT_TASK_PENDING_TTL_S`.
- Do not expire `leased`, `running`, or `finalizing` by pending TTL; scheduler
  lease expiry/requeue/failure owns those states.
- Result TTL is counted from `done_at` or `failed_at`; fall back to
  `updated_at` only for older rows without terminal timestamps.
- After result TTL, delete the payload file and mark metadata with
  `payload_evicted_at`, but keep a terminal tombstone.
- If payload deletion fails, do not mark `payload_evicted_at`; record the reaper
  error and retry on a later loop.
- Active rows may contain staged payload pointers. The reaper must treat model
  work `finalizing` staged paths as referenced until the finalizing lease expires.
  If an active row is requeued or restaged, the previous staged path is retained
  in task metadata as an abandoned staged payload so GC can classify it from
  `TaskStateStore` rather than from an unowned filesystem scan. The reaper
  deletes GC-eligible staged/abandoned payload files separately from terminal
  result payload eviction. V1 uses `MINT_TASK_RESULT_TTL_S` as the safety window
  after finalizing expiry or abandoned-path update time; a dedicated staged GC
  TTL can be added later if that coupling becomes too coarse.
- `staged_payload_checksum` and `staged_payload_size_bytes` are schema-reserved
  for future recovery. V1 GC attribution does not populate or depend on them.
- `retrieve_future` for a terminal row whose payload was evicted returns a
  stable `{"error": "Known terminal future evicted", ...}` payload, not HTTP
  404 and not pending.
- After `MINT_TASK_TOMBSTONE_TTL_S`, delete task rows, events, and any remaining
  payload residue.
- Future payload reaper metrics are separate from checkpoint/artifact reaper
  metrics.

Future reaper metrics:

- `mint_task_future_reaper_rows_total{action="expire_pending|evict_payload|gc_staged_payload|delete_tombstone"}`
- `mint_task_future_payload_evict_errors_total`

## Detached actor hygiene

Detached actors do not hot-reload. Changes to `TaskStateStore`, `ModelWorkScheduler`, `ModelRuntimeActor`, or `MaintenanceCronActor` require killing the matching detached actors in the target namespace before restart.
