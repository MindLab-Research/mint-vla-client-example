# Async futures (Mint polling protocol)

Many endpoints return `{"request_id": "<uuid>"}` immediately and complete the work asynchronously. The polling surface is the Tinker contract and must be preserved.

## Retrieve semantics (`POST /api/v1/retrieve_future`)

Implementation: `mint_server/routes/futures.py` backed by `TaskFutureService` in `mint_server/backend/task_state_store.py`.

- HTTP 408: `PENDING` (client should retry)
- HTTP 200 with `{"error": ...}`: terminal non-success (for example `FAILED`, `EXPIRED`, `RETRIEVED`)
- HTTP 200 with result payload: `DONE`
- HTTP 404: unknown `request_id` (server has forgotten it)

Pending retrieval uses bounded long-polling for local futures. When a local
future is still pending, the server keeps the HTTP request open until the task
becomes terminal or `MINT_RETRIEVE_FUTURE_WAIT_TIMEOUT_S` elapses. The default
wait timeout is 20s. If the timeout elapses with no terminal state, the response
remains the existing SDK-compatible HTTP 408 pending shape. The wait is native
to the detached `TaskStateStore` actor: API workers register a bounded waiter
through `TaskFutureService`, and `TaskStateStore` wakes the waiter when that
`request_id` changes status or metadata. API workers must not implement their
own repeated status polling loop for local futures.

This is a load-shedding and latency optimization, not a different future state.
It reduces client-side high-frequency polling while preserving the Tinker/Mint
polling contract. Gateway-routed futures still use the upstream retrieve
response plus the local pending throttle, because the router is not the source
of truth for the upstream future state.

Important detail: `TaskStateStore` is the single durable source of truth for task state, result metadata, and terminal future indexes. After returning `DONE` or `FAILED`, the server marks the `request_id` as `RETRIEVED` while retaining terminal metadata and payload pointers. A second `retrieve_future` is served idempotently from `TaskStateStore` when the payload or error is still retained; if the payload has been removed but the terminal task record remains, the route returns `{"error": "Known terminal future evicted", ...}` rather than treating the request as unknown.

## Why futures exist in Mint

Most work runs on Ray GPU actors and can exceed typical HTTP request lifetimes. The futures protocol keeps the HTTP surface stable and matches the Mint SDK client contract. Do not silently change status codes (for example, 408 to 202) or switch to streaming without updating the client contract.

## Where futures live

`TaskFutureService` is an in-process facade. Durable future state lives in a
`FutureStateStore` RocksDB component owned by the existing detached
`TaskStateStore` actor. There must not be a separate `FutureStateStore` Ray
actor or a separate future-state detached lifecycle. Session/index/billing
metadata also remains in the same `TaskStateStore` actor. Result payloads are
written through the in-process `TaskPayloadStore` filesystem helper.
`TaskPayloadStore` is not a Ray actor and has no lifecycle to reconcile.

`FutureStateStore` owns the high-frequency future path:

- `request_id -> status`
- request metadata used by scheduling, retrieval, cleanup, and failure fanout
- scheduler assignment/claim/finalization fields
- terminal result/error payload pointers
- staged and abandoned payload pointers for GC
- active and terminal indexes used by retrieve, scheduler hydration, and reaper

The production implementation is a persistent RocksDB-backed KV store opened
inside `TaskStateStore`. This is not a cache of SQLite state: for futures, the
`FutureStateStore` component is the source of truth. SQLite `TaskStateStore`
must not be updated on the future hot path. New writes go through
`TaskStateStore` into its `FutureStateStore` component.

The facade preserves the old Tinker future methods (`async_resolve`,
`async_fail`, `async_get_status`, etc.) while routing future state through
`FutureStateStore`. Completed result payloads use a staged commit protocol:
first `TaskStateStore` records the expected payload path in its future-state
component, then the payload store helper atomically publishes the vePFS JSON
file, then `TaskStateStore` commits the terminal status, checksum, size, and
result pointer. The existing `TaskStateStore` actor is the lifecycle owner for
these writes, but the KV helper must not use a single global lock for the
retrieve/scheduler hot path. Future writes use per-`request_id` locks only
while performing read-modify-write state transitions; retrieve/status reads are
plain point lookups. Future records keep explicit
status/domain/metadata/lease/result/staged/created/updated indexes. Scheduler
hydration and reapers use those indexes rather than full task scans. If the KV
backend exposes batch writes, the implementation should use them for the record
plus indexes.
Model-work finalization records the lease identity and
`finalizing_until`; direct in-process future resolution records the staged path
while leaving the task pending until terminal commit. This is primarily for GC
correctness: every non-temporary payload path is attributable to a future row
even if the process crashes between file publish and terminal metadata commit.

Direct in-process future resolution always uses a fresh `future__<uuid>` staged path, even if request metadata contains a model-work attempt id. Model-work attempt ids are valid only on the scheduler lease finalization path.

There is no separate future replay index or detached future actor. Retrieve
hot-cache entries are process-local accelerators only; restart recovery,
terminal replay, and payload-evicted detection all use `TaskStateStore`.

`FutureStateStore` uses `request_id` point lookups for retrieve and keeps
active-task indexes (`pending`, `queued`, `assigned`, `leased`, `running`,
`finalizing`) for scheduler hydration. Reapers use terminal-result,
lease-expiry, staged-payload, metadata, and updated indexes. Retrieve and task
status checks must never do full-table or full-keyspace scans.
`ModelWorkScheduler` must rebuild its in-memory projection from the store on
startup; it is not a second durable indexer.

## Admission and scheduling

Async endpoints that require model-runtime scheduling go through `ModelWorkScheduler`:
- API routes first create or ensure the task in `TaskStateStore` future state via `TaskFutureService`.
- The route appends a `ModelWorkItem` to the detached `ModelWorkScheduler` actor (`mint_model_work_scheduler` by default).
- `ModelWorkScheduler` keeps the hot domain backlog, per-replica subqueues, leases, and fairness state in memory.
- `ModelActorSupervisor` observes active scheduler domains and reconciles the matching desired runtime actors from config and placement JSON. A queued training domain can therefore create the runtime needed to claim it.
- Runtime actors claim from their scheduler-owned subqueue. Claiming is independent of `retrieve_future`; result polling reads `TaskStateStore` future state.
- Scheduler backlog and replica subqueues are rebuildable indexes over `TaskStateStore`, not durable authority. If a queue head no longer matches its durable task state during claim, the scheduler must reconcile that stale index entry instead of blocking later work in the same replica queue.
- Scheduler leases must include `attempt_id` and `scheduler_epoch`. `ModelEngineHost` owns terminal commit to `TaskStateStore` future state and lease completion/failure. Route-level `_do_*` functions may still use `TaskFutureService.async_resolve/async_fail` as an executor-local completion signal, but those calls are buffered while running under a model-work execution context and do not write terminal state directly. There is no scheduler-work fallback that writes terminal state through the facade.

On admission failure, the API must return HTTP 429 with a structured overload reason. V1 does not enforce a hard active-task cap; add one only if the active-task index becomes a measured bottleneck.

Pending tasks may wait for `ModelActorSupervisor` to create/register the desired
runtime actor. This must keep Mint async semantics: the client receives a
`request_id`, `retrieve_future` may wait on the server for a bounded interval
and returns HTTP 408 while the task is still pending, and the task eventually
becomes `DONE`, `FAILED`, `EXPIRED`, `CANCELLED`, or forgotten after retention.

TTL facts:

- `MINT_RETRIEVE_FUTURE_HOT_TTL_S` defaults to 300s and only controls the
  process-local retrieve hot cache.
- `MINT_RETRIEVE_FUTURE_GRACE_S` defaults to 600s for cached retrieve
  replay.
- `MINT_RETRIEVE_FUTURE_MIN_POLL_S` defaults to 1s and controls both the
  recommended pending retry interval and the local pending-throttle window
  used when long-polling is disabled.
- `MINT_RETRIEVE_FUTURE_WAIT_TIMEOUT_S` defaults to 20s and bounds how long a
  local `retrieve_future` request may wait for a pending task to become terminal
  before returning HTTP 408.
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
- `execution_serial_key` / `ordering_key` (uses
  `"training_session:{model_id}"` for model-bound training work)

Tagged requests are grouped by scheduler domain and assigned into runtime-owned subqueues. Scheduling semantics:

- Same session preserves FIFO order.
- Active leases are mutually exclusive by `ordering_key`, so the scheduler
  enforces same-session serialization for both sampling
  (`"session:{session_id}"`) and model-bound training
  (`"training_session:{model_id}"`).
- Affinity is the scheduler's sticky placement primitive. Work with the same
  affinity group prefers the same runtime replica when that replica is
  claimable, but it must still respect lease and ordering constraints.
- Legacy API-work-queue knobs such as `MINT_SCHEDULER_FAIRNESS`,
  `MINT_SCHEDULER_MAX_CONSECUTIVE`, `MINT_SCHEDULER_STARVATION_S`, and
  `MINT_SCHEDULER_COALESCE_MS` may remain in old env files or metadata for
  compatibility/historical diagnostics. They are not the authoritative
  scheduling algorithm in the `ModelWorkScheduler` architecture.
- There is no follow-up hold window anymore. A session only keeps the domain lease while its claimed request is still live; stale consumer generations release those leases during restart reconciliation.

This is a deliberate tradeoff: global strict FIFO across sessions is relaxed for
these tagged training ops to reduce cross-session thrash, while preserving
per-session ordering through scheduler leases.

## Reaping and cleanup

`MaintenanceCronActor` owns periodic cleanup. `TaskFutureService.async_reap()`
is the facade, but task/result retention policy belongs in `TaskStateStore`
future state and payload deletion is performed through the in-process
`TaskPayloadStore` helper.

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
  in future metadata as an abandoned staged payload so GC can classify it from
  `TaskStateStore` future state rather than from an unowned filesystem scan. The reaper
  deletes GC-eligible staged/abandoned payload files separately from terminal
  result payload eviction. V1 uses `MINT_TASK_RESULT_TTL_S` as the safety window
  after finalizing expiry or abandoned-path update time; a dedicated staged GC
  TTL can be added later if that coupling becomes too coarse.
- `staged_payload_checksum` and `staged_payload_size_bytes` are schema-reserved
  for future recovery. V1 GC attribution does not populate or depend on them.
- `retrieve_future` for a terminal row whose payload was evicted returns a
  stable `{"error": "Known terminal future evicted", ...}` payload, not HTTP
  404 and not pending.
- After `MINT_TASK_TOMBSTONE_TTL_S`, delete future records, events, and any
  remaining payload residue.
- Future payload reaper metrics are separate from checkpoint/artifact reaper
  metrics.

Future reaper metrics:

- `mint_task_future_reaper_rows_total{action="expire_pending|evict_payload|gc_staged_payload|delete_tombstone"}`
- `mint_task_future_payload_evict_errors_total`

## Detached actor hygiene

Detached actors do not hot-reload. Changes to `TaskStateStore`,
`ModelEngineHost`, or persistent storage code still require the matching
owner-specific restart or reconcile path. `ModelWorkScheduler` publishes
`code_identity`; the supervisor dependency ensure path may recreate a stale
scheduler via `stats(create_if_missing=True)`, while API request paths only
validate identity and fail fast. `MaintenanceCronActor` follows the same
owner-managed lifecycle under the supervisor. Changing the future-state KV
component requires restarting `TaskStateStore`; there is no separate
future-state actor to kill.

## Sampling backpressure policy

Local sampling executor backpressure is opt-in client backpressure, not the
normal scheduler admission path. `/api/v1/asample` and synchronous `sample_once`
may return HTTP 429 when the caller sends `X-Tinker-Sampling-Backpressure: 1`
and the local in-process sampling executor is already at
`MINT_MAX_INFLIGHT_SAMPLE_TASKS`. This local path is only a compatibility
pressure signal for callers that explicitly opted in.

The normal `/api/v1/asample` admission policy should be durable inflight
admission. Headerless requests should enter the model-work scheduler and receive
a future unless the durable system-wide inflight commitment exceeds configured
limits. Ordinary executor saturation is a queueing condition, not a rejection
condition.

V1 durable inflight admission is count-based. A task counts as inflight while it
is non-terminal:

- queued
- assigned
- leased
- running
- finalizing

Terminal statuses do not count:

- done
- failed
- cancelled
- expired
- retrieved
- tombstoned or forgotten after retention

Every durable sampling future must carry bounded admission metadata:

- `principal`: stable caller identity from the trusted auth boundary, such as
  `apikey:<id>`, `internal:<service>`, or `anonymous`
- `domain_key`: the same scheduler work domain used for model-work scheduling

Admission keeps two rebuildable counter views derived from `TaskStateStore`
future state:

- `inflight_by_domain[domain_key]`
- `inflight_by_principal_domain[(principal, domain_key)]`

Counters are incremented only after durable enqueue succeeds and decremented
only after a terminal state is persisted. On scheduler or admission actor
restart, the projection is rebuilt from the active future indexes in
`TaskStateStore`; admission must not full-scan RocksDB per request.

Initial V1 limits are intentionally generous operator defaults:

- `MINT_SAMPLING_MAX_INFLIGHT_PER_PRINCIPAL_DOMAIN=1024`
- `MINT_SAMPLING_MAX_INFLIGHT_PER_DOMAIN=10240`
- `MINT_SAMPLING_MAX_INFLIGHT_TOKENS_PER_PRINCIPAL_DOMAIN=0` (disabled)
- `MINT_SAMPLING_MAX_INFLIGHT_TOKENS_PER_DOMAIN=0` (disabled)
- `MINT_SAMPLING_INFLIGHT_ADMISSION_MODE=observe`

The admission mode should support an observe-first rollout:

- `MINT_SAMPLING_INFLIGHT_ADMISSION_MODE=off`: disabled
- `MINT_SAMPLING_INFLIGHT_ADMISSION_MODE=observe`: record gauges and
  would-reject counters, but still enqueue
- `MINT_SAMPLING_INFLIGHT_ADMISSION_MODE=enforce`: return structured HTTP 429
  when a configured limit is exceeded

In `observe` and `enforce`, emit low-cardinality default metrics:

- `mint_sampling_inflight_by_domain{domain_key}`
- `mint_sampling_inflight_principal_domain_max{domain_key}`
- `mint_sampling_inflight_tokens_by_domain{domain_key}`
- `mint_sampling_inflight_principal_domain_token_max{domain_key}`
- `mint_sampling_admission_would_reject_total{reason,domain_key}`
- `mint_sampling_admission_reject_total{reason,domain_key}`

Do not label default metrics by raw principal. If operators need exact
per-principal drilldown, add a separate explicitly bounded or hashed diagnostic
signal. Per-principal detail otherwise belongs in structured logs or sampled
traces, not default fleet metrics.

Structured 429 responses from durable inflight admission should be retryable and
machine-readable:

```json
{
  "error": "sampling_backpressure",
  "reason": "domain_inflight_limit_exceeded",
  "domain": "qwen3-4b-sampling",
  "principal": "apikey:<redacted>",
  "current": 10241,
  "limit": 10240,
  "retry_after_s": 5
}
```

Allowed durable inflight rejection reasons:

- `principal_domain_inflight_limit_exceeded`
- `domain_inflight_limit_exceeded`
- `principal_domain_token_budget_exceeded`
- `domain_token_budget_exceeded`

The legacy local executor 429 body remains plain and retryable:
`{"detail": "Sampling backpressure: server overloaded"}`. Server-side tests
should distinguish the two paths: local executor saturation only rejects
opt-in callers, while durable inflight admission may reject headerless callers
once `MINT_SAMPLING_INFLIGHT_ADMISSION_MODE=enforce`.
