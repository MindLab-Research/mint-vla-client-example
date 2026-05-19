# HTTP API boundary and internal routes

MinT has three HTTP surfaces. New endpoints must be placed by audience and compatibility contract, not by which Python module currently implements the handler.

## Public Tinker-compatible API: `/api/v1`

`/api/v1` is the user-facing compatibility surface for the Tinker SDK and cookbook.

Contract:
- Keep request/response semantics compatible with upstream Tinker unless an endpoint is explicitly documented as Mint-only.
- Preserve polling behavior such as `retrieve_future` returning pending/ready states in the SDK-compatible shape.
- Do not expose Ray actor inventory, placement groups, scheduler internals, debug dumps, or maintenance controls here.
- Route handlers may call internal backend services, but the HTTP contract should remain task/session/future/checkpoint oriented.

Examples:
- session creation and training/sampling submission
- future retrieval
- user-visible weight and checkpoint operations
- public health checks that do not perform expensive Ray diagnostics and only
  return `{"status":"ready"}` or `{"status":"unhealthy"}`

## Mint user extensions: `/api/v1/mint`

`/api/v1/mint` is user-facing but not Tinker-compatible. It is for Mint-specific features that clients may intentionally depend on.

Contract:
- Keep these APIs stable once shipped to users.
- Document the SDK/client expectation alongside the code.
- Do not use this prefix for operator-only controls.

## Internal control-plane API: `/internal`

`/internal` is for operators, deployment automation, and observability. It has no SDK compatibility guarantee.

Contract:
- Endpoints may be renamed or reshaped with the internal control plane.
- Mutating endpoints must enforce admin access when auth is enabled.
- Read endpoints should be cheap by default; costly refreshes must be explicit query parameters or separate endpoints.
- Internal routes may expose implementation details such as Ray actor names, replica queues, placement state, and task indexes.

Current categories:
- Health and observability: `/api/v1/healthz`, `/api/v1/internal/healthz`, `/internal/admission_stats`.
- Optional debug metrics: `/internal/metrics` when explicitly enabled; the default metrics path is OTel push from node collectors, not Prometheus scraping.
- Scheduler state: `/internal/model_work_scheduler`, `/internal/model_work_scheduler/debug_state`, `/internal/debug/scheduler_decisions`, `/internal/model_work_scheduler/noop`.
- Runtime desired state: `/internal/model_actor_supervisor`.
- Ray cluster diagnostics: `/internal/ray_cluster_health`, `/internal/ray_gcs_metrics`.
- Actor administration: `/internal/actors`, `/internal/actors/kill`.
- Maintenance actor diagnostics: `/internal/maintenance_cron_actor`.
- Usage and checkpoint operator views: `/internal/usage_logs`, `/internal/usage_summary/{account_id}`, `/internal/v1/checkpoints`.

`/api/v1/healthz` is intentionally public and minimal: it checks only the
business control-plane dependencies needed to accept and track work, caches a
successful value for 30s per API worker, and does not expose degraded internal
state.

`/api/v1/internal/healthz` is an internal lightweight health endpoint. It reads
the current `ModelActorSupervisor` summary snapshot and process-local
maintenance-cron/startup degraded markers, then returns a small component
summary. It must not perform per-request fanout to runtime actors and does not
promise scheduler, task, topology, or reaper summaries.

## Actor admin semantics

`/internal/actors` is an admin inventory view. It is not the scheduling source of truth.

Use it for:
- listing live actor inventory entries
- refreshing VLLM/Megatron observability metadata
- finding exact Ray actor names for operational cleanup

Do not use it for:
- deciding which model runtimes should exist
- claiming model work
- deriving task state

Those responsibilities belong to:
- `TaskStateStore`: durable task state and future metadata
- `ModelWorkScheduler`: hot scheduling projection, replica subqueues, and leases
- `ModelActorSupervisor`: desired model runtime reconciliation

`/internal/actors/kill` is an admin mutation endpoint. It may kill backend actors and release their placement groups. Busy actors require explicit force semantics from the request body; callers should prefer scheduler/supervisor reconciliation over manual kills when possible.

## Migration rule

Internal operational APIs must not remain under `/api/v1`. When a route is only used by scripts, runbooks, tests, or operators, move it to `/internal` and update those callers directly. The refactor branch does not keep legacy aliases unless an external compatibility requirement is explicitly stated.
