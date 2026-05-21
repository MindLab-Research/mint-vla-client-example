# Usage billing storage: durable outbox and PG sink

## Scope

This note documents the target usage billing design in `mint_server`:

- Keep chargeable usage tied to successful business work.
- Avoid route-level scattered PG writes by recording billing facts at the
  terminal task commit/finalization boundary.
- Persist billing events to SQLite first, colocated with `TaskStateStore`
  task state, then flush them to PostgreSQL from periodic maintenance.
- Preserve business availability when PostgreSQL is unavailable; PG outages
  should accumulate durable local outbox rows rather than fail completed work.

## Anchors and ownership

Business executors produce usage observations; they do not write billing rows
directly. The canonical billing write point is the first terminal commit for a
chargeable task.

Examples of usage observations:

- `sampling._do_sample`: prefill and generation token usage.
- `sampling._do_compute_logprobs`: prefill token usage.
- `training._do_forward`
- `training._do_forward_backward`
- `training._do_train_step`
- checkpoint/storage operations that produce chargeable artifacts.

V1 bills only completely successful work:

- `DONE`: bill usage observations attached to the successful result.
- `FAILED`, `CANCELLED`, `EXPIRED`, and pending timeout cleanup: do not bill.
- Partial work that produced tokens or consumed compute before a terminal
  failure is not billed in V1. Add a separate partial-usage design before
  charging failed work.

Async future paths must attach usage observations to task finalization metadata
or return them to the scheduler/runtime finalization layer. `TaskStateStore`
then converts those observations into durable `billing_outbox` rows in the same
SQLite transaction that commits terminal task state/result metadata.

Synchronous HTTP paths, including OpenAI-compatible responses, should use the
same billing outbox append path without creating retrieveable synthetic futures:
`TaskStateStore.append_billing_outbox(events, source="sync_http")`. They must
not bypass the outbox with ad hoc background PG writes. OpenAI-compatible
routes append to the local outbox before returning a successful response.

Requests without a billing context are not billed. Production paths should
surface this as an operational signal:

- Gateway/account/apikey context present: emit billing observations.
- Missing billing context: skip billing and increment
  `mint_billing_observation_skipped_total{reason="missing_billing_context"}`.
- `disabled`/`noop` usage modes may skip without creating outbox rows.

Request id policy:

- Tinker async paths use the gateway billing request id when available; otherwise
  they use the internal future request id.
- Synchronous OpenAI-compatible paths use the gateway request id when available;
  otherwise they use their generated request id.
- A single logical request can emit multiple events with the same `request_id`
  and different dimensions, for example sampling `prefill` and `sample`.
- Retries of the same logical request should reuse the billing request id for
  idempotency. New user requests must use a new request id.

`BillingObservation` is the business-fact schema passed to the terminal commit
or sync append path. It is intentionally separate from the PostgreSQL sink event:

```python
BillingObservation(
    account_id: str,
    apikey_id: str,
    request_id: str,
    charge_item: str,
    quantity: int,
    unit: str,
    model: str | None,
    route: str,
    dimension: str,
    metadata: dict[str, str | int | float | bool | None],
    observed_at: float,
)
```

The outbox adapter converts observations into `UsageEvent` rows by building a
stable label from bounded fields such as `model`, `route`, `dimension`, and
`unit`, then deriving `event_id`.

Do not expose billing outbox state through task lifecycle APIs:

- Scheduler does not read `billing_outbox`.
- `retrieve_future` does not read `billing_outbox`.
- Task lifecycle methods do not expose PostgreSQL concepts.
- `MaintenanceCronActor` only calls billing flush methods; it does not own task
  lifecycle decisions.
- `ModelActorSupervisor` owns the lifecycle of `MaintenanceCronActor`; the API
  server must not implicitly create cron actors during lifespan or request
  handling.

Each billing outbox row contains one normalized `UsageEvent` with:

- `request_id`
- `charge_item`
- `quantity`
- `account_id`, `apikey_id`
- `label`
- stable `event_id`

## Storage abstraction

`mint_server/usage_store.py` defines the PostgreSQL sink for finalized billing
events:

- `UsageStore` protocol (async write/query/summary/health)
- `PostgresUsageStore` (asyncpg pool + idempotent PG writes)

`TaskStateStore` owns the local durable outbox:

- `billing_outbox` rows are written before or during terminal task commit.
- Outbox rows are deleted from SQLite only after the corresponding PG insert
  transaction succeeds.
- The outbox is not the online reporting source of truth after flush;
  `billing.usage_event` remains the source for billing queries and summaries.
- `MaintenanceCronActor` owns the periodic flush loop for now. A dedicated
  `BillingActor` can be introduced later if billing flush throughput or
  isolation requires it, but it must remain a consumer, not the source of truth.
- API startup only checks PostgreSQL health to set degraded state. A PG outage
  must not block API startup because business success paths can still write
  durable local outbox rows.

Minimal `billing_outbox` schema:

- `outbox_id INTEGER PRIMARY KEY`
- `event_id TEXT UNIQUE`
- `event_json TEXT`
- `status TEXT` (`pending`, `flushing`, or `failed`)
- `claim_id TEXT`
- `claimed_until REAL`
- `attempt_count INTEGER`
- `last_error TEXT`
- `created_at REAL`
- `updated_at REAL`

## Concurrency and idempotency

Multi-replica concurrency is handled by PostgreSQL constraints:

- `event_id = uuid5(account_id, apikey_id, request_id, charge_item, label)`
- `billing.usage_event` has a unique index on `event_id`
- insert SQL uses `ON CONFLICT (event_id) DO NOTHING`

Result:

- Retry/duplicate delivery for the same logical usage event does not over-bill.
- Multi-row billing for a single request (for example sampling prefill + sample) is written in one PG transaction.
- `quantity` is not part of the idempotency key. Reusing the same logical
  event id with a different quantity is treated as a duplicate, not an update.
- A PostgreSQL conflict is a successful flush outcome. It commonly means PG was
  committed but SQLite deletion failed, or a retry/replay saw an already flushed
  event. The flusher should count it and delete the matching SQLite outbox row.
- SQLite-side duplicate `event_id` rows with conflicting quantities or payloads
  are an anomaly. Keep the earliest event, skip conflicting duplicates, and emit
  `mint_billing_outbox_conflict_total`.

## Flush state machine

`MaintenanceCronActor` periodically flushes `billing_outbox`:

1. Atomically claim up to N rows whose status is `pending`, or `flushing` with
   an expired `claimed_until`.
2. Write the claimed events to PostgreSQL in one transaction.
3. If the PG transaction succeeds, delete all claimed rows from SQLite,
   including rows that hit `ON CONFLICT (event_id) DO NOTHING`.
4. If PG fails transiently, release the claim or keep it retryable with an
   updated `attempt_count`, `last_error`, and retry time.
5. If PG fails permanently, mark rows `failed`, keep them for inspection/TTL, and
   degrade internal health.

Permanent PG errors include SQLSTATE class `23*` integrity/schema constraint
failures, unsupported charge items, and schema mismatch. Network failures,
timeouts, connection reset, and temporary server errors are transient.

## Failure semantics

- Normal successful terminal commits should write billing outbox rows in the
  same SQLite transaction as terminal task metadata.
- SQLite outbox write failure does not have to fail business work in V1. SQLite
  is expected to be local and reliable; failures must be explicit underbilling
  signals: write `billing_status="dropped"` plus error details to task metadata
  when possible, log the error, and increment
  `mint_billing_outbox_write_errors_total`.
- PostgreSQL unavailable: business work continues, outbox rows accumulate, and
  internal health is degraded. Startup PG health-check failure is also exposed
  as internal degraded state, not as a startup hard failure.
- PostgreSQL write succeeds but SQLite delete/mark-flushed fails: retry is safe
  because PG writes are idempotent by `event_id`.
- Outbox capacity should be monitored. If it grows beyond an operational
  threshold, admission may throttle or fail closed to protect disk and billing
  integrity.
- V1 health policy is degrade-only for backlog. Use env-controlled thresholds:
  `MINT_BILLING_OUTBOX_DEGRADED_ROWS=10000` and
  `MINT_BILLING_OUTBOX_DEGRADED_AGE_S=900`. Do not throttle until a separate
  admission policy is designed.
  `/api/v1/internal/healthz` also degrades when `billing_outbox` has failed rows
  or permanent flush errors.

Minimum metrics:

- `mint_billing_outbox_rows{status}`
- `mint_billing_outbox_oldest_age_s{status}`
- `mint_billing_outbox_flush_attempts_total{result="success|transient_error|permanent_error"}`
- `mint_billing_outbox_events_total{result="inserted|conflict|failed"}`
- `mint_billing_outbox_write_errors_total`
- `mint_billing_observation_skipped_total{reason}`

Do not use high-cardinality labels such as `request_id`, `event_id`, or raw
model names on billing metrics. `route`, `charge_item`, and `dimension` are
allowed when bounded.

## Operational modes

Configured by `MINT_USAGE_BACKEND`:

- `postgres`: durable SQLite outbox plus PostgreSQL sink.
- `disabled`/`noop`: allowed only for local development or explicit tests.

## Follow-ups

- Initial migration covers sampling/logprob and standard training success paths.
- Mint-only VLA/action/checkpoint routes are chargeable but are not part of the
  initial migration. Track them in GitHub issue #630 as remaining billing
  refactor scope and add billing observations for
  `/api/v1/mint/action_sessions/{action_session_id}/act`,
  `/api/v1/mint/vla/train_step`,
  `/api/v1/mint/checkpoints/interpolate`, and related Mint-only workload paths.
