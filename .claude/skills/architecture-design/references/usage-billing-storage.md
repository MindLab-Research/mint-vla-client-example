# Usage billing storage: direct PG with durable outbox

## Scope

This note documents usage billing storage migration in `mint_server`:

- Keep existing billing anchors in sampling/training success paths.
- Replace sync JSONL-only sink with async usage store abstraction.
- Add PostgreSQL backend with per-event idempotent writes for concurrent replicas.
- Preserve successful-request billing records by durably spooling to a local outbox when PG is temporarily unavailable.

## Anchors

Billing anchors remain at request-success points and now persist billing before resolving the async future:

- `sampling._do_sample`: prefill and generation token usage
- `training._do_forward_backward`
- `training._do_train_step`

Each anchor emits one or more `UsageEvent` rows with:

- `request_id`
- `charge_item`
- `quantity`
- `account_id`, `apikey_id`
- `label`

## Storage abstraction

`mint_server/usage_store.py` defines:

- `UsageStore` protocol (async write/query/summary/health)
- `PostgresUsageStore` (asyncpg pool + sqlite durable outbox)

## Concurrency and idempotency

Multi-replica concurrency is handled by PostgreSQL constraints:

- `idempotency_key = (request_id, charge_item, label)`
- insert SQL uses `ON CONFLICT (request_id, charge_item, label) DO NOTHING`

Result:

- Retry/duplicate delivery for the same logical usage event does not over-bill.
- Multi-row billing for a single request (for example sampling prefill + sample) is written in one PG transaction.

## Failure semantics

- Route handlers persist usage before resolving the async future.
- PG write attempts retry locally first.
- If PG remains unavailable, events are synchronously written to a sqlite outbox under `checkpoint_dir/.billing/` and later flushed to PG by a background task.
- The outbox is a retry buffer only; `billing.usage_event` remains the online source of truth.
- Server shutdown attempts a final outbox drain before closing the PG pool.

## Operational modes

Configured by `MINT_USAGE_BACKEND`:

- `postgres`: direct PG mode with durable local outbox fallback
