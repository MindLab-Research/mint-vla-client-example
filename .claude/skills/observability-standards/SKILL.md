---
name: observability-standards
description: |
  Observability standards for MinT/mint-server based on OpenTelemetry.

  Use for: trace/log/metric design, request_id/trace_id semantics, context propagation,
  error diagnostics, and OTLP exporter conventions.

  Triggers: "otel", "opentelemetry", "observability", "trace", "metrics", "logging",
  "trace_id", "request_id", "apmplus", "record_exception", "SLO", "incident"
---

# Observability Standards (MinT)

## Goal

Primary objective: **system stability + fast incident triage (low MTTR)**.

Observability design must answer, within minutes:

1. Which request/job failed?
2. Where did it fail (component, step, dependency)?
3. Why did it fail (timeout, resource, bad input, dependency)?
4. Is it isolated or widespread?
5. What should operator do next?

## Canonical Semantics

- `request_id`: external/business request identifier.
- `trace_id`: distributed tracing identifier (OTel context).
- Keep both concepts separate; never rename one into the other.
- Preserve existing API compatibility around `request_id`.

## Global Rules

1. Preserve existing `request_id` behavior unless explicitly requested.
2. Ensure trace/log/metric correlation through consistent context.
3. Prefer W3C `traceparent` extraction.
4. If no incoming trace exists, generate valid trace_id (32 lowercase hex, non-zero).
5. Do not overwrite an already active trace context with ad-hoc headers.
6. Observability changes should not change business behavior.
7. Exporter/setup failure must not block service startup.

## Trace Standards

### What To Trace

Trace only **state transitions and expensive/uncertain boundaries**:

1. HTTP ingress/egress (top-level SERVER span).
2. Queue boundaries (`enqueue`, `dequeue`, wait time).
3. External dependencies (Ray actor call, DB/Redis, remote HTTP/RPC).
4. Expensive compute stages (model load, forward/backward, weight save/load).
5. Retry/fallback paths.

### Depth Control (How Deep)

Keep trace depth intentionally bounded:

1. Default: 1 top-level request span + key child spans for major stages.
2. Avoid per-item/per-token/per-loop spans in hot loops.
3. Prefer span events or metrics for high-frequency inner-loop signals.
4. If a request creates too many spans, collapse low-value layers.

Rule of thumb:

- API request path: keep span count small and stable.
- Training/inference internals: trace stage boundaries, not every iteration detail.

### Required Span Attributes (Minimum)

For HTTP SERVER span:

- `http.method`
- `http.route` (template/path pattern when possible)
- `http.status_code`
- `request_id` (when available)

For worker/executor spans (as relevant):

- `op` / operation name
- `component` (api, queue, task_state_futures, trainer, sampler, gateway)
- `model_name` (if relevant)
- `queue_wait_ms` / `attempt` / `retry_count` (if relevant)

### Error Recording

1. On thrown exception: `span.record_exception(e)` and set `StatusCode.ERROR`.
2. On framework-converted `5xx` response (no exception bubbles): still record an explicit error event/exception surrogate.
3. Include failure reason category where possible (timeout, canceled, resource_exhausted, dependency_unavailable, validation).

### What Not To Put In Trace Attributes

Never include:

1. Huge payloads (vectors, logits, full prompts/responses, large JSON blobs).
2. High-cardinality raw values (raw URL with IDs, user free text, request body).
3. Secrets (`*_KEY`, `*_TOKEN`, `Authorization`, credentials).
4. Near-constant noise fields (for example vector length when always same and not diagnostic).

If needed for debugging, record compact summaries only:

- counts/sizes/buckets/hashes (non-reversible)
- bounded enums

## Logging Standards

Use structured logs; each line should be machine-parseable.

### Required Common Fields

Every log line should carry (when available):

- `timestamp`
- `level`
- `logger`
- `request_id` (or `-`)
- `trace_id` (or `-`)
- `component`
- `op` / action name

For background workers also include identifiers when relevant:

- `worker_idx`, `actor_name`, `job_id`, `model_name`, `session_id`

### Level Definitions

#### `DEBUG`

Use for detailed diagnostics that are noisy and disabled by default in production.

Allowed examples:

- branch decisions
- intermediate counters
- cache hit/miss details

Must not:

- flood hot loops
- include payload/secrets

#### `INFO`

Use for important lifecycle and successful state transitions.

Examples:

- request accepted
- enqueue/dequeue start/finish
- model/session created
- retry succeeded

Goal: reconstruct normal flow at coarse granularity.

#### `WARN`

Use for recoverable abnormal conditions or degraded behavior.

Examples:

- timeout with retry scheduled
- partial fallback
- temporary dependency slow/unavailable

Must include:

- impact scope
- action taken (`retrying`, `fallback=...`, `skipping...`)

#### `ERROR`

Use when request/job/stage failed or data correctness is at risk.

Must include enough info for first-response triage:

1. `error_type`, `error_message` (sanitized), stack trace.
2. Target/subject (`op`, `component`, `model`, `actor`, endpoint).
3. Correlation IDs (`request_id`, `trace_id`, `job_id`, `session_id` if any).
4. Timing context (timeout threshold, elapsed time, attempt number).
5. Immediate operator hint (`next_action`) when clear.

Avoid repetitive duplicates:

- one failure should have one primary ERROR summary per boundary.
- lower layers can log DEBUG/WARN but avoid ERROR storm.

### Logging Do/Don't

Do:

1. Log transitions and decisions.
2. Log with stable field names.
3. Log failure classification and remediation hint.

Don't:

1. Log full payload bodies by default.
2. Log secrets.
3. Emit identical ERROR repeatedly inside retry loop.

## Metrics Standards

Metrics are for **fleet health, trend, alerting**.

### Emit Metrics When

1. Needed for SLI/SLO and alerting.
2. Needed to detect systemic issues quickly (queue pressure, saturation, error rate).
3. Information cannot be reliably/cheaply derived from traces in practice.

### Do Not Emit Metrics When

1. It duplicates trace-only debugging detail with no operational value.
2. Label cardinality would explode.
3. The signal is unstable/noisy and not actionable.

### Recommended Metric Families

1. Availability:
   - request total / error total / error rate
2. Saturation and pressure:
   - queue depth, queue wait time, actor pool occupancy
3. Latency (only where needed for SLO or traces are sampled):
   - top-level endpoint/stage latency histograms
4. Dependency health:
   - timeout count, dependency failure count

### Label Cardinality Rules

Allowed labels (typically):

- `method`, `route`, `status_code`, `op`, `component`, bounded `model`

Avoid labels:

- `request_id`, `trace_id`, raw user/model input, free-form error text

## Context Propagation

1. Ingress middleware binds `trace_id` (and `request_id` if available).
2. Async queue/background payload carries minimal context (`trace_id`, optionally request_id if contract requires).
3. Worker restores context before logging/tracing.
4. Ensure context reset on request boundary to prevent cross-request contamination.

## Incident-First Error Template

When writing `ERROR` logs, ensure line contains:

1. What failed: `event`, `component`, `op`
2. Which unit failed: `request_id`, `trace_id`, `job_id/session_id`, `actor_name`
3. Why failed: `error_type`, categorized reason (`timeout`, `oom`, `dependency`, `validation`)
4. Impact: `status_code` / degraded mode / dropped or retried
5. Timing: `elapsed_ms`, `timeout_ms`, `attempt`
6. Next action hint: `retrying`, `restart_actor`, `check_dependency`, etc.

## OTLP Exporter and Runtime Safety

Environment-driven config:

- `OTEL_EXPORTER_OTLP_ENDPOINT`
- `OTEL_EXPORTER_OTLP_HEADERS`
- `OTEL_EXPORTER_OTLP_INSECURE`
- `OTEL_SERVICE_NAME` (default `mint`)
- `MINT_APMPLUS_APP_KEY` maps to `x-byteapm-appkey` when headers absent

Safety requirements:

1. Missing endpoint/dependency should not crash service.
2. Export failures should degrade gracefully.
3. Observability pipeline should not become request critical path.

## References

- Incident patterns and expected telemetry: `references/incident-playbook.md`

## Review Checklist

For observability-related PRs, verify:

1. Semantics
   - `request_id` and `trace_id` are not mixed.
2. Trace quality
   - key boundaries traced; depth controlled; errors recorded correctly.
3. Logging quality
   - correct levels used; ERROR lines are incident-actionable.
4. Metrics quality
   - only actionable metrics; bounded label cardinality.
5. Propagation
   - context survives async/queue/worker boundaries.
6. Safety
   - observability failures do not break business request flow.
7. Scope discipline
   - no unrelated scheduling/self-healing behavior bundled into observability-only PR.

## Non-goals

- Do not change scheduling/retry/queue self-healing behavior in an observability-only PR.
- Do not alter external API contracts unless explicitly requested.
- Do not add high-cardinality metrics/log attributes for convenience debugging.
