# Observability Practices

How to write observable code in MinT. Read this once; refer back when
reviewing or writing new subsystems.

## Three Signals, Three Questions

| Signal | Answers | When to reach for it |
|--------|---------|----------------------|
| **Trace** | "Where did the time go?" | A request is slow or fails and you need to find which hop. |
| **Metric** | "Is this systemic?" | You need to know if one request is an outlier or a pattern. |
| **Log** | "What happened?" | You need human-readable detail at a specific decision point. |

Don't use logs for timing (use traces). Don't use traces for fleet health
(use metrics). Don't use metrics for per-request detail (use traces or logs).

## Traceparent: The One Carrier

W3C `traceparent` is the sole trace-context carrier across process
boundaries. If an operation may outlive the HTTP request, you must propagate
it.

**Boundaries that always need propagation:**
- Queue enqueue → dequeue (traceparent goes into the task payload)
- Ray actor `.remote()` calls (traceparent goes into kwargs)
- Gateway forward (traceparent goes into HTTP headers)
- Webhook callbacks

**How to propagate:**
1. Producer: `get_current_traceparent()` — captures the active trace context.
2. Pass it through: function arg, dict field, or HTTP header.
3. Consumer: restore it before doing work — use `start_as_current_span_from_traceparent()`
   for a span, or `restore_trace_id_from_traceparent()` for log correlation only.

If you forget step 1, the consumer starts a new trace and the chain is broken.
If you forget step 3, the consumer's work is invisible in the original trace.

**Test:** After adding a new boundary, check the trace UI — the consumer span
should have the same `trace_id` as the HTTP ingress span. If not, propagation
is broken.

## Span Granularity

**Span at state transitions, not at hot loops.**

Good span boundaries:
- HTTP ingress/egress
- Queue enqueue/dequeue
- Ray actor call → result
- Model lifecycle: create, load, sample, forward, save
- Retry, fallback, remediation

Bad span boundaries:
- Per-token, per-item, per-batch-element
- Inner loops that execute thousands of times per request
- Synchronous in-process function calls (use a log instead)

For high-frequency signals inside a span, use `record_span_event_otel()` —
it annotates the current span without creating a new one.

## Log Format

**structlog structured calls. Never f-strings. Never `%s` interpolation. Never `[module]` prefix.**

```python
# Do this — event name first, then structured key=value kwargs
logger.info("task_admitted", op=op, status="admitted", duration_ms=ms)
logger.warning("request_failed", error_type=type(e).__name__, error=str(e))

# Not this — f-string (structlog can't extract structured fields)
logger.info(f"request_id={request_id} status=admitted")

# Not this — %s interpolation (key=value buried in format string, not queryable)
logger.info("request_id=%s status=admitted", request_id)

# Not this — [module] prefix (component is auto-injected)
logger.info("[admission] request_id=%s status=admitted", ...)
```

**Why:** structlog's processor chain auto-injects `component` (from `logger.name`),
`request_id`, `trace_id`, `hostname` into every log event as structured fields.
The first positional argument is the event name (a short snake_case identifier).
Business data goes in keyword arguments — they become queryable JSON fields in
VictoriaLogs, not text buried inside a format string.

**Convention:**
- Use `structlog.get_logger(__name__)`, not `logging.getLogger(__name__)`.
- First arg: concise snake_case event name (`task_admitted`, `lora_loaded`, `nan_detected`).
- Business data as `key=value` kwargs: `op=`, `status=`, `duration_ms=`, `error_type=`.
- Do not pass `request_id` or `trace_id` as kwargs — they are auto-injected.
- `error_type` via `type(e).__name__`, never `str(e)` alone.

**Log levels:**
- `INFO`: lifecycle transitions and successful completions with timing.
- `WARNING`: degraded but recoverable — retries, fallback, queue pressure.
- `ERROR`: failure or correctness risk. One per failure at the owning boundary,
  not one per retry attempt.

## What to Log at Each E2E Hop

Every request passes through: route → admission → enqueue → claim → execute
→ finalize → resolve. At minimum, each hop should log once:

| Hop | What to log |
|-----|-------------|
| Admission | `request_id`, `op`, decision (admitted/rejected), why, duration |
| Enqueue | `request_id`, `op`, `domain_key`, backlog depth |
| Claim → Execute | `request_id`, `op`, `actor_name`, queue wait time |
| Execute result | `request_id`, `op`, outcome (success/failure), exec duration, error type if failed |
| Resolve | `request_id`, status (done/failed/retrieved) |

If a hop has zero logging, you have a blind spot. When debugging, you should
be able to `grep request_id=xxx` and see the request's full lifecycle.

## Metric Design

**Metrics answer "is this systemic?" — they must be aggregatable.**

Cardinality is the enemy of aggregation. A metric label with unbounded values
(request_id, trace_id, error messages, file paths) makes the metric useless for
fleet dashboards and expensive to store.

**Bounded labels (good):**
- `op`, `component`, `status`, `reason`
- `base_model`, `actor_name`, `backend`
- `store`, `domain_key` (when domain set is small)

**Unbounded labels (never):**
- `request_id`, `trace_id`
- raw error messages
- file paths, checkpoint paths
- user text

**Metric families that matter:**
- Counters: operation counts, error counts, admission decisions
- Histograms: latency distributions, queue wait, execution duration
- Gauges: queue depth, active sessions, GPU memory (via observable callbacks)

**Adding a metric:** create the OTel instrument in `init_otel_logging()`,
add a `record_*_otel()` helper, and call it at the instrumentation site.
The helper should be a no-op when `_OTEL_ENABLED` is False.

## Error Observability

**One failure, one primary log, at the owning boundary.**

The "owning boundary" is the layer that has enough context to say what failed
and what to do next. For a vLLM timeout, that's the inference engine, not the
HTTP route. For a scheduler conflict, that's the scheduler, not the store.

Anti-patterns:
- Logging the same error at every layer as it propagates up.
- Logging inside a retry loop (log once after retries are exhausted).
- Logging `str(e)` without `type(e).__name__` (loses the exception type).

On spans: call `span.record_exception(e)` so the exception appears in the
trace. The `start_as_current_span` and `run_async_with_otel_span` helpers do
this automatically when an exception escapes.

## What Not to Observe

- Secrets, tokens, API keys, auth headers, process environments.
- Full prompts, responses, logits, weight vectors.
- Per-token or per-batch-element spans in hot loops.
- High-cardinality metric labels.
- Identical repeated ERROR lines in retry loops.
- Success at every micro-step (log success at the lifecycle boundary, not
  inside every function call).

## Review Checklist

- [ ] If the change crosses a process/queue boundary, is `traceparent` propagated?
- [ ] Are spans at state transitions, not in hot loops?
- [ ] Do logs use structlog structured calls (event name + kwargs, no f-strings/%s/[module])?
- [ ] Is `request_id` the first key in every log line?
- [ ] Are error logs at the owning boundary, with `error_type`?
- [ ] Are metric labels bounded (no `request_id`, no free-form text)?
- [ ] Is there a `record_*_otel()` helper for any new metric?
- [ ] Does the failure path record the exception on the span?
- [ ] Can you `grep request_id=xxx` and see the request's full lifecycle?
- [ ] Are no secrets, logits, or full prompts in any log line?
