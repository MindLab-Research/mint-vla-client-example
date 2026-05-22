# Observability Architecture

This reference defines MinT trace, log, and metric design. It is architecture
guidance, not an incident-query runbook.

Primary goal: keep production failures diagnosable within minutes without
making observability part of the request critical path.

## Current Runtime Shape

- MinT uses OpenTelemetry when `OTEL_EXPORTER_OTLP_ENDPOINT` is configured.
- Signals are exported by OTLP push from API workers and Ray actors.
- Exporter setup or export failure must never block service startup or request
  handling.
- API and actor processes use distinct `service.instance.id` values; metrics add
  `mint_instance_id` to separate cumulative counters by process.
- Do not assume Prometheus scrape is the primary source for MinT application
  metrics. Ray/GCS global metrics should be pushed by the head
  `NodeMetricsCollectorActor` from cached snapshots, not collected by dashboard
  refreshes through `/internal/metrics`.
- `NodeMetricsCollectorActor` pushes node-local OS/NVML metrics from every ready
  topology node. The observed Ray head node is also represented as `mint-head`
  for metrics-daemon purposes so it can push Ray live-state, placement-group,
  and GCS bridge health gauges without entering model placement or worker
  provisioning semantics.
- `/internal/metrics` is authenticated, opt-in, and debug/cached-only. Migrated
  dashboards must not require `up{job="mint-internal-metrics"}` for their normal
  runtime data path.

Relevant implementation:
- `mint_server/logging_context.py`
- `mint_server/config.py::otel_env_vars`
- route middleware in `mint_server/app.py`
- node and Ray/GCS metric push in `mint_server/backend/node_metrics_daemon.py`

## Identifier Semantics

- `request_id`: external/business request identifier. Preserve existing API
  compatibility.
- `trace_id`: distributed tracing identifier. It is propagated through W3C
  `traceparent`.
- Keep `request_id` and `trace_id` separate. Do not rename or overload one into
  the other.
- If no incoming trace context exists, generate a valid 32-character lowercase
  non-zero hex trace id.
- Propagate trace context across queue, future, Ray actor, gateway, and webhook
  boundaries whenever the operation may outlive the HTTP request.

## Trace Standards

Trace state transitions and uncertain or expensive boundaries:

- HTTP ingress/egress.
- Queue enqueue/dequeue and wait time.
- Ray actor calls and worker execution boundaries.
- Remote HTTP/RPC/gateway calls.
- Expensive model stages: create, load, sample, forward, forward_backward,
  optim_step, save_state, save_weights_for_sampler.
- Retry, fallback, stale-session, and remediation-related paths.

Avoid per-token, per-item, or hot-loop spans. Prefer span events or metrics for
high-frequency inner-loop signals.

Minimum useful span attributes:
- HTTP: `http.method`, `http.route`, `http.status_code`, `request_id` when
  available.
- Worker or queue stage: `component`, `op`, `model_name` when relevant,
  `attempt`/`retry_count` when relevant, bounded wait or elapsed timing fields.

On failure, call `record_exception` when there is an exception object and set
the span status to error. For framework-converted 5xx responses where no
exception escapes, add an explicit error event or exception surrogate.

## Log Standards

New logs should include stable fields where available:
- `request_id`
- `trace_id`
- `component`
- `op`
- target identifiers such as `model_name`, `actor_name`, `session_id`, or
  `request_type`

Do not assume all existing logs are fully structured. Improve logs at the
failure boundary being changed.

Use levels consistently:
- `INFO`: important lifecycle transitions and successful state changes.
- `WARNING`: degraded but recoverable conditions, retries, fallback, queue
  pressure, slow path.
- `ERROR`: request/job/stage failure or correctness risk. Include sanitized
  `error_type`, target, correlation ids, timing, and a short operator hint when
  it is clear.

Avoid repeated identical `ERROR` lines in retry loops. One failure should have
one primary incident-actionable error at the owning boundary.

Never log secrets, full prompts/responses, logits, vectors, process
environments, raw auth headers, or full `.secrets.env` contents.

## Metrics Standards

Metrics are for fleet health, alerting, and blast-radius checks. Use traces and
logs for per-request detail.

Current HTTP metric names:
- `mint_http_server_requests_total`
- `mint_http_server_errors_total`
- `mint_http_server_request_duration_ms`

Current HTTP attributes emitted by code:
- `http.method`
- `http.route`
- `http.status_code`
- `mint_instance_id`

Note: Victoria/Grafana may normalize attribute names into label names. When
editing dashboards, first inspect live label names rather than assuming
Prometheus-style labels such as `method`, `route`, or `status_code`.

Good metric families:
- request/error counts and endpoint latency
- queue depth, queue wait, admission decision, timeout counters
- actor pool occupancy and actor health
- dependency failure counts and latency

Avoid metric labels with high-cardinality values:
- `request_id`
- `trace_id`
- raw user text
- raw URL with IDs
- free-form error messages
- unbounded checkpoint/session paths

Use bounded labels such as route template, operation, component, model name,
actor type, decision, and failure category.

## Incident Patterns

Queue enqueue timeout:
- Trace: HTTP span plus queue enqueue span.
- Log once at the owning boundary with `request_id`, `trace_id`, queue actor,
  timeout, elapsed time, and `next_action=check_actor_health`.
- Metrics: timeout counter and queue depth/wait signal.

Upstream or gateway 5xx:
- Trace outbound call with peer/upstream alias, route, status, retry count.
- Warn on retry; error on final failure.
- Metrics: upstream error count by bounded alias/status class and latency by
  route template.

Worker OOM or resource exhausted:
- Trace the failing stage span.
- Log `request_id`, `trace_id`, actor/model, operation, error type, and whether
  the actor/session is contaminated or requires clean reload.
- Metrics: worker failure counter by bounded reason/component/model and
  saturation/backlog signals.

Slow request without hard error:
- Use traces to locate the slow stage.
- Use metrics to confirm whether the symptom is isolated or systemic.
- Warn only for thresholded sustained slow paths; avoid alerting on one-off
  queueing under expected load.

## Review Checklist

For observability-related PRs:
- Preserve `request_id` and `trace_id` semantics.
- Propagate `traceparent` across async/worker boundaries touched by the change.
- Keep spans bounded and attached to meaningful state transitions.
- Record errors on spans for thrown exceptions and converted 5xx responses.
- Add or update logs at the owning failure boundary with stable correlation
  fields.
- Keep metrics low-cardinality and actionable.
- Verify dashboard labels against live telemetry when changing Grafana JSON.
- Confirm exporter/logging/metrics failures cannot break business behavior.

Non-goals:
- Do not bundle scheduling, retry, or self-healing behavior into an
  observability-only PR.
- Do not change external API contracts for observability convenience.
