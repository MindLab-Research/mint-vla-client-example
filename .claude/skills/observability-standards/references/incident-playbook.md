# Incident Playbook (Observability)

## Purpose

This file provides concrete incident patterns and what to emit in traces, logs, and metrics.
Use it to reduce MTTR during production issues.

## Case 1: Queue Enqueue Timeout

Symptom:
- API returns 503 during enqueue.

Trace:
- Parent span: HTTP SERVER span.
- Child span: `queue.enqueue`.
- Attributes:
  - `component=api_work_queue`
  - `op=enqueue`
  - `timeout_ms`
  - `queue_actor`
- On failure:
  - `record_exception` with timeout exception
  - `status=ERROR`

Log:
- `ERROR` summary once per failed request:
  - `event=queue_enqueue_failed`
  - `request_id`, `trace_id`
  - `actor_name`, `timeout_ms`, `elapsed_ms`, `attempt`
  - `error_type`, `error_message`
  - `next_action=check_actor_health`

Metrics:
- Counter: `queue_enqueue_timeout_total{component,op}`
- Gauge/Hist: queue depth/wait time (low-cardinality labels only)

## Case 2: Upstream Dependency 5xx

Symptom:
- Gateway call fails with upstream 5xx.

Trace:
- Child span for outbound HTTP/RPC call.
- Attributes:
  - `peer.service`
  - `http.method`, `http.route`, `http.status_code`
  - `retry_count`
- Error:
  - record exception or surrogate for 5xx

Log:
- `WARN` for retry attempt.
- `ERROR` when final attempt fails:
  - include `upstream_alias`, `status_code`, `retry_count`, `duration_ms`

Metrics:
- Counter: `upstream_errors_total{upstream_alias,status_class}`
- Histogram: upstream latency per route template

## Case 3: Worker OOM / Resource Exhausted

Symptom:
- Training/inference stage fails due to OOM.

Trace:
- Stage span (`trainer.forward_backward` / `sampler.generate`).
- Attributes:
  - `model_name`, `component`, `stage`
  - `batch_size` (if bounded)
- Error classification:
  - `failure_reason=resource_exhausted`

Log:
- `ERROR`:
  - `event=worker_oom`
  - `request_id`, `trace_id`, `job_id`, `actor_name`, `model_name`
  - `error_type`, `error_message`
  - `next_action=reduce_concurrency_or_restart_actor`

Metrics:
- Counter: `worker_failures_total{reason,component,model}`
- Saturation metric: actor occupancy / queue backlog

## Case 4: Slow Request But No Hard Error

Symptom:
- User reports latency spike without explicit errors.

Trace:
- Ensure critical child spans exist (queue wait, dependency calls, compute stage).
- Compare p95/p99 span durations by stage.

Log:
- `WARN` for threshold exceed:
  - `event=request_slow`
  - `elapsed_ms`, `route`, `component_hotspot`

Metrics:
- Histograms for endpoint latency and queue wait.
- Alert on sustained p99 breach, not single spike.

## Anti-Patterns

Do not:
1. Add per-token/per-item spans in hot loops.
2. Put request_id/trace_id into metric labels.
3. Log raw prompts, vectors, logits, or secrets.
4. Emit repeated identical ERROR logs in retry loops.

## Fast Triage Query Checklist

When incident starts, operator should quickly filter by:

1. `request_id` or `trace_id` (single request drill-down)
2. `component + op + error_type` (blast radius)
3. latency percentiles by route/component
4. queue depth and timeout counters

