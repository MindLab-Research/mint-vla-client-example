---
name: telemetry-direct-query
description: |
  Direct VictoriaMetrics, VictoriaLogs, and VictoriaTraces querying with a local
  Python helper, HTTPie, or raw curl when `obsh` is unavailable.

  Use for: incident triage from error text, `request_id`, `trace_id`, endpoint,
  or a narrow time window when only raw HTTP access is available.

  Triggers: "victoria", "victoriametrics", "victorialogs", "victoriatraces",
  "telemetry query", "observability query", "promql", "logsql", "trace_id",
  "request_id", "httpie observability", "curl observability"
---

# telemetry-direct-query

Use this skill when `obsh` is unavailable and you must query Victoria directly.
Prefer the local Python helper. It removes most shell quoting errors and keeps
all three backends behind one CLI.

The three query surfaces are:

1. VictoriaLogs: LogSQL search and id-first narrowing
2. VictoriaTraces: Jaeger-compatible trace discovery and trace fetch
3. VictoriaMetrics: PromQL instant and range queries

Current recorded defaults from `/vePFS-Mindverse/user/nolanho/docs/grafana/README.md`:

- `VICTORIA_LOGS_URL=http://192.168.4.70:9428`
- `VICTORIA_TRACES_URL=http://192.168.4.70:10428`
- `VICTORIA_METRICS_URL=http://192.168.4.70:8428`

If the task provides a tunnel or another host, override these env vars instead of
editing command examples inline.

## Hard rules

- Start from the narrowest anchor you already have: exact error text, `request_id`, `trace_id`, endpoint, service, or small time window.
- Logs first for vague incidents. Trace first only when you already have `trace_id`.
- Use metrics to confirm whether an event is isolated or systemic. Metrics do not replace logs.
- Do not assume localhost unless you are explicitly on the observability host or inside a tunnel.
- Keep evidence short: command, time window, stable ids, and one to three proof lines.
- Use `mint_*` metric names by default. If only `tinker_*` appears, treat that as migration debt.

## Setup

```bash
export VICTORIA_LOGS_URL=${VICTORIA_LOGS_URL:-http://192.168.4.70:9428}
export VICTORIA_TRACES_URL=${VICTORIA_TRACES_URL:-http://192.168.4.70:10428}
export VICTORIA_METRICS_URL=${VICTORIA_METRICS_URL:-http://192.168.4.70:8428}
python .claude/skills/telemetry-direct-query/victoria_query.py --help
```

## Primary path: Python helper

Use `.claude/skills/telemetry-direct-query/victoria_query.py` first.
It is stdlib-only and wraps the documented Victoria APIs directly.

Common commands:

```bash
python .claude/skills/telemetry-direct-query/victoria_query.py logs \
  --query 'request_id:sample_12b39ee941342a4fe58b57354688f3c1' \
  --limit 20

python .claude/skills/telemetry-direct-query/victoria_query.py trace-services

python .claude/skills/telemetry-direct-query/victoria_query.py trace-search \
  --service mint \
  --operation 'POST /api/v1/retrieve_future' \
  --lookback 1h \
  --limit 20

python .claude/skills/telemetry-direct-query/victoria_query.py trace-get \
  --trace-id 042397caf580ea9b2d71eb2ea7332f99

python .claude/skills/telemetry-direct-query/victoria_query.py metrics-query \
  --query 'mint_metrics_up'

python .claude/skills/telemetry-direct-query/victoria_query.py metrics-range \
  --query 'rate(mint_http_server_requests_total[5m])' \
  --since 30m \
  --step 60s

python .claude/skills/telemetry-direct-query/victoria_query.py metrics-names
```

Useful helper flags:

- `--verbose`: print the resolved request URL to stderr
- `--compact`: emit compact JSON for shell pipelines
- `--timeout`: override the default 15s timeout
- `metrics-range --since 30m --end now`: avoid hand-calculating epochs

## Query workflow by backend

### 1. VictoriaLogs

Use logs to move from symptom text to stable ids.
Current documented direct API path from `/vePFS-Mindverse/user/nolanho/docs/grafana/victoria-otel-debug.md` is:

- `GET $VICTORIA_LOGS_URL/select/logsql/query?query=...&limit=...`

Primary examples:

```bash
python .claude/skills/telemetry-direct-query/victoria_query.py logs \
  --query '"CUDA out of memory"' \
  --limit 20

python .claude/skills/telemetry-direct-query/victoria_query.py logs \
  --query 'request_id:sample_12b39ee941342a4fe58b57354688f3c1' \
  --limit 20

python .claude/skills/telemetry-direct-query/victoria_query.py logs \
  --query 'trace_id:042397caf580ea9b2d71eb2ea7332f99' \
  --limit 20
```

After you extract `trace_id` or `request_id`, pivot instead of repeating broad text search.

### 2. VictoriaTraces

VictoriaTraces exposes a Jaeger-compatible query surface under
`$VICTORIA_TRACES_URL/select/jaeger/api/...`.

Primary examples:

```bash
python .claude/skills/telemetry-direct-query/victoria_query.py trace-services

python .claude/skills/telemetry-direct-query/victoria_query.py trace-search \
  --service mint \
  --operation 'POST /api/v1/retrieve_future' \
  --lookback 1h \
  --limit 20

python .claude/skills/telemetry-direct-query/victoria_query.py trace-get \
  --trace-id 042397caf580ea9b2d71eb2ea7332f99
```

If the Jaeger result looks sparse, cross-check the same `trace_id` in logs:

```bash
python .claude/skills/telemetry-direct-query/victoria_query.py logs \
  --query 'trace_id:042397caf580ea9b2d71eb2ea7332f99' \
  --limit 50
```

If you already know `trace_id`, skip service discovery and fetch the trace directly.

### 3. VictoriaMetrics

Current documented direct API paths from `/vePFS-Mindverse/user/nolanho/docs/grafana/victoria-otel-debug.md` are:

- `GET $VICTORIA_METRICS_URL/api/v1/query?query=...`
- `GET $VICTORIA_METRICS_URL/api/v1/query_range?query=...&start=...&end=...&step=...`
- `GET $VICTORIA_METRICS_URL/api/v1/label/__name__/values`

Primary examples:

```bash
python .claude/skills/telemetry-direct-query/victoria_query.py metrics-query \
  --query 'mint_metrics_up'

python .claude/skills/telemetry-direct-query/victoria_query.py metrics-query \
  --query 'rate(mint_http_server_requests_total{status_code=~"5.."}[5m])'

python .claude/skills/telemetry-direct-query/victoria_query.py metrics-range \
  --query 'rate(mint_http_server_requests_total[5m])' \
  --since 30m \
  --step 60s

python .claude/skills/telemetry-direct-query/victoria_query.py metrics-names
```

Use metrics to confirm rate, latency, queue depth, capacity pressure, or saturation after logs and traces have narrowed the failing path.

## Secondary path: HTTPie

Use HTTPie when you want raw HTTP visibility but still want less quoting pain than `curl`.

```bash
http --pretty=format GET "$VICTORIA_LOGS_URL/select/logsql/query" \
  query=='request_id:sample_12b39ee941342a4fe58b57354688f3c1' \
  limit:=20

http --pretty=format GET "$VICTORIA_TRACES_URL/select/jaeger/api/traces" \
  service==mint \
  operation=='POST /api/v1/retrieve_future' \
  lookback==1h \
  limit:=20

http --pretty=format GET "$VICTORIA_METRICS_URL/api/v1/query_range" \
  query=='rate(mint_http_server_requests_total[5m])' \
  start==1712462400 \
  end==1712464200 \
  step==60s
```

## Fallback path: curl

Use `curl` only when HTTPie is absent or you need exact raw URL semantics.
Keep `curl` as a low-level fallback, not the default workflow.

```bash
curl -sG "$VICTORIA_LOGS_URL/select/logsql/query" \
  --data-urlencode 'query=request_id:sample_12b39ee941342a4fe58b57354688f3c1' \
  --data-urlencode 'limit=20'

curl -s "$VICTORIA_TRACES_URL/select/jaeger/api/services"

curl -sG "$VICTORIA_TRACES_URL/select/jaeger/api/traces" \
  --data-urlencode 'service=mint' \
  --data-urlencode 'operation=POST /api/v1/retrieve_future' \
  --data-urlencode 'lookback=1h' \
  --data-urlencode 'limit=20'

curl -s "$VICTORIA_TRACES_URL/select/jaeger/api/traces/042397caf580ea9b2d71eb2ea7332f99"

curl -sG "$VICTORIA_METRICS_URL/api/v1/query" \
  --data-urlencode 'query=rate(mint_http_server_requests_total{status_code=~"5.."}[5m])'

curl -s "$VICTORIA_METRICS_URL/api/v1/label/__name__/values"
```

## Incident loop

1. Logs: search the exact symptom and pull out `trace_id` or `request_id`.
2. Traces: fetch the trace or recent trace set around the same service or operation.
3. Metrics: confirm whether the event appears in rates, latency, queue depth, or capacity metrics.
4. Record only the proof that changes the diagnosis.

## Current limit

I did not runtime-verify the live VictoriaTraces path in this skill. The Jaeger path is copied from the existing Grafana docs and should be treated as documented configuration, not runtime proof.
