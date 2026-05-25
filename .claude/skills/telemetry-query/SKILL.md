---
name: telemetry-query
description: |
  Query MinT production telemetry from VictoriaLogs, VictoriaTraces, and
  VictoriaMetrics with the project-bundled Python helper, HTTPie, or raw curl.

  Use for: incident triage from exact error text, request_id, trace_id, endpoint,
  metric name, Grafana/Victoria symptoms, or a narrow time window.

  Triggers: "telemetry", "telemetry query", "victoria", "victoriametrics",
  "victorialogs", "victoriatraces", "grafana", "promql", "logsql", "trace_id",
  "request_id", "metrics", "logs"
---

# telemetry-query

Use this skill for project-standard MinT telemetry queries. Prefer the bundled
Python helper first; it is stdlib-only and wraps the documented Victoria APIs.

## Query surfaces

1. VictoriaLogs: LogSQL search and id-first narrowing.
2. VictoriaTraces: Jaeger-compatible trace discovery and trace fetch.
3. VictoriaMetrics: PromQL instant and range queries.

Current recorded defaults from `/vePFS-Mindverse/user/nolanho/docs/grafana/README.md`:

- `VICTORIA_LOGS_URL=http://192.168.4.70:9428`
- `VICTORIA_TRACES_URL=http://192.168.4.70:10428`
- `VICTORIA_METRICS_URL=http://192.168.4.70:8428`

If current docs or the task provide different endpoints, set the environment
variables instead of editing commands inline.

## Hard rules

- Start from the narrowest anchor available: `request_id`, `trace_id`, exact
  error text, endpoint, service, metric name, or a small time window.
- Logs first for vague incidents. Trace first only when you already have a
  `trace_id`.
- Metrics confirm blast radius, rates, latency, queue pressure, or saturation.
  They do not replace logs/traces for root cause.
- Do not assume localhost unless explicitly running on the observability host or
  through a stated tunnel.
- Keep evidence short: command, time window, stable ids, hit count, and one to
  three proof lines.
- Prefer `mint_*` metric names. Treat `tinker_*`-only signals as migration debt.
- Never print credentials, signed URLs, process environments, or secret config.

## Setup

```bash
export VICTORIA_LOGS_URL=${VICTORIA_LOGS_URL:-http://192.168.4.70:9428}
export VICTORIA_TRACES_URL=${VICTORIA_TRACES_URL:-http://192.168.4.70:10428}
export VICTORIA_METRICS_URL=${VICTORIA_METRICS_URL:-http://192.168.4.70:8428}
python .claude/skills/telemetry-query/victoria_query.py --help
```

## Primary path

Use `.claude/skills/telemetry-query/victoria_query.py`.

Examples:

```bash
python .claude/skills/telemetry-query/victoria_query.py logs \
  --query 'request_id:sample_12b39ee941342a4fe58b57354688f3c1' \
  --limit 20

python .claude/skills/telemetry-query/victoria_query.py logs \
  --query '"CUDA out of memory"' \
  --limit 20

python .claude/skills/telemetry-query/victoria_query.py trace-services

python .claude/skills/telemetry-query/victoria_query.py trace-search \
  --service mint \
  --operation 'POST /api/v1/retrieve_future' \
  --lookback 1h \
  --limit 20

python .claude/skills/telemetry-query/victoria_query.py trace-get \
  --trace-id 042397caf580ea9b2d71eb2ea7332f99

python .claude/skills/telemetry-query/victoria_query.py metrics-query \
  --query 'mint_metrics_up'

python .claude/skills/telemetry-query/victoria_query.py metrics-range \
  --query 'rate(mint_http_server_requests_total[5m])' \
  --since 30m \
  --step 60s

python .claude/skills/telemetry-query/victoria_query.py metrics-names
```

Useful helper flags:
- `--verbose`: print the resolved request URL to stderr.
- `--compact`: emit compact JSON for shell pipelines.
- `--timeout`: override the default 15s timeout.
- `metrics-range --since 30m --end now`: avoid hand-calculating epochs.

## Workflow

1. Search logs by the exact symptom, `request_id`, or `trace_id`.
2. Extract stable ids and pivot instead of repeating broad text searches.
3. Fetch the trace when you have a trace id or need stage timing.
4. Query metrics to decide whether the event is isolated or systemic.
5. Record only proof that changes the diagnosis.

## Fallbacks

Use HTTPie when you need raw HTTP visibility:

```bash
http --pretty=format GET "$VICTORIA_LOGS_URL/select/logsql/query" \
  query=='request_id:sample_12b39ee941342a4fe58b57354688f3c1' \
  limit:=20

http --pretty=format GET "$VICTORIA_TRACES_URL/select/jaeger/api/traces" \
  service==mint \
  operation=='POST /api/v1/retrieve_future' \
  lookback==1h \
  limit:=20
```

Use curl only when HTTPie is absent or exact raw URL semantics matter:

```bash
curl -sG "$VICTORIA_LOGS_URL/select/logsql/query" \
  --data-urlencode 'query=request_id:sample_12b39ee941342a4fe58b57354688f3c1' \
  --data-urlencode 'limit=20'

curl -s "$VICTORIA_TRACES_URL/select/jaeger/api/services"

curl -sG "$VICTORIA_METRICS_URL/api/v1/query" \
  --data-urlencode 'query=rate(mint_http_server_requests_total{http.status_code=~"5.."}[5m])'
```

## Notes

- For dashboard edits, inspect live label names first. OTel attributes such as
  `http.status_code` may appear differently after Victoria/Grafana normalization.
- If VictoriaTraces returns sparse data, cross-check the same `trace_id` in logs.
