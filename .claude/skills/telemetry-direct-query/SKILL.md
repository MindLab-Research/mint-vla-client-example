---
name: telemetry-direct-query
description: |
  Direct VictoriaMetrics, VictoriaLogs, and VictoriaTraces querying with curl,
  HTTPie, or Python when `obsh` is unavailable.

  Use for: incident triage from error text, `request_id`, `trace_id`, endpoint,
  or a narrow time window when only raw HTTP access is available.

  Triggers: "victoria", "victoriametrics", "victorialogs", "victoriatraces",
  "telemetry query", "observability query", "promql", "logsql", "trace_id",
  "request_id", "curl observability", "httpie observability"
---

# telemetry-direct-query

Use this skill when `obsh` is unavailable and you must query Victoria directly.
The three query surfaces are:

1. VictoriaLogs: text search and id-first narrowing
2. VictoriaTraces: Jaeger-compatible trace search and trace fetch
3. VictoriaMetrics: PromQL instant and range queries

Do not guess endpoints from memory. Current recorded defaults from
`/vePFS-Mindverse/user/nolanho/docs/grafana/README.md` are:

- `VICTORIA_LOGS_URL=http://192.168.4.70:9428`
- `VICTORIA_TRACES_URL=http://192.168.4.70:10428`
- `VICTORIA_METRICS_URL=http://192.168.4.70:8428`

If the task provides a tunnel or another host, override these env vars instead of
editing commands inline.

## Hard rules

- Start from the narrowest known anchor: exact error text, `request_id`, `trace_id`, endpoint, service, or small time window.
- Logs first for vague incidents. Trace first only when you already have `trace_id`.
- Use metrics to confirm rate, latency, saturation, queue pressure, or resource effects. Metrics do not replace logs.
- Do not assume localhost unless you are explicitly on the observability host or inside a tunnel.
- Keep copied evidence short: command, time window, stable ids, and one to three proof lines.

## Setup

```bash
export VICTORIA_LOGS_URL=${VICTORIA_LOGS_URL:-http://192.168.4.70:9428}
export VICTORIA_TRACES_URL=${VICTORIA_TRACES_URL:-http://192.168.4.70:10428}
export VICTORIA_METRICS_URL=${VICTORIA_METRICS_URL:-http://192.168.4.70:8428}
```

Optional local helper:

```bash
python .claude/skills/telemetry-direct-query/victoria_query.py --help
```

## 1. Logs: VictoriaLogs

Current direct API pattern from `/vePFS-Mindverse/user/nolanho/docs/grafana/victoria-otel-debug.md`:

```bash
curl -s "$VICTORIA_LOGS_URL/select/logsql/query?query=trace_id:fedcba9876543210fedcba9876543210&limit=5"
```

Search by error text:

```bash
curl -sG "$VICTORIA_LOGS_URL/select/logsql/query" \
  --data-urlencode 'query="CUDA out of memory"' \
  --data-urlencode 'limit=20'
```

Search by `request_id` or `trace_id`:

```bash
curl -sG "$VICTORIA_LOGS_URL/select/logsql/query" \
  --data-urlencode 'query=request_id:sample_12b39ee941342a4fe58b57354688f3c1' \
  --data-urlencode 'limit=20'

curl -sG "$VICTORIA_LOGS_URL/select/logsql/query" \
  --data-urlencode 'query=trace_id:042397caf580ea9b2d71eb2ea7332f99' \
  --data-urlencode 'limit=20'
```

HTTPie equivalents:

```bash
http --pretty=format GET "$VICTORIA_LOGS_URL/select/logsql/query" \
  query=='request_id:sample_12b39ee941342a4fe58b57354688f3c1' \
  limit:=20
```

Python equivalent:

```bash
python - <<'PY'
import json
import os
import urllib.parse
import urllib.request

base = os.environ["VICTORIA_LOGS_URL"]
params = urllib.parse.urlencode(
    {
        "query": 'request_id:sample_12b39ee941342a4fe58b57354688f3c1',
        "limit": 20,
    }
)
with urllib.request.urlopen(f"{base}/select/logsql/query?{params}") as r:
    print(json.dumps(json.load(r), indent=2, sort_keys=True))
PY
```

Use logs to extract stable ids. Then pivot instead of repeating broad text search.

## 2. Traces: VictoriaTraces

VictoriaTraces exposes a Jaeger-compatible query surface under
`$VICTORIA_TRACES_URL/select/jaeger/api/...`.

Discover services first when the trace ownership is unclear:

```bash
curl -s "$VICTORIA_TRACES_URL/select/jaeger/api/services"
```

Search recent traces by service and optional operation:

```bash
curl -sG "$VICTORIA_TRACES_URL/select/jaeger/api/traces" \
  --data-urlencode 'service=mint' \
  --data-urlencode 'operation=POST /api/v1/retrieve_future' \
  --data-urlencode 'lookback=1h' \
  --data-urlencode 'limit=20'
```

Fetch one trace directly when you already have `trace_id`:

```bash
TRACE_ID=042397caf580ea9b2d71eb2ea7332f99
curl -s "$VICTORIA_TRACES_URL/select/jaeger/api/traces/$TRACE_ID"
```

Cross-check the same `trace_id` through log search when the Jaeger result looks sparse:

```bash
curl -sG "$VICTORIA_LOGS_URL/select/logsql/query" \
  --data-urlencode "query=trace_id:$TRACE_ID" \
  --data-urlencode 'limit=50'
```

HTTPie equivalents:

```bash
http --pretty=format GET "$VICTORIA_TRACES_URL/select/jaeger/api/services"

http --pretty=format GET "$VICTORIA_TRACES_URL/select/jaeger/api/traces" \
  service==mint \
  operation=='POST /api/v1/retrieve_future' \
  lookback==1h \
  limit:=20
```

Python equivalent:

```bash
python - <<'PY'
import json
import os
import urllib.parse
import urllib.request

base = os.environ["VICTORIA_TRACES_URL"]
params = urllib.parse.urlencode(
    {
        "service": "mint",
        "operation": "POST /api/v1/retrieve_future",
        "lookback": "1h",
        "limit": 20,
    }
)
with urllib.request.urlopen(f"{base}/select/jaeger/api/traces?{params}") as r:
    print(json.dumps(json.load(r), indent=2, sort_keys=True))
PY
```

If you already know `trace_id`, skip service discovery and fetch the trace directly.

## 3. Metrics: VictoriaMetrics

Current direct API pattern from `/vePFS-Mindverse/user/nolanho/docs/grafana/victoria-otel-debug.md`:

```bash
curl -s "$VICTORIA_METRICS_URL/api/v1/query?query=maple_test_counter_total"
```

Instant query for a metric or PromQL expression:

```bash
curl -sG "$VICTORIA_METRICS_URL/api/v1/query" \
  --data-urlencode 'query=mint_metrics_up'

curl -sG "$VICTORIA_METRICS_URL/api/v1/query" \
  --data-urlencode 'query=rate(mint_http_server_requests_total{status_code=~"5.."}[5m])'
```

Range query for trend confirmation:

```bash
START=$(date -u -d '30 minutes ago' +%s)
END=$(date -u +%s)

curl -sG "$VICTORIA_METRICS_URL/api/v1/query_range" \
  --data-urlencode 'query=rate(mint_http_server_requests_total[5m])' \
  --data-urlencode "start=$START" \
  --data-urlencode "end=$END" \
  --data-urlencode 'step=60s'
```

Discover available metric names:

```bash
curl -s "$VICTORIA_METRICS_URL/api/v1/label/__name__/values"
```

HTTPie equivalents:

```bash
http --pretty=format GET "$VICTORIA_METRICS_URL/api/v1/query" \
  query=='mint_metrics_up'

http --pretty=format GET "$VICTORIA_METRICS_URL/api/v1/query_range" \
  query=='rate(mint_http_server_requests_total[5m])' \
  start==1712462400 \
  end==1712464200 \
  step==60s
```

Python equivalent:

```bash
python - <<'PY'
import json
import os
import urllib.parse
import urllib.request

base = os.environ["VICTORIA_METRICS_URL"]
params = urllib.parse.urlencode(
    {
        "query": 'rate(mint_http_server_requests_total[5m])',
        "start": 1712462400,
        "end": 1712464200,
        "step": "60s",
    }
)
with urllib.request.urlopen(f"{base}/api/v1/query_range?{params}") as r:
    print(json.dumps(json.load(r), indent=2, sort_keys=True))
PY
```

Use `mint_*` metric names by default. If only `tinker_*` exists in raw data, treat that as migration debt rather than the preferred query prefix.

## Incident loop

1. Logs: search the exact symptom and pull out `trace_id` or `request_id`.
2. Traces: fetch the trace tree or recent trace set around the same service or operation.
3. Metrics: confirm whether the event is isolated or reflected in rates, latency, queue depth, or capacity metrics.
4. Record only the proof that changes the diagnosis.

## Local helper script

The helper wraps the direct APIs without `requests` or other third-party packages:

```bash
python .claude/skills/telemetry-direct-query/victoria_query.py logs \
  --query 'request_id:sample_12b39ee941342a4fe58b57354688f3c1' \
  --limit 20

python .claude/skills/telemetry-direct-query/victoria_query.py trace-get \
  --trace-id 042397caf580ea9b2d71eb2ea7332f99

python .claude/skills/telemetry-direct-query/victoria_query.py metrics-range \
  --query 'rate(mint_http_server_requests_total[5m])' \
  --start 1712462400 \
  --end 1712464200 \
  --step 60s
```

Use the helper when shell quoting becomes the main source of error.
