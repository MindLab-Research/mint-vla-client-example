#!/usr/bin/env python3
import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from typing import Any

DEFAULTS = {
    "metrics": "http://192.168.4.70:8428",
    "logs": "http://192.168.4.70:9428",
    "traces": "http://192.168.4.70:10428",
}


def env_url(kind: str) -> str:
    names = {
        "metrics": "VICTORIA_METRICS_URL",
        "logs": "VICTORIA_LOGS_URL",
        "traces": "VICTORIA_TRACES_URL",
    }
    return os.environ.get(names[kind], DEFAULTS[kind]).rstrip("/")


def request_json(base: str, path: str, params: dict[str, Any] | None = None) -> Any:
    qs = urllib.parse.urlencode(
        {k: v for k, v in (params or {}).items() if v is not None}, doseq=True
    )
    url = f"{base}{path}"
    if qs:
        url = f"{url}?{qs}"
    with urllib.request.urlopen(url) as resp:
        return json.load(resp)


def add_common_url_flags(parser: argparse.ArgumentParser, kind: str) -> None:
    parser.add_argument(
        "--url",
        default=env_url(kind),
        help=f"override {kind} base url",
    )


def cmd_logs(args: argparse.Namespace) -> int:
    out = request_json(
        args.url,
        "/select/logsql/query",
        {"query": args.query, "limit": args.limit},
    )
    json.dump(out, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def cmd_trace_services(args: argparse.Namespace) -> int:
    out = request_json(args.url, "/select/jaeger/api/services")
    json.dump(out, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def cmd_trace_search(args: argparse.Namespace) -> int:
    out = request_json(
        args.url,
        "/select/jaeger/api/traces",
        {
            "service": args.service,
            "operation": args.operation,
            "lookback": args.lookback,
            "limit": args.limit,
            "minDuration": args.min_duration,
            "maxDuration": args.max_duration,
            "tags": args.tags,
        },
    )
    json.dump(out, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def cmd_trace_get(args: argparse.Namespace) -> int:
    out = request_json(args.url, f"/select/jaeger/api/traces/{args.trace_id}")
    json.dump(out, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def cmd_metrics_query(args: argparse.Namespace) -> int:
    out = request_json(args.url, "/api/v1/query", {"query": args.query, "time": args.time})
    json.dump(out, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def cmd_metrics_range(args: argparse.Namespace) -> int:
    out = request_json(
        args.url,
        "/api/v1/query_range",
        {
            "query": args.query,
            "start": args.start,
            "end": args.end,
            "step": args.step,
        },
    )
    json.dump(out, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def cmd_metrics_names(args: argparse.Namespace) -> int:
    out = request_json(args.url, "/api/v1/label/__name__/values")
    json.dump(out, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Direct Victoria query helper for logs, traces, and metrics."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    logs = sub.add_parser("logs", help="query VictoriaLogs via LogSQL")
    add_common_url_flags(logs, "logs")
    logs.add_argument("--query", required=True, help="LogSQL query text")
    logs.add_argument("--limit", type=int, default=20, help="max records")
    logs.set_defaults(func=cmd_logs)

    tr_sv = sub.add_parser("trace-services", help="list traced services")
    add_common_url_flags(tr_sv, "traces")
    tr_sv.set_defaults(func=cmd_trace_services)

    tr_search = sub.add_parser("trace-search", help="search traces via Jaeger API")
    add_common_url_flags(tr_search, "traces")
    tr_search.add_argument("--service", required=True, help="service name")
    tr_search.add_argument("--operation", help="operation name")
    tr_search.add_argument("--lookback", default="1h", help="time window")
    tr_search.add_argument("--limit", type=int, default=20, help="max traces")
    tr_search.add_argument("--min-duration", help="Jaeger duration filter, e.g. 500ms")
    tr_search.add_argument("--max-duration", help="Jaeger duration filter")
    tr_search.add_argument(
        "--tags",
        help='Jaeger tags filter, e.g. {"http.status_code":"500"}',
    )
    tr_search.set_defaults(func=cmd_trace_search)

    tr_get = sub.add_parser("trace-get", help="fetch one trace by trace_id")
    add_common_url_flags(tr_get, "traces")
    tr_get.add_argument("--trace-id", required=True, help="trace id")
    tr_get.set_defaults(func=cmd_trace_get)

    m_query = sub.add_parser("metrics-query", help="instant PromQL query")
    add_common_url_flags(m_query, "metrics")
    m_query.add_argument("--query", required=True, help="PromQL expression")
    m_query.add_argument("--time", help="evaluation time")
    m_query.set_defaults(func=cmd_metrics_query)

    m_range = sub.add_parser("metrics-range", help="range PromQL query")
    add_common_url_flags(m_range, "metrics")
    m_range.add_argument("--query", required=True, help="PromQL expression")
    m_range.add_argument("--start", required=True, help="range start")
    m_range.add_argument("--end", required=True, help="range end")
    m_range.add_argument("--step", required=True, help="range step")
    m_range.set_defaults(func=cmd_metrics_range)

    m_names = sub.add_parser("metrics-names", help="list metric names")
    add_common_url_flags(m_names, "metrics")
    m_names.set_defaults(func=cmd_metrics_names)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
