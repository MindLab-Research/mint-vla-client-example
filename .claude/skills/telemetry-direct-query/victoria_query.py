#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

DEFAULTS = {
    "metrics": "http://192.168.4.70:8428",
    "logs": "http://192.168.4.70:9428",
    "traces": "http://192.168.4.70:10428",
}

DURATION_RE = re.compile(r"^(?P<value>\d+)(?P<unit>[smhdw])$")
DURATION_SCALE = {
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
    "w": 604800,
}


def env_url(kind: str) -> str:
    names = {
        "metrics": "VICTORIA_METRICS_URL",
        "logs": "VICTORIA_LOGS_URL",
        "traces": "VICTORIA_TRACES_URL",
    }
    return os.environ.get(names[kind], DEFAULTS[kind]).rstrip("/")


def parse_duration(value: str) -> int:
    match = DURATION_RE.fullmatch(value)
    if not match:
        raise argparse.ArgumentTypeError(
            f"invalid duration '{value}'; use forms like 30s, 15m, 2h, 1d"
        )
    return int(match.group("value")) * DURATION_SCALE[match.group("unit")]


def parse_epoch(value: str) -> float:
    if value == "now":
        return float(int(time.time()))
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid timestamp '{value}'; use unix epoch seconds or 'now'"
        ) from exc


def compute_range(start: str | None, end: str | None, since: str | None) -> tuple[str, str]:
    if since is not None:
        if start is not None:
            raise argparse.ArgumentTypeError("use either --start or --since, not both")
        end_ts = parse_epoch(end or "now")
        start_ts = end_ts - parse_duration(since)
        return normalize_ts(start_ts), normalize_ts(end_ts)
    if start is None or end is None:
        raise argparse.ArgumentTypeError(
            "metrics-range requires --start and --end, or --since with optional --end"
        )
    return normalize_ts(parse_epoch(start)), normalize_ts(parse_epoch(end))


def normalize_ts(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return str(value)


@dataclass
class VictoriaClient:
    base_url: str
    timeout: float = 15.0
    verbose: bool = False

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        qs = urllib.parse.urlencode(
            {k: v for k, v in (params or {}).items() if v is not None}, doseq=True
        )
        url = f"{self.base_url}{path}"
        if qs:
            url = f"{url}?{qs}"
        if self.verbose:
            print(url, file=sys.stderr)
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "mint-telemetry-direct-query/1",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} for {url}\n{body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"request failed for {url}: {exc.reason}") from exc


def dump_json(data: Any, compact: bool) -> None:
    if compact:
        json.dump(data, sys.stdout, separators=(",", ":"), sort_keys=True)
    else:
        json.dump(data, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def build_client(args: argparse.Namespace) -> VictoriaClient:
    return VictoriaClient(base_url=args.url.rstrip("/"), timeout=args.timeout, verbose=args.verbose)


def add_common_flags(parser: argparse.ArgumentParser, kind: str) -> None:
    parser.add_argument("--url", default=env_url(kind), help=f"override {kind} base url")
    parser.add_argument("--timeout", type=float, default=15.0, help="request timeout seconds")
    parser.add_argument("--compact", action="store_true", help="emit compact JSON")
    parser.add_argument("--verbose", action="store_true", help="print resolved request URL to stderr")


def cmd_logs(args: argparse.Namespace) -> int:
    data = build_client(args).get_json(
        "/select/logsql/query",
        {"query": args.query, "limit": args.limit},
    )
    dump_json(data, args.compact)
    return 0


def cmd_trace_services(args: argparse.Namespace) -> int:
    data = build_client(args).get_json("/select/jaeger/api/services")
    dump_json(data, args.compact)
    return 0


def cmd_trace_search(args: argparse.Namespace) -> int:
    data = build_client(args).get_json(
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
    dump_json(data, args.compact)
    return 0


def cmd_trace_get(args: argparse.Namespace) -> int:
    data = build_client(args).get_json(f"/select/jaeger/api/traces/{args.trace_id}")
    dump_json(data, args.compact)
    return 0


def cmd_metrics_query(args: argparse.Namespace) -> int:
    data = build_client(args).get_json(
        "/api/v1/query",
        {
            "query": args.query,
            "time": normalize_ts(parse_epoch(args.time)) if args.time is not None else None,
        },
    )
    dump_json(data, args.compact)
    return 0


def cmd_metrics_range(args: argparse.Namespace) -> int:
    start, end = compute_range(args.start, args.end, args.since)
    data = build_client(args).get_json(
        "/api/v1/query_range",
        {
            "query": args.query,
            "start": start,
            "end": end,
            "step": args.step,
        },
    )
    dump_json(data, args.compact)
    return 0


def cmd_metrics_names(args: argparse.Namespace) -> int:
    data = build_client(args).get_json("/api/v1/label/__name__/values")
    dump_json(data, args.compact)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Direct Victoria query helper for logs, traces, and metrics."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    logs = sub.add_parser("logs", help="query VictoriaLogs via LogSQL")
    add_common_flags(logs, "logs")
    logs.add_argument("--query", required=True, help="LogSQL query text")
    logs.add_argument("--limit", type=int, default=20, help="max records")
    logs.set_defaults(func=cmd_logs)

    tr_sv = sub.add_parser("trace-services", help="list traced services")
    add_common_flags(tr_sv, "traces")
    tr_sv.set_defaults(func=cmd_trace_services)

    tr_search = sub.add_parser("trace-search", help="search traces via Jaeger API")
    add_common_flags(tr_search, "traces")
    tr_search.add_argument("--service", required=True, help="service name")
    tr_search.add_argument("--operation", help="operation name")
    tr_search.add_argument("--lookback", default="1h", help="Jaeger lookback window")
    tr_search.add_argument("--limit", type=int, default=20, help="max traces")
    tr_search.add_argument("--min-duration", help="Jaeger duration filter, e.g. 500ms")
    tr_search.add_argument("--max-duration", help="Jaeger duration filter")
    tr_search.add_argument("--tags", help='Jaeger tags JSON, e.g. {"http.status_code":"500"}')
    tr_search.set_defaults(func=cmd_trace_search)

    tr_get = sub.add_parser("trace-get", help="fetch one trace by trace_id")
    add_common_flags(tr_get, "traces")
    tr_get.add_argument("--trace-id", required=True, help="trace id")
    tr_get.set_defaults(func=cmd_trace_get)

    m_query = sub.add_parser("metrics-query", help="instant PromQL query")
    add_common_flags(m_query, "metrics")
    m_query.add_argument("--query", required=True, help="PromQL expression")
    m_query.add_argument("--time", help="unix epoch seconds or 'now'")
    m_query.set_defaults(func=cmd_metrics_query)

    m_range = sub.add_parser("metrics-range", help="range PromQL query")
    add_common_flags(m_range, "metrics")
    m_range.add_argument("--query", required=True, help="PromQL expression")
    m_range.add_argument("--start", help="range start in unix epoch seconds")
    m_range.add_argument("--end", help="range end in unix epoch seconds or 'now'")
    m_range.add_argument("--since", help="relative lookback such as 30m or 2h")
    m_range.add_argument("--step", default="60s", help="range step")
    m_range.set_defaults(func=cmd_metrics_range)

    m_names = sub.add_parser("metrics-names", help="list metric names")
    add_common_flags(m_names, "metrics")
    m_names.set_defaults(func=cmd_metrics_names)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except (RuntimeError, argparse.ArgumentTypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
