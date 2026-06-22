#!/usr/bin/env python3
"""
High-level telemetry debugging helper for MinT, built on the Victoria MCP client.

Commands map to real debugging workflows and emit compact, human-readable output
(timestamps formatted, key fields extracted). Use --json for raw payloads.

All queries go through mcp_query.MCPClient (JSON-RPC 2.0 over Streamable HTTP).
Config (MCP_API_KEY, MCP_*_URL) is read from .env next to this file or the env.

Time formats differ by endpoint (handled here automatically):
  logs/metrics -> RFC3339,  traces -> Unix milliseconds.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from mcp_query import MCPClient, MCPError, dump_json, endpoint_url, get_api_key, load_dotenv


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def rfc3339(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def parse_ts(value: str) -> datetime:
    """Parse an RFC3339 / 'YYYY-MM-DD HH:MM:SS' / Unix-seconds timestamp to UTC."""
    txt = value.strip()
    if txt.isdigit():
        return datetime.fromtimestamp(int(txt), tz=timezone.utc)
    try:
        return datetime.fromisoformat(txt.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid timestamp '{value}'; use RFC3339 (2026-06-11T04:10:00Z) or Unix seconds"
        ) from exc


def _hits_total(payload: Any) -> int:
    buckets = payload.get("hits") if isinstance(payload, dict) else None
    if not buckets:
        return 0
    return sum(int(b.get("total", 0)) for b in buckets)


def _earliest_nonzero(payload: Any) -> str | None:
    buckets = payload.get("hits") if isinstance(payload, dict) else None
    if not buckets:
        return None
    first = buckets[0]
    for t, v in zip(first.get("timestamps", []), first.get("values", [])):
        if v:
            return t
    return None


def parse_ts(value: str) -> datetime:
    """Parse a user-supplied RFC3339 / 'YYYY-MM-DD HH:MM' / Unix-seconds time."""
    value = value.strip()
    if value.isdigit():
        return datetime.fromtimestamp(int(value), tz=timezone.utc)
    txt = value.replace("Z", "+00:00").replace(" ", "T", 1)
    dt = datetime.fromisoformat(txt)
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def resolve_window(args: argparse.Namespace) -> tuple[datetime, datetime, str]:
    """Resolve a query window from --start/--end or --lookback.

    Precise --start/--end (historical incident triage) takes precedence over the
    relative --lookback. Returns (start_dt, end_dt, human_label).
    """
    start = getattr(args, "start", None)
    end = getattr(args, "end", None)
    if start:
        start_dt = parse_ts(start)
        end_dt = parse_ts(end) if end else utcnow()
        return start_dt, end_dt, f"{rfc3339(start_dt)} ~ {rfc3339(end_dt)}"
    lookback = getattr(args, "lookback", 1440)
    end_dt = utcnow()
    start_dt = end_dt - timedelta(minutes=lookback)
    return start_dt, end_dt, f"last {lookback}m"


def fmt_time(row: dict[str, Any]) -> str:
    """Render a log row's timestamp as readable UTC.

    `mint` rows carry ns epoch in `_ray_timestamp_ns`; `mint-platform` rows
    only have RFC3339 `_time`. Try ns first, then RFC3339, else raw.
    """
    raw_ns = row.get("_ray_timestamp_ns")
    if raw_ns is not None:
        try:
            secs = int(raw_ns) / 1e9
            return datetime.fromtimestamp(secs, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )
        except (TypeError, ValueError):
            pass
    raw = row.get("_time")
    if raw:
        try:
            txt = str(raw).replace("Z", "+00:00")
            dt = datetime.fromisoformat(txt).astimezone(timezone.utc)
            return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except ValueError:
            return str(raw)
    return "(no timestamp)"


# Severity aliases -> canonical VictoriaLogs value (real values: ERROR/WARN/INFO/DEBUG).
SEVERITY_ALIASES = {
    "WARNING": "WARN",
    "ERR": "ERROR",
    "TRACE": "DEBUG",
    "CRITICAL": "ERROR",
    "FATAL": "ERROR",
}


def normalize_severity(value: str) -> str:
    """Map common severity aliases to the value VictoriaLogs actually stores."""
    up = value.strip().upper()
    return SEVERITY_ALIASES.get(up, up)


def dedup_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop exact-duplicate log rows (same time + msg + request_id)."""
    seen: set[tuple] = set()
    out: list[dict[str, Any]] = []
    for r in rows:
        key = (
            r.get("_ray_timestamp_ns") or r.get("_time"),
            r.get("_msg") or r.get("message"),
            r.get("request_id"),
            r.get("severity"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


# Fields worth surfacing by default when present (beyond ts/severity/service/msg).
DEFAULT_EXTRA_FIELDS = (
    "exception.type",
    "exception.message",
    "error",
    "request_id",
    "trace_id",
    "gateway_request_id",
    "host.name",
    "service.instance.id",
    "process.pid",
    "mint.cluster_id",
    "code.file.path",
    "code.function.name",
)


def is_blank(value: Any) -> bool:
    """VictoriaLogs uses '-' as the empty placeholder for absent string fields."""
    return value is None or value == "" or value == "-"


def parse_log_lines(payload: Any) -> list[dict[str, Any]]:
    """VictoriaLogs `query` text payload is newline-delimited JSON objects."""
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        return [payload]
    if not isinstance(payload, str) or not payload.strip():
        return []
    rows: list[dict[str, Any]] = []
    for line in payload.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        except json.JSONDecodeError:
            continue
    return rows


class Helper:
    def __init__(self, args: argparse.Namespace) -> None:
        key = get_api_key()
        self.json_out = getattr(args, "json", False)
        self.compact = getattr(args, "compact", False)
        self.fields = getattr(args, "fields", None)
        self.full = getattr(args, "full", False)
        self.logs = MCPClient(endpoint_url("logs"), key, timeout=args.timeout, verbose=args.verbose)
        self.traces = MCPClient(endpoint_url("traces"), key, timeout=args.timeout, verbose=args.verbose)
        self.metrics = MCPClient(endpoint_url("metrics"), key, timeout=args.timeout, verbose=args.verbose)

    # ---- logs ----

    def log_query(
        self, query: str, start_dt: datetime, end_dt: datetime, limit: int
    ) -> list[dict[str, Any]]:
        payload = self.logs.call_tool(
            "query",
            {"query": query, "start": rfc3339(start_dt), "end": rfc3339(end_dt), "limit": limit},
        )
        return dedup_rows(parse_log_lines(payload))

    def window_coverage_hint(self, start_dt: datetime, end_dt: datetime) -> str:
        """When 0 hits, report whether the *exact window* has any mint data.

        This catches the #718 trap: a day-level bucket shows millions of rows,
        but the incident minute itself is empty because retention rolled the
        early hours off. Probe the precise window first, then the wider library.
        """
        # 1) Does the exact requested window contain any mint logs at all?
        try:
            in_win = self.logs.call_tool(
                "hits",
                {
                    "query": 'service.name:"mint"',
                    "start": rfc3339(start_dt),
                    "end": rfc3339(end_dt),
                    "step": "1h",
                },
            )
        except MCPError:
            return ""
        win_total = _hits_total(in_win)
        if win_total > 0:
            return (
                f"提示: 该精确时间窗内 mint 服务有约 {win_total} 条日志，"
                "数据存在；0 命中更可能是查询过窄或字段值不匹配，可放宽 query 重试。"
            )
        # 2) Window is empty. Find where data actually starts (last 90d), so the
        #    user can tell "incident before retention" from "wrong window".
        wide_start = utcnow() - timedelta(days=90)
        try:
            wide = self.logs.call_tool(
                "hits",
                {"query": 'service.name:"mint"', "start": rfc3339(wide_start), "step": "1d"},
            )
        except MCPError:
            wide = None
        earliest_day = _earliest_nonzero(wide)
        if earliest_day is None:
            return (
                "提示: 该时间窗内 mint 服务无日志，最近 90 天也查不到数据，"
                "可能已过保留期 (后端 2026-06-10 做过 Victoria OTel 迁移)。"
            )
        # Day buckets are stamped at 00:00 even when the day's data only starts
        # hours later (the exact #718 trap). Drill into the earliest day at hour
        # resolution to report the *real* first data point.
        earliest = earliest_day
        try:
            day_start = parse_ts(earliest_day)
            day = self.logs.call_tool(
                "hits",
                {
                    "query": 'service.name:"mint"',
                    "start": rfc3339(day_start),
                    "end": rfc3339(day_start + timedelta(days=1)),
                    "step": "1h",
                },
            )
            hour = _earliest_nonzero(day)
            if hour:
                earliest = hour
        except (MCPError, ValueError):
            pass
        return (
            f"提示: 该精确时间窗 [{rfc3339(start_dt)} ~ {rfc3339(end_dt)}] 内 mint 服务 0 条日志，"
            f"但保留库中最早的实际数据出现在 {earliest} (已下钻到小时，非按天分桶的 00:00 假象)。"
            "若事故早于该时刻，telemetry 查不到属正常 (数据已被保留期滚出)。"
        )

    def print_logs(
        self,
        rows: list[dict[str, Any]],
        title: str,
        start_dt: datetime | None = None,
        end_dt: datetime | None = None,
    ) -> None:
        if self.json_out:
            dump_json(rows, self.compact)
            return
        print(f"=== {title} ({len(rows)} hits) ===\n")
        if not rows:
            print("(no log entries in this window)")
            if start_dt is not None and end_dt is not None:
                hint = self.window_coverage_hint(start_dt, end_dt)
                if hint:
                    print(hint)
            return
        extra = (
            [f.strip() for f in self.fields.split(",")] if self.fields else DEFAULT_EXTRA_FIELDS
        )
        msg_width = 100_000 if self.full else 300
        for i, r in enumerate(rows, 1):
            ts = fmt_time(r)
            sev = r.get("severity") or r.get("level") or ""
            svc = r.get("service.name") or r.get("service_name") or ""
            msg = (r.get("_msg") or r.get("message") or "").strip()
            print(f"{i}. {ts}  [{sev}] {svc}")
            print(f"   {msg[:msg_width]}")
            for k in extra:
                v = r.get(k)
                if not is_blank(v):
                    shown = str(v) if self.full else str(v)[:200]
                    print(f"   {k}: {shown}")
            print()

    # ---- traces ----

    def trace_services(self) -> Any:
        return self.traces.call_tool("services", {})

    def trace_search(
        self, service: str, operation: str | None, lookback_min: int,
        limit: int, min_duration: str | None,
    ) -> Any:
        start = ms(utcnow() - timedelta(minutes=lookback_min))
        end = ms(utcnow())
        return self.traces.call_tool(
            "traces",
            {
                "service": service,
                "operation": operation,
                "start": start,
                "end": end,
                "limit": limit,
                "minDuration": min_duration,
            },
        )

    def trace_get(self, trace_id: str) -> Any:
        return self.traces.call_tool("trace", {"trace_id": trace_id})

    # ---- metrics ----

    def metric_query(self, promql: str) -> Any:
        return self.metrics.call_tool("query", {"query": promql})

    def metric_names(self, match: str | None = None) -> list[str]:
        args = {"match": match} if match else {}
        payload = self.metrics.call_tool("metrics", args)
        if isinstance(payload, dict):
            names = payload.get("data", [])
        elif isinstance(payload, list):
            names = payload
        else:
            names = []
        return [n for n in names if isinstance(n, str)]


# ===== command handlers =====


def cmd_recent_errors(args: argparse.Namespace) -> int:
    h = Helper(args)
    start_dt, end_dt, label = resolve_window(args)
    rows = h.log_query("severity:ERROR", start_dt, end_dt, args.limit)
    h.print_logs(rows, f"Recent Errors ({label})", start_dt, end_dt)
    return 0


def cmd_find_logs(args: argparse.Namespace) -> int:
    h = Helper(args)
    start_dt, end_dt, label = resolve_window(args)
    query = args.query
    if args.severity:
        query = f"{query} AND severity:{normalize_severity(args.severity)}"
    rows = h.log_query(query, start_dt, end_dt, args.limit)
    h.print_logs(rows, f"Logs: {args.query} ({label})", start_dt, end_dt)
    return 0


def _request_id_query(ids: list[str]) -> str:
    """Build a LogsQL OR query over one or more request_ids."""
    if len(ids) == 1:
        return f'request_id:"{ids[0]}"'
    joined = " OR ".join(f'"{rid}"' for rid in ids)
    return f"request_id:({joined})"


def cmd_find_request(args: argparse.Namespace) -> int:
    h = Helper(args)
    start_dt, end_dt, _ = resolve_window(args)
    ids = args.request_id
    rows = h.log_query(_request_id_query(ids), start_dt, end_dt, args.limit)
    label = ids[0] if len(ids) == 1 else f"{len(ids)} request_ids"
    h.print_logs(rows, f"request_id={label}", start_dt, end_dt)
    return 0


def cmd_find_by_trace(args: argparse.Namespace) -> int:
    h = Helper(args)
    start_dt, end_dt, _ = resolve_window(args)
    rows = h.log_query(f'trace_id:"{args.trace_id}"', start_dt, end_dt, args.limit)
    h.print_logs(rows, f"logs for trace_id={args.trace_id}", start_dt, end_dt)
    return 0


def cmd_service_logs(args: argparse.Namespace) -> int:
    h = Helper(args)
    start_dt, end_dt, label = resolve_window(args)
    query = f'service.name:"{args.service}"'
    if args.severity:
        query = f"{query} AND severity:{normalize_severity(args.severity)}"
    rows = h.log_query(query, start_dt, end_dt, args.limit)
    h.print_logs(rows, f"service={args.service} ({label})", start_dt, end_dt)
    return 0


def cmd_services(args: argparse.Namespace) -> int:
    h = Helper(args)
    dump_json(h.trace_services(), h.compact)
    return 0


def cmd_slow_requests(args: argparse.Namespace) -> int:
    h = Helper(args)
    data = h.trace_search(
        args.service, args.operation, args.lookback, args.limit,
        f"{args.min_duration}ms",
    )
    dump_json(data, h.compact)
    return 0


def cmd_get_trace(args: argparse.Namespace) -> int:
    h = Helper(args)
    try:
        dump_json(h.trace_get(args.trace_id), h.compact)
    except MCPError as exc:
        if "404" in str(exc) or "not found" in str(exc).lower():
            print(
                f"trace {args.trace_id} not found via the single-trace endpoint.\n"
                "VictoriaTraces 单条拉取常因 trace 已过保留期、或该 id 仅出现在日志而\n"
                "未落 trace 而 404。可改用 `trace-services` + `slow-requests` 按服务/时间\n"
                "窗搜索，或回到 logs 用 `find-by-trace` 交叉验证该 trace_id 是否真有数据。",
                file=sys.stderr,
            )
            return 2
        raise
    return 0


def cmd_metric(args: argparse.Namespace) -> int:
    h = Helper(args)
    dump_json(h.metric_query(args.query), h.compact)
    return 0


def cmd_metric_inventory(args: argparse.Namespace) -> int:
    """List metric names grouped by family prefix, to spot coverage gaps.

    Surfaces whether control-plane families (mint_*) exist vs only engine
    metrics (sglang_*), and lets you grep a family with --match.
    """
    h = Helper(args)
    names = h.metric_names(args.match)
    if h.json_out:
        dump_json(names, h.compact)
        return 0
    groups: dict[str, list[str]] = {}
    for n in names:
        prefix = n.split("_", 1)[0].split(":", 1)[0]
        groups.setdefault(prefix, []).append(n)
    print(f"=== {len(names)} metrics in {len(groups)} families ===\n")
    for prefix in sorted(groups, key=lambda p: -len(groups[p])):
        members = sorted(groups[prefix])
        print(f"{prefix}_* ({len(members)})")
        if args.full:
            for m in members:
                print(f"   {m}")
        else:
            for m in members[:6]:
                print(f"   {m}")
            if len(members) > 6:
                print(f"   … +{len(members) - 6} more (--full to list)")
        print()
    return 0


def cmd_investigate(args: argparse.Namespace) -> int:
    h = Helper(args)
    if not args.request_id and not args.trace_id:
        print("provide --request-id (repeatable) or --trace-id", file=sys.stderr)
        return 2
    start_dt, end_dt, _ = resolve_window(args)
    if args.request_id:
        rows = h.log_query(_request_id_query(args.request_id), start_dt, end_dt, args.limit)
        label = args.request_id[0] if len(args.request_id) == 1 else f"{len(args.request_id)} request_ids"
        h.print_logs(rows, f"logs for request_id={label}", start_dt, end_dt)
    if args.trace_id:
        rows = h.log_query(f'trace_id:"{args.trace_id}"', start_dt, end_dt, args.limit)
        h.print_logs(rows, f"logs for trace_id={args.trace_id}", start_dt, end_dt)
        print("=== trace ===")
        try:
            dump_json(h.trace_get(args.trace_id), h.compact)
        except MCPError as exc:
            if "404" in str(exc) or "not found" in str(exc).lower():
                print(f"(trace {args.trace_id} 单条拉取 404；该 id 可能仅在日志未落 trace)", file=sys.stderr)
            else:
                raise
    return 0


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--verbose", action="store_true", help="log request methods to stderr")
    p.add_argument("--json", action="store_true", help="raw JSON output")
    p.add_argument("--compact", action="store_true", help="compact JSON")


def add_log_display(p: argparse.ArgumentParser) -> None:
    """Display flags for commands that render log rows."""
    p.add_argument(
        "--fields",
        help="comma-separated extra fields to show (default: a curated set "
        "incl. host.name/process.pid/exception.*); overrides the default set",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="do not truncate _msg or field values (see full stacktraces)",
    )


def add_window(p: argparse.ArgumentParser) -> None:
    """Precise --start/--end window for historical incident triage.

    Takes precedence over --lookback. Accepts RFC3339 (2026-06-11T04:10:00Z),
    'YYYY-MM-DD HH:MM', or Unix seconds. When 0 hits, the helper probes the
    exact window so you can tell 'wrong window / rolled off' from 'query too
    narrow' (the #718 trap: a busy day whose incident minute has no data yet).
    """
    p.add_argument("--start", help="window start (RFC3339 / 'YYYY-MM-DD HH:MM' / Unix s); overrides --lookback")
    p.add_argument("--end", help="window end (default: now when --start given)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MinT telemetry debug helper (MCP).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("recent-errors", help="recent ERROR-level logs")
    add_common(p)
    add_log_display(p)
    add_window(p)
    p.add_argument("--lookback", type=int, default=10080, help="lookback minutes (default 7d)")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_recent_errors)

    p = sub.add_parser("find-logs", help="search logs by LogsQL text")
    add_common(p)
    add_log_display(p)
    add_window(p)
    p.add_argument("query", help="LogsQL query, e.g. '_msg:\"CUDA out of memory\"'")
    p.add_argument("--severity", help="filter by severity (ERROR/WARN/INFO/DEBUG; WARNING is normalized)")
    p.add_argument("--lookback", type=int, default=10080, help="lookback minutes (default 7d)")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_find_logs)

    p = sub.add_parser("find-request", help="logs for one or more request_ids")
    add_common(p)
    add_log_display(p)
    add_window(p)
    p.add_argument("request_id", nargs="+", help="one or more request_ids (OR-matched)")
    p.add_argument("--lookback", type=int, default=10080, help="lookback minutes (default 7d)")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_find_request)

    p = sub.add_parser("find-by-trace", help="logs for a trace_id")
    add_common(p)
    add_log_display(p)
    add_window(p)
    p.add_argument("trace_id")
    p.add_argument("--lookback", type=int, default=10080, help="lookback minutes (default 7d)")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_find_by_trace)

    p = sub.add_parser("service-logs", help="logs for a service")
    add_common(p)
    add_log_display(p)
    add_window(p)
    p.add_argument("--service", default="mint")
    p.add_argument("--severity", help="filter by severity (ERROR/WARN/INFO/DEBUG)")
    p.add_argument("--lookback", type=int, default=10080, help="lookback minutes (default 7d)")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_service_logs)

    p = sub.add_parser("trace-services", help="list traced services")
    add_common(p)
    p.set_defaults(func=cmd_services)

    p = sub.add_parser("slow-requests", help="search slow traces (minDuration)")
    add_common(p)
    p.add_argument("--service", default="mint")
    p.add_argument("--operation")
    p.add_argument("--min-duration", type=int, default=1000, help="ms")
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_slow_requests)

    p = sub.add_parser("get-trace", help="fetch a trace by id")
    add_common(p)
    p.add_argument("trace_id")
    p.set_defaults(func=cmd_get_trace)

    p = sub.add_parser("metric", help="instant PromQL query")
    add_common(p)
    p.add_argument("query", help="PromQL expression")
    p.set_defaults(func=cmd_metric)

    p = sub.add_parser("metric-inventory", help="list metric names grouped by family")
    add_common(p)
    p.add_argument("--match", help="PromQL selector to filter, e.g. '{__name__=~\"mint_.*\"}'")
    p.add_argument("--full", action="store_true", help="list every metric name, not just samples")
    p.set_defaults(func=cmd_metric_inventory)

    p = sub.add_parser("investigate", help="logs (+trace) for request_id(s)/trace_id")
    add_common(p)
    add_log_display(p)
    add_window(p)
    p.add_argument("--request-id", action="append", help="request_id (repeatable, OR-matched)")
    p.add_argument("--trace-id")
    p.add_argument("--lookback", type=int, default=10080, help="lookback minutes (default 7d)")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_investigate)

    return parser


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except (MCPError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
