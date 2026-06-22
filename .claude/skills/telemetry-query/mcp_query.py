#!/usr/bin/env python3
"""
Victoria MCP query client (JSON-RPC 2.0 over Streamable HTTP).

Talks to the VictoriaLogs / VictoriaMetrics / VictoriaTraces MCP endpoints
documented in infra-cluster-iaac/USAGE.zh-CN.md.

Protocol (per endpoint, independent session):
  1. POST initialize           -> 200, response header `mcp-session-id`
  2. POST notifications/initialized (+session header) -> 202
  3. POST tools/list | tools/call (+session header)   -> 200

Tool results: result.content[0].text holds an (often escaped) JSON string,
or plain text, or "" when nothing matched. Always check result.isError first.

Auth: header `x-api-key: $MCP_API_KEY`. Never print the key.
Config is read from environment; `.env` next to this file is auto-loaded.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from json import JSONDecodeError
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "2024-11-05"

DEFAULT_ENDPOINTS = {
    "logs": "https://otelmcp.macaron.xin/logs/mcp",
    "metrics": "https://otelmcp.macaron.xin/metrics/mcp",
    "traces": "https://otelmcp.macaron.xin/traces/mcp",
}

ENV_NAMES = {
    "logs": "MCP_LOGS_URL",
    "metrics": "MCP_METRICS_URL",
    "traces": "MCP_TRACES_URL",
}


def load_dotenv() -> None:
    """Load KEY=VALUE pairs from a .env file next to this script (no override)."""
    env_path = Path(__file__).with_name(".env")
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def endpoint_url(kind: str) -> str:
    return os.environ.get(ENV_NAMES[kind], DEFAULT_ENDPOINTS[kind]).rstrip("/")


def get_api_key() -> str:
    key = os.environ.get("MCP_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "MCP_API_KEY not set. Put it in .claude/skills/telemetry-query/.env "
            "or export it. Source: K8s Secret monitoring/victorialogs-mcp-api-keys "
            "(client ai-client-1)."
        )
    return key


class MCPError(RuntimeError):
    """Tool-level error (result.isError) or protocol error."""


@dataclass
class MCPClient:
    """JSON-RPC 2.0 MCP client for one Victoria endpoint."""

    base_url: str
    api_key: str
    timeout: float = 30.0
    verbose: bool = False
    _session_id: str | None = field(default=None, init=False)
    _next_id: int = field(default=1, init=False)

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "x-api-key": self.api_key,
            "User-Agent": "mint-mcp-query/2",
        }
        if self._session_id:
            headers["mcp-session-id"] = self._session_id
        return headers

    def _post(self, payload: dict[str, Any]) -> tuple[int, dict[str, str], str]:
        body = json.dumps(payload).encode("utf-8")
        if self.verbose:
            method = payload.get("method")
            print(f"POST {self.base_url} method={method}", file=sys.stderr)
        req = urllib.request.Request(
            self.base_url, data=body, headers=self._headers(), method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.status, dict(resp.headers), resp.read().decode(
                    "utf-8", errors="replace"
                )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise MCPError(f"HTTP {exc.code} from {self.base_url}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise MCPError(f"request failed for {self.base_url}: {exc.reason}") from exc

    @staticmethod
    def _parse_jsonrpc(text: str) -> dict[str, Any]:
        """Parse a JSON-RPC response; tolerate SSE (`data:`) framing."""
        text = text.strip()
        if not text:
            return {}
        if text.startswith("data:"):
            chunks = [
                ln[len("data:"):].strip()
                for ln in text.splitlines()
                if ln.startswith("data:")
            ]
            text = "".join(chunks)
        try:
            return json.loads(text)
        except JSONDecodeError as exc:
            raise MCPError(f"non-JSON response: {exc.msg}: {text[:200]}") from exc

    def _ensure_session(self) -> None:
        if self._session_id is not None:
            return
        init_payload = {
            "jsonrpc": "2.0",
            "id": self._next_id,
            "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "mint-mcp-query", "version": "2.0"},
            },
        }
        self._next_id += 1
        status, headers, text = self._post(init_payload)
        session = headers.get("mcp-session-id") or headers.get("Mcp-Session-Id")
        if not session:
            # Some servers embed it only in body; fall back gracefully.
            data = self._parse_jsonrpc(text)
            session = (data.get("result") or {}).get("sessionId")
        if not session:
            raise MCPError(
                f"initialize did not return a session id (HTTP {status})"
            )
        self._session_id = session
        # Standard MCP handshake completion (server returns 202).
        self._post(
            {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        )

    def _rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._ensure_session()
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id,
            "method": method,
            "params": params,
        }
        self._next_id += 1
        _, _, text = self._post(payload)
        data = self._parse_jsonrpc(text)
        if "error" in data:
            err = data["error"]
            raise MCPError(f"JSON-RPC error {err.get('code')}: {err.get('message')}")
        return data.get("result", {})

    def list_tools(self) -> list[dict[str, Any]]:
        result = self._rpc("tools/list", {})
        return result.get("tools", [])

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool and return parsed payload. Raises MCPError on isError."""
        clean = {k: v for k, v in arguments.items() if v is not None}
        result = self._rpc("tools/call", {"name": name, "arguments": clean})
        content = result.get("content") or []
        texts = [c.get("text", "") for c in content if c.get("type") == "text"]
        joined = "\n".join(t for t in texts if t)
        if result.get("isError"):
            raise MCPError(f"tool '{name}' error: {joined or '(no detail)'}")
        return _maybe_json(joined)


def _maybe_json(text: str) -> Any:
    """Tool text payloads are often escaped JSON; parse when possible."""
    text = text.strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except JSONDecodeError:
        return text


def dump_json(data: Any, compact: bool) -> None:
    if compact:
        json.dump(data, sys.stdout, separators=(",", ":"), ensure_ascii=False)
    else:
        json.dump(data, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


def make_client(kind: str, args: argparse.Namespace) -> MCPClient:
    return MCPClient(
        base_url=endpoint_url(kind),
        api_key=get_api_key(),
        timeout=args.timeout,
        verbose=args.verbose,
    )


# Free-text/identifier args that must always stay strings on every endpoint,
# even when they happen to look numeric.
ALWAYS_STRING_ARGS = frozenset(
    {"query", "match", "field", "service", "operation", "service_name", "trace_id"}
)

# Time-ish args whose JSON type DIFFERS by endpoint:
#  - logs/metrics: the backend rejects numeric timestamps (`... wrong type: float64`),
#    so start/end/time/step must be sent as strings (RFC3339 or Unix seconds).
#  - traces: the schema declares these as numbers (Unix MILLISECONDS); sending a
#    string fails with `... wrong type: string`.
TIME_ARGS_BY_ENDPOINT: dict[str, frozenset[str]] = {
    "logs": frozenset({"start", "end", "time", "step", "lookback"}),
    "metrics": frozenset({"start", "end", "time", "step"}),
    # traces start/end/minDuration/maxDuration/lookback are numeric -> let json.loads run.
    "traces": frozenset(),
}


def parse_kv_args(items: list[str] | None, endpoint: str) -> dict[str, Any]:
    """Parse repeated --arg key=value into a dict, with endpoint-aware typing.

    Identifier/free-text args always stay strings. Time args are forced to
    strings on logs/metrics (the backend rejects numeric timestamps) but parsed
    as JSON on traces (the schema wants numeric Unix milliseconds). Everything
    else is parsed as JSON when possible so `limit=50` / `nocache=false` work.
    """
    string_keys = ALWAYS_STRING_ARGS | TIME_ARGS_BY_ENDPOINT.get(endpoint, frozenset())
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"invalid --arg '{item}', use key=value")
        key, _, value = item.partition("=")
        key = key.strip()
        if key in string_keys:
            out[key] = value
            continue
        try:
            out[key] = json.loads(value)
        except JSONDecodeError:
            out[key] = value
    return out


# ===== Generic commands =====


def cmd_tools(args: argparse.Namespace) -> int:
    """List tools for one endpoint (name + required params)."""
    client = make_client(args.endpoint, args)
    tools = client.list_tools()
    if args.raw:
        dump_json(tools, args.compact)
        return 0
    summary = []
    for tool in tools:
        schema = tool.get("inputSchema") or {}
        props = list((schema.get("properties") or {}).keys())
        required = schema.get("required") or []
        summary.append(
            {
                "name": tool.get("name"),
                "required": required,
                "params": props,
                "description": (tool.get("description") or "").strip()[:120],
            }
        )
    dump_json(summary, args.compact)
    return 0


def cmd_call(args: argparse.Namespace) -> int:
    """Call an arbitrary tool with --arg key=value pairs."""
    client = make_client(args.endpoint, args)
    arguments = parse_kv_args(args.arg, args.endpoint)
    result = client.call_tool(args.tool, arguments)
    dump_json(result, args.compact)
    return 0


# ===== Convenience commands (high-frequency log triage) =====


def _rfc3339_since(minutes: int) -> str:
    from datetime import datetime, timedelta, timezone

    dt = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def cmd_hits(args: argparse.Namespace) -> int:
    """Bucket log hit counts over time — probe where data actually lives."""
    client = make_client("logs", args)
    start = args.start or _rfc3339_since(args.since)
    result = client.call_tool(
        "hits",
        {"query": args.query, "start": start, "end": args.end, "step": args.step},
    )
    dump_json(result, args.compact)
    return 0


def cmd_field_names(args: argparse.Namespace) -> int:
    """List field names present for a LogsQL query (discover real field names)."""
    client = make_client("logs", args)
    start = args.start or _rfc3339_since(args.since)
    result = client.call_tool(
        "field_names", {"query": args.query, "start": start, "end": args.end}
    )
    dump_json(result, args.compact)
    return 0


def cmd_field_values(args: argparse.Namespace) -> int:
    """List observed values of one field (e.g. which severities/services exist)."""
    client = make_client("logs", args)
    start = args.start or _rfc3339_since(args.since)
    result = client.call_tool(
        "field_values",
        {"query": args.query, "field": args.field, "start": start, "end": args.end, "limit": args.limit},
    )
    dump_json(result, args.compact)
    return 0


def add_common_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--timeout", type=float, default=30.0, help="request timeout seconds")
    parser.add_argument("--compact", action="store_true", help="emit compact JSON")
    parser.add_argument("--verbose", action="store_true", help="log request methods to stderr")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Victoria MCP query client (JSON-RPC 2.0 over Streamable HTTP)."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    tools = sub.add_parser("tools", help="list available tools for an endpoint")
    add_common_flags(tools)
    tools.add_argument("endpoint", choices=["logs", "metrics", "traces"])
    tools.add_argument("--raw", action="store_true", help="dump full tool schemas")
    tools.set_defaults(func=cmd_tools)

    call = sub.add_parser("call", help="call any tool with key=value arguments")
    add_common_flags(call)
    call.add_argument("endpoint", choices=["logs", "metrics", "traces"])
    call.add_argument("tool", help="tool name (see `tools <endpoint>`)")
    call.add_argument(
        "--arg",
        action="append",
        metavar="KEY=VALUE",
        help="tool argument; value parsed as JSON when possible. Repeatable.",
    )
    call.set_defaults(func=cmd_call)

    hits = sub.add_parser("hits", help="logs: bucket hit counts over time (probe data boundary)")
    add_common_flags(hits)
    hits.add_argument("--query", default="service.name:\"mint\"", help="LogsQL query (default: mint service)")
    hits.add_argument("--since", type=int, default=43200, help="lookback minutes when --start omitted (default 30d)")
    hits.add_argument("--start", help="explicit RFC3339/Unix start (overrides --since)")
    hits.add_argument("--end", help="RFC3339/Unix end")
    hits.add_argument("--step", default="1d", help="bucket width (default 1d)")
    hits.set_defaults(func=cmd_hits)

    fnames = sub.add_parser("field-names", help="logs: list field names for a query")
    add_common_flags(fnames)
    fnames.add_argument("--query", default="*", help="LogsQL query (default: all)")
    fnames.add_argument("--since", type=int, default=10080, help="lookback minutes when --start omitted (default 7d)")
    fnames.add_argument("--start", help="explicit RFC3339/Unix start (overrides --since)")
    fnames.add_argument("--end", help="RFC3339/Unix end")
    fnames.set_defaults(func=cmd_field_names)

    fvalues = sub.add_parser("field-values", help="logs: list observed values of one field")
    add_common_flags(fvalues)
    fvalues.add_argument("field", help="field name, e.g. severity / service.name")
    fvalues.add_argument("--query", default="*", help="LogsQL query (default: all)")
    fvalues.add_argument("--since", type=int, default=10080, help="lookback minutes when --start omitted (default 7d)")
    fvalues.add_argument("--start", help="explicit RFC3339/Unix start (overrides --since)")
    fvalues.add_argument("--end", help="RFC3339/Unix end")
    fvalues.add_argument("--limit", type=int, default=100, help="max values")
    fvalues.set_defaults(func=cmd_field_values)

    return parser


def main() -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except (MCPError, RuntimeError, argparse.ArgumentTypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
