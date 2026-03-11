#!/usr/bin/env python3
"""Unified operations CLI for Mint/tinker-server.

This tool is intended to replace ad-hoc scripts in `scripts/wip` for routine
operations and recovery workflows.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html as html_lib
import http.server
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


DEFAULT_REMOTE_PYTHON = "/root/tinker_project/tinker-server-auth/.venv31213/bin/python"
DEFAULT_SUPERVISOR_PROGRAM = "tinker-server-auth"

# Ensure repo root is importable when this file is executed as:
#   python scripts/ops/mint_ops.py ...
# In that mode, sys.path[0] is scripts/ops, so sibling package imports like
# `tinker_server.*` can fail without this.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class _ReusableThreadingHTTPServer(http.server.ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def _parse_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(_fmt(x) for x in row) + " |")
    return out


def _run(cmd: list[str], *, timeout_s: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)


def _strip_flag_with_value(argv: list[str], flag: str) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == flag:
            i += 2
            continue
        if token.startswith(f"{flag}="):
            i += 1
            continue
        out.append(token)
        i += 1
    return out


def _rewrite_output_path_flag(argv: list[str], flag: str, new_path: str) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == flag:
            out.append(token)
            out.append(new_path)
            i += 2
            continue
        if token.startswith(f"{flag}="):
            out.append(f"{flag}={new_path}")
            i += 1
            continue
        out.append(token)
        i += 1
    return out


def _argv_has_flag(argv: list[str], flag: str) -> bool:
    for token in argv:
        if token == flag or token.startswith(f"{flag}="):
            return True
    return False


def _rewrite_or_append_output_flag(argv: list[str], flag: str, remote_path: str) -> list[str]:
    if _argv_has_flag(argv, flag):
        return _rewrite_output_path_flag(argv, flag, remote_path)
    return argv + [flag, remote_path]


def _extract_port_from_tokens(tokens: list[str], flag: str) -> int | None:
    for i, tok in enumerate(tokens):
        if tok == flag and i + 1 < len(tokens):
            try:
                return int(tokens[i + 1])
            except ValueError:
                return None
        if tok.startswith(f"{flag}="):
            try:
                return int(tok.split("=", 1)[1])
            except ValueError:
                return None
    return None


def _is_mint_ops_server_cmd(tokens: list[str], target_port: int) -> bool:
    if not any("mint_ops.py" in t for t in tokens):
        return False

    is_status_serve = ("status" in tokens) and ("--serve" in tokens or "-s" in tokens)
    is_ops_server = "ops-server" in tokens
    if not (is_status_serve or is_ops_server):
        return False

    port = _extract_port_from_tokens(tokens, "--serve-port")
    if port is None:
        port = _extract_port_from_tokens(tokens, "--server-port")
    if port is None:
        port = 8765
    return int(port) == int(target_port)


def _kill_stale_mint_ops_servers(port: int) -> list[int]:
    target_port = int(port)
    me = os.getpid()
    stale: list[int] = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        pid = int(p.name)
        if pid == me:
            continue
        try:
            raw = (p / "cmdline").read_bytes()
        except Exception:
            continue
        if not raw:
            continue
        tokens = [x.decode("utf-8", "ignore") for x in raw.split(b"\x00") if x]
        if not _is_mint_ops_server_cmd(tokens, target_port):
            continue
        stale.append(pid)

    if not stale:
        return []

    for pid in stale:
        try:
            os.kill(pid, 15)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass
    time.sleep(0.6)
    for pid in stale:
        try:
            os.kill(pid, 0)
            os.kill(pid, 9)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass
    return stale


def _run_remote_with_local_copy(
    *,
    args: argparse.Namespace,
    argv: list[str],
    local_md_out: Path | None,
    local_json_out: Path | None,
    local_html_out: Path | None,
) -> int:
    wants_copy = bool(local_md_out or local_json_out or local_html_out)
    remote_argv = list(argv)
    remote_tmp_paths: dict[str, str] = {}

    if wants_copy:
        suffix = f"{int(time.time())}_{os.getpid()}"
        if local_md_out:
            remote_md = f"/tmp/mint_ops_{suffix}.md"
            remote_tmp_paths["md"] = remote_md
            remote_argv = _rewrite_or_append_output_flag(remote_argv, "--md-out", remote_md)
        if local_json_out:
            remote_json = f"/tmp/mint_ops_{suffix}.json"
            remote_tmp_paths["json"] = remote_json
            remote_argv = _rewrite_or_append_output_flag(remote_argv, "--json-out", remote_json)
        if local_html_out:
            remote_html = f"/tmp/mint_ops_{suffix}.html"
            remote_tmp_paths["html"] = remote_html
            remote_argv = _rewrite_or_append_output_flag(remote_argv, "--html-out", remote_html)

    cmd = ["ssh", args.host, args.remote_python, os.path.abspath(__file__)] + remote_argv
    remote = subprocess.run(cmd)
    if remote.returncode != 0:
        return int(remote.returncode)

    if local_md_out and "md" in remote_tmp_paths:
        local_md_out.parent.mkdir(parents=True, exist_ok=True)
        cp = subprocess.run(["scp", f"{args.host}:{remote_tmp_paths['md']}", str(local_md_out)])
        if cp.returncode != 0:
            return int(cp.returncode)
        subprocess.run(["ssh", args.host, "rm", "-f", remote_tmp_paths["md"]], check=False)
        print(f"downloaded markdown to local: {local_md_out}")

    if local_json_out and "json" in remote_tmp_paths:
        local_json_out.parent.mkdir(parents=True, exist_ok=True)
        cp = subprocess.run(["scp", f"{args.host}:{remote_tmp_paths['json']}", str(local_json_out)])
        if cp.returncode != 0:
            return int(cp.returncode)
        subprocess.run(["ssh", args.host, "rm", "-f", remote_tmp_paths["json"]], check=False)
        print(f"downloaded json to local: {local_json_out}")

    if local_html_out and "html" in remote_tmp_paths:
        local_html_out.parent.mkdir(parents=True, exist_ok=True)
        cp = subprocess.run(["scp", f"{args.host}:{remote_tmp_paths['html']}", str(local_html_out)])
        if cp.returncode != 0:
            return int(cp.returncode)
        subprocess.run(["ssh", args.host, "rm", "-f", remote_tmp_paths["html"]], check=False)
        print(f"downloaded html to local: {local_html_out}")

    return 0


def _serve_status_html(
    *,
    html_path: Path,
    bind: str,
    port: int,
    refresh_fn: Callable[[], int] | None,
    kill_stale_ops: bool,
) -> int:
    html_path = html_path.resolve()
    root_dir = html_path.parent
    html_name = html_path.name

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a: Any, **kw: Any) -> None:
            super().__init__(*a, directory=str(root_dir), **kw)

        def end_headers(self) -> None:
            self.send_header("Cache-Control", "no-store, max-age=0")
            super().end_headers()

        def do_GET(self) -> None:
            if self.path in ("/", ""):
                self.send_response(302)
                self.send_header("Location", f"/{html_name}")
                self.end_headers()
                return
            return super().do_GET()

        def do_POST(self) -> None:
            if self.path != "/refresh":
                self.send_error(404, "Not Found")
                return
            if refresh_fn is None:
                payload = {"ok": False, "error": "refresh is not enabled for this report"}
                raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(405)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return

            rc = 1
            err = ""
            try:
                rc = int(refresh_fn())
            except Exception as e:
                rc = 1
                err = f"{type(e).__name__}: {e}"

            ok = rc == 0
            payload = {
                "ok": ok,
                "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "error": err if not ok else "",
            }
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200 if ok else 500)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    if kill_stale_ops:
        stale = _kill_stale_mint_ops_servers(int(port))
        if stale:
            print(f"killed stale mint_ops servers on port {int(port)}: {stale}")

    try:
        server = _ReusableThreadingHTTPServer((bind, int(port)), _Handler)
    except OSError as e:
        if getattr(e, "errno", None) == 98:
            raise RuntimeError(
                f"port {int(port)} is still in use; run `ss -ltnp | rg :{int(port)}\\b` and kill stale mint_ops/ssh tunnel"
            ) from e
        raise
    url = f"http://{bind}:{port}/{html_name}"
    print(f"serving status html at: {url}")
    print("press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _serve_status_api(
    *,
    bind: str,
    port: int,
    actor_limit: int,
    cache_ttl_s: float,
    snapshot_fn: Callable[[], dict[str, Any]],
    ops_args: argparse.Namespace | None = None,
    kill_stale_ops: bool = True,
) -> int:
    cache_lock = threading.Lock()
    cache: dict[str, Any] = {"snapshot": None, "updated_ts": 0.0}
    actor_limit = max(1, int(actor_limit))
    cache_ttl_s = max(0.0, float(cache_ttl_s))

    def _snapshot(force: bool) -> dict[str, Any]:
        now = time.time()
        with cache_lock:
            cached = cache.get("snapshot")
            updated_ts = float(cache.get("updated_ts", 0.0))
            if not force and cached is not None and (now - updated_ts) <= cache_ttl_s:
                return cached

        fresh = snapshot_fn()
        with cache_lock:
            cache["snapshot"] = fresh
            cache["updated_ts"] = time.time()
        return fresh

    def _query_truthy(query: dict[str, list[str]], key: str) -> bool:
        v = (query.get(key, [""])[0] or "").strip().lower()
        return v in {"1", "true", "yes", "y", "on"}

    class _Handler(http.server.BaseHTTPRequestHandler):
        server_version = "MintOpsHTTP/1.0"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(int(status))
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self._send_bytes(status, raw, "application/json; charset=utf-8")

        def _read_json_body(self) -> dict[str, Any]:
            raw_len = self.headers.get("Content-Length", "0").strip() or "0"
            try:
                n = max(0, int(raw_len))
            except ValueError:
                n = 0
            raw = self.rfile.read(n) if n > 0 else b""
            if not raw:
                return {}
            try:
                payload = json.loads(raw.decode("utf-8", "replace"))
            except json.JSONDecodeError as e:
                raise ValueError(f"invalid json body: {e}") from e
            if not isinstance(payload, dict):
                raise ValueError("json body must be object")
            return payload

        def _require_ops_args(self) -> argparse.Namespace:
            if ops_args is None:
                raise RuntimeError("ops operations are disabled")
            return ops_args

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            query = urllib.parse.parse_qs(parsed.query)
            force_refresh = _query_truthy(query, "refresh")

            if path in ("", "/"):
                self.send_response(302)
                self.send_header("Location", "/status.html")
                self.end_headers()
                return

            if path in ("/healthz", "/api/v1/healthz"):
                self._send_json(200, {"ok": True, "service": "mint-ops", "now": dt.datetime.now(dt.timezone.utc).isoformat()})
                return

            if path in ("/status.html", "/status", "/api/v1/status.html"):
                try:
                    snap = _snapshot(force=force_refresh)
                    html = _status_html(snap, actor_limit=actor_limit).encode("utf-8")
                    self._send_bytes(200, html, "text/html; charset=utf-8")
                except Exception as e:
                    self._send_json(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})
                return

            if path in ("/status.md", "/api/v1/status.md"):
                try:
                    snap = _snapshot(force=force_refresh)
                    md = _status_markdown(snap, actor_limit=actor_limit).encode("utf-8")
                    self._send_bytes(200, md, "text/markdown; charset=utf-8")
                except Exception as e:
                    self._send_json(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})
                return

            if path in ("/status.json", "/api/v1/status", "/api/v1/status.json"):
                fmt = (query.get("format", ["json"])[0] or "json").strip().lower()
                try:
                    snap = _snapshot(force=force_refresh)
                    if fmt == "json":
                        self._send_json(200, snap)
                    elif fmt in {"md", "markdown"}:
                        md = _status_markdown(snap, actor_limit=actor_limit).encode("utf-8")
                        self._send_bytes(200, md, "text/markdown; charset=utf-8")
                    elif fmt == "html":
                        html = _status_html(snap, actor_limit=actor_limit).encode("utf-8")
                        self._send_bytes(200, html, "text/html; charset=utf-8")
                    else:
                        self._send_json(400, {"ok": False, "error": f"unsupported format={fmt!r}"})
                except Exception as e:
                    self._send_json(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})
                return

            self._send_json(404, {"ok": False, "error": "not found"})

        def do_POST(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path

            if path in ("/refresh", "/api/v1/status/refresh"):
                try:
                    snap = _snapshot(force=True)
                    self._send_json(
                        200,
                        {
                            "ok": True,
                            "generated_at_utc": snap.get("generated_at_utc", dt.datetime.now(dt.timezone.utc).isoformat()),
                        },
                    )
                except Exception as e:
                    self._send_json(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})
                return

            if path in ("/api/v1/deploy/actor/kill",):
                try:
                    args = self._require_ops_args()
                    payload = self._read_json_body()
                    actor_type = str(payload.get("actor_type", "")).strip()
                    if actor_type not in {"vllm", "megatron", "dense", "all"}:
                        self._send_json(400, {"ok": False, "error": "actor_type must be one of: vllm, megatron, dense, all"})
                        return
                    model_name_raw = payload.get("model_name", None)
                    model_name = str(model_name_raw).strip() if model_name_raw is not None else None
                    data = _actor_kill_operation(args, actor_type=actor_type, model_name=model_name or None)
                    self._send_json(200, {"ok": True, "result": data})
                except ValueError as e:
                    self._send_json(400, {"ok": False, "error": str(e)})
                except Exception as e:
                    self._send_json(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})
                return

            if path in ("/api/v1/deploy/actor/rebuild",):
                try:
                    args = self._require_ops_args()
                    payload = self._read_json_body()
                    kind = str(payload.get("kind", "training")).strip().lower()
                    if kind not in {"vllm", "training"}:
                        self._send_json(400, {"ok": False, "error": "kind must be one of: vllm, training"})
                        return

                    models_raw = payload.get("models", [])
                    if isinstance(models_raw, list):
                        models = [str(x).strip() for x in models_raw if str(x).strip()]
                    else:
                        models = []
                    if not models:
                        model_single = str(payload.get("model", "")).strip()
                        if model_single:
                            models = [model_single]
                    if not models:
                        self._send_json(400, {"ok": False, "error": "at least one model is required (model or models[])"})
                        return

                    sample_ping = bool(payload.get("sample_ping", False))
                    lora_rank = int(payload.get("lora_rank", 16))
                    poll_timeout_s = float(payload.get("poll_timeout_s", 900.0))
                    poll_interval_s = float(payload.get("poll_interval_s", 2.0))
                    data = _actor_rebuild_operation(
                        args,
                        kind=kind,
                        models=models,
                        sample_ping=sample_ping,
                        lora_rank=lora_rank,
                        poll_timeout_s=poll_timeout_s,
                        poll_interval_s=poll_interval_s,
                    )
                    self._send_json(200, data)
                except ValueError as e:
                    self._send_json(400, {"ok": False, "error": str(e)})
                except Exception as e:
                    self._send_json(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})
                return

            if path in ("/api/v1/deploy/pg/remove",):
                try:
                    args = self._require_ops_args()
                    payload = self._read_json_body()
                    names_raw = payload.get("names", [])
                    names = [str(x).strip() for x in names_raw if str(x).strip()] if isinstance(names_raw, list) else []
                    state_raw = str(payload.get("state", "")).strip().upper()
                    state = state_raw if state_raw in {"PENDING", "CREATED"} else None
                    only_gpu = bool(payload.get("only_gpu", False))
                    apply = bool(payload.get("apply", False))
                    data = _pg_remove_operation(
                        args,
                        names=names,
                        state=state,
                        only_gpu=only_gpu,
                        apply=apply,
                    )
                    data["ok"] = True
                    self._send_json(200, data)
                except ValueError as e:
                    self._send_json(400, {"ok": False, "error": str(e)})
                except Exception as e:
                    self._send_json(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})
                return

            if path in ("/api/v1/deploy/server/restart",):
                try:
                    args = self._require_ops_args()
                    payload = self._read_json_body()
                    clean_dirty = bool(payload.get("clean_dirty", False))
                    wait_healthz_s = float(payload.get("wait_healthz_s", 60.0))
                    data = _server_restart_operation(
                        args,
                        clean_dirty=clean_dirty,
                        wait_healthz_s=wait_healthz_s,
                    )
                    self._send_json(200, data)
                except ValueError as e:
                    self._send_json(400, {"ok": False, "error": str(e)})
                except Exception as e:
                    self._send_json(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})
                return

            if path in ("/api/v1/deploy/node/scale", "/api/v1/cronjob/apply"):
                self._send_json(501, {"ok": False, "error": "TODO: not implemented yet"})
                return

            self._send_json(404, {"ok": False, "error": "not found"})

    if kill_stale_ops:
        stale = _kill_stale_mint_ops_servers(int(port))
        if stale:
            print(f"killed stale mint_ops servers on port {int(port)}: {stale}")

    try:
        server = _ReusableThreadingHTTPServer((bind, int(port)), _Handler)
    except OSError as e:
        if getattr(e, "errno", None) == 98:
            raise RuntimeError(
                f"port {int(port)} is still in use; run `ss -ltnp | rg :{int(port)}\\b` and kill stale mint_ops/ssh tunnel"
            ) from e
        raise
    print(f"mint-ops server listening at: http://{bind}:{int(port)}/")
    print("status page: /status.html")
    print("status api:  /api/v1/status?format=json|md|html")
    print(
        "deploy api:  /api/v1/deploy/actor/kill | /api/v1/deploy/actor/rebuild | "
        "/api/v1/deploy/pg/remove | /api/v1/deploy/server/restart"
    )
    print("press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _maybe_exec_remote(args: argparse.Namespace) -> int | None:
    if not args.host:
        return None

    argv = sys.argv[1:]
    argv = _strip_flag_with_value(argv, "--host")
    argv = _strip_flag_with_value(argv, "--remote-python")
    local_md_out = getattr(args, "md_out", None)
    local_json_out = getattr(args, "json_out", None)
    local_html_out = getattr(args, "html_out", None)

    if args.subcommand == "status" and bool(getattr(args, "serve", False)) and bool(getattr(args, "direct", False)):
        local_port = int(getattr(args, "direct_local_port", 0) or args.serve_port)
        remote_port = int(args.serve_port)
        argv = [t for t in argv if t not in ("--direct",)]
        argv = _strip_flag_with_value(argv, "--direct-local-port")
        argv = _strip_flag_with_value(argv, "--json-out")
        argv = _strip_flag_with_value(argv, "--md-out")
        argv = _strip_flag_with_value(argv, "--html-out")
        print(f"direct tunnel url: http://127.0.0.1:{local_port}/status.html")
        print(f"remote endpoint: {args.host}:127.0.0.1:{remote_port}")
        cmd = [
            "ssh",
            "-L",
            f"{local_port}:127.0.0.1:{remote_port}",
            args.host,
            args.remote_python,
            os.path.abspath(__file__),
        ] + argv
        os.execvp("ssh", cmd)
        return None

    if args.subcommand == "status" and bool(getattr(args, "serve", False)):
        if local_html_out is None:
            local_html_out = Path("mint_ops_status.html")
            args.html_out = local_html_out
        # Do not start a remote server; local serve will handle refresh by pulling remotely.
        argv = [t for t in argv if t not in ("--serve", "-s")]
        argv = _strip_flag_with_value(argv, "--serve-port")
        argv = _strip_flag_with_value(argv, "--serve-bind")

        def _refresh_remote() -> int:
            return _run_remote_with_local_copy(
                args=args,
                argv=argv,
                local_md_out=Path(local_md_out) if local_md_out else None,
                local_json_out=Path(local_json_out) if local_json_out else None,
                local_html_out=Path(local_html_out) if local_html_out else None,
            )

        rc = _refresh_remote()
        if rc != 0:
            return rc

        return _serve_status_html(
            html_path=Path(local_html_out),
            bind=str(args.serve_bind),
            port=int(args.serve_port),
            refresh_fn=_refresh_remote,
        )

    wants_copy = bool(local_md_out or local_json_out or local_html_out)
    if not wants_copy:
        cmd = ["ssh", args.host, args.remote_python, os.path.abspath(__file__)] + argv
        os.execvp("ssh", cmd)
        return None

    return _run_remote_with_local_copy(
        args=args,
        argv=argv,
        local_md_out=Path(local_md_out) if local_md_out else None,
        local_json_out=Path(local_json_out) if local_json_out else None,
        local_html_out=Path(local_html_out) if local_html_out else None,
    )


def _parse_environ_bytes(raw: bytes) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in raw.split(b"\x00"):
        if b"=" not in item:
            continue
        k, v = item.split(b"=", 1)
        env[k.decode("utf-8", "ignore")] = v.decode("utf-8", "ignore")
    return env


def _find_run_server_processes(program_name: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / "cmdline").read_bytes()
            if b"scripts/run_server.py" not in cmd:
                continue
            env = _parse_environ_bytes((p / "environ").read_bytes())
        except Exception:
            continue

        out.append(
            {
                "pid": int(p.name),
                "cmdline": cmd.replace(b"\x00", b" ").decode("utf-8", "ignore").strip(),
                "supervisor_process_name": env.get("SUPERVISOR_PROCESS_NAME"),
                "supervisor_group_name": env.get("SUPERVISOR_GROUP_NAME"),
                "is_supervisor_managed": env.get("SUPERVISOR_PROCESS_NAME") == program_name,
                "env": env,
            }
        )

    out.sort(key=lambda x: x["pid"])
    return out


def _pick_primary_server_process(program_name: str) -> dict[str, Any]:
    ps = _find_run_server_processes(program_name)
    if not ps:
        raise RuntimeError("No scripts/run_server.py process found")
    ps.sort(key=lambda x: (x["is_supervisor_managed"], x["pid"]), reverse=True)
    return ps[0]


def _resolve_api_key(args: argparse.Namespace, *, required: bool) -> str | None:
    if args.no_auth:
        if required:
            raise RuntimeError("API key required but --no-auth is set")
        return None

    if args.api_key:
        return args.api_key

    env_key = os.environ.get("TINKER_API_KEY") or os.environ.get("MINT_API_KEY")
    if env_key:
        return env_key

    try:
        proc = _pick_primary_server_process(args.program)
    except Exception:
        if required:
            raise
        return None

    key = proc["env"].get("TINKER_API_KEY")
    if key:
        return key

    if required:
        raise RuntimeError("Could not resolve API key from args/env/server process")
    return None


def _http_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None,
    headers: dict[str, str] | None,
    timeout_s: float,
) -> tuple[int, dict[str, Any] | list[Any] | str]:
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url=url, data=body, method=method.upper())
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", "replace")
            if not raw:
                return resp.status, ""
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw


def _base_url(args: argparse.Namespace) -> str:
    return f"http://localhost:{args.port}"


def _admin_headers(args: argparse.Namespace, *, required: bool) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    key = _resolve_api_key(args, required=required)
    if key:
        headers["X-API-Key"] = key
    return headers


def _api_get_actors(args: argparse.Namespace, *, actor_type: str | None, model_name: str | None) -> dict[str, Any]:
    q: list[str] = []
    if actor_type:
        q.append(f"type={urllib.parse.quote(actor_type)}")
    if model_name:
        q.append(f"model_name={urllib.parse.quote(model_name)}")
    suffix = f"?{'&'.join(q)}" if q else ""
    url = _base_url(args) + f"/api/v1/actors{suffix}"
    st, data = _http_json(
        "GET",
        url,
        payload=None,
        headers=_admin_headers(args, required=True),
        timeout_s=args.timeout_s,
    )
    if st != 200 or not isinstance(data, dict):
        raise RuntimeError(f"GET /actors failed status={st} body={data!r}")
    return data


def _api_healthz(args: argparse.Namespace) -> tuple[int, dict[str, Any] | list[Any] | str]:
    return _http_json(
        "GET",
        _base_url(args) + "/api/v1/healthz",
        payload=None,
        headers=None,
        timeout_s=args.timeout_s,
    )


def _api_capabilities(args: argparse.Namespace) -> tuple[int, dict[str, Any] | list[Any] | str]:
    headers: dict[str, str] = {}
    if not args.no_auth:
        key = _resolve_api_key(args, required=False)
        if key:
            headers["X-API-Key"] = key
    return _http_json(
        "GET",
        _base_url(args) + "/api/v1/get_server_capabilities",
        payload=None,
        headers=headers or None,
        timeout_s=args.timeout_s,
    )


def _api_create_session(args: argparse.Namespace, *, tag: str) -> str:
    st, data = _http_json(
        "POST",
        _base_url(args) + "/api/v1/create_session",
        payload={
            "tags": [tag],
            "user_metadata": {},
            "sdk_version": "scripts/ops/mint_ops.py",
        },
        headers=_admin_headers(args, required=True),
        timeout_s=args.timeout_s,
    )
    if st != 200 or not isinstance(data, dict) or not data.get("session_id"):
        raise RuntimeError(f"create_session failed status={st} body={data!r}")
    return str(data["session_id"])


def _api_create_sampling_session(args: argparse.Namespace, *, session_id: str, model: str) -> str:
    st, data = _http_json(
        "POST",
        _base_url(args) + "/api/v1/create_sampling_session",
        payload={
            "session_id": session_id,
            "sampling_session_seq_id": 0,
            "base_model": model,
        },
        headers=_admin_headers(args, required=True),
        timeout_s=max(args.timeout_s, 60.0),
    )
    if st != 200 or not isinstance(data, dict) or not data.get("sampling_session_id"):
        raise RuntimeError(f"create_sampling_session failed status={st} body={data!r}")
    return str(data["sampling_session_id"])


def _api_asample_ping(
    args: argparse.Namespace,
    *,
    sampling_session_id: str,
    poll_timeout_s: float,
    poll_interval_s: float,
) -> dict[str, Any]:
    st, fut = _http_json(
        "POST",
        _base_url(args) + "/api/v1/asample",
        payload={
            "sampling_session_id": sampling_session_id,
            "num_samples": 1,
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3, 4]}]},
            "sampling_params": {"max_tokens": 1, "temperature": 0.0},
        },
        headers=_admin_headers(args, required=True),
        timeout_s=args.timeout_s,
    )
    if st != 200 or not isinstance(fut, dict) or not fut.get("request_id"):
        raise RuntimeError(f"asample failed status={st} body={fut!r}")
    request_id = str(fut["request_id"])
    return _poll_future(
        args,
        request_id=request_id,
        timeout_s=poll_timeout_s,
        interval_s=poll_interval_s,
    )


def _api_create_model(
    args: argparse.Namespace,
    *,
    session_id: str,
    model: str,
    lora_rank: int,
) -> dict[str, Any]:
    st, data = _http_json(
        "POST",
        _base_url(args) + "/api/v1/create_model",
        payload={
            "session_id": session_id,
            "model_seq_id": 0,
            "base_model": model,
            "user_metadata": {},
            "lora_config": {"rank": int(lora_rank)},
        },
        headers=_admin_headers(args, required=True),
        timeout_s=max(args.timeout_s, 60.0),
    )
    if st != 200 or not isinstance(data, dict) or not data.get("request_id"):
        raise RuntimeError(f"create_model failed status={st} body={data!r}")
    return data


def _poll_future(
    args: argparse.Namespace,
    *,
    request_id: str,
    timeout_s: float,
    interval_s: float,
) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last: Any = None
    last_status: int | None = None
    while time.time() < deadline:
        st, data = _http_json(
            "POST",
            _base_url(args) + "/api/v1/retrieve_future",
            payload={"request_id": request_id},
            headers=_admin_headers(args, required=True),
            timeout_s=args.timeout_s,
        )
        last, last_status = data, st
        if st == 408:
            time.sleep(interval_s)
            continue
        if st == 200 and isinstance(data, dict) and not data.get("error"):
            return data
        raise RuntimeError(f"retrieve_future failed status={st} body={data!r}")

    raise TimeoutError(
        f"retrieve_future timeout request_id={request_id} timeout_s={timeout_s} "
        f"last_status={last_status} last={last!r}"
    )


def _ray_init(address: str) -> Any:
    import ray

    ray.init(address=address, ignore_reinit_error=True, logging_level="ERROR")
    return ray


def _collect_nodes(ray: Any) -> tuple[list[dict[str, Any]], dict[str, str]]:
    nodes: list[dict[str, Any]] = []
    id_to_ip: dict[str, str] = {}
    for n in ray.nodes():
        node_id = n.get("NodeID", "")
        ip = n.get("NodeManagerAddress", "")
        id_to_ip[node_id] = ip
        res = n.get("Resources", {}) or {}
        nodes.append(
            {
                "node_id": node_id,
                "node_id_short": node_id[:8],
                "ip": ip,
                "alive": bool(n.get("Alive", False)),
                "gpu_total": int(float(res.get("GPU", 0))),
                "cpu_total": int(float(res.get("CPU", 0))),
            }
        )
    nodes.sort(key=lambda x: (not x["alive"], -x["gpu_total"], x["ip"]))
    return nodes, id_to_ip


def _collect_placement_groups(ray: Any, *, include_removed: bool, id_to_ip: dict[str, str]) -> list[dict[str, Any]]:
    raw = ray.util.placement_group_table()
    pgs = list(raw.values()) if isinstance(raw, dict) else list(raw)
    out: list[dict[str, Any]] = []

    for pg in pgs:
        state = str(pg.get("state", "UNKNOWN"))
        if state == "REMOVED" and not include_removed:
            continue
        bundles = pg.get("bundles") or {}
        bundles_to_node_id = pg.get("bundles_to_node_id") or {}
        node_bundle_counts: Counter[str] = Counter()
        node_gpu_counts: defaultdict[str, float] = defaultdict(float)
        requested_gpu = 0.0
        for bundle_idx, resources in bundles.items():
            resources = resources or {}
            gpu = float(resources.get("GPU", 0) or 0)
            requested_gpu += gpu
            node_id = bundles_to_node_id.get(bundle_idx) or bundles_to_node_id.get(str(bundle_idx)) or ""
            if not node_id:
                continue
            ip = id_to_ip.get(node_id, f"<{node_id[:8]}>")
            node_bundle_counts[ip] += 1
            node_gpu_counts[ip] += gpu
        out.append(
            {
                "pg_id": pg.get("placement_group_id", ""),
                "name": str(pg.get("name", "")),
                "state": state,
                "strategy": str(pg.get("strategy", "")),
                "bundle_count": len(bundles),
                "requested_gpu": requested_gpu,
                "node_distribution": {
                    ip: {"bundles": node_bundle_counts[ip], "gpu": node_gpu_counts[ip]}
                    for ip in sorted(node_bundle_counts.keys())
                },
            }
        )

    state_order = {"PENDING": 0, "CREATED": 1}
    out.sort(key=lambda x: (state_order.get(x["state"], 2), x["name"]))
    return out


def _collect_ray_actor_details(ray: Any, *, id_to_ip: dict[str, str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    try:
        from ray.util.state import list_actors

        gcs_addr = ray.get_runtime_context().gcs_address
        head_ip = gcs_addr.split(":", 1)[0]
        dashboard_url = f"http://{head_ip}:8265"
        for a in list_actors(address=dashboard_url, limit=10000):
            req = a.required_resources or {}
            rows.append(
                {
                    "name": a.name or "",
                    "class_name": a.class_name or "",
                    "state": a.state or "",
                    "namespace": a.ray_namespace or "",
                    "node_id": a.node_id or "",
                    "ip": id_to_ip.get(a.node_id or "", ""),
                    "pid": a.pid,
                    "num_gpus": float(req.get("GPU", 0)),
                    "num_cpus": float(req.get("CPU", 0)),
                }
            )
        rows.sort(key=lambda x: (-x["num_gpus"], x["state"], x["class_name"], x["name"]))
        return {"source": "ray.util.state.list_actors", "actors": rows}
    except Exception as e1:
        try:
            raw = ray.state.actors()  # type: ignore[attr-defined]
            for info in raw.values():
                req = info.get("RequiredResources") or {}
                node_id = info.get("Address", {}).get("NodeID", "")
                rows.append(
                    {
                        "name": info.get("Name", ""),
                        "class_name": info.get("ActorClassName", ""),
                        "state": info.get("State", ""),
                        "namespace": info.get("Namespace", ""),
                        "node_id": node_id,
                        "ip": id_to_ip.get(node_id, ""),
                        "pid": info.get("Pid", 0),
                        "num_gpus": float(req.get("GPU", 0)),
                        "num_cpus": float(req.get("CPU", 0)),
                    }
                )
            rows.sort(key=lambda x: (-x["num_gpus"], x["state"], x["class_name"], x["name"]))
            return {"source": "ray.state.actors", "actors": rows, "_warning": f"fallback due to {e1!r}"}
        except Exception as e2:
            return {"source": "none", "actors": [], "_error": f"state query failed: {e1!r}; {e2!r}"}


def _build_node_probe_remote() -> Any:
    import ray

    @ray.remote(num_cpus=0, max_retries=0)
    def _probe_node() -> dict[str, Any]:
        import os
        import shutil
        import socket as _socket
        import subprocess as _subprocess
        import time
        from pathlib import Path as _Path

        result: dict[str, Any] = {
            "hostname": _socket.gethostname(),
            "timestamp": time.time(),
        }

        try:
            load1, load5, load15 = os.getloadavg()
            result["loadavg"] = {"1m": load1, "5m": load5, "15m": load15}
        except Exception as e:
            result["loadavg_error"] = repr(e)

        try:
            mem_total_kb = 0
            mem_avail_kb = 0
            for line in _Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    mem_total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    mem_avail_kb = int(line.split()[1])
            result["memory"] = {
                "total_gb": round(mem_total_kb / (1024 * 1024), 3),
                "available_gb": round(mem_avail_kb / (1024 * 1024), 3),
                "used_gb": round((mem_total_kb - mem_avail_kb) / (1024 * 1024), 3),
            }
        except Exception as e:
            result["memory_error"] = repr(e)

        disk: dict[str, Any] = {}
        for mount in ["/", "/tmp", "/vePFS-Mindverse"]:
            if not os.path.exists(mount):
                continue
            try:
                u = shutil.disk_usage(mount)
                disk[mount] = {
                    "total_gb": round(u.total / (1024 ** 3), 3),
                    "used_gb": round(u.used / (1024 ** 3), 3),
                    "free_gb": round(u.free / (1024 ** 3), 3),
                    "used_pct": round((u.used / u.total) * 100.0, 2) if u.total else 0.0,
                }
            except Exception as e:
                disk[mount] = {"_error": repr(e)}
        result["disk"] = disk

        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            proc = _subprocess.run(cmd, capture_output=True, text=True, timeout=8)
            if proc.returncode == 0:
                gpus: list[dict[str, Any]] = []
                for line in proc.stdout.splitlines():
                    parts = [x.strip() for x in line.split(",")]
                    if len(parts) != 6:
                        continue
                    gpus.append(
                        {
                            "index": int(parts[0]),
                            "name": parts[1],
                            "memory_total_mb": float(parts[2]),
                            "memory_used_mb": float(parts[3]),
                            "util_gpu_pct": float(parts[4]),
                            "temperature_c": float(parts[5]),
                        }
                    )
                result["gpus"] = gpus
            else:
                result["gpus_error"] = proc.stderr.strip() or proc.stdout.strip() or f"rc={proc.returncode}"
        except Exception as e:
            result["gpus_error"] = repr(e)

        return result

    return _probe_node


def _collect_machine_probes(ray: Any, *, nodes: list[dict[str, Any]], timeout_s: float) -> list[dict[str, Any]]:
    probe = _build_node_probe_remote()
    out: list[dict[str, Any]] = []

    futures: list[tuple[dict[str, Any], Any]] = []
    for node in nodes:
        if not node["alive"]:
            continue
        ip = node["ip"]
        try:
            fut = probe.options(resources={f"node:{ip}": 0.001}).remote()
            futures.append((node, fut))
        except Exception as e:
            out.append({"ip": ip, "node_id": node["node_id"], "_schedule_error": repr(e)})

    for node, fut in futures:
        try:
            payload = ray.get(fut, timeout=timeout_s)
            payload["ip"] = node["ip"]
            payload["node_id"] = node["node_id"]
            out.append(payload)
        except Exception as e:
            out.append({"ip": node["ip"], "node_id": node["node_id"], "_error": repr(e)})

    out.sort(key=lambda x: x.get("ip", ""))
    return out


def _env_bool(proc_env: dict[str, str], key: str, default: bool) -> bool:
    raw = proc_env.get(key)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _maybe_hf_model(model_name_or_path: str) -> str | None:
    s = str(model_name_or_path or "").strip()
    if not s:
        return None
    try:
        from tinker_server.backend.model_registry import maybe_normalize_model_name

        return maybe_normalize_model_name(s)
    except Exception:
        return None


def _build_rebuild_model_options(
    *,
    managed_actors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    try:
        from tinker_server.backend.model_registry import MODEL_CONFIGS, get_model_config
    except Exception:
        return []

    seen: set[str] = set()
    models: list[str] = []

    for m in sorted(MODEL_CONFIGS.keys()):
        if m in seen:
            continue
        seen.add(m)
        models.append(m)

    for a in managed_actors:
        m = _maybe_hf_model(str(a.get("base_model") or "")) or str(a.get("base_model") or "").strip()
        if not m or m in seen:
            continue
        seen.add(m)
        models.append(m)

    items: list[dict[str, Any]] = []
    for m in models:
        default_kind = "training"
        is_moe = None
        num_parameters = None
        try:
            cfg = get_model_config(m)
            is_moe = bool(cfg.is_moe)
            num_parameters = float(cfg.num_parameters)
            default_kind = "training" if is_moe else "vllm"
        except Exception:
            default_kind = "training"
        items.append(
            {
                "model": m,
                "default_kind": default_kind,
                "is_moe": is_moe,
                "num_parameters_b": num_parameters,
            }
        )

    items.sort(
        key=lambda x: (
            -(float(x["num_parameters_b"]) if isinstance(x.get("num_parameters_b"), (float, int)) else -1.0),
            str(x.get("model") or ""),
        )
    )
    return items


def _gateway_routed_models(proc_env: dict[str, str]) -> set[str]:
    raw = str(proc_env.get("TINKER_GATEWAY_CONFIG_JSON", "") or "").strip()
    if not raw:
        return set()
    try:
        payload = json.loads(raw)
    except Exception:
        return set()
    if not isinstance(payload, dict):
        return set()
    model_to_upstream = payload.get("model_to_upstream") or {}
    if not isinstance(model_to_upstream, dict):
        return set()

    out: set[str] = set()
    for model in model_to_upstream.keys():
        s = str(model or "").strip()
        if not s:
            continue
        out.add(_maybe_hf_model(s) or s)
    return out


def _is_training_actor_type(actor_type: str) -> bool:
    return str(actor_type or "").strip() in {"dense", "megatron"}


def _actor_model_ids(actor_payload: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    base_model_raw = str(actor_payload.get("base_model") or "").strip()
    if base_model_raw:
        out.add(base_model_raw)
        base_model_hf = _maybe_hf_model(base_model_raw)
        if base_model_hf:
            out.add(base_model_hf)

    metadata = actor_payload.get("metadata")
    if isinstance(metadata, dict):
        for key in ("model_key", "model_name", "base_model"):
            s = str(metadata.get(key) or "").strip()
            if not s:
                continue
            out.add(s)
            s_hf = _maybe_hf_model(s)
            if s_hf:
                out.add(s_hf)
    return out


def _compute_actor_readiness(
    *,
    proc_env: dict[str, str],
    managed_actors: list[dict[str, Any]],
) -> dict[str, Any]:
    managed_entries: list[dict[str, Any]] = []
    for a in managed_actors:
        actor_name = str(a.get("actor_name") or "").strip()
        actor_type = str(a.get("actor_type") or "").strip()
        model_ids = _actor_model_ids(a)
        base_model_raw = str(a.get("base_model") or "").strip()
        base_model_hf = str(_maybe_hf_model(base_model_raw) or "").strip()
        creating = bool(a.get("creating"))
        managed_entries.append(
            {
                "actor_name": actor_name,
                "actor_type": actor_type,
                "base_model_raw": base_model_raw,
                "base_model_hf": base_model_hf,
                "model_ids": sorted(model_ids),
                "creating": creating,
                "protected": bool(a.get("protected")),
            }
        )

    creating_managed = [m for m in managed_entries if m.get("creating")]

    expected: list[dict[str, Any]] = []
    models_csv = str(proc_env.get("MINT_PERSISTENT_MODELS", "") or "").strip()
    prewarm_training = _env_bool(proc_env, "MINT_PERSISTENT_PREWARM_TRAINING", True)
    prewarm_inference = _env_bool(proc_env, "MINT_PERSISTENT_PREWARM_INFERENCE", True)
    multi_lora_enabled = _env_bool(proc_env, "TINKER_ENABLE_MULTI_LORA", True)
    routed_upstream_models = _gateway_routed_models(proc_env)

    persistent_models: list[str] = []
    skipped_gateway_models: list[str] = []
    if models_csv:
        for raw in [x.strip() for x in models_csv.split(",") if x.strip()]:
            model = _maybe_hf_model(raw) or raw
            if model in routed_upstream_models:
                if model not in skipped_gateway_models:
                    skipped_gateway_models.append(model)
                continue
            if model not in persistent_models:
                persistent_models.append(model)

    try:
        from tinker_server.backend.model_registry import get_model_config
    except Exception:
        get_model_config = None  # type: ignore[assignment]

    skipped_unknown_models: list[str] = []
    for model in persistent_models:
        if get_model_config is not None:
            try:
                get_model_config(model)
            except Exception:
                skipped_unknown_models.append(model)
                continue

        if prewarm_training:
            expected.append({"actor_type": "training", "model": model, "source": "persistent_training"})
        if prewarm_inference and multi_lora_enabled:
            expected.append({"actor_type": "vllm", "model": model, "source": "persistent_inference"})

    def _matches(entry: dict[str, Any], actor_type: str, model: str) -> bool:
        model_ids = set(str(x) for x in (entry.get("model_ids") or []))
        if model not in model_ids:
            return False
        et = str(entry.get("actor_type") or "")
        if actor_type == "training":
            return _is_training_actor_type(et)
        return et == actor_type

    expected_status: list[dict[str, Any]] = []
    for e in expected:
        actor_type = str(e["actor_type"])
        model = str(e["model"])
        matches = [m for m in managed_entries if _matches(m, actor_type, model)]
        ready_count = sum(1 for m in matches if not m.get("creating"))
        creating_count = sum(1 for m in matches if m.get("creating"))
        if ready_count > 0:
            status = "ready"
        elif creating_count > 0:
            status = "creating"
        else:
            status = "missing"
        expected_status.append(
            {
                **e,
                "status": status,
                "ready_count": ready_count,
                "creating_count": creating_count,
                "actor_names": [str(m.get("actor_name") or "") for m in matches if str(m.get("actor_name") or "")],
            }
        )

    not_ready_expected = [x for x in expected_status if str(x.get("status")) != "ready"]
    return {
        "creating_managed": creating_managed,
        "expected_status": expected_status,
        "not_ready_expected": not_ready_expected,
        "counts": {
            "creating_managed": len(creating_managed),
            "expected_total": len(expected_status),
            "expected_not_ready": len(not_ready_expected),
            "gateway_routed_skipped": len(skipped_gateway_models),
            "unknown_model_skipped": len(skipped_unknown_models),
            "inference_disabled_by_multi_lora": int(prewarm_inference and not multi_lora_enabled),
        },
        "skipped_gateway_models": skipped_gateway_models,
        "skipped_unknown_models": skipped_unknown_models,
    }


def _status_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    proc = _pick_primary_server_process(args.program)
    proc_env = proc.get("env", {}) if isinstance(proc.get("env", {}), dict) else {}
    key = _resolve_api_key(args, required=False)

    st_health, health_data = _api_healthz(args)
    st_caps, caps_data = _api_capabilities(args)
    actors_data: dict[str, Any] | str = {"_skipped": "no-auth"}
    if key:
        try:
            actors_data = _api_get_actors(args, actor_type=None, model_name=None)
        except Exception as e:
            actors_data = {"_error": repr(e)}

    ray = _ray_init(args.address)
    nodes, id_to_ip = _collect_nodes(ray)
    pgs = _collect_placement_groups(ray, include_removed=args.include_removed_pg, id_to_ip=id_to_ip)
    actor_details = _collect_ray_actor_details(ray, id_to_ip=id_to_ip)
    try:
        named_actors = ray.util.list_named_actors(all_namespaces=True)
        named_payload: dict[str, Any] = {"count": len(named_actors), "actors": named_actors}
    except Exception as e:
        named_payload = {"count": 0, "actors": [], "_error": repr(e)}

    node_pg_gpu: defaultdict[str, float] = defaultdict(float)
    node_pg_names: defaultdict[str, list[str]] = defaultdict(list)
    pg_name_to_ips: dict[str, list[str]] = {}
    for pg in pgs:
        if pg["state"] == "REMOVED":
            continue
        pg_name = str(pg.get("name", "") or "")
        pg_ips = sorted([str(ip) for ip in (pg.get("node_distribution", {}) or {}).keys() if str(ip)])
        if pg_name:
            pg_name_to_ips[pg_name] = pg_ips
        for ip, info in pg["node_distribution"].items():
            gpu = float(info["gpu"])
            if gpu <= 0:
                continue
            node_pg_gpu[ip] += gpu
            node_pg_names[ip].append(f"{pg['name']}({int(gpu)})")

    managed_actors = actors_data.get("actors", []) if isinstance(actors_data, dict) else []

    actor_name_to_ips: defaultdict[str, set[str]] = defaultdict(set)
    for a in actor_details.get("actors", []):
        if str(a.get("state", "")).upper() != "ALIVE":
            continue
        name = str(a.get("name") or "")
        ip = str(a.get("ip") or "")
        if name and ip:
            actor_name_to_ips[name].add(ip)

    node_model_hints: defaultdict[str, set[str]] = defaultdict(set)
    for m in managed_actors:
        base_model = str(m.get("base_model") or "").strip()
        actor_name = str(m.get("actor_name") or "").strip()
        pg_name = str(m.get("pg_name") or "").strip()
        ips: set[str] = set()
        if pg_name:
            ips.update(pg_name_to_ips.get(pg_name, []))
        if actor_name:
            ips.update(actor_name_to_ips.get(actor_name, set()))
        if not base_model:
            continue
        for ip in ips:
            node_model_hints[ip].add(base_model)

    alive_actor_count_by_ip: Counter[str] = Counter()
    alive_actor_label_counts_by_ip: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for a in actor_details.get("actors", []):
        if str(a.get("state", "")).upper() != "ALIVE":
            continue
        ip = str(a.get("ip", "") or "")
        if ip:
            alive_actor_count_by_ip[ip] += 1
            actor_name = str(a.get("name") or "")
            class_name = str(a.get("class_name") or "")
            label = actor_name or class_name or "<unnamed>"
            if not actor_name and class_name == "MegatronRankWorker":
                hints = sorted(node_model_hints.get(ip, set()))
                if len(hints) == 1:
                    label = f"MegatronRankWorker [{hints[0]}]"
                elif len(hints) > 1:
                    label = f"MegatronRankWorker [models={len(hints)}]"
            gpu_req = float(a.get("num_gpus", 0) or 0)
            if gpu_req > 0:
                label = f"{label} (gpu={int(gpu_req)})"
            alive_actor_label_counts_by_ip[ip][label] += 1

    for n in nodes:
        ip = n["ip"]
        n["gpu_reserved"] = int(node_pg_gpu.get(ip, 0))
        n["gpu_idle_est"] = max(n["gpu_total"] - n["gpu_reserved"], 0)
        n["pg_labels"] = sorted(node_pg_names.get(ip, []))
        n["model_hints"] = sorted(node_model_hints.get(ip, set()))
        n["ray_actor_count"] = int(alive_actor_count_by_ip.get(ip, 0))
        label_counter = alive_actor_label_counts_by_ip.get(ip, Counter())
        n["ray_actor_names"] = [
            (f"{label} x{cnt}" if cnt > 1 else label)
            for label, cnt in sorted(label_counter.items(), key=lambda kv: (-kv[1], kv[0]))
        ]

    probes: list[dict[str, Any]] = []
    if not args.skip_machine_probe:
        probes = _collect_machine_probes(ray, nodes=nodes, timeout_s=args.timeout_s)

    actor_readiness = _compute_actor_readiness(proc_env=proc_env, managed_actors=managed_actors)
    rebuild_model_options = _build_rebuild_model_options(managed_actors=managed_actors)
    pending_pg_count = sum(1 for pg in pgs if pg["state"] == "PENDING")
    managed_counts_by_type = Counter(a.get("actor_type", "unknown") for a in managed_actors)

    return {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "port": args.port,
        "ray_address": args.address,
        "server_process": {
            "pid": proc["pid"],
            "supervisor_process_name": proc["supervisor_process_name"],
            "tinker_port": proc_env.get("TINKER_PORT"),
            "otel_service_name": proc_env.get("OTEL_SERVICE_NAME"),
            "otel_endpoint": proc_env.get("OTEL_EXPORTER_OTLP_ENDPOINT"),
            "has_tinker_api_key": bool(proc_env.get("TINKER_API_KEY")),
            "has_apm_app_key": bool(proc_env.get("MINT_APMPLUS_APP_KEY")),
        },
        "http": {
            "healthz_status": st_health,
            "healthz": health_data,
            "capabilities_status": st_caps,
            "capabilities": caps_data,
            "actors": actors_data,
        },
        "ray": {
            "cluster_resources": ray.cluster_resources() or {},
            "available_resources": ray.available_resources() or {},
            "nodes": nodes,
            "placement_groups": pgs,
            "pending_pg_count": pending_pg_count,
            "named_actors": named_payload,
            "actor_details": actor_details,
        },
        "machine_probes": probes,
        "actor_readiness": actor_readiness,
        "ops_ui": {
            "rebuild_model_options": rebuild_model_options,
        },
        "summary": {
            "gpu_total": int((ray.cluster_resources() or {}).get("GPU", 0)),
            "gpu_available": int((ray.available_resources() or {}).get("GPU", 0)),
            "gpu_nodes_alive": sum(1 for n in nodes if n["alive"] and n["gpu_total"] > 0),
            "cpu_only_nodes_alive": sum(1 for n in nodes if n["alive"] and n["gpu_total"] == 0),
            "dead_nodes": sum(1 for n in nodes if not n["alive"]),
            "placement_groups": len(pgs),
            "pending_placement_groups": pending_pg_count,
            "managed_actors": len(managed_actors),
            "managed_actor_counts_by_type": dict(managed_counts_by_type),
            "creating_actors": int(actor_readiness.get("counts", {}).get("creating_managed", 0)),
            "not_ready_expected_actors": int(actor_readiness.get("counts", {}).get("expected_not_ready", 0)),
        },
    }


def _gpu_probe_summary(probe: dict[str, Any]) -> str:
    gpus = probe.get("gpus", [])
    if not isinstance(gpus, list) or not gpus:
        return str(probe.get("gpus_error", "-"))
    util_avg = sum(float(x.get("util_gpu_pct", 0)) for x in gpus) / len(gpus)
    used = sum(float(x.get("memory_used_mb", 0)) for x in gpus)
    total = sum(float(x.get("memory_total_mb", 0)) for x in gpus)
    return f"{len(gpus)} gpu, util_avg={util_avg:.1f}%, mem={used/1024:.1f}/{total/1024:.1f}GiB"


def _status_markdown(snapshot: dict[str, Any], *, actor_limit: int) -> str:
    summary = snapshot["summary"]
    ray_data = snapshot["ray"]
    nodes = ray_data["nodes"]
    pgs = ray_data["placement_groups"]
    actors_payload = snapshot["http"]["actors"]
    managed_actors = actors_payload.get("actors", []) if isinstance(actors_payload, dict) else []
    probes = snapshot.get("machine_probes", [])

    lines: list[str] = []
    lines.append("# Mint Ops Status")
    lines.append("")
    lines.append(f"- generated_at_utc: `{snapshot['generated_at_utc']}`")
    lines.append(f"- host: `{snapshot['host']}`")
    lines.append(f"- ray_address: `{snapshot['ray_address']}`")
    lines.append(f"- server_pid: `{snapshot['server_process']['pid']}`")
    lines.append("")

    lines.append("## Overview")
    lines.extend(
        _md_table(
            ["metric", "value"],
            [
                ["gpu_total", summary["gpu_total"]],
                ["gpu_available", summary["gpu_available"]],
                ["gpu_nodes_alive", summary["gpu_nodes_alive"]],
                ["cpu_only_nodes_alive", summary["cpu_only_nodes_alive"]],
                ["dead_nodes", summary["dead_nodes"]],
                ["placement_groups", summary["placement_groups"]],
                ["pending_placement_groups", summary["pending_placement_groups"]],
                ["managed_actors", summary["managed_actors"]],
                ["managed_actor_counts_by_type", json.dumps(summary["managed_actor_counts_by_type"], ensure_ascii=False)],
                ["healthz_status", snapshot["http"]["healthz_status"]],
                ["capabilities_status", snapshot["http"]["capabilities_status"]],
            ],
        )
    )
    lines.append("")

    lines.append("## GPU Topology")
    node_rows: list[list[Any]] = []
    for n in nodes:
        actor_names = ", ".join(n.get("ray_actor_names", [])) or "-"
        node_rows.append(
            [
                n["ip"],
                "alive" if n["alive"] else "dead",
                n["gpu_total"],
                n["gpu_reserved"],
                n["gpu_idle_est"],
                n["cpu_total"],
                n.get("ray_actor_count", 0),
                actor_names,
                len(n.get("pg_labels", [])),
            ]
        )
    lines.extend(
        _md_table(
            ["ip", "state", "gpu_total", "gpu_reserved", "gpu_idle_est", "cpu_total", "ray_actors", "ray_actor_names", "pg_count"],
            node_rows,
        )
    )
    lines.append("")

    lines.append("## Placement Groups")
    if pgs:
        pg_rows: list[list[Any]] = []
        for pg in pgs[:actor_limit]:
            dist = ", ".join(f"{ip}:{int(info['gpu'])}" for ip, info in pg["node_distribution"].items()) or "-"
            pg_rows.append(
                [
                    pg["name"],
                    pg["state"],
                    int(pg["requested_gpu"]),
                    pg["bundle_count"],
                    pg["strategy"] or "-",
                    dist,
                ]
            )
        lines.extend(_md_table(["name", "state", "gpu", "bundles", "strategy", "node_distribution(gpu)"], pg_rows))
        if len(pgs) > actor_limit:
            lines.append(f"- truncated: showing {actor_limit}/{len(pgs)} placement groups")
    else:
        lines.append("- no placement groups")
    lines.append("")

    lines.append("## Managed Actors")
    if managed_actors:
        rows: list[list[Any]] = []
        for a in managed_actors[:actor_limit]:
            rows.append(
                [
                    a.get("actor_name", ""),
                    a.get("actor_type", ""),
                    a.get("base_model", ""),
                    a.get("num_gpus", 0),
                    round(float(a.get("idle_time", 0)), 1),
                    round(float(a.get("age", 0)), 1),
                    "yes" if a.get("protected") else "no",
                    "yes" if a.get("creating") else "no",
                ]
            )
        lines.extend(_md_table(["actor_name", "type", "base_model", "gpu", "idle_s", "age_s", "protected", "creating"], rows))
        if len(managed_actors) > actor_limit:
            lines.append(f"- truncated: showing {actor_limit}/{len(managed_actors)} managed actors")
    else:
        lines.append(f"- unavailable: `{json.dumps(actors_payload, ensure_ascii=False)}`")
    lines.append("")

    lines.append("## Machine Status")
    if probes:
        rows = []
        for p in probes:
            mem = p.get("memory", {})
            root_disk = p.get("disk", {}).get("/", {})
            pfs_disk = p.get("disk", {}).get("/vePFS-Mindverse", {})
            rows.append(
                [
                    p.get("ip", ""),
                    p.get("hostname", ""),
                    p.get("loadavg", {}).get("1m", "-"),
                    f"{mem.get('used_gb', '-')}/{mem.get('total_gb', '-')}GiB",
                    root_disk.get("used_pct", "-"),
                    pfs_disk.get("used_pct", "-"),
                    _gpu_probe_summary(p),
                    p.get("_error") or p.get("_schedule_error") or "-",
                ]
            )
        lines.extend(_md_table(["ip", "hostname", "load1", "mem_used/total", "root_used_pct", "pfs_used_pct", "gpu_probe", "error"], rows))
    else:
        lines.append("- machine probe skipped")
    lines.append("")

    return "\n".join(lines) + "\n"


def _status_html(snapshot: dict[str, Any], *, actor_limit: int) -> str:
    # Prefer external template for UI iteration without touching Python logic.
    template_path = Path(__file__).with_name("mint_ops_console.html")
    if template_path.exists():
        payload = json.dumps(snapshot, ensure_ascii=False).replace("</", "<\\/")
        template = template_path.read_text(encoding="utf-8")
        return (
            template.replace("__INITIAL_SNAPSHOT_JSON__", payload).replace(
                "__ACTOR_LIMIT__", str(max(1, int(actor_limit)))
            )
        )

    payload = json.dumps(snapshot, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Mint Ops Console</title>
  <style>
    :root {{
      --bg-0: #f4f7f9;
      --bg-1: #eef5f7;
      --panel: #ffffff;
      --text: #112026;
      --muted: #5f7077;
      --line: #d7e3e8;
      --ok: #1f8f5f;
      --warn: #bd7b1a;
      --bad: #c03a2b;
      --ink: #083344;
      --accent: #0f766e;
      --accent-2: #155e75;
      --shadow: 0 10px 30px rgba(8, 39, 52, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--text);
      font-family: "IBM Plex Sans", "Avenir Next", "Helvetica Neue", sans-serif;
      background:
        radial-gradient(circle at 10% -20%, #d3f8ef 0%, transparent 45%),
        radial-gradient(circle at 90% -30%, #d7f0ff 0%, transparent 42%),
        linear-gradient(180deg, var(--bg-1), var(--bg-0));
      min-height: 100vh;
    }}
    .app {{
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr);
      min-height: 100vh;
    }}
    .sidebar {{
      border-right: 1px solid var(--line);
      background: linear-gradient(180deg, #0f1723 0%, #102430 100%);
      color: #d8ebf5;
      padding: 18px 14px;
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
    }}
    .brand {{
      font-family: "IBM Plex Mono", "Consolas", monospace;
      font-size: 14px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #8ecfe6;
      margin-bottom: 10px;
    }}
    .title {{
      font-size: 22px;
      line-height: 1.15;
      margin: 0 0 14px;
      color: #f2fbff;
      font-family: "Space Grotesk", "IBM Plex Sans", sans-serif;
      font-weight: 700;
    }}
    .tab-btn {{
      width: 100%;
      border: 1px solid transparent;
      background: rgba(255, 255, 255, 0.02);
      color: #d8ebf5;
      border-radius: 10px;
      padding: 11px 12px;
      text-align: left;
      font-size: 14px;
      margin-bottom: 8px;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
      transition: all 160ms ease;
    }}
    .tab-btn:hover {{ background: rgba(255, 255, 255, 0.08); }}
    .tab-btn.active {{
      border-color: rgba(148, 220, 255, 0.35);
      background: linear-gradient(135deg, rgba(20, 115, 142, 0.42), rgba(12, 68, 86, 0.42));
      transform: translateX(2px);
    }}
    .badge {{
      background: rgba(255, 255, 255, 0.16);
      border: 1px solid rgba(255, 255, 255, 0.2);
      padding: 0 7px;
      border-radius: 999px;
      font-size: 11px;
      line-height: 20px;
      min-width: 26px;
      text-align: center;
    }}
    .side-meta {{
      margin-top: 16px;
      padding-top: 12px;
      border-top: 1px dashed rgba(255,255,255,0.18);
      font-size: 12px;
      color: #a8c2ce;
      line-height: 1.5;
    }}
    .main {{
      padding: 20px 20px 28px;
      max-width: 100%;
      overflow-x: hidden;
    }}
    .topbar {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 14px;
    }}
    .btn {{
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--text);
      border-radius: 10px;
      padding: 9px 12px;
      font-weight: 600;
      font-size: 13px;
      cursor: pointer;
      transition: all 140ms ease;
      box-shadow: 0 2px 0 rgba(17, 32, 38, 0.03);
    }}
    .btn:hover {{ transform: translateY(-1px); box-shadow: var(--shadow); }}
    .btn.primary {{ background: linear-gradient(135deg, #0f766e, #155e75); color: #fff; border-color: #0f766e; }}
    .btn.warn {{ background: linear-gradient(135deg, #efb84d, #d2872d); color: #1e1e1e; border-color: #c57d22; }}
    .btn.danger {{ background: linear-gradient(135deg, #e4664b, #cc4427); color: #fff; border-color: #bc321b; }}
    .btn:disabled {{ opacity: 0.45; cursor: not-allowed; transform: none; box-shadow: none; }}
    .hint {{ font-size: 12px; color: var(--muted); }}
    .tab-panel {{
      display: none;
      animation: rise 260ms ease both;
    }}
    .tab-panel.active {{ display: block; }}
    @keyframes rise {{
      from {{ opacity: 0; transform: translateY(12px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 13px;
      padding: 10px 12px;
      box-shadow: var(--shadow);
    }}
    .card .k {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }}
    .card .v {{ margin-top: 4px; font-size: 24px; font-weight: 700; color: var(--ink); font-family: "Space Grotesk", "IBM Plex Sans", sans-serif; }}
    .panel {{
      border: 1px solid var(--line);
      border-radius: 14px;
      background: var(--panel);
      margin-bottom: 12px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .panel h3 {{
      margin: 0;
      padding: 12px 14px;
      font-size: 15px;
      color: var(--accent-2);
      background: linear-gradient(90deg, rgba(15,118,110,0.08), rgba(255,255,255,0));
      border-bottom: 1px solid var(--line);
    }}
    .panel .body {{ padding: 12px 14px 14px; }}
    .table-wrap {{ overflow: auto; max-height: 440px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    th, td {{
      text-align: left;
      padding: 8px 7px;
      border-bottom: 1px solid #eaf0f3;
      vertical-align: top;
      word-break: break-word;
    }}
    th {{
      position: sticky;
      top: 0;
      z-index: 2;
      background: #f8fbfd;
      color: #385563;
      font-weight: 700;
    }}
    .pill {{
      display: inline-block;
      padding: 2px 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      font-size: 11px;
      font-weight: 700;
      background: #fff;
    }}
    .pill.ok {{ color: var(--ok); border-color: rgba(31,143,95,0.25); }}
    .pill.warn {{ color: var(--warn); border-color: rgba(189,123,26,0.25); }}
    .pill.bad {{ color: var(--bad); border-color: rgba(192,58,43,0.28); }}
    .grid-2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    .form-row {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-bottom: 8px;
    }}
    .form-row.single {{ grid-template-columns: 1fr; }}
    label {{
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 5px;
      font-weight: 600;
    }}
    input, select, textarea {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 9px;
      background: #fff;
      padding: 8px 9px;
      font-size: 13px;
      color: var(--text);
      font-family: "IBM Plex Sans", sans-serif;
    }}
    textarea {{ min-height: 100px; resize: vertical; }}
    .checks {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 8px;
    }}
    .check {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      font-size: 12px;
      color: var(--muted);
    }}
    .check input {{ width: 14px; height: 14px; margin: 0; }}
    pre.result {{
      margin: 8px 0 0;
      background: #0f1723;
      color: #d4f0ff;
      border-radius: 10px;
      border: 1px solid #253343;
      font-family: "IBM Plex Mono", "Consolas", monospace;
      font-size: 12px;
      line-height: 1.45;
      padding: 10px;
      max-height: 300px;
      overflow: auto;
      white-space: pre-wrap;
    }}
    .todo {{
      border: 1px dashed #95a8b2;
      border-radius: 10px;
      padding: 10px;
      background: linear-gradient(135deg, rgba(255,248,231,0.68), rgba(255,255,255,0.8));
      color: #6b5c35;
      font-size: 13px;
    }}
    .mono {{ font-family: "IBM Plex Mono", "Consolas", monospace; }}
    .split {{
      display: grid;
      grid-template-columns: minmax(420px, 1fr) minmax(420px, 1fr);
      gap: 12px;
    }}
    @media (max-width: 1080px) {{
      .app {{ grid-template-columns: 1fr; }}
      .sidebar {{
        position: relative;
        height: auto;
        border-right: 0;
        border-bottom: 1px solid rgba(255,255,255,0.12);
      }}
      .tabs {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
      }}
      .tab-btn {{ margin-bottom: 0; }}
      .split, .grid-2 {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="brand">mint.ops</div>
      <h1 class="title">Deployment Console</h1>
      <div class="tabs">
        <button class="tab-btn active" data-tab="dashboard">Dashboard <span class="badge" id="badgeNodes">-</span></button>
        <button class="tab-btn" data-tab="deploy">Deploy <span class="badge" id="badgeActors">-</span></button>
        <button class="tab-btn" data-tab="cronjob">Cronjob <span class="badge">TODO</span></button>
      </div>
      <div class="side-meta">
        <div>Generated: <span class="mono" id="metaGenerated">-</span></div>
        <div>Host: <span class="mono" id="metaHost">-</span></div>
        <div>Ray: <span class="mono" id="metaRay">-</span></div>
        <div>PID: <span class="mono" id="metaPid">-</span></div>
      </div>
    </aside>

    <main class="main">
      <div class="topbar">
        <button class="btn primary" id="refreshBtn">Refresh Snapshot</button>
        <span class="hint" id="refreshHint">ready</span>
        <label class="check"><input type="checkbox" id="autoRefresh" /> auto refresh</label>
        <select id="autoRefreshSec" style="width:auto">
          <option value="5">5s</option>
          <option value="10" selected>10s</option>
          <option value="20">20s</option>
          <option value="30">30s</option>
        </select>
      </div>

      <section class="tab-panel active" data-tab="dashboard">
        <div class="cards">
          <div class="card"><div class="k">GPU Total</div><div class="v" id="cardGpuTotal">-</div></div>
          <div class="card"><div class="k">GPU Available</div><div class="v" id="cardGpuAvail">-</div></div>
          <div class="card"><div class="k">Nodes Alive</div><div class="v" id="cardNodesAlive">-</div></div>
          <div class="card"><div class="k">Pending PG</div><div class="v" id="cardPgPending">-</div></div>
          <div class="card"><div class="k">Managed Actors</div><div class="v" id="cardActors">-</div></div>
          <div class="card"><div class="k">Healthz</div><div class="v" id="cardHealthz">-</div></div>
        </div>

        <div class="panel">
          <h3>Nodes / GPU / Actor Distribution</h3>
          <div class="body table-wrap">
            <table>
              <thead>
                <tr>
                  <th>ip</th><th>state</th><th>gpu_total</th><th>gpu_reserved</th><th>gpu_idle_est</th><th>gpu_used_pct(est)</th><th>actors</th><th>pg_labels</th>
                </tr>
              </thead>
              <tbody id="nodesBody"></tbody>
            </table>
          </div>
        </div>

        <div class="split">
          <div class="panel">
            <h3>Placement Groups Distribution</h3>
            <div class="body table-wrap">
              <table>
                <thead><tr><th>name</th><th>state</th><th>gpu</th><th>bundles</th><th>strategy</th><th>node_distribution</th></tr></thead>
                <tbody id="pgBody"></tbody>
              </table>
            </div>
          </div>
          <div class="panel">
            <h3>Actors Distribution / Status</h3>
            <div class="body table-wrap">
              <table>
                <thead><tr><th>actor</th><th>type</th><th>model</th><th>gpu</th><th>idle_s</th><th>status</th><th>protected</th></tr></thead>
                <tbody id="actorsBody"></tbody>
              </table>
            </div>
          </div>
        </div>
      </section>

      <section class="tab-panel" data-tab="deploy">
        <div class="grid-2">
          <div class="panel">
            <h3>2.1 Node Scale</h3>
            <div class="body">
              <div class="todo">TODO: 节点增减（接 Volcano/Aliyun 集群生命周期接口）。当前版本先提供 Dashboard 可视化和 PG/Actor 运维操作。</div>
            </div>
          </div>
          <div class="panel">
            <h3>2.2 Placement Group</h3>
            <div class="body">
              <div class="form-row single">
                <div>
                  <label>PG names (comma separated, optional)</label>
                  <input id="pgNames" placeholder="pg_a,pg_b" />
                </div>
              </div>
              <div class="form-row">
                <div>
                  <label>state filter</label>
                  <select id="pgState">
                    <option value="">(none)</option>
                    <option value="PENDING">PENDING</option>
                    <option value="CREATED">CREATED</option>
                  </select>
                </div>
                <div>
                  <label>wait / restart flush</label>
                  <input id="restartWait" type="number" min="10" step="1" value="60" />
                </div>
              </div>
              <div class="checks">
                <label class="check"><input id="pgOnlyGpu" type="checkbox" checked /> only GPU PG</label>
                <label class="check"><input id="restartCleanDirty" type="checkbox" /> clean dirty run_server</label>
              </div>
              <div class="topbar" style="margin-bottom:0">
                <button class="btn" id="pgPreviewBtn">Preview Remove</button>
                <button class="btn warn" id="pgApplyBtn">Apply Remove</button>
                <button class="btn" id="serverRestartBtn">Restart Server (flush config)</button>
              </div>
              <pre class="result" id="pgResult">ready</pre>
            </div>
          </div>
        </div>

        <div class="panel">
          <h3>2.3 Actor Operations</h3>
          <div class="body">
            <div class="form-row">
              <div>
                <label>actor type</label>
                <select id="actorType">
                  <option value="vllm">vllm</option>
                  <option value="megatron">megatron</option>
                  <option value="dense">dense</option>
                  <option value="all">all</option>
                </select>
              </div>
              <div>
                <label>model name (optional)</label>
                <input id="actorModel" placeholder="Qwen/Qwen3-30B-A3B-Instruct-2507" />
              </div>
            </div>
            <div class="topbar" style="margin-bottom:0">
              <button class="btn danger" id="actorKillBtn">Kill Actor(s)</button>
            </div>
            <pre class="result" id="actorResult">ready</pre>
            <div class="table-wrap" style="margin-top:10px">
              <table>
                <thead><tr><th>actor</th><th>type</th><th>model</th><th>gpu</th><th>creating</th><th>action</th></tr></thead>
                <tbody id="deployActorsBody"></tbody>
              </table>
            </div>
          </div>
        </div>
      </section>

      <section class="tab-panel" data-tab="cronjob">
        <div class="panel">
          <h3>Cronjob</h3>
          <div class="body">
            <div class="todo">TODO: 定时任务编排（例如健康巡检、Actor 回收、状态归档）。下一步可以把 `verify` 和 `status` 组合成可配置 cron 策略。</div>
          </div>
        </div>
      </section>
    </main>
  </div>

  <script id="initialSnapshot" type="application/json">{payload}</script>
  <script>
    const ACTOR_LIMIT = {max(1, int(actor_limit))};
    let state = {{ snapshot: JSON.parse(document.getElementById('initialSnapshot').textContent || '{{}}') }};
    let autoTimer = null;

    function esc(v) {{
      return String(v ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }}
    function n(v, d=0) {{
      const x = Number(v);
      return Number.isFinite(x) ? x : d;
    }}
    function asList(v) {{
      return Array.isArray(v) ? v : [];
    }}
    function statusPill(text, kind) {{
      const cls = kind || 'warn';
      return `<span class="pill ${{cls}}">${{esc(text)}}</span>`;
    }}
    function pretty(obj) {{
      try {{ return JSON.stringify(obj, null, 2); }} catch (_) {{ return String(obj); }}
    }}
    async function apiGetJson(path) {{
      const resp = await fetch(path);
      const body = await resp.json().catch(() => ({{ ok: false, error: 'invalid json' }}));
      if (!resp.ok) {{
        throw new Error(body.error || ('HTTP ' + resp.status));
      }}
      return body;
    }}
    async function apiPost(path, payload) {{
      const resp = await fetch(path, {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify(payload || {{}}),
      }});
      const body = await resp.json().catch(() => ({{ ok: false, error: 'invalid json' }}));
      if (!resp.ok) {{
        throw new Error(body.error || ('HTTP ' + resp.status));
      }}
      return body;
    }}
    function setHint(msg) {{
      document.getElementById('refreshHint').textContent = msg;
    }}
    function attachTabs() {{
      const buttons = Array.from(document.querySelectorAll('.tab-btn'));
      const panels = Array.from(document.querySelectorAll('.tab-panel'));
      for (const btn of buttons) {{
        btn.addEventListener('click', () => {{
          const tab = btn.getAttribute('data-tab');
          buttons.forEach(x => x.classList.toggle('active', x === btn));
          panels.forEach(p => p.classList.toggle('active', p.getAttribute('data-tab') === tab));
        }});
      }}
    }}
    function renderMeta(snap) {{
      document.getElementById('metaGenerated').textContent = String(snap.generated_at_utc || '-');
      document.getElementById('metaHost').textContent = String(snap.host || '-');
      document.getElementById('metaRay').textContent = String(snap.ray_address || '-');
      document.getElementById('metaPid').textContent = String((snap.server_process || {{}}).pid || '-');
    }}
    function renderCards(snap) {{
      const summary = snap.summary || {{}};
      document.getElementById('cardGpuTotal').textContent = String(summary.gpu_total ?? '-');
      document.getElementById('cardGpuAvail').textContent = String(summary.gpu_available ?? '-');
      document.getElementById('cardNodesAlive').textContent = String(summary.gpu_nodes_alive ?? '-');
      document.getElementById('cardPgPending').textContent = String(summary.pending_placement_groups ?? '-');
      document.getElementById('cardActors').textContent = String(summary.managed_actors ?? '-');
      document.getElementById('cardHealthz').textContent = String((snap.http || {{}}).healthz_status ?? '-');
      document.getElementById('badgeNodes').textContent = String(asList((snap.ray || {{}}).nodes).length);
      document.getElementById('badgeActors').textContent = String(asList(((snap.http || {{}}).actors || {{}}).actors).length);
    }}
    function renderNodesTable(snap) {{
      const nodes = asList((snap.ray || {{}}).nodes);
      const rows = nodes.map((node) => {{
        const total = n(node.gpu_total);
        const reserved = n(node.gpu_reserved);
        const usedPct = total > 0 ? (100 * reserved / total) : 0;
        const pgLabels = asList(node.pg_labels).slice(0, 5).join(', ') || '-';
        const alive = !!node.alive;
        return `<tr>
          <td class="mono">${{esc(node.ip || '-')}}</td>
          <td>${{statusPill(alive ? 'alive' : 'dead', alive ? 'ok' : 'bad')}}</td>
          <td>${{total}}</td>
          <td>${{reserved}}</td>
          <td>${{n(node.gpu_idle_est)}}</td>
          <td>${{usedPct.toFixed(1)}}%</td>
          <td>${{n(node.ray_actor_count)}}</td>
          <td>${{esc(pgLabels)}}</td>
        </tr>`;
      }});
      document.getElementById('nodesBody').innerHTML = rows.join('') || "<tr><td colspan='8'>no nodes</td></tr>";
    }}
    function renderPgTable(snap) {{
      const pgs = asList((snap.ray || {{}}).placement_groups).slice(0, ACTOR_LIMIT);
      const rows = pgs.map((pg) => {{
        const distObj = pg.node_distribution || {{}};
        const dist = Object.keys(distObj).map((ip) => `${{ip}}:${{Number((distObj[ip] || {{}}).gpu || 0)}}`).join(', ') || '-';
        const state = String(pg.state || 'UNKNOWN');
        const cls = state === 'CREATED' ? 'ok' : (state === 'PENDING' ? 'warn' : 'bad');
        return `<tr>
          <td class="mono">${{esc(pg.name || '-')}}</td>
          <td>${{statusPill(state, cls)}}</td>
          <td>${{n(pg.requested_gpu)}}</td>
          <td>${{n(pg.bundle_count)}}</td>
          <td>${{esc(pg.strategy || '-')}}</td>
          <td>${{esc(dist)}}</td>
        </tr>`;
      }});
      document.getElementById('pgBody').innerHTML = rows.join('') || "<tr><td colspan='6'>no placement groups</td></tr>";
    }}
    function renderActorsTable(snap) {{
      const actorsPayload = (snap.http || {{}}).actors || {{}};
      const actors = asList(actorsPayload.actors).slice(0, ACTOR_LIMIT);
      const rows = actors.map((a) => {{
        const creating = !!a.creating;
        const cls = creating ? 'warn' : 'ok';
        return `<tr>
          <td class="mono">${{esc(a.actor_name || '-')}}</td>
          <td>${{esc(a.actor_type || '-')}}</td>
          <td>${{esc(a.base_model || '-')}}</td>
          <td>${{n(a.num_gpus)}}</td>
          <td>${{n(a.idle_time).toFixed(1)}}</td>
          <td>${{statusPill(creating ? 'creating/pending' : 'ready', cls)}}</td>
          <td>${{a.protected ? 'yes' : 'no'}}</td>
        </tr>`;
      }});
      document.getElementById('actorsBody').innerHTML = rows.join('') || "<tr><td colspan='7'>no managed actors</td></tr>";

      const deployRows = actors.map((a) => `<tr>
        <td class="mono">${{esc(a.actor_name || '-')}}</td>
        <td>${{esc(a.actor_type || '-')}}</td>
        <td>${{esc(a.base_model || '-')}}</td>
        <td>${{n(a.num_gpus)}}</td>
        <td>${{a.creating ? 'yes' : 'no'}}</td>
        <td><button class="btn danger quick-kill" data-type="${{esc(a.actor_type || '')}}" data-model="${{esc(a.base_model || '')}}">kill</button></td>
      </tr>`);
      document.getElementById('deployActorsBody').innerHTML = deployRows.join('') || "<tr><td colspan='6'>no managed actors</td></tr>";
    }}
    function renderAll() {{
      const snap = state.snapshot || {{}};
      renderMeta(snap);
      renderCards(snap);
      renderNodesTable(snap);
      renderPgTable(snap);
      renderActorsTable(snap);
    }}
    async function refreshSnapshot(force) {{
      setHint(force ? 'refreshing...' : 'loading...');
      const btn = document.getElementById('refreshBtn');
      btn.disabled = true;
      try {{
        const data = await apiGetJson('/api/v1/status?format=json' + (force ? '&refresh=1' : ''));
        state.snapshot = data;
        renderAll();
        setHint('updated: ' + (data.generated_at_utc || '-'));
      }} catch (err) {{
        setHint('refresh failed: ' + (err && err.message ? err.message : String(err)));
      }} finally {{
        btn.disabled = false;
      }}
    }}
    function parseCsv(input) {{
      return String(input || '')
        .split(',')
        .map(s => s.trim())
        .filter(Boolean);
    }}
    function setResult(id, data) {{
      document.getElementById(id).textContent = pretty(data);
    }}
    async function runPgRemove(apply) {{
      const names = parseCsv(document.getElementById('pgNames').value);
      const state = String(document.getElementById('pgState').value || '');
      const onlyGpu = !!document.getElementById('pgOnlyGpu').checked;
      const payload = {{ names, state, only_gpu: onlyGpu, apply: !!apply }};
      setResult('pgResult', {{ status: 'running', payload }});
      try {{
        const data = await apiPost('/api/v1/deploy/pg/remove', payload);
        setResult('pgResult', data);
        await refreshSnapshot(true);
      }} catch (err) {{
        setResult('pgResult', {{ ok: false, error: String(err && err.message ? err.message : err) }});
      }}
    }}
    async function runActorKill(actorType, modelName) {{
      const payload = {{ actor_type: actorType, model_name: modelName || '' }};
      setResult('actorResult', {{ status: 'running', payload }});
      try {{
        const data = await apiPost('/api/v1/deploy/actor/kill', payload);
        setResult('actorResult', data);
        await refreshSnapshot(true);
      }} catch (err) {{
        setResult('actorResult', {{ ok: false, error: String(err && err.message ? err.message : err) }});
      }}
    }}
    async function runServerRestart() {{
      const cleanDirty = !!document.getElementById('restartCleanDirty').checked;
      const waitS = Number(document.getElementById('restartWait').value || '60');
      setResult('pgResult', {{ status: 'running', action: 'server restart', clean_dirty: cleanDirty, wait_healthz_s: waitS }});
      try {{
        const data = await apiPost('/api/v1/deploy/server/restart', {{
          clean_dirty: cleanDirty,
          wait_healthz_s: waitS,
        }});
        setResult('pgResult', data);
        await refreshSnapshot(true);
      }} catch (err) {{
        setResult('pgResult', {{ ok: false, error: String(err && err.message ? err.message : err) }});
      }}
    }}
    function bindActions() {{
      document.getElementById('refreshBtn').addEventListener('click', () => refreshSnapshot(true));
      document.getElementById('pgPreviewBtn').addEventListener('click', () => runPgRemove(false));
      document.getElementById('pgApplyBtn').addEventListener('click', async () => {{
        if (!window.confirm('Apply placement group removal now?')) return;
        await runPgRemove(true);
      }});
      document.getElementById('serverRestartBtn').addEventListener('click', async () => {{
        if (!window.confirm('Restart tinker-server now?')) return;
        await runServerRestart();
      }});
      document.getElementById('actorKillBtn').addEventListener('click', async () => {{
        const actorType = String(document.getElementById('actorType').value || '').trim();
        const model = String(document.getElementById('actorModel').value || '').trim();
        if (!window.confirm(`Kill actor_type=${{actorType}}${{model ? (' model=' + model) : ''}} ?`)) return;
        await runActorKill(actorType, model);
      }});
      document.getElementById('deployActorsBody').addEventListener('click', async (ev) => {{
        const target = ev.target;
        if (!target || !(target instanceof HTMLElement)) return;
        if (!target.classList.contains('quick-kill')) return;
        const actorType = String(target.getAttribute('data-type') || '').trim();
        const model = String(target.getAttribute('data-model') || '').trim();
        if (!actorType) return;
        if (!window.confirm(`Kill actor_type=${{actorType}} model=${{model || '-'}} ?`)) return;
        await runActorKill(actorType, model);
      }});

      function resetAutoTimer() {{
        if (autoTimer) {{
          clearInterval(autoTimer);
          autoTimer = null;
        }}
        const enabled = !!document.getElementById('autoRefresh').checked;
        const sec = Number(document.getElementById('autoRefreshSec').value || '10');
        if (!enabled || !Number.isFinite(sec) || sec <= 0) {{
          return;
        }}
        autoTimer = setInterval(() => {{
          refreshSnapshot(false).catch(() => null);
        }}, sec * 1000);
      }}
      document.getElementById('autoRefresh').addEventListener('change', resetAutoTimer);
      document.getElementById('autoRefreshSec').addEventListener('change', resetAutoTimer);
    }}

    attachTabs();
    bindActions();
    renderAll();
  </script>
</body>
</html>
"""


def _write_status_outputs(
    *,
    args: argparse.Namespace,
    snapshot: dict[str, Any],
    print_messages: bool,
) -> tuple[str, str]:
    md = _status_markdown(snapshot, actor_limit=max(1, args.actor_limit))
    html_report = _status_html(snapshot, actor_limit=max(1, args.actor_limit))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        if print_messages:
            print(f"wrote json: {args.json_out}")
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(md, encoding="utf-8")
        if print_messages:
            print(f"wrote markdown: {args.md_out}")
    if args.html_out:
        args.html_out.parent.mkdir(parents=True, exist_ok=True)
        args.html_out.write_text(html_report, encoding="utf-8")
        if print_messages:
            print(f"wrote html: {args.html_out}")
    return md, html_report


def _cmd_status(args: argparse.Namespace) -> int:
    snap = _status_snapshot(args)
    md, _html_report = _write_status_outputs(args=args, snapshot=snap, print_messages=True)
    if not args.md_out and not args.html_out and not args.serve:
        print(md)
    if args.json:
        print(json.dumps(snap, ensure_ascii=False, indent=2))

    if args.serve:
        if args.html_out is None:
            return _serve_status_api(
                bind=str(args.serve_bind),
                port=int(args.serve_port),
                actor_limit=max(1, args.actor_limit),
                cache_ttl_s=float(args.cache_ttl_s),
                snapshot_fn=lambda: _status_snapshot(args),
                ops_args=args,
                kill_stale_ops=bool(args.kill_stale_ops),
            )

        def _refresh_local() -> int:
            try:
                fresh = _status_snapshot(args)
                _write_status_outputs(args=args, snapshot=fresh, print_messages=False)
                return 0
            except Exception:
                return 1

        return _serve_status_html(
            html_path=Path(args.html_out),
            bind=str(args.serve_bind),
            port=int(args.serve_port),
            refresh_fn=_refresh_local,
            kill_stale_ops=bool(args.kill_stale_ops),
        )
    return 0


def _cmd_ops_server(args: argparse.Namespace) -> int:
    return _serve_status_api(
        bind=str(args.bind),
        port=int(args.server_port),
        actor_limit=max(1, args.actor_limit),
        cache_ttl_s=float(args.cache_ttl_s),
        snapshot_fn=lambda: _status_snapshot(args),
        ops_args=args,
        kill_stale_ops=bool(args.kill_stale_ops),
    )


def _server_restart_operation(
    args: argparse.Namespace,
    *,
    clean_dirty: bool,
    wait_healthz_s: float,
) -> dict[str, Any]:
    before = _find_run_server_processes(args.program)

    cleaned_dirty_pids: list[int] = []
    if clean_dirty:
        dirty = [p for p in before if not p["is_supervisor_managed"]]
        cleaned_dirty_pids = [int(p["pid"]) for p in dirty]
        for p in dirty:
            try:
                os.kill(int(p["pid"]), 15)
            except ProcessLookupError:
                pass
        if dirty:
            time.sleep(1.5)
            for p in dirty:
                try:
                    os.kill(int(p["pid"]), 0)
                    os.kill(int(p["pid"]), 9)
                except ProcessLookupError:
                    pass

    cp = _run(["supervisorctl", "restart", args.program], timeout_s=60)
    if cp.returncode != 0:
        raise RuntimeError(
            f"supervisor restart failed rc={cp.returncode} stdout={cp.stdout.strip()!r} stderr={cp.stderr.strip()!r}"
        )

    deadline = time.time() + float(wait_healthz_s)
    last_status: int | None = None
    last_body: Any = None
    while time.time() < deadline:
        st, data = _api_healthz(args)
        last_status = st
        last_body = data
        if st == 200:
            break
        time.sleep(1.5)

    after = _find_run_server_processes(args.program)
    if not after:
        raise RuntimeError("No run_server.py found after restart")

    primary = _pick_primary_server_process(args.program)
    healthz_ready = last_status == 200
    if not healthz_ready:
        raise RuntimeError(
            f"healthz not ready after restart: status={last_status} body={last_body!r}"
        )

    return {
        "ok": True,
        "cleaned_dirty_pids": cleaned_dirty_pids,
        "before_run_server_pids": [int(p["pid"]) for p in before],
        "after_run_server_pids": [int(p["pid"]) for p in after],
        "supervisor_stdout": cp.stdout.strip(),
        "healthz_status": last_status,
        "healthz_body": last_body,
        "primary_server": {
            "pid": int(primary["pid"]),
            "is_supervisor_managed": bool(primary["is_supervisor_managed"]),
            "supervisor_process_name": primary["supervisor_process_name"],
            "tinker_port": primary["env"].get("TINKER_PORT"),
            "otel_endpoint": primary["env"].get("OTEL_EXPORTER_OTLP_ENDPOINT"),
            "has_apm_app_key": bool(primary["env"].get("MINT_APMPLUS_APP_KEY")),
        },
    }


def _actor_kill_operation(
    args: argparse.Namespace,
    *,
    actor_type: str,
    model_name: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"actor_type": actor_type}
    if model_name:
        payload["model_name"] = model_name
    st, data = _http_json(
        "POST",
        _base_url(args) + "/api/v1/actors/kill",
        payload=payload,
        headers=_admin_headers(args, required=True),
        timeout_s=max(args.timeout_s, 60.0),
    )
    if st != 200 or not isinstance(data, dict):
        raise RuntimeError(f"POST /actors/kill failed status={st} body={data!r}")
    return data


def _actor_rebuild_operation(
    args: argparse.Namespace,
    *,
    kind: str,
    models: list[str],
    sample_ping: bool,
    lora_rank: int,
    poll_timeout_s: float,
    poll_interval_s: float,
) -> dict[str, Any]:
    if kind not in {"vllm", "training"}:
        raise RuntimeError("kind must be one of: vllm, training")
    if not models:
        raise RuntimeError("at least one model is required")

    results: list[dict[str, Any]] = []
    for model in models:
        item: dict[str, Any] = {"model": model, "kind": kind, "status": "UNKNOWN"}
        t0 = time.time()
        try:
            session_id = _api_create_session(args, tag="scripts/ops/mint_ops.py:actor-rebuild")
            item["session_id"] = session_id
            if kind == "vllm":
                sampling_session_id = _api_create_sampling_session(args, session_id=session_id, model=model)
                item["sampling_session_id"] = sampling_session_id
                if sample_ping:
                    _ = _api_asample_ping(
                        args,
                        sampling_session_id=sampling_session_id,
                        poll_timeout_s=poll_timeout_s,
                        poll_interval_s=poll_interval_s,
                    )
            else:
                created = _api_create_model(args, session_id=session_id, model=model, lora_rank=int(lora_rank))
                item["create_model"] = created
                request_id = str(created["request_id"])
                item["future"] = _poll_future(
                    args,
                    request_id=request_id,
                    timeout_s=float(poll_timeout_s),
                    interval_s=float(poll_interval_s),
                )
            item["status"] = "PASS"
        except Exception as e:
            item["status"] = "FAIL"
            item["error"] = f"{type(e).__name__}: {e}"
        finally:
            item["elapsed_s"] = round(time.time() - t0, 2)
        results.append(item)

    ok = all(x.get("status") == "PASS" for x in results)
    return {"ok": bool(ok), "results": results}


def _pg_remove_operation(
    args: argparse.Namespace,
    *,
    names: list[str] | None,
    state: str | None,
    only_gpu: bool,
    apply: bool,
) -> dict[str, Any]:
    ray = _ray_init(args.address)
    _nodes, id_to_ip = _collect_nodes(ray)
    pgs = _collect_placement_groups(ray, include_removed=True, id_to_ip=id_to_ip)

    name_set = set(names or [])
    targets: list[dict[str, Any]] = []
    for pg in pgs:
        if pg["state"] == "REMOVED":
            continue
        if name_set and pg["name"] in name_set:
            targets.append(pg)
            continue
        if state and pg["state"] == state:
            targets.append(pg)

    if only_gpu:
        targets = [pg for pg in targets if float(pg["requested_gpu"]) > 0]

    targets.sort(key=lambda x: x["name"])
    result: dict[str, Any] = {"targets": targets, "apply": bool(apply), "removed": [], "failed": []}
    if not targets or not apply:
        return result

    from ray.util.placement_group import PlacementGroup

    removed: list[str] = []
    failed: list[dict[str, str]] = []
    for pg in targets:
        try:
            pg_id_bytes = bytes.fromhex(str(pg["pg_id"]))
            handle = PlacementGroup(ray.PlacementGroupID(pg_id_bytes))
            ray.util.remove_placement_group(handle)
            removed.append(pg["name"])
        except Exception as e:
            failed.append({"name": pg["name"], "error": f"{type(e).__name__}: {e}"})
    result["removed"] = removed
    result["failed"] = failed
    return result


def _cmd_server_restart(args: argparse.Namespace) -> int:
    result = _server_restart_operation(
        args,
        clean_dirty=bool(args.clean_dirty),
        wait_healthz_s=float(args.wait_healthz_s),
    )
    print("before_run_server_pids:", result["before_run_server_pids"])
    print("cleaning_dirty_pids:", result["cleaned_dirty_pids"] or "none")
    if result.get("supervisor_stdout"):
        print(str(result["supervisor_stdout"]))
    print("after_run_server_pids:", result["after_run_server_pids"])
    print("primary_server:", json.dumps(result["primary_server"], ensure_ascii=False))
    print("healthz: ready")
    return 0


def _cmd_actor_list(args: argparse.Namespace) -> int:
    data = _api_get_actors(args, actor_type=args.actor_type, model_name=args.model_name)
    actors = data.get("actors", [])
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0

    print(f"total_gpus_used={data.get('total_gpus_used', '?')} actors={len(actors)}")
    rows = []
    for a in actors:
        rows.append(
            [
                a.get("actor_name", ""),
                a.get("actor_type", ""),
                a.get("base_model", ""),
                a.get("num_gpus", 0),
                round(float(a.get("idle_time", 0)), 1),
                "yes" if a.get("protected") else "no",
                a.get("current_session") or "-",
            ]
        )
    for line in _md_table(["actor_name", "type", "base_model", "gpu", "idle_s", "protected", "session"], rows):
        print(line)
    return 0


def _cmd_actor_kill(args: argparse.Namespace) -> int:
    data = _actor_kill_operation(
        args,
        actor_type=str(args.actor_type),
        model_name=str(args.model_name) if args.model_name else None,
    )
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


def _cmd_actor_rebuild(args: argparse.Namespace) -> int:
    models = list(args.model or [])
    data = _actor_rebuild_operation(
        args,
        kind=str(args.kind),
        models=models,
        sample_ping=bool(args.sample_ping),
        lora_rank=int(args.lora_rank),
        poll_timeout_s=float(args.poll_timeout_s),
        poll_interval_s=float(args.poll_interval_s),
    )
    print(json.dumps({"results": data.get("results", [])}, ensure_ascii=False, indent=2))
    return 0 if bool(data.get("ok")) else 1


def _cmd_pg_list(args: argparse.Namespace) -> int:
    ray = _ray_init(args.address)
    nodes, id_to_ip = _collect_nodes(ray)
    pgs = _collect_placement_groups(ray, include_removed=args.include_removed_pg, id_to_ip=id_to_ip)
    if args.json:
        print(json.dumps({"nodes": nodes, "placement_groups": pgs}, ensure_ascii=False, indent=2))
        return 0

    rows = []
    for pg in pgs:
        if args.state and pg["state"] != args.state:
            continue
        dist = ", ".join(f"{ip}:{int(info['gpu'])}" for ip, info in pg["node_distribution"].items()) or "-"
        rows.append([pg["name"], pg["state"], int(pg["requested_gpu"]), pg["bundle_count"], pg["strategy"] or "-", dist])
    for line in _md_table(["name", "state", "gpu", "bundles", "strategy", "node_distribution(gpu)"], rows):
        print(line)
    print(f"placement_groups={len(rows)}")
    return 0


def _cmd_pg_remove(args: argparse.Namespace) -> int:
    result = _pg_remove_operation(
        args,
        names=list(args.name or []),
        state=str(args.state) if args.state else None,
        only_gpu=bool(args.only_gpu),
        apply=bool(args.apply),
    )
    print(json.dumps({"targets": result.get("targets", []), "apply": result.get("apply", False)}, ensure_ascii=False, indent=2))
    targets = result.get("targets", [])
    if not targets:
        return 0
    if not bool(result.get("apply")):
        print("dry-run only. pass --apply to remove targets.")
        return 0
    print(json.dumps({"removed": result.get("removed", []), "failed": result.get("failed", [])}, ensure_ascii=False, indent=2))
    return 0 if not result.get("failed") else 1


def _cmd_verify(args: argparse.Namespace) -> int:
    report: dict[str, Any] = {"checks": []}

    st, data = _api_healthz(args)
    ok = st == 200
    report["checks"].append({"name": "healthz", "ok": ok, "status": st, "body": data})

    st_caps, caps = _api_capabilities(args)
    caps_ok = st_caps == 200 and isinstance(caps, dict) and isinstance(caps.get("supported_models"), list)
    report["checks"].append(
        {
            "name": "get_server_capabilities",
            "ok": caps_ok,
            "status": st_caps,
            "supported_models_count": len(caps.get("supported_models", [])) if isinstance(caps, dict) else 0,
            "body": caps if not caps_ok else {"supported_models_count": len(caps.get("supported_models", []))},
        }
    )

    try:
        actors = _api_get_actors(args, actor_type=None, model_name=None)
        report["checks"].append(
            {
                "name": "actors",
                "ok": True,
                "actors_count": len(actors.get("actors", [])),
                "total_gpus_used": actors.get("total_gpus_used"),
            }
        )
    except Exception as e:
        report["checks"].append({"name": "actors", "ok": False, "error": f"{type(e).__name__}: {e}"})

    for model in args.sampling_model or []:
        item: dict[str, Any] = {"name": f"sampling_smoke:{model}", "ok": False}
        try:
            session_id = _api_create_session(args, tag="scripts/ops/mint_ops.py:verify")
            sampling_session_id = _api_create_sampling_session(args, session_id=session_id, model=model)
            _ = _api_asample_ping(
                args,
                sampling_session_id=sampling_session_id,
                poll_timeout_s=args.poll_timeout_s,
                poll_interval_s=args.poll_interval_s,
            )
            item["ok"] = True
            item["session_id"] = session_id
            item["sampling_session_id"] = sampling_session_id
        except Exception as e:
            item["error"] = f"{type(e).__name__}: {e}"
        report["checks"].append(item)

    all_ok = all(bool(x.get("ok")) for x in report["checks"])
    report["ok"] = all_ok
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all_ok else 1


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mint unified ops CLI")
    p.add_argument("--host", default=None, help="SSH host to run command remotely")
    p.add_argument("--remote-python", default=DEFAULT_REMOTE_PYTHON, help="Remote python path for --host mode")
    p.add_argument("--port", type=int, default=18000, help="tinker-server API port")
    p.add_argument("--address", default="auto", help="Ray address")
    p.add_argument("--timeout-s", type=float, default=20.0, help="HTTP/RPC timeout")
    p.add_argument("--api-key", default=None, help="API key override")
    p.add_argument("--no-auth", action="store_true", help="Disable API-key auth")
    p.add_argument("--program", default=DEFAULT_SUPERVISOR_PROGRAM, help="supervisor program name")
    p.add_argument("--json", action="store_true", help="Print JSON payload in addition to command output")

    sub = p.add_subparsers(dest="subcommand", required=True)

    sp = sub.add_parser("status", help="Collect integrated cluster status")
    sp.add_argument("--skip-machine-probe", action="store_true", help="Skip per-node machine probe")
    sp.add_argument("--include-removed-pg", action="store_true", help="Include REMOVED placement groups")
    sp.add_argument("--actor-limit", type=int, default=200, help="Max rows in markdown tables")
    sp.add_argument("--json-out", type=Path, default=None, help="Write JSON snapshot to file")
    sp.add_argument("--md-out", type=Path, default=None, help="Write markdown report to file")
    sp.add_argument("--html-out", type=Path, default=None, help="Write HTML report to file")
    sp.add_argument("-s", "--serve", action="store_true", help="Serve local HTML report with refresh endpoint")
    sp.add_argument("--serve-bind", default="127.0.0.1", help="Bind address for --serve (default: 127.0.0.1)")
    sp.add_argument("--serve-port", type=int, default=8765, help="Port for --serve (default: 8765)")
    sp.add_argument(
        "--cache-ttl-s",
        type=float,
        default=5.0,
        help="In-memory status cache TTL in seconds when serving without --html-out",
    )
    sp.add_argument(
        "--direct",
        action="store_true",
        help="With --host and --serve, tunnel to remote server directly (no remote temp file + scp)",
    )
    sp.add_argument(
        "--direct-local-port",
        type=int,
        default=None,
        help="Local tunnel port for --direct (default: same as --serve-port)",
    )
    sp.add_argument(
        "--kill-stale-ops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Kill stale mint_ops server processes on the same serve port before binding",
    )
    sp.set_defaults(func=_cmd_status)

    sp = sub.add_parser("ops-server", help="Run Mint ops HTTP server for status HTML/JSON/MD")
    sp.add_argument("--bind", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    sp.add_argument("--server-port", type=int, default=8765, help="Ops server port (default: 8765)")
    sp.add_argument("--skip-machine-probe", action="store_true", help="Skip per-node machine probe")
    sp.add_argument("--include-removed-pg", action="store_true", help="Include REMOVED placement groups")
    sp.add_argument("--actor-limit", type=int, default=200, help="Max rows in status markdown/html tables")
    sp.add_argument("--cache-ttl-s", type=float, default=5.0, help="Status cache TTL seconds")
    sp.add_argument(
        "--kill-stale-ops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Kill stale mint_ops server processes on the same serve port before binding",
    )
    sp.set_defaults(func=_cmd_ops_server)

    sp = sub.add_parser("server-restart", help="Restart tinker server via supervisord")
    sp.add_argument("--clean-dirty", action="store_true", help="Kill non-supervisor run_server.py before restart")
    sp.add_argument("--wait-healthz-s", type=float, default=60.0, help="Wait budget for healthz ready")
    sp.set_defaults(func=_cmd_server_restart)

    sp = sub.add_parser("actor-list", help="List managed actors from /api/v1/actors")
    sp.add_argument("--actor-type", choices=["vllm", "megatron", "dense"], default=None, help="Filter actor type")
    sp.add_argument("--model-name", default=None, help="Filter base model")
    sp.set_defaults(func=_cmd_actor_list)

    sp = sub.add_parser("actor-kill", help="Kill managed actors via /api/v1/actors/kill")
    sp.add_argument("--actor-type", required=True, choices=["vllm", "megatron", "dense", "all"], help="Actor type")
    sp.add_argument("--model-name", default=None, help="Optional model_name filter")
    sp.set_defaults(func=_cmd_actor_kill)

    sp = sub.add_parser("actor-rebuild", help="Rebuild actor by forcing API creation path")
    sp.add_argument("--kind", choices=["vllm", "training"], default="vllm", help="Creation path kind")
    sp.add_argument("--model", action="append", default=[], help="Base model name (repeatable)")
    sp.add_argument("--models", default=None, help="Comma-separated models")
    sp.add_argument("--sample-ping", action="store_true", help="For vllm: run asample ping after session creation")
    sp.add_argument("--lora-rank", type=int, default=16, help="For training: LoRA rank for create_model")
    sp.add_argument("--poll-timeout-s", type=float, default=900.0, help="Future polling timeout")
    sp.add_argument("--poll-interval-s", type=float, default=2.0, help="Future polling interval")
    sp.set_defaults(func=_cmd_actor_rebuild)

    sp = sub.add_parser("pg-list", help="List placement groups and topology")
    sp.add_argument("--state", choices=["PENDING", "CREATED", "REMOVED"], default=None, help="Filter by state")
    sp.add_argument("--include-removed-pg", action="store_true", help="Include REMOVED PGs")
    sp.set_defaults(func=_cmd_pg_list)

    sp = sub.add_parser("pg-remove", help="Remove placement groups (dry-run by default)")
    sp.add_argument("--name", action="append", default=[], help="PG name to remove (repeatable)")
    sp.add_argument("--state", choices=["PENDING", "CREATED"], default=None, help="Remove all PGs in this state")
    sp.add_argument("--only-gpu", action="store_true", help="Only target GPU-consuming PGs")
    sp.add_argument("--apply", action="store_true", help="Apply removal (default: dry-run)")
    sp.set_defaults(func=_cmd_pg_remove)

    sp = sub.add_parser("verify", help="Run operational verification checks")
    sp.add_argument("--sampling-model", action="append", default=[], help="Optional sampling smoke model (repeatable)")
    sp.add_argument("--sampling-models", default=None, help="Comma-separated sampling smoke models")
    sp.add_argument("--poll-timeout-s", type=float, default=300.0, help="Future polling timeout")
    sp.add_argument("--poll-interval-s", type=float, default=2.0, help="Future polling interval")
    sp.set_defaults(func=_cmd_verify)

    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    remote_rc = _maybe_exec_remote(args)
    if remote_rc is not None:
        return int(remote_rc)

    if getattr(args, "models", None):
        args.model.extend(_parse_csv(args.models))
    if getattr(args, "sampling_models", None):
        args.sampling_model.extend(_parse_csv(args.sampling_models))

    # De-duplicate while preserving order.
    if hasattr(args, "model"):
        seen = set()
        dedup = []
        for x in args.model:
            if x in seen:
                continue
            seen.add(x)
            dedup.append(x)
        args.model = dedup
    if hasattr(args, "sampling_model"):
        seen = set()
        dedup = []
        for x in args.sampling_model:
            if x in seen:
                continue
            seen.add(x)
            dedup.append(x)
        args.sampling_model = dedup

    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
