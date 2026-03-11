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
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


DEFAULT_REMOTE_PYTHON = "/root/tinker_project/tinker-server-auth/.venv31213/bin/python"
DEFAULT_SUPERVISOR_PROGRAM = "tinker-server-auth"


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

    server = http.server.ThreadingHTTPServer((bind, int(port)), _Handler)
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


def _maybe_exec_remote(args: argparse.Namespace) -> int | None:
    if not args.host:
        return None

    argv = sys.argv[1:]
    argv = _strip_flag_with_value(argv, "--host")
    argv = _strip_flag_with_value(argv, "--remote-python")
    local_md_out = getattr(args, "md_out", None)
    local_json_out = getattr(args, "json_out", None)
    local_html_out = getattr(args, "html_out", None)

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


def _status_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    proc = _pick_primary_server_process(args.program)
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
    for pg in pgs:
        if pg["state"] == "REMOVED":
            continue
        for ip, info in pg["node_distribution"].items():
            gpu = float(info["gpu"])
            if gpu <= 0:
                continue
            node_pg_gpu[ip] += gpu
            node_pg_names[ip].append(f"{pg['name']}({int(gpu)})")

    alive_actor_count_by_ip: Counter[str] = Counter()
    alive_actor_names_by_ip: defaultdict[str, list[str]] = defaultdict(list)
    for a in actor_details.get("actors", []):
        if str(a.get("state", "")).upper() != "ALIVE":
            continue
        ip = a.get("ip", "")
        if ip:
            alive_actor_count_by_ip[ip] += 1
            actor_name = str(a.get("name") or a.get("class_name") or "<unnamed>")
            gpu_req = float(a.get("num_gpus", 0) or 0)
            if gpu_req > 0:
                actor_name = f"{actor_name} (gpu={int(gpu_req)})"
            alive_actor_names_by_ip[ip].append(actor_name)

    for n in nodes:
        ip = n["ip"]
        n["gpu_reserved"] = int(node_pg_gpu.get(ip, 0))
        n["gpu_idle_est"] = max(n["gpu_total"] - n["gpu_reserved"], 0)
        n["pg_labels"] = sorted(node_pg_names.get(ip, []))
        n["ray_actor_count"] = int(alive_actor_count_by_ip.get(ip, 0))
        n["ray_actor_names"] = sorted(alive_actor_names_by_ip.get(ip, []))

    probes: list[dict[str, Any]] = []
    if not args.skip_machine_probe:
        probes = _collect_machine_probes(ray, nodes=nodes, timeout_s=args.timeout_s)

    managed_actors = actors_data.get("actors", []) if isinstance(actors_data, dict) else []
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
            "tinker_port": proc["env"].get("TINKER_PORT"),
            "otel_service_name": proc["env"].get("OTEL_SERVICE_NAME"),
            "otel_endpoint": proc["env"].get("OTEL_EXPORTER_OTLP_ENDPOINT"),
            "has_tinker_api_key": bool(proc["env"].get("TINKER_API_KEY")),
            "has_apm_app_key": bool(proc["env"].get("MINT_APMPLUS_APP_KEY")),
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
    summary = snapshot["summary"]
    ray_data = snapshot["ray"]
    nodes = ray_data["nodes"]
    pgs = ray_data["placement_groups"]
    actors_payload = snapshot["http"]["actors"]
    managed_actors = actors_payload.get("actors", []) if isinstance(actors_payload, dict) else []
    probes = snapshot.get("machine_probes", [])

    def esc(x: Any) -> str:
        return html_lib.escape(str(x))

    def render_bar(pct: float) -> str:
        pct = max(0.0, min(100.0, pct))
        tone = "var(--ok)"
        if pct >= 90:
            tone = "var(--bad)"
        elif pct >= 70:
            tone = "var(--warn)"
        return (
            '<div class="bar-wrap"><div class="bar-fill" '
            f'style="width:{pct:.1f}%;background:{tone}"></div></div>'
            f'<span class="bar-text">{pct:.1f}%</span>'
        )

    overview_cards = [
        ("GPU Total", summary["gpu_total"]),
        ("GPU Available", summary["gpu_available"]),
        ("GPU Nodes Alive", summary["gpu_nodes_alive"]),
        ("Pending PG", summary["pending_placement_groups"]),
        ("Managed Actors", summary["managed_actors"]),
        ("Healthz", snapshot["http"]["healthz_status"]),
    ]

    node_rows_html: list[str] = []
    for n in nodes:
        gpu_total = int(n["gpu_total"])
        gpu_reserved = int(n["gpu_reserved"])
        used_pct = (100.0 * gpu_reserved / gpu_total) if gpu_total > 0 else 0.0
        state = "alive" if n["alive"] else "dead"
        state_class = "ok" if n["alive"] else "bad"
        pg_labels = ", ".join(n.get("pg_labels", [])) or "-"
        actor_names = n.get("ray_actor_names", [])
        actor_detail = "<br/>".join(esc(x) for x in actor_names) if actor_names else "-"
        node_rows_html.append(
            "<tr>"
            f"<td>{esc(n['ip'])}</td>"
            f'<td><span class="pill {state_class}">{esc(state)}</span></td>'
            f"<td>{gpu_total}</td>"
            f"<td>{gpu_reserved}</td>"
            f"<td>{int(n['gpu_idle_est'])}</td>"
            f"<td>{render_bar(used_pct)}</td>"
            f"<td>{int(n.get('ray_actor_count', 0))}</td>"
            f"<td>{actor_detail}</td>"
            f"<td>{esc(pg_labels)}</td>"
            "</tr>"
        )

    pg_rows_html: list[str] = []
    for pg in pgs[:actor_limit]:
        dist = ", ".join(f"{ip}:{int(info['gpu'])}" for ip, info in pg["node_distribution"].items()) or "-"
        state_class = "warn" if pg["state"] == "PENDING" else ("ok" if pg["state"] == "CREATED" else "")
        pg_rows_html.append(
            "<tr>"
            f"<td>{esc(pg['name'])}</td>"
            f'<td><span class="pill {state_class}">{esc(pg["state"])}</span></td>'
            f"<td>{int(pg['requested_gpu'])}</td>"
            f"<td>{int(pg['bundle_count'])}</td>"
            f"<td>{esc(pg['strategy'] or '-')}</td>"
            f"<td>{esc(dist)}</td>"
            "</tr>"
        )

    actor_rows_html: list[str] = []
    for a in managed_actors[:actor_limit]:
        actor_rows_html.append(
            "<tr>"
            f"<td>{esc(a.get('actor_name', ''))}</td>"
            f"<td>{esc(a.get('actor_type', ''))}</td>"
            f"<td>{esc(a.get('base_model', ''))}</td>"
            f"<td>{esc(a.get('num_gpus', 0))}</td>"
            f"<td>{esc(round(float(a.get('idle_time', 0)), 1))}</td>"
            f"<td>{esc(round(float(a.get('age', 0)), 1))}</td>"
            f"<td>{'yes' if a.get('protected') else 'no'}</td>"
            f"<td>{'yes' if a.get('creating') else 'no'}</td>"
            "</tr>"
        )

    probe_rows_html: list[str] = []
    for p in probes:
        mem = p.get("memory", {})
        root_disk = p.get("disk", {}).get("/", {})
        pfs_disk = p.get("disk", {}).get("/vePFS-Mindverse", {})
        root_pct = float(root_disk.get("used_pct", 0) or 0)
        pfs_pct = float(pfs_disk.get("used_pct", 0) or 0)
        probe_rows_html.append(
            "<tr>"
            f"<td>{esc(p.get('ip', ''))}</td>"
            f"<td>{esc(p.get('hostname', ''))}</td>"
            f"<td>{esc(p.get('loadavg', {}).get('1m', '-'))}</td>"
            f"<td>{esc(mem.get('used_gb', '-'))}/{esc(mem.get('total_gb', '-'))} GiB</td>"
            f"<td>{render_bar(root_pct)}</td>"
            f"<td>{render_bar(pfs_pct)}</td>"
            f"<td>{esc(_gpu_probe_summary(p))}</td>"
            f"<td>{esc(p.get('_error') or p.get('_schedule_error') or '-')}</td>"
            "</tr>"
        )

    cards_html = "".join(
        f'<div class="card"><div class="k">{esc(k)}</div><div class="v">{esc(v)}</div></div>'
        for k, v in overview_cards
    )

    pg_truncated = ""
    if len(pgs) > actor_limit:
        pg_truncated = f'<p class="note">truncated: showing {actor_limit}/{len(pgs)} placement groups</p>'
    actor_truncated = ""
    if len(managed_actors) > actor_limit:
        actor_truncated = f'<p class="note">truncated: showing {actor_limit}/{len(managed_actors)} managed actors</p>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Mint Ops Status</title>
  <style>
    :root {{
      --bg: #f7f8fa;
      --panel: #ffffff;
      --text: #1f2937;
      --muted: #6b7280;
      --line: #e5e7eb;
      --ok: #16a34a;
      --warn: #d97706;
      --bad: #dc2626;
      --accent: #0f766e;
    }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    .wrap {{ max-width: 1400px; margin: 20px auto; padding: 0 16px 40px; }}
    .hdr {{
      background: linear-gradient(135deg, #f0fdfa 0%, #ecfeff 45%, #f8fafc 100%);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px 18px;
      margin-bottom: 16px;
    }}
    .hdr h1 {{ margin: 0; font-size: 22px; }}
    .meta {{ color: var(--muted); font-size: 13px; margin-top: 8px; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px;
    }}
    .card .k {{ color: var(--muted); font-size: 12px; }}
    .card .v {{ margin-top: 3px; font-weight: 700; font-size: 19px; }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px 12px 14px;
      margin-bottom: 14px;
    }}
    section h2 {{
      margin: 0 0 10px;
      font-size: 16px;
      color: var(--accent);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      table-layout: fixed;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      text-align: left;
      padding: 7px 6px;
      vertical-align: middle;
      word-wrap: break-word;
    }}
    th {{
      background: #f9fafb;
      font-weight: 600;
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    .tbl-wrap {{
      overflow: auto;
      max-height: 440px;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .tbl-wrap.no-scroll {{
      overflow: visible;
      max-height: none;
    }}
    .pill {{
      display: inline-block;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 11px;
      font-weight: 600;
      border: 1px solid var(--line);
      background: #fff;
    }}
    .pill.ok {{ color: var(--ok); border-color: color-mix(in srgb, var(--ok) 30%, white); }}
    .pill.warn {{ color: var(--warn); border-color: color-mix(in srgb, var(--warn) 30%, white); }}
    .pill.bad {{ color: var(--bad); border-color: color-mix(in srgb, var(--bad) 30%, white); }}
    .bar-wrap {{
      width: 110px;
      height: 8px;
      border-radius: 999px;
      background: #edf2f7;
      display: inline-block;
      vertical-align: middle;
      margin-right: 6px;
      overflow: hidden;
    }}
    .bar-fill {{ height: 100%; border-radius: 999px; }}
    .bar-text {{ color: var(--muted); font-size: 11px; }}
    .note {{ margin: 8px 2px 0; color: var(--muted); font-size: 12px; }}
    .toolbar {{
      margin-top: 10px;
      display: flex;
      gap: 10px;
      align-items: center;
    }}
    .btn {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 6px 10px;
      cursor: pointer;
      font-size: 12px;
      font-weight: 600;
    }}
    .btn:hover {{ background: #f3f4f6; }}
    .hint {{ color: var(--muted); font-size: 12px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hdr">
      <h1>Mint Ops Status</h1>
      <div class="meta">
        generated_at_utc={esc(snapshot['generated_at_utc'])} |
        host={esc(snapshot['host'])} |
        ray_address={esc(snapshot['ray_address'])} |
        server_pid={esc(snapshot['server_process']['pid'])}
      </div>
      <div class="toolbar">
        <button class="btn" id="refreshBtn" onclick="refreshReport()">Refresh</button>
        <span class="hint" id="refreshHint">POST /refresh (serve mode) or browser reload fallback</span>
      </div>
    </div>

    <div class="cards">{cards_html}</div>

    <section>
      <h2>GPU Topology</h2>
      <div class="tbl-wrap no-scroll">
        <table>
          <thead>
            <tr>
              <th>ip</th><th>state</th><th>gpu_total</th><th>gpu_reserved</th><th>gpu_idle_est</th>
              <th>gpu_used_pct</th><th>ray_actors</th><th>ray_actor_names</th><th>pg_labels</th>
            </tr>
          </thead>
          <tbody>
            {"".join(node_rows_html)}
          </tbody>
        </table>
      </div>
    </section>

    <section>
      <h2>Placement Groups</h2>
      <div class="tbl-wrap">
        <table>
          <thead>
            <tr><th>name</th><th>state</th><th>gpu</th><th>bundles</th><th>strategy</th><th>node_distribution</th></tr>
          </thead>
          <tbody>
            {"".join(pg_rows_html)}
          </tbody>
        </table>
      </div>
      {pg_truncated}
    </section>

    <section>
      <h2>Managed Actors</h2>
      <div class="tbl-wrap">
        <table>
          <thead>
            <tr><th>actor_name</th><th>type</th><th>base_model</th><th>gpu</th><th>idle_s</th><th>age_s</th><th>protected</th><th>creating</th></tr>
          </thead>
          <tbody>
            {"".join(actor_rows_html)}
          </tbody>
        </table>
      </div>
      {actor_truncated}
    </section>

    <section>
      <h2>Machine Status</h2>
      <div class="tbl-wrap">
        <table>
          <thead>
            <tr><th>ip</th><th>hostname</th><th>load1</th><th>mem_used/total</th><th>root_used</th><th>pfs_used</th><th>gpu_probe</th><th>error</th></tr>
          </thead>
          <tbody>
            {"".join(probe_rows_html) if probe_rows_html else "<tr><td colspan='8'>machine probe skipped</td></tr>"}
          </tbody>
        </table>
      </div>
    </section>
  </div>
  <script>
    async function refreshReport() {{
      const btn = document.getElementById('refreshBtn');
      const hint = document.getElementById('refreshHint');
      btn.disabled = true;
      hint.textContent = 'Refreshing...';
      try {{
        const resp = await fetch('/refresh', {{ method: 'POST' }});
        if (!resp.ok) {{
          throw new Error('HTTP ' + resp.status);
        }}
        const payload = await resp.json();
        if (!payload.ok) {{
          throw new Error(payload.error || 'refresh failed');
        }}
        hint.textContent = 'Refreshed at ' + (payload.generated_at_utc || '');
        window.location.reload();
      }} catch (err) {{
        hint.textContent = 'Refresh endpoint unavailable; reloading page';
        window.location.reload();
      }}
    }}
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
    if args.serve and args.html_out is None:
        args.html_out = Path("mint_ops_status.html")

    snap = _status_snapshot(args)
    md, _html_report = _write_status_outputs(args=args, snapshot=snap, print_messages=True)
    if not args.md_out and not args.html_out:
        print(md)
    if args.json:
        print(json.dumps(snap, ensure_ascii=False, indent=2))

    if args.serve:
        if args.html_out is None:
            raise RuntimeError("--serve requires html output path")

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
        )
    return 0


def _cmd_server_restart(args: argparse.Namespace) -> int:
    before = _find_run_server_processes(args.program)
    print("before_run_server_pids:", [p["pid"] for p in before])

    if args.clean_dirty:
        dirty = [p for p in before if not p["is_supervisor_managed"]]
        if dirty:
            print("cleaning_dirty_pids:", [p["pid"] for p in dirty])
            for p in dirty:
                try:
                    os.kill(int(p["pid"]), 15)
                except ProcessLookupError:
                    pass
            time.sleep(1.5)
            for p in dirty:
                try:
                    os.kill(int(p["pid"]), 0)
                    os.kill(int(p["pid"]), 9)
                except ProcessLookupError:
                    pass
        else:
            print("cleaning_dirty_pids: none")

    cp = _run(["supervisorctl", "restart", args.program], timeout_s=60)
    if cp.returncode != 0:
        print(cp.stdout, end="")
        print(cp.stderr, end="", file=sys.stderr)
        raise RuntimeError(f"supervisor restart failed rc={cp.returncode}")
    print(cp.stdout.strip())

    deadline = time.time() + args.wait_healthz_s
    last: tuple[int, Any] | None = None
    while time.time() < deadline:
        st, data = _api_healthz(args)
        last = (st, data)
        if st == 200:
            break
        time.sleep(1.5)

    after = _find_run_server_processes(args.program)
    print("after_run_server_pids:", [p["pid"] for p in after])

    if not after:
        raise RuntimeError("No run_server.py found after restart")

    primary = _pick_primary_server_process(args.program)
    print(
        "primary_server:",
        json.dumps(
            {
                "pid": primary["pid"],
                "is_supervisor_managed": primary["is_supervisor_managed"],
                "supervisor_process_name": primary["supervisor_process_name"],
                "tinker_port": primary["env"].get("TINKER_PORT"),
                "otel_endpoint": primary["env"].get("OTEL_EXPORTER_OTLP_ENDPOINT"),
                "has_apm_app_key": bool(primary["env"].get("MINT_APMPLUS_APP_KEY")),
            },
            ensure_ascii=False,
        ),
    )

    if last is None or last[0] != 200:
        raise RuntimeError(f"healthz not ready after restart: last={last!r}")
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
    payload: dict[str, Any] = {"actor_type": args.actor_type}
    if args.model_name:
        payload["model_name"] = args.model_name
    st, data = _http_json(
        "POST",
        _base_url(args) + "/api/v1/actors/kill",
        payload=payload,
        headers=_admin_headers(args, required=True),
        timeout_s=max(args.timeout_s, 60.0),
    )
    if st != 200 or not isinstance(data, dict):
        raise RuntimeError(f"POST /actors/kill failed status={st} body={data!r}")
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


def _cmd_actor_rebuild(args: argparse.Namespace) -> int:
    models = args.model or []
    if not models:
        raise RuntimeError("at least one --model is required")

    results: list[dict[str, Any]] = []
    for model in models:
        item: dict[str, Any] = {"model": model, "kind": args.kind, "status": "UNKNOWN"}
        t0 = time.time()
        try:
            session_id = _api_create_session(args, tag="scripts/ops/mint_ops.py:actor-rebuild")
            item["session_id"] = session_id
            if args.kind == "vllm":
                sampling_session_id = _api_create_sampling_session(args, session_id=session_id, model=model)
                item["sampling_session_id"] = sampling_session_id
                if args.sample_ping:
                    _ = _api_asample_ping(
                        args,
                        sampling_session_id=sampling_session_id,
                        poll_timeout_s=args.poll_timeout_s,
                        poll_interval_s=args.poll_interval_s,
                    )
            else:
                created = _api_create_model(args, session_id=session_id, model=model, lora_rank=args.lora_rank)
                item["create_model"] = created
                request_id = str(created["request_id"])
                item["future"] = _poll_future(
                    args,
                    request_id=request_id,
                    timeout_s=args.poll_timeout_s,
                    interval_s=args.poll_interval_s,
                )
            item["status"] = "PASS"
        except Exception as e:
            item["status"] = "FAIL"
            item["error"] = f"{type(e).__name__}: {e}"
        finally:
            item["elapsed_s"] = round(time.time() - t0, 2)
        results.append(item)

    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
    return 0 if all(x["status"] == "PASS" for x in results) else 1


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
    ray = _ray_init(args.address)
    nodes, id_to_ip = _collect_nodes(ray)
    pgs = _collect_placement_groups(ray, include_removed=True, id_to_ip=id_to_ip)

    names = set(args.name or [])
    targets: list[dict[str, Any]] = []
    for pg in pgs:
        if pg["state"] == "REMOVED":
            continue
        if names and pg["name"] in names:
            targets.append(pg)
            continue
        if args.state and pg["state"] == args.state:
            targets.append(pg)

    if args.only_gpu:
        targets = [pg for pg in targets if float(pg["requested_gpu"]) > 0]

    targets.sort(key=lambda x: x["name"])
    print(json.dumps({"targets": targets, "apply": args.apply}, ensure_ascii=False, indent=2))
    if not targets:
        return 0
    if not args.apply:
        print("dry-run only. pass --apply to remove targets.")
        return 0

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

    print(json.dumps({"removed": removed, "failed": failed}, ensure_ascii=False, indent=2))
    return 0 if not failed else 1


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
    sp.set_defaults(func=_cmd_status)

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
