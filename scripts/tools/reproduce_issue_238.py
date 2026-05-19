#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class Env:
    base_url: str
    api_key: str
    ray_address: str
    ray_namespace: str
    ssh_host: str
    server_root: str
    mint_code_root: str
    port: int
    pidfile: str
    logfile: str


def _require_env(key: str) -> str:
    v = os.environ.get(key)
    if not v:
        raise SystemExit(f"error: missing env {key}")
    return v


def _coalesce(*vals: str | None) -> str | None:
    for v in vals:
        if v:
            return v
    return None


def _env() -> Env:
    port = int(os.environ.get("MINT_PORT", "10238"))
    base_url = os.environ.get("MINT_BASE_URL", f"http://localhost:{port}")
    api_key = _coalesce(os.environ.get("MINT_API_KEY"), os.environ.get("MINT_API_KEY"), "dummy") or "dummy"
    ray_address = _require_env("RAY_ADDRESS")
    ray_namespace = _coalesce(os.environ.get("MINT_RAY_NAMESPACE"), os.environ.get("MINT_RAY_NAMESPACE"))
    if not ray_namespace:
        raise SystemExit("error: missing env MINT_RAY_NAMESPACE or MINT_RAY_NAMESPACE")

    ssh_host = os.environ.get("MINT_ISSUE_SSH_HOST", "mint-dev")
    server_root = os.environ.get("MINT_ISSUE_SERVER_ROOT", "/root/mint_project/mint-server-issue-238")
    mint_code_root = _require_env("MINT_CODE_ROOT")
    pidfile = os.environ.get("MINT_ISSUE_PIDFILE", "/tmp/mint_server_issue_238.pid")
    logfile = os.environ.get("MINT_ISSUE_LOGFILE", "/tmp/mint_server_issue_238.log")
    return Env(
        base_url=str(base_url).rstrip("/"),
        api_key=str(api_key),
        ray_address=str(ray_address),
        ray_namespace=str(ray_namespace),
        ssh_host=str(ssh_host),
        server_root=str(server_root),
        mint_code_root=str(mint_code_root),
        port=int(port),
        pidfile=str(pidfile),
        logfile=str(logfile),
    )


def _ssh(env: Env, cmd: str) -> str:
    full = f"set -euo pipefail; {cmd}"
    p = subprocess.run(
        ["ssh", env.ssh_host, full],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return p.stdout


def _stop_server(env: Env) -> None:
    _ssh(env, f"test -f {env.pidfile} && xargs -r kill < {env.pidfile} || true; sleep 1")


def _start_server(env: Env) -> int:
    py_path = f"{env.server_root}:${{PYTHONPATH-}}"
    out = _ssh(
        env,
        "cd "
        + env.server_root
        + " && nohup bash -c "
        + json.dumps(
            "export MINT_PORT={port}; "
            "export MINT_RAY_NAMESPACE={ns}; "
            "export MINT_RAY_NAMESPACE={ns}; "
            "export MINT_CODE_ROOT={mint_root}; "
            "export MINT_TELEMETRY=0; "
            "PYTHONPATH={py_path} python scripts/run_server.py".format(
                port=int(env.port),
                ns=env.ray_namespace,
                mint_root=env.mint_code_root,
                py_path=py_path,
            )
        )
        + f" >> {env.logfile} 2>&1 & echo $! > {env.pidfile}; cat {env.pidfile}",
    ).strip()
    try:
        return int(out.splitlines()[-1].strip())
    except Exception as e:
        raise RuntimeError(f"failed to parse server pid from ssh output: {out!r}") from e


def _wait_ready(env: Env, *, timeout_s: float) -> None:
    deadline = time.time() + float(timeout_s)
    last_err: str | None = None
    while time.time() < deadline:
        try:
            r = requests.get(env.base_url + "/api/v1/healthz", timeout=2.0)
            if r.status_code == 200:
                return
            last_err = f"healthz status={r.status_code} body={r.text[:200]!r}"
        except Exception as e:
            last_err = f"healthz error: {type(e).__name__}: {e}"
        time.sleep(0.25)
    raise TimeoutError(f"server not ready within {timeout_s}s: {last_err}")


def _kill_namespace_actors(env: Env) -> None:
    out = _ssh(
        env,
        "RAY_ADDRESS="
        + json.dumps(env.ray_address)
        + " MINT_RAY_NAMESPACE="
        + json.dumps(env.ray_namespace)
        + " MINT_RAY_NAMESPACE="
        + json.dumps(env.ray_namespace)
        + " python - <<'PY'\n"
        "import os\n"
        "import ray\n"
        "ray.init(address=os.environ['RAY_ADDRESS'], ignore_reinit_error=True)\n"
        "ns = os.environ['MINT_RAY_NAMESPACE']\n"
        "actors = ray.util.list_named_actors(all_namespaces=True)\n"
        "killed = 0\n"
        "for a in actors:\n"
        "    if a.get('namespace') != ns:\n"
        "        continue\n"
        "    name = a.get('name')\n"
        "    if not name:\n"
        "        continue\n"
        "    try:\n"
        "        ray.kill(ray.get_actor(name, namespace=ns))\n"
        "        killed += 1\n"
        "    except Exception as e:\n"
        "        raise RuntimeError(f'kill_failed name={name!r} namespace={ns!r} err={e!r}')\n"
        "print(f'killed={killed} namespace={ns}')\n"
        "PY",
    )
    if "killed=" not in out:
        raise RuntimeError(f"unexpected kill output: {out!r}")


def _get_server_job_id(env: Env, pid: int) -> str:
    out = _ssh(
        env,
        "RAY_ADDRESS="
        + json.dumps(env.ray_address)
        + " python - <<'PY'\n"
        "import os\n"
        "import ray\n"
        "ray.init(address=os.environ['RAY_ADDRESS'], ignore_reinit_error=True)\n"
        "from ray._private.state import jobs\n"
        f"pid = int({int(pid)})\n"
        "for j in jobs():\n"
        "    if int(j.get('DriverPid') or 0) != pid:\n"
        "        continue\n"
        "    if int(j.get('EndTime') or 0) != 0:\n"
        "        continue\n"
        "    jid = j.get('JobID')\n"
        "    if isinstance(jid, str) and jid:\n"
        "        print(jid)\n"
        "        raise SystemExit(0)\n"
        "raise SystemExit('error: active job id not found for pid=' + str(pid))\n"
        "PY",
    ).strip()
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return lines[-1]


def _post_noop(env: Env) -> str:
    r = requests.post(env.base_url + "/internal/model_work_scheduler/noop", timeout=10.0)
    r.raise_for_status()
    payload = r.json()
    rid = payload.get("request_id")
    if not isinstance(rid, str) or not rid:
        raise RuntimeError(f"invalid noop response: {payload!r}")
    return rid


def _poll_future(env: Env, request_id: str, *, timeout_s: float) -> tuple[bool, int, str]:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        r = requests.post(
            env.base_url + "/api/v1/retrieve_future",
            json={"request_id": request_id},
            headers={"X-API-Key": env.api_key},
            timeout=10.0,
        )
        if r.status_code != 408:
            return True, int(r.status_code), r.text
        time.sleep(0.2)
    return False, 408, '{"queue_state":"active"}'


def _debug_state(env: Env) -> dict[str, Any]:
    r = requests.get(env.base_url + "/internal/model_work_scheduler/debug_state", timeout=10.0)
    r.raise_for_status()
    d = r.json()
    if not isinstance(d, dict):
        raise TypeError(f"debug_state returned non-dict: {type(d)}")
    return d


def main() -> int:
    env = _env()
    print(f"base_url={env.base_url} namespace={env.ray_namespace} port={env.port}", flush=True)

    print("restart_clean_slate=1", flush=True)
    _stop_server(env)
    _kill_namespace_actors(env)
    pid_a = _start_server(env)
    _wait_ready(env, timeout_s=30.0)
    job_a = _get_server_job_id(env, pid_a)
    print(f"server_a pid={pid_a} job_id={job_a}", flush=True)

    rid1 = _post_noop(env)
    ok1, code1, body1 = _poll_future(env, rid1, timeout_s=5.0)
    if not ok1:
        print(f"FAIL pre_restart still_pending request_id={rid1}", flush=True)
        print(json.dumps(_debug_state(env), sort_keys=True)[:2000], flush=True)
        return 2
    print(f"pre_restart done request_id={rid1} status={code1}", flush=True)

    print("restart_keep_queue_actor=1", flush=True)
    _stop_server(env)
    pid_b = _start_server(env)
    _wait_ready(env, timeout_s=30.0)
    job_b = _get_server_job_id(env, pid_b)
    print(f"server_b pid={pid_b} job_id={job_b}", flush=True)

    rid2 = _post_noop(env)
    ok2, code2, body2 = _poll_future(env, rid2, timeout_s=5.0)
    if not ok2:
        dbg = _debug_state(env)
        print(f"FAIL post_restart still_pending request_id={rid2}", flush=True)
        print(
            "debug_state stats={stats} last_dequeue={deq}".format(
                stats=dbg.get("stats"),
                deq=(dbg.get("recent_dequeues") or [])[-1] if isinstance(dbg.get("recent_dequeues"), list) else None,
            ),
            flush=True,
        )
        return 1

    print(f"post_restart done request_id={rid2} status={code2}", flush=True)
    _ = body1, body2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
