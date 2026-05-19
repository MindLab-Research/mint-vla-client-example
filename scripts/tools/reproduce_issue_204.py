from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid
from typing import Any

import requests


BASE_URL = os.environ.get("MINT_BASE_URL")
if not BASE_URL:
    port = os.environ.get("MINT_PORT", "10204")
    BASE_URL = f"http://localhost:{port}"
BASE_URL = BASE_URL.rstrip("/")

API_KEY = os.environ.get("MINT_API_KEY", "dummy")

SSH_HOST = os.environ.get("MINT_SSH_HOST", "mint-dev").strip()
SERVER_LOG_PATH = os.environ.get("MINT_SERVER_LOG_PATH", "/tmp/mint_server_issue_204.log").strip()

BASE_MODEL = os.environ.get("MINT_BASE_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")

POLL_TIMEOUT_S = float(os.environ.get("MINT_POLL_TIMEOUT_S", "1800"))
POLL_SLEEP_S = float(os.environ.get("MINT_POLL_SLEEP_S", "2.0"))

CREATE_SESSION_TIMEOUT_S = float(os.environ.get("MINT_CREATE_SESSION_TIMEOUT_S", "30"))
CREATE_SAMPLING_TIMEOUT_S = float(os.environ.get("MINT_CREATE_SAMPLING_TIMEOUT_S", "120"))
ASAMPLE_TIMEOUT_S = float(os.environ.get("MINT_ASAMPLE_TIMEOUT_S", "60"))


def _headers() -> dict[str, str]:
    if not API_KEY:
        return {}
    return {"X-API-Key": API_KEY}


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _post_json(path: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    r = requests.post(f"{BASE_URL}{path}", headers=_headers(), json=payload, timeout=timeout_s)
    if r.status_code != 200:
        raise RuntimeError(f"POST {path} -> {r.status_code}: {r.text[:500]!r}")
    data = r.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"POST {path} returned non-dict json: {type(data)}")
    return data


def _poll_future(request_id: str) -> dict[str, Any]:
    start = time.time()
    while True:
        if time.time() - start > POLL_TIMEOUT_S:
            raise TimeoutError(f"retrieve_future timed out after {POLL_TIMEOUT_S:.1f}s (request_id={request_id})")
        r = requests.post(
            f"{BASE_URL}/api/v1/retrieve_future",
            headers=_headers(),
            json={"request_id": request_id},
            timeout=30.0,
        )
        if r.status_code == 408:
            time.sleep(POLL_SLEEP_S)
            continue
        if r.status_code != 200:
            raise RuntimeError(f"retrieve_future -> {r.status_code}: {r.text[:500]!r}")
        data = r.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"retrieve_future returned non-dict json: {type(data)}")
        return data


def _ssh_read_forwarded_logs_present() -> bool:
    # Ray-forwarded logs typically appear with a worker prefix like:
    #   "(MultiNodeVLLMEngine pid=..., ip=...) ..."
    cmd = (
        f"rg -n \"\\(MultiNodeVLLMEngine pid=\" {SERVER_LOG_PATH} | head -n 1"
        if SERVER_LOG_PATH
        else "false"
    )
    out = subprocess.check_output(["ssh", SSH_HOST, cmd], text=True, stderr=subprocess.STDOUT)
    return bool(out.strip())


def main() -> int:
    print(
        f"BASE_URL={BASE_URL} base_model={BASE_MODEL} ssh_host={SSH_HOST} server_log={SERVER_LOG_PATH}",
        flush=True,
    )
    try:
        r = requests.get(f"{BASE_URL}/api/v1/healthz", headers=_headers(), timeout=5.0)
        if r.status_code != 200:
            return _fail(f"healthz -> {r.status_code}: {r.text[:200]!r}")

        session = _post_json(
            "/api/v1/create_session",
            payload={
                "tags": ["scripts/tools/reproduce_issue_204.py", f"repro-204-{uuid.uuid4().hex[:8]}"],
                "user_metadata": {},
                "sdk_version": "scripts/tools/reproduce_issue_204.py",
            },
            timeout_s=CREATE_SESSION_TIMEOUT_S,
        )
        session_id = session.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            return _fail(f"create_session missing/invalid session_id: {session!r}")

        sampling = _post_json(
            "/api/v1/create_sampling_session",
            payload={
                "session_id": session_id,
                "sampling_session_seq_id": 0,
                "base_model": BASE_MODEL,
                "lora_rank": 32,
            },
            timeout_s=CREATE_SAMPLING_TIMEOUT_S,
        )
        sampling_session_id = sampling.get("sampling_session_id")
        if not isinstance(sampling_session_id, str) or not sampling_session_id:
            return _fail(f"create_sampling_session missing/invalid sampling_session_id: {sampling!r}")

        out = _post_json(
            "/api/v1/asample",
            payload={
                "sampling_session_id": sampling_session_id,
                "seq_id": 0,
                "num_samples": 1,
                "prompt": {"chunks": [{"tokens": [1, 2], "type": "encoded_text"}]},
                "sampling_params": {"max_tokens": 1, "temperature": 0.0, "top_k": -1, "top_p": 1.0},
            },
            timeout_s=ASAMPLE_TIMEOUT_S,
        )
        request_id = out.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            return _fail(f"asample missing request_id: {out!r}")

        res = _poll_future(request_id)
        if res.get("error"):
            return _fail(f"asample returned error: {res.get('error')!r}")

        if _ssh_read_forwarded_logs_present():
            return _fail(
                "Ray worker logs are being forwarded into the API server log "
                "(expected fix is to disable log_to_driver by default)."
            )

        print("PASS")
        return 0
    except Exception as e:
        return _fail(str(e))


if __name__ == "__main__":
    raise SystemExit(main())

