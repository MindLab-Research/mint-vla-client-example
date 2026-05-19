#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
import uuid
from typing import Any

import requests


BASE_URL = (os.environ.get("MINT_BASE_URL") or "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("MINT_API_KEY", "dummy")
MODEL = os.environ.get("MINT_MODEL", "Qwen/Qwen3-0.6B")
PENDING_WAIT_S = float(os.environ.get("MINT_PENDING_WAIT_S", "30"))
FINAL_TIMEOUT_S = float(os.environ.get("MINT_FINAL_TIMEOUT_S", "180"))
MAX_TOKENS = int(os.environ.get("MINT_MAX_TOKENS", "512"))


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    return headers


def _fail(message: str) -> int:
    print(f"FAIL: {message}", file=sys.stderr)
    return 1


def _post_json(path: str, payload: dict[str, Any], *, expected_status: int = 200, timeout_s: float = 60.0) -> tuple[int, Any]:
    response = requests.post(
        f"{BASE_URL}{path}",
        headers=_headers(),
        json=payload,
        timeout=timeout_s,
    )
    body: Any
    try:
        body = response.json()
    except Exception:
        body = response.text
    if response.status_code != expected_status:
        raise RuntimeError(f"POST {path} -> {response.status_code}: {body!r}")
    return response.status_code, body


def _wait_until_pending(request_id: str) -> None:
    deadline = time.time() + PENDING_WAIT_S
    while time.time() < deadline:
        response = requests.post(
            f"{BASE_URL}/api/v1/retrieve_future",
            headers=_headers(),
            json={"request_id": request_id},
            timeout=30.0,
        )
        if response.status_code == 408:
            return
        if response.status_code != 200:
            raise RuntimeError(f"retrieve_future -> {response.status_code}: {response.text[:500]!r}")
        payload = response.json()
        raise RuntimeError(f"request completed before kill attempt; payload={payload!r}")
    raise RuntimeError(f"request_id={request_id} did not enter pending state within {PENDING_WAIT_S:.1f}s")


def _wait_until_done(request_id: str) -> dict[str, Any]:
    deadline = time.time() + FINAL_TIMEOUT_S
    while time.time() < deadline:
        response = requests.post(
            f"{BASE_URL}/api/v1/retrieve_future",
            headers=_headers(),
            json={"request_id": request_id},
            timeout=30.0,
        )
        if response.status_code == 408:
            time.sleep(0.5)
            continue
        if response.status_code != 200:
            raise RuntimeError(f"retrieve_future -> {response.status_code}: {response.text[:500]!r}")
        payload = response.json()
        if isinstance(payload, dict) and "error" in payload:
            raise RuntimeError(f"retrieve_future returned error payload: {payload!r}")
        if not isinstance(payload, dict):
            raise RuntimeError(f"retrieve_future returned unexpected payload: {payload!r}")
        return payload
    raise RuntimeError(f"request_id={request_id} did not complete within {FINAL_TIMEOUT_S:.1f}s")


def main() -> int:
    try:
        requests.get(f"{BASE_URL}/api/v1/healthz", timeout=5.0).raise_for_status()

        _, session = _post_json(
            "/api/v1/create_session",
            {
                "tags": ["issue-364", "busy-actor-kill-guard"],
                "user_metadata": {"purpose": "issue_364_busy_actor_kill_guard"},
                "sdk_version": "scripts/tools/reproduce_issue_364_busy_actor_kill_guard.py",
            },
        )
        session_id = str(session["session_id"])

        _, sampling = _post_json(
            "/api/v1/create_sampling_session",
            {
                "session_id": session_id,
                "sampling_session_seq_id": 364,
                "base_model": MODEL,
            },
            timeout_s=90.0,
        )
        sampling_session_id = str(sampling["sampling_session_id"])

        client_tag = uuid.uuid4().hex[:8]
        _, submitted = _post_json(
            "/api/v1/asample",
            {
                "sampling_session_id": sampling_session_id,
                "seq_id": 0,
                "num_samples": 1,
                "prompt": {"chunks": [{"tokens": [11, 12, 13, 14], "type": "encoded_text"}]},
                "sampling_params": {
                    "max_tokens": MAX_TOKENS,
                    "temperature": 0.0,
                    "top_k": 1,
                    "top_p": 1.0,
                },
                "client_request_id": f"issue364-busy-guard-{client_tag}",
            },
            timeout_s=60.0,
        )
        request_id = str(submitted["request_id"])

        _wait_until_pending(request_id)

        kill_response = requests.post(
            f"{BASE_URL}/internal/actors/kill",
            headers=_headers(),
            json={"actor_type": "vllm", "model_name": MODEL},
            timeout=30.0,
        )
        if kill_response.status_code != 409:
            raise RuntimeError(
                f"expected kill guard 409, got {kill_response.status_code}: {kill_response.text[:500]!r}"
            )

        final_payload = _wait_until_done(request_id)
        print(
            f"PASS: kill blocked with 409 while request stayed alive; "
            f"request_id={request_id} session_id={session_id} final_keys={sorted(final_payload.keys())}"
        )
        return 0
    except Exception as exc:
        return _fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
