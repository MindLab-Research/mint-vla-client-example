#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
import uuid
from typing import Any

import requests


BASE_URL = (os.environ.get("TINKER_BASE_URL") or "http://localhost:10164").rstrip("/")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")

INFER_MODEL = os.environ.get("TINKER_INFER_MODEL", "Qwen/Qwen3-0.6B")

POLL_TIMEOUT_S = float(os.environ.get("TINKER_POLL_TIMEOUT_S", "120"))
HTTP_TIMEOUT_S = float(os.environ.get("TINKER_HTTP_TIMEOUT_S", "30"))
IDLE_WAIT_S = float(os.environ.get("TINKER_IDLE_WAIT_S", "2.0"))

MAX_TOKENS = int(os.environ.get("TINKER_MAX_TOKENS", "2048"))


def _headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY} if API_KEY else {}


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr, flush=True)
    return 1


def _get_json(path: str) -> tuple[int, dict[str, Any]]:
    url = f"{BASE_URL}{path}"
    resp = requests.get(url, headers=_headers(), timeout=HTTP_TIMEOUT_S)
    try:
        data = resp.json()
    except Exception:
        data = {"_non_json_body": resp.text[:400]}
    if not isinstance(data, dict):
        data = {"_non_dict_json": str(type(data))}
    return resp.status_code, data


def _post_json(path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    url = f"{BASE_URL}{path}"
    resp = requests.post(url, headers=_headers(), json=payload, timeout=HTTP_TIMEOUT_S)
    try:
        data = resp.json()
    except Exception:
        data = {"_non_json_body": resp.text[:400]}
    if not isinstance(data, dict):
        data = {"_non_dict_json": str(type(data))}
    return resp.status_code, data


def _poll_future(request_id: str, *, timeout_s: float) -> tuple[int, dict[str, Any]]:
    start = time.time()
    while True:
        elapsed = time.time() - start
        if elapsed > timeout_s:
            raise TimeoutError(f"retrieve_future timeout request_id={request_id} elapsed_s={elapsed:.1f}")
        status, data = _post_json("/api/v1/retrieve_future", {"request_id": request_id})
        if status in (200, 404):
            return status, data
        if status != 408:
            return status, data
        time.sleep(0.5)


def _get_vllm_actor_entry() -> dict[str, Any] | None:
    status, data = _get_json(f"/api/v1/actors?type=vllm&model_name={INFER_MODEL}")
    if status != 200:
        raise RuntimeError(f"GET /actors returned {status}: {data!r}")
    actors = data.get("actors")
    if not isinstance(actors, list):
        raise RuntimeError(f"GET /actors missing actors list: {data!r}")
    for a in actors:
        if isinstance(a, dict) and a.get("base_model") == INFER_MODEL:
            return a
    return None


def _create_sampling_session() -> str:
    status, sess = _post_json(
        "/api/v1/create_session",
        {"tags": ["scripts/tools/reproduce_issue_164.py"], "user_metadata": {}, "sdk_version": "repro-164"},
    )
    if status != 200:
        raise RuntimeError(f"create_session returned {status}: {sess!r}")
    session_id = sess.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise RuntimeError(f"create_session missing session_id: {sess!r}")

    status, out = _post_json(
        "/api/v1/create_sampling_session",
        {"session_id": session_id, "sampling_session_seq_id": 0, "base_model": INFER_MODEL},
    )
    if status != 200:
        raise RuntimeError(f"create_sampling_session returned {status}: {out!r}")
    sampling_session_id = out.get("sampling_session_id")
    if not isinstance(sampling_session_id, str) or not sampling_session_id:
        raise RuntimeError(f"create_sampling_session missing sampling_session_id: {out!r}")
    return sampling_session_id


def _asample(sampling_session_id: str, *, request_id: str, max_tokens: int) -> str:
    status, out = _post_json(
        "/api/v1/asample",
        {
            "sampling_session_id": sampling_session_id,
            "seq_id": 0,
            "num_samples": 1,
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 1, 1, 1]}]},
            "sampling_params": {"max_tokens": int(max_tokens), "temperature": 0.0, "top_k": 1, "top_p": 1.0},
            "request_id": request_id,
        },
    )
    if status != 200:
        raise RuntimeError(f"asample returned {status}: {out!r}")
    rid = out.get("request_id")
    if not isinstance(rid, str) or not rid:
        raise RuntimeError(f"asample missing request_id: {out!r}")
    return rid


def _assert_busy_vllm_actor_not_idle_while_future_pending(sampling_session_id: str) -> None:
    # Warmup: force vLLM actor creation.
    warm_rid = _asample(sampling_session_id, request_id=f"repro-164-warm-{uuid.uuid4()}", max_tokens=1)
    status, data = _poll_future(warm_rid, timeout_s=POLL_TIMEOUT_S)
    if status != 200 or "error" in data:
        raise RuntimeError(f"warmup sample failed status={status} body={data!r}")

    # Ensure the vLLM actor exists and is considered idle under a short idle timeout.
    a = _get_vllm_actor_entry()
    if a is None:
        raise RuntimeError(f"vLLM actor not found for base_model={INFER_MODEL!r} (after warmup)")
    print(f"[warmup] actor_name={a.get('actor_name')} idle={a.get('idle')} idle_time={a.get('idle_time')}", flush=True)

    time.sleep(IDLE_WAIT_S)
    a = _get_vllm_actor_entry()
    if a is None:
        raise RuntimeError(f"vLLM actor missing after idle wait base_model={INFER_MODEL!r}")
    print(f"[pre] idle_wait_s={IDLE_WAIT_S} idle={a.get('idle')} idle_time={a.get('idle_time')}", flush=True)

    # Start a request that should take > idle_timeout_s, then verify the actor
    # does not remain idle while the future is pending.
    infer_rid = _asample(sampling_session_id, request_id=f"repro-164-infer-{uuid.uuid4()}", max_tokens=MAX_TOKENS)
    start = time.time()

    saw_pending = False
    while True:
        elapsed = time.time() - start
        if elapsed > POLL_TIMEOUT_S:
            raise TimeoutError(f"infer did not finish within poll_timeout_s={POLL_TIMEOUT_S}")

        st, fut = _post_json("/api/v1/retrieve_future", {"request_id": infer_rid})
        if st == 408:
            saw_pending = True
            a = _get_vllm_actor_entry()
            if a is None:
                raise RuntimeError("vLLM actor missing while inference future pending")
            idle = a.get("idle")
            idle_time = a.get("idle_time")
            print(f"[pending] dt_s={elapsed:.1f} idle={idle} idle_time={idle_time}", flush=True)
            if idle is True:
                raise RuntimeError(
                    "ResourcePool marks vLLM actor idle while inference is in-flight "
                    f"(idle_time={idle_time}, infer_request_id={infer_rid})"
                )
            time.sleep(0.5)
            continue

        if st == 200:
            if not saw_pending:
                raise RuntimeError(
                    "Inference finished too quickly; cannot validate keepalive. "
                    f"Try increasing TINKER_MAX_TOKENS (currently {MAX_TOKENS})."
                )
            if "error" in fut:
                raise RuntimeError(f"infer future returned error: {fut!r}")
            seqs = fut.get("sequences")
            if not isinstance(seqs, list) or not seqs:
                raise RuntimeError(f"infer future missing sequences: {fut!r}")
            return

        raise RuntimeError(f"retrieve_future returned unexpected status={st} body={fut!r}")


def main() -> int:
    try:
        st, health = _get_json("/api/v1/healthz")
        if st != 200:
            return _fail(f"healthz returned {st}: {health!r}")

        sampling_session_id = _create_sampling_session()
        _assert_busy_vllm_actor_not_idle_while_future_pending(sampling_session_id)
        print("PASS", flush=True)
        return 0
    except Exception as e:
        return _fail(str(e))


if __name__ == "__main__":
    raise SystemExit(main())
