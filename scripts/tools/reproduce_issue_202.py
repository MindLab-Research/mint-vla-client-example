from __future__ import annotations

import os
import sys
import time
from typing import Any

import requests


BASE_URL = os.environ.get("MINT_BASE_URL")
if not BASE_URL:
    port = os.environ.get("MINT_PORT", "10202")
    BASE_URL = f"http://localhost:{port}"
BASE_URL = BASE_URL.rstrip("/")

API_KEY = os.environ.get("MINT_API_KEY", "dummy")

BASE_MODEL = os.environ.get("MINT_BASE_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")

TOPK_PROMPT_LOGPROBS = int(os.environ.get("MINT_TOPK_PROMPT_LOGPROBS", "1"))

POLL_TIMEOUT_S = float(os.environ.get("MINT_POLL_TIMEOUT_S", "1800"))
POLL_SLEEP_S = float(os.environ.get("MINT_POLL_SLEEP_S", "2.0"))

CREATE_SESSION_TIMEOUT_S = float(os.environ.get("MINT_CREATE_SESSION_TIMEOUT_S", "30"))
CREATE_SAMPLING_TIMEOUT_S = float(os.environ.get("MINT_CREATE_SAMPLING_TIMEOUT_S", "120"))
ASAMPLE_TIMEOUT_S = float(os.environ.get("MINT_ASAMPLE_TIMEOUT_S", "60"))


def _headers() -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["X-API-Key"] = API_KEY
    return h


def _post_json(path: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    url = f"{BASE_URL}{path}"
    resp = requests.post(url, headers=_headers(), json=payload, timeout=timeout_s)
    if resp.status_code >= 400:
        raise RuntimeError(f"POST {path} -> {resp.status_code}: {resp.text[:400]!r}")
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"POST {path} -> non-dict json: {data!r}")
    return data


def _create_session() -> str:
    out = _post_json(
        "/api/v1/create_session",
        {
            "tags": [],
            "user_metadata": {},
            "sdk_version": "reproduce_issue_202",
            "type": "create_session",
        },
        timeout_s=CREATE_SESSION_TIMEOUT_S,
    )
    session_id = out.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise RuntimeError(f"create_session missing/invalid session_id: {out!r}")
    return session_id


def _create_sampling_session(session_id: str) -> str:
    out = _post_json(
        "/api/v1/create_sampling_session",
        {
            "session_id": session_id,
            "sampling_session_seq_id": 0,
            "base_model": BASE_MODEL,
            "lora_rank": 32,
        },
        timeout_s=CREATE_SAMPLING_TIMEOUT_S,
    )
    sampling_session_id = out.get("sampling_session_id")
    if not isinstance(sampling_session_id, str) or not sampling_session_id:
        raise RuntimeError(f"create_sampling_session missing/invalid sampling_session_id: {out!r}")
    return sampling_session_id


def _asample_once(*, sampling_session_id: str) -> str:
    out = _post_json(
        "/api/v1/asample",
        {
            "sampling_session_id": sampling_session_id,
            "seq_id": 0,
            "num_samples": 1,
            "prompt": {"chunks": [{"tokens": [1, 2], "type": "encoded_text"}]},
            "sampling_params": {
                "max_tokens": 1,
                "temperature": 1.0,
                "top_k": -1,
                "top_p": 1.0,
                "stop": None,
                "seed": None,
            },
            "prompt_logprobs": True,
            "topk_prompt_logprobs": TOPK_PROMPT_LOGPROBS,
            "include_prompt_logprobs": False,
        },
        timeout_s=ASAMPLE_TIMEOUT_S,
    )
    request_id = out.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise RuntimeError(f"asample missing/invalid request_id: {out!r}")
    return request_id


def _retrieve_future(request_id: str) -> dict[str, Any] | None:
    url = f"{BASE_URL}/api/v1/retrieve_future"
    resp = requests.post(url, headers=_headers(), json={"request_id": request_id}, timeout=30.0)
    if resp.status_code == 408:
        return None
    if resp.status_code >= 400:
        raise RuntimeError(f"retrieve_future -> {resp.status_code}: {resp.text[:400]!r}")
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"retrieve_future -> non-dict json: {data!r}")
    return data


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def main() -> int:
    print(
        f"BASE_URL={BASE_URL} base_model={BASE_MODEL} topk_prompt_logprobs={TOPK_PROMPT_LOGPROBS}",
        flush=True,
    )

    session_id = _create_session()
    sampling_session_id = _create_sampling_session(session_id)
    request_id = _asample_once(sampling_session_id=sampling_session_id)

    t0 = time.time()
    while True:
        if time.time() - t0 > POLL_TIMEOUT_S:
            return _fail(f"timeout waiting for retrieve_future: request_id={request_id}")
        out = _retrieve_future(request_id)
        if out is None:
            time.sleep(POLL_SLEEP_S)
            continue

        err = out.get("error")
        if err:
            return _fail(f"future error: {err!r}")

        topk = out.get("topk_prompt_logprobs")
        if not isinstance(topk, list) or len(topk) != 2:
            return _fail(f"missing/invalid topk_prompt_logprobs: {topk!r}")
        if topk[0] is not None:
            return _fail(f"topk_prompt_logprobs[0] expected None, got: {topk[0]!r}")
        if not isinstance(topk[1], list) or not topk[1]:
            return _fail(f"topk_prompt_logprobs[1] expected non-empty list, got: {topk[1]!r}")
        pair0 = topk[1][0]
        if not (isinstance(pair0, (list, tuple)) and len(pair0) == 2):
            return _fail(f"topk_prompt_logprobs[1][0] expected pair, got: {pair0!r}")

        print("PASS")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

