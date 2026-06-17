#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures
import os
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import Any

import requests

from mint_server.models.types import ModelInput, SampleRequest, SamplingParams


@dataclass(frozen=True)
class Env:
    base_url: str
    internal_api_token: str
    flood_count: int
    timeout_s: float


def _fail(message: str) -> int:
    print(f"FAIL: {message}", flush=True)
    return 1


def _env() -> Env:
    base_url = (os.environ.get("MINT_BASE_URL") or "http://localhost:8000").rstrip("/")
    internal_api_token = (os.environ.get("MINT_INTERNAL_API_TOKEN") or "").strip()
    if not internal_api_token:
        raise SystemExit("error: missing env MINT_INTERNAL_API_TOKEN")
    flood_count = int(os.environ.get("ISSUE324_FLOOD_COUNT") or "4")
    timeout_s = float(os.environ.get("ISSUE324_TIMEOUT_S") or "40")
    return Env(
        base_url=base_url,
        internal_api_token=internal_api_token,
        flood_count=flood_count,
        timeout_s=timeout_s,
    )


def _headers(*, user_id: str, apikey_id: str, env: Env) -> dict[str, str]:
    return {
        "X-MinT-User-Id": user_id,
        "X-MinT-User-Role": "user",
        "X-MinT-Account-Id": user_id,
        "X-MinT-Apikey-Id": apikey_id,
        "X-MinT-Request-Id": f"issue324-{uuid.uuid4().hex}",
        "X-Internal-Token": env.internal_api_token,
    }


def _payload() -> dict[str, Any]:
    req = SampleRequest(
        sampling_session_id="missing-session",
        num_samples=1,
        prompt=ModelInput.from_ints([1, 2, 3]),
        sampling_params=SamplingParams(max_tokens=4),
    )
    return req.model_dump(mode="json")


def _post_asample(env: Env, *, user_id: str, apikey_id: str) -> tuple[int, Any]:
    try:
        r = requests.post(
            env.base_url + "/api/v1/asample",
            json=_payload(),
            headers=_headers(user_id=user_id, apikey_id=apikey_id, env=env),
            timeout=env.timeout_s,
        )
    except Exception as e:
        return -1, f"{type(e).__name__}: {e}"
    try:
        body = r.json()
    except Exception:
        body = r.text
    return int(r.status_code), body


def _debug_state(env: Env) -> dict[str, Any]:
    r = requests.get(env.base_url + "/internal/model_work_scheduler/debug_state", timeout=env.timeout_s)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise TypeError(f"debug_state returned non-dict: {type(data)}")
    return data


def main() -> int:
    env = _env()
    health = requests.get(env.base_url + "/api/v1/healthz", timeout=env.timeout_s)
    if health.status_code != 200:
        return _fail(f"healthz status={health.status_code} body={health.text[:200]!r}")

    user_id = "aaaaaaaaaaaaaaaaaaaaaaaa"
    key_a = "bbbbbbbbbbbbbbbbbbbbbbbb"
    key_b = "cccccccccccccccccccccccc"

    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=env.flood_count + 4) as executor:
        futures = [
            executor.submit(_post_asample, env, user_id=user_id, apikey_id=key_a)
            for _ in range(env.flood_count)
        ]
        futures.append(executor.submit(_post_asample, env, user_id=user_id, apikey_id=key_b))
    results = [f.result() for f in futures]
    elapsed = time.time() - started

    status_counts = Counter(status for status, _body in results)
    key_a_results = results[:-1]
    key_b_status, key_b_body = results[-1]
    key_a_429 = [body for status, body in key_a_results if status == 429]
    key_a_200 = [body for status, body in key_a_results if status == 200]
    transport_errors = [body for status, body in results if status == -1]

    print(
        f"flood_count={env.flood_count} dt_s={elapsed:.2f} "
        f"status_counts={dict(sorted(status_counts.items()))}",
        flush=True,
    )
    if transport_errors:
        print(f"transport_errors={transport_errors[:3]}", flush=True)

    if not key_a_200:
        return _fail("key A never reached the queue; expected some 200 admissions before throttling")
    if not key_a_429:
        return _fail("key A never received 429 during flood; per-key throttle did not apply")
    if key_b_status != 200:
        return _fail(f"key B should remain admissible while key A is throttled, got status={key_b_status} body={key_b_body!r}")

    first_429 = key_a_429[0]
    if not isinstance(first_429, dict):
        return _fail(f"429 body is not JSON: {first_429!r}")
    detail = first_429.get("detail") if isinstance(first_429.get("detail"), dict) else first_429
    assert detail is not None
    if detail.get("code") != "sampling_principal_backpressure":
        return _fail(f"unexpected 429 code: {first_429!r}")
    assert detail is not None
    if detail.get("scope") != "api_key":
        return _fail(f"unexpected 429 scope: {first_429!r}")

    try:
        debug_state = _debug_state(env)
        queue_stats = debug_state.get("stats") if isinstance(debug_state, dict) else None
        if isinstance(queue_stats, dict):
            print(f"queue_stats.by_apikey_id={queue_stats.get('by_apikey_id')}", flush=True)
    except Exception as e:
        print(f"debug_state_unavailable={type(e).__name__}: {e}", flush=True)

    print(
        "PASS: key A was throttled independently and key B stayed admissible",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
