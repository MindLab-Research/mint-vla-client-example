import os
import sys
import time
import uuid
from typing import Any

import requests


BASE_URL = os.environ.get("TINKER_BASE_URL")
if not BASE_URL:
    port = os.environ.get("TINKER_PORT", "8000")
    BASE_URL = f"http://localhost:{port}"
BASE_URL = BASE_URL.rstrip("/")

API_KEY = os.environ.get("TINKER_API_KEY", "dummy")

MODEL = os.environ.get("TINKER_MODEL", "Qwen/Qwen3-0.6B")
POLL_TIMEOUT_S = float(os.environ.get("TINKER_POLL_TIMEOUT_S", "1800"))
POLL_SLEEP_S = float(os.environ.get("TINKER_POLL_SLEEP_S", "2.0"))


def _headers() -> dict[str, str]:
    if not API_KEY:
        return {}
    return {"X-API-Key": API_KEY}


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _model_to_vllm_actor_name(model_name: str) -> str:
    if "/" in model_name:
        model_part = model_name.split("/")[-1]
    else:
        model_part = model_name
    safe_name = model_part.lower().replace(" ", "_")
    return f"tinker_vllm_{safe_name}"


def _get(path: str, *, timeout_s: float, expect_status: int = 200) -> requests.Response:
    url = f"{BASE_URL}{path}"
    resp = requests.get(url, headers=_headers(), timeout=timeout_s)
    if resp.status_code != expect_status:
        raise RuntimeError(f"GET {path} -> {resp.status_code} (expected {expect_status}): {resp.text[:500]!r}")
    return resp


def _get_json(path: str, *, timeout_s: float, expect_status: int = 200) -> dict[str, Any]:
    resp = _get(path, timeout_s=timeout_s, expect_status=expect_status)
    if expect_status != 200:
        return {}
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"GET {path} returned non-dict json: {type(data)}")
    return data


def _post(path: str, payload: dict[str, Any], *, timeout_s: float, expect_status: int = 200) -> requests.Response:
    url = f"{BASE_URL}{path}"
    resp = requests.post(url, headers=_headers(), json=payload, timeout=timeout_s)
    if resp.status_code != expect_status:
        raise RuntimeError(f"POST {path} -> {resp.status_code} (expected {expect_status}): {resp.text[:500]!r}")
    return resp


def _post_json(path: str, payload: dict[str, Any], *, timeout_s: float, expect_status: int = 200) -> dict[str, Any]:
    resp = _post(path, payload, timeout_s=timeout_s, expect_status=expect_status)
    if expect_status != 200:
        return {}
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"POST {path} returned non-dict json: {type(data)}")
    return data


def _poll_future(request_id: str) -> dict[str, Any]:
    deadline = time.time() + POLL_TIMEOUT_S
    while time.time() < deadline:
        resp = requests.post(
            f"{BASE_URL}/api/v1/retrieve_future",
            headers=_headers(),
            json={"request_id": request_id},
            timeout=30.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            if not isinstance(data, dict):
                raise RuntimeError(f"retrieve_future returned non-dict json: {type(data)}")
            return data
        if resp.status_code == 408:
            time.sleep(POLL_SLEEP_S)
            continue
        raise RuntimeError(
            f"POST /api/v1/retrieve_future -> {resp.status_code}: {resp.text[:500]!r}"
        )
    raise TimeoutError(f"retrieve_future timed out after {POLL_TIMEOUT_S:.1f}s request_id={request_id}")


def main() -> int:
    try:
        _get_json("/api/v1/healthz", timeout_s=10.0)

        session = _post_json(
            "/api/v1/create_session",
            {
                "tags": ["scripts/tools/reproduce_issue_39.py", f"repro-39-{uuid.uuid4().hex[:8]}"],
                "user_metadata": {},
                "sdk_version": "scripts/tools/reproduce_issue_39.py",
            },
            timeout_s=30.0,
        )
        session_id = session.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            return _fail(f"create_session missing session_id: {session!r}")

        sampling = _post_json(
            "/api/v1/create_sampling_session",
            {"session_id": session_id, "sampling_session_seq_id": 0, "base_model": MODEL},
            timeout_s=90.0,
        )
        sampling_session_id = sampling.get("sampling_session_id")
        if not isinstance(sampling_session_id, str) or not sampling_session_id:
            return _fail(f"create_sampling_session missing sampling_session_id: {sampling!r}")

        fut = _post_json(
            "/api/v1/asample",
            {
                "sampling_session_id": sampling_session_id,
                "seq_id": 0,
                "num_samples": 1,
                "prompt": {"chunks": [{"tokens": [1, 1, 1], "type": "encoded_text"}]},
                "sampling_params": {"max_tokens": 1, "temperature": 0.0, "top_k": -1, "top_p": 1.0},
            },
            timeout_s=60.0,
        )
        request_id = fut.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            return _fail(f"asample missing request_id: {fut!r}")

        out = _poll_future(request_id)
        if "error" in out:
            return _fail(f"retrieve_future error: {out.get('error')!r}")

        # New unified actor endpoints must exist and use ModelActorRegistry schema.
        vllm = _get_json("/api/v1/actors?type=vllm", timeout_s=30.0)
        actors = vllm.get("actors")
        if not isinstance(actors, list):
            return _fail(f"/api/v1/actors returned invalid actors: {actors!r}")

        expected_actor_name = _model_to_vllm_actor_name(MODEL)
        if not any(isinstance(a, dict) and a.get("actor_name") == expected_actor_name for a in actors):
            return _fail(f"expected vLLM actor not present in /api/v1/actors: {expected_actor_name!r}")

        for t in ["megatron", "dense"]:
            d = _get_json(f"/api/v1/actors?type={t}", timeout_s=30.0)
            if not isinstance(d.get("actors"), list):
                return _fail(f"/api/v1/actors?type={t} returned invalid actors: {d!r}")

        killed = _post_json(
            "/api/v1/actors/kill",
            {"actor_type": "vllm", "model_name": MODEL},
            timeout_s=30.0,
        )
        if not isinstance(killed.get("killed"), int):
            return _fail(f"/api/v1/actors/kill missing killed count: {killed!r}")

        print("PASS")
        return 0
    except Exception as e:
        return _fail(str(e))


if __name__ == "__main__":
    raise SystemExit(main())
