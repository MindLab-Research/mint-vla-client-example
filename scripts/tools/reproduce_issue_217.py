import os
import sys
import time
import uuid
from typing import Any

import requests


BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")

MODEL = os.environ.get("TINKER_MODEL") or "Qwen/Qwen3-30B-A3B-Instruct-2507"
POLL_DELAY_S = float(os.environ.get("TINKER_POLL_DELAY_S", "0.2"))


def _headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY} if API_KEY else {}


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _get_json(url: str, timeout_s: float = 30.0) -> dict[str, Any]:
    resp = requests.get(url, headers=_headers(), timeout=timeout_s)
    if resp.status_code != 200:
        raise RuntimeError(f"GET {url} returned {resp.status_code}: {resp.text[:200]!r}")
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"GET {url} returned non-dict json: {type(data)}")
    return data


def _post(url: str, payload: dict[str, Any], timeout_s: float = 60.0) -> requests.Response:
    return requests.post(url, headers=_headers(), json=payload, timeout=timeout_s)


def _post_json(url: str, payload: dict[str, Any], timeout_s: float = 60.0) -> dict[str, Any]:
    resp = _post(url, payload, timeout_s=timeout_s)
    if resp.status_code != 200:
        raise RuntimeError(f"POST {url} returned {resp.status_code}: {resp.text[:200]!r}")
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"POST {url} returned non-dict json: {type(data)}")
    return data


def _require_model_in_caps(caps: dict[str, Any], model_name: str) -> None:
    models = caps.get("supported_models")
    if not isinstance(models, list):
        raise RuntimeError(f"supported_models missing/invalid: {models!r}")
    for m in models:
        if isinstance(m, dict) and m.get("model_name") == model_name:
            return
    raise RuntimeError(f"model {model_name!r} not present in supported_models")


def main() -> int:
    try:
        caps = _get_json(f"{BASE_URL}/api/v1/get_server_capabilities", timeout_s=30.0)
        _require_model_in_caps(caps, MODEL)

        session = _post_json(
            f"{BASE_URL}/api/v1/create_session",
            {
                "tags": ["scripts/tools/reproduce_issue_217.py", f"repro-217-{uuid.uuid4().hex[:8]}"],
                "user_metadata": {},
                "sdk_version": "scripts/tools/reproduce_issue_217.py",
            },
            timeout_s=30.0,
        )
        session_id = session.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            return _fail(f"create_session missing session_id: {session!r}")

        sampling = _post_json(
            f"{BASE_URL}/api/v1/create_sampling_session",
            {"session_id": session_id, "sampling_session_seq_id": 0, "base_model": MODEL},
            timeout_s=600.0,
        )
        sampling_session_id = sampling.get("sampling_session_id")
        if not isinstance(sampling_session_id, str) or not sampling_session_id:
            return _fail(f"create_sampling_session missing sampling_session_id: {sampling!r}")

        fut = _post_json(
            f"{BASE_URL}/api/v1/asample",
            {
                "sampling_session_id": sampling_session_id,
                "seq_id": 0,
                "num_samples": 1,
                "prompt": {"chunks": [{"tokens": [1, 1, 1], "type": "encoded_text"}]},
                "sampling_params": {"max_tokens": 64, "temperature": 0.0, "top_k": 1, "top_p": 1.0},
            },
            timeout_s=60.0,
        )
        request_id = fut.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            return _fail(f"asample missing request_id: {fut!r}")
        if not request_id.startswith("gw:"):
            return _fail(f"asample request_id not gateway-encoded: {request_id!r}")

        time.sleep(POLL_DELAY_S)
        poll_r = _post(f"{BASE_URL}/api/v1/retrieve_future", {"request_id": request_id}, timeout_s=30.0)
        if poll_r.status_code == 404:
            return _fail(f"retrieve_future returned 404 for gateway request_id={request_id!r}: {poll_r.text[:200]!r}")
        if poll_r.status_code != 503:
            return _fail(
                f"expected retrieve_future 503 (lost future diagnostic), got {poll_r.status_code}: {poll_r.text[:200]!r}"
            )

        payload = poll_r.json()
        if not isinstance(payload, dict) or "detail" not in payload:
            return _fail(f"retrieve_future 503 missing JSON detail: {payload!r}")

        print("PASS")
        return 0
    except Exception as e:
        return _fail(str(e))


if __name__ == "__main__":
    raise SystemExit(main())

