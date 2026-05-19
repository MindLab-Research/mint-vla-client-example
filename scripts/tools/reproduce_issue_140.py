import os
import sys
import time
import uuid
from typing import Any

import requests

BASE_URL = os.environ.get("MINT_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("MINT_API_KEY", "dummy")

BASE_MODEL = os.environ.get("MINT_MODEL", "Qwen/Qwen3-0.6B")
IDLE_S = float(os.environ.get("MINT_IDLE_S", "360"))
POLL_TIMEOUT_S = float(os.environ.get("MINT_POLL_TIMEOUT_S", "600"))


def _headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY} if API_KEY else {}


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _get_json(url: str, *, timeout_s: float) -> dict[str, Any]:
    resp = requests.get(url, headers=_headers(), timeout=timeout_s)
    if resp.status_code != 200:
        raise RuntimeError(f"GET {url} returned {resp.status_code}: {resp.text[:400]!r}")
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"GET {url} returned non-dict json: {type(data)}")
    return data


def _post_json(url: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    resp = requests.post(url, headers=_headers(), json=payload, timeout=timeout_s)
    if resp.status_code != 200:
        raise RuntimeError(f"POST {url} returned {resp.status_code}: {resp.text[:400]!r}")
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


def _poll_future(request_id: str) -> dict[str, Any]:
    url = f"{BASE_URL}/api/v1/retrieve_future"
    deadline = time.time() + POLL_TIMEOUT_S
    while time.time() < deadline:
        resp = requests.post(url, headers=_headers(), json={"request_id": request_id}, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if not isinstance(data, dict):
                raise RuntimeError(f"retrieve_future returned non-dict json: {type(data)}")
            return data
        if resp.status_code == 408:
            time.sleep(2)
            continue
        raise RuntimeError(f"POST {url} returned {resp.status_code}: {resp.text[:400]!r}")
    raise TimeoutError(f"retrieve_future timed out after {POLL_TIMEOUT_S}s (request_id={request_id})")


def main() -> int:
    try:
        caps = _get_json(f"{BASE_URL}/api/v1/get_server_capabilities", timeout_s=30.0)
        _require_model_in_caps(caps, BASE_MODEL)

        session = _post_json(
            f"{BASE_URL}/api/v1/create_session",
            {
                "tags": ["scripts/tools/reproduce_issue_140.py", f"repro-140-{uuid.uuid4().hex[:8]}"],
                "user_metadata": {},
                "sdk_version": "scripts/tools/reproduce_issue_140.py",
            },
            timeout_s=30.0,
        )
        session_id = session.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            return _fail(f"create_session missing session_id: {session!r}")

        sampling = _post_json(
            f"{BASE_URL}/api/v1/create_sampling_session",
            {"session_id": session_id, "sampling_session_seq_id": 0, "base_model": BASE_MODEL},
            timeout_s=90.0,
        )
        sampling_session_id = sampling.get("sampling_session_id")
        if not isinstance(sampling_session_id, str) or not sampling_session_id:
            return _fail(f"create_sampling_session missing sampling_session_id: {sampling!r}")

        prompt_tokens = [1, 1, 1]
        fut1 = _post_json(
            f"{BASE_URL}/api/v1/asample",
            {
                "sampling_session_id": sampling_session_id,
                "seq_id": 0,
                "num_samples": 1,
                "prompt": {"chunks": [{"tokens": prompt_tokens, "type": "encoded_text"}]},
                "sampling_params": {"max_tokens": 1, "temperature": 0.0, "top_k": -1, "top_p": 1.0},
            },
            timeout_s=60.0,
        )
        request_id_1 = fut1.get("request_id")
        if not isinstance(request_id_1, str) or not request_id_1:
            return _fail(f"asample(1) missing request_id: {fut1!r}")
        res1 = _poll_future(request_id_1)
        if "error" in res1:
            return _fail(f"retrieve_future(1) returned error: {res1.get('error')!r}")

        time.sleep(IDLE_S)

        fut2 = _post_json(
            f"{BASE_URL}/api/v1/asample",
            {
                "sampling_session_id": sampling_session_id,
                "seq_id": 1,
                "num_samples": 1,
                "prompt": {"chunks": [{"tokens": prompt_tokens, "type": "encoded_text"}]},
                "sampling_params": {"max_tokens": 1, "temperature": 0.0, "top_k": -1, "top_p": 1.0},
            },
            timeout_s=60.0,
        )
        request_id_2 = fut2.get("request_id")
        if not isinstance(request_id_2, str) or not request_id_2:
            return _fail(f"asample(2) missing request_id: {fut2!r}")
        res2 = _poll_future(request_id_2)
        if "error" in res2:
            return _fail(f"retrieve_future(2) returned error: {res2.get('error')!r}")

        print("PASS")
        return 0
    except Exception as e:
        return _fail(str(e))


if __name__ == "__main__":
    raise SystemExit(main())
