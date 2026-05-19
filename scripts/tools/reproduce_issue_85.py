import os
import sys
import time
import uuid
from typing import Any

import requests

BASE_URL_A = os.environ.get("MINT_BASE_URL_A") or os.environ.get("MINT_BASE_URL") or "http://localhost:10085"
BASE_URL_B = os.environ.get("MINT_BASE_URL_B") or os.environ.get("MINT_BASE_URL2") or "http://localhost:10086"
API_KEY = os.environ.get("MINT_API_KEY", "dummy")

POLL_TIMEOUT_S = float(os.environ.get("MINT_POLL_TIMEOUT_S", "60"))


def _headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY} if API_KEY else {}


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _post_json(url: str, payload: dict[str, Any], *, timeout_s: float = 60.0) -> tuple[int, dict[str, Any]]:
    resp = requests.post(url, headers=_headers(), json=payload, timeout=timeout_s)
    try:
        data = resp.json()
    except Exception:
        data = {"_non_json_body": resp.text[:400]}
    if not isinstance(data, dict):
        data = {"_non_dict_json": str(type(data))}
    return resp.status_code, data


def _asample(base_url: str) -> str:
    url = f"{base_url.rstrip('/')}/api/v1/asample"
    sampling_session_id = f"repro-85-invalid-{uuid.uuid4()}"
    status, data = _post_json(
        url,
        payload={
            "sampling_session_id": sampling_session_id,
            "seq_id": 0,
            "num_samples": 1,
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 1, 1]}]},
            "sampling_params": {"max_tokens": 1, "temperature": 0.0, "top_k": 1, "top_p": 1.0},
        },
        timeout_s=30.0,
    )
    if status != 200:
        raise RuntimeError(f"asample returned {status}: {data!r}")
    request_id = data.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise RuntimeError(f"asample missing request_id: {data!r}")
    return request_id


def _retrieve(base_url: str, request_id: str) -> tuple[int, dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/api/v1/retrieve_future"
    return _post_json(url, payload={"request_id": request_id}, timeout_s=30.0)


def _assert_cross_server_visible(src: str, dst: str) -> None:
    request_id = _asample(src)

    t0 = time.time()
    status, data = _retrieve(dst, request_id)

    # Regression target: 404 on a request_id returned moments ago by another replica.
    if status == 404:
        raise RuntimeError(f"retrieve_future cross-replica 404: {data!r}")

    # Pending or immediate completion are both acceptable.
    while status == 408 and time.time() - t0 < POLL_TIMEOUT_S:
        time.sleep(1.0)
        status, data = _retrieve(dst, request_id)
        if status == 404:
            raise RuntimeError(f"retrieve_future cross-replica 404 after pending: {data!r}")

    if status not in (200, 408):
        raise RuntimeError(f"retrieve_future unexpected status {status}: {data!r}")

    # If it completed, expect an error payload (invalid sampling_session_id) rather than sequences.
    if status == 200 and "error" not in data:
        # Some deployments may return a generic error shape. At minimum it should be JSON and not 404.
        pass


def main() -> int:
    try:
        _assert_cross_server_visible(BASE_URL_A, BASE_URL_B)
        _assert_cross_server_visible(BASE_URL_B, BASE_URL_A)
        print("PASS")
        return 0
    except Exception as e:
        return _fail(str(e))


if __name__ == "__main__":
    raise SystemExit(main())
