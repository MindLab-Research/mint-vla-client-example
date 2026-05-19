import os
import sys
import uuid

import requests

BASE_URL = os.environ.get("MINT_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("MINT_API_KEY", "dummy")


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _post_json(path: str, payload: dict, *, timeout_s: float) -> requests.Response:
    url = f"{BASE_URL}{path}"
    headers = {"X-API-Key": API_KEY}
    return requests.post(url, json=payload, headers=headers, timeout=timeout_s)


def _assert_unknown_request_id_404(request_id: str) -> int:
    resp = _post_json("/api/v1/retrieve_future", {"request_id": request_id}, timeout_s=30.0)
    if resp.status_code != 404:
        return _fail(f"retrieve_future({request_id!r}) expected 404, got {resp.status_code}: {resp.text}")
    try:
        body = resp.json()
    except Exception:
        return _fail(f"retrieve_future({request_id!r}) returned non-JSON 404: {resp.text}")
    detail = body.get("detail")
    if request_id not in str(detail):
        return _fail(f"retrieve_future({request_id!r}) 404 detail missing request_id: {body!r}")
    return 0


def main() -> int:
    # Regression guard: unknown request IDs should be a clean 404, not a 5xx.
    random_uuid = str(uuid.uuid4())
    rc = _assert_unknown_request_id_404(random_uuid)
    if rc != 0:
        return rc

    random_garbage = f"not-a-uuid-{uuid.uuid4().hex}"
    rc = _assert_unknown_request_id_404(random_garbage)
    if rc != 0:
        return rc

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
