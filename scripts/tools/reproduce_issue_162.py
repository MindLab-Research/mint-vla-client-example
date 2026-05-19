import os
import sys
from typing import Any

import requests

BASE_URL = os.environ.get("MINT_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("MINT_API_KEY", "dummy")


def _headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY} if API_KEY else {}


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _get_json(url: str) -> dict[str, Any]:
    resp = requests.get(url, headers=_headers(), timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"GET {url} returned {resp.status_code}: {resp.text[:200]!r}")
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"GET {url} returned non-dict json: {type(data)}")
    return data


def _expect_status(url: str, expected: int) -> None:
    resp = requests.get(url, headers=_headers(), timeout=10)
    if resp.status_code != expected:
        raise RuntimeError(f"GET {url} expected {expected}, got {resp.status_code}: {resp.text[:200]!r}")


def main() -> int:
    try:
        health = _get_json(f"{BASE_URL}/api/v1/healthz")
        if health.get("status") != "ready":
            return _fail(f"healthz unexpected payload: {health!r}")

        root = _get_json(f"{BASE_URL}/")
        if root.get("status") != "ready" or root.get("healthz") != "/api/v1/healthz":
            return _fail(f"root unexpected payload: {root!r}")

        for path in ("/doc", "/doc/", "/docs", "/docs/"):
            _expect_status(f"{BASE_URL}{path}", 404)

        print("PASS")
        return 0
    except Exception as e:
        return _fail(str(e))


if __name__ == "__main__":
    raise SystemExit(main())
