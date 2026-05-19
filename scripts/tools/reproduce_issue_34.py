import os
import sys

import requests

BASE_URL = os.environ.get("MINT_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("MINT_API_KEY", "dummy")


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def main() -> int:
    url = f"{BASE_URL.rstrip('/')}/api/v1/get_server_capabilities"
    try:
        r = requests.get(url, headers={"X-API-Key": API_KEY}, timeout=10)
    except Exception as e:
        return _fail(f"GET {url} failed: {e}")

    if r.status_code != 200:
        return _fail(f"GET {url} returned {r.status_code}: {r.text[:200]!r}")

    try:
        data = r.json()
    except Exception as e:
        return _fail(f"Response is not JSON: {e}")

    models = data.get("supported_models")
    if not isinstance(models, list) or not models:
        return _fail(f"supported_models missing or empty: {models!r}")

    for i, entry in enumerate(models):
        if not isinstance(entry, dict):
            return _fail(f"supported_models[{i}] is not an object: {entry!r}")
        name = entry.get("model_name")
        if not isinstance(name, str) or not name:
            return _fail(f"supported_models[{i}].model_name missing/invalid: {name!r}")
        if "max_context_length" not in entry:
            return _fail(f"{name}: missing max_context_length in /get_server_capabilities response")
        mcl = entry["max_context_length"]
        if not isinstance(mcl, int) or mcl <= 0:
            return _fail(f"{name}: invalid max_context_length={mcl!r}")

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

