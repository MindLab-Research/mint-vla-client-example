import os
import sys

import requests


BASE_URL = os.environ.get("MINT_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("MINT_API_KEY", "dummy")

ISSUE_NUMBER = 225


def _headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY} if API_KEY else {}


def _fail(msg: str) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return 1


def main() -> int:
    print(f"issue={ISSUE_NUMBER} base_url={BASE_URL}")

    payload = {
        # Some deployments require this field by schema even though the server currently ignores it.
        "session_id": f"repro_issue_{ISSUE_NUMBER}",
        # This is not a valid model name or HF cache path, but it contains a substring that currently
        # triggers silent model identification fallback (qwen--qwen3-0.6b -> Qwen/Qwen3-0.6B).
        "base_model": "xxx qwen--qwen3-0.6b yyy",
    }
    r = requests.post(
        f"{BASE_URL}/api/v1/create_sampling_session",
        headers=_headers(),
        json=payload,
        timeout=30.0,
    )

    if r.status_code == 200:
        return _fail(
            "BUG: create_sampling_session accepted an invalid base_model via substring fallback. "
            f"status={r.status_code} body={r.text!r}"
        )

    if r.status_code == 400:
        print(f"PASS: rejected invalid base_model. status={r.status_code} body={r.text!r}")
        return 0

    return _fail(f"Unexpected status={r.status_code} body={r.text!r}")


if __name__ == "__main__":
    raise SystemExit(main())

