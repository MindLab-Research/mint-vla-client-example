import os
import sys
import uuid
from typing import Any

import requests

BASE_URL = os.environ.get("MINT_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("MINT_API_KEY", "dummy")


def _headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY} if API_KEY else {}


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _get_json(url: str, *, timeout_s: float = 30.0) -> dict[str, Any]:
    resp = requests.get(url, headers=_headers(), timeout=timeout_s)
    if resp.status_code != 200:
        raise RuntimeError(f"GET {url} returned {resp.status_code}: {resp.text[:200]!r}")
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"GET {url} returned non-dict json: {type(data)}")
    return data


def _post_json(url: str, payload: dict[str, Any], *, timeout_s: float = 60.0) -> dict[str, Any]:
    resp = requests.post(url, headers=_headers(), json=payload, timeout=timeout_s)
    if resp.status_code != 200:
        raise RuntimeError(f"POST {url} returned {resp.status_code}: {resp.text[:200]!r}")
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"POST {url} returned non-dict json: {type(data)}")
    return data


def _make_prompt_tokens(n: int) -> list[int]:
    # Deterministic long prompt to avoid request completing before first poll.
    base = [1, 2, 3, 4]
    out = base * (n // len(base)) + base[: n % len(base)]
    return out


def main() -> int:
    try:
        caps = _get_json(f"{BASE_URL}/api/v1/get_server_capabilities", timeout_s=30.0)
        supported = caps.get("supported_models") or []
        names = [m.get("model_name") for m in supported if isinstance(m, dict)]
        base_model = os.environ.get("MINT_BASE_MODEL") or "Qwen/Qwen3-0.6B"
        if base_model not in names and names:
            base_model = str(names[0])

        sess = _post_json(
            f"{BASE_URL}/api/v1/create_session",
            {
                "tags": ["scripts/tools/reproduce_issue_24.py"],
                "user_metadata": {},
                "sdk_version": "scripts/tools/reproduce_issue_24.py",
            },
            timeout_s=30.0,
        )
        session_id = sess.get("session_id")
        if not session_id:
            return _fail(f"create_session missing session_id: {sess!r}")

        sampling = _post_json(
            f"{BASE_URL}/api/v1/create_sampling_session",
            {"session_id": session_id, "sampling_session_seq_id": 0, "base_model": base_model},
            timeout_s=120.0,
        )
        sampling_session_id = sampling.get("sampling_session_id")
        if not sampling_session_id:
            return _fail(f"create_sampling_session missing sampling_session_id: {sampling!r}")

        for prompt_len in (4096, 8192, 16384):
            for max_tokens in (1, 16, 64):
                seq_id = uuid.uuid4().int % 1000000000
                fut = _post_json(
                    f"{BASE_URL}/api/v1/asample",
                    {
                        "sampling_session_id": sampling_session_id,
                        "seq_id": seq_id,
                        "num_samples": 1,
                        "prompt": {"chunks": [{"tokens": _make_prompt_tokens(prompt_len), "type": "encoded_text"}]},
                        "sampling_params": {
                            "max_tokens": max_tokens,
                            "temperature": 0.0,
                            "top_k": -1,
                            "top_p": 1.0,
                        },
                    },
                    timeout_s=30.0,
                )
                request_id = fut.get("request_id")
                if not request_id:
                    return _fail(f"asample missing request_id: {fut!r}")

                resp = requests.post(
                    f"{BASE_URL}/api/v1/retrieve_future",
                    headers=_headers(),
                    json={"request_id": request_id},
                    timeout=30,
                )
                if resp.status_code == 200:
                    # Completed before first poll; make the request heavier and try again.
                    continue
                if resp.status_code != 408:
                    return _fail(f"retrieve_future status_code={resp.status_code} expected 408: {resp.text[:200]!r}")

                ra = resp.headers.get("Retry-After")
                try:
                    ra_i = int(ra)
                except Exception:
                    return _fail(f"Retry-After={ra!r} expected int header (headers={dict(resp.headers)!r})")
                if ra_i < 1:
                    return _fail(f"Retry-After={ra_i!r} expected >= 1 (headers={dict(resp.headers)!r})")

                body = resp.json()
                if not (isinstance(body, dict) and body.get("queue_state") == "active"):
                    return _fail(f"body={body!r} expected queue_state='active'")
                if "retry_after_s" in body and body.get("retry_after_s") != ra_i:
                    return _fail(f"retry_after_s={body.get('retry_after_s')!r} expected {ra_i}")

                print("PASS")
                return 0

        return _fail("retrieve_future returned 200 on first poll for all attempts (cannot observe 408 headers)")
    except Exception as e:
        return _fail(str(e))


if __name__ == "__main__":
    raise SystemExit(main())
