import os
import sys
import time
from typing import Any

import requests

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")


def _headers() -> dict[str, str]:
    if API_KEY:
        return {"X-API-Key": API_KEY}
    return {}


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _get_json(url: str, timeout: float = 30.0) -> dict[str, Any]:
    resp = requests.get(url, headers=_headers(), timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"GET {url} returned {resp.status_code}: {resp.text[:200]!r}")
    return resp.json()


def _post_json(url: str, payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    resp = requests.post(url, headers=_headers(), json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"POST {url} returned {resp.status_code}: {resp.text[:200]!r}")
    return resp.json()


def _poll_future(request_id: str, timeout_s: float = 600.0) -> dict[str, Any]:
    url = f"{BASE_URL}/api/v1/retrieve_future"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = requests.post(url, headers=_headers(), json={"request_id": request_id}, timeout=30)
        if resp.status_code == 408:
            time.sleep(1)
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"POST {url} returned {resp.status_code}: {resp.text[:200]!r}")
        data = resp.json()
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(str(data["error"]))
        return data
    raise RuntimeError(f"retrieve_future timed out after {timeout_s}s (request_id={request_id})")


def _pick_training_model() -> str:
    caps = _get_json(f"{BASE_URL}/api/v1/get_server_capabilities", timeout=30.0)
    models = caps.get("supported_models")
    if not isinstance(models, list) or not models:
        raise RuntimeError(f"supported_models missing/empty: {models!r}")

    names: list[str] = []
    for entry in models:
        if isinstance(entry, dict) and isinstance(entry.get("model_name"), str):
            names.append(entry["model_name"])

    # Issue #81 is a Megatron-path bug (MoE models); force a known MoE model if available.
    for prefer in (
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
    ):
        if prefer in names:
            return prefer

    for name in names:
        if "A3B" in name or "A22B" in name:
            return name

    raise RuntimeError(
        "No MoE (Megatron) model found in supported_models; cannot exercise the Megatron multi-chunk path."
    )


def main() -> int:
    model_id: str | None = None
    try:
        base_model = _pick_training_model()
        session = _post_json(
            f"{BASE_URL}/api/v1/create_session",
            {
                "tags": ["scripts/tools/reproduce_issue_81.py"],
                "user_metadata": {},
                "sdk_version": "scripts/tools/reproduce_issue_81.py",
            },
        )
        session_id = session.get("session_id")
        if not session_id:
            return _fail(f"create_session missing session_id: {session!r}")

        fut = _post_json(
            f"{BASE_URL}/api/v1/create_model",
            {
                "session_id": session_id,
                "model_seq_id": 0,
                "base_model": base_model,
                "lora_config": {"rank": 4},
            },
            timeout=60.0,
        )
        request_id = fut.get("request_id")
        if not request_id:
            return _fail(f"create_model missing request_id: {fut!r}")

        created = _poll_future(request_id, timeout_s=1200.0)
        model_id = created.get("model_id")
        if not model_id:
            return _fail(f"create_model future missing model_id: {created!r}")

        # Multi-chunk ModelInput: tokens are split across multiple chunks, but loss fields
        # are sized to the full concatenated sequence.
        data_item = {
            "model_input": {
                "chunks": [
                    {"type": "encoded_text", "tokens": [1]},
                    {"type": "encoded_text", "tokens": [2, 3]},
                ]
            },
            "loss_fn_inputs": {
                "target_tokens": {"data": [1, 2, 3]},
                "weights": {"data": [1.0, 1.0, 1.0]},
            },
        }

        forward_fut = _post_json(
            f"{BASE_URL}/api/v1/forward",
            {
                "model_id": model_id,
                "forward_input": {
                    "data": [data_item],
                    "loss_fn": "cross_entropy",
                },
                "seq_id": 0,
            },
            timeout=60.0,
        )
        forward_request_id = forward_fut.get("request_id")
        if not forward_request_id:
            return _fail(f"forward missing request_id: {forward_fut!r}")

        _poll_future(forward_request_id, timeout_s=600.0)
    except Exception as e:
        return _fail(str(e))
    finally:
        if model_id:
            try:
                requests.delete(
                    f"{BASE_URL}/api/v1/models/{model_id}",
                    headers=_headers(),
                    timeout=60,
                )
            except Exception:
                pass

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
