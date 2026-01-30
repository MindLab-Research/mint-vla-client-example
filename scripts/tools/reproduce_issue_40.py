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
LORA_RANK = int(os.environ.get("TINKER_LORA_RANK", "8"))
POLL_TIMEOUT_S = float(os.environ.get("TINKER_POLL_TIMEOUT_S", "1800"))
POLL_SLEEP_S = float(os.environ.get("TINKER_POLL_SLEEP_S", "2.0"))


def _headers() -> dict[str, str]:
    if not API_KEY:
        return {}
    return {"X-API-Key": API_KEY}


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _get_json(path: str, *, timeout_s: float, expect_status: int = 200) -> dict[str, Any]:
    url = f"{BASE_URL}{path}"
    resp = requests.get(url, headers=_headers(), timeout=timeout_s)
    if resp.status_code != expect_status:
        raise RuntimeError(f"GET {path} -> {resp.status_code} (expected {expect_status}): {resp.text[:500]!r}")
    if expect_status != 200:
        return {}
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"GET {path} returned non-dict json: {type(data)}")
    return data


def _post_json(path: str, payload: dict[str, Any], *, timeout_s: float, expect_status: int = 200) -> dict[str, Any]:
    url = f"{BASE_URL}{path}"
    resp = requests.post(url, headers=_headers(), json=payload, timeout=timeout_s)
    if resp.status_code != expect_status:
        raise RuntimeError(f"POST {path} -> {resp.status_code} (expected {expect_status}): {resp.text[:500]!r}")
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
        raise RuntimeError(f"POST /api/v1/retrieve_future -> {resp.status_code}: {resp.text[:500]!r}")
    raise TimeoutError(f"retrieve_future timed out after {POLL_TIMEOUT_S:.1f}s request_id={request_id}")


def main() -> int:
    try:
        _get_json("/api/v1/healthz", timeout_s=10.0)

        req = _post_json(
            "/api/v1/create_model",
            {
                "session_id": f"repro-40-{uuid.uuid4().hex[:8]}",
                "model_seq_id": 0,
                "base_model": MODEL,
                "user_metadata": {"repro": "scripts/tools/reproduce_issue_40.py"},
                "lora_config": {"rank": LORA_RANK},
                "type": "create_model",
            },
            timeout_s=30.0,
        )
        request_id = req.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            return _fail(f"create_model missing request_id: {req!r}")

        out = _poll_future(request_id)
        if "error" in out:
            return _fail(f"retrieve_future error: {out.get('error')!r}")

        model_id = out.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            return _fail(f"create_model missing model_id: {out!r}")

        backend = out.get("backend")
        if backend != "peft":
            return _fail(f"expected dense training backend 'peft', got {backend!r} (base_model={MODEL!r})")

        dense = _get_json("/api/v1/actors?type=dense", timeout_s=30.0)
        actors = dense.get("actors")
        if not isinstance(actors, list):
            return _fail(f"/api/v1/actors?type=dense returned invalid actors: {dense!r}")

        match = None
        for a in actors:
            if isinstance(a, dict) and a.get("current_session") == model_id:
                match = a
                break
        if match is None:
            return _fail(f"expected dense trainer with current_session={model_id!r} not found in /api/v1/actors")

        metadata = match.get("metadata")
        if not isinstance(metadata, dict):
            return _fail(f"dense actor missing metadata dict: {match!r}")
        if not isinstance(metadata.get("max_lora_rank"), int):
            return _fail(f"dense actor metadata missing int max_lora_rank: {metadata!r}")
        if metadata.get("actual_rank") != LORA_RANK:
            return _fail(f"dense actor metadata actual_rank mismatch: got {metadata.get('actual_rank')!r} want {LORA_RANK!r}")

        # Best-effort cleanup: delete the training model to release GPUs.
        try:
            resp = requests.delete(
                f"{BASE_URL}/api/v1/models/{model_id}",
                headers=_headers(),
                timeout=60.0,
            )
            if resp.status_code not in (200, 404):
                return _fail(f"DELETE /api/v1/models/{model_id} -> {resp.status_code}: {resp.text[:500]!r}")
        except Exception:
            pass

        print("PASS")
        return 0
    except Exception as e:
        return _fail(str(e))


if __name__ == "__main__":
    raise SystemExit(main())

