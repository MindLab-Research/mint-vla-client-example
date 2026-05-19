import os
import sys
import uuid
from typing import Any

import requests

BASE_URL = os.environ.get("MINT_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("MINT_API_KEY", "dummy")

SUPPORTED_MODEL = os.environ.get("MINT_SUPPORTED_MODEL", "Qwen/Qwen3-0.6B")
UNSUPPORTED_MODEL = os.environ.get("MINT_UNSUPPORTED_MODEL", "Qwen/Qwen3-4B-Instruct-2507")

EXPECT_ALLOW_UNSUPPORTED = os.environ.get("MINT_EXPECT_ALLOW_UNSUPPORTED_MODELS", "0").strip() in ("1", "true", "yes", "on")


def _headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY} if API_KEY else {}


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _post(url: str, payload: dict[str, Any]) -> requests.Response:
    return requests.post(url, headers=_headers(), json=payload, timeout=30)


def _expect_status(resp: requests.Response, expected: int, *, ctx: str) -> None:
    if resp.status_code != expected:
        raise RuntimeError(f"{ctx}: expected {expected}, got {resp.status_code}: {resp.text[:400]!r}")


def _expect_json_key(resp: requests.Response, key: str, *, ctx: str) -> str:
    try:
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"{ctx}: expected json, got decode err={e!r} body={resp.text[:400]!r}") from e
    if not isinstance(data, dict):
        raise RuntimeError(f"{ctx}: expected dict json, got {type(data)}: {data!r}")
    val = data.get(key)
    if not isinstance(val, str) or not val:
        raise RuntimeError(f"{ctx}: missing/invalid {key}: {data!r}")
    return val


def _check_sampling(base_model: str, *, expect_ok: bool) -> None:
    session_id = f"repro-163-{uuid.uuid4().hex[:8]}"
    resp = _post(
        f"{BASE_URL}/api/v1/create_sampling_session",
        {
            "session_id": session_id,
            "base_model": base_model,
        },
    )
    if expect_ok:
        _expect_status(resp, 200, ctx=f"create_sampling_session base_model={base_model!r}")
        _expect_json_key(resp, "sampling_session_id", ctx="create_sampling_session response")
    else:
        _expect_status(resp, 400, ctx=f"create_sampling_session base_model={base_model!r}")


def _check_create_model(base_model: str, *, expect_ok: bool) -> None:
    session_id = f"repro-163-{uuid.uuid4().hex[:8]}"
    resp = _post(
        f"{BASE_URL}/api/v1/create_model",
        {
            "session_id": session_id,
            "model_seq_id": 0,
            "base_model": base_model,
            "lora_config": {"rank": 8},
        },
    )
    if expect_ok:
        _expect_status(resp, 200, ctx=f"create_model base_model={base_model!r}")
        _expect_json_key(resp, "request_id", ctx="create_model response")
    else:
        _expect_status(resp, 400, ctx=f"create_model base_model={base_model!r}")


def _check_create_model_from_state(base_model: str, *, expect_ok: bool) -> None:
    session_id = f"repro-163-{uuid.uuid4().hex[:8]}"
    resp = _post(
        f"{BASE_URL}/api/v1/create_model_from_state",
        {
            "session_id": session_id,
            "model_seq_id": 0,
            "base_model": base_model,
            "state_path": "mint://ckpt_nonexistent",
            "lora_config": {"rank": 8},
        },
    )
    if expect_ok:
        _expect_status(resp, 200, ctx=f"create_model_from_state base_model={base_model!r}")
        _expect_json_key(resp, "request_id", ctx="create_model_from_state response")
    else:
        _expect_status(resp, 400, ctx=f"create_model_from_state base_model={base_model!r}")


def main() -> int:
    try:
        _check_sampling(SUPPORTED_MODEL, expect_ok=True)

        _check_sampling(UNSUPPORTED_MODEL, expect_ok=EXPECT_ALLOW_UNSUPPORTED)
        _check_create_model(UNSUPPORTED_MODEL, expect_ok=EXPECT_ALLOW_UNSUPPORTED)
        _check_create_model_from_state(UNSUPPORTED_MODEL, expect_ok=EXPECT_ALLOW_UNSUPPORTED)

        print("PASS")
        return 0
    except Exception as e:
        return _fail(str(e))


if __name__ == "__main__":
    raise SystemExit(main())
