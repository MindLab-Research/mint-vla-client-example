from __future__ import annotations

import os
import sys
import time
import uuid
from typing import Any

import requests


BASE_URL = os.environ.get("TINKER_BASE_URL")
if not BASE_URL:
    port = os.environ.get("TINKER_PORT", "10207")
    BASE_URL = f"http://localhost:{port}"
BASE_URL = BASE_URL.rstrip("/")

API_KEY = os.environ.get("TINKER_API_KEY", "dummy")

BASE_MODEL = os.environ.get("TINKER_BASE_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")

# Must be a shared-filesystem path accessible to vLLM workers.
# It must contain adapter_model.safetensors (create_sampling_session validates existence).
MODEL_PATH = os.environ.get(
    "TINKER_MODEL_PATH",
    "/vePFS-Mindverse/share/tinker_checkpoints/issue_207_bad_adapter",
).strip()

POLL_TIMEOUT_S = float(os.environ.get("TINKER_POLL_TIMEOUT_S", "1800"))
POLL_SLEEP_S = float(os.environ.get("TINKER_POLL_SLEEP_S", "2.0"))

CREATE_SESSION_TIMEOUT_S = float(os.environ.get("TINKER_CREATE_SESSION_TIMEOUT_S", "30"))
CREATE_SAMPLING_TIMEOUT_S = float(os.environ.get("TINKER_CREATE_SAMPLING_TIMEOUT_S", "120"))
ASAMPLE_TIMEOUT_S = float(os.environ.get("TINKER_ASAMPLE_TIMEOUT_S", "60"))


def _headers() -> dict[str, str]:
    if not API_KEY:
        return {}
    return {"X-API-Key": API_KEY}


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _post_json(path: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    r = requests.post(f"{BASE_URL}{path}", headers=_headers(), json=payload, timeout=timeout_s)
    if r.status_code != 200:
        raise RuntimeError(f"POST {path} -> {r.status_code}: {r.text[:500]!r}")
    data = r.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"POST {path} returned non-dict json: {type(data)}")
    return data


def _get_json(path: str, *, timeout_s: float) -> dict[str, Any]:
    r = requests.get(f"{BASE_URL}{path}", headers=_headers(), timeout=timeout_s)
    if r.status_code != 200:
        raise RuntimeError(f"GET {path} -> {r.status_code}: {r.text[:500]!r}")
    data = r.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"GET {path} returned non-dict json: {type(data)}")
    return data


def _poll_future(request_id: str) -> dict[str, Any]:
    start = time.time()
    while True:
        if time.time() - start > POLL_TIMEOUT_S:
            raise TimeoutError(f"retrieve_future timed out after {POLL_TIMEOUT_S:.1f}s (request_id={request_id})")
        r = requests.post(
            f"{BASE_URL}/api/v1/retrieve_future",
            headers=_headers(),
            json={"request_id": request_id},
            timeout=30.0,
        )
        if r.status_code == 408:
            time.sleep(POLL_SLEEP_S)
            continue
        if r.status_code != 200:
            raise RuntimeError(f"retrieve_future -> {r.status_code}: {r.text[:500]!r}")
        data = r.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"retrieve_future returned non-dict json: {type(data)}")
        return data


def _require_model_in_caps(model_name: str) -> None:
    caps = _get_json("/api/v1/get_server_capabilities", timeout_s=30.0)
    models = caps.get("supported_models")
    if not isinstance(models, list):
        raise RuntimeError(f"supported_models missing/invalid: {models!r}")
    for m in models:
        if isinstance(m, dict) and m.get("model_name") == model_name:
            return
    raise RuntimeError(f"model {model_name!r} not present in supported_models (wrong server?)")


def _asample_once(*, sampling_session_id: str, seq_id: int) -> dict[str, Any]:
    prompt_tokens = [1, 1, 1]
    out = _post_json(
        "/api/v1/asample",
        payload={
            "sampling_session_id": sampling_session_id,
            "seq_id": int(seq_id),
            "num_samples": 1,
            "prompt": {"chunks": [{"tokens": prompt_tokens, "type": "encoded_text"}]},
            "sampling_params": {"max_tokens": 1, "temperature": 0.0, "top_k": -1, "top_p": 1.0},
        },
        timeout_s=ASAMPLE_TIMEOUT_S,
    )
    request_id = out.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise RuntimeError(f"asample missing request_id: {out!r}")
    return _poll_future(request_id)


def main() -> int:
    try:
        _get_json("/api/v1/healthz", timeout_s=5.0)
        _require_model_in_caps(BASE_MODEL)

        session = _post_json(
            "/api/v1/create_session",
            payload={
                "tags": ["scripts/tools/reproduce_issue_207.py", f"repro-207-{uuid.uuid4().hex[:8]}"],
                "user_metadata": {},
                "sdk_version": "scripts/tools/reproduce_issue_207.py",
            },
            timeout_s=CREATE_SESSION_TIMEOUT_S,
        )
        session_id = session.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            return _fail(f"create_session missing/invalid session_id: {session!r}")

        sampling = _post_json(
            "/api/v1/create_sampling_session",
            payload={
                "session_id": session_id,
                "sampling_session_seq_id": 0,
                "base_model": BASE_MODEL,
                "model_path": MODEL_PATH,
                "lora_rank": 32,
            },
            timeout_s=CREATE_SAMPLING_TIMEOUT_S,
        )
        sampling_session_id = sampling.get("sampling_session_id")
        if not isinstance(sampling_session_id, str) or not sampling_session_id:
            return _fail(f"create_sampling_session missing/invalid sampling_session_id: {sampling!r}")

        res1 = _asample_once(sampling_session_id=sampling_session_id, seq_id=0)
        err1 = res1.get("error")
        if not err1:
            return _fail(
                "asample(1) unexpectedly succeeded; this repro requires a broken adapter "
                f"(model_path={MODEL_PATH!r}) to force add_lora failure."
            )
        if isinstance(err1, str) and "already has lora_int_id" in err1:
            return _fail(f"asample(1) unexpectedly hit lora_int_id conflict: {err1[:400]!r}")

        res2 = _asample_once(sampling_session_id=sampling_session_id, seq_id=1)
        err2 = res2.get("error")
        if not err2:
            return _fail("asample(2) unexpectedly succeeded; expected another add_lora failure.")
        if isinstance(err2, str) and "already has lora_int_id" in err2:
            return _fail(f"asample(2) hit lora_int_id conflict: {err2[:400]!r}")

        print("PASS")
        return 0
    except Exception as e:
        return _fail(str(e))


if __name__ == "__main__":
    raise SystemExit(main())

