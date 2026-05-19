import os
import sys
import time
from typing import Any

import requests


BASE_URL = os.environ.get("MINT_BASE_URL")
if not BASE_URL:
    port = os.environ.get("MINT_PORT", "8000")
    BASE_URL = f"http://localhost:{port}"
BASE_URL = BASE_URL.rstrip("/")

API_KEY = os.environ.get("MINT_API_KEY", "dummy")

MODEL = os.environ.get("MINT_MODEL", "moonshotai/Moonlight-16B-A3B-Instruct")
LORA_RANK = int(os.environ.get("MINT_LORA_RANK", "32"))
POLL_TIMEOUT_S = float(os.environ.get("MINT_POLL_TIMEOUT_S", "600"))
POLL_SLEEP_S = float(os.environ.get("MINT_POLL_SLEEP_S", "1.0"))
TOKENIZER_TIMEOUT_S = float(os.environ.get("MINT_TOKENIZER_TIMEOUT_S", "1200"))


def _headers() -> dict[str, str]:
    if not API_KEY:
        return {}
    return {"X-API-Key": API_KEY}


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _post_json(path: str, body: dict[str, Any], *, timeout_s: float = 60.0) -> dict[str, Any]:
    url = f"{BASE_URL}{path}"
    r = requests.post(url, headers=_headers(), json=body, timeout=timeout_s)
    if r.status_code != 200:
        raise RuntimeError(f"POST {path} -> {r.status_code}: {r.text[:500]!r}")
    out = r.json()
    if not isinstance(out, dict):
        raise RuntimeError(f"POST {path} returned non-dict json: {type(out)}")
    return out


def _get_json(path: str, *, timeout_s: float = 30.0) -> dict[str, Any]:
    url = f"{BASE_URL}{path}"
    r = requests.get(url, headers=_headers(), timeout=timeout_s)
    if r.status_code != 200:
        raise RuntimeError(f"GET {path} -> {r.status_code}: {r.text[:500]!r}")
    out = r.json()
    if not isinstance(out, dict):
        raise RuntimeError(f"GET {path} returned non-dict json: {type(out)}")
    return out


def _delete(path: str, *, timeout_s: float = 60.0) -> None:
    url = f"{BASE_URL}{path}"
    r = requests.delete(url, headers=_headers(), timeout=timeout_s)
    if r.status_code != 200:
        raise RuntimeError(f"DELETE {path} -> {r.status_code}: {r.text[:500]!r}")


def _poll_future(request_id: str) -> dict[str, Any]:
    t0 = time.time()
    while True:
        url = f"{BASE_URL}/api/v1/retrieve_future"
        r = requests.post(url, headers=_headers(), json={"request_id": request_id}, timeout=30.0)
        if r.status_code == 408:
            if time.time() - t0 > POLL_TIMEOUT_S:
                raise RuntimeError(f"retrieve_future timeout after {POLL_TIMEOUT_S:.1f}s request_id={request_id}")
            time.sleep(POLL_SLEEP_S)
            continue
        if r.status_code != 200:
            raise RuntimeError(f"retrieve_future -> {r.status_code}: {r.text[:500]!r}")
        out = r.json()
        if not isinstance(out, dict):
            raise RuntimeError(f"retrieve_future returned non-dict json: {type(out)}")
        return out


def main() -> int:
    try:
        _get_json("/api/v1/healthz", timeout_s=5.0)

        session_id = _post_json("/api/v1/create_session", {"tags": ["issue-150"]}).get("session_id")
        if not isinstance(session_id, str) or not session_id:
            return _fail(f"create_session returned invalid session_id={session_id!r}")

        # Create a training model. For MLA MoE models (Moonlight/K2/DeepSeekV3) this should
        # start a Megatron actor in Ray.
        create_model = _post_json(
            "/api/v1/create_model",
            {
                "session_id": session_id,
                "model_seq_id": 0,
                "base_model": MODEL,
                "lora_config": {"rank": LORA_RANK},
            },
            timeout_s=30.0,
        )
        request_id = create_model.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            return _fail(f"create_model returned invalid request_id={request_id!r}")

        out = _poll_future(request_id)
        if "error" in out:
            err = out.get("error")
            if isinstance(err, str) and "NVTE_FUSED_ATTN" not in err:
                return _fail(f"create_model failed (unexpected error, missing NVTE_FUSED_ATTN): {err}")
            return _fail(f"create_model failed: {err!r}")

        backend = out.get("backend")
        model_id = out.get("model_id")
        if backend != "megatron":
            return _fail(f"create_model backend={backend!r} expected 'megatron' (model={MODEL!r})")
        if not isinstance(model_id, str) or not model_id:
            return _fail(f"create_model returned invalid model_id={model_id!r}")

        # Ensure the Megatron actor actually initialized. create_model intentionally does not
        # wait for __ray_ready__ for megatron workers; calling tokenizer forces a real RPC
        # through the worker group (and will surface actor-creation failures).
        try:
            _get_json(f"/api/v1/models/{model_id}/tokenizer", timeout_s=TOKENIZER_TIMEOUT_S)
        finally:
            _delete(f"/api/v1/models/{model_id}")

        print("PASS")
        return 0
    except Exception as e:
        return _fail(str(e))


if __name__ == "__main__":
    raise SystemExit(main())
