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

MODEL = os.environ.get("MINT_MODEL", "Qwen/Qwen3-0.6B")
LORA_RANK = int(os.environ.get("MINT_LORA_RANK", "32"))

POLL_TIMEOUT_S = float(os.environ.get("MINT_POLL_TIMEOUT_S", "1800"))
POLL_SLEEP_S = float(os.environ.get("MINT_POLL_SLEEP_S", "2.0"))


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
        r = requests.post(
            f"{BASE_URL}/api/v1/retrieve_future",
            headers=_headers(),
            json={"request_id": request_id},
            timeout=30.0,
        )
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


def _require_model_in_caps(model_name: str) -> None:
    caps = _get_json("/api/v1/get_server_capabilities", timeout_s=30.0)
    models = caps.get("supported_models")
    if not isinstance(models, list):
        raise RuntimeError(f"supported_models missing/invalid: {models!r}")
    for m in models:
        if isinstance(m, dict) and m.get("model_name") == model_name:
            return
    raise RuntimeError(f"model {model_name!r} not present in supported_models (wrong server?)")


def main() -> int:
    model_id: str | None = None
    try:
        _get_json("/api/v1/healthz", timeout_s=5.0)
        _require_model_in_caps(MODEL)

        session_id = _post_json("/api/v1/create_session", {"tags": ["issue-148"]}, timeout_s=30.0).get("session_id")
        if not isinstance(session_id, str) or not session_id:
            return _fail(f"create_session returned invalid session_id={session_id!r}")

        create_model = _post_json(
            "/api/v1/create_model",
            {"session_id": session_id, "model_seq_id": 0, "base_model": MODEL, "lora_config": {"rank": LORA_RANK}},
            timeout_s=30.0,
        )
        request_id = create_model.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            return _fail(f"create_model returned invalid request_id={request_id!r}")

        out = _poll_future(request_id)
        if "error" in out:
            return _fail(f"create_model failed: {out.get('error')!r}")
        model_id = out.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            return _fail(f"create_model returned invalid model_id={model_id!r}")

        save_state = _post_json(
            "/api/v1/save_state",
            {"model_id": model_id, "path": f"issue-148-{int(time.time())}"},
            timeout_s=30.0,
        )
        request_id = save_state.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            return _fail(f"save_state returned invalid request_id={request_id!r}")

        out = _poll_future(request_id)
        if "error" in out:
            return _fail(f"save_state failed: {out.get('error')!r}")

        path = out.get("path")
        if not isinstance(path, str) or not path:
            return _fail(f"save_state returned invalid path={path!r} payload_keys={sorted(out.keys())!r}")
        if not path.startswith("mint://"):
            return _fail(f"save_state returned non-mint path={path!r} mint_path={out.get('mint_path')!r}")

        tinker_path = out.get("tinker_path")
        if not isinstance(tinker_path, str) or not tinker_path:
            return _fail(f"save_state response missing tinker_path payload_keys={sorted(out.keys())!r}")
        if not tinker_path.startswith("mint://"):
            return _fail(f"save_state returned invalid tinker_path={tinker_path!r}")

        print("PASS")
        return 0
    except Exception as e:
        return _fail(str(e))
    finally:
        if model_id:
            try:
                _delete(f"/api/v1/models/{model_id}", timeout_s=30.0)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
