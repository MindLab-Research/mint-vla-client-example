import os
import sys
import time
import uuid
from typing import Any

import requests

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")

BASE_MODEL = os.environ.get("TINKER_MODEL", "Qwen/Qwen3-0.6B")
LORA_RANK = int(os.environ.get("TINKER_LORA_RANK", "8"))

CREATE_TIMEOUT_S = float(os.environ.get("TINKER_CREATE_MODEL_TIMEOUT_S", "3600"))
SAVE_TIMEOUT_S = float(os.environ.get("TINKER_SAVE_STATE_TIMEOUT_S", "3600"))
RESUME_TIMEOUT_S = float(os.environ.get("TINKER_CREATE_MODEL_FROM_STATE_TIMEOUT_S", "3600"))


def _headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY} if API_KEY else {}


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _post_json(url: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    resp = requests.post(url, headers=_headers(), json=payload, timeout=timeout_s)
    if resp.status_code != 200:
        raise RuntimeError(f"POST {url} returned {resp.status_code}: {resp.text[:400]!r}")
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"POST {url} returned non-dict json: {type(data)}")
    return data


def _poll_future(request_id: str, *, timeout_s: float) -> dict[str, Any]:
    url = f"{BASE_URL}/api/v1/retrieve_future"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = requests.post(url, headers=_headers(), json={"request_id": request_id}, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if not isinstance(data, dict):
                raise RuntimeError(f"retrieve_future returned non-dict json: {type(data)}")
            return data
        if resp.status_code == 408:
            time.sleep(2)
            continue
        raise RuntimeError(f"POST {url} returned {resp.status_code}: {resp.text[:400]!r}")
    raise TimeoutError(f"retrieve_future timed out after {timeout_s}s (request_id={request_id})")


def _delete_model(model_id: str) -> None:
    try:
        requests.delete(f"{BASE_URL}/api/v1/models/{model_id}", headers=_headers(), timeout=60)
    except Exception:
        pass


def main() -> int:
    created_model_id: str | None = None
    resumed_model_id: str | None = None
    try:
        session_id = f"repro-86-{uuid.uuid4().hex[:8]}"

        created = _post_json(
            f"{BASE_URL}/api/v1/create_model",
            {
                "session_id": session_id,
                "model_seq_id": 0,
                "base_model": BASE_MODEL,
                "lora_config": {"rank": LORA_RANK},
            },
            timeout_s=60.0,
        )
        if "request_id" in created:
            created = _poll_future(str(created["request_id"]), timeout_s=CREATE_TIMEOUT_S)

        created_model_id = created.get("model_id")
        if not created_model_id:
            return _fail(f"create_model missing model_id: {created!r}")

        # Official contract: save_state(name=...) returns a resolvable URI like
        # tinker://<model_id>/<name>. Repro uses a named checkpoint.
        checkpoint_name = f"issue86_{uuid.uuid4().hex[:8]}"
        saved = _post_json(
            f"{BASE_URL}/api/v1/save_state",
            {"model_id": created_model_id, "path": checkpoint_name},
            timeout_s=60.0,
        )
        if "request_id" in saved:
            saved = _poll_future(str(saved["request_id"]), timeout_s=SAVE_TIMEOUT_S)
        if "error" in saved:
            return _fail(f"save_state failed: {saved.get('error')!r}")

        resume_path = f"tinker://{created_model_id}/{checkpoint_name}"

        # Mimic "resume after restart": delete the training model, then create a new one from state.
        _delete_model(created_model_id)
        created_model_id = None

        resumed = _post_json(
            f"{BASE_URL}/api/v1/create_model_from_state",
            {
                "session_id": session_id,
                "model_seq_id": 1,
                "base_model": BASE_MODEL,
                "state_path": resume_path,
                "lora_config": {"rank": LORA_RANK},
                "load_optimizer": True,
            },
            timeout_s=60.0,
        )
        if "request_id" in resumed:
            resumed = _poll_future(str(resumed["request_id"]), timeout_s=RESUME_TIMEOUT_S)
        if "error" in resumed:
            return _fail(f"create_model_from_state failed for {resume_path!r}: {resumed.get('error')!r}")

        resumed_model_id = resumed.get("model_id")
        if not resumed_model_id:
            return _fail(f"create_model_from_state missing model_id: {resumed!r}")

        print("PASS")
        return 0
    except Exception as e:
        return _fail(str(e))
    finally:
        if created_model_id:
            _delete_model(created_model_id)
        if resumed_model_id:
            _delete_model(resumed_model_id)


if __name__ == "__main__":
    raise SystemExit(main())
