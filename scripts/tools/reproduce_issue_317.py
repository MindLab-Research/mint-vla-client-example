from __future__ import annotations

import os
import sys
import time
import uuid
from typing import Any

import requests


BASE_URL = (os.environ.get("MINT_BASE_URL") or f"http://localhost:{os.environ.get('MINT_PORT', '10317')}").rstrip("/")
API_KEY = os.environ.get("MINT_API_KEY", "dummy")
BASE_MODEL = os.environ.get("MINT_BASE_MODEL", "Qwen/Qwen3-0.6B")
LORA_RANK = int(os.environ.get("MINT_LORA_RANK", "8"))
SAVE_TIMEOUT_S = float(os.environ.get("MINT_SAVE_TIMEOUT_S", "20"))
FUTURE_TIMEOUT_S = float(os.environ.get("MINT_FUTURE_TIMEOUT_S", "900"))
LIST_COMPLETE_TIMEOUT_S = float(os.environ.get("MINT_LIST_COMPLETE_TIMEOUT_S", "120"))
POLL_SLEEP_S = float(os.environ.get("MINT_POLL_SLEEP_S", "1.0"))
PENDING_CACHE_STATUSES = {"pending", "in_progress"}


def _headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY} if API_KEY else {}


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr, flush=True)
    return 1


def _post_json(path: str, payload: dict[str, Any], *, timeout_s: float = 60.0) -> tuple[int, dict[str, Any]]:
    resp = requests.post(f"{BASE_URL}{path}", headers=_headers(), json=payload, timeout=timeout_s)
    try:
        data = resp.json()
    except Exception:
        data = {"_non_json_body": resp.text[:800]}
    if not isinstance(data, dict):
        data = {"_non_dict_json": str(type(data))}
    return resp.status_code, data


def _get_json(path: str, *, timeout_s: float = 60.0) -> tuple[int, dict[str, Any]]:
    resp = requests.get(f"{BASE_URL}{path}", headers=_headers(), timeout=timeout_s)
    try:
        data = resp.json()
    except Exception:
        data = {"_non_json_body": resp.text[:800]}
    if not isinstance(data, dict):
        data = {"_non_dict_json": str(type(data))}
    return resp.status_code, data


def _poll_future(request_id: str, *, timeout_s: float = FUTURE_TIMEOUT_S) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status, data = _post_json("/api/v1/retrieve_future", {"request_id": request_id}, timeout_s=30.0)
        if status == 200:
            return data
        if status != 408:
            raise RuntimeError(f"retrieve_future returned {status}: {data!r}")
        time.sleep(POLL_SLEEP_S)
    raise TimeoutError(f"retrieve_future timed out after {timeout_s:.1f}s (request_id={request_id})")


def _create_model(*, session_id: str, model_seq_id: int, state_path: str | None = None) -> str:
    if state_path is None:
        path = "/api/v1/create_model"
        payload = {
            "session_id": session_id,
            "model_seq_id": model_seq_id,
            "base_model": BASE_MODEL,
            "lora_config": {"rank": LORA_RANK},
        }
    else:
        path = "/api/v1/create_model_from_state"
        payload = {
            "session_id": session_id,
            "model_seq_id": model_seq_id,
            "base_model": BASE_MODEL,
            "state_path": state_path,
            "lora_config": {"rank": LORA_RANK},
            "load_optimizer": True,
        }
    status, data = _post_json(path, payload, timeout_s=60.0)
    if status != 200:
        raise RuntimeError(f"{path} returned {status}: {data!r}")
    request_id = data.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise RuntimeError(f"{path} missing request_id: {data!r}")
    result = _poll_future(request_id)
    model_id = result.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError(f"{path} future missing model_id: {result!r}")
    return model_id


def _delete_model(model_id: str) -> None:
    try:
        requests.delete(f"{BASE_URL}/api/v1/models/{model_id}", headers=_headers(), timeout=60)
    except Exception:
        pass


def _checkpoint_entry(model_id: str, checkpoint_name: str) -> dict[str, Any] | None:
    status, data = _get_json(f"/api/v1/training_runs/{model_id}/checkpoints")
    if status == 404:
        return None
    if status != 200:
        raise RuntimeError(f"list_checkpoints returned {status}: {data!r}")
    items = data.get("checkpoints")
    if not isinstance(items, list):
        raise RuntimeError(f"list_checkpoints invalid payload: {data!r}")
    checkpoint_id = f"weights/{checkpoint_name}"
    for item in items:
        if isinstance(item, dict) and item.get("checkpoint_id") == checkpoint_id:
            return item
    return None


def main() -> int:
    primary_model_id: str | None = None
    checkpoint_name = f"issue317-{uuid.uuid4().hex[:8]}"
    try:
        health_status, health = _get_json("/api/v1/healthz", timeout_s=10.0)
        if health_status != 200:
            return _fail(f"healthz returned {health_status}: {health!r}")

        primary_model_id = _create_model(
            session_id=f"issue317-primary-{uuid.uuid4().hex[:8]}",
            model_seq_id=0,
        )

        started = time.time()
        status, submitted = _post_json(
            "/api/v1/save_state",
            {"model_id": primary_model_id, "path": checkpoint_name},
            timeout_s=60.0,
        )
        if status != 200:
            return _fail(f"save_state returned {status}: {submitted!r}")
        request_id = submitted.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            return _fail(f"save_state missing request_id: {submitted!r}")

        save_result = _poll_future(request_id, timeout_s=SAVE_TIMEOUT_S)
        elapsed = time.time() - started
        if elapsed >= SAVE_TIMEOUT_S:
            return _fail(f"save_state future stayed blocked for {elapsed:.1f}s")

        if save_result.get("mirror_status") != "pending":
            return _fail(f"save_state did not return mirror_status=pending: {save_result!r}")
        if save_result.get("storage_tier") != "persistent_cache":
            return _fail(f"save_state did not return storage_tier=persistent_cache: {save_result!r}")
        filesystem_path = save_result.get("filesystem_path")
        if not isinstance(filesystem_path, str) or "/persistent_cache/" not in filesystem_path:
            return _fail(f"save_state filesystem_path is not cache-backed: {save_result!r}")
        tinker_path = save_result.get("tinker_path") or save_result.get("path")
        if not isinstance(tinker_path, str) or not tinker_path:
            return _fail(f"save_state missing checkpoint URI: {save_result!r}")

        listed = _checkpoint_entry(primary_model_id, checkpoint_name)
        if listed is None:
            return _fail("list_checkpoints did not surface the pending cache checkpoint")
        if listed.get("mirror_status") not in PENDING_CACHE_STATUSES or listed.get("storage_tier") != "persistent_cache":
            return _fail(f"list_checkpoints did not expose cache-backed pre-complete state: {listed!r}")

        deadline = time.time() + LIST_COMPLETE_TIMEOUT_S
        while time.time() < deadline:
            listed = _checkpoint_entry(primary_model_id, checkpoint_name)
            if listed and listed.get("mirror_status") == "complete" and listed.get("storage_tier") == "persistent_tos":
                print("PASS", flush=True)
                return 0
            time.sleep(POLL_SLEEP_S)

        return _fail(f"mirror did not reach complete state within {LIST_COMPLETE_TIMEOUT_S:.1f}s")
    except Exception as e:
        return _fail(str(e))
    finally:
        if primary_model_id:
            _delete_model(primary_model_id)


if __name__ == "__main__":
    raise SystemExit(main())
