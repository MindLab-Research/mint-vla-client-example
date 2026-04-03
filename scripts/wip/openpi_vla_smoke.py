#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from typing import Any

import requests

PNG_1X1_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z0FcAAAAASUVORK5CYII="


def _headers(api_key: str | None) -> dict[str, str]:
    if not api_key:
        return {}
    return {"X-API-Key": api_key}


def _get_json(base_url: str, path: str, headers: dict[str, str], *, timeout_s: float = 30.0) -> dict[str, Any]:
    resp = requests.get(f"{base_url}{path}", headers=headers, timeout=timeout_s)
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict):
        raise TypeError(f"GET {path} returned non-dict JSON: {type(payload)}")
    return payload


def _post_json(base_url: str, path: str, headers: dict[str, str], payload: dict[str, Any], *, timeout_s: float = 120.0) -> dict[str, Any]:
    resp = requests.post(f"{base_url}{path}", headers=headers, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    body = resp.json()
    if not isinstance(body, dict):
        raise TypeError(f"POST {path} returned non-dict JSON: {type(body)}")
    return body


def _poll_future(base_url: str, headers: dict[str, str], request_id: str, *, timeout_s: float = 1800.0) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = requests.post(f"{base_url}/api/v1/retrieve_future", headers=headers, json={"request_id": request_id}, timeout=60.0)
        if resp.status_code == 408:
            time.sleep(1.0)
            continue
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, dict):
            raise TypeError(f"retrieve_future returned non-dict JSON: {type(payload)}")
        return payload
    raise TimeoutError(f"retrieve_future timed out after {timeout_s}s for request_id={request_id}")


def _await_result(base_url: str, headers: dict[str, str], payload: dict[str, Any], *, timeout_s: float = 1800.0) -> dict[str, Any]:
    request_id = payload.get("request_id")
    if isinstance(request_id, str) and request_id:
        return _poll_future(base_url, headers, request_id, timeout_s=timeout_s)
    return payload


def _image_chunk() -> dict[str, Any]:
    return {"type": "image", "data": PNG_1X1_BASE64, "format": "png", "expected_tokens": 256}


def _observation(prompt_tokens: list[int], *, state_dim: int) -> dict[str, Any]:
    return {
        "state": {"data": [0.0] * state_dim, "shape": [state_dim], "dtype": "float32"},
        "model_input": {
            "chunks": [_image_chunk(), _image_chunk(), _image_chunk(), {"type": "encoded_text", "tokens": prompt_tokens}],
        },
    }


def _fast_datum() -> dict[str, Any]:
    return {
        "observation": _observation([11, 12, 13], state_dim=8),
        "supervision": {
            "target_tokens": {"data": [21, 22], "shape": [2], "dtype": "int64"},
            "weights": {"data": [1.0, 1.0], "shape": [2], "dtype": "float32"},
            "token_ar_mask": {"data": [1, 1], "shape": [2], "dtype": "int64"},
        },
    }


def _pi05_datum() -> dict[str, Any]:
    return {
        "observation": _observation([11, 12, 13], state_dim=8),
        "supervision": {
            "actions": {"data": [0.0] * (10 * 7), "shape": [10, 7], "dtype": "float32"},
        },
    }


def _create_model(base_url: str, headers: dict[str, str], *, base_model: str) -> tuple[str, dict[str, Any]]:
    payload = {
        "session_id": f"smoke-{uuid.uuid4().hex[:12]}",
        "model_seq_id": 0,
        "base_model": base_model,
        "lora_config": {"rank": 16, "train_attn": True, "train_mlp": True, "train_unembed": True},
        "user_metadata": {"script": "scripts/wip/openpi_vla_smoke.py"},
    }
    result = _await_result(base_url, headers, _post_json(base_url, "/api/v1/create_model", headers, payload))
    model_id = result.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError(f"create_model missing model_id: {result!r}")
    return model_id, result


def _delete_model(base_url: str, headers: dict[str, str], model_id: str) -> None:
    try:
        requests.delete(f"{base_url}/api/v1/models/{model_id}", headers=headers, timeout=120.0)
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("TINKER_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--api-key", default=os.environ.get("TINKER_API_KEY", "dummy"))
    parser.add_argument("--model", choices=["openpi/pi0-fast-libero-low-mem-finetune", "openpi/pi05-libero-low-mem-finetune"], default="openpi/pi0-fast-libero-low-mem-finetune")
    parser.add_argument("--skip-action", action="store_true")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    headers = _headers(args.api_key)
    actors_before = _get_json(base_url, "/api/v1/actors", headers)
    model_id = ""
    action_session_id = ""
    try:
        model_id, create_result = _create_model(base_url, headers, base_model=args.model)
        datum = _fast_datum() if "pi0-fast" in args.model else _pi05_datum()
        train_result = _await_result(base_url, headers, _post_json(base_url, "/api/v1/mint/vla/train_step", headers, {"model_id": model_id, "loss_fn": "cross_entropy" if "pi0-fast" in args.model else "flow_matching", "data": [datum]}))
        save_result: dict[str, Any] = {}
        action_result: dict[str, Any] = {}
        if not args.skip_action:
            save_result = _await_result(base_url, headers, _post_json(base_url, "/api/v1/save_weights_for_sampler", headers, {"model_id": model_id, "path": f"smoke_sampler_{uuid.uuid4().hex[:8]}"}))
            model_path = save_result.get("path")
            if not isinstance(model_path, str) or not model_path:
                raise RuntimeError(f"save_weights_for_sampler missing path: {save_result!r}")
            action_created = _post_json(base_url, "/api/v1/mint/action_sessions", headers, {"session_id": f"smoke-action-{uuid.uuid4().hex[:12]}", "base_model": args.model, "model_path": model_path})
            action_session_id = action_created["action_session_id"]
            action_result = _await_result(base_url, headers, _post_json(base_url, f"/api/v1/mint/action_sessions/{action_session_id}/act", headers, {"observation": datum["observation"]}))
        actors_after = _get_json(base_url, "/api/v1/actors", headers)
        payload = {
            "model": args.model,
            "model_id": model_id,
            "create_result": create_result,
            "train_result": train_result,
            "save_result": save_result,
            "action_session_id": action_session_id,
            "action_result": action_result,
            "actors_before": actors_before,
            "actors_after": actors_after,
        }
        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        print(json.dumps(payload, indent=2))
        return 0
    finally:
        if action_session_id:
            try:
                requests.delete(f"{base_url}/api/v1/mint/action_sessions/{action_session_id}", headers=headers, timeout=120.0)
            except Exception:
                pass
        if model_id:
            _delete_model(base_url, headers, model_id)


if __name__ == "__main__":
    raise SystemExit(main())
