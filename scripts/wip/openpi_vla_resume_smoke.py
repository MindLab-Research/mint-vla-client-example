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


def _train_datum() -> dict[str, Any]:
    return {
        "observation": {
            "state": {"data": [0.0] * 8, "shape": [8], "dtype": "float32"},
            "model_input": {
                "chunks": [
                    {"type": "image", "data": PNG_1X1_BASE64, "format": "png", "expected_tokens": 256},
                    {"type": "image", "data": PNG_1X1_BASE64, "format": "png", "expected_tokens": 256},
                    {"type": "image", "data": PNG_1X1_BASE64, "format": "png", "expected_tokens": 256},
                    {"type": "encoded_text", "tokens": [11, 12, 13]},
                ]
            },
        },
        "supervision": {
            "target_tokens": {"data": [21, 22], "shape": [2], "dtype": "int64"},
            "weights": {"data": [1.0, 1.0], "shape": [2], "dtype": "float32"},
            "token_ar_mask": {"data": [1, 1], "shape": [2], "dtype": "int64"},
        },
    }


def _create_model(base_url: str, headers: dict[str, str], *, base_model: str, tag: str) -> str:
    payload = {
        "session_id": f"resume-smoke-{tag}-{uuid.uuid4().hex[:8]}",
        "model_seq_id": 0,
        "base_model": base_model,
        "lora_config": {"rank": 16, "train_attn": True, "train_mlp": True, "train_unembed": True},
        "user_metadata": {"script": "scripts/wip/openpi_vla_resume_smoke.py"},
    }
    result = _await_result(base_url, headers, _post_json(base_url, "/api/v1/create_model", headers, payload))
    model_id = result.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError(f"create_model missing model_id: {result!r}")
    return model_id


def _create_model_from_state(base_url: str, headers: dict[str, str], *, base_model: str, state_path: str) -> str:
    payload = {
        "session_id": f"resume-smoke-restored-{uuid.uuid4().hex[:8]}",
        "model_seq_id": 0,
        "base_model": base_model,
        "state_path": state_path,
        "lora_config": {"rank": 16, "train_attn": True, "train_mlp": True, "train_unembed": True},
        "load_optimizer": True,
        "user_metadata": {"script": "scripts/wip/openpi_vla_resume_smoke.py"},
    }
    result = _await_result(base_url, headers, _post_json(base_url, "/api/v1/create_model_from_state", headers, payload))
    model_id = result.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError(f"create_model_from_state missing model_id: {result!r}")
    return model_id


def _train_step(base_url: str, headers: dict[str, str], *, model_id: str) -> dict[str, Any]:
    payload = {"model_id": model_id, "loss_fn": "cross_entropy", "data": [_train_datum()]}
    return _await_result(base_url, headers, _post_json(base_url, "/api/v1/mint/vla/train_step", headers, payload))


def _save_state(base_url: str, headers: dict[str, str], *, model_id: str) -> str:
    payload = {"model_id": model_id, "path": f"resume_smoke_{uuid.uuid4().hex[:8]}"}
    result = _await_result(base_url, headers, _post_json(base_url, "/api/v1/save_state", headers, payload))
    path = result.get("path")
    if not isinstance(path, str) or not path:
        raise RuntimeError(f"save_state missing path: {result!r}")
    return path


def _delete_model(base_url: str, headers: dict[str, str], model_id: str) -> None:
    try:
        requests.delete(f"{base_url}/api/v1/models/{model_id}", headers=headers, timeout=120.0)
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("MINT_BASE_URL") or os.environ.get("TINKER_BASE_URL") or "http://localhost:8000")
    parser.add_argument("--api-key", default=os.environ.get("MINT_API_KEY") or os.environ.get("TINKER_API_KEY") or "")
    parser.add_argument("--model", default="openpi/pi0-fast-libero-low-mem-finetune")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    headers = _headers(args.api_key)
    model_a = ""
    model_b = ""
    try:
        model_a = _create_model(base_url, headers, base_model=args.model, tag="pre")
        pre = _train_step(base_url, headers, model_id=model_a)
        state_path = _save_state(base_url, headers, model_id=model_a)
        model_b = _create_model_from_state(base_url, headers, base_model=args.model, state_path=state_path)
        post = _train_step(base_url, headers, model_id=model_b)
        payload = {
            "model": args.model,
            "model_a": model_a,
            "model_b": model_b,
            "state_path": state_path,
            "pre_metrics": pre.get("metrics", {}),
            "post_metrics": post.get("metrics", {}),
        }
        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        print(json.dumps(payload, indent=2))
        return 0
    finally:
        if model_b:
            _delete_model(base_url, headers, model_b)
        if model_a:
            _delete_model(base_url, headers, model_a)


if __name__ == "__main__":
    raise SystemExit(main())
