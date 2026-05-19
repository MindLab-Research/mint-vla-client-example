#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
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
            time.sleep(0.5)
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


def _datum(seed: int) -> dict[str, Any]:
    token0 = 20 + (seed % 11)
    token1 = 40 + (seed % 13)
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
            "target_tokens": {"data": [token0, token1], "shape": [2], "dtype": "int64"},
            "weights": {"data": [1.0, 1.0], "shape": [2], "dtype": "float32"},
            "token_ar_mask": {"data": [1, 1], "shape": [2], "dtype": "int64"},
        },
    }


def _create_model(base_url: str, headers: dict[str, str], *, model: str, tag: str) -> str:
    payload = {
        "session_id": f"dual-train-{tag}-{uuid.uuid4().hex[:8]}",
        "model_seq_id": 0,
        "base_model": model,
        "lora_config": {"rank": 16, "train_attn": True, "train_mlp": True, "train_unembed": True},
        "user_metadata": {"script": "scripts/wip/openpi_vla_dual_train_isolation.py", "tag": tag},
    }
    result = _await_result(base_url, headers, _post_json(base_url, "/api/v1/create_model", headers, payload))
    model_id = result.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError(f"create_model missing model_id: {result!r}")
    return model_id


def _train_step(base_url: str, headers: dict[str, str], *, model_id: str, seed: int) -> float:
    payload = {"model_id": model_id, "loss_fn": "cross_entropy", "data": [_datum(seed)]}
    result = _await_result(base_url, headers, _post_json(base_url, "/api/v1/mint/vla/train_step", headers, payload))
    return float(result.get("metrics", {}).get("loss:mean", 0.0))


def _delete_model(base_url: str, headers: dict[str, str], model_id: str) -> None:
    try:
        requests.delete(f"{base_url}/api/v1/models/{model_id}", headers=headers, timeout=120.0)
    except Exception:
        pass


def _worker(base_url: str, headers: dict[str, str], *, model: str, tag: str, steps: int, start_event: threading.Event) -> dict[str, Any]:
    start_event.wait()
    model_id = ""
    losses: list[float] = []
    try:
        model_id = _create_model(base_url, headers, model=model, tag=tag)
        for i in range(steps):
            losses.append(_train_step(base_url, headers, model_id=model_id, seed=(hash(tag) & 0xFFFF) + i))
        return {"tag": tag, "ok": True, "model_id": model_id, "losses": losses}
    except Exception as exc:
        return {"tag": tag, "ok": False, "error": repr(exc)}
    finally:
        if model_id:
            _delete_model(base_url, headers, model_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("MINT_BASE_URL") or os.environ.get("MINT_BASE_URL") or "http://localhost:8000")
    parser.add_argument("--api-key", default=os.environ.get("MINT_API_KEY") or os.environ.get("MINT_API_KEY") or "")
    parser.add_argument("--model", default="openpi/pi0-fast-libero-low-mem-finetune")
    parser.add_argument("--steps-per-model", type=int, default=3)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    headers = _headers(args.api_key)
    start_event = threading.Event()
    results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_worker, base_url, headers, model=args.model, tag="A", steps=args.steps_per_model, start_event=start_event),
            pool.submit(_worker, base_url, headers, model=args.model, tag="B", steps=args.steps_per_model, start_event=start_event),
        ]
        start_event.set()
        for fut in as_completed(futures):
            results.append(fut.result())

    summary = {
        "model": args.model,
        "steps_per_model": args.steps_per_model,
        "ok": sum(1 for r in results if r.get("ok")),
        "failed": sum(1 for r in results if not r.get("ok")),
        "results": sorted(results, key=lambda r: r["tag"]),
    }
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
