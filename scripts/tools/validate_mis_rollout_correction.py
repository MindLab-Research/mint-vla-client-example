#!/usr/bin/env python3
"""Validate session-level MIS rollout_correction wiring end-to-end.

This script runs a minimal real training request:
1) create_model with session-level rollout_correction_config configured for Seq-MIS
2) forward_backward with loss_fn=importance_sampling (no per-step rollout config)

Run locally against dev server (via SSH tunnel):
  TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=dummy \
  python scripts/tools/validate_mis_rollout_correction.py \
    --base-model Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from typing import Any

import requests


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


BASE_URL = _env("TINKER_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = _env("TINKER_API_KEY", "dummy")


def _headers(api_key: str) -> dict[str, str]:
    return {"X-API-Key": api_key} if api_key else {}


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    if resp.status_code != 200:
        raise RuntimeError(f"POST {url} returned {resp.status_code}: {resp.text[:400]!r}")
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"POST {url} returned non-dict json: {type(data)}")
    return data


def _poll_future(base_url: str, headers: dict[str, str], request_id: str, *, timeout_s: float, interval_s: float) -> dict[str, Any]:
    url = f"{base_url}/api/v1/retrieve_future"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = requests.post(url, headers=headers, json={"request_id": request_id}, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if not isinstance(data, dict):
                raise RuntimeError(f"retrieve_future returned non-dict json: {type(data)}")
            return data
        if resp.status_code == 408:
            time.sleep(interval_s)
            continue
        raise RuntimeError(f"POST {url} returned {resp.status_code}: {resp.text[:400]!r}")
    raise TimeoutError(f"retrieve_future timed out after {timeout_s}s (request_id={request_id})")


def _delete_model(base_url: str, headers: dict[str, str], model_id: str) -> None:
    try:
        requests.delete(f"{base_url}/api/v1/models/{model_id}", headers=headers, timeout=60)
    except Exception:
        pass


def _build_rl_datum() -> dict[str, Any]:
    # Keep token IDs small and deterministic.
    # For RL losses, lengths of weights/logprobs/advantages must match target length.
    tokens = [10, 11, 12, 13, 14, 15]  # seq_len=6
    input_tokens = tokens[:-1]  # len=5
    target_tokens = tokens[1:]  # len=5

    weights = [0.0, 1.0, 1.0, 1.0, 1.0]
    old_logprobs = [0.0, 0.0, 0.0, 0.0, 0.0]
    advantages = [0.0, 1.0, -1.5, 0.7, -0.2]

    return {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": input_tokens}]},
        "loss_fn_inputs": {
            "target_tokens": {"data": target_tokens, "shape": [len(target_tokens)], "dtype": "int64"},
            "weights": {"data": weights, "shape": [len(weights)], "dtype": "float32"},
            "logprobs": {"data": old_logprobs, "shape": [len(old_logprobs)], "dtype": "float32"},
            "advantages": {"data": advantages, "shape": [len(advantages)], "dtype": "float32"},
        },
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate MIS rollout_correction_config path")
    p.add_argument("--base-url", default=BASE_URL)
    p.add_argument("--api-key", default=API_KEY)
    p.add_argument("--base-model", default=_env("TINKER_MODEL", "Qwen/Qwen3-0.6B"))
    p.add_argument("--lora-rank", type=int, default=int(_env("TINKER_LORA_RANK", "8")))
    p.add_argument("--create-timeout-s", type=float, default=float(_env("TINKER_CREATE_MODEL_TIMEOUT_S", "3600")))
    p.add_argument("--forward-backward-timeout-s", type=float, default=float(_env("TINKER_FORWARD_BACKWARD_TIMEOUT_S", "1800")))
    p.add_argument("--poll-interval-s", type=float, default=2.0)
    p.add_argument(
        "--mis-threshold", type=float, default=1.1, help="rollout_correction_config.rollout_is_threshold"
    )
    p.add_argument("--skip-cleanup", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    base_url = str(args.base_url).rstrip("/")
    headers = _headers(str(args.api_key))
    model_id: str | None = None

    try:
        session_id = f"validate-mis-{uuid.uuid4().hex[:8]}"
        created = _post_json(
            f"{base_url}/api/v1/create_model",
            headers,
            {
                "session_id": session_id,
                "model_seq_id": 0,
                "base_model": args.base_model,
                "lora_config": {"rank": int(args.lora_rank)},
                "rollout_correction_config": {
                    "rollout_is": "sequence",
                    "rollout_is_threshold": float(args.mis_threshold),
                    "rollout_rs": "seq_sum_k1",
                    "rollout_rs_threshold": "0.5_2.0",
                    "bypass_mode": True,
                    "loss_type": "reinforce",
                },
            },
            timeout_s=60.0,
        )
        if "request_id" in created:
            created = _poll_future(
                base_url,
                headers,
                str(created["request_id"]),
                timeout_s=float(args.create_timeout_s),
                interval_s=float(args.poll_interval_s),
            )
        if "error" in created:
            return _fail(f"create_model failed: {created.get('error')!r}")
        model_id = created.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            return _fail(f"create_model missing/invalid model_id: {created!r}")

        datum = _build_rl_datum()
        fb = _post_json(
            f"{base_url}/api/v1/forward_backward",
            headers,
            {
                "model_id": model_id,
                "forward_backward_input": {
                    "data": [datum],
                    "loss_fn": "importance_sampling",
                },
            },
            timeout_s=60.0,
        )
        if "request_id" in fb:
            fb = _poll_future(
                base_url,
                headers,
                str(fb["request_id"]),
                timeout_s=float(args.forward_backward_timeout_s),
                interval_s=float(args.poll_interval_s),
            )
        if "error" in fb:
            return _fail(f"forward_backward failed: {fb.get('error')!r}")

        outs = fb.get("loss_fn_outputs")
        if not isinstance(outs, list) or not outs:
            return _fail(f"forward_backward missing loss_fn_outputs: {fb!r}")

        print("PASS: MIS rollout_correction request succeeded and response was valid")
        return 0
    except Exception as e:
        return _fail(str(e))
    finally:
        if model_id and not args.skip_cleanup:
            _delete_model(base_url, headers, model_id)


if __name__ == "__main__":
    raise SystemExit(main())
