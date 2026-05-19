#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class Env:
    base_url: str
    api_key: str
    timeout_s: float
    poll_timeout_s: float
    poll_sleep_s: float


def _fail(message: str) -> int:
    print(f"FAIL: {message}", flush=True)
    return 1


def _env() -> Env:
    return Env(
        base_url=(os.environ.get("MINT_BASE_URL") or "http://localhost:8000").rstrip("/"),
        api_key=os.environ.get("MINT_API_KEY") or "dummy",
        timeout_s=float(os.environ.get("ISSUE398_TIMEOUT_S") or "120"),
        poll_timeout_s=float(os.environ.get("ISSUE398_POLL_TIMEOUT_S") or "240"),
        poll_sleep_s=float(os.environ.get("ISSUE398_POLL_SLEEP_S") or "1.0"),
    )


def _headers(env: Env) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "X-API-Key": env.api_key,
    }


def _request_payload(model_name: str) -> dict[str, Any]:
    return {
        "base_model": model_name,
        "num_samples": 1,
        "prompt": {
            "chunks": [
                {
                    "type": "encoded_text",
                    "tokens": [151644, 8948, 198],
                }
            ]
        },
        "sampling_params": {
            "max_tokens": 8,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
        },
    }


def _post_json(env: Env, path: str, payload: dict[str, Any]) -> requests.Response:
    return requests.post(
        env.base_url + path,
        json=payload,
        headers=_headers(env),
        timeout=env.timeout_s,
    )


def _poll_future(env: Env, request_id: str) -> dict[str, Any]:
    deadline = time.time() + env.poll_timeout_s
    while True:
        response = _post_json(env, "/api/v1/retrieve_future", {"request_id": request_id})
        if response.status_code == 408:
            if time.time() >= deadline:
                raise TimeoutError(f"retrieve_future timed out for request_id={request_id}")
            time.sleep(env.poll_sleep_s)
            continue
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise TypeError(f"retrieve_future returned non-dict payload: {type(payload)}")
        return payload


def _assert_sample_payload(payload: dict[str, Any]) -> list[int]:
    sequences = payload.get("sequences")
    if not isinstance(sequences, list) or len(sequences) != 1:
        raise AssertionError(f"expected exactly one sequence, got {payload!r}")
    sequence = sequences[0]
    if not isinstance(sequence, dict):
        raise AssertionError(f"sequence payload is not a dict: {sequence!r}")
    tokens = sequence.get("tokens")
    if not isinstance(tokens, list) or not all(isinstance(tok, int) for tok in tokens):
        raise AssertionError(f"sequence tokens are invalid: {sequence!r}")
    return tokens


def _run_case(env: Env, model_name: str) -> None:
    response = _post_json(env, "/api/v1/asample", _request_payload(model_name))
    if response.status_code >= 400:
        try:
            body = response.json()
        except Exception:
            body = response.text
        raise AssertionError(f"POST /api/v1/asample failed for {model_name}: status={response.status_code} body={body!r}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise AssertionError(f"asample response is not a dict for {model_name}: {payload!r}")
    request_id = payload.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise AssertionError(f"missing request_id for {model_name}: {payload!r}")
    result = _poll_future(env, request_id)
    tokens = _assert_sample_payload(result)
    print(f"PASS case model={model_name} request_id={request_id} output_tokens={len(tokens)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Qwen/Qwen3-0.6B", "Qwen/Qwen3-30B-A3B-Instruct-2507"],
        help="Base models to validate through /api/v1/asample using direct base_model selector",
    )
    args = parser.parse_args()
    env = _env()

    health = requests.get(env.base_url + "/api/v1/healthz", headers=_headers(env), timeout=env.timeout_s)
    if health.status_code != 200:
        return _fail(f"healthz status={health.status_code} body={health.text[:200]!r}")

    for model_name in args.models:
        try:
            _run_case(env, model_name)
        except Exception as e:
            return _fail(f"{model_name}: {type(e).__name__}: {e}")

    print("PASS: /api/v1/asample accepted direct base_model selectors for all requested models", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
