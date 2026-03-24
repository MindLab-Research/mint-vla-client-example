#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
import uuid
from typing import Any

import requests


BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")
BASE_MODEL = os.environ.get("TINKER_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
LORA_RANK = int(os.environ.get("TINKER_LORA_RANK", "8"))
LEARNING_RATE = float(os.environ.get("TINKER_LEARNING_RATE", "1e-4"))
SMALL_SEQ_LEN = int(os.environ.get("TINKER_SMALL_SEQ_LEN", "256"))
LARGE_SEQ_LEN = int(os.environ.get("TINKER_LARGE_SEQ_LEN", "0"))
CREATE_TIMEOUT_S = float(os.environ.get("TINKER_CREATE_TIMEOUT_S", "3600"))
POLL_TIMEOUT_S = float(os.environ.get("TINKER_POLL_TIMEOUT_S", "3600"))
REQUEST_TIMEOUT_S = float(os.environ.get("TINKER_REQUEST_TIMEOUT_S", "120"))
POLL_REQUEST_TIMEOUT_S = float(os.environ.get("TINKER_POLL_REQUEST_TIMEOUT_S", "180"))


def _headers() -> dict[str, str]:
    if not API_KEY:
        return {}
    return {"X-API-Key": API_KEY}


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _post_json(path: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    resp = requests.post(f"{BASE_URL}{path}", json=payload, headers=_headers(), timeout=timeout_s)
    if resp.status_code != 200:
        raise RuntimeError(f"POST {path} -> {resp.status_code}: {resp.text[:800]!r}")
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"POST {path} returned non-dict JSON: {type(data)}")
    return data


def _get_json(path: str, *, timeout_s: float) -> dict[str, Any]:
    resp = requests.get(f"{BASE_URL}{path}", headers=_headers(), timeout=timeout_s)
    if resp.status_code != 200:
        raise RuntimeError(f"GET {path} -> {resp.status_code}: {resp.text[:800]!r}")
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"GET {path} returned non-dict JSON: {type(data)}")
    return data


def _poll_future(request_id: str, *, timeout_s: float) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = requests.post(
            f"{BASE_URL}/api/v1/retrieve_future",
            json={"request_id": request_id},
            headers=_headers(),
            timeout=POLL_REQUEST_TIMEOUT_S,
        )
        if resp.status_code == 408:
            time.sleep(2.0)
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"retrieve_future({request_id}) -> {resp.status_code}: {resp.text[:800]!r}")
        data = resp.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"retrieve_future returned non-dict JSON: {type(data)}")
        return data
    raise TimeoutError(f"retrieve_future timed out after {timeout_s}s for request_id={request_id}")


def _await_maybe_async(result: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    request_id = result.get("request_id")
    if isinstance(request_id, str) and request_id:
        return _poll_future(request_id, timeout_s=timeout_s)
    return result


def _delete_model(model_id: str) -> None:
    try:
        requests.delete(f"{BASE_URL}/api/v1/models/{model_id}", headers=_headers(), timeout=120.0)
    except Exception:
        pass


def _get_max_context_len() -> int:
    caps = _get_json("/api/v1/get_server_capabilities", timeout_s=30.0)
    models = caps.get("supported_models")
    if not isinstance(models, list):
        raise RuntimeError(f"capabilities missing supported_models: {caps!r}")
    for model in models:
        if not isinstance(model, dict):
            continue
        if model.get("model_name") != BASE_MODEL:
            continue
        max_context_len = model.get("max_context_length")
        if isinstance(max_context_len, int) and max_context_len > 8:
            return max_context_len
    raise RuntimeError(f"could not find max_context_length for {BASE_MODEL!r}")


def _resolve_large_seq_len() -> int:
    if LARGE_SEQ_LEN > 0:
        return LARGE_SEQ_LEN
    max_context = _get_max_context_len()
    # Use the full advertised model length for the transition-sequence repro.
    return max_context


def _make_datum(seq_len: int) -> dict[str, Any]:
    if seq_len < 8:
        raise ValueError(f"seq_len must be >= 8, got {seq_len}")
    tokens = [10 + (idx % 64) for idx in range(seq_len)]
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = [1.0] * len(target_tokens)
    return {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": input_tokens}]},
        "loss_fn_inputs": {
            "target_tokens": {
                "data": target_tokens,
                "shape": [len(target_tokens)],
                "dtype": "int64",
            },
            "weights": {
                "data": weights,
                "shape": [len(weights)],
                "dtype": "float32",
            },
        },
    }


def _create_model(session_id: str, model_seq_id: int) -> tuple[str, str]:
    created = _post_json(
        "/api/v1/create_model",
        {
            "session_id": session_id,
            "model_seq_id": model_seq_id,
            "base_model": BASE_MODEL,
            "lora_config": {"rank": LORA_RANK},
            "learning_rate": LEARNING_RATE,
            "user_metadata": {"issue": 370, "script": "reproduce_issue_370_transition_sequence.py"},
        },
        timeout_s=60.0,
    )
    created = _await_maybe_async(created, timeout_s=CREATE_TIMEOUT_S)
    model_id = created.get("model_id")
    backend = created.get("backend")
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError(f"create_model missing model_id: {created!r}")
    if not isinstance(backend, str) or not backend:
        raise RuntimeError(f"create_model missing backend: {created!r}")
    print(f"created model_id={model_id} backend={backend}", flush=True)
    return model_id, backend


def _forward_backward(model_id: str, datum: dict[str, Any], *, label: str) -> dict[str, Any]:
    t0 = time.time()
    result = _post_json(
        "/api/v1/forward_backward",
        {
            "model_id": model_id,
            "forward_backward_input": {"data": [datum], "loss_fn": "cross_entropy"},
        },
        timeout_s=REQUEST_TIMEOUT_S,
    )
    result = _await_maybe_async(result, timeout_s=POLL_TIMEOUT_S)
    if "error" in result:
        raise RuntimeError(f"{label}: forward_backward failed: {result.get('error')!r}")
    elapsed = time.time() - t0
    metrics = result.get("metrics")
    loss = None if not isinstance(metrics, dict) else metrics.get("loss:mean")
    print(f"{label}: forward_backward ok elapsed_s={elapsed:.3f} loss={loss}", flush=True)
    return result


def _forward(model_id: str, datum: dict[str, Any], *, label: str) -> dict[str, Any]:
    t0 = time.time()
    result = _post_json(
        "/api/v1/forward",
        {
            "model_id": model_id,
            "forward_input": {"data": [datum], "loss_fn": "cross_entropy", "loss_fn_config": {}},
        },
        timeout_s=REQUEST_TIMEOUT_S,
    )
    result = _await_maybe_async(result, timeout_s=POLL_TIMEOUT_S)
    if "error" in result:
        raise RuntimeError(f"{label}: forward failed: {result.get('error')!r}")
    elapsed = time.time() - t0
    metrics = result.get("metrics")
    loss = None if not isinstance(metrics, dict) else metrics.get("loss:mean")
    print(f"{label}: forward ok elapsed_s={elapsed:.3f} loss={loss}", flush=True)
    return result


def _optim_step(model_id: str, *, label: str) -> dict[str, Any]:
    t0 = time.time()
    result = _post_json(
        "/api/v1/optim_step",
        {
            "model_id": model_id,
            "adam_params": {
                "learning_rate": LEARNING_RATE,
                "beta1": 0.9,
                "beta2": 0.95,
                "eps": 1e-12,
            },
        },
        timeout_s=REQUEST_TIMEOUT_S,
    )
    result = _await_maybe_async(result, timeout_s=POLL_TIMEOUT_S)
    if "error" in result:
        raise RuntimeError(f"{label}: optim_step failed: {result.get('error')!r}")
    elapsed = time.time() - t0
    print(f"{label}: optim_step ok elapsed_s={elapsed:.3f}", flush=True)
    return result


def main() -> int:
    suffix = uuid.uuid4().hex[:8]
    session_a = f"issue370-a-{suffix}"
    session_b = f"issue370-b-{suffix}"
    model_a: str | None = None
    model_b: str | None = None
    try:
        large_seq_len = _resolve_large_seq_len()
        small_datum = _make_datum(SMALL_SEQ_LEN)
        large_datum = _make_datum(large_seq_len)
        print(
            f"transition_repro model={BASE_MODEL} small_seq_len={SMALL_SEQ_LEN} large_seq_len={large_seq_len}",
            flush=True,
        )

        model_a, backend_a = _create_model(session_a, 0)
        model_b, backend_b = _create_model(session_b, 0)
        if backend_a != "megatron" or backend_b != "megatron":
            raise RuntimeError(
                f"expected Megatron backend for transition repro, got backend_a={backend_a!r} backend_b={backend_b!r}"
            )

        _forward_backward(model_a, small_datum, label="warm_a_small")
        _optim_step(model_a, label="warm_a_small")
        _forward(model_b, small_datum, label="warm_b_small")

        _forward_backward(model_a, large_datum, label="A_forward_backward_large")
        _forward(model_b, large_datum, label="B_forward_large")
        _optim_step(model_a, label="A_optim_after_B_forward")

        print("PASS: transition sequence completed without server-side failure", flush=True)
        return 0
    except Exception as exc:
        return _fail(str(exc))
    finally:
        for model_id in (model_b, model_a):
            if model_id:
                _delete_model(model_id)


if __name__ == "__main__":
    raise SystemExit(main())
