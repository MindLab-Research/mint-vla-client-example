#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import subprocess
import sys
import time
import uuid
from typing import Any

import requests


def _load_api_key() -> str:
    api_key = os.environ.get("TINKER_API_KEY")
    if api_key:
        return api_key

    try:
        result = subprocess.run(
            ["zsh", "-lc", "printf %s \"${TINKER_API_KEY-}\""],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""

    return result.stdout.strip()


BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = _load_api_key()
BASE_MODEL = os.environ.get("TINKER_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
LORA_RANK = int(os.environ.get("TINKER_LORA_RANK", "8"))
LEARNING_RATE = float(os.environ.get("TINKER_LEARNING_RATE", "1e-4"))

CREATE_TIMEOUT_S = float(os.environ.get("TINKER_CREATE_MODEL_TIMEOUT_S", "3600"))
FWDBWD_TIMEOUT_S = float(os.environ.get("TINKER_FORWARD_BACKWARD_TIMEOUT_S", "3600"))
OPTIM_TIMEOUT_S = float(os.environ.get("TINKER_OPTIM_STEP_TIMEOUT_S", "3600"))
REQUEST_TIMEOUT_S = float(os.environ.get("TINKER_REQUEST_TIMEOUT_S", "60"))
POLL_REQUEST_TIMEOUT_S = float(os.environ.get("TINKER_POLL_REQUEST_TIMEOUT_S", "30"))
POLL_INTERVAL_S = float(os.environ.get("TINKER_POLL_INTERVAL_S", "2"))


def _headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY} if API_KEY else {}


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _post_json(path: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    resp = requests.post(f"{BASE_URL}{path}", headers=_headers(), json=payload, timeout=timeout_s)
    if resp.status_code != 200:
        raise RuntimeError(f"POST {path} -> {resp.status_code}: {resp.text[:600]!r}")
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"POST {path} returned non-dict JSON: {type(data)}")
    return data


def _delete_model(model_id: str) -> None:
    try:
        requests.delete(f"{BASE_URL}/api/v1/models/{model_id}", headers=_headers(), timeout=REQUEST_TIMEOUT_S)
    except Exception:
        pass


def _poll_future(request_id: str, *, timeout_s: float) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = requests.post(
            f"{BASE_URL}/api/v1/retrieve_future",
            headers=_headers(),
            json={"request_id": request_id},
            timeout=POLL_REQUEST_TIMEOUT_S,
        )
        if resp.status_code == 408:
            time.sleep(POLL_INTERVAL_S)
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"retrieve_future({request_id}) -> {resp.status_code}: {resp.text[:600]!r}")
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


def _require_metrics(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError(f"result missing metrics dict: {result!r}")
    return metrics


def _require_metric(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    if value is None:
        raise RuntimeError(f"metrics missing {key!r}: {metrics!r}")
    if not isinstance(value, (int, float)):
        raise RuntimeError(f"metrics[{key!r}] is not numeric: {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise RuntimeError(f"metrics[{key!r}] is non-finite: {value!r}")
    return value


def _extract_logprobs(result: dict[str, Any]) -> list[float]:
    outputs = result.get("loss_fn_outputs")
    if not isinstance(outputs, list) or not outputs or not isinstance(outputs[0], dict):
        raise RuntimeError(f"missing loss_fn_outputs in result: {result!r}")
    logprobs_obj = outputs[0].get("logprobs")
    if isinstance(logprobs_obj, dict):
        data = logprobs_obj.get("data")
        if not isinstance(data, list):
            raise RuntimeError(f"logprobs.data missing/invalid: {logprobs_obj!r}")
        return [float(x) for x in data]
    if isinstance(logprobs_obj, list):
        return [float(x) for x in logprobs_obj]
    raise RuntimeError(f"unexpected logprobs payload: {logprobs_obj!r}")


def _assert_close(name: str, actual: float, expected: float, *, atol: float = 1e-3, rtol: float = 1e-3) -> None:
    diff = abs(actual - expected)
    limit = max(atol, rtol * max(abs(actual), abs(expected), 1.0))
    if diff > limit:
        raise RuntimeError(
            f"{name} mismatch: actual={actual:.6f} expected={expected:.6f} abs_diff={diff:.6f} limit={limit:.6f}"
        )


def _create_model() -> tuple[str, str]:
    session_id = f"repro-343-{uuid.uuid4().hex[:8]}"
    created = _post_json(
        "/api/v1/create_model",
        {
            "session_id": session_id,
            "model_seq_id": 0,
            "base_model": BASE_MODEL,
            "lora_config": {"rank": LORA_RANK},
            "learning_rate": LEARNING_RATE,
            "user_metadata": {"issue": 343},
        },
        timeout_s=REQUEST_TIMEOUT_S,
    )
    created = _await_maybe_async(created, timeout_s=CREATE_TIMEOUT_S)
    if "error" in created:
        raise RuntimeError(f"create_model failed: {created.get('error')!r}")
    model_id = created.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError(f"create_model missing model_id: {created!r}")
    backend = created.get("backend")
    print(f"created model_id={model_id} backend={backend} base_model={BASE_MODEL}", flush=True)
    return model_id, str(backend)


def _forward(model_id: str, datum: dict[str, Any]) -> dict[str, Any]:
    result = _post_json(
        "/api/v1/forward",
        {
            "model_id": model_id,
            "forward_input": {
                "data": [datum],
                "loss_fn": "cross_entropy",
                "loss_fn_config": {},
            },
        },
        timeout_s=REQUEST_TIMEOUT_S,
    )
    result = _await_maybe_async(result, timeout_s=FWDBWD_TIMEOUT_S)
    if "error" in result:
        raise RuntimeError(f"forward failed: {result.get('error')!r}")
    return result


def _forward_backward(model_id: str, datum: dict[str, Any], *, loss_fn: str) -> dict[str, Any]:
    result = _post_json(
        "/api/v1/forward_backward",
        {
            "model_id": model_id,
            "forward_backward_input": {"data": [datum], "loss_fn": loss_fn, "loss_fn_config": {}},
        },
        timeout_s=REQUEST_TIMEOUT_S,
    )
    result = _await_maybe_async(result, timeout_s=FWDBWD_TIMEOUT_S)
    if "error" in result:
        raise RuntimeError(f"forward_backward({loss_fn}) failed: {result.get('error')!r}")
    return result


def _optim_step(model_id: str) -> dict[str, Any]:
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
    result = _await_maybe_async(result, timeout_s=OPTIM_TIMEOUT_S)
    if "error" in result:
        raise RuntimeError(f"optim_step failed: {result.get('error')!r}")
    return result


def _make_cross_entropy_datum() -> tuple[dict[str, Any], list[float]]:
    tokens = [10, 11, 12, 13, 14, 15]
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = [1.0] * len(target_tokens)
    datum = {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": input_tokens}]},
        "loss_fn_inputs": {
            "target_tokens": {"data": target_tokens, "shape": [len(target_tokens)], "dtype": "int64"},
            "weights": {"data": weights, "shape": [len(weights)], "dtype": "float32"},
        },
    }
    return datum, weights


def _make_importance_sampling_datum() -> tuple[dict[str, Any], list[float], list[float], list[float]]:
    tokens = [20, 21, 22, 23, 24, 25]
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = [1.0] * len(target_tokens)
    old_logprobs = [0.0] * len(target_tokens)
    advantages = [0.0, 1.0, -2.0, 0.5, 0.0]
    datum = {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": input_tokens}]},
        "loss_fn_inputs": {
            "target_tokens": {"data": target_tokens, "shape": [len(target_tokens)], "dtype": "int64"},
            "weights": {"data": weights, "shape": [len(weights)], "dtype": "float32"},
            "logprobs": {"data": old_logprobs, "shape": [len(old_logprobs)], "dtype": "float32"},
            "advantages": {"data": advantages, "shape": [len(advantages)], "dtype": "float32"},
        },
    }
    return datum, weights, old_logprobs, advantages


def _verify_cross_entropy(model_id: str) -> None:
    datum, weights = _make_cross_entropy_datum()
    result = _forward_backward(model_id, datum, loss_fn="cross_entropy")
    metrics = _require_metrics(result)

    loss_sum = _require_metric(metrics, "loss:sum")
    loss_mean = _require_metric(metrics, "loss:mean")
    num_tokens = _require_metric(metrics, "num_tokens:sum")
    logprobs = _extract_logprobs(result)

    expected_sum = -sum(lp * wt for lp, wt in zip(logprobs, weights))
    _assert_close("cross_entropy loss:sum", loss_sum, expected_sum)

    if num_tokens <= 0:
        raise RuntimeError(f"cross_entropy returned invalid num_tokens:sum={num_tokens}")
    if loss_mean == 0.0:
        raise RuntimeError(f"cross_entropy returned suspicious loss:mean=0.0 metrics={metrics!r}")

    # Verify loss:sum / num_tokens:sum ≈ loss:mean (contract consistency)
    expected_mean = loss_sum / num_tokens
    _assert_close("cross_entropy loss:sum/num_tokens vs loss:mean", loss_mean, expected_mean)

    print(
        "cross_entropy ok "
        f"loss_sum={loss_sum:.6f} expected_sum={expected_sum:.6f} loss_mean={loss_mean:.6f} num_tokens={num_tokens:.0f}",
        flush=True,
    )


def _verify_importance_sampling(model_id: str) -> None:
    datum, weights, old_logprobs, advantages = _make_importance_sampling_datum()
    result = _forward_backward(model_id, datum, loss_fn="importance_sampling")
    metrics = _require_metrics(result)

    loss_sum = _require_metric(metrics, "loss:sum")
    loss_mean = _require_metric(metrics, "loss:mean")
    new_logprobs = _extract_logprobs(result)

    expected_sum = 0.0
    for lp_new, lp_old, adv, wt in zip(new_logprobs, old_logprobs, advantages, weights):
        log_ratio = max(-20.0, min(20.0, lp_new - lp_old))
        ratio = math.exp(log_ratio)
        expected_sum += -(ratio * adv * wt)
    _assert_close("importance_sampling loss:sum", loss_sum, expected_sum, atol=1e-2, rtol=1e-2)

    if not math.isfinite(loss_mean):
        raise RuntimeError(f"importance_sampling returned non-finite loss:mean={loss_mean!r}")

    print(
        "importance_sampling ok "
        f"loss_sum={loss_sum:.6f} expected_sum={expected_sum:.6f} loss_mean={loss_mean:.6f}",
        flush=True,
    )

    optim_result = _optim_step(model_id)
    optim_metrics = _require_metrics(optim_result)
    grad_norm = optim_metrics.get("grad_norm")
    if grad_norm is None:
        grad_norm = optim_metrics.get("grad_norm:last")
    if grad_norm is None or not isinstance(grad_norm, (int, float)) or not math.isfinite(float(grad_norm)):
        raise RuntimeError(f"optim_step missing finite grad_norm metric: {optim_metrics!r}")
    step_value = optim_metrics.get("step")
    print(f"optim_step ok grad_norm={float(grad_norm):.6f} step={step_value!r}", flush=True)


def _verify_forward(model_id: str) -> None:
    """Verify forward (read-only) returns numerically consistent loss metrics."""
    datum, weights = _make_cross_entropy_datum()
    result = _forward(model_id, datum)
    metrics = _require_metrics(result)

    loss_sum = _require_metric(metrics, "loss:sum")
    loss_mean = _require_metric(metrics, "loss:mean")
    num_tokens = _require_metric(metrics, "num_tokens:sum")
    logprobs = _extract_logprobs(result)

    expected_sum = -sum(lp * wt for lp, wt in zip(logprobs, weights))
    _assert_close("forward loss:sum", loss_sum, expected_sum)

    if num_tokens <= 0:
        raise RuntimeError(f"forward returned invalid num_tokens:sum={num_tokens}")
    expected_mean = loss_sum / num_tokens
    _assert_close("forward loss:sum/num_tokens vs loss:mean", loss_mean, expected_mean)

    print(
        "forward ok "
        f"loss_sum={loss_sum:.6f} expected_sum={expected_sum:.6f} "
        f"loss_mean={loss_mean:.6f} num_tokens={num_tokens:.0f}",
        flush=True,
    )


def main() -> int:
    model_id: str | None = None
    try:
        model_id, _backend = _create_model()
        _verify_cross_entropy(model_id)
        _verify_forward(model_id)
        _verify_importance_sampling(model_id)
        print("PASS issue #343 reproduction/verification", flush=True)
        return 0
    except Exception as e:
        return _fail(str(e))
    finally:
        if model_id:
            _delete_model(model_id)


if __name__ == "__main__":
    raise SystemExit(main())
