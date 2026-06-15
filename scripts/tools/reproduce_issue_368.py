#!/usr/bin/env python3
"""Reproduce and validate issue 368 against a live server.

This script exercises the two service-side behaviors that caused the incident:
1. `current_step` in detached training state must advance after a real optim step.
2. A training session must be auto-terminated after heartbeat loss.

The script talks to the HTTP API directly so it can run in isolated environments
without depending on a specific Mint/Tinker SDK install.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

import requests
from transformers import AutoTokenizer


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_BASE_URL = "http://127.0.0.1:8001"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


@dataclass
class Evidence:
    base_url: str
    base_model: str
    ray_address: str
    ray_namespace: str
    session_id: str
    model_id: str
    create_model_request_id: str
    forward_backward_request_id: str
    optim_step_request_id: str
    optim_step_metrics: dict[str, Any]
    model_info_after_step: dict[str, Any]
    detached_training_session_after_step: dict[str, Any]
    stale_after_s: float
    cleanup_wait_s: float
    synthetic_future_request_id: str
    synthetic_future_status: str
    synthetic_future_error: str | None
    training_run_deleted: bool
    model_deleted: bool


def _coalesce(*values: str | None) -> str | None:
    for value in values:
        if value:
            return value
    return None


def _headers(args: argparse.Namespace) -> dict[str, str]:
    api_key = _coalesce(args.api_key, os.environ.get("MINT_API_KEY"), os.environ.get("MINT_API_KEY"))
    return {"X-API-Key": api_key} if api_key else {}


def _base_url(args: argparse.Namespace) -> str:
    return (
        _coalesce(args.base_url, os.environ.get("MINT_BASE_URL"), os.environ.get("MINT_BASE_URL"), DEFAULT_BASE_URL)
        .rstrip("/")
    )


def _get(url: str, *, headers: dict[str, str], timeout_s: float) -> requests.Response:
    response = requests.get(url, headers=headers, timeout=timeout_s)
    return response


def _post(url: str, *, headers: dict[str, str], payload: dict[str, Any], timeout_s: float) -> requests.Response:
    response = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    return response


def _configure_ray_access(*, ray_address: str, ray_namespace: str) -> None:
    os.environ["MINT_RAY_GCS_ADDRESS"] = str(ray_address)
    os.environ["MINT_RAY_NAMESPACE"] = str(ray_namespace)
    os.environ["MINT_RAY_NAMESPACE"] = str(ray_namespace)


def _ensure_ray_initialized(*, ray_address: str, ray_namespace: str) -> None:
    _configure_ray_access(ray_address=ray_address, ray_namespace=ray_namespace)
    import ray

    if ray.is_initialized():
        ray.shutdown()
    ray.init(address=str(ray_address), namespace=str(ray_namespace), ignore_reinit_error=True)
    try:
        from mint_server.backend.stores.task_state_store import task_state_futures

        task_state_futures._task_state._reset_ray_actor()
    except Exception:
        pass


def _poll_future(
    base_url: str,
    request_id: str,
    *,
    headers: dict[str, str],
    timeout_s: float,
    poll_interval_s: float = 2.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while True:
        if time.monotonic() > deadline:
            raise TimeoutError(f"timeout while waiting for request_id={request_id}")
        response = _post(
            f"{base_url}/api/v1/retrieve_future",
            headers=headers,
            payload={"request_id": request_id},
            timeout_s=max(30.0, poll_interval_s + 5.0),
        )
        if response.status_code == 408:
            time.sleep(poll_interval_s)
            continue
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and payload.get("error"):
            raise RuntimeError(f"request_id={request_id} failed: {payload['error']}")
        return payload


def _build_cross_entropy_datum(tokenizer: Any, *, max_len: int) -> dict[str, Any]:
    prompt = "Question: What is 13 + 29?\nAnswer:"
    completion = " 42"

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise RuntimeError("tokenizer is missing eos_token_id")
    completion_tokens = completion_tokens + [int(eos_id)]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else int(eos_id)

    tokens = prompt_tokens + completion_tokens
    weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)

    if len(tokens) < max_len:
        pad = max_len - len(tokens)
        tokens = tokens + [pad_id] * pad
        weights = weights + [0.0] * pad
    else:
        tokens = tokens[:max_len]
        weights = weights[:max_len]

    return {
        "model_input": {
            "chunks": [
                {
                    "type": "encoded_text",
                    "tokens": tokens[:-1],
                }
            ]
        },
        "loss_fn_inputs": {
            "target_tokens": {
                "data": tokens[1:],
                "shape": [len(tokens) - 1],
                "dtype": "int64",
            },
            "weights": {
                "data": weights[1:],
                "shape": [len(weights) - 1],
                "dtype": "float32",
            },
        },
    }


def _wait_for_model_absence(
    base_url: str,
    model_id: str,
    *,
    headers: dict[str, str],
    stale_after_s: float,
    timeout_s: float,
) -> tuple[bool, bool, float]:
    deadline = time.monotonic() + timeout_s
    while True:
        model_response = _get(f"{base_url}/api/v1/models/{model_id}", headers=headers, timeout_s=10.0)
        run_response = _get(f"{base_url}/api/v1/training_runs/{model_id}", headers=headers, timeout_s=10.0)
        if model_response.status_code == 404 and run_response.status_code == 404:
            waited = timeout_s - max(0.0, deadline - time.monotonic())
            return True, True, waited
        if time.monotonic() > deadline:
            return model_response.status_code == 404, run_response.status_code == 404, timeout_s
        time.sleep(max(1.0, min(5.0, stale_after_s / 3.0)))


def _get_detached_training_session_info(model_id: str, *, ray_address: str, ray_namespace: str) -> dict[str, Any]:
    _ensure_ray_initialized(ray_address=ray_address, ray_namespace=ray_namespace)
    from mint_server.backend.stores.training_session_store import get_training_session_info

    info = get_training_session_info(model_id)
    if not isinstance(info, dict):
        raise AssertionError(f"detached training session info missing for model_id={model_id}")
    return info


def _create_synthetic_pending_training_future(model_id: str, *, ray_address: str, ray_namespace: str) -> str:
    _ensure_ray_initialized(ray_address=ray_address, ray_namespace=ray_namespace)
    from mint_server.backend.stores.task_state_store import task_state_futures

    request_id = f"issue368-synthetic-{uuid.uuid4().hex}"
    asyncio.run(task_state_futures.async_create_with_id(request_id))
    asyncio.run(
        task_state_futures.async_mark_running(
            request_id,
            meta={
                "op": "training.optim_step",
                "model_id": model_id,
                "queue_state": "running",
                "queue_state_reason": "issue368_synthetic_pending_future",
            },
        )
    )
    return request_id


def _wait_for_synthetic_future_failure(
    request_id: str,
    *,
    ray_address: str,
    ray_namespace: str,
    timeout_s: float,
) -> tuple[str, str | None]:
    _ensure_ray_initialized(ray_address=ray_address, ray_namespace=ray_namespace)
    from mint_server.backend.stores.task_state_store import FutureStatus, task_state_futures

    deadline = time.monotonic() + timeout_s
    while True:
        status = asyncio.run(task_state_futures.async_get_status(request_id))
        if status == FutureStatus.FAILED:
            return status.value, asyncio.run(task_state_futures.async_get_error(request_id))
        if time.monotonic() > deadline:
            raise TimeoutError(f"synthetic future {request_id} did not fail within {timeout_s}s (status={status.value})")
        time.sleep(1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--future-timeout-s", type=float, default=900.0)
    parser.add_argument("--stale-after-s", type=float, required=True)
    parser.add_argument("--cleanup-timeout-s", type=float, default=90.0)
    parser.add_argument("--ray-address", default=_coalesce(os.environ.get("MINT_RAY_GCS_ADDRESS"), "192.168.36.5:26379"))
    parser.add_argument("--ray-namespace", default=_coalesce(os.environ.get("MINT_RAY_NAMESPACE"), "mint_local_nolanho"))
    parser.add_argument("--report-json", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = _base_url(args)
    headers = _headers(args)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    datum = _build_cross_entropy_datum(tokenizer, max_len=int(args.max_len))

    session_response = _post(
        f"{base_url}/api/v1/create_session",
        headers=headers,
        payload={"tags": ["issue-368"], "user_metadata": {"issue": 368}, "sdk_version": "reproduce_issue_368"},
        timeout_s=30.0,
    )
    session_response.raise_for_status()
    session_id = session_response.json()["session_id"]

    create_model_response = _post(
        f"{base_url}/api/v1/create_model",
        headers=headers,
        payload={
            "session_id": session_id,
            "model_seq_id": 0,
            "base_model": args.base_model,
            "lora_config": {
                "rank": int(args.lora_rank),
                "train_attn": True,
                "train_mlp": True,
                "train_unembed": True,
            },
            "user_metadata": {"issue": 368, "script": "reproduce_issue_368"},
            "type": "create_model",
        },
        timeout_s=60.0,
    )
    create_model_response.raise_for_status()
    create_model_request_id = create_model_response.json()["request_id"]
    create_model_result = _poll_future(
        base_url,
        create_model_request_id,
        headers=headers,
        timeout_s=float(args.future_timeout_s),
    )
    model_id = create_model_result["model_id"]

    # Prime the heartbeat store, then stop heartbeating after the training step.
    heartbeat_response = _post(
        f"{base_url}/api/v1/session_heartbeat",
        headers=headers,
        payload={"session_id": session_id, "type": "session_heartbeat"},
        timeout_s=10.0,
    )
    heartbeat_response.raise_for_status()

    forward_backward_response = _post(
        f"{base_url}/api/v1/forward_backward",
        headers=headers,
        payload={
            "model_id": model_id,
            "seq_id": 0,
            "forward_backward_input": {
                "data": [datum],
                "loss_fn": "cross_entropy",
            },
        },
        timeout_s=60.0,
    )
    forward_backward_response.raise_for_status()
    forward_backward_request_id = forward_backward_response.json()["request_id"]
    _poll_future(
        base_url,
        forward_backward_request_id,
        headers=headers,
        timeout_s=float(args.future_timeout_s),
    )

    optim_step_response = _post(
        f"{base_url}/api/v1/optim_step",
        headers=headers,
        payload={
            "model_id": model_id,
            "seq_id": 1,
            "adam_params": {
                "learning_rate": float(args.learning_rate),
                "beta1": 0.9,
                "beta2": 0.95,
                "eps": 1e-12,
            },
            "type": "optim_step",
        },
        timeout_s=60.0,
    )
    optim_step_response.raise_for_status()
    optim_step_request_id = optim_step_response.json()["request_id"]
    optim_step_result = _poll_future(
        base_url,
        optim_step_request_id,
        headers=headers,
        timeout_s=float(args.future_timeout_s),
    )

    model_info_response = _get(f"{base_url}/api/v1/models/{model_id}", headers=headers, timeout_s=30.0)
    model_info_response.raise_for_status()
    model_info = model_info_response.json()
    current_step = int(model_info.get("current_step", -1))
    if current_step < 1:
        raise AssertionError(f"expected current_step >= 1 after optim_step, got {current_step}")
    detached_info = _get_detached_training_session_info(
        model_id,
        ray_address=str(args.ray_address),
        ray_namespace=str(args.ray_namespace),
    )
    detached_current_step = int(detached_info.get("current_step", -1))
    if detached_current_step < 1:
        raise AssertionError(
            f"expected detached current_step >= 1 after optim_step, got {detached_current_step}"
        )

    synthetic_future_request_id = _create_synthetic_pending_training_future(
        model_id,
        ray_address=str(args.ray_address),
        ray_namespace=str(args.ray_namespace),
    )

    model_deleted, run_deleted, cleanup_wait_s = _wait_for_model_absence(
        base_url,
        model_id,
        headers=headers,
        stale_after_s=float(args.stale_after_s),
        timeout_s=float(args.cleanup_timeout_s),
    )
    if not model_deleted or not run_deleted:
        raise AssertionError(
            "stale cleanup did not delete training state in time: "
            f"model_deleted={model_deleted} run_deleted={run_deleted}"
        )
    synthetic_future_status, synthetic_future_error = _wait_for_synthetic_future_failure(
        synthetic_future_request_id,
        ray_address=str(args.ray_address),
        ray_namespace=str(args.ray_namespace),
        timeout_s=float(args.cleanup_timeout_s),
    )
    if "stale heartbeat" not in str(synthetic_future_error or ""):
        raise AssertionError(
            "synthetic future failed for the wrong reason: "
            f"request_id={synthetic_future_request_id} error={synthetic_future_error!r}"
        )

    evidence = Evidence(
        base_url=base_url,
        base_model=args.base_model,
        ray_address=str(args.ray_address),
        ray_namespace=str(args.ray_namespace),
        session_id=session_id,
        model_id=model_id,
        create_model_request_id=create_model_request_id,
        forward_backward_request_id=forward_backward_request_id,
        optim_step_request_id=optim_step_request_id,
        optim_step_metrics=optim_step_result.get("metrics") or {},
        model_info_after_step=model_info,
        detached_training_session_after_step=detached_info,
        stale_after_s=float(args.stale_after_s),
        cleanup_wait_s=float(cleanup_wait_s),
        synthetic_future_request_id=synthetic_future_request_id,
        synthetic_future_status=synthetic_future_status,
        synthetic_future_error=synthetic_future_error,
        training_run_deleted=bool(run_deleted),
        model_deleted=bool(model_deleted),
    )

    payload = asdict(evidence)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
