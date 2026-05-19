#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any

import requests


BASE_URL = os.environ.get("MINT_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("MINT_API_KEY", "dummy")
BASE_MODEL = os.environ.get("MINT_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
LORA_RANK = int(os.environ.get("MINT_LORA_RANK", "8"))
LEARNING_RATE = float(os.environ.get("MINT_LEARNING_RATE", "1e-4"))
WARMUP_STEPS = int(os.environ.get("MINT_WARMUP_STEPS", "5"))
COMPARE_STEPS = int(os.environ.get("MINT_COMPARE_STEPS", "3"))
POLL_TIMEOUT_S = float(os.environ.get("MINT_POLL_TIMEOUT_S", "3600"))
POLL_REQUEST_TIMEOUT_S = float(os.environ.get("MINT_POLL_REQUEST_TIMEOUT_S", "180"))
CREATE_TIMEOUT_S = float(os.environ.get("MINT_CREATE_TIMEOUT_S", "3600"))
SAVE_TIMEOUT_S = float(os.environ.get("MINT_SAVE_TIMEOUT_S", "3600"))
RESUME_TIMEOUT_S = float(os.environ.get("MINT_RESUME_TIMEOUT_S", "3600"))


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}


def _fail(msg: str) -> int:
    print(f"FAIL: {msg}", file=sys.stderr)
    return 1


def _post_json(path: str, payload: dict[str, Any], *, timeout_s: float = 60.0) -> dict[str, Any]:
    resp = requests.post(f"{BASE_URL}{path}", json=payload, headers=_headers(), timeout=timeout_s)
    if resp.status_code != 200:
        raise RuntimeError(f"POST {path} -> {resp.status_code}: {resp.text[:800]!r}")
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"POST {path} returned non-dict JSON: {type(data)}")
    return data


def _delete(path: str, *, timeout_s: float = 60.0) -> None:
    resp = requests.delete(f"{BASE_URL}{path}", headers=_headers(), timeout=timeout_s)
    if resp.status_code != 200:
        raise RuntimeError(f"DELETE {path} -> {resp.status_code}: {resp.text[:800]!r}")


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


@dataclass
class TrainStep:
    index: int
    loss: float


def _get_loss(result: dict[str, Any]) -> float:
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError(f"forward_backward missing metrics: {result!r}")
    value = metrics.get("loss:mean")
    if value is None:
        raise RuntimeError(f"forward_backward missing metrics['loss:mean']: {result!r}")
    return float(value)


def _make_training_batch() -> list[dict[str, Any]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    pairs = [
        ("Translate to French: hello -> ", "bonjour"),
        ("Translate to French: goodbye -> ", "au revoir"),
        ("Translate to French: thank you -> ", "merci"),
        ("Translate to French: please -> ", "s'il vous plait"),
    ]
    data: list[dict[str, Any]] = []
    for prompt, target in pairs:
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        target_tokens = tokenizer.encode(target, add_special_tokens=False)
        full_tokens = prompt_tokens + target_tokens
        loss_mask = [0.0] * len(prompt_tokens) + [1.0] * len(target_tokens)
        data.append(
            {
                "model_input": {"chunks": [{"tokens": full_tokens[:-1], "type": "encoded_text"}]},
                "loss_fn_inputs": {
                    "target_tokens": {
                        "data": full_tokens[1:],
                        "shape": [len(full_tokens) - 1],
                        "dtype": "int64",
                    },
                    "loss_mask": {
                        "data": loss_mask[1:],
                        "shape": [len(loss_mask) - 1],
                        "dtype": "float32",
                    },
                },
            }
        )
    return data


def _create_model(session_id: str, model_seq_id: int) -> tuple[str, str]:
    created = _post_json(
        "/api/v1/create_model",
        {
            "session_id": session_id,
            "model_seq_id": model_seq_id,
            "base_model": BASE_MODEL,
            "lora_config": {"rank": LORA_RANK},
            "learning_rate": LEARNING_RATE,
        },
        timeout_s=60.0,
    )
    created = _await_maybe_async(created, timeout_s=CREATE_TIMEOUT_S)
    model_id = created.get("model_id")
    backend = created.get("backend")
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError(f"create_model missing model_id: {created!r}")
    if backend != "megatron":
        raise RuntimeError(f"expected backend='megatron', got {backend!r} for {BASE_MODEL!r}")
    return model_id, backend


def _train_step(model_id: str, batch: list[dict[str, Any]], step_idx: int) -> TrainStep:
    fb = _await_maybe_async(
        _post_json(
            "/api/v1/forward_backward",
            {
                "model_id": model_id,
                "forward_backward_input": {"data": batch, "loss_fn": "cross_entropy"},
            },
            timeout_s=120.0,
        ),
        timeout_s=POLL_TIMEOUT_S,
    )
    loss = _get_loss(fb)
    _await_maybe_async(
        _post_json(
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
            timeout_s=60.0,
        ),
        timeout_s=POLL_TIMEOUT_S,
    )
    print(f"step={step_idx} loss={loss:.6f}", flush=True)
    return TrainStep(index=step_idx, loss=loss)


def _save_state(model_id: str, checkpoint_name: str) -> str:
    saved = _await_maybe_async(
        _post_json("/api/v1/save_state", {"model_id": model_id, "path": checkpoint_name}, timeout_s=60.0),
        timeout_s=SAVE_TIMEOUT_S,
    )
    path = saved.get("path")
    if not isinstance(path, str) or not path:
        raise RuntimeError(f"save_state missing checkpoint path: {saved!r}")
    print(f"checkpoint={path}", flush=True)
    return path


def _resume_from_state(session_id: str, model_seq_id: int, state_path: str) -> str:
    resumed = _post_json(
        "/api/v1/create_model_from_state",
        {
            "session_id": session_id,
            "model_seq_id": model_seq_id,
            "base_model": BASE_MODEL,
            "state_path": state_path,
            "lora_config": {"rank": LORA_RANK},
            "load_optimizer": True,
            "user_metadata": {"issue": 283},
        },
        timeout_s=60.0,
    )
    resumed = _await_maybe_async(resumed, timeout_s=RESUME_TIMEOUT_S)
    model_id = resumed.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError(f"create_model_from_state missing model_id: {resumed!r}")
    return model_id


def _compare_losses(uninterrupted: list[TrainStep], resumed: list[TrainStep]) -> tuple[bool, str]:
    if len(uninterrupted) != len(resumed):
        return False, f"length mismatch: uninterrupted={len(uninterrupted)} resumed={len(resumed)}"

    diffs = []
    ratios = []
    for a, b in zip(uninterrupted, resumed):
        diff = abs(a.loss - b.loss)
        denom = max(abs(a.loss), 1e-8)
        ratio = diff / denom
        diffs.append(diff)
        ratios.append(ratio)
        print(
            f"compare step_ref={a.index} uninterrupted={a.loss:.6f} resumed={b.loss:.6f} "
            f"abs_diff={diff:.6f} rel_diff={ratio:.6f}",
            flush=True,
        )

    max_abs = max(diffs)
    max_rel = max(ratios)
    mean_rel = sum(ratios) / len(ratios)

    # Acceptance: resumed trajectory should track uninterrupted continuation closely.
    # A resume bug in optimizer state produces a large first-step spike, not small noise.
    ok = max_rel <= 0.15 and mean_rel <= 0.10
    summary = f"max_abs={max_abs:.6f} max_rel={max_rel:.6f} mean_rel={mean_rel:.6f}"
    return ok, summary


def _compare_resume_to_presave(last_presave: TrainStep, resumed: list[TrainStep]) -> tuple[bool, str]:
    if not resumed:
        return False, "no resumed steps recorded"

    first = resumed[0].loss
    max_resumed = max(step.loss for step in resumed)
    final = resumed[-1].loss
    first_ratio = first / max(last_presave.loss, 1e-8)
    max_ratio = max_resumed / max(last_presave.loss, 1e-8)

    for step in resumed:
        rel = step.loss / max(last_presave.loss, 1e-8)
        print(
            f"resume_compare presave_step={last_presave.index} presave_loss={last_presave.loss:.6f} "
            f"resumed_step={step.index} resumed_loss={step.loss:.6f} rel_to_presave={rel:.6f}",
            flush=True,
        )

    # Optimizer restore should not produce a large loss spike relative to the
    # last pre-save step, and the resumed trajectory should keep descending.
    ok = max_ratio <= 1.25 and final <= first
    summary = (
        f"presave_loss={last_presave.loss:.6f} first_resumed={first:.6f} "
        f"final_resumed={final:.6f} first_ratio={first_ratio:.6f} max_ratio={max_ratio:.6f}"
    )
    return ok, summary


def main() -> int:
    batch = _make_training_batch()
    suffix = uuid.uuid4().hex[:8]
    session_a = f"issue283-a-{suffix}"
    session_b = f"issue283-b-{suffix}"
    model_a: str | None = None
    model_b: str | None = None
    try:
        model_a, _ = _create_model(session_a, 0)
        last_presave: TrainStep | None = None
        for step_idx in range(1, WARMUP_STEPS + 1):
            last_presave = _train_step(model_a, batch, step_idx)

        if last_presave is None:
            raise RuntimeError("no pre-save training steps ran")

        checkpoint_path = _save_state(model_a, f"issue283-{suffix}")
        print(
            f"presave_reference step={last_presave.index} loss={last_presave.loss:.6f}",
            flush=True,
        )
        model_b = _resume_from_state(session_b, 0, checkpoint_path)
        resumed: list[TrainStep] = []
        for compare_idx in range(1, COMPARE_STEPS + 1):
            resumed.append(_train_step(model_b, batch, compare_idx))

        ok, summary = _compare_resume_to_presave(last_presave, resumed)
        if not ok:
            return _fail(f"resume trajectory diverged from pre-save behavior: {summary}")

        print(f"PASS: resumed trajectory stayed aligned with pre-save loss ({summary})", flush=True)
        return 0
    except Exception as exc:
        return _fail(str(exc))
    finally:
        for model_id in (model_b, model_a):
            if model_id:
                try:
                    _delete(f"/api/v1/models/{model_id}", timeout_s=120.0)
                except Exception as exc:
                    print(f"WARN: cleanup failed for {model_id}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
