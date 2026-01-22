#!/usr/bin/env python3
"""Issue #44 regression tools (client-side, run locally).

Subcommands:
- auto-swap: verify session swap preserves loss continuity
- concurrent: verify interleaved A/B training behaves like solo runs (coarse check)
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
import uuid
from typing import Any

import requests

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_SEQ_ID = 0


def _coalesce(*values: str | None) -> str | None:
    for v in values:
        if v:
            return v
    return None


def _base_url(args: argparse.Namespace) -> str:
    return (
        _coalesce(args.base_url, os.environ.get("TINKER_BASE_URL"), os.environ.get("MINT_BASE_URL"), DEFAULT_BASE_URL)
        .rstrip("/")
    )


def _headers(args: argparse.Namespace) -> dict[str, str]:
    api_key = _coalesce(args.api_key, os.environ.get("TINKER_API_KEY"), os.environ.get("MINT_API_KEY"))
    return {"X-API-Key": api_key} if api_key else {}


def _poll_future(base_url: str, headers: dict[str, str], request_id: str, timeout_s: float) -> dict:
    poll_url = f"{base_url}/api/v1/retrieve_future"
    start = time.time()

    while time.time() - start < timeout_s:
        resp = requests.post(poll_url, headers=headers, json={"request_id": request_id}, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 408:
            time.sleep(2)
            continue
        resp.raise_for_status()

    raise TimeoutError(f"Operation did not complete within {timeout_s}s")


def _create_model(base_url: str, headers: dict[str, str], *, session_id: str, base_model: str, lora_rank: int, timeout_s: float) -> str:
    url = f"{base_url}/api/v1/create_model"
    payload = {
        "session_id": session_id,
        "model_seq_id": MODEL_SEQ_ID,
        "base_model": base_model,
        "lora_config": {"rank": int(lora_rank)},
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    result = resp.json()
    if "request_id" in result:
        result = _poll_future(base_url, headers, result["request_id"], timeout_s=timeout_s)
    return result.get("model_id", f"{session_id}_{MODEL_SEQ_ID}")


def _forward_backward(base_url: str, headers: dict[str, str], *, model_id: str, data: list, loss_fn: str, timeout_s: float) -> dict:
    url = f"{base_url}/api/v1/forward_backward"
    payload = {"model_id": model_id, "forward_backward_input": {"data": data, "loss_fn": loss_fn}}
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    result = resp.json()
    if "request_id" in result:
        result = _poll_future(base_url, headers, result["request_id"], timeout_s=timeout_s)
    return result


def _optim_step(base_url: str, headers: dict[str, str], *, model_id: str, learning_rate: float, timeout_s: float) -> dict:
    url = f"{base_url}/api/v1/optim_step"
    payload = {"model_id": model_id, "adam_params": {"learning_rate": float(learning_rate)}}
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    result = resp.json()
    if "request_id" in result:
        result = _poll_future(base_url, headers, result["request_id"], timeout_s=timeout_s)
    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    a = sub.add_parser("auto-swap")
    a.add_argument("--base-url", default=None)
    a.add_argument("--api-key", default=None)
    a.add_argument("--model", default=DEFAULT_MODEL)
    a.add_argument("--lora-rank", type=int, default=8)
    a.add_argument("--steps", type=int, default=3)
    a.add_argument("--learning-rate", type=float, default=1e-4)
    a.add_argument("--timeout-s", type=float, default=900.0)
    a.add_argument("--max-loss-diff", type=float, default=0.5)

    c = sub.add_parser("concurrent")
    c.add_argument("--base-url", default=None)
    c.add_argument("--api-key", default=None)
    c.add_argument("--model", default=DEFAULT_MODEL)
    c.add_argument("--lora-rank", type=int, default=8)
    c.add_argument("--steps", type=int, default=5)
    c.add_argument("--batch-size", type=int, default=4)
    c.add_argument("--learning-rate", type=float, default=1e-4)
    c.add_argument("--timeout-s", type=float, default=900.0)

    return p.parse_args()


def _create_datum(prompt: str, response: str, tokenizer: Any) -> dict:
    full_text = f"{prompt}{response}"
    tokens = tokenizer.encode(full_text)
    prompt_tokens = tokenizer.encode(prompt)
    target_tokens = tokens[1:] + [tokenizer.eos_token_id or 0]
    loss_mask = [0.0] * len(prompt_tokens) + [1.0] * (len(tokens) - len(prompt_tokens))

    return {
        "model_input": {"chunks": [{"tokens": tokens, "type": "encoded_text"}]},
        "loss_fn_inputs": {
            "target_tokens": {"data": target_tokens, "shape": [len(target_tokens)], "dtype": "int64"},
            "loss_mask": {"data": loss_mask, "shape": [len(loss_mask)], "dtype": "float32"},
        },
    }


def _auto_swap(args: argparse.Namespace) -> int:
    base_url = _base_url(args)
    headers = _headers(args)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    datum_a = _create_datum("What is 2+2? ", "The answer is 4.", tokenizer)
    datum_b = _create_datum("What is 3+3? ", "The answer is 6.", tokenizer)

    sess_a = f"issue44-A-{uuid.uuid4().hex[:8]}"
    sess_b = f"issue44-B-{uuid.uuid4().hex[:8]}"

    model_a = _create_model(base_url, headers, session_id=sess_a, base_model=args.model, lora_rank=args.lora_rank, timeout_s=args.timeout_s)
    for _ in range(int(args.steps)):
        _forward_backward(base_url, headers, model_id=model_a, data=[datum_a], loss_fn="cross_entropy", timeout_s=args.timeout_s)
        _optim_step(base_url, headers, model_id=model_a, learning_rate=args.learning_rate, timeout_s=args.timeout_s)

    res_a = _forward_backward(base_url, headers, model_id=model_a, data=[datum_a], loss_fn="cross_entropy", timeout_s=args.timeout_s)
    loss_a = res_a.get("metrics", {}).get("loss:mean", 0.0)

    model_b = _create_model(base_url, headers, session_id=sess_b, base_model=args.model, lora_rank=args.lora_rank, timeout_s=args.timeout_s)
    for _ in range(int(args.steps)):
        _forward_backward(base_url, headers, model_id=model_b, data=[datum_b], loss_fn="cross_entropy", timeout_s=args.timeout_s)
        _optim_step(base_url, headers, model_id=model_b, learning_rate=args.learning_rate, timeout_s=args.timeout_s)

    res_a_restored = _forward_backward(base_url, headers, model_id=model_a, data=[datum_a], loss_fn="cross_entropy", timeout_s=args.timeout_s)
    loss_a_restored = res_a_restored.get("metrics", {}).get("loss:mean", 0.0)

    diff = abs(float(loss_a) - float(loss_a_restored))
    print(f"loss_after_training={loss_a:.6f} loss_after_restore={loss_a_restored:.6f} diff={diff:.6f}", flush=True)
    return 0 if diff < float(args.max_loss_diff) else 1


def _make_arithmetic_data(n: int, tokenizer: Any) -> list[dict]:
    data: list[dict] = []
    for _ in range(n):
        x, y = random.randint(0, 100), random.randint(0, 100)
        data.append(_create_datum(f"What is {x} + {y}? ", str(x + y), tokenizer))
    return data


def _make_countdown_data(n: int, tokenizer: Any) -> list[dict]:
    problems = [
        ([44, 19, 35], 98, "44 + 35 + 19"),
        ([10, 5, 2], 17, "10 + 5 + 2"),
        ([20, 8, 3], 15, "20 - 8 + 3"),
        ([50, 25, 10], 65, "50 + 25 - 10"),
        ([100, 50, 25], 75, "100 - 50 + 25"),
        ([30, 20, 10], 40, "30 + 20 - 10"),
        ([15, 7, 3], 25, "15 + 7 + 3"),
        ([80, 40, 20], 60, "80 - 40 + 20"),
        ([12, 6, 4], 14, "12 + 6 - 4"),
        ([90, 45, 15], 60, "90 - 45 + 15"),
    ]
    data: list[dict] = []
    for i in range(n):
        nums, target, solution = problems[i % len(problems)]
        data.append(_create_datum(f"Using the numbers {nums}, create an equation that equals {target}. ", f"<answer> {solution} = {target} </answer>", tokenizer))
    return data


def _train_solo(base_url: str, headers: dict[str, str], *, base_model: str, lora_rank: int, data: list[dict], steps: int, batch_size: int, lr: float, timeout_s: float) -> list[float]:
    session_id = f"issue44-solo-{uuid.uuid4().hex[:8]}"
    model_id = _create_model(base_url, headers, session_id=session_id, base_model=base_model, lora_rank=lora_rank, timeout_s=timeout_s)
    losses: list[float] = []
    for step in range(steps):
        batch = data[step * batch_size : (step + 1) * batch_size]
        res = _forward_backward(base_url, headers, model_id=model_id, data=batch, loss_fn="cross_entropy", timeout_s=timeout_s)
        _optim_step(base_url, headers, model_id=model_id, learning_rate=lr, timeout_s=timeout_s)
        loss = float(res.get("metrics", {}).get("loss:mean", 0.0))
        losses.append(loss)
    return losses


def _train_interleaved(base_url: str, headers: dict[str, str], *, base_model: str, lora_rank: int, data_a: list[dict], data_b: list[dict], steps: int, batch_size: int, lr: float, timeout_s: float) -> tuple[list[float], list[float]]:
    session_a = f"issue44-A-{uuid.uuid4().hex[:8]}"
    session_b = f"issue44-B-{uuid.uuid4().hex[:8]}"
    model_a = _create_model(base_url, headers, session_id=session_a, base_model=base_model, lora_rank=lora_rank, timeout_s=timeout_s)
    model_b = _create_model(base_url, headers, session_id=session_b, base_model=base_model, lora_rank=lora_rank, timeout_s=timeout_s)

    la: list[float] = []
    lb: list[float] = []
    for step in range(steps):
        ba = data_a[step * batch_size : (step + 1) * batch_size]
        bb = data_b[step * batch_size : (step + 1) * batch_size]
        ra = _forward_backward(base_url, headers, model_id=model_a, data=ba, loss_fn="cross_entropy", timeout_s=timeout_s)
        rb = _forward_backward(base_url, headers, model_id=model_b, data=bb, loss_fn="cross_entropy", timeout_s=timeout_s)
        _optim_step(base_url, headers, model_id=model_a, learning_rate=lr, timeout_s=timeout_s)
        _optim_step(base_url, headers, model_id=model_b, learning_rate=lr, timeout_s=timeout_s)
        la.append(float(ra.get("metrics", {}).get("loss:mean", 0.0)))
        lb.append(float(rb.get("metrics", {}).get("loss:mean", 0.0)))
    return la, lb


def _concurrent(args: argparse.Namespace) -> int:
    base_url = _base_url(args)
    headers = _headers(args)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    random.seed(42)

    data_a = _make_arithmetic_data(int(args.steps) * int(args.batch_size), tokenizer)
    data_b = _make_countdown_data(int(args.steps) * int(args.batch_size), tokenizer)

    solo_a = _train_solo(
        base_url,
        headers,
        base_model=args.model,
        lora_rank=args.lora_rank,
        data=data_a,
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.learning_rate),
        timeout_s=float(args.timeout_s),
    )
    solo_b = _train_solo(
        base_url,
        headers,
        base_model=args.model,
        lora_rank=args.lora_rank,
        data=data_b,
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.learning_rate),
        timeout_s=float(args.timeout_s),
    )
    inter_a, inter_b = _train_interleaved(
        base_url,
        headers,
        base_model=args.model,
        lora_rank=args.lora_rank,
        data_a=data_a,
        data_b=data_b,
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.learning_rate),
        timeout_s=float(args.timeout_s),
    )

    print(f"solo_a={solo_a}", flush=True)
    print(f"solo_b={solo_b}", flush=True)
    print(f"interleaved_a={inter_a}", flush=True)
    print(f"interleaved_b={inter_b}", flush=True)
    return 0


def main() -> int:
    args = _parse_args()
    cmd = args.cmd or "auto-swap"
    if cmd == "auto-swap":
        return _auto_swap(args)
    if cmd == "concurrent":
        return _concurrent(args)
    print(f"unknown cmd: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

