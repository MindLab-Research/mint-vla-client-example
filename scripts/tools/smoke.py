#!/usr/bin/env python3
"""Smoke checks (client-side, run locally).

Subcommands:
- service: basic HTTP health/session checks
- dense-train: minimal dense training loop (cross-entropy)
"""

from __future__ import annotations

import argparse
import datetime
import os
import random
import time
from typing import Any

import requests
from dotenv import load_dotenv


DEFAULT_BASE_URL = "http://localhost:8000"


def _ts() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _coalesce(*values: str | None) -> str | None:
    for v in values:
        if v:
            return v
    return None


def _load_env() -> None:
    load_dotenv()
    repo_root_env = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
    if os.path.exists(repo_root_env):
        load_dotenv(repo_root_env, override=False)


def _base_url(args: argparse.Namespace) -> str:
    return (
        _coalesce(args.base_url, os.environ.get("TINKER_BASE_URL"), os.environ.get("MINT_BASE_URL"), DEFAULT_BASE_URL)
        .rstrip("/")
    )


def _headers(args: argparse.Namespace) -> dict[str, str]:
    api_key = _coalesce(args.api_key, os.environ.get("TINKER_API_KEY"), os.environ.get("MINT_API_KEY"))
    return {"X-API-Key": api_key} if api_key else {}


def _get(url: str, headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
    r = requests.get(url, headers=headers, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def _post(url: str, headers: dict[str, str], payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    svc = sub.add_parser("service", help="Health/capabilities/create_session smoke checks")
    svc.add_argument("--base-url", default=None)
    svc.add_argument("--api-key", default=None)
    svc.add_argument("--timeout-s", type=float, default=10.0)
    svc.add_argument("--create-sampling-session", action="store_true")
    svc.add_argument("--base-model", default="Qwen/Qwen3-0.6B")

    dt = sub.add_parser("dense-train", help="Dense training smoke check (cross-entropy)")
    dt.add_argument("--base-url", default=None)
    dt.add_argument("--api-key", default=None)
    dt.add_argument("--model", default="Qwen/Qwen3-0.6B")
    dt.add_argument("--lora-rank", type=int, default=16)
    dt.add_argument("--steps", type=int, default=2)
    dt.add_argument("--max-len", type=int, default=512)
    dt.add_argument("--learning-rate", type=float, default=5e-5)
    dt.add_argument("--heartbeat-s", type=float, default=30.0)
    dt.add_argument("--call-timeout-s", type=float, default=900.0)

    return p.parse_args()


def _service(args: argparse.Namespace) -> int:
    _load_env()
    base_url = _base_url(args)
    headers = _headers(args)
    timeout_s = float(args.timeout_s)

    t0 = time.time()
    health = _get(f"{base_url}/api/v1/healthz", headers, timeout_s=timeout_s)
    print(f"[{_ts()}] healthz ok dt_s={time.time()-t0:.2f} body={health}", flush=True)

    t0 = time.time()
    caps = _get(f"{base_url}/api/v1/get_server_capabilities", headers, timeout_s=timeout_s)
    n = len(caps.get("supported_models") or [])
    print(f"[{_ts()}] capabilities ok dt_s={time.time()-t0:.2f} supported_models={n}", flush=True)

    t0 = time.time()
    sess = _post(
        f"{base_url}/api/v1/create_session",
        headers,
        payload={"tags": ["scripts/tools/smoke.py"], "user_metadata": {}, "sdk_version": "scripts/tools/smoke.py"},
        timeout_s=timeout_s,
    )
    session_id = sess.get("session_id")
    if not session_id:
        raise RuntimeError(f"create_session missing session_id: {sess}")
    print(f"[{_ts()}] create_session ok dt_s={time.time()-t0:.2f} session_id={session_id}", flush=True)

    if args.create_sampling_session:
        t0 = time.time()
        out = _post(
            f"{base_url}/api/v1/create_sampling_session",
            headers,
            payload={"session_id": session_id, "sampling_session_seq_id": 0, "base_model": args.base_model},
            timeout_s=max(timeout_s, 60.0),
        )
        sampling_session_id = out.get("sampling_session_id")
        if not sampling_session_id:
            raise RuntimeError(f"create_sampling_session missing sampling_session_id: {out}")
        print(
            f"[{_ts()}] create_sampling_session ok dt_s={time.time()-t0:.2f} base_model={args.base_model} "
            f"sampling_session_id={sampling_session_id}",
            flush=True,
        )

    return 0


def _wait_future(fut: Any, *, label: str, timeout_s: float, heartbeat_s: float) -> Any:
    start = time.time()
    while True:
        elapsed = time.time() - start
        remaining = timeout_s - elapsed
        if remaining <= 0:
            raise TimeoutError(f"timeout while waiting {label} elapsed_s={elapsed:.0f}")
        try:
            return fut.result(timeout=min(heartbeat_s, max(0.5, remaining)))
        except TimeoutError:
            print(f"[{_ts()}] waiting {label} elapsed_s={elapsed:.0f}", flush=True)


def _make_dense_datum(tokenizer: Any, *, max_len: int) -> Any:
    from mint import types

    random.seed(42)
    a = random.randint(10, 99)
    b = random.randint(10, 99)

    prompt = f"Question: What is {a} + {b}?\nAnswer:"
    completion = f" {a + b}"

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
    eos_id = tokenizer.eos_token_id
    completion_tokens = completion_tokens + [eos_id]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)
    tokens = prompt_tokens + completion_tokens

    if len(tokens) < max_len:
        pad = max_len - len(tokens)
        tokens = tokens + [pad_id] * pad
        weights = weights + [0.0] * pad
    else:
        tokens = tokens[:max_len]
        weights = weights[:max_len]

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    target_weights = weights[1:]

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={"target_tokens": target_tokens, "weights": target_weights},
    )


def _compute_weighted_loss(loss_fn_outputs: Any, weights: list[float]) -> float:
    total_loss = 0.0
    total_weight = 0.0

    for out in loss_fn_outputs:
        logprobs = out["logprobs"]
        if hasattr(logprobs, "tolist"):
            logprobs = logprobs.tolist()
        for lp, wt in zip(logprobs, weights):
            total_loss += -float(lp) * float(wt)
            total_weight += float(wt)

    return total_loss / max(total_weight, 1.0)


def _dense_train(args: argparse.Namespace) -> int:
    _load_env()
    base_url = _base_url(args)
    api_key = _coalesce(args.api_key, os.environ.get("MINT_API_KEY"), os.environ.get("TINKER_API_KEY"))

    import mint
    from mint import types

    print(
        f"[{_ts()}] start dense-train base_url={base_url} model={args.model} rank={args.lora_rank} steps={args.steps}",
        flush=True,
    )
    sc = mint.ServiceClient(base_url=base_url, api_key=api_key)
    tc = sc.create_lora_training_client(
        base_model=args.model,
        rank=args.lora_rank,
        train_mlp=True,
        train_attn=True,
        train_unembed=True,
    )
    tok = tc.get_tokenizer()

    datum = _make_dense_datum(tok, max_len=args.max_len)
    weights = datum.loss_fn_inputs["weights"]
    if hasattr(weights, "tolist"):
        weights = weights.tolist()

    for step in range(int(args.steps)):
        t0 = time.time()
        fw_fut = tc.forward_backward(data=[datum], loss_fn="cross_entropy")
        fw_res = _wait_future(
            fw_fut,
            label=f"forward_backward step={step + 1}/{args.steps}",
            timeout_s=float(args.call_timeout_s),
            heartbeat_s=float(args.heartbeat_s),
        )
        loss = _compute_weighted_loss(fw_res.loss_fn_outputs, weights)
        fw_s = time.time() - t0

        t0 = time.time()
        opt_fut = tc.optim_step(types.AdamParams(learning_rate=float(args.learning_rate)))
        _wait_future(
            opt_fut,
            label=f"optim_step step={step + 1}/{args.steps}",
            timeout_s=float(args.call_timeout_s),
            heartbeat_s=float(args.heartbeat_s),
        )
        opt_s = time.time() - t0

        print(f"[{_ts()}] step={step + 1}/{args.steps} loss={loss:.6f} fw_s={fw_s:.2f} opt_s={opt_s:.2f}", flush=True)

    return 0


def main() -> int:
    args = _parse_args()
    cmd = args.cmd or "service"

    if cmd == "service":
        return _service(args)
    if cmd == "dense-train":
        return _dense_train(args)

    raise SystemExit(f"unknown cmd: {cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
