#!/usr/bin/env python3
"""Pressure test: concurrent 32k-context training across supported models.

Runs three concurrent SFT loops (forward_backward + optim_step) at a fixed
sequence length (default: 32000 tokens):
  - Qwen/Qwen3-0.6B
  - Qwen/Qwen3-4B
  - Qwen/Qwen3-30B-A3B-Instruct-2507

Target server:
  - Set MINT_BASE_URL or TINKER_BASE_URL (expected to be a localhost tunnel).
  - Set MINT_API_KEY or TINKER_API_KEY if your deployment requires auth.

Exit codes:
  - 0: all tasks completed
  - 1: any task failed or timed out (possible freeze)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

DEFAULT_BASE_URL = "http://localhost:18000"


@dataclass(frozen=True)
class TaskConfig:
    model: str
    lora_rank: int
    max_context_len: int
    steps: int
    learning_rate: float
    call_timeout_s: float


def _coalesce(*values: str | None) -> str | None:
    for v in values:
        if v:
            return v
    return None


def _load_env() -> None:
    # Load env from the CWD (common case) and from the repo root (when invoked via wrapper).
    load_dotenv()
    repo_root_env = Path(__file__).resolve().parents[2] / ".env"
    if repo_root_env.exists():
        load_dotenv(repo_root_env, override=False)

    if "MINT_BASE_URL" not in os.environ and "TINKER_BASE_URL" not in os.environ:
        os.environ["MINT_BASE_URL"] = DEFAULT_BASE_URL


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=None, help="Overrides MINT_BASE_URL/TINKER_BASE_URL")
    p.add_argument("--api-key", default=None, help="Overrides MINT_API_KEY/TINKER_API_KEY")
    p.add_argument("--max-context-len", type=int, default=32000)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--call-timeout-s", type=float, default=900.0)
    return p.parse_args()


def _make_sft_datum(example: dict[str, str], tokenizer: Any, max_length: int) -> Any:
    from mint import types

    prompt = f"Question: {example['question']}\nAnswer:"
    completion = f" {example['answer']}"

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)

    eos_token_id = tokenizer.eos_token_id
    completion_tokens = completion_tokens + [eos_token_id]

    prompt_weights = [0] * len(prompt_tokens)
    completion_weights = [1] * len(completion_tokens)

    all_tokens = prompt_tokens + completion_tokens
    all_weights = prompt_weights + completion_weights

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id
    current_len = len(all_tokens)

    if current_len < max_length:
        padding_len = max_length - current_len
        all_tokens = all_tokens + [pad_token_id] * padding_len
        all_weights = all_weights + [0] * padding_len
    elif current_len > max_length:
        all_tokens = all_tokens[:max_length]
        all_weights = all_weights[:max_length]

    input_tokens = all_tokens[:-1]
    target_tokens = all_tokens[1:]
    weights = all_weights[1:]

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={"target_tokens": target_tokens, "weights": weights},
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


def _run_task(base_url: str, api_key: str | None, cfg: TaskConfig) -> dict[str, Any]:
    import mint
    from mint import types

    try:
        sc = mint.ServiceClient(base_url=base_url, api_key=api_key)
        tc = sc.create_lora_training_client(
            base_model=cfg.model,
            rank=cfg.lora_rank,
            train_mlp=True,
            train_attn=True,
            train_unembed=True,
        )
        tokenizer = tc.get_tokenizer()
    except Exception as e:
        return {
            "model": cfg.model,
            "status": "setup_error",
            "error": f"{type(e).__name__}: {e}",
            "losses": [],
        }

    random.seed(42)
    a = random.randint(10, 99)
    b = random.randint(10, 99)
    ex = {"question": f"What is {a} * {b}?", "answer": str(a * b)}

    try:
        datum = _make_sft_datum(ex, tokenizer, max_length=cfg.max_context_len)
        weights = datum.loss_fn_inputs["weights"]
        if hasattr(weights, "tolist"):
            weights = weights.tolist()
    except Exception as e:
        return {
            "model": cfg.model,
            "status": "datum_error",
            "error": f"{type(e).__name__}: {e}",
            "losses": [],
        }

    losses: list[float] = []

    for step in range(cfg.steps):
        try:
            fw_fut = tc.forward_backward(data=[datum], loss_fn="cross_entropy")
            fw_res = fw_fut.result(timeout=cfg.call_timeout_s)
        except Exception as e:
            return {
                "model": cfg.model,
                "status": "forward_backward_error",
                "step": step + 1,
                "error": f"{type(e).__name__}: {e}",
                "losses": losses,
            }

        try:
            loss = _compute_weighted_loss(fw_res.loss_fn_outputs, weights)
            losses.append(loss)
        except Exception as e:
            return {
                "model": cfg.model,
                "status": "loss_compute_error",
                "step": step + 1,
                "error": f"{type(e).__name__}: {e}",
                "losses": losses,
            }

        try:
            opt_fut = tc.optim_step(types.AdamParams(learning_rate=cfg.learning_rate))
            opt_fut.result(timeout=cfg.call_timeout_s)
        except Exception as e:
            return {
                "model": cfg.model,
                "status": "optim_step_error",
                "step": step + 1,
                "error": f"{type(e).__name__}: {e}",
                "losses": losses,
            }

        print(f"[{cfg.model}] step {step + 1}/{cfg.steps}: loss={loss:.4f}", flush=True)

    return {"model": cfg.model, "status": "ok", "losses": losses}


def main() -> int:
    _load_env()
    args = _parse_args()

    base_url = _coalesce(args.base_url, os.environ.get("MINT_BASE_URL"), os.environ.get("TINKER_BASE_URL"), DEFAULT_BASE_URL)
    api_key = _coalesce(args.api_key, os.environ.get("MINT_API_KEY"), os.environ.get("TINKER_API_KEY"))

    print(f"Base URL: {base_url}", flush=True)
    if api_key:
        print("API key: set", flush=True)
    else:
        print("API key: missing (this is OK only if server allows unauthenticated access)", flush=True)

    models = [
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
    ]

    cfgs = [
        TaskConfig(
            model=m,
            lora_rank=args.lora_rank,
            max_context_len=args.max_context_len,
            steps=args.steps,
            learning_rate=args.learning_rate,
            call_timeout_s=args.call_timeout_s,
        )
        for m in models
    ]

    start = time.time()
    results: list[dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cfgs)) as ex:
        futs = {ex.submit(_run_task, base_url, api_key, cfg): cfg.model for cfg in cfgs}
        try:
            for fut in concurrent.futures.as_completed(futs, timeout=max(60.0, args.call_timeout_s * max(1, args.steps))):
                try:
                    results.append(fut.result())
                except Exception as e:
                    results.append({
                        "model": futs[fut],
                        "status": "exception",
                        "error": f"{type(e).__name__}: {e}",
                    })
        except concurrent.futures.TimeoutError:
            for fut, model in futs.items():
                if not fut.done():
                    results.append({"model": model, "status": "timeout", "error": "task timed out (possible freeze)"})

    elapsed = time.time() - start
    print(f"Elapsed: {elapsed:.2f}s", flush=True)

    failed = [r for r in results if r.get("status") != "ok"]
    for r in results:
        status = r.get("status")
        model = r.get("model")
        losses = r.get("losses", [])
        if losses:
            print(f"Result: {model} status={status} loss0={losses[0]:.4f} lossN={losses[-1]:.4f}", flush=True)
        else:
            print(f"Result: {model} status={status} error={r.get('error', '')}", flush=True)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
