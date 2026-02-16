#!/usr/bin/env python3
"""Pressure-test K2 Megatron trainer path only (no inference actor usage)."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def _ts() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_env() -> None:
    load_dotenv()
    for name in (".env", ".secrets.env"):
        env_file = _repo_root() / name
        if env_file.exists():
            load_dotenv(env_file, override=False)


def _coalesce(*vals: str | None) -> str | None:
    for value in vals:
        if value:
            return value
    return None


def _wait_future(fut: Any, *, label: str, timeout_s: float, heartbeat_s: float) -> Any:
    start = time.time()
    while True:
        elapsed = time.time() - start
        if elapsed >= timeout_s:
            raise TimeoutError(f"timeout waiting {label} elapsed_s={elapsed:.1f}")
        try:
            return fut.result(timeout=min(heartbeat_s, max(0.5, timeout_s - elapsed)))
        except TimeoutError:
            print(f"[{_ts()}] waiting {label} elapsed_s={elapsed:.1f}", flush=True)


def _make_prompt_tokens(tokenizer: Any, prompt_len: int) -> list[int]:
    filler_ids = tokenizer.encode(" a", add_special_tokens=False) or tokenizer.encode(
        "0", add_special_tokens=False
    )
    if not filler_ids:
        raise RuntimeError("failed to get filler token id from tokenizer")
    return [int(filler_ids[0])] * int(prompt_len)


def _chunk_datums(datums: list[Any], microbatch: int) -> list[list[Any]]:
    if microbatch <= 0 or microbatch >= len(datums):
        return [datums]
    return [datums[i : i + microbatch] for i in range(0, len(datums), microbatch)]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pressure-test Megatron forward_backward + optim_step without inference"
    )
    p.add_argument(
        "--base-url",
        default=_coalesce(
            os.environ.get("TINKER_BASE_URL"),
            os.environ.get("MINT_BASE_URL"),
            "http://localhost:18000",
        ),
    )
    p.add_argument(
        "--api-key",
        default=_coalesce(
            os.environ.get("TINKER_API_KEY"),
            os.environ.get("MINT_API_KEY"),
            "",
        ),
    )
    p.add_argument("--model", default="moonshotai/Kimi-K2-Instruct")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--train-mlp", action="store_true", default=True)
    p.add_argument("--no-train-mlp", dest="train_mlp", action="store_false")
    p.add_argument("--train-attn", action="store_true", default=True)
    p.add_argument("--no-train-attn", dest="train_attn", action="store_false")
    p.add_argument("--train-unembed", action="store_true", default=True)
    p.add_argument("--no-train-unembed", dest="train_unembed", action="store_false")
    p.add_argument("--train-prompts", type=int, default=8)
    p.add_argument("--train-groups", type=int, default=8)
    p.add_argument("--train-context-len", type=int, default=32000)
    p.add_argument("--train-steps", type=int, default=1)
    p.add_argument("--train-microbatch", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--loss-fn", choices=("ppo", "importance_sampling"), default="ppo")
    p.add_argument("--call-timeout-s", type=float, default=10800.0)
    p.add_argument("--heartbeat-s", type=float, default=30.0)
    p.add_argument("--run-dir", default=None)
    return p.parse_args()


def main() -> int:
    _load_env()
    args = _parse_args()

    import mint
    from mint import types

    base_url = str(args.base_url).rstrip("/")
    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else (_repo_root() / "results" / "benchmarks" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f"k2_megatron_train_only_{args.model.replace('/', '_')}.json"

    print(
        f"[{_ts()}] start base_url={base_url} model={args.model} rank={args.rank} "
        f"train_mlp={args.train_mlp} train_attn={args.train_attn} train_unembed={args.train_unembed}",
        flush=True,
    )

    service_client = mint.ServiceClient(base_url=base_url, api_key=args.api_key or None)
    training_client = service_client.create_lora_training_client(
        base_model=args.model,
        rank=int(args.rank),
        train_mlp=bool(args.train_mlp),
        train_attn=bool(args.train_attn),
        train_unembed=bool(args.train_unembed),
    )
    print(
        f"[{_ts()}] training_client model_id={getattr(training_client, 'model_id', None)}",
        flush=True,
    )
    if str(args.model).startswith("moonshotai/Kimi-K2-"):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    else:
        tokenizer = training_client.get_tokenizer()

    train_batch_size = int(args.train_prompts) * int(args.train_groups)
    if train_batch_size < 1:
        raise ValueError("train_prompts * train_groups must be >= 1")
    if int(args.train_context_len) < 2:
        raise ValueError("train_context_len must be >= 2")

    train_tokens = _make_prompt_tokens(tokenizer, prompt_len=int(args.train_context_len))
    input_tokens = train_tokens[:-1]
    target_tokens = train_tokens[1:]
    seq_len = len(input_tokens)

    base_datum = types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={
            "target_tokens": target_tokens,
            "logprobs": [0.0] * seq_len,
            "advantages": [1.0] * seq_len,
        },
    )
    datums = [base_datum] * train_batch_size
    microbatches = _chunk_datums(datums, int(args.train_microbatch))
    print(
        f"[{_ts()}] datums={train_batch_size} microbatch={int(args.train_microbatch)} "
        f"num_microbatches={len(microbatches)}",
        flush=True,
    )

    step_metrics: list[dict[str, Any]] = []
    training_error: str | None = None
    for step in range(int(args.train_steps)):
        try:
            fw_s = 0.0
            for micro_idx, micro in enumerate(microbatches):
                t0 = time.time()
                fw_fut = training_client.forward_backward(micro, loss_fn=str(args.loss_fn))
                _wait_future(
                    fw_fut,
                    label=f"forward_backward step={step + 1} microbatch={micro_idx + 1}/{len(microbatches)}",
                    timeout_s=float(args.call_timeout_s),
                    heartbeat_s=float(args.heartbeat_s),
                )
                fw_s += time.time() - t0

            t0 = time.time()
            opt_fut = training_client.optim_step(
                types.AdamParams(learning_rate=float(args.learning_rate))
            )
            _wait_future(
                opt_fut,
                label=f"optim_step step={step + 1}",
                timeout_s=float(args.call_timeout_s),
                heartbeat_s=float(args.heartbeat_s),
            )
            opt_s = time.time() - t0

            tokens_per_step = train_batch_size * seq_len
            step_total = fw_s + opt_s
            rec = {
                "step": step + 1,
                "batch_size_datums": int(train_batch_size),
                "context_len": int(args.train_context_len),
                "tokens_per_datum": int(seq_len),
                "tokens_per_step": int(tokens_per_step),
                "train_microbatch": int(args.train_microbatch),
                "num_microbatches": int(len(microbatches)),
                "forward_backward_s": float(fw_s),
                "optim_step_s": float(opt_s),
                "step_total_s": float(step_total),
                "forward_backward_tokens_per_s": float(tokens_per_step / fw_s if fw_s > 0 else 0.0),
                "step_tokens_per_s": float(tokens_per_step / step_total if step_total > 0 else 0.0),
            }
            step_metrics.append(rec)
            print(
                f"[{_ts()}] step {step + 1}/{int(args.train_steps)} "
                f"forward_backward_s={fw_s:.2f} optim_step_s={opt_s:.2f} "
                f"tokens_per_step={tokens_per_step}",
                flush=True,
            )
        except Exception as e:
            training_error = f"{type(e).__name__}: {e}"
            step_metrics.append(
                {
                    "step": step + 1,
                    "batch_size_datums": int(train_batch_size),
                    "context_len": int(args.train_context_len),
                    "tokens_per_datum": int(seq_len),
                    "tokens_per_step": int(train_batch_size * seq_len),
                    "train_microbatch": int(args.train_microbatch),
                    "num_microbatches": int(len(microbatches)),
                    "error": training_error,
                }
            )
            break

    output = {
        "timestamp": _ts(),
        "base_url": base_url,
        "model": args.model,
        "rank": int(args.rank),
        "train_mlp": bool(args.train_mlp),
        "train_attn": bool(args.train_attn),
        "train_unembed": bool(args.train_unembed),
        "loss_fn": str(args.loss_fn),
        "train_microbatch": int(args.train_microbatch),
        "training_speed_long_context": {
            "train_prompts": int(args.train_prompts),
            "train_groups": int(args.train_groups),
            "effective_batch_size": int(train_batch_size),
            "train_context_len": int(args.train_context_len),
            "train_steps": int(args.train_steps),
            "step_metrics": step_metrics,
            "error": training_error,
        },
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"[{_ts()}] wrote {out_path}", flush=True)
    print(json.dumps(output, indent=2), flush=True)
    return 2 if training_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
