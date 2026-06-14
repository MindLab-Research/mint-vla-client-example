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
    env_file = _repo_root() / ".env"
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark model sampling and training speed")
    p.add_argument(
        "--base-url",
        default=_coalesce(
            os.environ.get("MINT_BASE_URL"),
            os.environ.get("MINT_BASE_URL"),
            "http://localhost:8000",
        ),
    )
    p.add_argument(
        "--api-key",
        default=_coalesce(
            os.environ.get("MINT_API_KEY"),
            os.environ.get("MINT_API_KEY"),
            "dummy",
        ),
    )
    p.add_argument("--model", required=True)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--call-timeout-s", type=float, default=10800.0)
    p.add_argument("--heartbeat-s", type=float, default=30.0)

    p.add_argument("--sampling-num-samples", type=int, default=1)
    p.add_argument("--long-prefill-tokens", type=int, default=30000)
    p.add_argument("--short-generation-tokens", type=int, default=256)
    p.add_argument("--moderate-prefill-tokens", type=int, default=8192)
    p.add_argument("--long-generation-tokens", type=int, default=4096)

    p.add_argument("--train-prompts", type=int, default=8)
    p.add_argument("--train-groups", type=int, default=8)
    p.add_argument("--train-context-len", type=int, default=30000)
    p.add_argument("--train-steps", type=int, default=1)
    p.add_argument("--skip-transfer-and-sampling", action="store_true")
    p.add_argument("--run-dir", default=None)
    return p.parse_args()


def _make_prompt_tokens(tokenizer: Any, prompt_len: int) -> list[int]:
    filler_ids = tokenizer.encode(" a", add_special_tokens=False) or tokenizer.encode(
        "0", add_special_tokens=False
    )
    if not filler_ids:
        raise RuntimeError("failed to get filler token id from tokenizer")
    return [int(filler_ids[0])] * int(prompt_len)


def _run_sampling_case(
    *,
    sampling_client: Any,
    model_input_type: Any,
    sampling_params_type: Any,
    prompt_tokens: list[int],
    max_tokens: int,
    num_samples: int,
    timeout_s: float,
    heartbeat_s: float,
    case_name: str,
) -> dict[str, Any]:
    t0 = time.time()
    fut = sampling_client.sample(
        prompt=model_input_type.from_ints(tokens=prompt_tokens),
        num_samples=int(num_samples),
        sampling_params=sampling_params_type(
            max_tokens=int(max_tokens),
            temperature=0.7,
            top_k=-1,
            top_p=1.0,
        ),
    )
    result = _wait_future(
        fut,
        label=f"sample:{case_name}",
        timeout_s=float(timeout_s),
        heartbeat_s=float(heartbeat_s),
    )
    elapsed_s = time.time() - t0
    sequences = result.sequences if result is not None else []
    generated_tokens = sum(len(seq.tokens) for seq in sequences)
    return {
        "case": case_name,
        "prefill_tokens": len(prompt_tokens),
        "max_decode_tokens": int(max_tokens),
        "num_samples": int(num_samples),
        "num_sequences": len(sequences),
        "decode_tokens": int(generated_tokens),
        "elapsed_s": float(elapsed_s),
        "decode_tokens_per_s": float(generated_tokens / elapsed_s if elapsed_s > 0 else 0.0),
        "prefill_plus_decode_tokens_per_s": float(
            (len(prompt_tokens) + generated_tokens) / elapsed_s if elapsed_s > 0 else 0.0
        ),
    }


def main() -> int:
    _load_env()
    args = _parse_args()

    import mint
    from mint import types

    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else (_repo_root() / "results" / "benchmarks" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f"model_speed_{args.model.replace('/', '_')}.json"

    base_url = str(args.base_url).rstrip("/")
    print(
        f"[{_ts()}] start benchmark base_url={base_url} model={args.model} rank={args.rank}",
        flush=True,
    )

    service_client = mint.ServiceClient(base_url=base_url, api_key=args.api_key)
    training_client = service_client.create_lora_training_client(
        base_model=args.model,
        rank=int(args.rank),
        train_mlp=True,
        train_attn=True,
        train_unembed=True,
    )
    tokenizer = training_client.get_tokenizer()

    transfer: dict[str, Any] | None = None
    sampling_results: list[dict[str, Any]] = []
    if not args.skip_transfer_and_sampling:
        save_name = f"speedbench-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        t0 = time.time()
        save_fut = training_client.save_weights_for_sampler(name=save_name)
        save_res = _wait_future(
            save_fut,
            label="save_weights_for_sampler",
            timeout_s=float(args.call_timeout_s),
            heartbeat_s=float(args.heartbeat_s),
        )
        save_s = time.time() - t0
        sampling_path = save_res.path

        t0 = time.time()
        sampling_client = service_client.create_sampling_client(
            model_path=sampling_path,
            base_model=args.model,
        )
        create_sampling_client_s = time.time() - t0

        warmup_prompt = _make_prompt_tokens(tokenizer, prompt_len=32)
        warmup_res = _run_sampling_case(
            sampling_client=sampling_client,
            model_input_type=types.ModelInput,
            sampling_params_type=types.SamplingParams,
            prompt_tokens=warmup_prompt,
            max_tokens=1,
            num_samples=1,
            timeout_s=float(args.call_timeout_s),
            heartbeat_s=float(args.heartbeat_s),
            case_name="warmup_after_transfer",
        )
        transfer_total_s = (
            float(save_s) + float(create_sampling_client_s) + float(warmup_res["elapsed_s"])
        )
        transfer = {
            "save_weights_for_sampler_s": float(save_s),
            "create_sampling_client_s": float(create_sampling_client_s),
            "warmup_after_transfer_s": float(warmup_res["elapsed_s"]),
            "total_s": float(transfer_total_s),
            "model_path": str(sampling_path),
        }

        long_prefill_prompt = _make_prompt_tokens(tokenizer, prompt_len=int(args.long_prefill_tokens))
        moderate_prefill_prompt = _make_prompt_tokens(
            tokenizer, prompt_len=int(args.moderate_prefill_tokens)
        )
        sampling_results = [
            _run_sampling_case(
                sampling_client=sampling_client,
                model_input_type=types.ModelInput,
                sampling_params_type=types.SamplingParams,
                prompt_tokens=long_prefill_prompt,
                max_tokens=int(args.short_generation_tokens),
                num_samples=int(args.sampling_num_samples),
                timeout_s=float(args.call_timeout_s),
                heartbeat_s=float(args.heartbeat_s),
                case_name="long_prefill_short_generation",
            ),
            _run_sampling_case(
                sampling_client=sampling_client,
                model_input_type=types.ModelInput,
                sampling_params_type=types.SamplingParams,
                prompt_tokens=moderate_prefill_prompt,
                max_tokens=int(args.long_generation_tokens),
                num_samples=int(args.sampling_num_samples),
                timeout_s=float(args.call_timeout_s),
                heartbeat_s=float(args.heartbeat_s),
                case_name="moderate_prefill_long_generation",
            ),
        ]

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

    training_results: list[dict[str, Any]] = []
    training_error: str | None = None
    for step in range(int(args.train_steps)):
        t0 = time.time()
        try:
            fw_fut = training_client.forward_backward(datums, loss_fn="ppo")
            _wait_future(
                fw_fut,
                label=f"forward_backward step={step + 1}",
                timeout_s=float(args.call_timeout_s),
                heartbeat_s=float(args.heartbeat_s),
            )
            fw_s = time.time() - t0

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

            total_tokens = train_batch_size * seq_len
            training_results.append(
                {
                    "step": step + 1,
                    "batch_size_datums": int(train_batch_size),
                    "context_len": int(args.train_context_len),
                    "tokens_per_datum": int(seq_len),
                    "tokens_per_step": int(total_tokens),
                    "forward_backward_s": float(fw_s),
                    "optim_step_s": float(opt_s),
                    "step_total_s": float(fw_s + opt_s),
                    "forward_backward_tokens_per_s": float(
                        total_tokens / fw_s if fw_s > 0 else 0.0
                    ),
                    "step_tokens_per_s": float(
                        total_tokens / (fw_s + opt_s) if (fw_s + opt_s) > 0 else 0.0
                    ),
                }
            )
        except Exception as e:
            training_error = f"{type(e).__name__}: {e}"
            training_results.append(
                {
                    "step": step + 1,
                    "batch_size_datums": int(train_batch_size),
                    "context_len": int(args.train_context_len),
                    "tokens_per_datum": int(seq_len),
                    "tokens_per_step": int(train_batch_size * seq_len),
                    "forward_backward_elapsed_before_error_s": float(time.time() - t0),
                    "error": training_error,
                }
            )
            break

    output = {
        "timestamp": _ts(),
        "base_url": base_url,
        "model": args.model,
        "rank": int(args.rank),
        "skip_transfer_and_sampling": bool(args.skip_transfer_and_sampling),
        "transfer_checkpoint_to_inferencer": transfer,
        "sampling_speed": sampling_results,
        "training_speed_long_context": {
            "train_prompts": int(args.train_prompts),
            "train_groups": int(args.train_groups),
            "effective_batch_size": int(train_batch_size),
            "train_context_len": int(args.train_context_len),
            "train_steps": int(args.train_steps),
            "step_metrics": training_results,
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
