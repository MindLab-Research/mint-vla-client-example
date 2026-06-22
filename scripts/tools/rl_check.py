#!/usr/bin/env python3
"""RL e2e check for mint-server.

Runs a minimal RL training loop (sample → reward → forward_backward → optim_step
→ save_weights → sample again) against a mint-server deployment, collecting
timing evidence and a reward trajectory. Designed for dev validation after
code changes that affect actor naming, placement, or training/inference flow.

Requires: mindlab-toolkit (provides `import mint` which wraps tinker SDK).

Usage:
  # Dev server (no auth, localhost:8000)
  MINT_BASE_URL=http://localhost:8000 python rl_check.py --model Qwen/Qwen3-0.6B

  # Production
  MINT_BASE_URL=https://mint.macaron.xin MINT_API_KEY=<key> \
    python rl_check.py --model Qwen/Qwen3-0.6B

  # Multiple models
  python rl_check.py --model Qwen/Qwen3-0.6B --model Qwen/Qwen3-4B-Instruct-2507

  # Custom RL params
  python rl_check.py --model Qwen/Qwen3-0.6B --steps 20 --group-size 8 --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mint
from mint import types


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _now() -> float:
    return time.monotonic()


# ---------------------------------------------------------------------------
# Task: countdown arithmetic
# ---------------------------------------------------------------------------

def _make_prompt(target: int) -> str:
    """Generate a countdown task: 'Count down from N to 1.'"""
    return f"Count down from {target} to 1.\n{target},"


def _expected_answer(target: int) -> str:
    return ", ".join(str(i) for i in range(target, 0, -1))


def _reward(generated_text: str, target: int) -> float:
    """1.0 if the generated text contains the correct countdown sequence, else 0.0."""
    expected = _expected_answer(target)
    return 1.0 if expected in generated_text else 0.0


# ---------------------------------------------------------------------------
# Tokenizer helper
# ---------------------------------------------------------------------------

def _get_tokenizer(training_client: Any, base_model: str) -> Any:
    """Get a tokenizer from the training client."""
    return training_client.get_tokenizer()


def _encode(tokenizer: Any, text: str) -> list[int]:
    """Encode text to token IDs."""
    return tokenizer.encode(text)


def _decode(tokenizer: Any, token_ids: list[int]) -> str:
    """Decode token IDs to text."""
    return tokenizer.decode(token_ids)


# ---------------------------------------------------------------------------
# RL step
# ---------------------------------------------------------------------------

def run_rl_step(
    *,
    step_idx: int,
    training_client: Any,
    sampling_client: Any,
    tokenizer: Any,
    group_size: int,
    max_tokens: int,
    learning_rate: float,
    timeout_s: float,
) -> dict[str, Any]:
    """Run one RL step: sample → reward → forward_backward → optim_step.

    Returns a dict with metrics for this step.
    """
    step_t0 = _now()
    target = random.randint(5, 20)
    prompt = _make_prompt(target)
    prompt_tokens = _encode(tokenizer, prompt)

    all_rewards: list[float] = []
    training_datums: list[types.Datum] = []

    for _ in range(group_size):
        sample_future = sampling_client.sample(
            prompt=types.ModelInput.from_ints(prompt_tokens),
            num_samples=1,
            sampling_params=types.SamplingParams(
                max_tokens=max_tokens,
                temperature=0.7,
            ),
        )
        sample_result = sample_result_wait(sample_future, timeout_s=timeout_s)
        # SDK v2: SampleResponse.sequences[0].tokens instead of SampleResponse.tokens
        seq = sample_result.sequences[0]
        generated_tokens = [int(t) for t in seq.tokens]
        print(f"    [sample] generated {len(generated_tokens)} tokens: {generated_tokens[:10]}...", flush=True)
        generated_text = _decode(tokenizer, generated_tokens)
        r = _reward(generated_text, target)
        print(f"    [reward] {r:.1f} text={generated_text[:60]!r}", flush=True)
        all_rewards.append(r)

        full_tokens = prompt_tokens + generated_tokens
        # Build loss target: the generated tokens are the targets.
        # input_tokens = everything except the last token.
        # target_tokens = everything except the first token (shifted by 1).
        input_tokens = full_tokens[:-1]
        target_tokens = full_tokens[1:]

        # Loss mask: only on the generated portion.
        loss_mask = [0.0] * len(prompt_tokens) + [1.0] * len(generated_tokens)
        loss_mask = loss_mask[:len(target_tokens)]

        if not loss_mask or sum(loss_mask) == 0:
            continue

        loss_fn_inputs = {
            "target_tokens": types.TensorData(data=target_tokens, dtype="int64", shape=[len(target_tokens)]),
            "weights": types.TensorData(data=loss_mask, dtype="float32", shape=[len(loss_mask)]),
        }

        training_datums.append(
            types.Datum(
                model_input=types.ModelInput.from_ints(tokens=input_tokens),
                loss_fn_inputs=loss_fn_inputs,
            )
        )

    avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    accuracy = sum(1 for r in all_rewards if r > 0) / len(all_rewards) if all_rewards else 0.0

    if training_datums:
        fwdbwd_future = training_client.forward_backward(
            training_datums,
            loss_fn="cross_entropy",
        )
        _wait_future(fwdbwd_future, "forward_backward", step_idx, timeout_s)

        optim_future = training_client.optim_step(
            types.AdamParams(learning_rate=learning_rate),
        )
        _wait_future(optim_future, "optim_step", step_idx, timeout_s)

    step_s = _now() - step_t0
    return {
        "step": step_idx + 1,
        "target": target,
        "avg_reward": round(avg_reward, 4),
        "accuracy": round(accuracy, 4),
        "num_datums": len(training_datums),
        "elapsed_s": round(step_s, 2),
    }


def sample_result_wait(future: Any, *, timeout_s: float) -> Any:
    """Wait for a sample future with timeout."""
    return future.result(timeout=timeout_s if timeout_s > 0 else None)


def _wait_future(future: Any, label: str, step_idx: int, timeout_s: float) -> Any:
    """Wait for a future with a heartbeat print."""
    t0 = _now()
    while True:
        try:
            return future.result(timeout=30 if timeout_s <= 0 else min(30, timeout_s))
        except TimeoutError:
            elapsed = _now() - t0
            if timeout_s > 0 and elapsed > timeout_s:
                raise
            print(f"  [step {step_idx + 1}] {label} waiting... ({elapsed:.0f}s)", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_one_model(
    *,
    service_client: mint.ServiceClient,
    base_model: str,
    num_steps: int,
    group_size: int,
    learning_rate: float,
    max_tokens: int,
    lora_rank: int,
    timeout_s: float,
) -> list[dict[str, Any]]:
    """Run RL loop for one model. Returns list of per-step metrics."""
    print(f"\n{'='*60}")
    print(f"Model: {base_model}")
    print(f"Steps: {num_steps}, Group size: {group_size}, LR: {learning_rate}")
    print(f"LoRA rank: {lora_rank}, Max tokens: {max_tokens}")
    print(f"{'='*60}\n")

    # Create training client
    t0 = _now()
    training_client = service_client.create_lora_training_client(
        base_model=base_model,
        rank=lora_rank,
    )
    print(f"Training client created ({_now() - t0:.1f}s)")

    tokenizer = _get_tokenizer(training_client, base_model)

    # Create initial sampling client on base model (no LoRA needed for step 0).
    t0 = _now()
    sampling_client = service_client.create_sampling_client(
        base_model=base_model,
    )
    print(f"Sampling client created ({_now() - t0:.1f}s)")

    all_metrics: list[dict[str, Any]] = []

    for step in range(num_steps):
        metrics = run_rl_step(
            step_idx=step,
            training_client=training_client,
            sampling_client=sampling_client,
            tokenizer=tokenizer,
            group_size=group_size,
            max_tokens=max_tokens,
            learning_rate=learning_rate,
            timeout_s=timeout_s,
        )
        all_metrics.append(metrics)
        print(
            f"  [step {metrics['step']:>3}/{num_steps}] "
            f"reward={metrics['avg_reward']:.3f} "
            f"acc={metrics['accuracy']:.1%} "
            f"datums={metrics['num_datums']} "
            f"elapsed={metrics['elapsed_s']:.1f}s"
        )

        # Save weights and get new sampling client every few steps
        if (step + 1) % 5 == 0 and step + 1 < num_steps:
            t0 = _now()
            name = f"rl_check_step_{step + 1}"
            save_future = training_client.save_weights_for_sampler(name=name)
            _wait_future(save_future, f"save_weights({name})", step, timeout_s)
            # save_weights_for_sampler registers a multi-LoRA sampling session
            # internally. Use the returned session for subsequent sampling.
            result = save_future.result()
            sampling_session_id = getattr(result, "sampling_session_id", None) or getattr(result, "session_id", None)
            if sampling_session_id:
                sampling_client = service_client.create_sampling_client(
                    sampling_session_id=sampling_session_id,
                )
            else:
                # Fallback: create from base model (no LoRA)
                sampling_client = service_client.create_sampling_client(
                    base_model=base_model,
                )
            print(f"  Weights saved + sampling client refreshed ({_now() - t0:.1f}s)")

    return all_metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RL e2e check for mint-server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", "-m", action="append", dest="models", required=True,
        help="Base model name (repeatable for multiple models)",
    )
    parser.add_argument("--steps", type=int, default=10, help="RL steps per model (default: 10)")
    parser.add_argument("--group-size", type=int, default=4, help="Samples per step (default: 4)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max generation tokens (default: 64)")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument(
        "--timeout-s", type=float,
        default=float(os.environ.get("MINT_TEST_TIMEOUT_S", "600")),
        help="Per-request timeout seconds (default: 600, env: MINT_TEST_TIMEOUT_S)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Write JSON results to this directory (default: /tmp/rl_check/<timestamp>)",
    )
    args = parser.parse_args()

    print(f"MINT_BASE_URL={os.environ.get('MINT_BASE_URL', 'not set')!r}")
    print(f"Timestamp: {_ts()}")

    # Create service client
    t0 = _now()
    service_client = mint.ServiceClient()
    print(f"Service client connected ({_now() - t0:.1f}s)")

    # List available models
    try:
        caps = service_client.get_server_capabilities()
        available = {m.model_name for m in caps.supported_models}
        print(f"Server supports {len(available)} models")
    except Exception as e:
        print(f"Warning: could not list models: {e}")
        available = set()

    # Output dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir or f"/tmp/rl_check/{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    all_results: dict[str, Any] = {
        "timestamp": _ts(),
        "base_url": os.environ.get("MINT_BASE_URL", "default"),
        "models": {},
    }

    exit_code = 0

    for model in args.models:
        if available and model not in available:
            print(f"\nSKIP: {model} not in server capabilities")
            all_results["models"][model] = {"status": "skipped", "reason": "not in capabilities"}
            continue

        try:
            metrics = run_one_model(
                service_client=service_client,
                base_model=model,
                num_steps=args.steps,
                group_size=args.group_size,
                learning_rate=args.lr,
                max_tokens=args.max_tokens,
                lora_rank=args.lora_rank,
                timeout_s=args.timeout_s,
            )

            rewards = [m["avg_reward"] for m in metrics]
            all_results["models"][model] = {
                "status": "pass" if all(m["num_datums"] > 0 for m in metrics) else "warn",
                "steps_completed": len(metrics),
                "final_reward": rewards[-1] if rewards else None,
                "avg_reward": round(sum(rewards) / len(rewards), 4) if rewards else None,
                "reward_trajectory": rewards,
                "metrics": metrics,
            }

            # Save per-model JSON
            (out_dir / f"{model.replace('/', '_')}.json").write_text(
                json.dumps(all_results["models"][model], indent=2)
            )

        except Exception as e:
            print(f"\nFAIL: {model}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            all_results["models"][model] = {"status": "fail", "error": str(e)}
            exit_code = 1

    # Write summary
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n{'='*60}")
    print("Summary:")
    for model, result in all_results["models"].items():
        status = result["status"]
        extra = ""
        if status == "pass":
            extra = f" final_reward={result.get('final_reward')}"
        elif status == "fail":
            extra = f" error={result.get('error', '')[:80]}"
        print(f"  {model}: {status}{extra}")
    print(f"Results: {summary_path}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
