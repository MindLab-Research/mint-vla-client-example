#!/usr/bin/env python3
"""Test if LoRA is actually applied during Megatron forward pass.

Hypothesis: LoRA might not be applied during forward-only mode (eval_mode).

Test:
1. Fresh LoRA → Megatron forward → logprobs_fresh
2. Train 5 steps (LoRA weights become non-zero)
3. Trained LoRA → Megatron forward (NO export to vLLM) → logprobs_trained
4. Compare: if logprobs_fresh ≈ logprobs_trained, LoRA is NOT applied during forward

If LoRA isn't applied, Megatron would always output base model logprobs,
regardless of training, which would explain the divergence from vLLM.
"""

import asyncio
import os

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker
import torch


async def get_megatron_logprobs(training_client, tokenizer, tokens):
    """Get Megatron logprobs for a sequence WITHOUT involving vLLM."""
    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.ones(len(tokens), dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.zeros(len(tokens), dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.zeros(len(tokens), dtype=torch.float32)),
        }
    )

    fwd_future = await training_client.forward_async([datum], loss_fn="importance_sampling")
    fwd_result = await fwd_future.result_async()

    if not fwd_result.loss_fn_outputs:
        return None

    return fwd_result.loss_fn_outputs[0]["logprobs"].to_torch().tolist()


async def do_training_step(training_client, tokenizer, prompt: str, response: str):
    """Do a single training step."""
    full_text = prompt + response
    tokens = tokenizer.encode(full_text, add_special_tokens=False)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

    prompt_len = len(prompt_tokens)
    mask = [0.0] * prompt_len + [1.0] * (len(tokens) - prompt_len)

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.ones(len(tokens), dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.zeros(len(tokens), dtype=torch.float32)),
        }
    )

    fwd_bwd_future = await training_client.forward_backward_async([datum], loss_fn="importance_sampling")
    await fwd_bwd_future.result_async()

    optim_future = await training_client.optim_step_async(
        adam_params=tinker.AdamParams(learning_rate=1e-4)  # Higher LR for visible effect
    )
    await optim_future.result_async()


async def main():
    print("=" * 70)
    print("TEST: Is LoRA applied during Megatron forward?")
    print("=" * 70)

    model_name = "moonshotai/Moonlight-16B-A3B-Instruct"
    base_url = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Test sequence - simple math question
    test_prompt = "<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n<|im_start|>assistant\n"
    test_response = "4<|im_end|>"
    test_tokens = tokenizer.encode(test_prompt + test_response, add_special_tokens=False)

    print(f"\nTest sequence: {len(test_tokens)} tokens")
    print(f"Decoded: {tokenizer.decode(test_tokens)[:100]!r}...")

    # Create training session
    print("\n[1] Creating training session with fresh LoRA...")
    service_client = tinker.ServiceClient(base_url=base_url)
    training_client = await service_client.create_lora_training_client_async(model_name, rank=16)

    # Get Megatron logprobs with FRESH LoRA
    print("\n[2] Getting Megatron logprobs with FRESH LoRA (before training)...")
    logprobs_fresh = await get_megatron_logprobs(training_client, tokenizer, test_tokens)

    if not logprobs_fresh:
        print("ERROR: Could not get logprobs")
        return

    print(f"    First 10 logprobs: {logprobs_fresh[:10]}")

    # Train 5 steps on related data
    print("\n[3] Training 5 steps to update LoRA weights...")
    training_examples = [
        ("<|im_start|>user\nWhat is 1+1?<|im_end|>\n<|im_start|>assistant\n", "2<|im_end|>"),
        ("<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n", "4<|im_end|>"),
        ("<|im_start|>user\nWhat is 3+3?<|im_end|>\n<|im_start|>assistant\n", "6<|im_end|>"),
        ("<|im_start|>user\nWhat is 4+4?<|im_end|>\n<|im_start|>assistant\n", "8<|im_end|>"),
        ("<|im_start|>user\nWhat is 5+5?<|im_end|>\n<|im_start|>assistant\n", "10<|im_end|>"),
    ]

    for i, (prompt, response) in enumerate(training_examples):
        await do_training_step(training_client, tokenizer, prompt, response)
        print(f"    Step {i+1}/5 completed")

    # Get Megatron logprobs with TRAINED LoRA (NO export to vLLM)
    print("\n[4] Getting Megatron logprobs with TRAINED LoRA (NO vLLM involved)...")
    logprobs_trained = await get_megatron_logprobs(training_client, tokenizer, test_tokens)

    if not logprobs_trained:
        print("ERROR: Could not get logprobs")
        return

    print(f"    First 10 logprobs: {logprobs_trained[:10]}")

    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON: Fresh LoRA vs Trained LoRA (Megatron only, no vLLM)")
    print("=" * 70)

    diffs = [t - f for f, t in zip(logprobs_fresh, logprobs_trained)]

    print(f"\n{'pos':>4} | {'token':>8} | {'text':>12} | {'fresh':>10} | {'trained':>10} | {'diff':>10}")
    print("-" * 70)

    prompt_len = len(tokenizer.encode(test_prompt, add_special_tokens=False))
    for i in range(min(20, len(test_tokens))):
        tok = test_tokens[i]
        text = tokenizer.decode([tok])
        f = logprobs_fresh[i]
        t = logprobs_trained[i]
        d = diffs[i]
        marker = " *** " if abs(d) > 0.1 else ""
        print(f"{i:>4} | {tok:>8} | {repr(text):>12} | {f:>10.4f} | {t:>10.4f} | {d:>+10.4f}{marker}")

    # Summary
    mean_diff = sum(abs(d) for d in diffs) / len(diffs)
    max_diff = max(abs(d) for d in diffs)
    significant = sum(1 for d in diffs if abs(d) > 0.1)

    print("\n" + "=" * 70)
    print(f"Mean |diff|: {mean_diff:.4f}")
    print(f"Max |diff|: {max_diff:.4f}")
    print(f"Significant diffs (|d| > 0.1): {significant}/{len(diffs)}")
    print("=" * 70)

    if max_diff < 0.01:
        print("\nCONCLUSION: LoRA is NOT applied during Megatron forward!")
        print("  - Fresh and trained logprobs are nearly identical")
        print("  - This explains why Megatron diverges from vLLM after training")
        print("  - BUG: eval_mode or forward_only doesn't apply LoRA weights")
    elif significant > len(diffs) * 0.2:
        print("\nCONCLUSION: LoRA IS applied during Megatron forward")
        print("  - Logprobs changed significantly after training")
        print("  - Bug must be elsewhere (export, alignment, etc.)")
    else:
        print("\nCONCLUSION: Mixed - some positions affected, others not")
        print("  - LoRA may be partially applied")
        print("  - Or training didn't affect these positions much")


if __name__ == "__main__":
    asyncio.run(main())
