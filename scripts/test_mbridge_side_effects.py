#!/usr/bin/env python3
"""Test if mbridge export has side effects on Megatron's forward pass.

Purpose: Determine if calling export_adapter_weights corrupts Megatron's state.

Test procedure:
1. Create training session with fresh LoRA
2. Do a few training steps
3. Call Megatron forward and record logprobs (BEFORE export)
4. Call export_adapter_weights (mbridge export)
5. Call Megatron forward and record logprobs (AFTER export)
6. Compare - if different, export has side effects
"""

import asyncio
import os
import sys
import time

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker
import torch


async def get_megatron_logprobs(training_client: tinker.TrainingClient, tokenizer, prompt: str, response: str):
    """Get Megatron logprobs for a prompt+response pair."""
    full_text = prompt + response
    tokens = tokenizer.encode(full_text, add_special_tokens=False)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_len = len(prompt_tokens)

    # SFT format
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    mask = [0.0] * (prompt_len - 1) + [1.0] * (len(input_tokens) - prompt_len + 1)
    logprobs = [0.0] * len(input_tokens)

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.zeros(len(input_tokens), dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.tensor(logprobs, dtype=torch.float32)),
        }
    )

    fwd_future = await training_client.forward_async([datum], loss_fn="importance_sampling")
    fwd_result = await fwd_future.result_async()

    if not fwd_result.loss_fn_outputs:
        return None

    return fwd_result.loss_fn_outputs[0]["logprobs"].to_torch().tolist()


async def do_training_step(training_client: tinker.TrainingClient, tokenizer, prompt: str, response: str):
    """Do a single training step using SFT format."""
    full_text = prompt + response
    tokens = tokenizer.encode(full_text, add_special_tokens=False)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_len = len(prompt_tokens)

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    mask = [0.0] * (prompt_len - 1) + [1.0] * (len(input_tokens) - prompt_len + 1)
    advantages = [0.0] * (prompt_len - 1) + [1.0] * (len(input_tokens) - prompt_len + 1)
    logprobs = [0.0] * len(input_tokens)

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.tensor(advantages, dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.tensor(logprobs, dtype=torch.float32)),
        }
    )

    fwd_bwd_future = await training_client.forward_backward_async([datum], loss_fn="importance_sampling")
    await fwd_bwd_future.result_async()

    optim_future = await training_client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=5e-5))
    await optim_future.result_async()


async def main():
    print("=" * 70)
    print("MBRIDGE EXPORT SIDE EFFECTS TEST")
    print("=" * 70)

    model_name = "moonshotai/Moonlight-16B-A3B-Instruct"
    base_url = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")

    # Load tokenizer
    print("\n[1] Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Create training client
    print("\n[2] Creating training session with fresh LoRA...")
    service_client = tinker.ServiceClient(base_url=base_url)
    training_client = await service_client.create_lora_training_client_async(model_name, rank=16)

    # Do a few training steps
    print("\n[3] Doing 3 training steps...")
    training_examples = [
        ("<|im_start|>user\nWhat is 7+8?<|im_end|>\n<|im_start|>assistant\n", "The answer is 15.<|im_end|>"),
        ("<|im_start|>user\nWhat is 9+6?<|im_end|>\n<|im_start|>assistant\n", "The answer is 15.<|im_end|>"),
        ("<|im_start|>user\nWhat is 5+10?<|im_end|>\n<|im_start|>assistant\n", "The answer is 15.<|im_end|>"),
    ]

    for i, (prompt, response) in enumerate(training_examples):
        t0 = time.time()
        await do_training_step(training_client, tokenizer, prompt, response)
        print(f"  Step {i+1}/3 completed in {time.time()-t0:.1f}s")

    # Test prompt/response
    test_prompt = "<|im_start|>user\nWhat is 7+8?<|im_end|>\n<|im_start|>assistant\n"
    test_response = "The answer is 15.<|im_end|>"

    # Get Megatron logprobs BEFORE export
    print("\n[4] Getting Megatron logprobs BEFORE export...")
    logprobs_before = await get_megatron_logprobs(training_client, tokenizer, test_prompt, test_response)
    print(f"  Got {len(logprobs_before)} logprobs")

    # Trigger mbridge export by calling save_weights_and_get_sampling_client_async
    print("\n[5] Triggering mbridge export (save_weights_and_get_sampling_client_async)...")
    t0 = time.time()
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()
    print(f"  Export completed in {time.time()-t0:.1f}s")

    # Get Megatron logprobs AFTER export
    print("\n[6] Getting Megatron logprobs AFTER export...")
    logprobs_after = await get_megatron_logprobs(training_client, tokenizer, test_prompt, test_response)
    print(f"  Got {len(logprobs_after)} logprobs")

    # Compare logprobs
    print("\n" + "=" * 70)
    print("COMPARISON: BEFORE vs AFTER export")
    print("=" * 70)

    # Get response token positions
    prompt_tokens = tokenizer.encode(test_prompt, add_special_tokens=False)
    prompt_len = len(prompt_tokens)
    full_tokens = tokenizer.encode(test_prompt + test_response, add_special_tokens=False)
    response_tokens = full_tokens[prompt_len:]

    print(f"\nResponse tokens: {[tokenizer.decode([t]) for t in response_tokens[:10]]}")
    print(f"\n{'pos':>4} | {'tok':>8} | {'text':>15} | {'BEFORE':>12} | {'AFTER':>12} | {'DIFF':>12}")
    print("-" * 80)

    diffs = []
    for i in range(min(20, len(response_tokens))):
        pos = prompt_len - 1 + i  # SFT format position
        tok = response_tokens[i] if i < len(response_tokens) else 0
        text = tokenizer.decode([tok])

        before = logprobs_before[pos] if pos < len(logprobs_before) else float('nan')
        after = logprobs_after[pos] if pos < len(logprobs_after) else float('nan')
        diff = after - before

        diffs.append(diff)
        print(f"{pos:>4} | {tok:>8} | {repr(text):>15} | {before:>12.4f} | {after:>12.4f} | {diff:>+12.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if diffs:
        max_diff = max(abs(d) for d in diffs)
        mean_diff = sum(abs(d) for d in diffs) / len(diffs)
        print(f"\nMax absolute diff: {max_diff:.6f}")
        print(f"Mean absolute diff: {mean_diff:.6f}")

        if max_diff > 0.01:
            print("\n>>> WARNING: Export has SIDE EFFECTS! Logprobs changed after export.")
        else:
            print("\n>>> OK: No significant side effects detected.")
    else:
        print("\nNo comparison data available.")


if __name__ == "__main__":
    asyncio.run(main())
