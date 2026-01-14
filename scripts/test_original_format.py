#!/usr/bin/env python3
"""Test logprob comparison using ORIGINAL cookbook datum format.

The original cookbook format:
- input = full_sequence[:-1] (remove last token)
- target = full_sequence[1:] (remove first token)
- len(input) == len(target)

This differs from the "full-sequence" format I tried earlier.
"""

import asyncio
import os
import torch

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker


async def get_megatron_logprobs_original_format(training_client, full_tokens, prompt_len):
    """Get logprobs from Megatron using ORIGINAL cookbook format."""
    # Original format: input=[:-1], target=[1:]
    input_tokens = full_tokens[:-1]
    target_tokens = full_tokens[1:]

    # Mask: 0 for prompt positions, 1 for response positions
    # After removing last token, prompt_len stays same, response shrinks by 1
    # Position i in input predicts target[i] = full_tokens[i+1]
    # Response tokens are full_tokens[prompt_len:], so positions predicting them are prompt_len-1, prompt_len, ...
    # Actually: position prompt_len-1 predicts full_tokens[prompt_len] (first response)
    # position len(input)-1 predicts full_tokens[len(full)-1] = last token
    mask = [0.0] * (prompt_len - 1) + [1.0] * (len(input_tokens) - prompt_len + 1)

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.zeros(len(input_tokens), dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.zeros(len(input_tokens), dtype=torch.float32)),
        }
    )

    fwd_future = await training_client.forward_async([datum], loss_fn="importance_sampling")
    fwd_result = await fwd_future.result_async()
    return fwd_result.loss_fn_outputs[0]["logprobs"].to_torch().tolist()


async def do_sft_step_original_format(training_client, full_tokens, prompt_len, lr=5e-5):
    """Do one SFT step using ORIGINAL cookbook format."""
    input_tokens = full_tokens[:-1]
    target_tokens = full_tokens[1:]

    # Mask for response positions (predicting response tokens)
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

    fwd_bwd = await training_client.forward_backward_async([datum], loss_fn="importance_sampling")
    await fwd_bwd.result_async()
    optim = await training_client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=lr))
    await optim.result_async()


async def main():
    print("=" * 70)
    print("TEST: Original Cookbook Datum Format")
    print("Format: input=[:-1], target=[1:]")
    print("=" * 70)

    model_name = "moonshotai/Moonlight-16B-A3B-Instruct"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Test sequence
    prompt = "<|im_start|>user\nWhat is 2+3?<|im_end|>\n<|im_start|>assistant\n"
    response = "5<|im_end|>"

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    full_tokens = tokenizer.encode(prompt + response, add_special_tokens=False)
    prompt_len = len(prompt_tokens)
    response_tokens = full_tokens[prompt_len:]

    print(f"\nPrompt: {prompt!r}")
    print(f"Response: {response!r}")
    print(f"Prompt length: {prompt_len}, Response length: {len(response_tokens)}, Full length: {len(full_tokens)}")
    print(f"Response tokens: {response_tokens} = {[tokenizer.decode([t]) for t in response_tokens]}")

    # Create training client
    print("\n[1] Creating training session...")
    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])
    training_client = await service_client.create_lora_training_client_async(model_name, rank=16)

    # Get logprobs BEFORE training
    print("\n[2] Megatron logprobs BEFORE training (original format)...")
    lp_before = await get_megatron_logprobs_original_format(training_client, full_tokens, prompt_len)

    print("\n  Response token logprobs BEFORE:")
    for i, tok in enumerate(response_tokens):
        # Position that predicts this token: prompt_len - 1 + i
        pos = prompt_len - 1 + i
        if pos < len(lp_before):
            print(f"    pos {pos} predicts {tok} ({tokenizer.decode([tok])!r}) -> logprob {lp_before[pos]:.4f}")

    # Do SFT steps
    num_steps = 10
    print(f"\n[3] Doing {num_steps} SFT steps...")
    for i in range(num_steps):
        await do_sft_step_original_format(training_client, full_tokens, prompt_len, lr=1e-4)
        if (i + 1) % 5 == 0:
            print(f"    Step {i+1}/{num_steps} done")

    # Get logprobs AFTER training
    print("\n[4] Megatron logprobs AFTER training (original format)...")
    lp_after = await get_megatron_logprobs_original_format(training_client, full_tokens, prompt_len)

    print("\n  Response token logprobs AFTER:")
    for i, tok in enumerate(response_tokens):
        pos = prompt_len - 1 + i
        if pos < len(lp_after):
            print(f"    pos {pos} predicts {tok} ({tokenizer.decode([tok])!r}) -> logprob {lp_after[pos]:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  {'Token':>12} | {'BEFORE':>10} | {'AFTER':>10} | {'CHANGE':>10}")
    print(f"  {'-'*50}")

    total_change = 0
    for i, tok in enumerate(response_tokens):
        pos = prompt_len - 1 + i
        if pos < len(lp_before) and pos < len(lp_after):
            before = lp_before[pos]
            after = lp_after[pos]
            change = after - before
            total_change += change
            indicator = "+" if change > 0 else ""
            print(f"  {tokenizer.decode([tok]):>12} | {before:>10.4f} | {after:>10.4f} | {indicator}{change:>9.4f}")

    n = len(response_tokens)
    print(f"\n  Mean logprob change: {total_change / n if n > 0 else 0:+.4f}")

    if total_change > 0:
        print("\n  RESULT: Logprobs INCREASED (SFT working correctly)")
    else:
        print("\n  RESULT: Logprobs DECREASED (unexpected)")

    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
