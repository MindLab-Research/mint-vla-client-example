#!/usr/bin/env python
"""Detailed analysis of logprob mismatch between sampling and training."""

import asyncio
import os

import tinker
import torch
from tinker import ModelInput, SamplingParams, TensorData


async def main():
    base_url = os.environ.get("TINKER_BASE_URL", "http://localhost:18000")
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    lora_rank = 8
    max_tokens = 100

    print(f"base_url: {base_url}")
    print(f"model: {model_name}")

    service_client = tinker.ServiceClient(base_url=base_url)
    training_client = await service_client.create_lora_training_client_async(
        base_model=model_name, rank=lora_rank
    )
    tokenizer = training_client.get_tokenizer()

    prompt_text = """Using the numbers [15, 16, 45], create an equation that equals 76.
You can use basic arithmetic operations (+, -, *, /) and each number can only be used once.
"""
    prompt_tokens = tokenizer.encode(prompt_text)

    sampling_client = await training_client.save_weights_and_get_sampling_client_async()

    sample_result = await sampling_client.sample_async(
        prompt=ModelInput.from_ints(prompt_tokens),
        num_samples=1,
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0),
    )

    sampled_tokens = sample_result.sequences[0].tokens
    sampled_logprobs = sample_result.sequences[0].logprobs

    full_tokens = prompt_tokens + sampled_tokens
    full_logprobs = [0.0] * len(prompt_tokens) + sampled_logprobs

    input_tokens = full_tokens[:-1]
    target_tokens = full_tokens[1:]
    target_logprobs = full_logprobs[1:]
    mask = [0.0] * len(prompt_tokens) + [1.0] * len(sampled_tokens)
    mask = mask[1:]
    advantages = mask.copy()

    datum = tinker.Datum(
        model_input=ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "logprobs": TensorData.from_torch(torch.tensor(target_logprobs, dtype=torch.float32)),
            "advantages": TensorData.from_torch(torch.tensor(advantages, dtype=torch.float32)),
            "mask": TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
        },
    )

    fwd_bwd_future = await training_client.forward_backward_async([datum], loss_fn="importance_sampling")
    fwd_bwd_result = await fwd_bwd_future.result_async()
    training_logprobs = fwd_bwd_result.loss_fn_outputs[0]["logprobs"].to_torch()

    print("\n" + "=" * 100)
    print("TOKEN-BY-TOKEN COMPARISON (response tokens only)")
    print("=" * 100)
    print(f"{'Pos':>4} {'Token':>8} {'Text':>15} {'Sampled':>12} {'Training':>12} {'Diff':>12} {'AbsDiff':>10}")
    print("-" * 100)

    mask_tensor = torch.tensor(mask)
    masked_indices = (mask_tensor > 0.5).nonzero(as_tuple=True)[0]

    abs_diffs = []
    for i, idx in enumerate(masked_indices):
        idx = int(idx)
        token = target_tokens[idx]
        token_str = tokenizer.decode([token]).replace("\n", "\\n")[:15]
        sampled_lp = target_logprobs[idx]
        training_lp = training_logprobs[idx].item()
        diff = sampled_lp - training_lp
        abs_diff = abs(diff)
        abs_diffs.append(abs_diff)

        marker = "  "
        if abs_diff > 0.1:
            marker = "**"
        elif abs_diff > 0.01:
            marker = "* "

        print(f"{idx:>4} {token:>8} {token_str:>15} {sampled_lp:>12.6f} {training_lp:>12.6f} {diff:>12.6f} {abs_diff:>10.6f} {marker}")

    abs_diffs_tensor = torch.tensor(abs_diffs)
    print("-" * 100)
    print(f"Response tokens: {len(masked_indices)}")
    print(f"Mean abs diff: {abs_diffs_tensor.mean():.6f}")
    print(f"Max abs diff: {abs_diffs_tensor.max():.6f}")
    print(f"Std abs diff: {abs_diffs_tensor.std():.6f}")

    # Show last 5 tokens specifically
    print("\n" + "=" * 60)
    print("LAST 5 TOKENS (potential boundary issue):")
    print("=" * 60)
    for i in range(-5, 0):
        if abs(i) <= len(masked_indices):
            idx = int(masked_indices[i])
            token = target_tokens[idx]
            token_str = tokenizer.decode([token]).replace("\n", "\\n")
            sampled_lp = target_logprobs[idx]
            training_lp = training_logprobs[idx].item()
            diff = sampled_lp - training_lp
            print(f"  Position {idx}: token={token} '{token_str}' sampled={sampled_lp:.6f} training={training_lp:.6f} diff={diff:.6f}")


if __name__ == "__main__":
    asyncio.run(main())
