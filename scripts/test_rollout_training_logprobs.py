#!/usr/bin/env python
"""Unit test to verify consistency between rollout logprobs and training logprobs.

This test ensures that:
1. Logprobs from sampling (used for importance ratio calculation)
2. Logprobs from forward_backward (used for new policy calculation)

Are consistent for the same model weights and token sequence.

Usage:
    python test_rollout_training_logprobs.py --model_name Qwen/Qwen2.5-7B-Instruct --max_tokens 4096
    python test_rollout_training_logprobs.py --model_name Qwen/Qwen3-30B-A3B-Instruct-2507 --max_tokens 4096
    python test_rollout_training_logprobs.py --model_name Qwen/Qwen3-30B-A3B-Instruct-2507 --max_tokens 4096 --exclude_last_token

Environment variables:
    TINKER_BASE_URL: Server base URL if --base_url is not specified
"""

import asyncio
import logging
import os
import sys

import tinker
import torch
from tinker import ModelInput, SamplingParams, TensorData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_rollout_training_logprobs_consistency(
    base_url: str | None = None,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    lora_rank: int = 8,
    exclude_last_token: bool = False,
    max_tokens: int = 50,
):
    """Test consistency between rollout logprobs and training logprobs."""
    print("=" * 60)
    print("Testing rollout logprobs vs training logprobs consistency")
    print("=" * 60)

    print(f"Creating training client: {model_name}, LoRA rank={lora_rank}")
    service_client = tinker.ServiceClient(base_url=base_url)
    training_client = await service_client.create_lora_training_client_async(
        base_model=model_name, rank=lora_rank
    )
    # Use AutoTokenizer directly with trust_remote_code for Moonlight
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    prompt_text = """Using the numbers [15, 16, 45], create an equation that equals 76.
You can use basic arithmetic operations (+, -, *, /) and each number can only be used once.
"""
    prompt_tokens = tokenizer.encode(prompt_text)
    prompt_model_input = ModelInput.from_ints(prompt_tokens)
    print(f"Prompt tokens: {len(prompt_tokens)}")

    print("Creating sampling client...")
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()

    print("Sampling sequence and getting logprobs...")
    sample_result = await sampling_client.sample_async(
        prompt=prompt_model_input,
        num_samples=1,
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0),
    )

    sampled_sequence = sample_result.sequences[0]
    sampled_tokens = sampled_sequence.tokens
    sampled_logprobs = sampled_sequence.logprobs

    if sampled_logprobs is None:
        raise ValueError("No logprobs in sampling result!")

    # print(f"Sampled tokens: {sampled_tokens}")
    # print(f"Sampled logprobs: {sampled_logprobs}")

    full_sequence_tokens = prompt_tokens + sampled_tokens
    full_sequence_text = tokenizer.decode(full_sequence_tokens)
    print(f"Full sequence: {full_sequence_text}")

    assert len(sampled_logprobs) == len(
        sampled_tokens
    ), f"sampled_logprobs length {len(sampled_logprobs)} != sampled_tokens length {len(sampled_tokens)}"

    full_logprobs = [0.0] * len(prompt_tokens) + sampled_logprobs
    # CRITICAL FIX: Pass full sequence to Megatron (including last token)
    # This shifts the "garbage rolled position" from N-1 to N
    # After roll: labels = [t₁, ..., t_N, t₀] where t_N is now CORRECT at position N-1
    # The garbage t₀ is at position N, which we mask out
    input_tokens = full_sequence_tokens  # Include last token
    target_tokens = full_sequence_tokens[1:] + [full_sequence_tokens[0]]  # Shifted, dummy at end
    target_logprobs = full_logprobs[1:] + [0.0]  # Shifted, dummy at end

    assert len(input_tokens) == len(target_tokens) == len(
        target_logprobs
    ), f"Length mismatch: input={len(input_tokens)}, target={len(target_tokens)}, logprobs={len(target_logprobs)}"

    advantages = [0.0] * len(prompt_tokens) + [1.0] * len(sampled_tokens)
    # Advantages also needs dummy at end to match length
    advantages = advantages[1:] + [0.0]
    mask = [0.0] * len(prompt_tokens) + [1.0] * len(sampled_tokens)
    mask = mask[1:] + [0.0]  # Dummy position masked out

    datum = tinker.Datum(
        model_input=ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "logprobs": TensorData.from_torch(torch.tensor(target_logprobs, dtype=torch.float32)),
            "advantages": TensorData.from_torch(torch.tensor(advantages, dtype=torch.float32)),
            "mask": TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
        },
    )

    print(f"Constructed Datum: input_len={len(input_tokens)}, target_len={len(target_tokens)}")

    # print("Running forward_backward to get training logprobs...")
    fwd_bwd_future = await training_client.forward_backward_async(
        [datum], loss_fn="importance_sampling"
    )
    fwd_bwd_result = await fwd_bwd_future.result_async()

    training_logprobs = fwd_bwd_result.loss_fn_outputs[0]["logprobs"].to_torch()
    # print(f"ForwardBackwardOutput.loss_fn_outputs.: len={len(training_logprobs)}")

    mask_tensor = torch.tensor(mask, dtype=torch.float32)
    sampled_logprobs_tensor = torch.tensor(target_logprobs, dtype=torch.float32)

    mask_bool = mask_tensor > 0.5
    sampled_logprobs_masked = sampled_logprobs_tensor[mask_bool]
    training_logprobs_masked = training_logprobs[mask_bool]

    # Exclude the last token from comparison
    if exclude_last_token:
        sampled_logprobs_masked = sampled_logprobs_masked[:-1]
        training_logprobs_masked = training_logprobs_masked[:-1]

    # print(f"Mask=1 positions: {mask_bool.sum().item()}")
    # print(f"Sampled logprobs (masked): {sampled_logprobs_masked.tolist()}")
    # print(f"Training logprobs (masked): {training_logprobs_masked.tolist()}")

    diff = sampled_logprobs_masked - training_logprobs_masked

    max_diff = torch.abs(diff).max().item()
    mean_diff = torch.abs(diff).mean().item()
    mse = (diff ** 2).mean().item()
    relative_diff = torch.abs(diff / (sampled_logprobs_masked.abs() + 1e-8))
    max_relative_diff = relative_diff.max().item()

    print("\nDetailed comparison (all tokens in mask):")
    masked_indices = torch.where(mask_bool)[0]
    if exclude_last_token:
        masked_indices = masked_indices[:-1]
    for i, idx in enumerate(masked_indices):
        token_idx = int(idx.item())
        token = target_tokens[token_idx]
        token_str = tokenizer.decode([token])
        sampled_lp = sampled_logprobs_tensor[token_idx].item()
        training_lp = training_logprobs[token_idx].item()
        diff_val = diff[i].item()
        rel_diff_val = relative_diff[i].item()
        print(
            f"  {token_idx} Token {token}: '{token_str}' "
            f"| sampled={sampled_lp:.6f}, training={training_lp:.6f}, diff={diff_val:.6f}, rel_diff={rel_diff_val:.2f}"
        )

    print("=" * 60)
    print("Comparison results:")
    print(f"  logprob diff mse: {mse:.6f}")
    print(f"  logprob diff mean: {mean_diff:.6f}")
    print(f"  logprob diff max: {max_diff:.6f}")
    print(f"  logprob diff max relative: {max_relative_diff:.6f}")
    print("=" * 60)

    tolerance = 1e-2
    if mean_diff < tolerance:
        print("✅ Test passed! Rollout logprobs and training logprobs are consistent")
        return True
    else:
        print(f"❌ Test failed! Mean diff {mean_diff} exceeds tolerance {tolerance}")
        print("This may cause incorrect importance ratio calculation, affecting RL training")
        return False


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test rollout and training logprobs consistency")
    parser.add_argument("--base_url", type=str, default=None, help="Tinker server base URL")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name",
    )
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--max_tokens", type=int, default=50, help="Max tokens")
    parser.add_argument("--exclude_last_token", action="store_true", help="Exclude last token from comparison")

    args = parser.parse_args()

    base_url = args.base_url or os.environ.get("TINKER_BASE_URL")

    success = await test_rollout_training_logprobs_consistency(
        base_url=base_url, model_name=args.model_name, lora_rank=args.lora_rank, max_tokens=args.max_tokens, exclude_last_token=args.exclude_last_token
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
