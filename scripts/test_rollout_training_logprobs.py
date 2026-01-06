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

    # Only compare logprobs on generated tokens (not prompt tokens)
    # After the full-sequence fix:
    # - input_tokens has len(prompt) + len(sampled) tokens
    # - training_logprobs has same length
    # - We only have vLLM logprobs for sampled tokens (positions len(prompt) to len(prompt)+len(sampled)-1)
    # Note: position indices are BEFORE the [1:] shift that was applied to mask/target_logprobs

    # Extract logprobs for generated token positions only
    # The shift means: original position i corresponds to shifted position i-1
    # Generated tokens start at original position len(prompt_tokens), which is shifted position len(prompt_tokens)-1
    gen_start = len(prompt_tokens) - 1  # First generated token position after shift
    gen_end = gen_start + len(sampled_tokens)  # One past last generated token

    # Get vLLM logprobs (already shifted in target_logprobs)
    vllm_logprobs = torch.tensor(target_logprobs[gen_start:gen_end], dtype=torch.float32)
    megatron_logprobs = training_logprobs[gen_start:gen_end]

    # Sanity check
    assert len(vllm_logprobs) == len(sampled_tokens), f"vLLM logprobs length mismatch: {len(vllm_logprobs)} vs {len(sampled_tokens)}"
    assert len(megatron_logprobs) == len(sampled_tokens), f"Megatron logprobs length mismatch: {len(megatron_logprobs)} vs {len(sampled_tokens)}"

    diff = vllm_logprobs - megatron_logprobs

    max_diff = torch.abs(diff).max().item()
    mean_diff = torch.abs(diff).mean().item()
    mse = (diff ** 2).mean().item()
    relative_diff = torch.abs(diff / (vllm_logprobs.abs() + 1e-8))
    max_relative_diff = relative_diff.max().item()

    print(f"\nDetailed comparison (generated tokens only, {len(sampled_tokens)} tokens):")
    for i in range(len(sampled_tokens)):
        token = sampled_tokens[i]
        token_str = tokenizer.decode([token])
        vllm_lp = vllm_logprobs[i].item()
        meg_lp = megatron_logprobs[i].item()
        d = diff[i].item()
        rel = relative_diff[i].item()
        print(f"  {i} Token {token}: {repr(token_str)} | vllm={vllm_lp:.6f}, megatron={meg_lp:.6f}, diff={d:.6f}, rel_diff={rel:.2f}")

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
