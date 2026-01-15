#!/usr/bin/env python3
"""Reproduce GitHub Issue #51 - TensorData compatibility in RL training.

Tests whether using TensorData.from_torch() vs plain lists causes issues
when running against MinT with Tinker SDK.

Usage:
    # Test with tinker SDK
    TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=dummy python scripts/reproduce_issue51.py

    # Test with mint SDK
    MINT_BASE_URL=http://localhost:8000 MINT_API_KEY=dummy python scripts/reproduce_issue51.py --mint
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mint", action="store_true", help="Use mint SDK instead of tinker")
    args = parser.parse_args()

    if args.mint:
        print("Testing with MINT SDK")
        import mint as sdk
        from mint import TensorData
        from mint import types
        base_url = os.environ.get("MINT_BASE_URL", "http://localhost:8000")
        api_key = os.environ.get("MINT_API_KEY", "dummy")
    else:
        print("Testing with TINKER SDK")
        import tinker as sdk
        from tinker import TensorData
        from tinker import types
        base_url = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
        api_key = os.environ.get("TINKER_API_KEY", "dummy")

    import torch

    print(f"TensorData class: {TensorData}")
    print()

    # Create service client
    service_client = sdk.ServiceClient()

    # Create training client - use different models for MinT vs Tinker
    # If no TINKER_BASE_URL set, SDK uses official Tinker server
    is_official_tinker = not args.mint and not os.environ.get("TINKER_BASE_URL")
    if is_official_tinker:
        BASE_MODEL = "meta-llama/Llama-3.2-1B"  # Tinker supports this
        print("Detected: Official Tinker server")
    else:
        BASE_MODEL = "Qwen/Qwen3-0.6B"  # MinT supports this
        print("Detected: MinT server")
    print(f"Creating training client for {BASE_MODEL}...")
    training_client = service_client.create_lora_training_client(
        base_model=BASE_MODEL,
        rank=16,
    )
    tokenizer = training_client.get_tokenizer()
    print("Training client created.\n")

    # Test 1: SFT-style datum with plain lists (should work)
    print("=" * 60)
    print("TEST 1: SFT-style with plain lists")
    print("=" * 60)

    prompt = "Question: What is 2 + 2?\nAnswer:"
    completion = " 4"

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False) + [tokenizer.eos_token_id]

    all_tokens = prompt_tokens + completion_tokens
    input_tokens = all_tokens[:-1]
    target_tokens = all_tokens[1:]
    weights = [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(completion_tokens)

    sft_datum = types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={
            "target_tokens": target_tokens,  # plain list
            "weights": weights,              # plain list
        }
    )

    try:
        result = training_client.forward_backward([sft_datum], loss_fn="cross_entropy").result()
        print(f"SUCCESS: loss_fn_outputs count = {len(result.loss_fn_outputs)}")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
    print()

    # Test 2: RL-style datum with TensorData.from_torch() (issue claims this fails)
    print("=" * 60)
    print("TEST 2: RL-style with TensorData.from_torch()")
    print("=" * 60)

    # Simulating RL training data (like in notebook)
    logprobs = [0.0] * len(input_tokens)
    advantages = [0.0] * (len(prompt_tokens) - 1) + [0.5] * len(completion_tokens)

    rl_datum = types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.int64)),
            "weights": TensorData.from_torch(torch.tensor(weights, dtype=torch.float32)),
            "logprobs": TensorData.from_torch(torch.tensor(logprobs, dtype=torch.float32)),
            "advantages": TensorData.from_torch(torch.tensor(advantages, dtype=torch.float32)),
        },
    )

    try:
        result = training_client.forward_backward([rl_datum], loss_fn="importance_sampling").result()
        print(f"SUCCESS: loss_fn_outputs count = {len(result.loss_fn_outputs)}")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    print()

    # Test 3: RL-style with plain lists (proposed fix)
    print("=" * 60)
    print("TEST 3: RL-style with plain lists (proposed fix)")
    print("=" * 60)

    rl_datum_fixed = types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={
            "target_tokens": target_tokens,
            "weights": weights,
            "logprobs": logprobs,
            "advantages": advantages,
        },
    )

    try:
        result = training_client.forward_backward([rl_datum_fixed], loss_fn="importance_sampling").result()
        print(f"SUCCESS: loss_fn_outputs count = {len(result.loss_fn_outputs)}")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
