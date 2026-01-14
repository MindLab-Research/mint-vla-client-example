#!/usr/bin/env python3
"""Load checkpoint into FRESH Megatron and check logprobs.

The hypothesis: Same weights produce correct results in vLLM but wrong in Megatron.
This script tests whether loading the checkpoint into a fresh Megatron shows the same bug.
"""

import asyncio
import json
import os

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker
import torch
import numpy as np

MODEL_NAME = "moonshotai/Moonlight-16B-A3B-Instruct"


async def main():
    from transformers import AutoTokenizer

    # Load saved data
    print("Loading saved data...")
    with open("/tmp/debug_checkpoint_data.json", "r") as f:
        data = json.load(f)

    input_tokens = data["input_tokens"]
    target_tokens = data["target_tokens"]
    corrupted_positions = data["corrupted_positions"]
    megatron_fresh_original = data["megatron_fresh"]
    megatron_trained_original = data["megatron_trained"]
    checkpoint_path = data["checkpoint_path"]

    print(f"Corrupted positions: {corrupted_positions}")
    print(f"Checkpoint: {checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    # Create a FRESH Megatron training client
    print("\n" + "=" * 70)
    print("Creating FRESH Megatron training client")
    print("=" * 70)
    fresh_client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)

    mask = [1.0] * len(input_tokens)
    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.ones(len(input_tokens), dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.zeros(len(input_tokens), dtype=torch.float32)),
        }
    )

    # Get FRESH Megatron logprobs
    print("\n" + "=" * 70)
    print("Getting FRESH Megatron logprobs (before loading checkpoint)")
    print("=" * 70)
    fwd = await fresh_client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    fresh_logprobs = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())
    print(f"Fresh logprobs at corrupted positions:")
    for pos in corrupted_positions[:5]:
        target_str = tokenizer.decode([target_tokens[pos]])
        print(f"  pos {pos}: {fresh_logprobs[pos]:.4f} (target={repr(target_str)})")

    # Now load the checkpoint
    print("\n" + "=" * 70)
    print("Loading checkpoint into Megatron")
    print("=" * 70)
    # The load_state API should load the checkpoint
    load_result = await (await fresh_client.load_state_async(checkpoint_path)).result_async()
    print(f"Load result: {load_result}")

    # Get logprobs AFTER loading checkpoint
    print("\n" + "=" * 70)
    print("Getting Megatron logprobs AFTER loading checkpoint")
    print("=" * 70)
    fwd = await fresh_client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    loaded_logprobs = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    # Compare
    print("\n" + "=" * 100)
    print("COMPARISON: Fresh vs Loaded Checkpoint vs Original Trained")
    print("=" * 100)
    print(f"{'Pos':<4} {'Target':<12} {'Fresh Now':<12} {'Loaded Now':<12} {'Orig Fresh':<12} {'Orig Train':<12}")
    print("-" * 80)

    for pos in corrupted_positions:
        target_str = repr(tokenizer.decode([target_tokens[pos]]))[:10]
        fresh_now = fresh_logprobs[pos]
        loaded_now = loaded_logprobs[pos]
        orig_fresh = megatron_fresh_original[pos]
        orig_train = megatron_trained_original[pos]

        print(f"{pos:<4} {target_str:<12} {fresh_now:<12.4f} {loaded_now:<12.4f} {orig_fresh:<12.4f} {orig_train:<12.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    loaded_matches_trained = np.allclose(loaded_logprobs, megatron_trained_original, rtol=0.1)
    print(f"Loaded logprobs match original trained: {loaded_matches_trained}")

    if loaded_matches_trained:
        print("BUG CONFIRMED: Loading checkpoint into fresh Megatron reproduces the issue!")
        print("The bug is in how Megatron loads/applies the LoRA weights, not in training.")
    else:
        print("Loaded logprobs differ from original trained - need more investigation")


if __name__ == "__main__":
    asyncio.run(main())
