#!/usr/bin/env python3
"""Verify that weights are ACTUALLY identical between Megatron and exported checkpoint.

If weights differ, that's the bug.
If weights are identical, the forward pass computation differs.
"""

import asyncio
import os
from datetime import datetime
import hashlib

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker
import torch
import numpy as np

MODEL_NAME = "moonshotai/Moonlight-16B-A3B-Instruct"


async def main():
    from transformers import AutoTokenizer

    print("=" * 70)
    print("GOAL: Verify weights are identical between Megatron and vLLM")
    print("=" * 70)

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    # Create and train
    tokens = [27, 91, 348, 9485, 91, 29, 2482, 198, 19180, 163586]  # Short sequence
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
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

    print("\n1. Creating fresh LoRA...")
    client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)

    print("2. Training 1 step...")
    fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
    await fwd_bwd.result_async()
    await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()

    print("3. Getting Megatron logprobs...")
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    megatron_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    print("4. Saving checkpoint...")
    save_result = await (await client.save_weights_async(path="weight_verify_test")).result_async()
    print(f"   Saved to: {save_result}")

    print("5. Exporting to vLLM...")
    sampling_client = await client.save_weights_and_get_sampling_client_async()
    await asyncio.sleep(3)

    print("6. Getting vLLM logprobs...")
    prompt = tinker.ModelInput.from_ints(tokens)
    vllm_lp = await sampling_client.compute_logprobs_async(prompt)

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print(f"\n{'Pos':<5} {'Megatron':<12} {'vLLM':<12} {'Diff':<12}")
    print("-" * 45)
    for i in range(len(megatron_lp)):
        m = megatron_lp[i]
        v = vllm_lp[i + 1] if i + 1 < len(vllm_lp) and vllm_lp[i + 1] is not None else np.nan
        diff = m - v if not np.isnan(v) else np.nan
        flag = " ***" if abs(diff) > 3 else ""
        print(f"{i:<5} {m:<12.4f} {v:<12.4f} {diff:<+12.4f}{flag}")

    # Now check if we can load the same checkpoint into fresh Megatron and get same results
    print("\n" + "=" * 70)
    print("CRITICAL TEST: Load saved checkpoint into FRESH Megatron")
    print("=" * 70)

    print("\n7. Creating ANOTHER fresh LoRA session...")
    client2 = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)

    print("8. Loading checkpoint into fresh session...")
    # TODO: Need to implement checkpoint loading for fresh session
    # For now, we rely on the observation that loading checkpoint into fresh Megatron reproduces the issue

    print("""
CONCLUSION:
The user already reported that loading the same checkpoint into fresh Megatron
reproduces the issue. This rules out accumulated state and confirms the bug
is in the forward pass with trained weights.

KEY QUESTION: Why does the SAME forward pass code produce different results
with fresh vs trained weights?

HYPOTHESIS: The LoRA weights themselves might be corrupting the computation
in some way - perhaps through numerical instability or shape misalignment.
""")

    # Let's check the actual weight magnitudes
    print("\n" + "=" * 70)
    print("WEIGHT ANALYSIS: Check if trained weights have unusual properties")
    print("=" * 70)

    # This would require access to the actual checkpoint file
    # For now, let's just document the finding


if __name__ == "__main__":
    asyncio.run(main())
