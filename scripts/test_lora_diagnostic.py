#!/usr/bin/env python3
"""Minimal test to trigger LoRA diagnostic output."""

import asyncio
import os

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker
import torch
import numpy as np

MODEL_NAME = "moonshotai/Moonlight-16B-A3B-Instruct"


async def main():
    print("=" * 70)
    print("TESTING LORA DIAGNOSTIC OUTPUT")
    print("=" * 70)

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    # Minimal test sequence
    tokens = [27, 91, 348, 9485, 91, 29, 2482, 198, 19180]  # 9 tokens
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

    print("\n1. Create fresh LoRA...")
    client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)

    print("\n2. Forward pass with FRESH LoRA (expect lora contribution = 0)...")
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    fresh_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())
    print(f"   Fresh logprobs: {fresh_lp}")

    print("\n3. Train 1 step...")
    fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
    await fwd_bwd.result_async()
    await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()

    print("\n4. Forward pass with TRAINED LoRA (this is where the bug manifests)...")
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    trained_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())
    print(f"   Trained logprobs: {trained_lp}")

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    for i, (f, t) in enumerate(zip(fresh_lp, trained_lp)):
        delta = t - f
        flag = " ***" if delta < -5 else ""
        print(f"Position {i}: fresh={f:.4f}, trained={t:.4f}, delta={delta:+.4f}{flag}")

    print("\nDONE - check worker logs for [LORA-DIAG] output")


if __name__ == "__main__":
    asyncio.run(main())
