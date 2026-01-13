#!/usr/bin/env python3
"""Test if topk_indices/topk_logits propagate from Megatron to API output."""

import asyncio
import os
os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker
import torch

MODEL_NAME = "moonshotai/Moonlight-16B-A3B-Instruct"
TEST_TOKENS = [100, 200, 300, 400, 500]  # Simple test tokens

async def main():
    print("Creating Megatron training client...")
    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])
    client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)
    print("Client created")

    # Create a simple datum
    input_tokens = TEST_TOKENS[:-1]
    target_tokens = TEST_TOKENS[1:]
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

    print("\nRunning forward pass...")
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()

    print(f"\nResult type: {type(result)}")
    print(f"Result keys: {dir(result)}")

    output = result.loss_fn_outputs[0]
    print(f"\nloss_fn_outputs[0] type: {type(output)}")
    print(f"loss_fn_outputs[0] keys: {list(output.keys())}")

    # Check for topk
    if "topk_indices" in output:
        topk_indices = output["topk_indices"]
        print(f"\ntopk_indices found!")
        print(f"  Type: {type(topk_indices)}")
        if hasattr(topk_indices, "data"):
            print(f"  Data type: {type(topk_indices.data)}")
            print(f"  Data sample: {topk_indices.data[:2] if topk_indices.data else 'empty'}")
    else:
        print("\ntopk_indices NOT in output")

    if "topk_logits" in output:
        topk_logits = output["topk_logits"]
        print(f"\ntopk_logits found!")
        print(f"  Type: {type(topk_logits)}")
        if hasattr(topk_logits, "data"):
            print(f"  Data type: {type(topk_logits.data)}")
            print(f"  Data sample: {topk_logits.data[:2] if topk_logits.data else 'empty'}")
    else:
        print("\ntopk_logits NOT in output")

    print("\n--- Done ---")

if __name__ == "__main__":
    asyncio.run(main())
