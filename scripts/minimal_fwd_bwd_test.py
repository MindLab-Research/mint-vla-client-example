#!/usr/bin/env python3
"""Minimal test to reproduce forward_backward failure."""

import os
import sys

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import asyncio
import tinker
import torch
import numpy as np

MODEL_NAME = "moonshotai/Moonlight-16B-A3B-Instruct"

TEST_TEXT = "<|im_start|>user\nCount down from 10 to 1, one number per line.<|im_end|>\n<|im_start|>assistant\n10\n9\n8\n7\n6\n5\n4\n3\n2\n1<|im_end|>"


async def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    mask = [1.0] * len(input_tokens)

    print(f"Token count: {len(tokens)}")
    print(f"input_tokens: {len(input_tokens)}")
    print(f"target_tokens: {len(target_tokens)}")
    print(f"mask: {len(mask)}")

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])
    client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.ones(len(input_tokens), dtype=torch.float32)),
        },
    )

    # Exactly like test script: forward first
    print("Running forward_async...")
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    print(f'Forward done, logprobs len: {len(result.loss_fn_outputs[0]["logprobs"].to_numpy())}')

    # Then export to vLLM (exactly like test script)
    print("Running save_weights_and_get_sampling_client_async...")
    sampling_client = await client.save_weights_and_get_sampling_client_async()
    print("Export done")
    await asyncio.sleep(2)

    # Then forward_backward (exactly like test script Phase 3)
    print("Running forward_backward_async...")
    fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
    await fwd_bwd.result_async()
    print("SUCCESS: forward_backward_async completed!")


if __name__ == "__main__":
    asyncio.run(main())
