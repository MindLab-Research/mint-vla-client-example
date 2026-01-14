#!/usr/bin/env python3
"""Check what LoRA keys are exported and loaded by vLLM.

This script traces the exact export and load paths to understand
why Megatron and vLLM behave differently after training.
"""

import asyncio
import os

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker
import torch
import numpy as np

MODEL_NAME = "moonshotai/Moonlight-16B-A3B-Instruct"


async def main():
    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    tokens = tokenizer.encode("Hello world!", add_special_tokens=False)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    mask = [1.0] * len(input_tokens)

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    print("\n1. Create fresh LoRA client...")
    client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)

    print("\n2. Train 3 steps to get non-zero LoRA weights...")
    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.ones(len(input_tokens), dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.zeros(len(input_tokens), dtype=torch.float32)),
        }
    )

    for step in range(3):
        fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
        await fwd_bwd.result_async()
        await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()
    print("   Done!")

    print("\n3. Export to vLLM and get sampling client...")
    print("   (Check server logs for export details)")
    sampling_client = await client.save_weights_and_get_sampling_client_async()

    print(f"\n4. Sampling client session ID: {sampling_client.session_id}")

    # Simple test
    prompt = tinker.ModelInput.from_ints(tokens)
    lp = await sampling_client.compute_logprobs_async(prompt)
    print(f"\n5. Logprobs computed: {len([x for x in lp if x is not None])} values")

    print("\n" + "=" * 70)
    print("Check server logs at /tmp/tinker_server.log for:")
    print("  - 'use_per_expert_lora=' to see if per-expert expansion was enabled")
    print("  - 'Expanded' or 'Filtered' for MLP LoRA handling")
    print("  - 'EP-expanded' for expert-parallel expansion")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
