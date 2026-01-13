#!/usr/bin/env python3
"""Verify fresh LoRA weights are zero."""

import os
import sys

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import asyncio
import tinker
import torch
import numpy as np

MODEL_NAME = "moonshotai/Moonlight-16B-A3B-Instruct"


async def main():
    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    print("Creating fresh training client...")
    client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)

    print("Exporting fresh weights to vLLM...")
    sampling_client = await client.save_weights_and_get_sampling_client_async()

    # The export happens during get_sampling_client_async
    # Let's check what the exported weights look like

    print("\nExported. Now check the state_dict...")

    # Actually, let me just test if the base model (without LoRA) gives same results
    # by using the Megatron's raw forward vs vLLM

    TEST_TEXT = "<|im_start|>user\nCount down from 10 to 1.<|im_end|>"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)
    input_tokens = tokens[:-1]

    print(f"\nInput: {len(input_tokens)} tokens")

    # Get Megatron logprobs
    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.tensor([1.0] * len(input_tokens), dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(tokens[1:], dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.ones(len(input_tokens), dtype=torch.float32)),
        },
    )
    lp_result = await client.get_log_probs_async([datum])
    meg_lp = lp_result[0].log_probs

    print(f"Megatron fresh logprobs: {len(meg_lp)} positions")
    for i in range(min(10, len(meg_lp))):
        print(f"  pos={i}: {meg_lp[i]:.4f}")

    # Get vLLM logprobs
    sample_result = await sampling_client.sample_async(
        prompt=tinker.ModelInput.from_ints(input_tokens),
        num_samples=1,
        sampling_params=tinker.SamplingParams(max_tokens=1, temperature=0.0),
        include_prompt_logprobs=True,
    )

    if sample_result.prompt_logprobs:
        vllm_lp = sample_result.prompt_logprobs
        print(f"\nvLLM fresh logprobs: {len(vllm_lp)} positions")
        for i in range(min(10, len(vllm_lp))):
            print(f"  pos={i}: {vllm_lp[i]:.4f if vllm_lp[i] is not None else 'None'}")
    else:
        print("\nvLLM didn't return prompt_logprobs")

    print("\n" + "="*70)
    print("COMPARISON (should be close if fresh LoRA has no effect):")
    print("="*70)

    for i in range(min(10, len(meg_lp))):
        meg = meg_lp[i]
        vllm = vllm_lp[i] if vllm_lp and i < len(vllm_lp) and vllm_lp[i] is not None else float('nan')
        diff = meg - vllm if not np.isnan(vllm) else float('nan')
        print(f"  pos={i}: Meg={meg:8.4f}, vLLM={vllm:8.4f}, diff={diff:+8.4f}")


if __name__ == "__main__":
    asyncio.run(main())
