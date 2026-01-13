#!/usr/bin/env python3
"""Investigate why most Megatron positions show -0.0000 after training."""

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

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])
    client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.ones(len(input_tokens), dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.zeros(len(input_tokens), dtype=torch.float32)),
        },
    )

    # Get fresh logprobs
    print("=" * 80)
    print("FRESH MODEL LOGPROBS")
    print("=" * 80)

    fwd0 = await client.forward_async([datum], loss_fn="importance_sampling")
    res0 = await fwd0.result_async()
    lp0 = np.array(res0.loss_fn_outputs[0]["logprobs"].to_numpy())

    print(f"\nFresh Megatron (first 20 positions):")
    for i in range(min(20, len(lp0))):
        target_str = tokenizer.decode([target_tokens[i]])[:8]
        print(f"  pos={i:2d}: {lp0[i]:8.4f} (target={repr(target_str)})")

    # Train 3 steps
    print("\n" + "=" * 80)
    print("TRAINING 3 STEPS")
    print("=" * 80)

    for step in range(3):
        fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
        await fwd_bwd.result_async()
        await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()

    # Get trained logprobs
    fwd1 = await client.forward_async([datum], loss_fn="importance_sampling")
    res1 = await fwd1.result_async()
    lp1 = np.array(res1.loss_fn_outputs[0]["logprobs"].to_numpy())

    print(f"\nTrained Megatron (first 20 positions):")
    for i in range(min(20, len(lp1))):
        target_str = tokenizer.decode([target_tokens[i]])[:8]
        delta = lp1[i] - lp0[i]
        print(f"  pos={i:2d}: {lp1[i]:8.4f} (delta={delta:+8.4f}, target={repr(target_str)})")

    # Export to vLLM
    sampling = await client.save_weights_and_get_sampling_client_async()
    await asyncio.sleep(2)
    vllm_res = await sampling.sample_async(
        prompt=tinker.ModelInput.from_ints(input_tokens),
        num_samples=1,
        sampling_params=tinker.SamplingParams(max_tokens=1, temperature=0.0),
        include_prompt_logprobs=True,
    )
    vllm_lp = vllm_res.prompt_logprobs

    print(f"\nTrained vLLM (first 20 positions):")
    for i in range(min(20, len(vllm_lp))):
        if vllm_lp[i] is not None:
            target_str = tokenizer.decode([input_tokens[i]])[:8]  # Note: vLLM[i] is for input_tokens[i]
            print(f"  pos={i:2d}: {vllm_lp[i]:8.4f} (token={repr(target_str)})")

    # Compare aligned
    print("\n" + "=" * 80)
    print("ALIGNED COMPARISON: Megatron[i] vs vLLM[i+1]")
    print("=" * 80)
    print(f"\n{'Pos':<4} {'Target':<10} {'Meg Fresh':<10} {'Meg Train':<10} {'vLLM':<10} {'Meg-vLLM':<10}")
    print("-" * 60)

    # Count positions by behavior
    near_zero_count = 0
    large_negative_count = 0
    for i in range(min(20, len(lp1))):
        target_str = tokenizer.decode([target_tokens[i]])[:8]
        meg_fresh = lp0[i]
        meg_train = lp1[i]
        vllm_aligned = vllm_lp[i+1] if i+1 < len(vllm_lp) and vllm_lp[i+1] is not None else float('nan')
        diff = meg_train - vllm_aligned if not np.isnan(vllm_aligned) else float('nan')

        if not np.isnan(meg_train):
            if meg_train > -0.01:  # Near zero
                near_zero_count += 1
            elif meg_train < -10:  # Large negative
                large_negative_count += 1

        print(f"{i:<4} {target_str:<10} {meg_fresh:<10.4f} {meg_train:<10.4f} {vllm_aligned:<10.4f} {diff:+10.4f}")

    print(f"\nPositions with trained logprob > -0.01 (near zero): {near_zero_count}")
    print(f"Positions with trained logprob < -10 (large negative): {large_negative_count}")

    # Check statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Megatron fresh mean: {np.mean(lp0):.4f}")
    print(f"Megatron trained mean: {np.mean(lp1):.4f}")
    print(f"Megatron trained min: {np.min(lp1):.4f}")
    print(f"Megatron trained max: {np.max(lp1):.4f}")
    print(f"Positions with logprob == 0.0: {np.sum(lp1 == 0.0)}")
    print(f"Positions with logprob > -0.001: {np.sum(lp1 > -0.001)}")


if __name__ == "__main__":
    asyncio.run(main())
