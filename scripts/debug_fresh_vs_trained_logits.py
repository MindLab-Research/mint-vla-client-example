#!/usr/bin/env python3
"""Debug: compare logits at position 7 before and after training."""

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

    print(f"Token 3922 = {repr(tokenizer.decode([3922]))}")
    print(f"Token 16 = {repr(tokenizer.decode([16]))}")
    print(f"Token at position 7 (input) = {input_tokens[7]} = {repr(tokenizer.decode([input_tokens[7]]))}")
    print(f"Token at position 7 (target) = {target_tokens[7]} = {repr(tokenizer.decode([target_tokens[7]]))}")
    print()

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

    print("=" * 70)
    print("STEP 1: FRESH MODEL - forward only")
    print("=" * 70)
    print("Server will log: [LOGIT] lines for positions 7, 8, 23")
    print()

    fwd0 = await client.forward_async([datum], loss_fn="importance_sampling")
    res0 = await fwd0.result_async()
    lp0 = np.array(res0.loss_fn_outputs[0]["logprobs"].to_numpy())
    print(f"Megatron fresh pos 7 logprob: {lp0[7]:.4f}")
    print(f"Megatron fresh pos 8 logprob: {lp0[8]:.4f}")

    print()
    print("=" * 70)
    print("STEP 2: ONE TRAINING STEP")
    print("=" * 70)

    fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
    await fwd_bwd.result_async()
    await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()
    print("Training step complete")

    print()
    print("=" * 70)
    print("STEP 3: TRAINED MODEL - forward only")
    print("=" * 70)

    fwd1 = await client.forward_async([datum], loss_fn="importance_sampling")
    res1 = await fwd1.result_async()
    lp1 = np.array(res1.loss_fn_outputs[0]["logprobs"].to_numpy())
    print(f"Megatron trained pos 7 logprob: {lp1[7]:.4f} (delta={lp1[7]-lp0[7]:+.4f})")
    print(f"Megatron trained pos 8 logprob: {lp1[8]:.4f} (delta={lp1[8]-lp0[8]:+.4f})")

    print()
    print("=" * 70)
    print("STEP 4: Export to vLLM and compare")
    print("=" * 70)

    sampling = await client.save_weights_and_get_sampling_client_async()
    await asyncio.sleep(2)
    vllm_res = await sampling.sample_async(
        prompt=tinker.ModelInput.from_ints(input_tokens),
        num_samples=1,
        sampling_params=tinker.SamplingParams(max_tokens=1, temperature=0.0),
        include_prompt_logprobs=True,
    )
    vllm_lp = vllm_res.prompt_logprobs

    print(f"vLLM trained pos 8 logprob (aligned with Meg pos 7): {vllm_lp[8]:.4f}")
    print(f"vLLM trained pos 9 logprob (aligned with Meg pos 8): {vllm_lp[9]:.4f}")
    print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Position 7 (target='Count'):")
    print(f"  Fresh Megatron: {lp0[7]:.4f}")
    print(f"  Trained Megatron: {lp1[7]:.4f}")
    print(f"  Trained vLLM: {vllm_lp[8]:.4f}")
    print(f"  Megatron-vLLM diff: {lp1[7] - vllm_lp[8]:.4f}")
    print()
    print("Check server logs at: /vePFS-Mindverse/share/code/raw_logit_diag.log")
    print("Look for pattern in logits before and after training.")


if __name__ == "__main__":
    asyncio.run(main())
