#!/usr/bin/env python3
"""Debug: trace MoE routing and expert assignment."""

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

    print("=" * 70)
    print("HYPOTHESIS: MoE routing differs between train_mode and eval_mode")
    print("=" * 70)
    print()
    print("Position 7: input=\\n, target=Count (token 3922)")
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

    print("Test 1: Fresh model forward (eval_mode)")
    fwd0 = await client.forward_async([datum], loss_fn="importance_sampling")
    res0 = await fwd0.result_async()
    lp0 = np.array(res0.loss_fn_outputs[0]["logprobs"].to_numpy())
    print(f"  pos 7 logprob: {lp0[7]:.4f}")

    print("\nTest 2: Train ONE step")
    fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
    await fwd_bwd.result_async()
    await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()
    print("  Done")

    print("\nTest 3: After training, forward (eval_mode)")
    fwd1 = await client.forward_async([datum], loss_fn="importance_sampling")
    res1 = await fwd1.result_async()
    lp1 = np.array(res1.loss_fn_outputs[0]["logprobs"].to_numpy())
    print(f"  pos 7 logprob: {lp1[7]:.4f} (delta from fresh: {lp1[7]-lp0[7]:+.4f})")

    print("\nTest 4: After training, forward TWICE more (check stability)")
    for i in range(2):
        fwd = await client.forward_async([datum], loss_fn="importance_sampling")
        res = await fwd.result_async()
        lp = np.array(res.loss_fn_outputs[0]["logprobs"].to_numpy())
        print(f"  Forward {i+2}: pos 7 logprob = {lp[7]:.4f}")

    print("\nTest 5: Create a NEW client, load the same checkpoint, check forward")
    # Save first
    save_result = await (await client.save_state_async(name="moe_debug")).result_async()
    print(f"  Saved to: {save_result.path}")

    # Create new client
    client2 = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)

    # Fresh forward
    fwd2_fresh = await client2.forward_async([datum], loss_fn="importance_sampling")
    res2_fresh = await fwd2_fresh.result_async()
    lp2_fresh = np.array(res2_fresh.loss_fn_outputs[0]["logprobs"].to_numpy())
    print(f"  Client 2 fresh: pos 7 logprob = {lp2_fresh[7]:.4f}")

    # Load checkpoint
    await client2.load_state_async(save_result.path)

    # Forward after load
    fwd2_loaded = await client2.forward_async([datum], loss_fn="importance_sampling")
    res2_loaded = await fwd2_loaded.result_async()
    lp2_loaded = np.array(res2_loaded.loss_fn_outputs[0]["logprobs"].to_numpy())
    print(f"  Client 2 loaded: pos 7 logprob = {lp2_loaded[7]:.4f}")

    print("\nTest 6: vLLM with same checkpoint")
    sampling = await client.save_weights_and_get_sampling_client_async()
    await asyncio.sleep(2)
    vllm_res = await sampling.sample_async(
        prompt=tinker.ModelInput.from_ints(input_tokens),
        num_samples=1,
        sampling_params=tinker.SamplingParams(max_tokens=1, temperature=0.0),
        include_prompt_logprobs=True,
    )
    vllm_lp = vllm_res.prompt_logprobs
    print(f"  vLLM pos 8 (aligned): {vllm_lp[8]:.4f}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Fresh:                      {lp0[7]:.4f}")
    print(f"After training (same):      {lp1[7]:.4f}")
    print(f"New client loaded:          {lp2_loaded[7]:.4f}")
    print(f"vLLM:                       {vllm_lp[8]:.4f}")
    print()
    if abs(lp1[7] - lp2_loaded[7]) < 0.1:
        print("Training client and loaded client match → checkpoint save/load is correct")
    if abs(lp1[7] - vllm_lp[8]) > 5:
        print(f"Megatron-vLLM diff = {lp1[7] - vllm_lp[8]:.1f} nats → BUG in Megatron forward")


if __name__ == "__main__":
    asyncio.run(main())
