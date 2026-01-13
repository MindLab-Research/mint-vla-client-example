#!/usr/bin/env python3
"""Debug single training step: trace weights and logprobs."""

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
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    TEST_TEXT = "<|im_start|>user\nCount down from 10 to 1, one number per line.<|im_end|>\n<|im_start|>assistant\n10\n9\n8\n7\n6\n5\n4\n3\n2\n1<|im_end|>"
    tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    mask = [1.0] * len(input_tokens)

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    print("=" * 70)
    print("Creating fresh Megatron LoRA client...")
    print("=" * 70)
    client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)

    # Build datum EXACTLY like test_megatron_vllm_logprob_mismatch.py
    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.ones(len(input_tokens), dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.zeros(len(input_tokens), dtype=torch.float32)),
        },
    )

    print("\n" + "=" * 70)
    print("STEP 0: Fresh model (before any training)")
    print("=" * 70)

    fwd0 = await client.forward_async([datum], loss_fn="importance_sampling")
    res0 = await fwd0.result_async()
    lp0 = np.array(res0.loss_fn_outputs[0]["logprobs"].to_numpy())
    print(f"Megatron pos 7 logprob: {lp0[7]:.4f}")

    # Export fresh and check vLLM (same order as test script)
    print("\nExporting fresh weights to vLLM...")
    sampling0 = await client.save_weights_and_get_sampling_client_async()
    await asyncio.sleep(1)

    vllm_res0 = await sampling0.sample_async(
        prompt=tinker.ModelInput.from_ints(input_tokens),
        num_samples=1,
        sampling_params=tinker.SamplingParams(max_tokens=1, temperature=0.0),
        include_prompt_logprobs=True,
    )
    vllm_lp0 = vllm_res0.prompt_logprobs
    print(f"vLLM pos 8 logprob (aligned): {vllm_lp0[8]:.4f}")
    print(f"Fresh model diff: {abs(lp0[7] - vllm_lp0[8]):.4f}")

    print("\n" + "=" * 70)
    print("STEP 1: Do ONE training step (forward_backward + optim_step)")
    print("=" * 70)

    # Do one forward-backward pass (EXACTLY like test script)
    fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
    await fwd_bwd.result_async()
    print(f"Forward-backward done")

    # Do one optimizer step
    await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()
    print(f"Optimizer step done")

    print("\n" + "=" * 70)
    print("STEP 2: Get logprobs IMMEDIATELY after training (same client)")
    print("=" * 70)

    fwd1 = await client.forward_async([datum], loss_fn="importance_sampling")
    res1 = await fwd1.result_async()
    lp1 = np.array(res1.loss_fn_outputs[0]["logprobs"].to_numpy())
    print(f"Megatron pos 7 logprob: {lp1[7]:.4f} (was {lp0[7]:.4f}, delta={lp1[7]-lp0[7]:+.4f})")

    # Export and check vLLM
    sampling1 = await client.save_weights_and_get_sampling_client_async()
    await asyncio.sleep(1)

    vllm_res1 = await sampling1.sample_async(
        prompt=tinker.ModelInput.from_ints(input_tokens),
        num_samples=1,
        sampling_params=tinker.SamplingParams(max_tokens=1, temperature=0.0),
        include_prompt_logprobs=True,
    )
    vllm_lp1 = vllm_res1.prompt_logprobs
    print(f"vLLM pos 8 logprob (aligned): {vllm_lp1[8]:.4f} (was {vllm_lp0[8]:.4f}, delta={vllm_lp1[8]-vllm_lp0[8]:+.4f})")
    print(f"Trained model diff: {abs(lp1[7] - vllm_lp1[8]):.4f}")

    # Save checkpoint for later analysis
    save_result = await (await client.save_state_async(name="step1_debug")).result_async()
    print(f"\nCheckpoint: {save_result.path}")

    print("\n" + "=" * 70)
    print("STEP 3: Load checkpoint into FRESH Megatron client and compare")
    print("=" * 70)

    # Create a NEW client
    client2 = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)

    # Get logprobs from fresh client (should match step 0)
    fwd2_fresh = await client2.forward_async([datum], loss_fn="importance_sampling")
    res2_fresh = await fwd2_fresh.result_async()
    lp2_fresh = np.array(res2_fresh.loss_fn_outputs[0]["logprobs"].to_numpy())
    print(f"New client (fresh) pos 7: {lp2_fresh[7]:.4f}")

    # Load the step1 checkpoint
    print(f"\nLoading checkpoint: {save_result.path}")
    await client2.load_state_async(save_result.path)

    # Get logprobs after loading
    fwd2_loaded = await client2.forward_async([datum], loss_fn="importance_sampling")
    res2_loaded = await fwd2_loaded.result_async()
    lp2_loaded = np.array(res2_loaded.loss_fn_outputs[0]["logprobs"].to_numpy())
    print(f"New client (loaded) pos 7: {lp2_loaded[7]:.4f}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Fresh Megatron:        {lp0[7]:.4f}")
    print(f"After train (same):    {lp1[7]:.4f} (delta={lp1[7]-lp0[7]:+.4f})")
    print(f"After load (new):      {lp2_loaded[7]:.4f} (delta from fresh={lp2_loaded[7]-lp2_fresh[7]:+.4f})")
    print(f"vLLM fresh:            {vllm_lp0[8]:.4f}")
    print(f"vLLM after train:      {vllm_lp1[8]:.4f} (delta={vllm_lp1[8]-vllm_lp0[8]:+.4f})")

    print(f"\n>>> Key question: Does 'After train (same)' match 'After load (new)'?")
    if abs(lp1[7] - lp2_loaded[7]) < 0.5:
        print(f">>> YES: Save/load preserves weights. Bug is in Megatron forward pass.")
    else:
        print(f">>> NO: Difference {abs(lp1[7] - lp2_loaded[7]):.4f}. Bug may be in save/load.")


if __name__ == "__main__":
    asyncio.run(main())
