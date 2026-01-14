#!/usr/bin/env python3
"""Compare LoRA weights in Megatron memory vs exported to vLLM."""

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
    print("=" * 70)
    print("COMPARING LORA WEIGHTS: MEGATRON MEMORY VS VLLM EXPORT")
    print("=" * 70)

    # Set up tinker client
    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])
    client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)

    # Get tokens for one sample
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    TEST_TEXT = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello"
    tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.ones(len(input_tokens), dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.ones(len(input_tokens), dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.zeros(len(input_tokens), dtype=torch.float32)),
        },
    )

    # Train 3 steps to get non-zero LoRA weights
    print("\n1. Training 3 steps to get non-zero LoRA weights...")
    for i in range(3):
        fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
        await fwd_bwd.result_async()
        await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()
    print("   Done")

    # Save checkpoint (this is what gets loaded back into Megatron)
    print("\n2. Saving Megatron checkpoint...")
    save_result = await (await client.save_state_async(name="weight_compare")).result_async()
    print(f"   Saved to: {save_result.path}")

    # Export for vLLM (this is what vLLM uses)
    print("\n3. Exporting for vLLM...")
    sampling = await client.save_weights_and_get_sampling_client_async()
    print("   Exported")

    # The question is: are the weights the same?
    # Let's check by comparing forward pass results

    # Create fresh client and load the checkpoint
    print("\n4. Creating fresh client and loading checkpoint...")
    client2 = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)

    # Fresh forward
    fwd_fresh = await client2.forward_async([datum], loss_fn="importance_sampling")
    res_fresh = await fwd_fresh.result_async()
    lp_fresh = np.array(res_fresh.loss_fn_outputs[0]["logprobs"].to_numpy())
    print(f"   Fresh logprobs: {lp_fresh[:5]}")

    # Load checkpoint
    await client2.load_state_async(save_result.path)

    # Forward after loading
    fwd_loaded = await client2.forward_async([datum], loss_fn="importance_sampling")
    res_loaded = await fwd_loaded.result_async()
    lp_loaded = np.array(res_loaded.loss_fn_outputs[0]["logprobs"].to_numpy())
    print(f"   Loaded logprobs: {lp_loaded[:5]}")

    # vLLM forward
    await asyncio.sleep(2)
    vllm_res = await sampling.sample_async(
        prompt=tinker.ModelInput.from_ints(input_tokens),
        num_samples=1,
        sampling_params=tinker.SamplingParams(max_tokens=1, temperature=0.0),
        include_prompt_logprobs=True,
    )
    vllm_lp = np.array([x if x is not None else 0.0 for x in vllm_res.prompt_logprobs])
    print(f"   vLLM logprobs: {vllm_lp[1:6]}")  # Offset by 1 for alignment

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"Fresh:  {lp_fresh[:5]}")
    print(f"Loaded: {lp_loaded[:5]}")
    print(f"vLLM:   {vllm_lp[1:6]}")

    # Key comparison: fresh vs vLLM should be similar, loaded should match training client
    diff_fresh_vllm = np.abs(lp_fresh - vllm_lp[1:len(lp_fresh)+1]).mean()
    diff_loaded_vllm = np.abs(lp_loaded - vllm_lp[1:len(lp_loaded)+1]).mean()

    print(f"\nMean |fresh - vLLM|: {diff_fresh_vllm:.4f}")
    print(f"Mean |loaded - vLLM|: {diff_loaded_vllm:.4f}")

    if diff_loaded_vllm > 2:
        print(f"\n>>> BUG: Loaded checkpoint produces different results than vLLM!")
        print(f">>> Same weights, different outputs.")


if __name__ == "__main__":
    asyncio.run(main())
