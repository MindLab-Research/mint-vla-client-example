#!/usr/bin/env python3
"""Debug script to inspect LoRA export state dict.

Run this LOCALLY after a training session to see what keys are exported.
"""

import asyncio
import os
import sys

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker
import torch


async def main():
    print("=" * 70)
    print("DEBUG: Inspecting LoRA Export State Dict")
    print("=" * 70)

    model_name = "moonshotai/Moonlight-16B-A3B-Instruct"
    base_url = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")

    # Create client
    print("\n[1] Creating training session with fresh LoRA...")
    service_client = tinker.ServiceClient(base_url=base_url)
    training_client = await service_client.create_lora_training_client_async(model_name, rank=16)

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Do a quick training step to get non-zero weights
    print("\n[2] Doing a training step to get non-zero LoRA weights...")
    prompt = "<|im_start|>user\nWhat is 1+1?<|im_end|>\n<|im_start|>assistant\n"
    response = "2<|im_end|>"
    full_text = prompt + response
    tokens = tokenizer.encode(full_text, add_special_tokens=False)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_len = len(prompt_tokens)
    mask = [0.0] * prompt_len + [1.0] * (len(tokens) - prompt_len)

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.ones(len(tokens), dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.zeros(len(tokens), dtype=torch.float32)),
        }
    )

    fwd_bwd = await training_client.forward_backward_async([datum], loss_fn="importance_sampling")
    await fwd_bwd.result_async()
    optim = await training_client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=5e-5))
    await optim.result_async()
    print("   Training step completed")

    # Get sampling client (this triggers export)
    print("\n[3] Getting sampling client (exports LoRA to vLLM)...")
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()
    print("   Export completed")

    # Now inspect what was saved
    # The checkpoint is at a temp path, but we can also check the saved files
    print("\n[4] Inspecting exported checkpoint...")

    # The checkpoint path is returned by save_checkpoint
    # Let's call the server API to get info about the exported state

    # For debugging, let's sample with vLLM and check logprobs
    test_prompt = "<|im_start|>user\nWhat is 2+3?<|im_end|>\n<|im_start|>assistant\n"
    prompt_tokens_test = tokenizer.encode(test_prompt, add_special_tokens=False)
    model_input = tinker.ModelInput.from_ints(prompt_tokens_test)

    resp = await sampling_client.sample_async(
        prompt=model_input,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=20,
            temperature=0.0,  # Greedy
            seed=42,
        ),
    )

    seq = resp.sequences[0]
    print(f"\n[5] vLLM generated: {tokenizer.decode(seq.tokens)!r}")
    print(f"    vLLM logprobs (first 5): {seq.logprobs[:5] if seq.logprobs else 'None'}")

    # Check the debug log for exported keys
    print("\n[6] Check server log for export details:")
    print("    ssh volcano 'tail -200 /vePFS-Mindverse/share/code/tinker-server/debug_lora_export.log'")


if __name__ == "__main__":
    asyncio.run(main())
