#!/usr/bin/env python3
"""Test LoRA tensor values before and after training.

Purpose: Verify that the exported LoRA weights reflect training changes.
"""

import asyncio
import os
import sys
import time
import tempfile

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker
import torch


async def do_training_step(training_client: tinker.TrainingClient, tokenizer, prompt: str, response: str):
    """Do a single training step using SFT format."""
    full_text = prompt + response
    tokens = tokenizer.encode(full_text, add_special_tokens=False)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_len = len(prompt_tokens)

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    mask = [0.0] * (prompt_len - 1) + [1.0] * (len(input_tokens) - prompt_len + 1)
    advantages = [0.0] * (prompt_len - 1) + [1.0] * (len(input_tokens) - prompt_len + 1)
    logprobs = [0.0] * len(input_tokens)

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.tensor(advantages, dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.tensor(logprobs, dtype=torch.float32)),
        }
    )

    fwd_bwd_future = await training_client.forward_backward_async([datum], loss_fn="importance_sampling")
    await fwd_bwd_future.result_async()

    optim_future = await training_client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=5e-5))
    await optim_future.result_async()


async def main():
    print("=" * 70)
    print("LORA TENSOR VALUE VERIFICATION")
    print("=" * 70)

    model_name = "moonshotai/Moonlight-16B-A3B-Instruct"
    base_url = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")

    # Load tokenizer
    print("\n[1] Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Create clients
    print("\n[2] Creating training session with fresh LoRA...")
    service_client = tinker.ServiceClient(base_url=base_url)
    training_client = await service_client.create_lora_training_client_async(model_name, rank=16)

    # Get initial weights via save_weights
    print("\n[3] Getting initial LoRA weights (should be zeros)...")
    t0 = time.time()
    sampling_client_before = await training_client.save_weights_and_get_sampling_client_async()
    print(f"    Export took {time.time()-t0:.1f}s")

    # Check saved adapter file
    print("\n[4] Checking saved adapter file (fresh LoRA)...")
    from safetensors import safe_open

    # Find the most recent adapter file
    temp_dirs = [d for d in os.listdir("/tmp") if d.startswith("tinker_lora_")]
    if temp_dirs:
        temp_dirs.sort(key=lambda x: os.path.getmtime(f"/tmp/{x}"), reverse=True)
        latest_adapter = f"/tmp/{temp_dirs[0]}/adapter_model.safetensors"
        print(f"    Found: {latest_adapter}")

        if os.path.exists(latest_adapter):
            with safe_open(latest_adapter, framework="pt") as f:
                keys = list(f.keys())
                print(f"    Keys: {len(keys)}")
                for k in keys[:5]:
                    tensor = f.get_tensor(k)
                    norm = tensor.norm().item()
                    max_val = tensor.abs().max().item()
                    print(f"      {k[:60]}: norm={norm:.6f}, max={max_val:.6f}")

    # Do training steps
    print("\n[5] Doing 5 training steps...")
    training_examples = [
        ("<|im_start|>user\nWhat is 1+1?<|im_end|>\n<|im_start|>assistant\n", "2<|im_end|>"),
        ("<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n", "4<|im_end|>"),
        ("<|im_start|>user\nWhat is 3+3?<|im_end|>\n<|im_start|>assistant\n", "6<|im_end|>"),
        ("<|im_start|>user\nWhat is 4+4?<|im_end|>\n<|im_start|>assistant\n", "8<|im_end|>"),
        ("<|im_start|>user\nWhat is 5+5?<|im_end|>\n<|im_start|>assistant\n", "10<|im_end|>"),
    ]

    for i, (prompt, response) in enumerate(training_examples):
        t0 = time.time()
        await do_training_step(training_client, tokenizer, prompt, response)
        print(f"    Step {i+1}/5 completed in {time.time()-t0:.1f}s")

    # Get trained weights
    print("\n[6] Getting trained LoRA weights...")
    t0 = time.time()
    sampling_client_after = await training_client.save_weights_and_get_sampling_client_async()
    print(f"    Export took {time.time()-t0:.1f}s")

    # Check saved adapter file after training
    print("\n[7] Checking saved adapter file (trained LoRA)...")
    temp_dirs = [d for d in os.listdir("/tmp") if d.startswith("tinker_lora_")]
    if temp_dirs:
        temp_dirs.sort(key=lambda x: os.path.getmtime(f"/tmp/{x}"), reverse=True)
        latest_adapter = f"/tmp/{temp_dirs[0]}/adapter_model.safetensors"
        print(f"    Found: {latest_adapter}")

        if os.path.exists(latest_adapter):
            with safe_open(latest_adapter, framework="pt") as f:
                keys = list(f.keys())
                print(f"    Keys: {len(keys)}")

                nonzero_count = 0
                total_norm = 0.0
                for k in keys:
                    tensor = f.get_tensor(k)
                    norm = tensor.norm().item()
                    total_norm += norm
                    if norm > 1e-8:
                        nonzero_count += 1

                print(f"    Total keys: {len(keys)}")
                print(f"    Non-zero keys: {nonzero_count}")
                print(f"    Total norm: {total_norm:.6f}")

                print(f"\n    Sample weights (first 5):")
                for k in keys[:5]:
                    tensor = f.get_tensor(k)
                    norm = tensor.norm().item()
                    max_val = tensor.abs().max().item()
                    print(f"      {k[:60]}: norm={norm:.6f}, max={max_val:.6f}")

                if nonzero_count == 0:
                    print("\n    >>> PROBLEM: All weights are still zero after training!")
                    print("        This means training didn't update the LoRA weights, or export didn't capture them.")
                else:
                    print(f"\n    >>> OK: {nonzero_count}/{len(keys)} weights have non-zero values.")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
