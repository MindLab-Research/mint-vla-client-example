#!/usr/bin/env python3
"""Verify LoRA tensor values at each stage of export.

Purpose: Diagnose why trained LoRA weights don't transfer from Megatron to vLLM.
Tests the hypothesis that the state_dict contains zero values when passed to vLLM.
"""

import asyncio
import os
import sys
import time

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

    # Get initial weights DIRECTLY via API (not via sampling client)
    print("\n[3] Getting initial LoRA weights (should be zeros)...")
    print("    Calling get_lora_state_dict directly...")

    # Access the underlying HTTP client to call raw API
    import httpx
    async with httpx.AsyncClient(timeout=300.0) as client:
        # Get lora state dict directly
        response = await client.post(
            f"{base_url}/api/v1/get_lora_state_dict",
            json={"session_id": training_client._session_id},
        )
        if response.status_code != 200:
            print(f"    ERROR: {response.status_code} - {response.text}")
        else:
            # The tensors would be serialized - let's use the save_weights flow instead
            print(f"    Response: {response.status_code}")

    # Use save_weights to get the actual weights
    print("\n[4] Exporting fresh LoRA to vLLM...")
    t0 = time.time()
    sampling_client_before = await training_client.save_weights_and_get_sampling_client_async()
    print(f"    Export took {time.time()-t0:.1f}s")

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
    print("\n[6] Exporting trained LoRA to vLLM...")
    t0 = time.time()
    sampling_client_after = await training_client.save_weights_and_get_sampling_client_async()
    print(f"    Export took {time.time()-t0:.1f}s")

    # Compare logprobs to verify training effect
    print("\n[7] Comparing logprobs before/after training...")
    test_prompt = "<|im_start|>user\nWhat is 2+3?<|im_end|>\n<|im_start|>assistant\n"
    test_response = "5<|im_end|>"

    prompt_tokens = tokenizer.encode(test_prompt, add_special_tokens=False)
    full_tokens = tokenizer.encode(test_prompt + test_response, add_special_tokens=False)
    model_input = tinker.ModelInput.from_ints(full_tokens)

    lp_before = await sampling_client_before.compute_logprobs_async(model_input)
    lp_after = await sampling_client_after.compute_logprobs_async(model_input)

    prompt_len = len(prompt_tokens)
    response_tokens = full_tokens[prompt_len:]

    print(f"\n  Logprob comparison for response tokens:")
    print(f"  {'pos':>4} | {'tok':>8} | {'text':>15} | {'before':>12} | {'after':>12} | {'change':>12}")
    print(f"  {'-' * 75}")

    total_change = 0.0
    for i, tok in enumerate(response_tokens):
        text = tokenizer.decode([tok])
        pos = prompt_len + i
        lp_b = lp_before[pos] if pos < len(lp_before) else float('nan')
        lp_a = lp_after[pos] if pos < len(lp_after) else float('nan')
        change = lp_a - lp_b
        total_change += abs(change)
        print(f"  {pos:>4} | {tok:>8} | {repr(text):>15} | {lp_b:>12.4f} | {lp_a:>12.4f} | {change:>+12.4f}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Total absolute change: {total_change:.4f}")

    if total_change < 0.1:
        print("\n  >>> PROBLEM: vLLM logprobs barely changed after training!")
        print("      This indicates LoRA export is NOT correctly transferring trained weights.")
        print("      Root cause: state_dict passed to vLLM may contain stale/zero values.")
    else:
        print("\n  >>> OK: vLLM logprobs changed after training.")
        print("      LoRA export appears to be working.")


if __name__ == "__main__":
    asyncio.run(main())
