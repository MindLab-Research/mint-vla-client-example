#!/usr/bin/env python3
"""Compare LoRA weights: Megatron internal vs exported to vLLM.

Hypothesis: Export process corrupts or misaligns LoRA weights.

Test:
1. Train a few steps
2. Get Megatron's internal LoRA state dict
3. Export to vLLM
4. Load vLLM adapter and compare weights
"""

import asyncio
import os

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker
import torch
import requests


async def do_training_step(training_client, tokenizer, prompt: str, response: str):
    """Do a single training step."""
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

    fwd_bwd_future = await training_client.forward_backward_async([datum], loss_fn="importance_sampling")
    await fwd_bwd_future.result_async()

    optim_future = await training_client.optim_step_async(
        adam_params=tinker.AdamParams(learning_rate=1e-4)
    )
    await optim_future.result_async()


async def main():
    print("=" * 70)
    print("TEST: Compare Megatron internal LoRA vs exported weights")
    print("=" * 70)

    model_name = "moonshotai/Moonlight-16B-A3B-Instruct"
    base_url = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Create training session
    print("\n[1] Creating training session...")
    service_client = tinker.ServiceClient(base_url=base_url)
    training_client = await service_client.create_lora_training_client_async(model_name, rank=16)

    # Train 3 steps
    print("\n[2] Training 3 steps...")
    training_examples = [
        ("<|im_start|>user\nWhat is 1+1?<|im_end|>\n<|im_start|>assistant\n", "2<|im_end|>"),
        ("<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n", "4<|im_end|>"),
        ("<|im_start|>user\nWhat is 3+3?<|im_end|>\n<|im_start|>assistant\n", "6<|im_end|>"),
    ]

    for i, (prompt, response) in enumerate(training_examples):
        await do_training_step(training_client, tokenizer, prompt, response)
        print(f"    Step {i+1}/3 completed")

    # Export to vLLM and get adapter path
    print("\n[3] Exporting to vLLM...")
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()
    print(f"    Export completed")

    # Get the adapter path from vLLM session
    # The sampling_client should have info about the adapter
    print(f"\n[4] Sampling client info:")
    print(f"    Session ID: {sampling_client._session_id if hasattr(sampling_client, '_session_id') else 'N/A'}")

    # Try to get internal state through HTTP API
    # Check if there's a way to inspect the exported adapter

    # Get Megatron's internal LoRA weights via training client
    print(f"\n[5] Getting Megatron internal LoRA weights...")

    # Use the session info endpoint if available
    try:
        session_info = await training_client.get_info_async()
        print(f"    Training session info: {session_info}")
    except Exception as e:
        print(f"    Could not get session info: {e}")

    # For now, let's check if there's a difference in logprobs
    # by comparing vLLM compute vs Megatron forward on the same sequence

    test_tokens = tokenizer.encode("<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n4", add_special_tokens=False)

    # Get vLLM logprobs
    print(f"\n[6] Getting vLLM logprobs...")
    vllm_logprobs = await sampling_client.compute_logprobs_async(
        tinker.ModelInput.from_ints(test_tokens)
    )
    print(f"    Got {len(vllm_logprobs)} logprobs")
    print(f"    First 10: {vllm_logprobs[:10]}")

    # Get Megatron logprobs
    print(f"\n[7] Getting Megatron logprobs...")
    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(test_tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.ones(len(test_tokens), dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(test_tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.zeros(len(test_tokens), dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.zeros(len(test_tokens), dtype=torch.float32)),
        }
    )
    fwd_future = await training_client.forward_async([datum], loss_fn="importance_sampling")
    fwd_result = await fwd_future.result_async()
    megatron_logprobs = fwd_result.loss_fn_outputs[0]["logprobs"].to_torch().tolist()
    print(f"    Got {len(megatron_logprobs)} logprobs")
    print(f"    First 10: {megatron_logprobs[:10]}")

    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON: vLLM vs Megatron logprobs (same sequence, trained LoRA)")
    print("=" * 70)

    print(f"\n{'pos':>4} | {'token':>8} | {'text':>12} | {'vLLM':>10} | {'Megatron':>10} | {'diff':>10}")
    print("-" * 70)

    for i in range(min(25, len(test_tokens))):
        tok = test_tokens[i]
        text = tokenizer.decode([tok])
        v = vllm_logprobs[i] if i < len(vllm_logprobs) else float('nan')
        m = megatron_logprobs[i] if i < len(megatron_logprobs) else float('nan')
        d = v - m if not (isinstance(v, float) and v != v) else float('nan')
        marker = " *** " if abs(d) > 1 else ""
        print(f"{i:>4} | {tok:>8} | {repr(text):>12} | {v:>10.4f} | {m:>10.4f} | {d:>+10.4f}{marker}")

    # Summary
    diffs = [vllm_logprobs[i] - megatron_logprobs[i] for i in range(min(len(vllm_logprobs), len(megatron_logprobs)))]
    mean_diff = sum(abs(d) for d in diffs) / len(diffs)
    max_diff = max(abs(d) for d in diffs)
    above_1 = sum(1 for d in diffs if abs(d) > 1)

    print("\n" + "=" * 70)
    print(f"Mean |diff|: {mean_diff:.4f}")
    print(f"Max |diff|: {max_diff:.4f}")
    print(f"Tokens with |diff| > 1: {above_1}/{len(diffs)}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
