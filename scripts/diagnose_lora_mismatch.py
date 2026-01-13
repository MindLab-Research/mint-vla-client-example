#!/usr/bin/env python3
"""Diagnose LoRA weight mismatch between Megatron and vLLM.

This script:
1. Creates training session
2. Trains 1 step
3. Gets LoRA weights from Megatron directly
4. Exports to vLLM format
5. Compares the actual tensor values

Key hypothesis: The export is correct but computation differs.
"""

import asyncio
import os
import sys
import time

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker
import torch
import httpx


async def do_training_step(training_client: tinker.TrainingClient, tokenizer, prompt: str, response: str):
    """Do a single training step."""
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

    optim_future = await training_client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=5e-4))
    await optim_future.result_async()


async def main():
    print("=" * 80)
    print("LORA MISMATCH DIAGNOSTIC")
    print("=" * 80)

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

    # Do training steps
    print("\n[3] Doing 3 training steps with high LR...")
    training_examples = [
        ("<|im_start|>user\nWhat is 1+1?<|im_end|>\n<|im_start|>assistant\n", "2<|im_end|>"),
        ("<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n", "4<|im_end|>"),
        ("<|im_start|>user\nWhat is 3+3?<|im_end|>\n<|im_start|>assistant\n", "6<|im_end|>"),
    ]

    for i, (prompt, response) in enumerate(training_examples):
        t0 = time.time()
        await do_training_step(training_client, tokenizer, prompt, response)
        print(f"    Step {i+1}/3 completed in {time.time()-t0:.1f}s")

    # Export to vLLM
    print("\n[4] Exporting LoRA to vLLM...")
    t0 = time.time()
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()
    print(f"    Export took {time.time()-t0:.1f}s")

    # Check exported safetensors
    print("\n[5] Checking exported weights...")
    from safetensors import safe_open

    temp_dirs = [d for d in os.listdir("/tmp") if d.startswith("tinker_lora_")]
    if not temp_dirs:
        print("    ERROR: No exported LoRA found!")
        return

    temp_dirs.sort(key=lambda x: os.path.getmtime(f"/tmp/{x}"), reverse=True)
    adapter_path = f"/tmp/{temp_dirs[0]}/adapter_model.safetensors"
    print(f"    Adapter path: {adapter_path}")

    # Analyze exported weights
    with safe_open(adapter_path, framework="pt") as f:
        keys = list(f.keys())
        print(f"\n    Total keys: {len(keys)}")

        # Find a layer 0 expert 0 gate_proj lora_B
        target_key = None
        for k in keys:
            if "layers.0.mlp.experts.0.gate_proj.lora_B" in k:
                target_key = k
                break

        if target_key:
            tensor = f.get_tensor(target_key)
            print(f"\n    Sample expert weight: {target_key}")
            print(f"    Shape: {tensor.shape}")
            print(f"    Norm: {tensor.norm().item():.6f}")
            print(f"    Max abs: {tensor.abs().max().item():.6f}")
            print(f"    First 5 values: {tensor.flatten()[:5].tolist()}")

            # For ETP=1, the exported weight should be full [8192, 16]
            if tensor.shape[0] == 8192:
                print(f"    >>> CORRECT: Full intermediate size (8192)")
            else:
                print(f"    >>> WARNING: Unexpected size {tensor.shape[0]}")

        # Check multiple experts to see if they're clones
        print("\n    Checking if experts are clones (for ETP=1 shared LoRA):")
        expert_weights = []
        for eid in range(min(8, 64)):  # Check first 8 experts
            key = f"base_model.model.model.layers.0.mlp.experts.{eid}.gate_proj.lora_B.weight"
            if key in keys:
                t = f.get_tensor(key)
                expert_weights.append(t)
                print(f"      Expert {eid}: norm={t.norm().item():.4f}, max={t.abs().max().item():.4f}")

        if len(expert_weights) >= 2:
            # Check if they're identical (as expected for ETP=1)
            diff = (expert_weights[0] - expert_weights[1]).abs().max().item()
            if diff < 1e-6:
                print(f"    >>> Experts 0 and 1 are IDENTICAL (diff={diff:.2e}) - correct for shared LoRA")
            else:
                print(f"    >>> Experts 0 and 1 DIFFER (max_diff={diff:.4f}) - this could be a bug!")

    # Now test logprobs
    print("\n[6] Comparing logprobs between vLLM and Megatron...")
    test_prompt = "<|im_start|>user\nWhat is 2+3?<|im_end|>\n<|im_start|>assistant\n"
    test_response = "5<|im_end|>"

    prompt_tokens = tokenizer.encode(test_prompt, add_special_tokens=False)
    full_tokens = tokenizer.encode(test_prompt + test_response, add_special_tokens=False)
    prompt_len = len(prompt_tokens)

    print(f"    Prompt: {test_prompt!r}")
    print(f"    Response: {test_response!r}")
    print(f"    Prompt tokens: {len(prompt_tokens)}, Response tokens: {len(full_tokens) - len(prompt_tokens)}")

    # Get vLLM logprobs
    model_input = tinker.ModelInput.from_ints(full_tokens)
    vllm_logprobs = await sampling_client.compute_logprobs_async(model_input)

    # Get Megatron logprobs via training client
    # This requires calling the training endpoint
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            f"{base_url}/api/v1/compute_logprobs_trainer",
            json={
                "model_id": training_client._model_id,
                "input_ids": [full_tokens],
                "prompt_len": prompt_len,
            }
        )
        if response.status_code != 200:
            print(f"    ERROR: Failed to get Megatron logprobs: {response.text}")
            megatron_logprobs = None
        else:
            result = response.json()
            megatron_logprobs = result["logprobs"][0]

    # Compare
    if megatron_logprobs:
        print(f"\n    Token-by-token comparison:")
        print(f"    {'Pos':>4} | {'Token':>8} | {'Text':>15} | {'vLLM':>12} | {'Megatron':>12} | {'Diff':>12}")
        print(f"    {'-' * 80}")

        total_abs_diff = 0.0
        max_diff = 0.0
        for i, tid in enumerate(full_tokens[prompt_len:]):
            v_lp = vllm_logprobs[prompt_len + i] if prompt_len + i < len(vllm_logprobs) else float('nan')
            m_lp = megatron_logprobs[i] if i < len(megatron_logprobs) else float('nan')
            text = tokenizer.decode([tid])
            diff = abs(v_lp - m_lp)
            total_abs_diff += diff
            max_diff = max(max_diff, diff)
            flag = " ***" if diff > 1.0 else ""
            print(f"    {prompt_len + i:>4} | {tid:>8} | {repr(text):>15} | {v_lp:>12.4f} | {m_lp:>12.4f} | {diff:>12.4f}{flag}")

        print(f"\n    Summary:")
        print(f"    Total abs diff: {total_abs_diff:.4f}")
        print(f"    Max diff: {max_diff:.4f}")

        if max_diff > 10.0:
            print(f"\n    >>> MASSIVE DIVERGENCE DETECTED!")
            print(f"        This indicates a bug in LoRA export or application.")
        elif max_diff > 1.0:
            print(f"\n    >>> Significant divergence detected.")
        else:
            print(f"\n    >>> Logprobs match reasonably well.")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
