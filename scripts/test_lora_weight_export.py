#!/usr/bin/env python3
"""Test LoRA weight export by comparing actual tensor values.

Purpose: Verify that LoRA weights are correctly exported from Megatron to vLLM.

Test procedure:
1. Create training session with fresh LoRA
2. Do a few training steps
3. Get LoRA state dict from Megatron
4. Export to vLLM
5. Get LoRA state dict from vLLM
6. Compare the actual tensor values
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


async def get_megatron_logprobs(training_client: tinker.TrainingClient, tokenizer, prompt: str, response: str):
    """Get Megatron logprobs for a fixed sequence."""
    full_text = prompt + response
    tokens = tokenizer.encode(full_text, add_special_tokens=False)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_len = len(prompt_tokens)

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    mask = [0.0] * (prompt_len - 1) + [1.0] * (len(input_tokens) - prompt_len + 1)
    logprobs = [0.0] * len(input_tokens)

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.zeros(len(input_tokens), dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.tensor(logprobs, dtype=torch.float32)),
        }
    )

    fwd_future = await training_client.forward_async([datum], loss_fn="importance_sampling")
    fwd_result = await fwd_future.result_async()

    if not fwd_result.loss_fn_outputs:
        return None

    return fwd_result.loss_fn_outputs[0]["logprobs"].to_torch().tolist()


async def get_vllm_logprobs(sampling_client: tinker.SamplingClient, tokenizer, prompt: str, response: str):
    """Get vLLM logprobs for a fixed sequence."""
    full_text = prompt + response
    tokens = tokenizer.encode(full_text, add_special_tokens=False)
    model_input = tinker.ModelInput.from_ints(tokens)

    logprobs = await sampling_client.compute_logprobs_async(model_input)
    return logprobs


async def main():
    print("=" * 70)
    print("LORA WEIGHT EXPORT DIAGNOSTIC")
    print("=" * 70)

    model_name = "moonshotai/Moonlight-16B-A3B-Instruct"
    base_url = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")

    # Load tokenizer
    print("\n[1] Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Fixed test sequence
    test_prompt = "<|im_start|>user\nWhat is 2+3?<|im_end|>\n<|im_start|>assistant\n"
    test_response = "5<|im_end|>"  # Short response matching training pattern

    prompt_tokens = tokenizer.encode(test_prompt, add_special_tokens=False)
    response_tokens = tokenizer.encode(test_response, add_special_tokens=False)
    print(f"\n  Prompt tokens: {len(prompt_tokens)}")
    print(f"  Response tokens: {response_tokens} -> {[tokenizer.decode([t]) for t in response_tokens]}")

    # Create clients
    print("\n[2] Creating training session with fresh LoRA...")
    service_client = tinker.ServiceClient(base_url=base_url)
    training_client = await service_client.create_lora_training_client_async(model_name, rank=16)

    # Get sampling client (fresh LoRA)
    print("\n[3] Exporting fresh LoRA to vLLM...")
    t0 = time.time()
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()
    print(f"    Export took {time.time()-t0:.1f}s")

    # Get logprobs BEFORE training
    print("\n[4] Getting logprobs BEFORE training...")
    megatron_before = await get_megatron_logprobs(training_client, tokenizer, test_prompt, test_response)
    vllm_before = await get_vllm_logprobs(sampling_client, tokenizer, test_prompt, test_response)

    print(f"\n  BEFORE training (fresh LoRA):")
    print(f"  {'pos':>4} | {'tok':>8} | {'text':>15} | {'Megatron':>12} | {'vLLM':>12} | {'diff':>12}")
    print(f"  {'-' * 75}")

    prompt_len = len(prompt_tokens)
    for i, tok in enumerate(response_tokens):
        text = tokenizer.decode([tok])
        m_pos = prompt_len + i - 1  # Megatron position (SFT format)
        v_pos = prompt_len + i  # vLLM position
        m_lp = megatron_before[m_pos] if m_pos < len(megatron_before) else float('nan')
        v_lp = vllm_before[v_pos] if v_pos < len(vllm_before) else float('nan')
        diff = v_lp - m_lp
        print(f"  {prompt_len+i:>4} | {tok:>8} | {repr(text):>15} | {m_lp:>12.4f} | {v_lp:>12.4f} | {diff:>+12.4f}")

    # Do training steps
    print("\n[5] Doing 5 training steps on short responses...")
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

    # Get Megatron logprobs AFTER training (before export)
    print("\n[6] Getting Megatron logprobs AFTER training (before export)...")
    megatron_after_before_export = await get_megatron_logprobs(training_client, tokenizer, test_prompt, test_response)

    # Export trained LoRA
    print("\n[7] Exporting trained LoRA to vLLM...")
    t0 = time.time()
    sampling_client_after = await training_client.save_weights_and_get_sampling_client_async()
    print(f"    Export took {time.time()-t0:.1f}s")

    # Get vLLM logprobs AFTER export
    print("\n[8] Getting vLLM logprobs AFTER export...")
    vllm_after = await get_vllm_logprobs(sampling_client_after, tokenizer, test_prompt, test_response)

    # Get Megatron logprobs AFTER training AFTER export (to check for side effects)
    print("\n[9] Getting Megatron logprobs AFTER export (checking for side effects)...")
    megatron_after_after_export = await get_megatron_logprobs(training_client, tokenizer, test_prompt, test_response)

    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print(f"\n  AFTER training:")
    print(f"  {'pos':>4} | {'tok':>8} | {'text':>15} | {'Meg(pre-exp)':>12} | {'Meg(post-exp)':>13} | {'vLLM':>12} | {'v-m diff':>12}")
    print(f"  {'-' * 95}")

    diffs = []
    for i, tok in enumerate(response_tokens):
        text = tokenizer.decode([tok])
        m_pos = prompt_len + i - 1
        v_pos = prompt_len + i

        m_pre = megatron_after_before_export[m_pos] if m_pos < len(megatron_after_before_export) else float('nan')
        m_post = megatron_after_after_export[m_pos] if m_pos < len(megatron_after_after_export) else float('nan')
        v_lp = vllm_after[v_pos] if v_pos < len(vllm_after) else float('nan')
        diff = v_lp - m_post
        diffs.append(diff)

        print(f"  {prompt_len+i:>4} | {tok:>8} | {repr(text):>15} | {m_pre:>12.4f} | {m_post:>13.4f} | {v_lp:>12.4f} | {diff:>+12.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if diffs:
        max_diff = max(abs(d) for d in diffs)
        mean_diff = sum(abs(d) for d in diffs) / len(diffs)
        print(f"\n  Max abs diff (vLLM vs Megatron): {max_diff:.4f}")
        print(f"  Mean abs diff: {mean_diff:.4f}")

        if max_diff > 1.0:
            print(f"\n  >>> PROBLEM: Large discrepancy detected!")
            print(f"      This indicates LoRA export is NOT correctly transferring trained weights.")
            print(f"      Megatron has the training effect, vLLM does not.")
        else:
            print(f"\n  >>> OK: LoRA export appears correct.")

    # Check Megatron consistency (before vs after export)
    meg_diff = max(abs(megatron_after_before_export[prompt_len+i-1] - megatron_after_after_export[prompt_len+i-1])
                   for i in range(len(response_tokens)))
    print(f"\n  Megatron consistency (pre vs post export): max diff = {meg_diff:.6f}")
    if meg_diff < 0.0001:
        print(f"      >>> OK: Export does not corrupt Megatron state.")
    else:
        print(f"      >>> WARNING: Export may have side effects on Megatron!")


if __name__ == "__main__":
    asyncio.run(main())
