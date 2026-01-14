#!/usr/bin/env python3
"""Diagnose trained LoRA divergence between vLLM and Megatron.

This test:
1. Creates fresh training session
2. Gets vLLM logprobs (fresh LoRA)
3. Gets Megatron logprobs (fresh LoRA)
4. Trains 1 step
5. Exports trained LoRA to vLLM
6. Gets vLLM logprobs (trained LoRA)
7. Gets Megatron logprobs (trained LoRA)
8. Compares all four

Expected: fresh vLLM ≈ fresh Megatron, trained vLLM ≈ trained Megatron
Bug: if fresh matches but trained diverges
"""

import asyncio
import os
import time

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker
import torch


async def get_megatron_logprobs(training_client, tokens, prompt_len):
    """Get Megatron logprobs via forward pass."""
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    mask = [0.0] * (prompt_len - 1) + [1.0] * (len(input_tokens) - prompt_len + 1)

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.zeros(len(input_tokens), dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.zeros(len(input_tokens), dtype=torch.float32)),
        }
    )

    fwd_future = await training_client.forward_async([datum], loss_fn="importance_sampling")
    fwd_result = await fwd_future.result_async()

    if not fwd_result.loss_fn_outputs:
        return None

    logprobs = fwd_result.loss_fn_outputs[0]["logprobs"].to_torch()
    # Return only response token logprobs (after prompt)
    return logprobs[prompt_len-1:].tolist()


async def get_vllm_logprobs(sampling_client, tokens, prompt_len):
    """Get vLLM logprobs via compute_logprobs.

    vLLM compute_logprobs returns: logprobs[i] = log P(token[i+1] | token[0:i+1])
    For first response token t_P, we need logprobs[P-1].
    """
    model_input = tinker.ModelInput.from_ints(tokens)
    logprobs = await sampling_client.compute_logprobs_async(model_input)
    # Return only response token logprobs (after prompt)
    # First response token t_P has logprob at index P-1
    return list(logprobs)[prompt_len-1:]


async def do_training_step(training_client, tokenizer, prompt: str, response: str, lr=5e-4):
    """Do a single training step."""
    full_text = prompt + response
    tokens = tokenizer.encode(full_text, add_special_tokens=False)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_len = len(prompt_tokens)

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    mask = [0.0] * (prompt_len - 1) + [1.0] * (len(input_tokens) - prompt_len + 1)
    advantages = [1.0] * len(input_tokens)  # High advantage to force learning
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

    optim_future = await training_client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=lr))
    await optim_future.result_async()


def compare_logprobs(lp1, lp2, name1, name2, tokens, tokenizer, prompt_len):
    """Compare two logprob lists and print analysis."""
    print(f"\n{'pos':>4} | {'token':>8} | {'text':>15} | {name1:>12} | {name2:>12} | {'diff':>12}")
    print("-" * 80)

    max_diff = 0.0
    total_diff = 0.0
    count = 0

    for i, tid in enumerate(tokens[prompt_len:]):
        l1 = lp1[i] if i < len(lp1) else float('nan')
        l2 = lp2[i] if i < len(lp2) else float('nan')
        text = tokenizer.decode([tid])
        diff = l1 - l2 if not (l1 != l1 or l2 != l2) else 0.0
        total_diff += abs(diff)
        max_diff = max(max_diff, abs(diff))
        count += 1
        flag = " ***" if abs(diff) > 1.0 else ""
        print(f"{prompt_len + i:>4} | {tid:>8} | {repr(text):>15} | {l1:>12.4f} | {l2:>12.4f} | {diff:>+12.4f}{flag}")

    mean_diff = total_diff / max(count, 1)
    print(f"\nSummary: mean_diff={mean_diff:.4f}, max_diff={max_diff:.4f}")
    return mean_diff, max_diff


async def main():
    print("=" * 80)
    print("TRAINED LORA DIVERGENCE DIAGNOSTIC")
    print("=" * 80)

    model_name = "moonshotai/Moonlight-16B-A3B-Instruct"

    # Load tokenizer
    print("\n[1] Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Test sequence
    test_prompt = "<|im_start|>user\nWhat is 2+3?<|im_end|>\n<|im_start|>assistant\n"
    test_response = "5<|im_end|>"
    full_text = test_prompt + test_response

    prompt_tokens = tokenizer.encode(test_prompt, add_special_tokens=False)
    full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
    prompt_len = len(prompt_tokens)

    print(f"    Test: {repr(test_prompt[:50])}... -> {repr(test_response)}")
    print(f"    Tokens: prompt={prompt_len}, response={len(full_tokens)-prompt_len}")

    # Create training session
    print("\n[2] Creating training session with fresh LoRA...")
    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])
    training_client = await service_client.create_lora_training_client_async(model_name, rank=16)
    print(f"    model_id: {training_client.model_id}")

    # Get fresh LoRA sampling client
    print("\n[3] Exporting fresh LoRA to vLLM...")
    t0 = time.time()
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()
    print(f"    Export took {time.time()-t0:.1f}s")

    # Get logprobs from both systems with FRESH LoRA
    print("\n[4] Getting logprobs with FRESH LoRA...")
    t0 = time.time()
    fresh_vllm = await get_vllm_logprobs(sampling_client, full_tokens, prompt_len)
    print(f"    vLLM: {len(fresh_vllm)} logprobs in {time.time()-t0:.1f}s")

    t0 = time.time()
    fresh_megatron = await get_megatron_logprobs(training_client, full_tokens, prompt_len)
    print(f"    Megatron: {len(fresh_megatron)} logprobs in {time.time()-t0:.1f}s")

    print("\n--- FRESH LORA COMPARISON ---")
    fresh_mean, fresh_max = compare_logprobs(
        fresh_vllm, fresh_megatron, "vLLM", "Megatron",
        full_tokens, tokenizer, prompt_len
    )

    # Train 1 step
    print("\n[5] Training 1 step with high LR (5e-4)...")
    training_example = ("<|im_start|>user\nWhat is 1+1?<|im_end|>\n<|im_start|>assistant\n", "2<|im_end|>")
    t0 = time.time()
    await do_training_step(training_client, tokenizer, training_example[0], training_example[1], lr=5e-4)
    print(f"    Training step took {time.time()-t0:.1f}s")

    # Export trained LoRA to vLLM
    print("\n[6] Exporting TRAINED LoRA to vLLM...")
    t0 = time.time()
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()
    print(f"    Export took {time.time()-t0:.1f}s")

    # Get logprobs from both systems with TRAINED LoRA
    print("\n[7] Getting logprobs with TRAINED LoRA...")
    t0 = time.time()
    trained_vllm = await get_vllm_logprobs(sampling_client, full_tokens, prompt_len)
    print(f"    vLLM: {len(trained_vllm)} logprobs in {time.time()-t0:.1f}s")

    t0 = time.time()
    trained_megatron = await get_megatron_logprobs(training_client, full_tokens, prompt_len)
    print(f"    Megatron: {len(trained_megatron)} logprobs in {time.time()-t0:.1f}s")

    print("\n--- TRAINED LORA COMPARISON ---")
    trained_mean, trained_max = compare_logprobs(
        trained_vllm, trained_megatron, "vLLM", "Megatron",
        full_tokens, tokenizer, prompt_len
    )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Fresh LoRA:   mean_diff={fresh_mean:.4f}, max_diff={fresh_max:.4f}")
    print(f"  Trained LoRA: mean_diff={trained_mean:.4f}, max_diff={trained_max:.4f}")

    if fresh_max < 0.5 and trained_max > 5.0:
        print("\n  >>> BUG CONFIRMED: Fresh LoRA matches, trained LoRA DIVERGES!")
        print("      The LoRA export or vLLM loading has a bug for non-zero weights.")
    elif trained_max < 0.5:
        print("\n  >>> NO BUG: Both fresh and trained LoRA produce matching logprobs.")
    else:
        print(f"\n  >>> Some divergence detected. Fresh={fresh_max:.2f}, Trained={trained_max:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
