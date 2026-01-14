#!/usr/bin/env python3
"""Diagnose LoRA mismatch by comparing logprobs between vLLM and Megatron.

Uses the /forward endpoint to get Megatron logprobs.
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


async def do_training_step(training_client: tinker.TrainingClient, tokenizer, prompt: str, response: str, lr=5e-4):
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

    optim_future = await training_client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=lr))
    await optim_future.result_async()


async def get_megatron_logprobs(base_url: str, model_id: str, full_tokens: list[int]):
    """Get logprobs from Megatron via /forward endpoint."""
    async with httpx.AsyncClient(timeout=300.0) as client:
        # Submit forward request
        request_body = {
            "model_id": model_id,
            "forward_input": [{
                "model_input": {"token_ids": full_tokens},
            }]
        }
        response = await client.post(f"{base_url}/api/v1/forward", json=request_body)
        if response.status_code != 200:
            raise Exception(f"Forward request failed: {response.text}")

        future_result = response.json()
        future_id = future_result["future_id"]

        # Poll for result
        for _ in range(60):
            result_response = await client.post(
                f"{base_url}/api/v1/retrieve_future",
                json={"future_id": future_id}
            )
            if result_response.status_code == 200:
                result = result_response.json()
                return result.get("logprobs", [])
            await asyncio.sleep(1.0)

        raise Exception("Timeout waiting for forward result")


async def get_logprobs(base_url: str, training_client, sampling_client, full_tokens, prompt_len, tokenizer):
    """Get logprobs from both vLLM and Megatron."""
    # vLLM logprobs
    model_input = tinker.ModelInput.from_ints(full_tokens)
    vllm_logprobs = await sampling_client.compute_logprobs_async(model_input)

    # Megatron logprobs via forward pass
    megatron_result = await get_megatron_logprobs(base_url, training_client.model_id, full_tokens)

    # The forward result structure may vary - let's print and inspect
    print(f"    Megatron forward result structure: {type(megatron_result)}")
    if isinstance(megatron_result, list) and len(megatron_result) > 0:
        first = megatron_result[0]
        print(f"    First element: {type(first)}")
        if isinstance(first, dict):
            print(f"    Keys: {first.keys()}")
            if "logprobs" in first:
                megatron_logprobs = first["logprobs"]
            else:
                megatron_logprobs = megatron_result
        elif isinstance(first, (int, float)):
            megatron_logprobs = megatron_result
        else:
            megatron_logprobs = megatron_result
    else:
        megatron_logprobs = megatron_result

    return list(vllm_logprobs), megatron_logprobs


def compare_logprobs(vllm_lp, megatron_lp, full_tokens, prompt_len, tokenizer, label=""):
    """Compare logprobs and print detailed analysis."""
    print(f"\n    Token-by-token comparison {label}:")
    print(f"    {'Pos':>4} | {'Token':>8} | {'Text':>15} | {'vLLM':>12} | {'Megatron':>12} | {'Diff':>12}")
    print(f"    {'-' * 80}")

    total_abs_diff = 0.0
    max_diff = 0.0
    max_diff_pos = -1
    num_compared = 0

    for i, tid in enumerate(full_tokens[prompt_len:]):
        # vLLM logprobs are indexed from the prompt
        v_idx = prompt_len + i
        v_lp = vllm_lp[v_idx] if v_idx < len(vllm_lp) else float('nan')

        # Megatron logprobs - need to figure out indexing
        m_lp = megatron_lp[i] if i < len(megatron_lp) else float('nan')

        text = tokenizer.decode([tid])
        diff = abs(v_lp - m_lp) if not (v_lp != v_lp or m_lp != m_lp) else 0.0
        total_abs_diff += diff
        num_compared += 1
        if diff > max_diff:
            max_diff = diff
            max_diff_pos = i
        flag = " ***" if diff > 1.0 else ""
        print(f"    {prompt_len + i:>4} | {tid:>8} | {repr(text):>15} | {v_lp:>12.4f} | {m_lp:>12.4f} | {diff:>12.4f}{flag}")

    mean_diff = total_abs_diff / max(num_compared, 1)
    print(f"\n    Summary: mean_diff={mean_diff:.4f}, max_diff={max_diff:.4f} at pos {max_diff_pos}")
    return max_diff


async def main():
    print("=" * 80)
    print("LORA LOGPROBS COMPARISON: vLLM vs Megatron")
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

    # Test sequence
    test_prompt = "<|im_start|>user\nWhat is 2+3?<|im_end|>\n<|im_start|>assistant\n"
    test_response = "5"  # Keep it simple
    prompt_tokens = tokenizer.encode(test_prompt, add_special_tokens=False)
    full_tokens = tokenizer.encode(test_prompt + test_response, add_special_tokens=False)
    prompt_len = len(prompt_tokens)

    print(f"    Test: {test_prompt!r} -> {test_response!r}")
    print(f"    Tokens: prompt={prompt_len}, response={len(full_tokens)-prompt_len}")

    # Get fresh LoRA logprobs
    print("\n[3] Exporting fresh LoRA and comparing logprobs...")
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()

    try:
        vllm_lp, megatron_lp = await get_logprobs(base_url, training_client, sampling_client, full_tokens, prompt_len, tokenizer)
        fresh_max_diff = compare_logprobs(vllm_lp, megatron_lp, full_tokens, prompt_len, tokenizer, "(FRESH LORA)")
    except Exception as e:
        print(f"    ERROR getting logprobs: {e}")
        fresh_max_diff = None

    # Train 1 step
    print("\n[4] Training 1 step with high LR (5e-4)...")
    training_example = ("<|im_start|>user\nWhat is 1+1?<|im_end|>\n<|im_start|>assistant\n", "2<|im_end|>")
    t0 = time.time()
    await do_training_step(training_client, tokenizer, training_example[0], training_example[1], lr=5e-4)
    print(f"    Training step took {time.time()-t0:.1f}s")

    # Export trained LoRA and compare
    print("\n[5] Exporting trained LoRA and comparing logprobs...")
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()

    try:
        vllm_lp_after, megatron_lp_after = await get_logprobs(base_url, training_client, sampling_client, full_tokens, prompt_len, tokenizer)
        trained_max_diff = compare_logprobs(vllm_lp_after, megatron_lp_after, full_tokens, prompt_len, tokenizer, "(AFTER 1 STEP)")
    except Exception as e:
        print(f"    ERROR getting logprobs: {e}")
        trained_max_diff = None

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if fresh_max_diff is not None:
        print(f"  Fresh LoRA max diff: {fresh_max_diff:.4f} nats")
    if trained_max_diff is not None:
        print(f"  Trained LoRA max diff: {trained_max_diff:.4f} nats")

    if fresh_max_diff is not None and trained_max_diff is not None:
        if fresh_max_diff < 1.0 and trained_max_diff > 10.0:
            print("\n  >>> BUG CONFIRMED: Fresh LoRA matches, trained LoRA diverges!")
        elif trained_max_diff < 1.0:
            print("\n  >>> OK: Both fresh and trained LoRA produce matching logprobs.")
        else:
            print(f"\n  >>> Some divergence detected, needs investigation.")


if __name__ == "__main__":
    asyncio.run(main())
