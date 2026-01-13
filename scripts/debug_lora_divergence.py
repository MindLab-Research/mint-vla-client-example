#!/usr/bin/env python3
"""Debug LoRA divergence between vLLM and Megatron.

This test:
1. Initializes a fresh training session
2. Trains for 1 step
3. Samples from vLLM with trained LoRA
4. Computes logprobs in BOTH vLLM and Megatron
5. Prints detailed comparison for each token
"""
import asyncio
import json
import os
import sys
from typing import Optional

import httpx
import tinker

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

async def main():
    base_url = os.environ["TINKER_BASE_URL"]

    # Create HTTP client
    client = type('Client', (), {})()
    client.http = httpx.AsyncClient(base_url=base_url, timeout=300.0)

    model_name = "moonshotai/Moonlight-16B-A3B-Instruct"
    rank = 16
    lr = 1e-4

    # Create training session
    print(f"Creating training session: {model_name}, rank={rank}")
    session_resp = await client.http.post(
        "/api/v1/create_session",
        json={
            "model_name": model_name,
            "max_lora_rank": rank,
        }
    )
    session_data = session_resp.json()
    session_id = session_data["session_id"]
    model_id = f"{session_id}_0"
    print(f"Session: {session_id}")

    # Initialize LoRA
    print("\nInitializing fresh LoRA...")
    init_resp = await client.http.post(
        "/api/v1/reinit_lora_weights",
        json={
            "model_id": model_id,
            "learning_rate": lr,
            "rank": rank,
        }
    )
    print(f"Init response: {init_resp.json()}")

    # Create a simple training batch
    prompt = "Calculate 5 + 3 = "
    completion = "8"

    # Construct token IDs
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    full_text = prompt + completion
    token_ids = tokenizer.encode(full_text, add_special_tokens=False)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    prompt_len = len(prompt_ids)
    total_len = len(token_ids)

    print(f"\nPrompt: {prompt!r}")
    print(f"Completion: {completion!r}")
    print(f"Prompt tokens: {prompt_ids}")
    print(f"Full tokens: {token_ids}")
    print(f"Prompt len: {prompt_len}, Total len: {total_len}")

    # Check logprobs BEFORE training using BOTH systems
    print("\n" + "="*60)
    print("BEFORE TRAINING - Comparing logprobs")
    print("="*60)

    # Get vLLM logprobs via sampling
    sample_resp = await client.http.post(
        "/api/v1/asample",
        json={
            "model_id": model_id,
            "prompts": [{"prompt_token_ids": prompt_ids}],
            "sampling_params": {
                "temperature": 0.0,
                "max_tokens": 10,
                "logprobs": 5,
                "prompt_logprobs": 1,
            }
        }
    )
    future_id = sample_resp.json()["future_id"]

    # Poll for result
    while True:
        result_resp = await client.http.post(
            "/api/v1/retrieve_future",
            json={"future_id": future_id}
        )
        if result_resp.status_code == 200:
            sample_result = result_resp.json()
            break
        await asyncio.sleep(0.5)

    # Get the sampled completion
    gen_output = sample_result["outputs"][0]
    gen_token_ids = gen_output.get("token_ids", [])
    print(f"\nvLLM generated: {tokenizer.decode(gen_token_ids)}")
    print(f"Token IDs: {gen_token_ids}")

    # Now compare on this specific sequence
    test_ids = prompt_ids + gen_token_ids[:5]  # First 5 generated tokens
    print(f"\nTest sequence: {test_ids}")

    # Get vLLM compute logprobs (separate from sampling)
    print("\n--- vLLM Compute Logprobs ---")
    compute_resp = await client.http.post(
        "/api/v1/compute_logprobs",
        json={
            "model_id": model_id,
            "input_ids": test_ids,
            "prompt_len": prompt_len,
        }
    )
    vllm_logprobs = compute_resp.json()["logprobs"]
    print(f"vLLM logprobs: {vllm_logprobs[:10]}")

    # Get Megatron compute logprobs
    print("\n--- Megatron Compute Logprobs ---")
    train_compute_resp = await client.http.post(
        "/api/v1/compute_logprobs_trainer",
        json={
            "model_id": model_id,
            "input_ids": [test_ids],  # Batched
            "prompt_len": prompt_len,
        }
    )
    megatron_logprobs = train_compute_resp.json()["logprobs"][0]
    print(f"Megatron logprobs: {megatron_logprobs[:10]}")

    # Compare token by token
    print("\n--- Token-by-token comparison (BEFORE training) ---")
    print(f"{'Pos':>4} | {'Token':>15} | {'vLLM':>12} | {'Megatron':>12} | {'Diff':>10}")
    print("-" * 70)

    for i, tid in enumerate(test_ids[prompt_len:]):
        if i >= len(vllm_logprobs) or i >= len(megatron_logprobs):
            break
        v_lp = vllm_logprobs[i]
        m_lp = megatron_logprobs[i]
        diff = abs(v_lp - m_lp)
        token_text = tokenizer.decode([tid])
        print(f"{prompt_len + i:4} | {token_text:>15} | {v_lp:12.4f} | {m_lp:12.4f} | {diff:10.4f}")

    # Train for 1 step
    print("\n" + "="*60)
    print("TRAINING 1 STEP")
    print("="*60)

    # Create training data with random rewards
    import random
    random.seed(42)

    # Create 4 trajectories with different rewards
    batch = []
    for i in range(4):
        # Generate completion
        sample_resp = await client.http.post(
            "/api/v1/asample",
            json={
                "model_id": model_id,
                "prompts": [{"prompt_token_ids": prompt_ids}],
                "sampling_params": {
                    "temperature": 1.0,
                    "max_tokens": 20,
                }
            }
        )
        future_id = sample_resp.json()["future_id"]
        while True:
            result_resp = await client.http.post(
                "/api/v1/retrieve_future",
                json={"future_id": future_id}
            )
            if result_resp.status_code == 200:
                sample_result = result_resp.json()
                break
            await asyncio.sleep(0.1)

        traj_tokens = sample_result["outputs"][0].get("token_ids", [])
        traj_logprobs = sample_result["outputs"][0].get("logprobs", [])

        # Random reward
        reward = random.uniform(-1, 1)

        batch.append({
            "input_ids": prompt_ids + traj_tokens,
            "prompt_len": prompt_len,
            "sampling_logprobs": traj_logprobs,
            "rewards": [reward],
        })
        print(f"  Trajectory {i}: {len(traj_tokens)} tokens, reward={reward:.3f}")

    # Train
    train_resp = await client.http.post(
        "/api/v1/train_step",
        json={
            "model_id": model_id,
            "batch": batch,
            "loss_fn": "ppo",
        }
    )
    train_result = train_resp.json()
    print(f"\nTrain result: loss={train_result.get('loss', 'N/A')}")

    # Check logprobs AFTER training
    print("\n" + "="*60)
    print("AFTER TRAINING - Comparing logprobs")
    print("="*60)

    # Get vLLM compute logprobs
    print("\n--- vLLM Compute Logprobs ---")
    compute_resp = await client.http.post(
        "/api/v1/compute_logprobs",
        json={
            "model_id": model_id,
            "input_ids": test_ids,
            "prompt_len": prompt_len,
        }
    )
    vllm_logprobs_after = compute_resp.json()["logprobs"]
    print(f"vLLM logprobs: {vllm_logprobs_after[:10]}")

    # Get Megatron compute logprobs
    print("\n--- Megatron Compute Logprobs ---")
    train_compute_resp = await client.http.post(
        "/api/v1/compute_logprobs_trainer",
        json={
            "model_id": model_id,
            "input_ids": [test_ids],
            "prompt_len": prompt_len,
        }
    )
    megatron_logprobs_after = train_compute_resp.json()["logprobs"][0]
    print(f"Megatron logprobs: {megatron_logprobs_after[:10]}")

    # Compare token by token
    print("\n--- Token-by-token comparison (AFTER training) ---")
    print(f"{'Pos':>4} | {'Token':>15} | {'vLLM':>12} | {'Megatron':>12} | {'Diff':>10} | {'vLLM Δ':>10}")
    print("-" * 85)

    max_diff = 0
    max_diff_pos = -1
    for i, tid in enumerate(test_ids[prompt_len:]):
        if i >= len(vllm_logprobs_after) or i >= len(megatron_logprobs_after):
            break
        v_lp = vllm_logprobs_after[i]
        m_lp = megatron_logprobs_after[i]
        diff = abs(v_lp - m_lp)
        v_delta = v_lp - vllm_logprobs[i] if i < len(vllm_logprobs) else 0
        token_text = tokenizer.decode([tid])

        flag = " ***" if diff > 1.0 else ""
        print(f"{prompt_len + i:4} | {token_text:>15} | {v_lp:12.4f} | {m_lp:12.4f} | {diff:10.4f} | {v_delta:10.4f}{flag}")

        if diff > max_diff:
            max_diff = diff
            max_diff_pos = prompt_len + i

    print(f"\nMax diff: {max_diff:.4f} at position {max_diff_pos}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    before_diffs = [abs(vllm_logprobs[i] - megatron_logprobs[i])
                   for i in range(min(len(vllm_logprobs), len(megatron_logprobs)))]
    after_diffs = [abs(vllm_logprobs_after[i] - megatron_logprobs_after[i])
                  for i in range(min(len(vllm_logprobs_after), len(megatron_logprobs_after)))]

    print(f"BEFORE training: mean={sum(before_diffs)/len(before_diffs):.4f}, max={max(before_diffs):.4f}")
    print(f"AFTER training:  mean={sum(after_diffs)/len(after_diffs):.4f}, max={max(after_diffs):.4f}")

    if max(after_diffs) > 1.0:
        print("\n!!! BUG CONFIRMED: Large divergence after training !!!")
    else:
        print("\nLogprobs match - no bug detected")

if __name__ == "__main__":
    asyncio.run(main())
