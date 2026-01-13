#!/usr/bin/env python
"""Test script to debug topk_prompt_logprobs API.

This script tests the top-K prompt logprobs functionality with both:
1. Base model session (no LoRA)
2. LoRA session from training

Uses pre-encoded token IDs (Qwen2.5 tokenizer).
"""

import asyncio
import os
import httpx

# Configuration
TINKER_BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# Qwen2.5 token IDs for "Hello, how are you today?"
TEST_TOKENS = [9707, 11, 1246, 525, 498, 3351, 30]


async def poll_future(client: httpx.AsyncClient, request_id: str, timeout: int = 120) -> dict | None:
    """Poll a future until completion or timeout."""
    for _ in range(timeout):
        resp = await client.post("/api/v1/retrieve_future", json={"request_id": request_id})
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 408:
            await asyncio.sleep(1)
        else:
            print(f"   Error: {resp.status_code} {resp.text}")
            return None
    print("   Timeout waiting for future")
    return None


async def test_topk_base_model(client: httpx.AsyncClient) -> None:
    """Test top-K with base model (no LoRA)."""
    print("\n=== Test 1: Base Model Session ===")

    # Create sampling session (base model)
    print("[1] Creating base model sampling session...")
    create_resp = await client.post("/api/v1/create_sampling_session", json={
        "session_id": "test-topk-base",
        "base_model": MODEL_NAME,
    })
    if create_resp.status_code != 200:
        print(f"   FAILED: {create_resp.status_code} {create_resp.text}")
        return

    session_id = create_resp.json().get("sampling_session_id")
    print(f"   Session: {session_id}")

    # Call asample with topk_prompt_logprobs
    print("[2] Calling asample with topk_prompt_logprobs=10...")
    sample_resp = await client.post("/api/v1/asample", json={
        "sampling_session_id": session_id,
        "prompt": {"chunks": [{"type": "encoded_text", "tokens": TEST_TOKENS}]},
        "sampling_params": {"max_tokens": 5, "temperature": 0.7},
        "num_samples": 1,
        "topk_prompt_logprobs": 10,
    })
    if sample_resp.status_code != 200:
        print(f"   FAILED: {sample_resp.status_code} {sample_resp.text}")
        return

    request_id = sample_resp.json().get("request_id")
    result = await poll_future(client, request_id)
    if result is None:
        return

    topk = result.get("topk_prompt_logprobs", [])
    print(f"   topk_prompt_logprobs length: {len(topk)}")
    non_empty = sum(1 for d in topk if d)
    print(f"   Non-empty positions: {non_empty}/{len(topk)}")
    if topk and len(topk) > 1:
        print(f"   Sample (pos 1): {topk[1]}")
    print(f"   RESULT: {'PASS' if non_empty > 0 else 'FAIL'}")


async def test_topk_lora_from_training(client: httpx.AsyncClient) -> None:
    """Test top-K with LoRA session from training."""
    print("\n=== Test 2: LoRA Session from Training ===")

    # Create training model
    print("[1] Creating training model...")
    create_resp = await client.post("/api/v1/create_model", json={
        "session_id": "test-topk-lora",
        "model_seq_id": 1,
        "base_model": MODEL_NAME,
        "lora_config": {"rank": 16, "alpha": 32},
    })
    if create_resp.status_code != 200:
        print(f"   FAILED: {create_resp.status_code} {create_resp.text}")
        return

    request_id = create_resp.json().get("request_id")
    result = await poll_future(client, request_id, timeout=180)
    if result is None:
        return

    model_id = result.get("model_id")
    print(f"   Model: {model_id}")

    # Save weights to register for sampling
    print("[2] Saving weights to register for sampling...")
    save_resp = await client.post("/api/v1/save_weights", json={
        "model_id": model_id,
        "path": "test-topk-checkpoint"
    })
    if save_resp.status_code != 200:
        print(f"   FAILED: {save_resp.status_code} {save_resp.text}")
        return

    request_id = save_resp.json().get("request_id")
    result = await poll_future(client, request_id, timeout=180)
    if result is None:
        return
    print("   Weights saved and registered")

    # Call asample with topk_prompt_logprobs
    print("[3] Calling asample with topk_prompt_logprobs=10...")
    sample_resp = await client.post("/api/v1/asample", json={
        "sampling_session_id": model_id,  # Use model_id as session_id
        "prompt": {"chunks": [{"type": "encoded_text", "tokens": TEST_TOKENS}]},
        "sampling_params": {"max_tokens": 5, "temperature": 0.7},
        "num_samples": 1,
        "topk_prompt_logprobs": 10,
    })
    if sample_resp.status_code != 200:
        print(f"   FAILED: {sample_resp.status_code} {sample_resp.text}")
        return

    request_id = sample_resp.json().get("request_id")
    result = await poll_future(client, request_id)
    if result is None:
        return

    topk = result.get("topk_prompt_logprobs", [])
    print(f"   topk_prompt_logprobs length: {len(topk)}")
    non_empty = sum(1 for d in topk if d)
    print(f"   Non-empty positions: {non_empty}/{len(topk)}")
    if topk and len(topk) > 1:
        print(f"   Sample (pos 1): {topk[1]}")
    print(f"   RESULT: {'PASS' if non_empty > 0 else 'FAIL'}")


async def main():
    """Run all tests."""
    async with httpx.AsyncClient(base_url=TINKER_BASE_URL, timeout=300) as client:
        await test_topk_base_model(client)
        await test_topk_lora_from_training(client)


if __name__ == "__main__":
    asyncio.run(main())
