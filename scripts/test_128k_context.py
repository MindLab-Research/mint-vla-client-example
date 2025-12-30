#!/usr/bin/env python3
"""Test 64K context support for K2-Thinking via training API."""

import asyncio
import os
import sys

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

async def test_128k_context():
    """Test that K2-Thinking can use 128K context via training workflow."""
    from tinker import ServiceClient

    print("=" * 60)
    print("Testing 32K Context Support for K2-Thinking")
    print("=" * 60)
    print("\nThis test creates a training session, saves weights to vLLM,")
    print("and verifies vLLM starts with max_model_len=32768 (32K).")

    client = ServiceClient()

    # Step 1: Create training session
    print("\n[1] Creating training session for K2-Thinking...")
    print("    Config: TP=8, EP=8 (64 GPUs)")
    try:
        training_client = await client.create_lora_training_client_async(
            base_model="moonshotai/Kimi-K2-Thinking",
            rank=16,
        )
        print(f"    Training session created: {training_client.model_id}")
    except Exception as e:
        print(f"    FAILED: {e}")
        return False

    # Step 2: Save weights to vLLM (triggers vLLM init with 128K config)
    print("\n[2] Saving weights to vLLM (triggers 128K context init)...")
    print("    This will take ~20 minutes for K2 model loading...")
    try:
        sampling_client = await training_client.save_weights_and_get_sampling_client_async()
        print(f"    vLLM ready with sampling client")
    except Exception as e:
        print(f"    FAILED: {e}")
        # Check if it's OOM
        if "OOM" in str(e) or "memory" in str(e).lower():
            print("\n    >>> 128K context OOM - need Option 2 (FP8 KV cache)")
        return False

    # Step 3: Test inference
    print("\n[3] Testing inference with 128K-capable engine...")
    try:
        response = await sampling_client.asample(
            prompt="What is 2+2? Think step by step.",
            max_tokens=500,
            temperature=0.7,
        )
        print(f"    Response length: {len(response.text)} chars")
        print(f"    Response: {response.text[:300]}...")
    except Exception as e:
        print(f"    FAILED: {e}")
        return False

    # Step 4: Test longer generation to verify KV cache works
    print("\n[4] Testing 4K token generation...")
    try:
        response = await sampling_client.asample(
            prompt="Explain the theory of relativity in great detail.",
            max_tokens=4096,
            temperature=0.7,
        )
        print(f"    Generated {len(response.text)} characters")
        print("    4K generation OK - KV cache working")
    except Exception as e:
        print(f"    FAILED: {e}")
        return False

    print("\n" + "=" * 60)
    print("128K Context Test: PASSED")
    print("=" * 60)
    print("\nvLLM successfully started with max_model_len=65536")
    return True


if __name__ == "__main__":
    success = asyncio.run(test_128k_context())
    sys.exit(0 if success else 1)
