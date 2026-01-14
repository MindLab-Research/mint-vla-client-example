#!/usr/bin/env python3
"""Test K2 inference only - assumes vLLM is already loaded."""

import asyncio
import os
import sys

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")


async def test_inference():
    """Test K2 inference using already-loaded vLLM."""
    from tinker import ServiceClient
    import tinker

    print("=" * 70)
    print("K2 Inference Test (assumes vLLM already loaded)")
    print("=" * 70)

    client = ServiceClient()

    # Step 1: Create training session
    print("\n[1] Creating training session...")
    try:
        training_client = await client.create_lora_training_client_async(
            base_model="moonshotai/Kimi-K2-Thinking",
            rank=16,
        )
        print(f"    Session: {training_client.model_id}")
    except Exception as e:
        print(f"    FAILED: {e}")
        return False

    # Step 2: Get sampling client (should be fast if vLLM already loaded)
    print("\n[2] Getting sampling client...")
    try:
        sampling_client = await training_client.save_weights_and_get_sampling_client_async()
        print("    Sampling client ready")
    except Exception as e:
        print(f"    FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 3: Test inference
    print("\n[3] Running inference...")
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "moonshotai/Kimi-K2-Thinking", trust_remote_code=True
        )
        prompt_text = "What is 2 + 2?"
        prompt = tinker.types.ModelInput.from_ints(tokenizer.encode(prompt_text))
        params = tinker.SamplingParams(max_tokens=2000, temperature=0.7)

        print(f"    Prompt: {prompt_text}")
        print("    Calling sample_async...")

        response = await sampling_client.sample_async(prompt, 1, params)

        generated_tokens = response.sequences[0].tokens
        generated_text = tokenizer.decode(generated_tokens)
        print(f"    Response: {generated_text[:200]}")
        print(f"    Token count: {len(generated_tokens)}")
    except Exception as e:
        print(f"    FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 70)
    print("K2 Inference Test: PASSED")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = asyncio.run(test_inference())
    sys.exit(0 if success else 1)
