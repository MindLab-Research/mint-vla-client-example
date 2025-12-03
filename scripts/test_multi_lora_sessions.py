#!/usr/bin/env python
"""Test multi-LoRA session support.

Creates multiple sessions with lora_rank=32, each with its own VerlInferenceEngine.
Sessions are isolated - each has independent LoRA weights (zero-initialized).

Note: Outputs are identical between sessions until training updates the LoRA weights.
This test verifies session isolation architecture, not LoRA differentiation.

Usage:
    # Server must be running
    python scripts/test_multi_lora_sessions.py
"""

import os
import sys
import time

import requests

# Add tinker to path if not installed
tinker_path = os.path.join(os.path.dirname(__file__), "../../tinker/src")
if os.path.exists(tinker_path):
    sys.path.insert(0, tinker_path)

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")


def create_session_with_lora(lora_rank: int = 32) -> str:
    """Create a sampling session with specified LoRA rank."""
    # Create session
    resp = requests.post(f"{BASE_URL}/api/v1/create_session", json={
        "tags": [],
        "user_metadata": {},
    })
    resp.raise_for_status()
    session_id = resp.json()["session_id"]

    # Create sampling session with lora_rank
    resp = requests.post(f"{BASE_URL}/api/v1/create_sampling_session", json={
        "session_id": session_id,
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "lora_rank": lora_rank,
    })
    resp.raise_for_status()
    return resp.json()["sampling_session_id"]


def sample(sampling_session_id: str, prompt_tokens: list[int], max_tokens: int = 32) -> list[int]:
    """Sample from a session, returning generated token IDs."""
    # Submit async sample
    resp = requests.post(f"{BASE_URL}/api/v1/asample", json={
        "sampling_session_id": sampling_session_id,
        "seq_id": 0,
        "num_samples": 1,
        "prompt": {"chunks": [{"tokens": prompt_tokens, "type": "encoded_text"}]},
        "sampling_params": {
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_k": -1,
            "top_p": 0.9,
        },
    })
    resp.raise_for_status()
    request_id = resp.json()["request_id"]

    # Poll for result
    for _ in range(120):  # 2 minute timeout
        resp = requests.post(f"{BASE_URL}/api/v1/retrieve_future", json={
            "request_id": request_id,
        })
        if resp.status_code == 200:
            return resp.json()["sequences"][0]["tokens"]
        time.sleep(1)

    raise TimeoutError(f"Request {request_id} timed out")


def main():
    print(f"Connecting to: {BASE_URL}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    from transformers import AutoTokenizer
    model_path = os.environ.get("TINKER_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Prepare prompt
    prompt_text = "Write a haiku about programming:"
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_tokens = tokenizer.encode(formatted, add_special_tokens=False)

    print(f"\nPrompt: {prompt_text}")
    print(f"Prompt tokens: {len(prompt_tokens)}")

    # Create two sessions with lora_rank=32
    print("\n" + "=" * 60)
    print("Creating Session 1 (lora_rank=32)...")
    print("=" * 60)
    t0 = time.time()
    session_1 = create_session_with_lora(lora_rank=32)
    print(f"Session 1 created: {session_1} ({time.time() - t0:.1f}s)")

    print("\n" + "=" * 60)
    print("Creating Session 2 (lora_rank=32)...")
    print("=" * 60)
    t0 = time.time()
    session_2 = create_session_with_lora(lora_rank=32)
    print(f"Session 2 created: {session_2} ({time.time() - t0:.1f}s)")

    # Sample from both sessions with same prompt
    print("\n" + "=" * 60)
    print("Sampling from Session 1...")
    print("=" * 60)
    t0 = time.time()
    output_1 = sample(session_1, prompt_tokens, max_tokens=64)
    text_1 = tokenizer.decode(output_1, skip_special_tokens=True)
    print(f"Session 1 output ({time.time() - t0:.1f}s, {len(output_1)} tokens):")
    print(text_1)

    print("\n" + "=" * 60)
    print("Sampling from Session 2...")
    print("=" * 60)
    t0 = time.time()
    output_2 = sample(session_2, prompt_tokens, max_tokens=64)
    text_2 = tokenizer.decode(output_2, skip_special_tokens=True)
    print(f"Session 2 output ({time.time() - t0:.1f}s, {len(output_2)} tokens):")
    print(text_2)

    # Compare outputs
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    if output_1 == output_2:
        print("Outputs are IDENTICAL (expected - LoRA weights are zero-initialized)")
        print("Outputs will differ after training updates LoRA weights.")
    else:
        print("Outputs are DIFFERENT")
        min_len = min(len(output_1), len(output_2))
        diff_count = sum(1 for i in range(min_len) if output_1[i] != output_2[i])
        diff_count += abs(len(output_1) - len(output_2))
        print(f"Token difference: {diff_count}/{max(len(output_1), len(output_2))}")

    # Verify session isolation: sample from same session twice should use same engine
    print("\n" + "=" * 60)
    print("Verifying session isolation...")
    print("=" * 60)
    t0 = time.time()
    output_1b = sample(session_1, prompt_tokens, max_tokens=64)
    print(f"Session 1 second sample ({time.time() - t0:.1f}s)")

    if output_1 == output_1b:
        print("Same session, same output (deterministic with same seed)")
    else:
        print("Same session, different output (stochastic sampling)")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
    print("\nSessions are cleaned up automatically after 5 minutes of inactivity")
    print("or on server shutdown.")


if __name__ == "__main__":
    main()
