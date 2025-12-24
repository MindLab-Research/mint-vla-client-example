#!/usr/bin/env python3
"""Reproduce issue #9: vLLM inference fails when prompt exceeds max_model_len.

The issue: vLLM's max_model_len defaults to 4096, but models like Qwen2.5-7B-Instruct
support 32K context. When prompt length > 4096, vLLM returns:
  ValueError: max_tokens must be at least 1, got -5904

Usage:
    # Against production
    python scripts/reproduce_issue_9.py

    # Against dev (bugfix server)
    TINKER_BASE_URL=http://localhost:8001 TINKER_API_KEY=dummy python scripts/reproduce_issue_9.py
"""

import os
import sys
import time
import uuid

try:
    import httpx
except ImportError:
    print("httpx not installed. Run: pip install httpx")
    sys.exit(1)

try:
    from transformers import AutoTokenizer
except ImportError:
    print("transformers not installed. Run: pip install transformers")
    sys.exit(1)

# Production config (overridable via env vars)
API_URL = os.environ.get("TINKER_BASE_URL", "https://mint-alpha.macaron.im")
API_KEY = os.environ.get("TINKER_API_KEY", "sk-mint-vr7P59S96QCRV1qU1wcu0cssk4bNDVPaAIdwfyM0sbg")

# Test model - Qwen2.5-7B-Instruct has 32K context but vLLM defaults to 4096
MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Generate ~5000 tokens of prompt (should work with 32K context, fails with 4096 limit)
LONG_PROMPT_TEXT = ("The quick brown fox jumps over the lazy dog. " * 500) + "\n\nPlease summarize:"


def make_request(client: httpx.Client, method: str, path: str, **kwargs):
    """Make API request with auth header."""
    headers = kwargs.pop("headers", {})
    headers["X-API-Key"] = API_KEY

    url = f"{API_URL}{path}"
    return client.request(method, url, headers=headers, **kwargs)


def test_long_prompt() -> bool:
    """Test inference with a prompt that exceeds vLLM's default 4096 max_model_len."""
    print(f"\n{'='*60}")
    print("Issue #9: vLLM max_model_len test")
    print(f"{'='*60}")
    print(f"API URL: {API_URL}")
    print(f"Model: {MODEL}")

    # Tokenize the prompt
    print("\nTokenizing prompt...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompt_tokens = tokenizer.encode(LONG_PROMPT_TEXT, add_special_tokens=False)
    num_tokens = len(prompt_tokens)
    print(f"Prompt: {num_tokens} tokens")
    print(f"  (vLLM default max_model_len=4096, model supports 32K)")
    print(f"  (This prompt should {'FAIL with 4096 limit' if num_tokens > 4096 else 'work'})")

    with httpx.Client(timeout=300.0) as client:
        # Step 1: Create sampling session
        print("\n[1] Creating sampling session...")
        resp = make_request(
            client, "POST", "/api/v1/create_sampling_session",
            json={
                "session_id": str(uuid.uuid4()),  # Required by API but ignored
                "base_model": MODEL,
            }
        )

        if resp.status_code != 200:
            print(f"FAILED: Could not create session: {resp.status_code}")
            print(f"Response: {resp.text}")
            return False

        data = resp.json()
        session_id = data.get("sampling_session_id")  # Use server-generated ID
        print(f"Session created: {session_id}")

        # Step 2: Submit async sample with long prompt
        print("\n[2] Submitting long prompt for sampling...")
        resp = make_request(
            client, "POST", "/api/v1/asample",
            json={
                "sampling_session_id": session_id,
                "num_samples": 1,
                "prompt": {
                    "chunks": [{"type": "encoded_text", "tokens": prompt_tokens}]
                },
                "sampling_params": {
                    "max_tokens": 50,
                    "temperature": 0.7,
                },
            }
        )

        if resp.status_code != 200:
            print(f"FAILED: asample request failed: {resp.status_code}")
            error_text = resp.text[:500]  # Truncate
            print(f"Response: {error_text}")
            # Check for the specific error
            if "max_tokens must be at least 1" in resp.text or "max_model_len" in resp.text.lower():
                print("\n" + "="*60)
                print("ISSUE #9 REPRODUCED: vLLM rejected long prompt")
                print("Error indicates prompt exceeds vLLM's max_model_len")
                print("="*60)
            return False

        data = resp.json()
        request_id = data.get("request_id")
        print(f"Request ID: {request_id}")

        # Step 3: Poll for result
        print("\n[3] Polling for result...")
        max_attempts = 60  # 2 minutes
        for attempt in range(max_attempts):
            resp = make_request(
                client, "POST", "/api/v1/retrieve_future",
                json={"request_id": request_id}
            )

            if resp.status_code == 200:
                result = resp.json()
                sequences = result.get("sequences", [])
                if sequences:
                    first_seq = sequences[0]
                    tokens = first_seq.get("tokens", [])
                    print(f"\nGeneration completed: {len(tokens)} tokens")
                    text = tokenizer.decode(tokens, skip_special_tokens=True)
                    print(f"Output: {text[:200]}...")
                    return True
                else:
                    print(f"Unexpected response: {result}")
                    return False
            elif resp.status_code == 408:
                # Still processing
                if attempt % 10 == 0:
                    print(f"    Attempt {attempt + 1}/{max_attempts}: processing...")
                time.sleep(2)
            else:
                print(f"FAILED: retrieve_future returned {resp.status_code}")
                error_text = resp.text[:500]
                print(f"Response: {error_text}")
                # Check for the specific error
                if "max_tokens must be at least 1" in resp.text or "max_model_len" in resp.text.lower():
                    print("\n" + "="*60)
                    print("ISSUE #9 REPRODUCED: vLLM rejected long prompt during generation")
                    print("="*60)
                return False

        print("TIMEOUT: Result not ready after polling")
        return False


def main():
    print("="*60)
    print("Issue #9 Reproduction Script")
    print("vLLM max_model_len should use model's max_position_embeddings")
    print("="*60)

    success = test_long_prompt()

    print("\n" + "="*60)
    if success:
        print("TEST PASSED: Long prompt inference works correctly")
        print("Issue #9 is FIXED (or was not present)")
    else:
        print("TEST FAILED: Long prompt inference failed")
        print("Issue #9 needs to be fixed")
    print("="*60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
