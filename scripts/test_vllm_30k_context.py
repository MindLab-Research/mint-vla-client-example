#!/usr/bin/env python3
"""Test vLLM with 30K context length for K2-Thinking.

Verifies that vLLM can handle 30,720 tokens input (matching Megatron training max with TP=32).
"""

import os
import sys
import time
import requests

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
MODEL = "moonshotai/Kimi-K2-Thinking"
CONTEXT_LENGTH = 30720  # 30k context target


def poll_future(request_id: str, timeout: int = 600):
    """Poll for async operation completion."""
    poll_url = f"{BASE_URL}/api/v1/retrieve_future"
    start = time.time()
    last_print = 0
    while time.time() - start < timeout:
        resp = requests.post(poll_url, json={"request_id": request_id}, headers=HEADERS, timeout=120)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 408:
            elapsed = time.time() - start
            if elapsed - last_print > 30:
                print(f"  Waiting... {elapsed:.0f}s")
                last_print = elapsed
            time.sleep(2)
            continue
        else:
            return {"error": f"HTTP {resp.status_code}: {resp.text}"}
    return {"error": f"Timeout after {timeout}s"}


def create_sampling_session():
    """Create vLLM sampling session."""
    print(f"Creating sampling session for {MODEL}...")
    print("  (vLLM init takes ~10 minutes for K2-Thinking)")
    resp = requests.post(
        f"{BASE_URL}/api/v1/create_sampling_session",
        json={"base_model": MODEL, "session_id": f"vllm_30k_test_{int(time.time())}"},
        headers=HEADERS,
        timeout=900,  # 15 min timeout for initial request
    )
    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code}: {resp.text}"

    data = resp.json()
    if "request_id" in data:
        # Async initialization
        print(f"  Waiting for vLLM initialization (request_id={data['request_id']})...")
        result = poll_future(data["request_id"], timeout=900)
        if "error" in result:
            return None, result["error"]
        return result.get("sampling_session_id"), None
    else:
        return data.get("sampling_session_id"), None


def test_long_context(session_id: str, context_tokens: int, gen_tokens: int = 10):
    """Test vLLM with long context input."""
    print(f"\nTesting {context_tokens} context tokens + {gen_tokens} generation tokens...")

    # Create a long prompt using repeating tokens
    # Token 100 is a common token, safe to repeat
    prompt_tokens = [100] * context_tokens

    payload = {
        "prompt": {
            "chunks": [{"tokens": prompt_tokens, "type": "encoded_text"}]
        },
        "sampling_session_id": session_id,
        "sampling_params": {
            "max_tokens": gen_tokens,
            "temperature": 0.0,  # Deterministic for testing
        },
        "num_samples": 1,
    }

    start = time.time()
    resp = requests.post(
        f"{BASE_URL}/api/v1/asample",
        json=payload,
        headers=HEADERS,
        timeout=60,
    )

    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}: {resp.text}", "time": time.time() - start}

    request_id = resp.json().get("request_id")
    if not request_id:
        return {"error": "No request_id in response", "time": time.time() - start}

    print(f"  Request ID: {request_id}")
    result = poll_future(request_id, timeout=600)
    elapsed = time.time() - start

    if "error" in result:
        return {"error": result["error"], "time": elapsed}

    # Extract generated tokens
    sequences = result.get("sequences", [])
    if not sequences:
        return {"error": "No sequences in result", "time": elapsed}

    seq = sequences[0]
    output_tokens = len(seq.get("tokens", []))
    stop_reason = seq.get("stop_reason", "unknown")

    return {
        "success": True,
        "input_tokens": context_tokens,
        "output_tokens": output_tokens,
        "stop_reason": stop_reason,
        "time": elapsed,
    }


def main():
    print("=" * 60)
    print("vLLM 30K Context Test")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Target context: {CONTEXT_LENGTH} tokens")
    print()

    # Create session
    session_id, error = create_sampling_session()
    if error:
        print(f"Error creating session: {error}")
        return 1

    print(f"Session ID: {session_id}")

    # Test various context lengths up to 30K
    test_lengths = [8192, 15360, 20480, 25600, 30720]

    print("\n" + "-" * 60)
    print("Testing context lengths...")
    print("-" * 60)

    results = {}
    for ctx_len in test_lengths:
        result = test_long_context(session_id, ctx_len, gen_tokens=10)
        results[ctx_len] = result

        if "error" in result:
            print(f"  {ctx_len} tokens: FAIL ({result['time']:.1f}s)")
            print(f"    Error: {result['error'][:200]}")
        else:
            print(f"  {ctx_len} tokens: PASS - generated {result['output_tokens']} tokens ({result['time']:.1f}s)")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    max_working = 0
    for ctx_len, result in results.items():
        status = "PASS" if "success" in result else "FAIL"
        print(f"  {ctx_len:>6} tokens: {status}")
        if "success" in result:
            max_working = ctx_len

    print(f"\nMax working context: {max_working} tokens")

    if max_working >= CONTEXT_LENGTH:
        print(f"\nvLLM supports {CONTEXT_LENGTH} context length!")
        return 0
    else:
        print(f"\nvLLM max context ({max_working}) < target ({CONTEXT_LENGTH})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
