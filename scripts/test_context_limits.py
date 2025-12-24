#!/usr/bin/env python3
"""Test context window limits for issue #9 fix.

Tests:
1. Prompt nearly filling context window (should work)
2. Prompt exceeding context window (should give clear error)
3. Dense model (Qwen2.5-7B-Instruct, 32K context)
4. MoE model (Qwen3-30B-A3B-Instruct-2507, 262K context)

Usage:
    TINKER_BASE_URL=http://localhost:8001 TINKER_API_KEY=dummy python scripts/test_context_limits.py
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

API_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8001")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")

# Test configurations
TESTS = [
    {
        "name": "Dense model - near limit",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "max_context": 32768,
        "prompt_tokens": 30000,  # Near 32K limit
        "max_new_tokens": 100,
        "should_succeed": True,
    },
    {
        "name": "Dense model - exceed limit",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "max_context": 32768,
        "prompt_tokens": 35000,  # Exceeds 32K limit
        "max_new_tokens": 100,
        "should_succeed": False,
    },
    {
        "name": "MoE model - moderate prompt",
        "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "max_context": 262144,
        "prompt_tokens": 8000,  # Moderate for MoE
        "max_new_tokens": 100,
        "should_succeed": True,
    },
]


def make_request(client: httpx.Client, method: str, path: str, **kwargs):
    """Make API request with auth header."""
    headers = kwargs.pop("headers", {})
    headers["X-API-Key"] = API_KEY
    url = f"{API_URL}{path}"
    return client.request(method, url, headers=headers, **kwargs)


def generate_prompt_tokens(tokenizer, target_length: int) -> list[int]:
    """Generate a prompt with approximately target_length tokens."""
    # Use a repeating pattern
    base_text = "The quick brown fox jumps over the lazy dog. "
    base_tokens = tokenizer.encode(base_text, add_special_tokens=False)

    # Calculate how many repetitions needed
    reps_needed = (target_length // len(base_tokens)) + 1
    full_tokens = (base_tokens * reps_needed)[:target_length]

    return full_tokens


def run_test(test_config: dict) -> tuple[bool, str]:
    """Run a single test and return (passed, message)."""
    name = test_config["name"]
    model = test_config["model"]
    prompt_tokens = test_config["prompt_tokens"]
    max_new_tokens = test_config["max_new_tokens"]
    should_succeed = test_config["should_succeed"]

    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Prompt tokens: {prompt_tokens:,}")
    print(f"Max context: {test_config['max_context']:,}")
    print(f"Expected: {'SUCCESS' if should_succeed else 'CLEAR ERROR'}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    except Exception as e:
        return False, f"Failed to load tokenizer: {e}"

    # Generate prompt
    print(f"Generating {prompt_tokens:,} token prompt...")
    tokens = generate_prompt_tokens(tokenizer, prompt_tokens)
    actual_len = len(tokens)
    print(f"Actual prompt length: {actual_len:,} tokens")

    with httpx.Client(timeout=600.0) as client:
        # Create session
        print("\nCreating sampling session...")
        try:
            resp = make_request(
                client, "POST", "/api/v1/create_sampling_session",
                json={"session_id": str(uuid.uuid4()), "base_model": model}
            )
        except Exception as e:
            return False, f"Connection error: {e}"

        if resp.status_code != 200:
            error_text = resp.text[:500]
            if not should_succeed:
                # Check if error is clear
                if "context" in error_text.lower() or "token" in error_text.lower() or "length" in error_text.lower():
                    return True, f"Got clear error as expected: {error_text[:200]}"
                return False, f"Got error but not clear: {error_text}"
            return False, f"Failed to create session: {resp.status_code} - {error_text}"

        session_id = resp.json().get("sampling_session_id")
        print(f"Session: {session_id}")

        # Submit sample
        print("\nSubmitting prompt...")
        try:
            resp = make_request(
                client, "POST", "/api/v1/asample",
                json={
                    "sampling_session_id": session_id,
                    "num_samples": 1,
                    "prompt": {"chunks": [{"type": "encoded_text", "tokens": tokens}]},
                    "sampling_params": {"max_tokens": max_new_tokens, "temperature": 0.7},
                }
            )
        except Exception as e:
            return False, f"Request error: {e}"

        if resp.status_code != 200:
            error_text = resp.text[:500]
            if not should_succeed:
                # Check if error message is clear about context/length
                error_lower = error_text.lower()
                if any(kw in error_lower for kw in ["context", "token", "length", "exceed", "limit", "too long"]):
                    return True, f"Got clear error as expected: {error_text[:200]}"
                return False, f"Got error but message unclear: {error_text}"
            return False, f"asample failed: {resp.status_code} - {error_text}"

        request_id = resp.json().get("request_id")
        print(f"Request ID: {request_id}")

        # Poll for result
        print("Polling for result...")
        max_attempts = 120  # 4 minutes for MoE init
        for attempt in range(max_attempts):
            try:
                resp = make_request(
                    client, "POST", "/api/v1/retrieve_future",
                    json={"request_id": request_id}
                )
            except Exception as e:
                return False, f"Poll error: {e}"

            if resp.status_code == 200:
                result = resp.json()
                sequences = result.get("sequences", [])
                if sequences:
                    output_tokens = len(sequences[0].get("tokens", []))
                    if should_succeed:
                        return True, f"Generated {output_tokens} tokens"
                    else:
                        return False, f"Expected failure but succeeded with {output_tokens} tokens"
                else:
                    error = result.get("error", "Unknown")
                    if not should_succeed:
                        if any(kw in str(error).lower() for kw in ["context", "token", "length", "exceed"]):
                            return True, f"Got clear error: {error}"
                    return False, f"Empty result: {result}"
            elif resp.status_code == 408:
                if attempt % 20 == 0:
                    print(f"  Attempt {attempt+1}/{max_attempts}: processing...")
                time.sleep(2)
            else:
                error_text = resp.text[:500]
                if not should_succeed:
                    error_lower = error_text.lower()
                    if any(kw in error_lower for kw in ["context", "token", "length", "exceed", "limit"]):
                        return True, f"Got clear error: {error_text[:200]}"
                return False, f"Poll failed: {resp.status_code} - {error_text}"

        return False, "Timeout waiting for result"


def main():
    print("="*60)
    print("Context Window Limit Tests")
    print("="*60)
    print(f"API URL: {API_URL}")

    # Check server health
    print("\nChecking server health...")
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = make_request(client, "GET", "/api/v1/healthz")
            if resp.status_code != 200:
                print(f"Server not healthy: {resp.status_code}")
                return 1
            print("Server is healthy")
    except Exception as e:
        print(f"Cannot connect to server: {e}")
        return 1

    results = []
    for test in TESTS:
        passed, message = run_test(test)
        results.append((test["name"], passed, message))
        status = "PASS" if passed else "FAIL"
        print(f"\nResult: {status} - {message}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_passed = True
    for name, passed, message in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            print(f"         {message}")
            all_passed = False

    print("="*60)
    if all_passed:
        print("All tests passed")
    else:
        print("Some tests failed")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
