#!/usr/bin/env python3
"""Test vLLM long sequence generation with unlimited polling.

This script submits a long generation request and polls until completion,
with no timeout limit, to verify K2 can generate long sequences.
"""

import argparse
import json
import time
import requests

BASE_URL = "http://localhost:8000"


def create_sampling_session(base_model: str) -> str:
    """Create a sampling session and return session_id."""
    resp = requests.post(
        f"{BASE_URL}/api/v1/create_sampling_session",
        json={"base_model": base_model, "session_id": f"long_gen_{int(time.time())}"},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["sampling_session_id"]


def submit_sample_request(prompt_tokens: list[int], max_tokens: int, session_id: str) -> str:
    """Submit async sample request, return request_id."""
    payload = {
        "prompt": {
            "chunks": [{"tokens": prompt_tokens, "type": "encoded_text"}]
        },
        "sampling_session_id": session_id,
        "sampling_params": {
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
        },
        "num_samples": 1,
    }

    resp = requests.post(
        f"{BASE_URL}/api/v1/asample",
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["request_id"]


def poll_forever(request_id: str, poll_interval: float = 2.0) -> dict:
    """Poll for result with no timeout - runs until completion."""
    start_time = time.time()
    poll_count = 0

    while True:
        poll_count += 1
        elapsed = time.time() - start_time

        try:
            resp = requests.post(
                f"{BASE_URL}/api/v1/retrieve_future",
                json={"request_id": request_id},
                timeout=30,
            )
        except requests.exceptions.Timeout:
            print(f"[{elapsed:.0f}s] Poll {poll_count}: HTTP timeout, retrying...")
            time.sleep(poll_interval)
            continue

        if resp.status_code == 200:
            result = resp.json()
            if "error" in result:
                print(f"[{elapsed:.0f}s] Poll {poll_count}: Error - {result['error']}")
                return result
            print(f"[{elapsed:.0f}s] Poll {poll_count}: COMPLETE!")
            return result
        elif resp.status_code == 408:
            # Still pending - normal behavior
            if poll_count % 10 == 1:  # Log every 10th poll
                print(f"[{elapsed:.0f}s] Poll {poll_count}: Still processing...")
            time.sleep(poll_interval)
        else:
            print(f"[{elapsed:.0f}s] Poll {poll_count}: Unexpected status {resp.status_code}")
            return {"error": f"HTTP {resp.status_code}: {resp.text}"}


def tokenize_prompt(prompt: str, model_name: str = "moonshotai/Kimi-K2-Thinking") -> list[int]:
    """Tokenize prompt using transformers."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return tokenizer.encode(prompt)


def main():
    parser = argparse.ArgumentParser(description="Test vLLM long generation")
    parser.add_argument("--max-tokens", type=int, default=1000, help="Max tokens to generate")
    parser.add_argument("--model", type=str, default="moonshotai/Kimi-K2-Thinking", help="Model ID")
    parser.add_argument("--prompt", type=str, default="Write a detailed essay about the history of mathematics, covering ancient civilizations through modern developments.", help="Prompt")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Seconds between polls")
    parser.add_argument("--prompt-tokens", type=str, default="", help="Comma-separated token IDs (bypasses tokenization)")
    args = parser.parse_args()

    print("=" * 60)
    print("vLLM Long Generation Test (No Timeout)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Max tokens: {args.max_tokens}")

    # Get prompt tokens
    if args.prompt_tokens:
        prompt_tokens = [int(t) for t in args.prompt_tokens.split(",")]
        print(f"Using provided tokens: {len(prompt_tokens)} tokens")
    else:
        print(f"Prompt: {args.prompt[:50]}...")
        print("Tokenizing prompt...")
        prompt_tokens = tokenize_prompt(args.prompt, args.model)
        print(f"Tokenized to {len(prompt_tokens)} tokens")
    print()

    # Create sampling session first
    print("Creating sampling session...")
    t0 = time.time()
    session_id = create_sampling_session(args.model)
    print(f"Session ID: {session_id}")

    # Submit request
    print("Submitting request...")
    request_id = submit_sample_request(prompt_tokens, args.max_tokens, session_id)
    print(f"Request ID: {request_id}")
    print()

    # Poll until done
    print("Polling for result (no timeout)...")
    result = poll_forever(request_id, args.poll_interval)

    total_time = time.time() - t0
    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)

    if "error" in result:
        print(f"ERROR: {result['error']}")
    elif "sequences" in result:
        sequences = result["sequences"]
        print(f"Number of sequences: {len(sequences)}")
        for i, seq in enumerate(sequences):
            tokens = seq.get("tokens", [])
            text = seq.get("text", "")
            print(f"\nSequence {i+1}:")
            print(f"  Tokens: {len(tokens)}")
            print(f"  Stop reason: {seq.get('stop_reason', 'N/A')}")
            if text:
                preview = text[:500] + "..." if len(text) > 500 else text
                print(f"  Text preview: {preview}")
    else:
        print(f"Unknown result format: {json.dumps(result, indent=2)[:500]}")

    print()
    print(f"Total time: {total_time:.1f}s")
    if "sequences" in result and result["sequences"]:
        tokens = len(result["sequences"][0].get("tokens", []))
        if tokens > 0:
            print(f"Generation speed: {tokens / total_time:.2f} tokens/s")


if __name__ == "__main__":
    main()
