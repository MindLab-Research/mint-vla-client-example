#!/usr/bin/env python
"""Test multi-session concurrent requests to tinker-server.

Creates multiple sampling sessions and sends requests simultaneously
to verify the server handles concurrent sessions correctly.

Usage:
    export TINKER_BASE_URL=http://localhost:8000
    python scripts/test_multi_session.py
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add tinker to path if not installed
tinker_path = os.path.join(os.path.dirname(__file__), "../../tinker/src")
if os.path.exists(tinker_path):
    sys.path.insert(0, tinker_path)

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")

from tinker import types
from tinker.lib.public_interfaces.service_client import ServiceClient


# Different prompts for each session
PROMPTS = [
    "What is the capital of France?",
    "What is 2 + 2?",
    "Name a color.",
    "What is the largest planet?",
    "Say hello.",
]


def create_session_and_sample(session_idx: int, tokenizer, prompt_text: str) -> dict:
    """Create a sampling session and send a request."""
    start = time.time()

    # Create independent ServiceClient and SamplingClient
    client = ServiceClient()
    sampling_client = client.create_sampling_client(
        base_model="Qwen/Qwen2.5-7B-Instruct"
    )
    session_id = sampling_client._sampling_session_id
    setup_time = time.time() - start

    # Prepare prompt with chat template
    messages = [{"role": "user", "content": prompt_text}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    prompt = types.ModelInput.from_ints(prompt_tokens)

    params = types.SamplingParams(max_tokens=32, temperature=0.7, top_p=0.9)

    # Send request
    sample_start = time.time()
    future = sampling_client.sample(prompt=prompt, num_samples=1, sampling_params=params)
    result = future.result(timeout=120)
    sample_time = time.time() - sample_start

    # Decode response
    response_text = tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)

    return {
        "session_idx": session_idx,
        "session_id": session_id[:8],
        "prompt": prompt_text,
        "response": response_text[:50],
        "tokens": len(result.sequences[0].tokens),
        "stop_reason": result.sequences[0].stop_reason,
        "setup_time": setup_time,
        "sample_time": sample_time,
    }


def main():
    num_sessions = int(os.environ.get("NUM_SESSIONS", "5"))
    print(f"Testing {num_sessions} concurrent sessions")
    print(f"Server: {os.environ.get('TINKER_BASE_URL')}")
    print("=" * 70)

    # Load tokenizer once
    print("Loading tokenizer...")
    from transformers import AutoTokenizer
    model_path = os.environ.get("TINKER_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Use prompts cyclically if num_sessions > len(PROMPTS)
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(num_sessions)]

    # Run all sessions concurrently
    print(f"\nStarting {num_sessions} concurrent requests...")
    start_time = time.time()

    results = []
    errors = []

    with ThreadPoolExecutor(max_workers=num_sessions) as executor:
        futures = {
            executor.submit(create_session_and_sample, i, tokenizer, prompts[i]): i
            for i in range(num_sessions)
        }

        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"  Session {idx}: completed ({result['tokens']} tokens)")
            except Exception as e:
                errors.append((idx, str(e)))
                print(f"  Session {idx}: FAILED - {e}")

    total_time = time.time() - start_time

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for r in sorted(results, key=lambda x: x["session_idx"]):
        print(f"\nSession {r['session_idx']} (id: {r['session_id']}...):")
        print(f"  Prompt: {r['prompt']}")
        print(f"  Response: {r['response']}...")
        print(f"  Tokens: {r['tokens']}, Stop: {r['stop_reason']}")
        print(f"  Setup: {r['setup_time']:.2f}s, Sample: {r['sample_time']:.2f}s")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total sessions: {num_sessions}")
    print(f"Successful: {len(results)}")
    print(f"Failed: {len(errors)}")
    print(f"Total time: {total_time:.2f}s")

    if results:
        avg_sample = sum(r["sample_time"] for r in results) / len(results)
        print(f"Avg sample time: {avg_sample:.2f}s")
        print(f"Throughput: {len(results) / total_time:.2f} requests/s")

    if errors:
        print("\nErrors:")
        for idx, err in errors:
            print(f"  Session {idx}: {err}")
        sys.exit(1)
    else:
        print("\nAll sessions completed successfully!")


if __name__ == "__main__":
    main()
