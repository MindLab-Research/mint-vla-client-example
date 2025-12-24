#!/usr/bin/env python3
"""Test MoE model context window for issue #9 fix."""

import os
import sys
import time
import uuid

import httpx
from transformers import AutoTokenizer

API_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8001")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")
MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"

def make_request(client, method, path, **kwargs):
    headers = kwargs.pop("headers", {})
    headers["X-API-Key"] = API_KEY
    return client.request(method, f"{API_URL}{path}", headers=headers, **kwargs)

def main():
    print(f"Testing MoE model: {MODEL}")
    print(f"API URL: {API_URL}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    # Generate 8K token prompt
    base = "The quick brown fox jumps over the lazy dog. "
    base_tokens = tokenizer.encode(base, add_special_tokens=False)
    prompt_tokens = (base_tokens * 1000)[:8000]
    print(f"Prompt: {len(prompt_tokens)} tokens")

    with httpx.Client(timeout=600.0) as client:
        # Create session
        print("\nCreating session...")
        resp = make_request(client, "POST", "/api/v1/create_sampling_session",
            json={"session_id": str(uuid.uuid4()), "base_model": MODEL})

        if resp.status_code != 200:
            print(f"FAILED: {resp.status_code} - {resp.text[:500]}")
            return 1

        session_id = resp.json()["sampling_session_id"]
        print(f"Session: {session_id}")

        # Submit sample
        print("\nSubmitting prompt (MoE init may take 5-10 min)...")
        resp = make_request(client, "POST", "/api/v1/asample",
            json={
                "sampling_session_id": session_id,
                "num_samples": 1,
                "prompt": {"chunks": [{"type": "encoded_text", "tokens": prompt_tokens}]},
                "sampling_params": {"max_tokens": 50, "temperature": 0.7},
            })

        if resp.status_code != 200:
            print(f"FAILED asample: {resp.status_code}")
            print(resp.text[:1000])
            return 1

        request_id = resp.json()["request_id"]
        print(f"Request: {request_id}")

        # Poll
        print("\nPolling (this will take time for MoE init)...")
        for attempt in range(180):  # 6 min
            resp = make_request(client, "POST", "/api/v1/retrieve_future",
                json={"request_id": request_id})

            if resp.status_code == 200:
                result = resp.json()
                if result.get("sequences"):
                    tokens = result["sequences"][0].get("tokens", [])
                    print(f"\nSUCCESS: Generated {len(tokens)} tokens")
                    text = tokenizer.decode(tokens, skip_special_tokens=True)
                    print(f"Output: {text[:200]}...")
                    return 0
                elif result.get("error"):
                    print(f"\nFAILED: {result['error'][:500]}")
                    return 1
            elif resp.status_code == 408:
                if attempt % 30 == 0:
                    print(f"  {attempt*2}s: processing...")
                time.sleep(2)
            else:
                print(f"\nFAILED poll: {resp.status_code} - {resp.text[:500]}")
                return 1

        print("\nTIMEOUT")
        return 1

if __name__ == "__main__":
    sys.exit(main())
