#!/usr/bin/env python3
"""Test MoE model with 260K prompt (near 262K limit) - OOM stress test."""

import os
import sys
import time
import uuid

import httpx
from transformers import AutoTokenizer

API_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8001")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")
MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
PROMPT_SIZE = 260000  # 260K tokens (near 262K limit)

def make_request(client, method, path, **kwargs):
    headers = kwargs.pop("headers", {})
    headers["X-API-Key"] = API_KEY
    return client.request(method, f"{API_URL}{path}", headers=headers, **kwargs)

def main():
    print(f"MoE 260K Prompt Stress Test")
    print(f"="*60)
    print(f"Model: {MODEL}")
    print(f"Context limit: 262,144 tokens")
    print(f"Test prompt: {PROMPT_SIZE:,} tokens")
    print(f"="*60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    # Generate 260K tokens
    print(f"Generating {PROMPT_SIZE:,} token prompt...")
    base = "The quick brown fox jumps over the lazy dog. "
    base_tokens = tokenizer.encode(base, add_special_tokens=False)
    prompt_tokens = (base_tokens * (PROMPT_SIZE // len(base_tokens) + 1))[:PROMPT_SIZE]
    print(f"Actual prompt: {len(prompt_tokens):,} tokens")

    with httpx.Client(timeout=1800.0) as client:  # 30 min timeout
        # Health check
        print("\nChecking server...")
        for _ in range(30):
            try:
                resp = make_request(client, "GET", "/api/v1/healthz")
                if resp.status_code == 200:
                    print("Server ready")
                    break
            except:
                pass
            time.sleep(2)
        else:
            print("Server not ready")
            return 1

        # Create session
        print("\nCreating session...")
        resp = make_request(client, "POST", "/api/v1/create_sampling_session",
            json={"session_id": str(uuid.uuid4()), "base_model": MODEL})

        if resp.status_code != 200:
            print(f"Session failed: {resp.status_code}")
            print(resp.text[:1000])
            return 1

        session_id = resp.json()["sampling_session_id"]
        print(f"Session: {session_id}")

        # Submit sample
        print(f"\nSubmitting {PROMPT_SIZE:,} token prompt...")
        print("(MoE init + long prompt processing may take 10+ minutes)")
        start_time = time.time()

        resp = make_request(client, "POST", "/api/v1/asample",
            json={
                "sampling_session_id": session_id,
                "num_samples": 1,
                "prompt": {"chunks": [{"type": "encoded_text", "tokens": prompt_tokens}]},
                "sampling_params": {"max_tokens": 50, "temperature": 0.7},
            })

        if resp.status_code != 200:
            elapsed = time.time() - start_time
            print(f"\nasample failed after {elapsed:.1f}s: {resp.status_code}")
            error_text = resp.text
            print(error_text[:2000])

            # Check for OOM indicators
            if "CUDA out of memory" in error_text or "OOM" in error_text:
                print("\n" + "="*60)
                print("RESULT: OOM - Model cannot handle 260K context with current GPU config")
                print("="*60)
            elif "exceeds model context limit" in error_text:
                print("\n" + "="*60)
                print("RESULT: Context limit exceeded (expected for prompts > 262K)")
                print("="*60)
            return 1

        request_id = resp.json()["request_id"]
        print(f"Request: {request_id}")

        # Poll for result
        print("\nPolling for result...")
        for attempt in range(600):  # 20 min polling
            try:
                resp = make_request(client, "POST", "/api/v1/retrieve_future",
                    json={"request_id": request_id})
            except Exception as e:
                print(f"Poll error: {e}")
                time.sleep(5)
                continue

            if resp.status_code == 200:
                result = resp.json()
                elapsed = time.time() - start_time

                if result.get("sequences"):
                    tokens = result["sequences"][0].get("tokens", [])
                    print(f"\n{'='*60}")
                    print(f"SUCCESS: Generated {len(tokens)} tokens in {elapsed:.1f}s")
                    print("="*60)
                    text = tokenizer.decode(tokens, skip_special_tokens=True)
                    print(f"Output: {text[:200]}...")
                    return 0

                elif result.get("error"):
                    error = result["error"]
                    print(f"\n{'='*60}")
                    print(f"FAILED after {elapsed:.1f}s:")
                    print("="*60)
                    print(error[:2000])

                    if "CUDA out of memory" in error or "OOM" in error:
                        print("\nRESULT: OOM during generation")
                    return 1

            elif resp.status_code == 408:
                if attempt % 60 == 0:
                    elapsed = time.time() - start_time
                    print(f"  {elapsed:.0f}s: still processing...")
                time.sleep(2)
            else:
                elapsed = time.time() - start_time
                print(f"\nPoll failed after {elapsed:.1f}s: {resp.status_code}")
                print(resp.text[:1000])
                return 1

        elapsed = time.time() - start_time
        print(f"\nTIMEOUT after {elapsed:.1f}s")
        return 1

if __name__ == "__main__":
    sys.exit(main())
