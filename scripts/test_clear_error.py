#!/usr/bin/env python3
"""Test that exceeding context limit gives clear error."""

import os
import sys
import time
import uuid

import httpx
from transformers import AutoTokenizer

API_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8001")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")
MODEL = "Qwen/Qwen2.5-7B-Instruct"

def make_request(client, method, path, **kwargs):
    headers = kwargs.pop("headers", {})
    headers["X-API-Key"] = API_KEY
    return client.request(method, f"{API_URL}{path}", headers=headers, **kwargs)

def main():
    print(f"Testing exceed context limit error message")
    print(f"Model: {MODEL} (32K context)")

    # Load tokenizer and generate 35K token prompt (exceeds 32K)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    base = "Test token. "
    base_tokens = tokenizer.encode(base, add_special_tokens=False)
    prompt_tokens = (base_tokens * 20000)[:35000]
    print(f"Prompt: {len(prompt_tokens):,} tokens (exceeds 32K limit)")

    with httpx.Client(timeout=300.0) as client:
        # Create session
        resp = make_request(client, "POST", "/api/v1/create_sampling_session",
            json={"session_id": str(uuid.uuid4()), "base_model": MODEL})
        if resp.status_code != 200:
            print(f"Session failed: {resp.text[:500]}")
            return 1
        session_id = resp.json()["sampling_session_id"]

        # Submit sample
        resp = make_request(client, "POST", "/api/v1/asample",
            json={
                "sampling_session_id": session_id,
                "num_samples": 1,
                "prompt": {"chunks": [{"type": "encoded_text", "tokens": prompt_tokens}]},
                "sampling_params": {"max_tokens": 50},
            })
        if resp.status_code != 200:
            print(f"asample failed: {resp.text[:500]}")
            return 1
        request_id = resp.json()["request_id"]

        # Poll for result
        for _ in range(30):
            resp = make_request(client, "POST", "/api/v1/retrieve_future",
                json={"request_id": request_id})
            if resp.status_code == 200:
                result = resp.json()
                if result.get("error"):
                    error = result["error"]
                    print(f"\n{'='*60}")
                    print("ERROR MESSAGE:")
                    print("="*60)
                    print(error)
                    print("="*60)

                    # Check if it's clear
                    if "Prompt length" in error and "exceeds model context limit" in error:
                        print("\nVERDICT: Clear error message")
                        return 0
                    else:
                        print("\nVERDICT: Error message NOT clear")
                        return 1
                elif result.get("sequences"):
                    print("ERROR: Expected failure but got success!")
                    return 1
            elif resp.status_code == 408:
                time.sleep(2)
            else:
                print(f"Poll error: {resp.status_code} - {resp.text[:500]}")
                return 1
        print("TIMEOUT")
        return 1

if __name__ == "__main__":
    sys.exit(main())
