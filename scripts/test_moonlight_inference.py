#!/usr/bin/env python3
"""Test Moonlight-16B-A3B inference with vLLM.

Moonlight uses the same DeepseekV3ForCausalLM architecture as Kimi-K2.
This test validates the DeepseekV3 code path with a smaller model.

Moonlight specs:
- 64 experts, 27 layers, hidden_size=2048
- Inference: TP=2 (2 GPUs)
- Training: TP=2, EP=8 (16 GPUs)
"""

import os
import sys
import time
import uuid

import requests

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")

MOONLIGHT_MODEL = "moonshotai/Moonlight-16B-A3B-Instruct"


def get_headers():
    return {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def poll_future(request_id: str, timeout: int = 600) -> dict:
    """Poll for async operation result."""
    poll_url = f"{BASE_URL}/api/v1/retrieve_future"
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.post(poll_url, json={"request_id": request_id}, headers=get_headers(), timeout=120)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 408:
            time.sleep(1)
            continue
        else:
            return {"error": f"HTTP {resp.status_code}: {resp.text}"}
    return {"error": f"Timeout after {timeout}s"}


def create_sampling_session(base_model: str) -> str:
    """Create a sampling session for inference-only."""
    session_id = f"moonlight_inf_{uuid.uuid4().hex[:8]}"
    url = f"{BASE_URL}/api/v1/create_sampling_session"
    payload = {
        "session_id": session_id,
        "base_model": base_model,
    }
    print(f"Creating sampling session: {session_id}")
    print(f"Base model: {base_model}")

    resp = requests.post(url, json=payload, headers=get_headers(), timeout=300)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

    result = resp.json()
    sampling_session_id = result.get("sampling_session_id")
    print(f"Sampling session created: {sampling_session_id}")
    return sampling_session_id


def sample(sampling_session_id: str, prompt_tokens: list, max_tokens: int = 50, temperature: float = 0.0) -> dict:
    """Sample from model."""
    url = f"{BASE_URL}/api/v1/asample"
    payload = {
        "sampling_session_id": sampling_session_id,
        "prompt": {"chunks": [{"tokens": prompt_tokens, "type": "encoded_text"}]},
        "sampling_params": {"max_tokens": max_tokens, "temperature": temperature},
        "num_samples": 1,
    }

    resp = requests.post(url, json=payload, headers=get_headers(), timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

    request_id = resp.json().get("request_id")
    return poll_future(request_id, timeout=300)


def main():
    print("=" * 70)
    print("Moonlight-16B-A3B Inference Test (DeepseekV3 Architecture)")
    print("=" * 70)
    print(f"Model: {MOONLIGHT_MODEL}")
    print(f"Config: TP=2, DP=1 (2 GPUs)")
    print("Purpose: Validate K2 code path with smaller model")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MOONLIGHT_MODEL,
        trust_remote_code=True,
    )
    print(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    # Check vLLM status first
    print("\nChecking vLLM status...")
    try:
        resp = requests.get(f"{BASE_URL}/api/v1/vllm_status", headers=get_headers(), timeout=30)
        status = resp.json()
        print(f"vLLM status: {status}")
    except Exception as e:
        print(f"Warning: Could not check vLLM status: {e}")

    # Step 1: Create sampling session
    print("\n" + "-" * 40)
    print("Step 1: Creating Moonlight sampling session...")
    print("This will initialize vLLM with Moonlight (may take 2-5 minutes)...")
    t0 = time.time()
    try:
        sampling_session_id = create_sampling_session(MOONLIGHT_MODEL)
        init_time = time.time() - t0
        print(f"Session created in {init_time:.1f}s")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 2: Sample
    print("\n" + "-" * 40)
    print("Step 2: Sampling from Moonlight...")
    prompt = "Q: What is 2+2?\nA:"
    print(f"Prompt: {prompt!r}")
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    print(f"Prompt tokens: {prompt_tokens}")

    t0 = time.time()
    try:
        result = sample(sampling_session_id, prompt_tokens, max_tokens=20, temperature=0.0)
        sample_time = time.time() - t0

        if "error" in result:
            print(f"FAILED: {result['error']}")
            return 1

        # Decode output
        sequences = result.get("sequences", [])
        if sequences:
            tokens = sequences[0].get("tokens", [])
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            print(f"Output tokens: {tokens[:20]}...")
            print(f"Output text: {text!r}")
        else:
            print("No sequences returned")

        print(f"Sample completed in {sample_time:.1f}s")
        sample_ok = True
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        sample_ok = False
        sample_time = 0

    # Summary
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"vLLM Init: {init_time:.1f}s")
    print(f"Sampling: {'OK' if sample_ok else 'FAILED'} ({sample_time:.1f}s)")

    if sample_ok:
        print("\nMoonlight (DeepseekV3) inference: WORKING")
        print("This validates the K2 code path works for DeepseekV3ForCausalLM architecture")
        return 0
    else:
        print("\nMoonlight inference: FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
