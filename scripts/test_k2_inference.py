#!/usr/bin/env python3
"""Test Kimi K2 inference with vLLM (TP=16 multi-node for LoRA support).

K2 inference with LoRA requires TP=16 = 16 GPUs across 2 nodes.
TP=8 OOMs when allocating LoRA buffers for 384 experts.
LoRA is REQUIRED for weight transfer from training to inference.

NOTE: K2 training requires 64 GPUs (TP=8, EP=8) due to FP8 conversion memory.
"""

import os
import sys
import time
import uuid

import requests

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")

K2_MODEL = "moonshotai/Kimi-K2-Thinking"
# Local path for offline tokenizer loading (Volcano has no internet)
K2_MODEL_PATH = "/vePFS-Mindverse/share/huggingface/hub/models--moonshotai--Kimi-K2-Thinking/snapshots/612681931a8c906ddb349f8ad0f582cb552189cd"


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
    session_id = f"k2_inf_{uuid.uuid4().hex[:8]}"
    url = f"{BASE_URL}/api/v1/create_sampling_session"
    payload = {
        "session_id": session_id,
        "base_model": base_model,
    }
    print(f"Creating sampling session: {session_id}")
    print(f"Base model: {base_model}")

    # K2 model loading takes ~20-30 minutes for 62 shards at ~22s/shard
    resp = requests.post(url, json=payload, headers=get_headers(), timeout=1800)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

    result = resp.json()
    sampling_session_id = result.get("sampling_session_id")
    print(f"Sampling session created: {sampling_session_id}")
    return sampling_session_id


def sample(sampling_session_id: str, prompt: str, max_tokens: int = 50, temperature: float = 0.7) -> dict:
    """Sample from model."""
    # First tokenize the prompt (use local path for offline environment)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(K2_MODEL_PATH, trust_remote_code=True, local_files_only=True)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)

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
    print("Kimi K2 Inference Test (vLLM, TP=16 multi-node for LoRA)")
    print("=" * 70)
    print(f"Model: {K2_MODEL}")
    print(f"Config: TP=16 across 2 nodes (16 GPUs) - LoRA ENABLED for weight transfer")
    print("Uses MultiNodeInferenceEngine with vLLM's native Ray distributed backend")
    print()

    # Check vLLM status first
    print("Checking vLLM status...")
    try:
        resp = requests.get(f"{BASE_URL}/api/v1/vllm_status", headers=get_headers(), timeout=30)
        status = resp.json()
        print(f"vLLM status: {status}")
    except Exception as e:
        print(f"Warning: Could not check vLLM status: {e}")

    # Step 1: Create sampling session
    print("\nStep 1: Creating K2 sampling session...")
    print("This will initialize vLLM with K2 (may take 5-10 minutes)...")
    t0 = time.time()
    try:
        sampling_session_id = create_sampling_session(K2_MODEL)
        init_time = time.time() - t0
        print(f"Session created in {init_time:.1f}s")
    except Exception as e:
        print(f"FAILED: {e}")
        return 1

    # Step 2: Sample
    print("\nStep 2: Sampling from K2...")
    prompt = "Q: What is 2+2?\nA:"
    print(f"Prompt: {prompt!r}")

    t0 = time.time()
    try:
        result = sample(sampling_session_id, prompt, max_tokens=20, temperature=0.0)
        sample_time = time.time() - t0

        if "error" in result:
            print(f"FAILED: {result['error']}")
            return 1

        # Decode output (use local path for offline environment)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(K2_MODEL_PATH, trust_remote_code=True, local_files_only=True)

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

    # Summary
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"vLLM Init: {init_time:.1f}s")
    print(f"Sampling: {'OK' if sample_ok else 'FAILED'}")

    if sample_ok:
        print("\nK2 vLLM inference (TP=16, LoRA enabled): WORKING")
        return 0
    else:
        print("\nK2 vLLM inference: FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
