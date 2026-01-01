#!/usr/bin/env python3
"""Test K2 training with 32K context using CP=2 folding.

Goal: Verify MoE Parallel Folding (CP=2, EP=8) enables 32K context training
on 64 GPUs (same GPU count as CP=1, EP=8 with 16K limit).

Memory target: 32K tokens / CP=2 = 16K per GPU ≈ same as 16K with CP=1
"""

import os
import sys
import requests
import time
import uuid

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")
K2_MODEL = "moonshotai/Kimi-K2-Thinking"
K2_MODEL_PATH = "/vePFS-Mindverse/share/huggingface/hub/models--moonshotai--Kimi-K2-Thinking/snapshots/612681931a8c906ddb349f8ad0f582cb552189cd"

HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def poll_future(request_id: str, timeout: int = 1800):
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


def create_session(model: str, lora_rank: int = 16, lr: float = 1e-5):
    """Create training session."""
    session_id = f"k2_32k_test_{uuid.uuid4().hex[:8]}"
    print(f"Creating session: {session_id}")
    print(f"  model: {model}")
    print(f"  lora_rank: {lora_rank}")
    print(f"  Expected config: TP=8, EP=8, CP=2 (MoE Parallel Folding)")

    resp = requests.post(
        f"{BASE_URL}/api/v1/create_model",
        json={
            "session_id": session_id,
            "model_seq_id": 1,
            "base_model": model,
            "lora_config": {"rank": lora_rank},
            "learning_rate": lr,
        },
        headers=HEADERS,
        timeout=60,
    )
    if resp.status_code != 200:
        print(f"Error: HTTP {resp.status_code}: {resp.text}")
        return None, None

    request_id = resp.json().get("request_id")
    print(f"Request ID: {request_id}")
    print("Waiting for model initialization (expect ~5-10 min for K2 with CP=2)...")

    result = poll_future(request_id, timeout=1200)
    if "error" in result:
        print(f"Error: {result['error']}")
        return None, None

    model_id = result.get("model_id")
    print(f"Session ready: model_id={model_id}")
    return session_id, model_id


def forward_backward(model_id: str, data: list, session_id: str = None):
    """Run forward-backward pass."""
    payload = {
        "model_id": model_id,
        "forward_backward_input": {
            "data": data,
            "loss_fn": "cross_entropy",
        },
    }
    if session_id:
        payload["session_id"] = session_id

    resp = requests.post(
        f"{BASE_URL}/api/v1/forward_backward",
        json=payload,
        headers=HEADERS,
        timeout=120,
    )

    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}: {resp.text}"}

    request_id = resp.json().get("request_id")
    if request_id:
        return poll_future(request_id, timeout=600)
    return resp.json()


def test_sequence_length(model_id: str, session_id: str, seq_len: int):
    """Test forward-backward with specific sequence length."""
    print(f"\nTesting sequence length: {seq_len} tokens")

    # Generate synthetic data: repeat a simple pattern
    # Use token IDs that are valid for Qwen/K2 (1-163839)
    base_tokens = [100] * seq_len  # Simple repeated token
    loss_mask = [1.0] * (seq_len - 1)

    api_data = [{
        "model_input": {"chunks": [{"tokens": base_tokens[:-1], "type": "encoded_text"}]},
        "loss_fn_inputs": {
            "target_tokens": {"data": base_tokens[1:], "shape": [seq_len - 1], "dtype": "int64"},
            "loss_mask": {"data": loss_mask, "shape": [seq_len - 1], "dtype": "float32"},
        },
    }]

    start = time.time()
    result = forward_backward(model_id, api_data, session_id)
    elapsed = time.time() - start

    if "error" in result:
        print(f"  FAILED: {result['error']}")
        return False, elapsed

    loss = result.get("metrics", {}).get("loss:mean", 0)
    tokens = result.get("metrics", {}).get("num_tokens:sum", 0)
    print(f"  PASS: loss={loss:.4f}, tokens={tokens:.0f}, time={elapsed:.1f}s")
    return True, elapsed


def main():
    print("=" * 70)
    print("K2 32K Context Test (MoE Parallel Folding: CP=2, EP=8)")
    print("=" * 70)

    # Check cluster
    print("\nChecking cluster resources...")
    try:
        resp = requests.get(f"{BASE_URL}/api/v1/resource_pool", headers=HEADERS, timeout=30)
        if resp.status_code == 200:
            resources = resp.json()
            gpus = resources.get("gpu_resources", {})
            print(f"  Total GPUs: {gpus.get('total', 'unknown')}")
            print(f"  Available: {gpus.get('available', 'unknown')}")
    except Exception as e:
        print(f"  Could not check resources: {e}")

    # Create session
    print("\n" + "-" * 50)
    print("Creating K2 training session...")
    print("  This will create Megatron worker group with CP=2 folding")
    print("-" * 50)

    start = time.time()
    session_id, model_id = create_session(K2_MODEL, lora_rank=16, lr=1e-5)
    if not session_id:
        print("Failed to create session")
        return 1

    print(f"Session created in {time.time() - start:.1f}s")

    # Note: We use synthetic tokens (fixed token ID), no tokenizer needed
    # The test validates memory with long sequences, not semantic content

    # Test sequence lengths
    print("\n" + "-" * 50)
    print("Testing sequence lengths...")
    print("-" * 50)

    results = {}

    # Start with known-working length (8K), then increase
    for seq_len in [8192, 16384, 32768]:
        success, elapsed = test_sequence_length(model_id, session_id, seq_len)
        results[seq_len] = {"success": success, "time": elapsed}

        if not success:
            print(f"\n  Stopping at {seq_len} - failed")
            break

    # Summary
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    for seq_len, result in results.items():
        status = "PASS" if result["success"] else "FAIL"
        print(f"  {seq_len:,} tokens: {status} ({result['time']:.1f}s)")

    max_working = max([s for s, r in results.items() if r["success"]], default=0)
    print(f"\nMaximum working context: {max_working:,} tokens")

    if max_working >= 32768:
        print("32K context with CP=2 folding: VERIFIED")
        return 0
    else:
        print(f"32K not reached. Check server logs for OOM details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
