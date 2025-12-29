#!/usr/bin/env python3
"""Test K2 training at various context lengths to calibrate memory model.

Tests 4K, 5K, 6K, 7K context to find exact working maximum.
"""

import os
import sys
import requests
import time
import uuid

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")
K2_MODEL = "moonshotai/Kimi-K2-Thinking"

HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


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
                print(f"    Waiting... {elapsed:.0f}s")
                last_print = elapsed
            time.sleep(2)
            continue
        else:
            return {"error": f"HTTP {resp.status_code}: {resp.text}"}
    return {"error": f"Timeout after {timeout}s"}


def create_session():
    """Create training session."""
    session_id = f"k2_calibration_{uuid.uuid4().hex[:8]}"
    print(f"Creating session: {session_id}")

    resp = requests.post(
        f"{BASE_URL}/api/v1/create_model",
        json={
            "session_id": session_id,
            "model_seq_id": 1,
            "base_model": K2_MODEL,
            "lora_config": {"rank": 16},
            "learning_rate": 1e-5,
        },
        headers=HEADERS,
        timeout=60,
    )
    if resp.status_code != 200:
        print(f"Error: HTTP {resp.status_code}: {resp.text}")
        return None, None

    request_id = resp.json().get("request_id")
    print(f"Request ID: {request_id}")
    print("Waiting for model initialization...")

    result = poll_future(request_id, timeout=1200)
    if "error" in result:
        print(f"Error: {result['error']}")
        return None, None

    model_id = result.get("model_id")
    print(f"Session ready: model_id={model_id}")
    return session_id, model_id


def test_context(model_id: str, session_id: str, seq_len: int):
    """Test forward-backward at specific context length."""
    print(f"\n  Testing {seq_len} tokens...", end="", flush=True)

    # Generate synthetic data
    base_tokens = [100] * seq_len
    loss_mask = [1.0] * (seq_len - 1)

    api_data = [{
        "model_input": {"chunks": [{"tokens": base_tokens[:-1], "type": "encoded_text"}]},
        "loss_fn_inputs": {
            "target_tokens": {"data": base_tokens[1:], "shape": [seq_len - 1], "dtype": "int64"},
            "loss_mask": {"data": loss_mask, "shape": [seq_len - 1], "dtype": "float32"},
        },
    }]

    payload = {
        "model_id": model_id,
        "forward_backward_input": {
            "data": api_data,
            "loss_fn": "cross_entropy",
        },
        "session_id": session_id,
    }

    start = time.time()
    resp = requests.post(
        f"{BASE_URL}/api/v1/forward_backward",
        json=payload,
        headers=HEADERS,
        timeout=120,
    )

    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}", "time": time.time() - start}

    request_id = resp.json().get("request_id")
    if request_id:
        result = poll_future(request_id, timeout=600)
        elapsed = time.time() - start
        if "error" in result:
            return {"error": result["error"], "time": elapsed}
        loss = result.get("metrics", {}).get("loss:mean", 0)
        return {"success": True, "loss": loss, "time": elapsed}

    return {"error": "No request_id", "time": time.time() - start}


def main():
    print("=" * 60)
    print("K2 Memory Calibration Test")
    print("=" * 60)

    session_id, model_id = create_session()
    if not session_id:
        return 1

    # Test various context lengths to find max working context
    # Based on calibrated memory model:
    # - 8K: ~67.88 GiB peak (known working)
    # - 14K: ~75.65 GiB peak (theoretical max with 2 GiB headroom)
    # - 16K: ~78.24 GiB peak (OOM observed)
    test_lengths = [8192, 10240, 12288, 13312, 14336, 15360]

    print("\n" + "-" * 60)
    print("Testing context lengths...")
    print("-" * 60)

    results = {}
    for seq_len in test_lengths:
        result = test_context(model_id, session_id, seq_len)
        results[seq_len] = result

        if "error" in result:
            print(f" FAIL ({result['time']:.1f}s)")
            print(f"    Error: {result['error'][:200]}")
            break
        else:
            print(f" PASS: loss={result['loss']:.4f} ({result['time']:.1f}s)")

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    max_working = 0
    for seq_len, r in results.items():
        status = "PASS" if "success" in r else "FAIL"
        print(f"  {seq_len:>6} tokens: {status}")
        if "success" in r:
            max_working = seq_len

    print(f"\nMax working context: {max_working} tokens")
    print(f"For 32K batch: need {32768 // max(max_working, 1)} micro-batches")

    return 0


if __name__ == "__main__":
    sys.exit(main())
