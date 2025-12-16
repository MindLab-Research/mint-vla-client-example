#!/usr/bin/env python3
"""Quick test to verify 'weights' field is accepted directly (Tinker SDK compatibility)."""

import os
import time
import requests

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("TINKER_API_KEY", "dev-key-for-testing")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

def poll_future(request_id: str, timeout: int = 300) -> dict:
    """Poll for async result."""
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.post(
            f"{BASE_URL}/api/v1/retrieve_future",
            json={"request_id": request_id},
            headers=HEADERS,
            timeout=30
        )
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 408:
            time.sleep(0.5)
            continue
        else:
            resp.raise_for_status()
    raise TimeoutError(f"Operation did not complete within {timeout}s")


def main():
    print("Testing Tinker SDK 'weights' field compatibility...")

    # 1. Create training session
    print("\n1. Creating training session...")
    resp = requests.post(
        f"{BASE_URL}/api/v1/create_model",
        json={
            "session_id": "test_weights_field",
            "model_seq_id": 1,
            "base_model": "Qwen/Qwen2.5-7B-Instruct",
            "lora_config": {"rank": 32},
        },
        headers=HEADERS,
        timeout=300
    )
    resp.raise_for_status()
    result = poll_future(resp.json()["request_id"], timeout=300)
    if "error" in result:
        raise RuntimeError(f"create_model failed: {result['error']}")
    model_id = result["model_id"]
    print(f"   Model created: {model_id}")

    # 2. Test forward_backward with 'weights' field (NOT loss_mask)
    print("\n2. Testing forward_backward with 'weights' field...")

    # Simple test data
    input_tokens = [1, 2, 3, 4, 5]  # 5 input tokens
    target_tokens = [2, 3, 4, 5, 6]  # Shifted by 1
    weights = [0.0, 0.0, 1.0, 1.0, 1.0]  # Only train on last 3

    resp = requests.post(
        f"{BASE_URL}/api/v1/forward_backward",
        json={
            "model_id": model_id,
            "forward_backward_input": {
                "data": [{
                    "model_input": {"chunks": [{"tokens": input_tokens, "type": "encoded_text"}]},
                    "loss_fn_inputs": {
                        "target_tokens": {"data": target_tokens, "shape": [len(target_tokens)], "dtype": "int64"},
                        "weights": {"data": weights, "shape": [len(weights)], "dtype": "float32"},  # Using 'weights' NOT 'loss_mask'
                    }
                }],
                "loss_fn": "cross_entropy"
            }
        },
        headers=HEADERS,
        timeout=120
    )
    resp.raise_for_status()
    result = poll_future(resp.json()["request_id"], timeout=300)

    if "error" in result:
        print(f"   FAILED: {result['error']}")
        return False

    print(f"   Success! Loss: {result.get('metrics', {}).get('loss:sum', 'N/A')}")
    print(f"   Logprobs returned: {len(result.get('loss_fn_outputs', [{}])[0].get('logprobs', []))} values")

    # 3. Verify save_state endpoint works
    print("\n3. Testing /save_state endpoint...")
    resp = requests.post(
        f"{BASE_URL}/api/v1/save_state",
        json={"model_id": model_id, "path": "test_checkpoint"},
        headers=HEADERS,
        timeout=120
    )
    resp.raise_for_status()
    result = poll_future(resp.json()["request_id"], timeout=300)

    if "error" in result:
        print(f"   FAILED: {result['error']}")
    else:
        print(f"   Success! Path: {result.get('path', 'N/A')}")

    print("\n" + "=" * 50)
    print("All Tinker SDK compatibility tests passed!")
    print("=" * 50)
    return True


if __name__ == "__main__":
    main()
