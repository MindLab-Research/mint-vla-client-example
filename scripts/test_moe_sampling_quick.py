#!/usr/bin/env python3
"""Quick test: MoE sampling to check hidden_size mismatch."""

import time
import uuid
import requests

BASE_URL = "http://localhost:8000"
API = f"{BASE_URL}/api/v1"
MOE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def poll_future(request_id, timeout=300):
    poll_url = f"{API}/retrieve_future"
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.post(poll_url, json={"request_id": request_id}, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 408:
            time.sleep(0.5)
            continue
        else:
            resp.raise_for_status()
    raise TimeoutError(f"Operation did not complete within {timeout}s")


def main():
    # Create MoE session
    session_id = f"moe_test_{uuid.uuid4().hex[:8]}"
    print(f"Creating MoE session: {session_id}")

    resp = requests.post(f"{API}/create_model", json={
        "session_id": session_id,
        "model_seq_id": 1,
        "base_model": MOE_MODEL,
        "lora_config": {"rank": 32},
        "learning_rate": 1e-4,
    }, timeout=300)
    resp.raise_for_status()
    result = poll_future(resp.json().get("request_id"), timeout=300)
    if "error" in result:
        print(f"Create failed: {result['error']}")
        return 1
    model_id = result.get("model_id")
    print(f"model_id: {model_id}")

    # Save weights
    print("Saving weights...")
    resp = requests.post(f"{API}/save_weights", json={"model_id": model_id, "name": "test"}, timeout=120)
    resp.raise_for_status()
    result = poll_future(resp.json().get("request_id"), timeout=120)
    if "error" in result:
        print(f"save_weights error: {result['error']}")
        return 1
    print(f"save_weights result: {result}")

    # Sample
    print("Sampling...")
    prompt_tokens = [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198]
    resp = requests.post(f"{API}/asample", json={
        "model_id": model_id,
        "prompt": {"chunks": [{"tokens": prompt_tokens, "type": "encoded_text"}]},
        "sampling_params": {"max_tokens": 10, "temperature": 0.0},
        "num_samples": 1,
    }, timeout=120)
    resp.raise_for_status()
    result = poll_future(resp.json().get("request_id"), timeout=120)
    print(f"sample result: {result}")

    if "error" in result:
        print(f"FAIL: sampling error: {result['error']}")
        return 1
    elif "sequences" not in result or len(result["sequences"]) == 0:
        print("FAIL: No sequences returned")
        return 1
    else:
        print(f"PASS: Got {len(result['sequences'])} sequences")
        return 0


if __name__ == "__main__":
    exit(main())
