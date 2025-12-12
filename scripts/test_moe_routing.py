#!/usr/bin/env python
"""MoE Routing Integration Test - Phase 2 Validation.

Tests that MoE models (Qwen3-30B-A3B) route to MegatronTrainingWorker.
Pure HTTP client test - no Ray dependency.

The server API returns "backend" field indicating routing:
- "megatron" = MegatronTrainingWorker (8 GPUs for MoE)
- None or other = TrainingWorker (1 GPU for dense)

Usage:
    TINKER_BASE_URL=http://localhost:8000 python scripts/test_moe_routing.py
"""

import os
import sys
import time
import json
import uuid
import requests

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")


def create_training_session(base_model: str, rank: int = 16) -> dict:
    """Create a training session via API.

    The create_model API is async - it returns a future that must be polled.
    """
    session_id = str(uuid.uuid4())
    model_seq_id = 1

    # Step 1: Submit create_model request
    url = f"{BASE_URL}/api/v1/create_model"
    payload = {
        "session_id": session_id,
        "model_seq_id": model_seq_id,
        "base_model": base_model,
        "lora_config": {"rank": rank},
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    future = resp.json()
    request_id = future.get("request_id")

    # Step 2: Poll for result
    poll_url = f"{BASE_URL}/api/v1/retrieve_future"
    max_wait = 600  # 10 minutes for MoE initialization
    start = time.time()

    while time.time() - start < max_wait:
        poll_resp = requests.post(poll_url, json={"request_id": request_id}, timeout=30)

        if poll_resp.status_code == 200:
            result = poll_resp.json()
            result["session_id"] = session_id  # Add for reference
            return result
        elif poll_resp.status_code == 408:
            # Still processing
            time.sleep(2)
            continue
        else:
            poll_resp.raise_for_status()

    raise TimeoutError(f"create_model did not complete within {max_wait}s")


def get_health():
    """Check server health."""
    url = f"{BASE_URL}/api/v1/healthz"
    resp = requests.get(url, timeout=10)
    return resp.json()


def test_moe_model_routing():
    """Test that MoE model (Qwen3-30B-A3B) routes to MegatronTrainingWorker."""
    print("=" * 70)
    print("TEST: MoE Model Routing to MegatronTrainingWorker")
    print("=" * 70)

    moe_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"

    print(f"\nCreating training session for: {moe_model}")
    print("Expected: backend='megatron' (MegatronTrainingWorker, 8 GPUs)")
    print("\nThis may take several minutes for Megatron initialization...")

    t0 = time.time()
    try:
        result = create_training_session(moe_model, rank=16)
        init_time = time.time() - t0

        print(f"\nSession created in {init_time:.2f}s")
        print(f"Response: {json.dumps(result, indent=2)}")

        backend = result.get("backend", "unknown")
        model_id = result.get("model_id", "unknown")

        print(f"\nBackend: {backend}")
        print(f"Model ID: {model_id}")

        if backend == "megatron":
            print("\n[PASS] MoE model routed to MegatronTrainingWorker")
            return True, model_id
        else:
            print(f"\n[FAIL] Expected backend='megatron', got: '{backend}'")
            return False, model_id

    except requests.exceptions.HTTPError as e:
        print(f"\n[FAIL] HTTP Error: {e}")
        if e.response is not None:
            print(f"Response: {e.response.text}")
        return False, None
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_dense_model_routing():
    """Test that dense model routes to TrainingWorker."""
    print("\n" + "=" * 70)
    print("TEST: Dense Model Routing to TrainingWorker (Control)")
    print("=" * 70)

    dense_model = "Qwen/Qwen2.5-7B-Instruct"

    print(f"\nCreating training session for: {dense_model}")
    print("Expected: backend != 'megatron' (TrainingWorker, 1 GPU)")

    t0 = time.time()
    try:
        result = create_training_session(dense_model, rank=16)
        init_time = time.time() - t0

        print(f"\nSession created in {init_time:.2f}s")
        print(f"Response: {json.dumps(result, indent=2)}")

        backend = result.get("backend", "")

        if backend != "megatron":
            print("\n[PASS] Dense model routed to TrainingWorker (not megatron)")
            return True
        else:
            print("\n[FAIL] Dense model incorrectly routed to megatron")
            return False

    except requests.exceptions.HTTPError as e:
        print(f"\n[FAIL] HTTP Error: {e}")
        if e.response is not None:
            print(f"Response: {e.response.text}")
        return False
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print(f"Server: {BASE_URL}")
    print("=" * 70)
    print("PHASE 2 MoE ROUTING INTEGRATION TEST")
    print("=" * 70)

    # Check server health first
    print("\nChecking server health...")
    try:
        health = get_health()
        print(f"Server status: {health.get('status')}")
    except Exception as e:
        print(f"Server not available: {e}")
        return 1

    results = {}

    # Test 1: MoE routing (this is the main Phase 2 test)
    moe_pass, moe_model_id = test_moe_model_routing()
    results["moe_routing"] = moe_pass

    # Test 2: Dense routing (control - verifies routing logic doesn't break dense)
    results["dense_routing"] = test_dense_model_routing()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_pass = True
    for test, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {test}: {status}")
        if not result:
            all_pass = False

    print("\n" + ("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED"))
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
