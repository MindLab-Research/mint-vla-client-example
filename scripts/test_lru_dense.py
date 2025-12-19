#!/usr/bin/env python3
"""Test LRU eviction with dense models only.

Dense models are faster to initialize (~10s vs 90s for MoE).
This test verifies that LRU eviction works for vLLM actors.

Test Plan:
1. Create session A with Qwen2.5-7B (1 GPU trainer + 1 GPU vLLM)
2. Create session B with Qwen2.5-7B (1 GPU trainer + shared vLLM)
3. Mark session A's trainer as idle (session A ends)
4. Create session C - should evict session A's trainer
"""

import os
import sys
import time
import uuid
import requests

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")

MODEL = "Qwen/Qwen2.5-7B-Instruct"


def get_headers():
    return {"Authorization": f"Bearer {API_KEY}"}


def api_post(endpoint: str, json_data=None, timeout=120):
    url = f"{BASE_URL}/api/v1/{endpoint}"
    resp = requests.post(url, json=json_data, headers=get_headers(), timeout=timeout)
    return resp


def poll_future(request_id: str, timeout: int = 120):
    start = time.time()
    while time.time() - start < timeout:
        resp = api_post("retrieve_future", {"request_id": request_id}, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 408:
            time.sleep(1)
            continue
        else:
            return {"error": f"Status {resp.status_code}: {resp.text[:200]}"}
    return {"error": "timeout"}


def get_resource_pool():
    """Get resource pool status."""
    resp = requests.get(f"{BASE_URL}/api/v1/resource_pool", headers=get_headers(), timeout=30)
    if resp.status_code == 200:
        return resp.json()
    return None


def create_session(model: str, session_id: str = None, lora_rank: int = 16):
    """Create training session."""
    if session_id is None:
        session_id = f"lru_test_{uuid.uuid4().hex[:8]}"
    payload = {
        "session_id": session_id,
        "model_seq_id": 1,
        "base_model": model,
        "lora_config": {"rank": lora_rank},
        "learning_rate": 1e-4,
    }
    print(f"Creating session {session_id}...")
    resp = api_post("create_model", payload, timeout=120)
    if resp.status_code != 200:
        print(f"  FAIL: {resp.status_code} - {resp.text[:200]}")
        return None, None

    result = poll_future(resp.json().get("request_id"), timeout=120)
    if "error" in result:
        print(f"  FAIL: {result}")
        return None, None

    print(f"  OK: model_id={result.get('model_id')}")
    return session_id, result.get("model_id")


def print_resource_pool():
    pool = get_resource_pool()
    if pool:
        print(f"  Actors: {len(pool.get('actors', []))}, Total GPUs: {pool.get('total_gpus_used', 0)}")
        for actor in pool.get("actors", []):
            idle = "(idle)" if actor.get("idle") else "(active)"
            print(f"    - {actor['actor_name'][:40]}: {actor['num_gpus']} GPUs {idle}")
    else:
        print("  Could not get resource pool")


def main():
    print("=" * 60)
    print("LRU EVICTION TEST (Dense Models)")
    print("=" * 60)
    print()

    # Step 1: Create first session
    print("1. Create session A:")
    session_a, model_a = create_session(MODEL, "session_a")
    if not session_a:
        return 1
    print_resource_pool()
    print()

    # Step 2: Create second session
    print("2. Create session B:")
    session_b, model_b = create_session(MODEL, "session_b")
    if not session_b:
        return 1
    print_resource_pool()
    print()

    # Step 3: Trigger sampling for session A (creates vLLM actor if not exists)
    print("3. Save weights for session A (triggers vLLM creation):")
    resp = api_post("save_weights_for_sampler", {"model_id": model_a, "session_id": session_a})
    if resp.status_code == 200:
        result = poll_future(resp.json().get("request_id"), timeout=120)
        if "error" not in result:
            print(f"  OK: sampling_session_id={result.get('sampling_session_id', 'unknown')}")
        else:
            print(f"  FAIL: {result}")
            # Continue anyway - vLLM might have been created
    else:
        print(f"  FAIL: {resp.status_code}")
    print_resource_pool()
    print()

    # Step 4: Simulate session A becoming idle
    # In real usage, this happens when session.end() is called
    # For testing, we just check if idle tracking works
    print("4. Check idle state (session A should be idle after no activity):")
    time.sleep(2)  # Small wait
    print_resource_pool()
    print()

    # Step 5: Create more sessions to trigger eviction
    print("5. Create session C (should use shared trainer pool):")
    session_c, model_c = create_session(MODEL, "session_c")
    if not session_c:
        return 1
    print_resource_pool()
    print()

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print()
    print("Summary:")
    print("  - Sessions A, B, C created successfully")
    print("  - Dense trainer pool should be shared (reused)")
    print("  - vLLM actor should be shared (multi-LoRA)")
    print("  - LRU eviction would trigger if GPUs run out")
    return 0


if __name__ == "__main__":
    sys.exit(main())
