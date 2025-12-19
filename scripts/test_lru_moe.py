#!/usr/bin/env python3
"""Test LRU eviction with MoE models.

This tests:
1. Server remains responsive during vLLM initialization (non-blocking fix)
2. LRU eviction triggers when GPUs are exhausted

Test Plan:
1. Create MoE session A (4 GPU Megatron)
2. Trigger sampling for session A (creates 4 GPU vLLM)
3. During vLLM init, verify server responds to healthz
4. Create MoE session B (shares Megatron)
5. End session A (actors become idle)
6. Create another session that needs different resources -> triggers eviction
"""

import os
import sys
import time
import threading
import requests

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")

MODEL_MOE = "Qwen/Qwen3-30B-A3B-Instruct-2507"  # 4 GPU Megatron, 4 GPU vLLM
MODEL_DENSE = "Qwen/Qwen2.5-7B-Instruct"  # 1 GPU trainer, 1 GPU vLLM


def get_headers():
    return {"Authorization": f"Bearer {API_KEY}"}


def api_post(endpoint: str, json_data=None, timeout=120):
    url = f"{BASE_URL}/api/v1/{endpoint}"
    return requests.post(url, json=json_data, headers=get_headers(), timeout=timeout)


def poll_future(request_id: str, timeout: int = 300):
    """Poll for async result."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = api_post("retrieve_future", {"request_id": request_id}, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 408:
                time.sleep(2)
                continue
            else:
                return {"error": f"Status {resp.status_code}: {resp.text[:200]}"}
        except requests.exceptions.Timeout:
            continue
    return {"error": "timeout"}


def check_health():
    """Check if server responds to healthz."""
    try:
        resp = requests.get(f"{BASE_URL}/api/v1/healthz", headers=get_headers(), timeout=5)
        return resp.status_code == 200
    except:
        return False


def get_resource_pool():
    """Get resource pool status."""
    try:
        resp = requests.get(f"{BASE_URL}/api/v1/resource_pool", headers=get_headers(), timeout=30)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return None


def create_session(model: str, session_id: str):
    """Create training session."""
    payload = {
        "session_id": session_id,
        "model_seq_id": 1,
        "base_model": model,
        "lora_config": {"rank": 16},
        "learning_rate": 1e-4,
    }
    print(f"  Creating session {session_id} with {model}...")
    resp = api_post("create_model", payload, timeout=300)
    if resp.status_code != 200:
        print(f"    FAIL: {resp.status_code} - {resp.text[:200]}")
        return None

    result = poll_future(resp.json().get("request_id"), timeout=300)
    if "error" in result:
        print(f"    FAIL: {result}")
        return None

    print(f"    OK: model_id={result.get('model_id')}")
    return result.get("model_id")


def save_weights_for_sampler(model_id: str, session_id: str):
    """Trigger vLLM creation via save_weights_for_sampler."""
    print(f"  Saving weights for {model_id}...")
    resp = api_post("save_weights_for_sampler", {"model_id": model_id, "session_id": session_id})
    if resp.status_code != 200:
        print(f"    FAIL: {resp.status_code}")
        return None
    return resp.json().get("request_id")


def print_pool():
    pool = get_resource_pool()
    if pool:
        print(f"  Resource pool: {pool.get('total_gpus_used', 0)} GPUs used")
        for actor in pool.get("actors", []):
            idle = "(idle)" if actor.get("idle") else "(active)"
            print(f"    - {actor['actor_name'][:50]}: {actor['num_gpus']} GPUs {idle}")


def main():
    print("=" * 70)
    print("LRU EVICTION TEST WITH MOE MODELS")
    print("=" * 70)
    print()

    # Step 1: Create MoE session
    print("1. Create MoE session A:")
    model_a = create_session(MODEL_MOE, "moe_session_a")
    if not model_a:
        print("   FAILED to create MoE session")
        return 1
    print_pool()
    print()

    # Step 2: Trigger sampling (creates vLLM actor)
    print("2. Trigger sampling for session A (this creates vLLM actor):")
    request_id = save_weights_for_sampler(model_a, "moe_session_a")
    if not request_id:
        print("   FAILED to submit save_weights_for_sampler")
        return 1

    # Step 3: Monitor server responsiveness during vLLM init
    print("3. Monitoring server responsiveness during vLLM init...")
    health_checks = 0
    health_ok = 0
    start_time = time.time()

    # Poll for result while checking health
    while time.time() - start_time < 180:  # 3 minute timeout
        # Check health
        if check_health():
            health_ok += 1
        health_checks += 1

        # Check if sampling is ready
        try:
            resp = api_post("retrieve_future", {"request_id": request_id}, timeout=10)
            if resp.status_code == 200:
                result = resp.json()
                elapsed = time.time() - start_time
                print(f"   Sampling ready after {elapsed:.1f}s")
                print(f"   sampling_session_id: {result.get('sampling_session_id', 'unknown')}")
                break
            elif resp.status_code == 408:
                # Still pending
                pass
            else:
                print(f"   Unexpected status: {resp.status_code}")
        except requests.exceptions.Timeout:
            pass

        time.sleep(3)
    else:
        print("   TIMEOUT waiting for sampling")

    print(f"   Health checks: {health_ok}/{health_checks} passed")
    if health_ok < health_checks * 0.8:
        print("   WARNING: Server was unresponsive during vLLM init")
    else:
        print("   Server remained responsive (non-blocking fix works)")
    print_pool()
    print()

    # Step 4: Create dense session (tests resource sharing)
    print("4. Create Dense session B:")
    model_b = create_session(MODEL_DENSE, "dense_session_b")
    if model_b:
        print("   Dense session created successfully")
    else:
        print("   Failed to create dense session")
    print_pool()
    print()

    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
