#!/usr/bin/env python3
"""Test LRU eviction fix for vLLM actors.

Test Plan:
1. Create vLLM actor for Model A (uses 4 GPUs - MoE)
2. Create vLLM actor for Model B (uses 4 GPUs - MoE)
3. Create vLLM actor for Model C (uses 1 GPU - Dense)
4. Create vLLM actor for Model D (uses 1 GPU - Dense)
   At this point, 10 GPUs are used by 4 vLLM actors
5. Try to create Megatron actor for MoE (needs 4 GPUs)
   With 16 GPUs total, 10 used, only 6 available
   Should NOT block - LRU eviction should evict idle vLLM actors

Before fix: Server hangs indefinitely at step 5
After fix: LRU evicts oldest vLLM actors, freeing GPUs for Megatron
"""

import os
import sys
import time
import uuid
import requests

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")

# Use models that share similar GPU requirements
MODEL_A = "Qwen/Qwen3-30B-A3B-Instruct-2507"  # MoE, 4 GPUs inference
MODEL_B = "Qwen/Qwen2.5-7B-Instruct"  # Dense, 1 GPU inference


def get_headers():
    return {"Authorization": f"Bearer {API_KEY}"}


def api_post(endpoint: str, json_data=None, timeout=300):
    url = f"{BASE_URL}/api/v1/{endpoint}"
    resp = requests.post(url, json=json_data, headers=get_headers(), timeout=timeout)
    return resp


def poll_future(request_id: str, timeout: int = 300):
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


def get_gpu_status():
    """Get current GPU usage from Ray."""
    import subprocess
    result = subprocess.run(
        ["ssh", "volcano", "python3", "-c", '''
import ray
ray.init(address="auto", ignore_reinit_error=True)
r = ray.available_resources()
t = ray.cluster_resources()
print(f"{int(r.get('GPU', 0))}/{int(t.get('GPU', 0))}")
'''],
        capture_output=True, text=True
    )
    return result.stdout.strip()


def get_resource_pool():
    """Get resource pool status."""
    resp = requests.get(f"{BASE_URL}/api/v1/resource_pool", headers=get_headers(), timeout=30)
    if resp.status_code == 200:
        return resp.json()
    return None


def create_session(model: str, lora_rank: int = 16, lr: float = 1e-4):
    """Create training session."""
    session_id = f"lru_test_{uuid.uuid4().hex[:8]}"
    payload = {
        "session_id": session_id,
        "model_seq_id": 1,
        "base_model": model,
        "lora_config": {"rank": lora_rank},
        "learning_rate": lr,
    }
    print(f"Creating session for {model}...")
    resp = api_post("create_model", payload, timeout=300)
    if resp.status_code != 200:
        print(f"  FAIL: {resp.status_code} - {resp.text[:200]}")
        return None, None

    result = poll_future(resp.json().get("request_id"), timeout=300)
    if "error" in result:
        print(f"  FAIL: {result}")
        return None, None

    print(f"  OK: model_id={result.get('model_id')}")
    return session_id, result.get("model_id")


def main():
    print("=" * 60)
    print("LRU EVICTION FIX TEST")
    print("=" * 60)
    print(f"Note: MINT_MIN_ACTOR_AGE should be 0 for this test")
    print()

    # Check initial state
    print("1. Initial state:")
    print(f"   GPU status: {get_gpu_status()}")
    pool = get_resource_pool()
    if pool:
        print(f"   Resource pool: {len(pool.get('actors', []))} actors, {pool.get('total_gpus_used', 0)} GPUs used")
    else:
        print("   Resource pool: unavailable")
    print()

    # Step 1: Create RL session with Model A (MoE - uses Megatron 4 GPUs + vLLM 4 GPUs)
    print("2. Create RL session with Model A (MoE - Qwen3-30B-A3B):")
    session_a, model_a = create_session(MODEL_A)
    if not session_a:
        print("   FAIL: Could not create session A")
        return 1
    print(f"   GPU status: {get_gpu_status()}")
    print()

    # Check resource pool
    pool = get_resource_pool()
    if pool:
        for actor in pool.get("actors", []):
            print(f"   Actor: {actor['actor_name']} ({actor['num_gpus']} GPUs)")
    print()

    # Step 2: Create RL session with Model B (Dense - uses Dense trainer 1 GPU + vLLM 1 GPU)
    print("3. Create RL session with Model B (Dense - Qwen2.5-7B):")
    session_b, model_b = create_session(MODEL_B)
    if not session_b:
        print("   FAIL: Could not create session B")
        return 1
    print(f"   GPU status: {get_gpu_status()}")
    print()

    # Check resource pool
    pool = get_resource_pool()
    if pool:
        for actor in pool.get("actors", []):
            idle_str = "(idle)" if actor.get("idle") else "(active)"
            print(f"   Actor: {actor['actor_name']} ({actor['num_gpus']} GPUs) {idle_str}")
    print()

    # Step 3: Trigger sampling for both models to create vLLM actors
    print("4. Trigger sampling to create vLLM actors:")

    # Sample from Model A
    print("   Sampling from Model A...")
    resp = api_post("save_weights_for_sampler", {"model_id": model_a, "session_id": session_a})
    if resp.status_code == 200:
        result = poll_future(resp.json().get("request_id"), timeout=300)
        if "error" not in result:
            print(f"   Model A sampling session: {result.get('sampling_session_id', 'unknown')}")
        else:
            print(f"   Model A sampling FAIL: {result}")
    else:
        print(f"   Model A save_weights FAIL: {resp.status_code}")

    # Sample from Model B
    print("   Sampling from Model B...")
    resp = api_post("save_weights_for_sampler", {"model_id": model_b, "session_id": session_b})
    if resp.status_code == 200:
        result = poll_future(resp.json().get("request_id"), timeout=300)
        if "error" not in result:
            print(f"   Model B sampling session: {result.get('sampling_session_id', 'unknown')}")
        else:
            print(f"   Model B sampling FAIL: {result}")
    else:
        print(f"   Model B save_weights FAIL: {resp.status_code}")

    print(f"   GPU status: {get_gpu_status()}")
    print()

    # Check resource pool
    pool = get_resource_pool()
    if pool:
        print("5. Resource pool after sampling:")
        for actor in pool.get("actors", []):
            idle_str = "(idle)" if actor.get("idle") else "(active)"
            print(f"   Actor: {actor['actor_name']} ({actor['num_gpus']} GPUs) {idle_str}")
        print(f"   Total GPUs used: {pool.get('total_gpus_used', 0)}")
    print()

    # Step 4: Try to create another MoE session - should trigger LRU eviction
    print("6. Create another MoE session (should trigger LRU eviction):")
    print("   Before fix: Server would hang here if not enough GPUs")
    print("   After fix: Should evict idle vLLM actors to free GPUs")

    start = time.time()
    session_c, model_c = create_session(MODEL_A)  # Another MoE session
    elapsed = time.time() - start

    if session_c:
        print(f"   PASS: Session created in {elapsed:.1f}s")
        print(f"   GPU status: {get_gpu_status()}")
    else:
        print(f"   FAIL: Could not create session C after {elapsed:.1f}s")
    print()

    # Final resource pool status
    pool = get_resource_pool()
    if pool:
        print("7. Final resource pool:")
        for actor in pool.get("actors", []):
            idle_str = "(idle)" if actor.get("idle") else "(active)"
            print(f"   Actor: {actor['actor_name']} ({actor['num_gpus']} GPUs) {idle_str}")
        print(f"   Total GPUs used: {pool.get('total_gpus_used', 0)}")
    print()

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
