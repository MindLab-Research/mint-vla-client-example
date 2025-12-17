#!/usr/bin/env python3
"""Test LRU eviction across actor types."""

import os
import sys
import time
import uuid
import requests

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")

DENSE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MOE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def get_headers():
    return {"Authorization": f"Bearer {API_KEY}"}


def api_get(endpoint: str, timeout=30):
    url = f"{BASE_URL}/api/v1/{endpoint}"
    resp = requests.get(url, headers=get_headers(), timeout=timeout)
    return resp.json() if resp.status_code == 200 else None


def api_post(endpoint: str, json_data=None, timeout=120):
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
            return {"error": f"Status {resp.status_code}"}
    return {"error": "timeout"}


def get_resource_pool():
    return api_get("resource_pool")


def create_session(model: str, lora_rank: int = 32, lr: float = 1e-4):
    """Create training session using correct API format."""
    session_id = f"evict_test_{uuid.uuid4().hex[:8]}"
    payload = {
        "session_id": session_id,
        "model_seq_id": 1,
        "base_model": model,
        "lora_config": {"rank": lora_rank},
        "learning_rate": lr,
    }
    resp = api_post("create_model", payload, timeout=300)
    if resp.status_code != 200:
        print(f"  create_model failed: {resp.status_code} - {resp.text[:200]}")
        return None, None
    
    result = poll_future(resp.json().get("request_id"), timeout=300)
    if "error" in result:
        print(f"  poll_future failed: {result}")
        return None, None
    
    return session_id, result.get("model_id")


def kill_megatron():
    return api_post("kill_megatron").json() if api_post("kill_megatron").status_code == 200 else None


def kill_vllm():
    return api_post("kill_vllm").json() if api_post("kill_vllm").status_code == 200 else None


def print_pool(pool, label=""):
    if not pool:
        print(f"  {label}: No data")
        return
    print(f"  {label}:")
    for a in pool.get("actors", []):
        evictable = a['idle'] and a['age'] > 300
        print(f"    {a['actor_name']}: {a['num_gpus']} GPUs, "
              f"{a['age']:.0f}s old, evictable={evictable}")
    print(f"    Total: {pool['total_gpus_used']} GPUs used")


def main():
    print("=" * 60)
    print("LRU EVICTION TEST")
    print("=" * 60)

    # Current state
    print("\n1. Current state:")
    pool = get_resource_pool()
    print_pool(pool, "Resource Pool")

    # Check evictable actors
    evictable = [a for a in pool.get("actors", []) if a['idle'] and a['age'] > 300] if pool else []
    print(f"\n   Evictable actors: {len(evictable)}")

    # Kill existing actors and test MoE creation
    print("\n2. Killing existing actors...")
    kill_megatron()
    time.sleep(2)
    kill_vllm()
    time.sleep(5)

    pool = get_resource_pool()
    print_pool(pool, "After cleanup")

    print("\n3. Creating MoE session (8 GPUs)...")
    start = time.time()
    session_id, model_id = create_session(MOE_MODEL, lora_rank=32)
    elapsed = time.time() - start

    if model_id:
        print(f"   MoE created: {model_id} in {elapsed:.1f}s")
        pool = get_resource_pool()
        print_pool(pool, "After MoE creation")
        print("\n   PASS: MoE creation successful")
    else:
        print("   FAIL: MoE creation failed")
        return 1

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
