#!/usr/bin/env python3
"""Test Issue 2: Concurrent sessions with resource pressure.

Tests what happens when:
1. Model A is actively in use (trainer/inferencer)
2. Another session tries to create trainer/inferencer for model B
3. Not enough resources available

Expected: Session B should either wait or trigger eviction.
"""
import asyncio
import os
import sys
import time
import uuid

import httpx
import requests

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")

# Use different models to ensure different trainers/inferers
MODEL_A = "Qwen/Qwen2.5-7B-Instruct"
MODEL_B = "Qwen/Qwen3-0.6B"


def get_headers():
    return {"Authorization": "Bearer dummy"}


def poll_future(request_id, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.post(
            f"{BASE_URL}/api/v1/retrieve_future",
            json={"request_id": request_id},
            headers=get_headers(),
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 408:
            time.sleep(0.5)
        else:
            return {"error": f"HTTP {resp.status_code}: {resp.text}"}
    return {"error": "Timeout"}


def get_resource_pool():
    resp = requests.get(f"{BASE_URL}/api/v1/resource_pool", headers=get_headers(), timeout=10)
    return resp.json()


def print_resources(label):
    pool = get_resource_pool()
    print(f"\n{label}:")
    print(f"  GPUs: {pool.get('total_gpus_used', 0)}")
    actors = pool.get("actors", [])
    for a in actors:
        busy = "BUSY" if not a["idle"] else "idle"
        print(f"    - {a['actor_name']}: {a['num_gpus']}GPU, {busy}")


def create_session(model_name, session_prefix):
    """Create a training session."""
    session_id = f"{session_prefix}_{uuid.uuid4().hex[:6]}"
    resp = requests.post(
        f"{BASE_URL}/api/v1/create_model",
        json={
            "session_id": session_id,
            "model_seq_id": 1,
            "base_model": model_name,
            "lora_config": {"rank": 32},
            "learning_rate": 1e-4,
        },
        headers=get_headers(),
        timeout=300,
    )
    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}: {resp.text}"}
    result = poll_future(resp.json().get("request_id"), timeout=300)
    if "error" in result:
        return {"error": result["error"]}
    return {"model_id": result.get("model_id"), "session_id": session_id}


def train_step(model_id):
    """Run one training step."""
    tokens = list(range(1, 51))  # 50 tokens for longer training
    datum = {
        "model_input": {"chunks": [{"tokens": tokens, "type": "encoded_text"}]},
        "loss_fn_inputs": {
            "target_tokens": {"data": tokens, "shape": [50], "dtype": "int64"},
            "loss_mask": {"data": [0.0] * 25 + [1.0] * 25, "shape": [50], "dtype": "float32"},
        },
    }

    resp = requests.post(
        f"{BASE_URL}/api/v1/forward_backward",
        json={"model_id": model_id, "forward_backward_input": {"data": [datum], "loss_fn": "cross_entropy"}},
        headers=get_headers(),
        timeout=120,
    )
    if resp.status_code != 200:
        return {"error": f"forward_backward HTTP {resp.status_code}"}
    
    result = poll_future(resp.json().get("request_id"))
    if "error" in result:
        return {"error": result["error"]}
    
    # Optim step
    resp = requests.post(
        f"{BASE_URL}/api/v1/optim_step",
        json={"model_id": model_id, "adam_params": {"learning_rate": 1e-4}},
        headers=get_headers(),
        timeout=60,
    )
    if resp.status_code != 200:
        return {"error": f"optim_step HTTP {resp.status_code}"}
    
    return poll_future(resp.json().get("request_id"))


async def run_session_with_busy_training(model_name, session_prefix, num_steps=3):
    """Run a session with multiple training steps (keeping trainer busy)."""
    model_short = model_name.split("/")[-1]
    print(f"\n  Starting session for {model_short}...")
    
    result = create_session(model_name, session_prefix)
    if "error" in result:
        return {"error": result["error"]}
    
    model_id = result["model_id"]
    print(f"  Created: {model_id}")
    
    # Run multiple training steps
    for i in range(num_steps):
        print(f"  Training step {i+1}/{num_steps}...")
        result = train_step(model_id)
        if "error" in result:
            return {"error": f"Training step {i+1}: {result['error']}"}
    
    print(f"  Session {model_short} completed {num_steps} steps")
    return {"model_id": model_id}


async def test_concurrent_pressure():
    """Test concurrent sessions with resource pressure."""
    print("\n" + "=" * 60)
    print("TEST: CONCURRENT SESSIONS WITH RESOURCE PRESSURE")
    print("=" * 60)
    print(f"Model A: {MODEL_A}")
    print(f"Model B: {MODEL_B}")
    
    print_resources("Initial state")
    
    # Start two sessions concurrently - both will need trainers
    print("\n[1] Starting two sessions CONCURRENTLY...")
    print("    Both sessions will keep their trainers busy with multiple training steps")
    
    start = time.time()
    
    # Run both sessions concurrently
    loop = asyncio.get_event_loop()
    task_a = loop.create_task(run_session_with_busy_training(MODEL_A, "pressure_a", num_steps=3))
    task_b = loop.create_task(run_session_with_busy_training(MODEL_B, "pressure_b", num_steps=3))
    
    results = await asyncio.gather(task_a, task_b, return_exceptions=True)
    
    elapsed = time.time() - start
    print(f"\n[2] Both sessions completed in {elapsed:.1f}s")
    
    # Check results
    all_ok = True
    for i, res in enumerate(results):
        model = MODEL_A if i == 0 else MODEL_B
        model_short = model.split("/")[-1]
        if isinstance(res, Exception):
            print(f"    [{model_short}] Exception: {res}")
            all_ok = False
        elif "error" in res:
            print(f"    [{model_short}] FAILED: {res['error']}")
            all_ok = False
        else:
            print(f"    [{model_short}] OK: {res.get('model_id')}")
    
    print_resources("After concurrent sessions")
    
    return all_ok


async def test_sequential_then_concurrent():
    """Test starting first session, then trying second while first is busy."""
    print("\n" + "=" * 60)
    print("TEST: FIRST SESSION BUSY, SECOND TRIES TO START")
    print("=" * 60)
    
    print_resources("Initial state")
    
    # Create first session
    print("\n[1] Creating first session (Model A)...")
    result_a = create_session(MODEL_A, "seq_a")
    if "error" in result_a:
        print(f"    FAILED: {result_a['error']}")
        return False
    model_id_a = result_a["model_id"]
    print(f"    Created: {model_id_a}")
    
    # Start training on first session (non-blocking) while creating second
    print("\n[2] Starting training on Model A while creating session for Model B...")
    
    async def train_a():
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: train_step(model_id_a))
    
    async def create_b():
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: create_session(MODEL_B, "seq_b"))
    
    start = time.time()
    
    # Start both concurrently
    train_task = asyncio.create_task(train_a())
    create_task = asyncio.create_task(create_b())
    
    train_result, create_result = await asyncio.gather(train_task, create_task)
    
    elapsed = time.time() - start
    print(f"\n[3] Both operations completed in {elapsed:.1f}s")
    
    # Check results
    all_ok = True
    if "error" in train_result:
        print(f"    Model A training: FAILED - {train_result['error']}")
        all_ok = False
    else:
        print("    Model A training: OK")
    
    if "error" in create_result:
        print(f"    Model B creation: FAILED - {create_result['error']}")
        all_ok = False
    else:
        print(f"    Model B creation: OK - {create_result.get('model_id')}")
    
    print_resources("Final state")
    
    return all_ok


async def main():
    print("=" * 60)
    print("ISSUE 2: CONCURRENT SESSIONS WITH RESOURCE PRESSURE")
    print("=" * 60)
    print(f"Base URL: {BASE_URL}")
    
    results = []
    
    # Test 1: Concurrent sessions
    try:
        passed = await test_concurrent_pressure()
        results.append(("Concurrent pressure", passed))
    except Exception as e:
        print(f"Exception: {e}")
        results.append(("Concurrent pressure", False))
    
    # Brief pause
    await asyncio.sleep(3)
    
    # Test 2: Sequential then concurrent
    try:
        passed = await test_sequential_then_concurrent()
        results.append(("Sequential then concurrent", passed))
    except Exception as e:
        print(f"Exception: {e}")
        results.append(("Sequential then concurrent", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        all_passed = all_passed and passed
    
    print(f"\nOVERALL: {'PASS' if all_passed else 'FAIL'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
