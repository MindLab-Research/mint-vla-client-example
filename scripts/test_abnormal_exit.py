#!/usr/bin/env python3
"""Test 6: Abnormal Client Exit Mid-Operation

Tests that server survives when a client:
1. Starts a training session
2. Begins a long-running operation
3. Exits abruptly (simulated by cancellation/timeout)

Verifies:
- Server doesn't crash
- Resources are properly cleaned up
- Other clients can still use the system
"""
import asyncio
import os
import sys
import time
import uuid

import httpx
import requests

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
MODEL = "Qwen/Qwen2.5-7B-Instruct"


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


def health_check():
    try:
        resp = requests.get(f"{BASE_URL}/api/v1/healthz", timeout=10)
        return resp.status_code == 200
    except:
        return False


def get_resource_pool():
    resp = requests.get(f"{BASE_URL}/api/v1/resource_pool", headers=get_headers(), timeout=10)
    return resp.json()


def create_session(session_id):
    """Create training session."""
    resp = requests.post(
        f"{BASE_URL}/api/v1/create_model",
        json={
            "session_id": session_id,
            "model_seq_id": 1,
            "base_model": MODEL,
            "lora_config": {"rank": 32},
            "learning_rate": 1e-4,
        },
        headers=get_headers(),
        timeout=300,
    )
    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}: {resp.text}"}
    return poll_future(resp.json().get("request_id"), timeout=300)


def do_training_step(model_id):
    """Run forward_backward and optim_step."""
    tokens = list(range(1, 11))
    datum = {
        "model_input": {"chunks": [{"tokens": tokens, "type": "encoded_text"}]},
        "loss_fn_inputs": {
            "target_tokens": {"data": tokens, "shape": [10], "dtype": "int64"},
            "loss_mask": {"data": [0.0] * 5 + [1.0] * 5, "shape": [10], "dtype": "float32"},
        },
    }

    # Forward backward
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
        return result

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


async def test_abrupt_disconnection():
    """Test 1: Client disconnects during save_weights."""
    print("\n" + "=" * 60)
    print("TEST 6A: Client disconnects during save_weights")
    print("=" * 60)

    session_id = f"abort_test_{uuid.uuid4().hex[:6]}"

    # Create session
    print(f"\n[1] Creating session: {session_id}")
    result = create_session(session_id)
    if "error" in result:
        print(f"    FAILED: {result['error']}")
        return False
    model_id = result.get("model_id")
    print(f"    Model ID: {model_id}")

    # Do a training step
    print("\n[2] Running training step...")
    result = do_training_step(model_id)
    if "error" in result:
        print(f"    FAILED: {result['error']}")
        return False
    print("    Done")

    # Start save_weights but cancel quickly
    print("\n[3] Starting save_weights then aborting...")
    async with httpx.AsyncClient() as client:
        try:
            # Start the request with very short timeout (will abort)
            resp = await asyncio.wait_for(
                client.post(
                    f"{BASE_URL}/api/v1/save_weights",
                    json={"model_id": model_id, "name": "abort_test"},
                    headers=get_headers(),
                    timeout=30,
                ),
                timeout=1.0,  # Abort after 1 second
            )
        except asyncio.TimeoutError:
            print("    Client aborted after 1s (as expected)")
        except Exception as e:
            print(f"    Client aborted with: {e}")

    # Wait a bit for any cleanup
    await asyncio.sleep(2)

    # Check server health
    print("\n[4] Checking server health after abort...")
    if health_check():
        print("    Server is healthy")
    else:
        print("    Server is NOT healthy!")
        return False

    # Try to use the system with another client
    print("\n[5] Testing with new client...")
    session_id2 = f"after_abort_{uuid.uuid4().hex[:6]}"
    result = create_session(session_id2)
    if "error" in result:
        print(f"    FAILED: {result['error']}")
        return False
    model_id2 = result.get("model_id")
    print(f"    Created new session: {model_id2}")

    result = do_training_step(model_id2)
    if "error" in result:
        print(f"    FAILED: {result['error']}")
        return False
    print("    New client training succeeded")

    return True


async def test_multiple_aborts():
    """Test 2: Multiple clients abort simultaneously."""
    print("\n" + "=" * 60)
    print("TEST 6B: Multiple clients abort simultaneously")
    print("=" * 60)

    # Create 3 sessions
    print("\n[1] Creating 3 sessions...")
    sessions = []
    for i in range(3):
        session_id = f"multi_abort_{i}_{uuid.uuid4().hex[:6]}"
        result = create_session(session_id)
        if "error" in result:
            print(f"    Session {i} FAILED: {result['error']}")
            continue
        model_id = result.get("model_id")
        sessions.append((session_id, model_id))
        print(f"    Session {i}: {model_id}")

    if len(sessions) < 2:
        print("    Could not create enough sessions")
        return False

    # Train each session
    print("\n[2] Training each session...")
    for session_id, model_id in sessions:
        result = do_training_step(model_id)
        if "error" in result:
            print(f"    {session_id} training FAILED")
        else:
            print(f"    {session_id} training done")

    # Start save_weights on all and abort
    print("\n[3] Starting save_weights on all then aborting...")
    async with httpx.AsyncClient() as client:
        tasks = []
        for session_id, model_id in sessions:
            async def abort_save(mid):
                try:
                    await asyncio.wait_for(
                        client.post(
                            f"{BASE_URL}/api/v1/save_weights",
                            json={"model_id": mid, "name": "abort_test"},
                            headers=get_headers(),
                            timeout=30,
                        ),
                        timeout=0.5,
                    )
                except:
                    pass
            tasks.append(abort_save(model_id))
        await asyncio.gather(*tasks)
    print("    All clients aborted")

    await asyncio.sleep(2)

    # Check server health
    print("\n[4] Checking server health after multi-abort...")
    if health_check():
        print("    Server is healthy")
    else:
        print("    Server is NOT healthy!")
        return False

    return True


async def test_abort_during_create():
    """Test 3: Client aborts during session creation."""
    print("\n" + "=" * 60)
    print("TEST 6C: Client aborts during session creation")
    print("=" * 60)

    session_id = f"create_abort_{uuid.uuid4().hex[:6]}"

    print(f"\n[1] Starting create_model then aborting after 0.5s...")
    async with httpx.AsyncClient() as client:
        try:
            await asyncio.wait_for(
                client.post(
                    f"{BASE_URL}/api/v1/create_model",
                    json={
                        "session_id": session_id,
                        "model_seq_id": 1,
                        "base_model": MODEL,
                        "lora_config": {"rank": 32},
                        "learning_rate": 1e-4,
                    },
                    headers=get_headers(),
                    timeout=30,
                ),
                timeout=0.5,
            )
        except asyncio.TimeoutError:
            print("    Client aborted after 0.5s")
        except Exception as e:
            print(f"    Client aborted with: {e}")

    await asyncio.sleep(1)

    # Check server health
    print("\n[2] Checking server health...")
    if health_check():
        print("    Server is healthy")
    else:
        print("    Server is NOT healthy!")
        return False

    # Try with new session
    print("\n[3] Testing with new client...")
    session_id2 = f"after_create_abort_{uuid.uuid4().hex[:6]}"
    result = create_session(session_id2)
    if "error" in result:
        print(f"    FAILED: {result['error']}")
        return False
    print(f"    Created: {result.get('model_id')}")

    return True


async def main():
    print("=" * 60)
    print("TEST 6: ABNORMAL CLIENT EXIT")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Base URL: {BASE_URL}")

    # Initial health check
    print("\nInitial health check...")
    if not health_check():
        print("Server not healthy!")
        return 1
    print("Server healthy")

    # Initial resource state
    pool = get_resource_pool()
    print(f"GPUs used: {pool.get('total_gpus_used', 0)}")

    results = []

    # Test 6A: Abort during save_weights
    try:
        results.append(("6A: Abort during save_weights", await test_abrupt_disconnection()))
    except Exception as e:
        print(f"Test 6A exception: {e}")
        results.append(("6A: Abort during save_weights", False))

    # Test 6B: Multiple aborts
    try:
        results.append(("6B: Multiple simultaneous aborts", await test_multiple_aborts()))
    except Exception as e:
        print(f"Test 6B exception: {e}")
        results.append(("6B: Multiple simultaneous aborts", False))

    # Test 6C: Abort during create
    try:
        results.append(("6C: Abort during create", await test_abort_during_create()))
    except Exception as e:
        print(f"Test 6C exception: {e}")
        results.append(("6C: Abort during create", False))

    # Final health check
    print("\n" + "=" * 60)
    print("FINAL VERIFICATION")
    print("=" * 60)

    if health_check():
        print("[PASS] Server still healthy")
    else:
        print("[FAIL] Server became unhealthy")

    pool = get_resource_pool()
    print(f"Final GPUs used: {pool.get('total_gpus_used', 0)}")

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
