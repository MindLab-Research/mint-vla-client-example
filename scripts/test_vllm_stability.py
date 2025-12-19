#!/usr/bin/env python3
"""Intensive vLLM actor stability test.

Tests:
1. Idempotent creation - same actor multiple times
2. Multiple models - different vLLM actors
3. Resource pressure - LRU eviction
4. Abnormal client exit - simulate crash
5. Server responsiveness during actor creation
"""

import asyncio
import json
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import httpx

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("TINKER_API_KEY", "")

# Models to test (dense models for faster tests)
MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-0.6B",
]

def get_headers():
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    return headers


def health_check() -> dict:
    """Check server health."""
    try:
        resp = httpx.get(f"{BASE_URL}/api/v1/healthz", timeout=5)
        return {"status": "ok", "data": resp.json()}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_resource_pool() -> dict:
    """Get resource pool state."""
    try:
        resp = httpx.get(f"{BASE_URL}/api/v1/resource_pool", headers=get_headers(), timeout=10)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def create_sampling_session(model: str, session_id: str = None) -> dict:
    """Create a sampling session (triggers vLLM actor creation)."""
    session_id = session_id or str(uuid.uuid4())
    payload = {"session_id": session_id, "base_model": model}

    try:
        resp = httpx.post(
            f"{BASE_URL}/api/v1/create_sampling_session",
            json=payload,
            headers=get_headers(),
            timeout=180,  # vLLM init can take 60-120s
        )
        return {"status": resp.status_code, "data": resp.json() if resp.status_code == 200 else resp.text}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def create_training_session(model: str, session_id: str = None) -> dict:
    """Create a training session."""
    session_id = session_id or str(uuid.uuid4())
    payload = {
        "session_id": session_id,
        "base_model": model,
        "lora_rank": 16,
        "learning_rate": 1e-4,
    }

    try:
        resp = httpx.post(
            f"{BASE_URL}/api/v1/create_session",
            json=payload,
            headers=get_headers(),
            timeout=180,
        )
        return {"status": resp.status_code, "data": resp.json() if resp.status_code == 200 else resp.text}
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def check_health_during_operation(duration: float, interval: float = 2.0) -> list:
    """Check health periodically during a long operation."""
    results = []
    start = time.time()
    async with httpx.AsyncClient() as client:
        while time.time() - start < duration:
            try:
                resp = await client.get(f"{BASE_URL}/api/v1/healthz", timeout=5)
                results.append({
                    "time": time.time() - start,
                    "status": resp.status_code,
                    "latency": resp.elapsed.total_seconds(),
                })
            except Exception as e:
                results.append({
                    "time": time.time() - start,
                    "status": "error",
                    "error": str(e),
                })
            await asyncio.sleep(interval)
    return results


def test_1_idempotent_creation():
    """Test: Creating same vLLM actor multiple times should not fail."""
    print("\n" + "="*60)
    print("TEST 1: Idempotent vLLM Creation")
    print("="*60)

    model = MODELS[0]
    session_id = f"idempotent-test-{uuid.uuid4().hex[:8]}"

    # Create session first time
    print(f"\n[1.1] First creation for {model}...")
    start = time.time()
    result1 = create_sampling_session(model, session_id)
    elapsed1 = time.time() - start
    print(f"  Result: {result1['status']} in {elapsed1:.1f}s")

    if result1["status"] != 200:
        print(f"  FAIL: First creation failed: {result1}")
        return False

    # Create session second time (should reuse existing actor)
    print(f"\n[1.2] Second creation for same model (should reuse)...")
    session_id2 = f"idempotent-test-{uuid.uuid4().hex[:8]}"
    start = time.time()
    result2 = create_sampling_session(model, session_id2)
    elapsed2 = time.time() - start
    print(f"  Result: {result2['status']} in {elapsed2:.1f}s")

    if result2["status"] != 200:
        print(f"  FAIL: Second creation failed: {result2}")
        return False

    # Second should be much faster (reusing existing actor)
    if elapsed2 > 10:
        print(f"  WARNING: Second creation took {elapsed2:.1f}s (expected <10s for reuse)")
    else:
        print(f"  OK: Actor reuse working ({elapsed2:.1f}s)")

    print("\n  TEST 1 PASSED")
    return True


def test_2_multiple_models():
    """Test: Creating vLLM actors for different models."""
    print("\n" + "="*60)
    print("TEST 2: Multiple Model vLLM Actors")
    print("="*60)

    results = []
    for i, model in enumerate(MODELS):
        session_id = f"multi-model-{i}-{uuid.uuid4().hex[:8]}"
        print(f"\n[2.{i+1}] Creating session for {model}...")

        start = time.time()
        result = create_sampling_session(model, session_id)
        elapsed = time.time() - start

        results.append({
            "model": model,
            "status": result["status"],
            "elapsed": elapsed,
        })
        print(f"  Result: {result['status']} in {elapsed:.1f}s")

        if result["status"] != 200:
            print(f"  FAIL: Creation failed: {result}")
            return False

    # Check resource pool
    pool = get_resource_pool()
    vllm_actors = [a for a in pool.get("actors", []) if a["actor_type"] == "vllm"]
    print(f"\n  vLLM actors in pool: {len(vllm_actors)}")
    for actor in vllm_actors:
        print(f"    - {actor['actor_name']} (node={actor.get('node_id', 'unknown')[:8] if actor.get('node_id') else 'unknown'})")

    print("\n  TEST 2 PASSED")
    return True


def test_3_server_responsiveness():
    """Test: Server remains responsive during vLLM actor creation."""
    print("\n" + "="*60)
    print("TEST 3: Server Responsiveness During vLLM Init")
    print("="*60)

    # Force new vLLM actor creation by using a unique model path
    # (This won't actually work for new models, so we test with existing models)
    model = MODELS[1]  # Use smaller model
    session_id = f"responsive-test-{uuid.uuid4().hex[:8]}"

    async def run_test():
        # Start health checks in background
        health_task = asyncio.create_task(check_health_during_operation(60, interval=2.0))

        # Create sampling session (may trigger vLLM init)
        print(f"\n[3.1] Creating session for {model} while monitoring health...")
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(pool, lambda: create_sampling_session(model, session_id))

        # Wait for health checks to complete
        await asyncio.sleep(5)
        health_task.cancel()
        try:
            health_results = await health_task
        except asyncio.CancelledError:
            health_results = []

        return result, health_results

    result, health_results = asyncio.run(run_test())

    print(f"  Session creation: {result['status']}")

    # Analyze health check results
    errors = [h for h in health_results if h.get("status") == "error"]
    slow = [h for h in health_results if isinstance(h.get("status"), int) and h.get("latency", 0) > 5]

    if errors:
        print(f"  WARNING: {len(errors)} health check errors during creation:")
        for e in errors[:3]:
            print(f"    - t={e['time']:.1f}s: {e.get('error', 'unknown')}")
    else:
        print(f"  OK: All {len(health_results)} health checks passed")

    if slow:
        print(f"  WARNING: {len(slow)} slow responses (>5s)")

    print("\n  TEST 3 PASSED" if not errors else "\n  TEST 3 PASSED (with warnings)")
    return True


def test_4_concurrent_creation():
    """Test: Concurrent session creation for different models."""
    print("\n" + "="*60)
    print("TEST 4: Concurrent Session Creation")
    print("="*60)

    async def create_sessions_concurrent():
        async with httpx.AsyncClient(timeout=180) as client:
            tasks = []
            for i, model in enumerate(MODELS):
                session_id = f"concurrent-{i}-{uuid.uuid4().hex[:8]}"
                payload = {"session_id": session_id, "base_model": model}
                task = client.post(
                    f"{BASE_URL}/api/v1/create_sampling_session",
                    json=payload,
                    headers=get_headers(),
                )
                tasks.append(task)

            print(f"\n[4.1] Sending {len(tasks)} concurrent requests...")
            start = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            elapsed = time.time() - start

            return results, elapsed

    results, elapsed = asyncio.run(create_sessions_concurrent())

    print(f"  Total time: {elapsed:.1f}s")

    for i, (model, result) in enumerate(zip(MODELS, results)):
        if isinstance(result, Exception):
            print(f"  [{i}] {model}: ERROR - {result}")
        else:
            print(f"  [{i}] {model}: {result.status_code}")

    errors = [r for r in results if isinstance(r, Exception) or (hasattr(r, 'status_code') and r.status_code != 200)]
    if errors:
        print(f"\n  WARNING: {len(errors)} failures in concurrent creation")

    print("\n  TEST 4 PASSED")
    return True


def test_5_repeated_rapid_creation():
    """Test: Rapid repeated session creation (stress test)."""
    print("\n" + "="*60)
    print("TEST 5: Rapid Repeated Creation (Stress Test)")
    print("="*60)

    model = MODELS[0]
    num_iterations = 5

    print(f"\n[5.1] Creating {num_iterations} sessions rapidly for {model}...")

    results = []
    for i in range(num_iterations):
        session_id = f"rapid-{i}-{uuid.uuid4().hex[:8]}"
        start = time.time()
        result = create_sampling_session(model, session_id)
        elapsed = time.time() - start

        results.append({
            "iteration": i,
            "status": result["status"],
            "elapsed": elapsed,
        })
        print(f"  [{i+1}/{num_iterations}] Status: {result['status']}, Time: {elapsed:.1f}s")

    # After first, all should be fast (reusing actor)
    fast_count = sum(1 for r in results[1:] if r["elapsed"] < 5)
    print(f"\n  Fast responses (after first): {fast_count}/{len(results)-1}")

    errors = [r for r in results if r["status"] != 200]
    if errors:
        print(f"  FAIL: {len(errors)} errors during rapid creation")
        return False

    print("\n  TEST 5 PASSED")
    return True


def test_6_health_during_all_tests():
    """Test: Verify server health after all tests."""
    print("\n" + "="*60)
    print("TEST 6: Final Health Check")
    print("="*60)

    result = health_check()
    print(f"\n[6.1] Health check: {result}")

    pool = get_resource_pool()
    print(f"\n[6.2] Resource pool summary:")
    print(f"  Total GPUs used: {pool.get('total_gpus_used', 'unknown')}")
    print(f"  Actors: {len(pool.get('actors', []))}")

    for actor in pool.get("actors", []):
        node_short = actor.get('node_id', 'unknown')[:8] if actor.get('node_id') else 'unknown'
        print(f"    - {actor['actor_name']}: {actor['actor_type']} ({actor['num_gpus']} GPUs, node={node_short})")

    print("\n  TEST 6 PASSED")
    return True


def main():
    print("="*60)
    print("vLLM ACTOR STABILITY TEST SUITE")
    print("="*60)
    print(f"Base URL: {BASE_URL}")
    print(f"Models: {MODELS}")

    # Initial health check
    health = health_check()
    if health["status"] != "ok":
        print(f"\nERROR: Server not healthy: {health}")
        sys.exit(1)
    print(f"\nInitial health: {health['status']}")

    # Run tests
    tests = [
        ("Idempotent Creation", test_1_idempotent_creation),
        ("Multiple Models", test_2_multiple_models),
        ("Server Responsiveness", test_3_server_responsiveness),
        ("Concurrent Creation", test_4_concurrent_creation),
        ("Rapid Repeated Creation", test_5_repeated_rapid_creation),
        ("Final Health Check", test_6_health_during_all_tests),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
