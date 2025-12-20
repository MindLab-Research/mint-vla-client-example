#!/usr/bin/env python3
"""Test vLLM actor creation from scratch with health monitoring.

This test verifies that the server remains responsive during vLLM
initialization (which can take 60-120s for MoE models).
"""

import asyncio
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("TINKER_API_KEY", "")

def get_headers():
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    return headers


async def monitor_health(stop_event: asyncio.Event, interval: float = 2.0) -> list:
    """Monitor server health until stop_event is set."""
    results = []
    start = time.time()
    async with httpx.AsyncClient() as client:
        while not stop_event.is_set():
            elapsed = time.time() - start
            try:
                resp = await client.get(f"{BASE_URL}/api/v1/healthz", timeout=10)
                latency = resp.elapsed.total_seconds()
                status = "ok" if resp.status_code == 200 else f"http_{resp.status_code}"
                results.append({
                    "time": elapsed,
                    "status": status,
                    "latency": latency,
                })
                print(f"  [{elapsed:6.1f}s] Health: {status} (latency={latency:.2f}s)")
            except httpx.TimeoutException:
                results.append({
                    "time": elapsed,
                    "status": "timeout",
                    "latency": 10.0,
                })
                print(f"  [{elapsed:6.1f}s] Health: TIMEOUT")
            except Exception as e:
                results.append({
                    "time": elapsed,
                    "status": "error",
                    "error": str(e),
                })
                print(f"  [{elapsed:6.1f}s] Health: ERROR - {e}")
            await asyncio.sleep(interval)
    return results


def create_sampling_session_sync(model: str, session_id: str) -> dict:
    """Create sampling session (blocking)."""
    payload = {"session_id": session_id, "base_model": model}
    try:
        resp = httpx.post(
            f"{BASE_URL}/api/v1/create_sampling_session",
            json=payload,
            headers=get_headers(),
            timeout=300,  # 5 min timeout for vLLM init
        )
        return {"status": resp.status_code, "data": resp.json() if resp.status_code == 200 else resp.text}
    except httpx.TimeoutException:
        return {"status": "timeout", "error": "Request timed out after 300s"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def test_vllm_creation_with_health_monitoring(model: str):
    """Test vLLM creation while monitoring server health."""
    session_id = f"vllm-init-test-{int(time.time())}"

    print(f"\nCreating sampling session for {model}")
    print(f"Session ID: {session_id}")
    print("\nMonitoring server health during vLLM initialization...")
    print("-" * 60)

    # Start health monitoring
    stop_event = asyncio.Event()
    monitor_task = asyncio.create_task(monitor_health(stop_event))

    # Create session in thread pool (blocking operation)
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=1) as pool:
        start = time.time()
        result = await loop.run_in_executor(
            pool,
            lambda: create_sampling_session_sync(model, session_id)
        )
        elapsed = time.time() - start

    # Stop health monitoring
    stop_event.set()
    await asyncio.sleep(0.5)  # Let monitor finish current iteration
    health_results = await monitor_task

    print("-" * 60)
    print(f"\nSession creation result: {result['status']} in {elapsed:.1f}s")

    # Analyze health results
    timeouts = [h for h in health_results if h.get("status") == "timeout"]
    errors = [h for h in health_results if h.get("status") == "error"]
    slow = [h for h in health_results if h.get("latency", 0) > 5]

    print(f"\nHealth monitoring summary:")
    print(f"  Total checks: {len(health_results)}")
    print(f"  Timeouts: {len(timeouts)}")
    print(f"  Errors: {len(errors)}")
    print(f"  Slow (>5s): {len(slow)}")

    if timeouts or errors:
        print("\n*** SERVER WAS UNRESPONSIVE ***")
        for h in (timeouts + errors)[:5]:
            print(f"  [{h['time']:.1f}s] {h.get('status')}: {h.get('error', '')}")
        return False

    print("\n*** SERVER REMAINED RESPONSIVE ***")
    return result["status"] == 200


async def main():
    print("="*60)
    print("vLLM CREATION WITH HEALTH MONITORING TEST")
    print("="*60)

    # Test with Qwen2.5-7B (smaller model, faster init)
    model = "Qwen/Qwen2.5-7B-Instruct"

    # Check current resource pool
    print("\nChecking resource pool before test...")
    try:
        resp = httpx.get(f"{BASE_URL}/api/v1/resource_pool", headers=get_headers(), timeout=10)
        pool = resp.json()
        vllm_actors = [a for a in pool.get("actors", []) if a["actor_type"] == "vllm"]
        model_actor = f"tinker_vllm_{model.split('/')[-1].lower()}"
        exists = any(a["actor_name"] == model_actor for a in vllm_actors)
        print(f"  vLLM actors: {len(vllm_actors)}")
        print(f"  Target actor ({model_actor}): {'EXISTS' if exists else 'NOT FOUND'}")
        if exists:
            print("\n  NOTE: vLLM actor already exists. This test will reuse it.")
            print("  For a true init test, kill the actor first.")
    except Exception as e:
        print(f"  Could not check resource pool: {e}")

    print("\n" + "="*60)
    print("STARTING TEST")
    print("="*60)

    success = await test_vllm_creation_with_health_monitoring(model)

    print("\n" + "="*60)
    print(f"RESULT: {'PASS' if success else 'FAIL'}")
    print("="*60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
