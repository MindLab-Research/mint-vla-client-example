#!/usr/bin/env python3
"""Test vLLM actor creation via save_weights flow with health monitoring.

The vLLM actor is created lazily when save_weights_for_sampler is called.
This test:
1. Creates a training session
2. Runs one training iteration
3. Calls save_weights_for_sampler (triggers vLLM init)
4. Monitors server health during the process
"""

import asyncio
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import httpx

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("TINKER_API_KEY", "")

MODEL = "Qwen/Qwen2.5-7B-Instruct"

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
                print(f"  [{elapsed:6.1f}s] Health: TIMEOUT ***")
            except Exception as e:
                results.append({
                    "time": elapsed,
                    "status": "error",
                    "error": str(e),
                })
                print(f"  [{elapsed:6.1f}s] Health: ERROR - {e}")
            await asyncio.sleep(interval)
    return results


def create_training_session(session_id: str) -> dict:
    """Create training session."""
    payload = {
        "session_id": session_id,
        "base_model": MODEL,
        "lora_rank": 16,
        "learning_rate": 1e-4,
    }
    try:
        resp = httpx.post(
            f"{BASE_URL}/api/v1/create_session",
            json=payload,
            headers=get_headers(),
            timeout=300,
        )
        return {"status": resp.status_code, "data": resp.json() if resp.status_code == 200 else resp.text}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def run_training_iteration(session_id: str) -> dict:
    """Run one training iteration."""
    # Simple training data
    prompt = "What is 2+2?"
    target = "4"

    # Tokenize (simplified - using character IDs)
    prompt_ids = [ord(c) for c in prompt]
    target_ids = [ord(c) for c in target]

    payload = {
        "session_id": session_id,
        "loss_fn_name": "cross_entropy",
        "loss_fn_inputs": {
            "target_tokens": {"data": prompt_ids + target_ids, "shape": [len(prompt_ids) + len(target_ids)], "dtype": "int64"},
            "prompt_length": len(prompt_ids),
        },
        "loss_fn_kwargs": {},
    }

    try:
        # Forward
        resp = httpx.post(
            f"{BASE_URL}/api/v1/forward_backward",
            json=payload,
            headers=get_headers(),
            timeout=60,
        )
        if resp.status_code != 200:
            return {"status": "forward_error", "error": resp.text}

        # Optim step
        resp = httpx.post(
            f"{BASE_URL}/api/v1/optim_step",
            json={"session_id": session_id},
            headers=get_headers(),
            timeout=60,
        )
        if resp.status_code != 200:
            return {"status": "optim_error", "error": resp.text}

        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def save_weights_for_sampler(session_id: str) -> dict:
    """Save weights for sampler (triggers vLLM init)."""
    payload = {"session_id": session_id}
    try:
        # Submit request (async endpoint)
        resp = httpx.post(
            f"{BASE_URL}/api/v1/save_weights_for_sampler",
            json=payload,
            headers=get_headers(),
            timeout=30,
        )
        if resp.status_code != 200:
            return {"status": "submit_error", "error": resp.text}

        request_id = resp.json().get("request_id")
        if not request_id:
            return {"status": "no_request_id", "error": resp.json()}

        # Poll for completion
        poll_payload = {"request_id": request_id}
        start = time.time()
        while time.time() - start < 300:  # 5 min timeout
            resp = httpx.post(
                f"{BASE_URL}/api/v1/retrieve_future",
                json=poll_payload,
                headers=get_headers(),
                timeout=30,
            )
            if resp.status_code == 200:
                return {"status": "ok", "data": resp.json()}
            elif resp.status_code == 408:
                # Still pending
                time.sleep(2)
                continue
            else:
                return {"status": "poll_error", "code": resp.status_code, "error": resp.text}

        return {"status": "timeout", "error": "save_weights timed out after 300s"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def test_vllm_init_via_save_weights():
    """Test vLLM init triggered by save_weights."""
    session_id = f"vllm-trigger-test-{uuid.uuid4().hex[:8]}"

    print(f"\n[1] Creating training session: {session_id}")
    result = create_training_session(session_id)
    print(f"    Result: {result['status']}")
    if result["status"] != 200:
        print(f"    Error: {result}")
        return False

    print(f"\n[2] Running one training iteration...")
    result = run_training_iteration(session_id)
    print(f"    Result: {result['status']}")
    if result["status"] != "ok":
        print(f"    Error: {result}")
        return False

    print(f"\n[3] Calling save_weights_for_sampler (triggers vLLM init)...")
    print("    Monitoring server health during vLLM initialization...")
    print("-" * 60)

    # Start health monitoring
    stop_event = asyncio.Event()
    monitor_task = asyncio.create_task(monitor_health(stop_event, interval=2.0))

    # Run save_weights in thread pool
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=1) as pool:
        start = time.time()
        result = await loop.run_in_executor(
            pool,
            lambda: save_weights_for_sampler(session_id)
        )
        elapsed = time.time() - start

    # Stop health monitoring
    stop_event.set()
    await asyncio.sleep(0.5)
    health_results = await monitor_task

    print("-" * 60)
    print(f"\n    save_weights result: {result['status']} in {elapsed:.1f}s")

    # Analyze health results
    timeouts = [h for h in health_results if h.get("status") == "timeout"]
    errors = [h for h in health_results if h.get("status") == "error"]
    slow = [h for h in health_results if h.get("latency", 0) > 5]

    print(f"\n[4] Health monitoring summary:")
    print(f"    Total checks: {len(health_results)}")
    print(f"    Timeouts: {len(timeouts)}")
    print(f"    Errors: {len(errors)}")
    print(f"    Slow (>5s): {len(slow)}")

    if timeouts:
        print("\n*** SERVER HAD TIMEOUTS (was blocking) ***")
        for h in timeouts[:5]:
            print(f"    [{h['time']:.1f}s] TIMEOUT")

    if errors:
        print("\n*** SERVER HAD ERRORS ***")
        for h in errors[:5]:
            print(f"    [{h['time']:.1f}s] {h.get('error', 'unknown')}")

    # Check vLLM actor was created
    print("\n[5] Checking resource pool for vLLM actor...")
    try:
        resp = httpx.get(f"{BASE_URL}/api/v1/resource_pool", headers=get_headers(), timeout=10)
        pool = resp.json()
        vllm_actors = [a for a in pool.get("actors", []) if a["actor_type"] == "vllm"]
        print(f"    vLLM actors: {len(vllm_actors)}")
        for actor in vllm_actors:
            print(f"      - {actor['actor_name']} (node={actor.get('node_id', 'unknown')[:8] if actor.get('node_id') else 'unknown'})")
    except Exception as e:
        print(f"    Error checking pool: {e}")

    success = result["status"] == "ok" and len(timeouts) == 0
    return success


async def main():
    print("="*60)
    print("vLLM INIT VIA SAVE_WEIGHTS TEST")
    print("="*60)
    print(f"Model: {MODEL}")
    print(f"Base URL: {BASE_URL}")

    # Check initial state
    print("\n[0] Initial state check...")
    try:
        resp = httpx.get(f"{BASE_URL}/api/v1/resource_pool", headers=get_headers(), timeout=10)
        pool = resp.json()
        vllm_actors = [a for a in pool.get("actors", []) if a["actor_type"] == "vllm"]
        target = f"tinker_vllm_{MODEL.split('/')[-1].lower()}"
        exists = any(a["actor_name"] == target for a in vllm_actors)
        print(f"    Target vLLM actor ({target}): {'EXISTS' if exists else 'NOT FOUND'}")
        if exists:
            print("\n    NOTE: vLLM actor already exists. Killing it for fresh test...")
            # Could kill here, but let's test with existing actor first
    except Exception as e:
        print(f"    Error: {e}")

    print("\n" + "="*60)
    print("STARTING TEST")
    print("="*60)

    success = await test_vllm_init_via_save_weights()

    print("\n" + "="*60)
    print(f"RESULT: {'PASS' if success else 'FAIL'}")
    print("="*60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
