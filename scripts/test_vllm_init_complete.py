#!/usr/bin/env python3
"""Test vLLM actor creation via save_weights flow with health monitoring.

Uses the correct Tinker API format to:
1. Create training session
2. Run one training iteration
3. Save weights (triggers vLLM init)
4. Monitor server health during vLLM initialization
"""

import asyncio
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import httpx
import requests

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")

MODEL = "Qwen/Qwen2.5-7B-Instruct"

def get_headers():
    return {"Authorization": f"Bearer {API_KEY}"}


def poll_future(request_id: str, timeout: int = 300, request_timeout: int = 120) -> dict:
    """Poll for async operation result."""
    poll_url = f"{BASE_URL}/api/v1/retrieve_future"
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.post(poll_url, json={"request_id": request_id},
                            headers=get_headers(), timeout=request_timeout)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 408:
            time.sleep(0.5)
            continue
        else:
            return {"error": f"HTTP {resp.status_code}: {resp.text}"}
    return {"error": f"Operation did not complete within {timeout}s"}


def create_session(base_model: str, lora_rank: int = 32, lr: float = 1e-4) -> tuple[str, str]:
    """Create training session. Returns (session_id, model_id)."""
    session_id = f"vllm_test_{uuid.uuid4().hex[:8]}"
    url = f"{BASE_URL}/api/v1/create_model"
    payload = {
        "session_id": session_id,
        "model_seq_id": 1,
        "base_model": base_model,
        "lora_config": {"rank": lora_rank},
        "learning_rate": lr,
    }
    resp = requests.post(url, json=payload, headers=get_headers(), timeout=300)
    if resp.status_code != 200:
        raise RuntimeError(f"Session creation failed: {resp.status_code} {resp.text}")

    result = poll_future(resp.json().get("request_id"), timeout=300)
    if "error" in result:
        raise RuntimeError(f"Session creation failed: {result['error']}")
    return session_id, result.get("model_id")


def make_sft_datum(input_tokens: list, target_tokens: list, loss_mask: list) -> dict:
    """Create a single SFT training datum."""
    return {
        "model_input": {"chunks": [{"tokens": input_tokens, "type": "encoded_text"}]},
        "loss_fn_inputs": {
            "target_tokens": {"data": target_tokens, "shape": [len(target_tokens)], "dtype": "int64"},
            "loss_mask": {"data": loss_mask, "shape": [len(loss_mask)], "dtype": "float32"},
        },
    }


def forward_backward(model_id: str, data: list, loss_fn: str = "cross_entropy") -> dict:
    """Run forward-backward pass."""
    url = f"{BASE_URL}/api/v1/forward_backward"
    payload = {
        "model_id": model_id,
        "forward_backward_input": {"data": data, "loss_fn": loss_fn},
    }
    resp = requests.post(url, json=payload, headers=get_headers(), timeout=120)
    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}: {resp.text}"}
    return poll_future(resp.json().get("request_id"), timeout=300)


def optim_step(model_id: str, lr: float = 1e-4) -> dict:
    """Run optimizer step."""
    url = f"{BASE_URL}/api/v1/optim_step"
    payload = {
        "model_id": model_id,
        "adam_params": {"learning_rate": lr, "beta1": 0.9, "beta2": 0.95, "eps": 1e-12},
    }
    resp = requests.post(url, json=payload, headers=get_headers(), timeout=60)
    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}: {resp.text}"}
    return poll_future(resp.json().get("request_id"), timeout=60)


async def monitor_health_async(stop_event: asyncio.Event, interval: float = 2.0) -> list:
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
                results.append({"time": elapsed, "status": "timeout", "latency": 10.0})
                print(f"  [{elapsed:6.1f}s] Health: TIMEOUT ***")
            except Exception as e:
                results.append({"time": elapsed, "status": "error", "error": str(e)})
                print(f"  [{elapsed:6.1f}s] Health: ERROR - {e}")
            await asyncio.sleep(interval)
    return results


def save_weights_blocking(model_id: str, name: str = "test") -> dict:
    """Save weights (triggers vLLM init) - blocking."""
    url = f"{BASE_URL}/api/v1/save_weights"
    payload = {"model_id": model_id, "name": name}
    resp = requests.post(url, json=payload, headers=get_headers(), timeout=120)
    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}: {resp.text}"}
    return poll_future(resp.json().get("request_id"), timeout=300, request_timeout=180)


async def test_vllm_init_with_health_monitoring():
    """Test vLLM init triggered by save_weights while monitoring health."""

    # Step 1: Create training session
    print(f"\n[1] Creating training session for {MODEL}...")
    try:
        session_id, model_id = create_session(MODEL, lora_rank=32, lr=1e-4)
        print(f"    Session: {session_id}")
        print(f"    Model ID: {model_id}")
    except Exception as e:
        print(f"    FAILED: {e}")
        return False

    # Step 2: Prepare training data (simple SFT)
    print(f"\n[2] Preparing training data...")
    # Use simple token sequences
    input_tokens = [1, 2, 3, 4, 5]  # "What is"
    target_tokens = [6, 7, 8, 9, 10]  # "the answer"
    loss_mask = [0.0] * len(input_tokens) + [1.0] * len(target_tokens)
    full_tokens = input_tokens + target_tokens
    datum = make_sft_datum(full_tokens, full_tokens, loss_mask)
    print(f"    Data prepared: {len(full_tokens)} tokens")

    # Step 3: Run forward_backward
    print(f"\n[3] Running forward_backward...")
    start = time.time()
    result = forward_backward(model_id, [datum])
    elapsed = time.time() - start
    if "error" in result:
        print(f"    FAILED: {result['error']}")
        return False
    loss = result.get("output", {}).get("metrics", {}).get("loss", "unknown")
    print(f"    Loss: {loss} (in {elapsed:.1f}s)")

    # Step 4: Run optim_step
    print(f"\n[4] Running optim_step...")
    start = time.time()
    result = optim_step(model_id, lr=1e-4)
    elapsed = time.time() - start
    if "error" in result:
        print(f"    FAILED: {result['error']}")
        return False
    print(f"    Done (in {elapsed:.1f}s)")

    # Step 5: Save weights (triggers vLLM init) while monitoring health
    print(f"\n[5] Calling save_weights (triggers vLLM init)...")
    print("    Monitoring server health during vLLM initialization...")
    print("-" * 60)

    stop_event = asyncio.Event()
    monitor_task = asyncio.create_task(monitor_health_async(stop_event, interval=2.0))

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=1) as pool:
        start = time.time()
        result = await loop.run_in_executor(
            pool,
            lambda: save_weights_blocking(model_id, "test")
        )
        elapsed = time.time() - start

    stop_event.set()
    await asyncio.sleep(0.5)
    health_results = await monitor_task

    print("-" * 60)
    if "error" in result:
        print(f"\n    save_weights FAILED: {result['error']}")
    else:
        print(f"\n    save_weights succeeded in {elapsed:.1f}s")
        path = result.get("checkpoint_path", "unknown")
        print(f"    Path: {path}")

    # Step 6: Analyze health results
    timeouts = [h for h in health_results if h.get("status") == "timeout"]
    errors = [h for h in health_results if h.get("status") == "error"]
    slow = [h for h in health_results if h.get("latency", 0) > 5]

    print(f"\n[6] Health monitoring summary:")
    print(f"    Total checks: {len(health_results)}")
    print(f"    Timeouts: {len(timeouts)}")
    print(f"    Errors: {len(errors)}")
    print(f"    Slow (>5s): {len(slow)}")

    if timeouts:
        print("\n*** SERVER HAD TIMEOUTS (was blocking) ***")
        for h in timeouts[:5]:
            print(f"    [{h['time']:.1f}s] TIMEOUT")
        return False

    if errors:
        print("\n*** SERVER HAD ERRORS ***")
        for h in errors[:5]:
            print(f"    [{h['time']:.1f}s] {h.get('error', 'unknown')}")

    # Step 7: Check vLLM actor was created
    print("\n[7] Checking resource pool for vLLM actor...")
    try:
        resp = requests.get(f"{BASE_URL}/api/v1/resource_pool", headers=get_headers(), timeout=10)
        pool = resp.json()
        vllm_actors = [a for a in pool.get("actors", []) if a["actor_type"] == "vllm"]
        print(f"    vLLM actors: {len(vllm_actors)}")
        for actor in vllm_actors:
            node = actor.get('node_id', 'unknown')
            node_short = node[:8] if node else 'unknown'
            print(f"      - {actor['actor_name']} (node={node_short})")
    except Exception as e:
        print(f"    Error checking pool: {e}")

    print("\n*** SERVER REMAINED RESPONSIVE ***")
    return "error" not in result


async def main():
    print("="*60)
    print("vLLM INIT VIA SAVE_WEIGHTS TEST")
    print("="*60)
    print(f"Model: {MODEL}")
    print(f"Base URL: {BASE_URL}")

    # Check initial vLLM state
    print("\n[0] Checking initial vLLM actor state...")
    try:
        resp = requests.get(f"{BASE_URL}/api/v1/resource_pool", headers=get_headers(), timeout=10)
        pool = resp.json()
        vllm_actors = [a for a in pool.get("actors", []) if a["actor_type"] == "vllm"]
        target = f"tinker_vllm_{MODEL.split('/')[-1].lower()}"
        exists = any(a["actor_name"] == target for a in vllm_actors)
        print(f"    vLLM actors: {len(vllm_actors)}")
        print(f"    Target ({target}): {'EXISTS' if exists else 'NOT FOUND'}")
        if exists:
            print("\n    NOTE: vLLM actor already exists.")
            print("    This test will verify reuse. For fresh init test, kill vLLM first.")
    except Exception as e:
        print(f"    Error: {e}")

    print("\n" + "="*60)
    print("STARTING TEST")
    print("="*60)

    success = await test_vllm_init_with_health_monitoring()

    print("\n" + "="*60)
    print(f"RESULT: {'PASS' if success else 'FAIL'}")
    print("="*60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
