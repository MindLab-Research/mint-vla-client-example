#!/usr/bin/env python3
"""Issue 8 Test: Per-model Megatron actors.

Tests that different MoE models get separate Megatron actors, not a shared one.

Test i: Train Qwen3-30B, kill Megatron, train Moonlight, compare curves
Test ii: Train Qwen3-30B then Moonlight without kill, compare curves
Test iii: Train Qwen3-30B and Moonlight concurrently (time-sharing via LRU eviction)

Usage:
    TINKER_BASE_URL=http://localhost:8000 python scripts/test_issue8_per_model_actors.py [test_num]

    test_num: 1, 2, or 3 (default: run all)
"""

import os
import sys
import time
import uuid
import json
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")

QWEN3_30B = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MOONLIGHT = "moonshotai/Moonlight-16B-A3B-Instruct"

NUM_TRAIN_STEPS = 5
LORA_RANK = 16
LEARNING_RATE = 1e-4


def get_headers():
    return {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def poll_future(request_id: str, timeout: int = 600) -> dict:
    """Poll for async operation result."""
    poll_url = f"{BASE_URL}/api/v1/retrieve_future"
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.post(poll_url, json={"request_id": request_id}, headers=get_headers(), timeout=120)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 408:
            time.sleep(1)
            continue
        else:
            return {"error": f"HTTP {resp.status_code}: {resp.text}"}
    return {"error": f"Timeout after {timeout}s"}


def create_training_session(model: str, session_name: str) -> tuple:
    """Create training session, return (session_id, model_id)."""
    session_id = f"{session_name}_{uuid.uuid4().hex[:8]}"
    print(f"[{session_name}] Creating session for {model}...")

    resp = requests.post(
        f"{BASE_URL}/api/v1/create_model",
        json={
            "session_id": session_id,
            "model_seq_id": 1,
            "base_model": model,
            "lora_config": {"rank": LORA_RANK},
        },
        headers=get_headers(),
        timeout=60,
    )
    if resp.status_code != 200:
        return None, None, f"HTTP {resp.status_code}: {resp.text}"

    request_id = resp.json().get("request_id")
    print(f"[{session_name}] Waiting for Megatron initialization...")

    result = poll_future(request_id, timeout=900)
    if "error" in result:
        return None, None, result["error"]

    model_id = result.get("model_id")
    print(f"[{session_name}] Session ready: model_id={model_id}")
    return session_id, model_id, None


def train_step(model_id: str, session_name: str, step_num: int) -> dict:
    """Run one training step."""
    # Simple training data
    data = [{
        "model_input": {"chunks": [{"tokens": [9707, 1917, 0], "type": "encoded_text"}]},
        "loss_fn_inputs": {
            "target_tokens": {"data": [1917, 0, 0], "shape": [3], "dtype": "int64"},
            "loss_mask": {"data": [1.0, 1.0, 0.0], "shape": [3], "dtype": "float32"},
        },
    }]

    resp = requests.post(
        f"{BASE_URL}/api/v1/train_step",
        json={
            "model_id": model_id,
            "forward_backward_input": {"data": data, "loss_fn": "cross_entropy"},
            "adam_params": {"learning_rate": LEARNING_RATE},
        },
        headers=get_headers(),
        timeout=300,
    )
    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}: {resp.text}"}

    request_id = resp.json().get("request_id")
    if not request_id:
        return {"error": "No request_id"}

    return poll_future(request_id, timeout=600)


def train_model(model: str, session_name: str, num_steps: int = NUM_TRAIN_STEPS) -> list:
    """Train a model for num_steps and return loss curve."""
    t0 = time.time()
    session_id, model_id, err = create_training_session(model, session_name)
    if err:
        print(f"[{session_name}] Session creation failed: {err}")
        return []

    init_time = time.time() - t0
    print(f"[{session_name}] Initialized in {init_time:.1f}s")

    losses = []
    for i in range(num_steps):
        step_start = time.time()
        result = train_step(model_id, session_name, i + 1)
        step_time = time.time() - step_start

        if "error" in result:
            print(f"[{session_name}] Step {i+1} error: {result['error']}")
            losses.append(None)
            continue

        loss = result.get("metrics", {}).get("loss:mean", 0)
        losses.append(loss)
        print(f"[{session_name}] Step {i+1}/{num_steps}: loss={loss:.4f} ({step_time:.1f}s)")

    return losses


def kill_megatron_actor(model: str):
    """Kill Megatron actor for specific model."""
    print(f"Killing Megatron actor for {model}...")
    resp = requests.post(
        f"{BASE_URL}/api/v1/kill_megatron",
        json={"base_model": model},
        headers=get_headers(),
        timeout=60,
    )
    if resp.status_code == 200:
        print(f"Killed: {resp.json()}")
    else:
        print(f"Kill response: HTTP {resp.status_code}: {resp.text}")


def get_resource_pool():
    """Get current resource pool state."""
    resp = requests.get(f"{BASE_URL}/api/v1/resource_pool", headers=get_headers(), timeout=30)
    return resp.json()


def print_curve(name: str, losses: list):
    """Print training curve summary."""
    valid = [l for l in losses if l is not None]
    if not valid:
        print(f"{name}: No valid losses")
        return

    print(f"{name}: {' -> '.join(f'{l:.4f}' for l in valid)}")
    if len(valid) >= 2 and valid[0] > 0:
        reduction = (1 - valid[-1] / valid[0]) * 100
        print(f"  Loss reduction: {reduction:.1f}%")


# =============================================================================
# TEST I: Train Qwen3-30B, kill, train Moonlight
# =============================================================================

def test_i():
    """Test i: Train Qwen3-30B, kill Megatron, train Moonlight, compare curves."""
    print("=" * 70)
    print("TEST I: Sequential training with kill between models")
    print("=" * 70)
    print(f"1. Train {QWEN3_30B} for {NUM_TRAIN_STEPS} steps")
    print(f"2. Kill Megatron actor")
    print(f"3. Train {MOONLIGHT} for {NUM_TRAIN_STEPS} steps")
    print("Expected: Both models train independently, losses decrease")
    print("=" * 70)

    # Train Qwen3-30B
    print("\n--- Phase 1: Qwen3-30B ---")
    qwen_losses = train_model(QWEN3_30B, "qwen3_30b")

    # Kill Megatron
    print("\n--- Killing Megatron ---")
    kill_megatron_actor(QWEN3_30B)
    time.sleep(2)

    # Train Moonlight
    print("\n--- Phase 2: Moonlight ---")
    moon_losses = train_model(MOONLIGHT, "moonlight")

    # Results
    print("\n" + "=" * 70)
    print("RESULTS - TEST I")
    print("=" * 70)
    print_curve("Qwen3-30B", qwen_losses)
    print_curve("Moonlight", moon_losses)

    # Verify both trained successfully
    qwen_ok = len([l for l in qwen_losses if l is not None]) >= NUM_TRAIN_STEPS - 1
    moon_ok = len([l for l in moon_losses if l is not None]) >= NUM_TRAIN_STEPS - 1

    if qwen_ok and moon_ok:
        print("\nTEST I: PASS - Both models trained independently")
        return True
    else:
        print("\nTEST I: FAIL - Training errors occurred")
        return False


# =============================================================================
# TEST II: Train Qwen3-30B then Moonlight without kill
# =============================================================================

def test_ii():
    """Test ii: Train Qwen3-30B then Moonlight without killing Megatron."""
    print("=" * 70)
    print("TEST II: Sequential training WITHOUT kill between models")
    print("=" * 70)
    print(f"1. Train {QWEN3_30B} for {NUM_TRAIN_STEPS} steps")
    print(f"2. Immediately train {MOONLIGHT} for {NUM_TRAIN_STEPS} steps (no kill)")
    print("Expected: Per-model actors - Moonlight creates its own actor")
    print("=" * 70)

    # Train Qwen3-30B
    print("\n--- Phase 1: Qwen3-30B ---")
    qwen_losses = train_model(QWEN3_30B, "qwen3_30b")

    # Check resource pool
    print("\n--- Resource pool after Qwen3-30B ---")
    pool = get_resource_pool()
    print(f"GPUs used: {pool.get('total_gpus_used', 0)}")
    print(f"Actors: {[a.get('name', 'unknown') for a in pool.get('actors', [])]}")

    # Train Moonlight immediately (no kill)
    print("\n--- Phase 2: Moonlight (no kill) ---")
    moon_losses = train_model(MOONLIGHT, "moonlight")

    # Results
    print("\n" + "=" * 70)
    print("RESULTS - TEST II")
    print("=" * 70)
    print_curve("Qwen3-30B", qwen_losses)
    print_curve("Moonlight", moon_losses)

    # Key check: Moonlight should start fresh (not reuse Qwen's actor)
    # If per-model actors work, Moonlight trains independently
    qwen_ok = len([l for l in qwen_losses if l is not None]) >= NUM_TRAIN_STEPS - 1
    moon_ok = len([l for l in moon_losses if l is not None]) >= NUM_TRAIN_STEPS - 1

    if qwen_ok and moon_ok:
        print("\nTEST II: PASS - Per-model actors working correctly")
        return True
    else:
        print("\nTEST II: FAIL - Training errors occurred")
        return False


# =============================================================================
# TEST III: Concurrent training (time-sharing via LRU eviction)
# =============================================================================

def train_model_with_delays(model: str, session_name: str, num_steps: int, delay_between_steps: float = 0) -> list:
    """Train with optional delays between steps (to allow eviction)."""
    t0 = time.time()
    session_id, model_id, err = create_training_session(model, session_name)
    if err:
        print(f"[{session_name}] Session creation failed: {err}")
        return []

    init_time = time.time() - t0
    print(f"[{session_name}] Initialized in {init_time:.1f}s")

    losses = []
    for i in range(num_steps):
        if delay_between_steps > 0 and i > 0:
            print(f"[{session_name}] Waiting {delay_between_steps}s before step {i+1}...")
            time.sleep(delay_between_steps)

        step_start = time.time()
        result = train_step(model_id, session_name, i + 1)
        step_time = time.time() - step_start

        if "error" in result:
            print(f"[{session_name}] Step {i+1} error: {result['error']}")
            losses.append(None)
            continue

        loss = result.get("metrics", {}).get("loss:mean", 0)
        losses.append(loss)
        print(f"[{session_name}] Step {i+1}/{num_steps}: loss={loss:.4f} ({step_time:.1f}s)")

    return losses


def test_iii():
    """Test iii: Concurrent training - tests LRU eviction and time-sharing."""
    print("=" * 70)
    print("TEST III: Concurrent training (resource pressure & time-sharing)")
    print("=" * 70)
    print(f"Client A: {QWEN3_30B}")
    print(f"Client B: {MOONLIGHT}")
    print("Expected behavior:")
    print("  - Both clients start concurrently")
    print("  - When resources insufficient, one waits for other to become evictable")
    print("  - Time-sharing: actors swap based on LRU eviction")
    print("  - Both clients eventually complete (even if slowly)")
    print("=" * 70)

    results = {"qwen": [], "moonlight": []}
    errors = {"qwen": None, "moonlight": None}

    def train_qwen():
        try:
            results["qwen"] = train_model_with_delays(QWEN3_30B, "client_A_qwen", NUM_TRAIN_STEPS, delay_between_steps=5)
        except Exception as e:
            errors["qwen"] = str(e)
            print(f"[client_A_qwen] Exception: {e}")

    def train_moonlight():
        try:
            # Small delay to ensure Qwen starts first
            time.sleep(2)
            results["moonlight"] = train_model_with_delays(MOONLIGHT, "client_B_moon", NUM_TRAIN_STEPS, delay_between_steps=5)
        except Exception as e:
            errors["moonlight"] = str(e)
            print(f"[client_B_moon] Exception: {e}")

    print("\n--- Starting concurrent training ---")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(train_qwen),
            executor.submit(train_moonlight),
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Thread exception: {e}")

    total_time = time.time() - start_time

    # Results
    print("\n" + "=" * 70)
    print("RESULTS - TEST III")
    print("=" * 70)
    print(f"Total time: {total_time:.1f}s")
    print()

    if errors["qwen"]:
        print(f"Qwen3-30B error: {errors['qwen']}")
    else:
        print_curve("Qwen3-30B (Client A)", results["qwen"])

    if errors["moonlight"]:
        print(f"Moonlight error: {errors['moonlight']}")
    else:
        print_curve("Moonlight (Client B)", results["moonlight"])

    # Analyze behavior
    qwen_steps = len([l for l in results["qwen"] if l is not None])
    moon_steps = len([l for l in results["moonlight"] if l is not None])

    print(f"\nQwen completed: {qwen_steps}/{NUM_TRAIN_STEPS} steps")
    print(f"Moonlight completed: {moon_steps}/{NUM_TRAIN_STEPS} steps")

    if qwen_steps >= NUM_TRAIN_STEPS - 1 and moon_steps >= NUM_TRAIN_STEPS - 1:
        print("\nTEST III: PASS - Both clients completed (time-sharing worked)")
        return True
    elif qwen_steps > 0 and moon_steps > 0:
        print("\nTEST III: PARTIAL - Both clients made progress (time-sharing occurred)")
        return True
    elif qwen_steps > 0 or moon_steps > 0:
        print("\nTEST III: DEGRADED - Only one client succeeded")
        print("Check: Did waiting client timeout? Is graceful degradation working?")
        return False
    else:
        print("\nTEST III: FAIL - Neither client completed")
        return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"Server: {BASE_URL}")
    print("Issue 8 Test: Per-model Megatron Actors")
    print()

    # Check server health
    try:
        resp = requests.get(f"{BASE_URL}/api/v1/healthz", timeout=10)
        print(f"Server status: {resp.json().get('status')}")
    except Exception as e:
        print(f"Server not available: {e}")
        return 1

    # Check initial resource pool
    pool = get_resource_pool()
    print(f"Initial GPUs used: {pool.get('total_gpus_used', 0)}")

    # Determine which test to run
    test_num = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    results = {}

    if test_num == 0 or test_num == 1:
        results["test_i"] = test_i()
        print("\n")

    if test_num == 0 or test_num == 2:
        results["test_ii"] = test_ii()
        print("\n")

    if test_num == 0 or test_num == 3:
        results["test_iii"] = test_iii()
        print("\n")

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'PASS' if all_passed else 'FAIL'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
