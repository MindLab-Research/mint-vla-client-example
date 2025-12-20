#!/usr/bin/env python3
"""Test Issue 1: LRU Eviction Mechanism Reliability

Tests LRU eviction with RL sessions (require both trainer + inferencer):
1. Start RL session with model A (creates trainer + vLLM)
2. Do some training and sampling
3. End session (but don't manually kill actors)
4. Start RL session with model B (should trigger LRU eviction if needed)
5. Verify model B works without manual actor killing

Uses different (A, B) combinations for thorough testing.
"""
import os
import sys
import time
import uuid

import requests

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")

# Model combinations to test
MODEL_COMBOS = [
    ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-0.6B"),
    ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B-Instruct-2507"),
    ("Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen2.5-7B-Instruct"),
]


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


def health_check():
    try:
        resp = requests.get(f"{BASE_URL}/api/v1/healthz", timeout=10)
        return resp.status_code == 200
    except:
        return False


def print_resource_state(label):
    pool = get_resource_pool()
    print(f"\n{label}:")
    print(f"  GPUs: {pool.get('total_gpus_used', 0)}/12")
    actors = pool.get("actors", [])
    trainers = [a for a in actors if a["actor_type"] in ("dense", "megatron")]
    vllms = [a for a in actors if a["actor_type"] == "vllm"]
    print(f"  Trainers: {len(trainers)}, vLLMs: {len(vllms)}")
    for a in actors:
        idle = "IDLE" if a["idle"] else "BUSY"
        print(f"    - {a['actor_name']}: {a['num_gpus']}GPU, {idle}")


def run_rl_session(model_name, session_name):
    """Run a simulated RL session (train + sample)."""
    session_id = f"{session_name}_{uuid.uuid4().hex[:6]}"
    model_short = model_name.split("/")[-1]

    print(f"\n  Creating session for {model_short}...")
    start = time.time()

    # Create session
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
        return {"error": f"create_model HTTP {resp.status_code}: {resp.text}"}

    result = poll_future(resp.json().get("request_id"), timeout=300)
    if "error" in result:
        return {"error": f"create_model: {result['error']}"}
    model_id = result.get("model_id")
    print(f"    Model ID: {model_id}")

    # Forward-backward (training)
    tokens = list(range(1, 21))  # 20 tokens
    datum = {
        "model_input": {"chunks": [{"tokens": tokens, "type": "encoded_text"}]},
        "loss_fn_inputs": {
            "target_tokens": {"data": tokens, "shape": [20], "dtype": "int64"},
            "loss_mask": {"data": [0.0] * 10 + [1.0] * 10, "shape": [20], "dtype": "float32"},
        },
    }

    print("    Training...")
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
        return {"error": f"forward_backward: {result['error']}"}

    # Optim step
    resp = requests.post(
        f"{BASE_URL}/api/v1/optim_step",
        json={"model_id": model_id, "adam_params": {"learning_rate": 1e-4}},
        headers=get_headers(),
        timeout=60,
    )
    if resp.status_code != 200:
        return {"error": f"optim_step HTTP {resp.status_code}"}
    result = poll_future(resp.json().get("request_id"))
    if "error" in result:
        return {"error": f"optim_step: {result['error']}"}

    # Save weights (triggers vLLM)
    print("    Saving weights (triggers vLLM)...")
    resp = requests.post(
        f"{BASE_URL}/api/v1/save_weights",
        json={"model_id": model_id, "name": "rl_test"},
        headers=get_headers(),
        timeout=120,
    )
    if resp.status_code != 200:
        return {"error": f"save_weights HTTP {resp.status_code}"}
    result = poll_future(resp.json().get("request_id"), timeout=600)
    if "error" in result:
        return {"error": f"save_weights: {result['error']}"}

    # Asample (inference)
    print("    Sampling...")
    resp = requests.post(
        f"{BASE_URL}/api/v1/asample",
        json={
            "model_id": model_id,
            "num_samples": 1,
            "prompt": {
                "chunks": [{"tokens": [1, 2, 3, 4, 5], "type": "encoded_text"}]
            },
            "sampling_params": {
                "max_tokens": 10,
                "temperature": 0.7,
            },
        },
        headers=get_headers(),
        timeout=120,
    )
    if resp.status_code != 200:
        return {"error": f"asample HTTP {resp.status_code}: {resp.text}"}
    result = poll_future(resp.json().get("request_id"))
    if "error" in result:
        return {"error": f"asample: {result['error']}"}

    elapsed = time.time() - start
    print(f"    RL session completed in {elapsed:.1f}s")

    return {"success": True, "model_id": model_id, "elapsed": elapsed}


def test_lru_eviction_combo(model_a, model_b, combo_num):
    """Test LRU eviction between two models."""
    model_a_short = model_a.split("/")[-1]
    model_b_short = model_b.split("/")[-1]

    print(f"\n{'=' * 60}")
    print(f"COMBO {combo_num}: {model_a_short} -> {model_b_short}")
    print("=" * 60)

    print_resource_state("Before session A")

    # Session A
    print(f"\n[1] RL Session with {model_a_short}")
    result_a = run_rl_session(model_a, f"lru_a_{combo_num}")
    if "error" in result_a:
        print(f"    FAILED: {result_a['error']}")
        return False

    print_resource_state("After session A (before B)")

    # Short wait (simulate session end)
    print("\n[2] Session A ended. Waiting 5s before session B...")
    print("    NOTE: Not manually killing any actors!")
    time.sleep(5)

    # Session B (should trigger LRU eviction if needed)
    print(f"\n[3] RL Session with {model_b_short}")
    result_b = run_rl_session(model_b, f"lru_b_{combo_num}")
    if "error" in result_b:
        print(f"    FAILED: {result_b['error']}")
        # Check if it's a resource error
        if "GPU" in result_b.get("error", "") or "resource" in result_b.get("error", "").lower():
            print("    (LRU eviction may not have worked properly)")
        return False

    print_resource_state("After session B")

    # Verify server health
    if health_check():
        print("\n[PASS] Server healthy after both sessions")
        return True
    else:
        print("\n[FAIL] Server became unhealthy")
        return False


def main():
    print("=" * 60)
    print("TEST ISSUE 1: LRU EVICTION MECHANISM RELIABILITY")
    print("=" * 60)
    print(f"Base URL: {BASE_URL}")
    print("Testing RL sessions with different model combinations")
    print("NO MANUAL ACTOR KILLING between sessions!")

    # Initial state
    print_resource_state("Initial state")

    if not health_check():
        print("\nFAILED: Server not healthy")
        return 1

    results = []
    for i, (model_a, model_b) in enumerate(MODEL_COMBOS, 1):
        try:
            passed = test_lru_eviction_combo(model_a, model_b, i)
            results.append((model_a, model_b, passed))
        except Exception as e:
            print(f"\nException in combo {i}: {e}")
            results.append((model_a, model_b, False))

        # Brief pause between combos
        time.sleep(3)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for model_a, model_b, passed in results:
        a_short = model_a.split("/")[-1]
        b_short = model_b.split("/")[-1]
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {a_short} -> {b_short}")
        all_passed = all_passed and passed

    print_resource_state("Final state")

    print(f"\nOVERALL: {'PASS' if all_passed else 'FAIL'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
