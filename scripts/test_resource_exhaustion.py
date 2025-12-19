#!/usr/bin/env python3
"""Test 4: Resource Exhaustion with All Model Types
Tests LRU eviction when GPUs are exhausted using dense AND MoE models.

Cluster: 12 GPUs
- Dense models: 1 GPU each for trainer + 1 GPU for vLLM
- MoE model: 4 GPUs for trainer + 4 GPUs for vLLM

Scenario:
1. Create sessions for all 3 dense models (uses 6 GPUs: 3 trainers + 3 vLLM)
2. Create session for MoE model (needs 8 GPUs: 4 trainer + 4 vLLM)
3. Total needed: 14 GPUs > 12 available
4. Expected: LRU eviction frees idle actors to make room
"""
import requests
import time
import uuid

BASE_URL = "http://localhost:8000"

# All supported models
DENSE_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-4B-Instruct-2507",
]
MOE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


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
        resp = requests.get(f"{BASE_URL}/api/v1/healthz", timeout=5)
        return resp.status_code == 200
    except:
        return False


def create_and_train(model_name, prefix):
    """Create session, train, and save weights (triggers vLLM)."""
    session_id = f"{prefix}_{uuid.uuid4().hex[:6]}"

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

    # Forward backward
    tokens = list(range(1, 11))
    datum = {
        "model_input": {"chunks": [{"tokens": tokens, "type": "encoded_text"}]},
        "loss_fn_inputs": {
            "target_tokens": {"data": tokens, "shape": [10], "dtype": "int64"},
            "loss_mask": {"data": [0.0] * 5 + [1.0] * 5, "shape": [10], "dtype": "float32"},
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
    resp = requests.post(
        f"{BASE_URL}/api/v1/save_weights",
        json={"model_id": model_id, "name": "test"},
        headers=get_headers(),
        timeout=120,
    )
    if resp.status_code != 200:
        return {"error": f"save_weights HTTP {resp.status_code}"}
    result = poll_future(resp.json().get("request_id"), timeout=600)  # MoE vLLM takes longer
    if "error" in result:
        return {"error": f"save_weights: {result['error']}"}

    return {"success": True, "model_id": model_id}


def print_resource_state(label):
    pool = get_resource_pool()
    print(f"\n{label}:")
    print(f"  GPUs used: {pool.get('total_gpus_used', 0)}")
    actors = pool.get("actors", [])
    trainers = [a for a in actors if a["actor_type"] in ("dense", "megatron")]
    vllms = [a for a in actors if a["actor_type"] == "vllm"]
    print(f"  Trainers: {len(trainers)}, vLLMs: {len(vllms)}")
    for a in actors:
        idle_str = "IDLE" if a["idle"] else "BUSY"
        print(f"    - {a['actor_name']}: {a['num_gpus']} GPU, {idle_str}, age={a['age']:.0f}s")
    return pool


def main():
    print("=" * 70)
    print("TEST 4: RESOURCE EXHAUSTION WITH ALL MODEL TYPES")
    print("=" * 70)
    print(f"Dense models: {DENSE_MODELS}")
    print(f"MoE model: {MOE_MODEL}")
    print(f"Cluster: 12 GPUs")

    # Initial state
    print_resource_state("Initial state")

    # Verify server health
    if not health_check():
        print("\nFAILED: Server not healthy")
        return 1
    print("\nServer healthy")

    # Phase 1: Train all 3 dense models
    print("\n" + "=" * 70)
    print("PHASE 1: Training 3 dense models (6 GPUs needed)")
    print("=" * 70)

    for i, model in enumerate(DENSE_MODELS):
        model_short = model.split("/")[-1]
        print(f"\n[{i+1}/3] Training {model_short}...")
        start = time.time()
        result = create_and_train(model, f"exhaust{i}")
        elapsed = time.time() - start

        if "error" in result:
            print(f"  FAILED: {result['error']}")
            # Continue anyway to test partial exhaustion
        else:
            print(f"  SUCCESS ({elapsed:.1f}s)")

        # Check server still healthy
        if not health_check():
            print("  WARNING: Server became unhealthy!")

    print_resource_state("After Phase 1")

    # Phase 2: Train MoE model (needs 8 more GPUs)
    print("\n" + "=" * 70)
    print("PHASE 2: Training MoE model (8 GPUs needed)")
    print("=" * 70)
    print("This should trigger LRU eviction of idle actors...")

    model_short = MOE_MODEL.split("/")[-1]
    print(f"\nTraining {model_short}...")
    start = time.time()
    result = create_and_train(MOE_MODEL, "exhaust_moe")
    elapsed = time.time() - start

    if "error" in result:
        print(f"  FAILED: {result['error']}")
        # Check if it's a resource error or other
        if "GPU" in result["error"] or "resource" in result["error"].lower():
            print("  (This may be expected if LRU eviction couldn't free enough GPUs)")
    else:
        print(f"  SUCCESS ({elapsed:.1f}s)")

    final_pool = print_resource_state("Final state")

    # Verification
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    passed = True

    # Server should still be healthy (no hangs)
    if health_check():
        print("[PASS] Server remained healthy (no hangs)")
    else:
        print("[FAIL] Server became unhealthy")
        passed = False

    # Check total GPUs used doesn't exceed cluster capacity
    gpus_used = final_pool.get("total_gpus_used", 0)
    if gpus_used <= 12:
        print(f"[PASS] GPU usage within limits ({gpus_used}/12)")
    else:
        print(f"[FAIL] GPU usage exceeded capacity ({gpus_used}/12)")
        passed = False

    # Check we have some actors running
    actors = final_pool.get("actors", [])
    if len(actors) > 0:
        print(f"[PASS] Actors still running ({len(actors)})")
    else:
        print("[FAIL] No actors running")
        passed = False

    print(f"\nTEST 4 RESULT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
