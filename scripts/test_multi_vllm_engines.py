#!/usr/bin/env python3
"""Test that each model gets its own vLLM engine.

Verifies the "one vLLM per model" architecture:
1. Create sessions for two different dense models
2. Train each model
3. Save weights (triggers vLLM engine creation)
4. Verify each model has its own vLLM actor
5. Sample from each to verify they work independently
"""

import os
import sys
import time
import uuid

import requests

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")

# Two different dense models (each needs 1 GPU)
MODEL_A = "Qwen/Qwen3-0.6B"
MODEL_B = "Qwen/Qwen2.5-7B-Instruct"


def get_headers():
    return {"Authorization": f"Bearer {API_KEY}"}


def poll_future(request_id: str, timeout: int = 300) -> dict:
    """Poll for async operation result."""
    poll_url = f"{BASE_URL}/api/v1/retrieve_future"
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.post(poll_url, json={"request_id": request_id}, headers=get_headers(), timeout=120)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 408:
            time.sleep(0.5)
            continue
        else:
            resp.raise_for_status()
    raise TimeoutError(f"Operation did not complete within {timeout}s")


def get_resource_pool() -> dict:
    """Get current resource pool status."""
    resp = requests.get(f"{BASE_URL}/api/v1/resource_pool")
    return resp.json()


def create_session(base_model: str, lora_rank: int = 32, lr: float = 1e-4) -> tuple[str, str]:
    """Create training session."""
    session_id = f"multi_vllm_test_{uuid.uuid4().hex[:8]}"
    url = f"{BASE_URL}/api/v1/create_model"
    payload = {
        "session_id": session_id,
        "model_seq_id": 1,
        "base_model": base_model,
        "lora_config": {"rank": lora_rank},
        "learning_rate": lr,
    }
    resp = requests.post(url, json=payload, headers=get_headers(), timeout=300)
    resp.raise_for_status()
    result = poll_future(resp.json().get("request_id"), timeout=300)
    if "error" in result:
        raise RuntimeError(f"Session creation failed: {result['error']}")
    return session_id, result.get("model_id")


def train_step(model_id: str, data: list, lr: float = 1e-4) -> dict:
    """Combined forward_backward + optim_step."""
    url = f"{BASE_URL}/api/v1/train_step"
    payload = {
        "model_id": model_id,
        "forward_backward_input": {"data": data, "loss_fn": "cross_entropy"},
        "adam_params": {"learning_rate": lr, "beta1": 0.9, "beta2": 0.95, "eps": 1e-12},
    }
    resp = requests.post(url, json=payload, headers=get_headers(), timeout=120)
    resp.raise_for_status()
    return poll_future(resp.json().get("request_id"), timeout=300)


def save_weights(model_id: str, name: str = "test") -> dict:
    """Save weights for sampling."""
    url = f"{BASE_URL}/api/v1/save_weights"
    payload = {"model_id": model_id, "name": name}
    resp = requests.post(url, json=payload, headers=get_headers(), timeout=300)
    resp.raise_for_status()
    # vLLM init takes ~80-120s first time (model load + torch.compile + CUDA graph)
    return poll_future(resp.json().get("request_id"), timeout=300)


def sample(model_id: str, prompt_tokens: list, max_tokens: int = 20, temperature: float = 0.0) -> dict:
    """Sample from model."""
    url = f"{BASE_URL}/api/v1/asample"
    payload = {
        "model_id": model_id,
        "prompt": {"chunks": [{"tokens": prompt_tokens, "type": "encoded_text"}]},
        "sampling_params": {"max_tokens": max_tokens, "temperature": temperature},
        "num_samples": 1,
    }
    resp = requests.post(url, json=payload, headers=get_headers(), timeout=120)
    resp.raise_for_status()
    return poll_future(resp.json().get("request_id"), timeout=180)


def make_sft_datum(input_tokens: list, target_tokens: list, loss_mask: list) -> dict:
    """Create SFT training datum."""
    return {
        "model_input": {"chunks": [{"tokens": input_tokens, "type": "encoded_text"}]},
        "loss_fn_inputs": {
            "target_tokens": {"data": target_tokens, "shape": [len(target_tokens)], "dtype": "int64"},
            "loss_mask": {"data": loss_mask, "shape": [len(loss_mask)], "dtype": "float32"},
        },
    }


def prepare_data() -> list:
    """Prepare SFT data using raw token IDs."""
    data = []
    for i in range(4):
        input_tokens = list(range(100, 120))
        target_tokens = list(range(101, 121))
        loss_mask = [0.0] * 10 + [1.0] * 10
        data.append(make_sft_datum(input_tokens, target_tokens, loss_mask))
    return data


def count_vllm_actors(pool: dict) -> dict:
    """Count vLLM actors by model."""
    vllm_actors = {}
    for actor in pool.get("actors", []):
        if actor.get("actor_type") == "vllm":
            name = actor["actor_name"]
            vllm_actors[name] = {
                "gpus": actor["num_gpus"],
                "base_model": actor.get("base_model", "unknown"),
            }
    return vllm_actors


def main():
    print("=" * 70)
    print("Multi-vLLM Engine Test")
    print("=" * 70)
    print(f"Model A: {MODEL_A}")
    print(f"Model B: {MODEL_B}")
    print()

    # Step 1: Check initial state
    print("Step 1: Checking initial resource pool...")
    pool = get_resource_pool()
    print(f"  GPUs used: {pool.get('total_gpus_used', 0)}")
    initial_vllm = count_vllm_actors(pool)
    print(f"  Initial vLLM actors: {len(initial_vllm)}")
    for name, info in initial_vllm.items():
        print(f"    - {name}: {info['gpus']} GPUs")

    # Step 2: Create and train Model A
    print(f"\nStep 2: Creating session for Model A ({MODEL_A})...")
    t0 = time.time()
    try:
        _, model_id_a = create_session(MODEL_A, lora_rank=32)
        print(f"  Session created in {time.time() - t0:.1f}s: {model_id_a}")
    except Exception as e:
        print(f"  FAILED: {e}")
        return 1

    print("  Training (2 iterations)...")
    data = prepare_data()
    for i in range(2):
        result = train_step(model_id_a, data)
        loss = result.get("metrics", {}).get("loss:mean", 0)
        print(f"    Iter {i+1}: loss={loss:.4f}")

    print("  Saving weights...")
    try:
        result_a = save_weights(model_id_a, name="multi_vllm_test_a")
        print(f"    Sampling registered: {result_a.get('sampling_registered', False)}")
    except Exception as e:
        print(f"  Save FAILED: {e}")
        return 1

    # Step 3: Create and train Model B
    print(f"\nStep 3: Creating session for Model B ({MODEL_B})...")
    t0 = time.time()
    try:
        _, model_id_b = create_session(MODEL_B, lora_rank=32)
        print(f"  Session created in {time.time() - t0:.1f}s: {model_id_b}")
    except Exception as e:
        print(f"  FAILED: {e}")
        return 1

    print("  Training (2 iterations)...")
    for i in range(2):
        result = train_step(model_id_b, data)
        loss = result.get("metrics", {}).get("loss:mean", 0)
        print(f"    Iter {i+1}: loss={loss:.4f}")

    print("  Saving weights...")
    try:
        result_b = save_weights(model_id_b, name="multi_vllm_test_b")
        print(f"    Sampling registered: {result_b.get('sampling_registered', False)}")
    except Exception as e:
        print(f"  Save FAILED: {e}")
        return 1

    # Step 4: Check vLLM actors
    print("\nStep 4: Verifying vLLM actors...")
    pool = get_resource_pool()
    final_vllm = count_vllm_actors(pool)
    print(f"  Total vLLM actors: {len(final_vllm)}")

    expected_actor_a = "tinker_vllm_qwen3-0.6b"
    expected_actor_b = "tinker_vllm_qwen2.5-7b-instruct"

    has_actor_a = expected_actor_a in final_vllm
    has_actor_b = expected_actor_b in final_vllm

    for name, info in final_vllm.items():
        marker = ""
        if name == expected_actor_a:
            marker = " <-- Model A"
        elif name == expected_actor_b:
            marker = " <-- Model B"
        print(f"    - {name}: {info['gpus']} GPUs{marker}")

    # Step 5: Sample from both
    print("\nStep 5: Sampling from both models...")
    prompt_tokens = [100, 101, 102, 103, 104]

    print(f"  Model A ({MODEL_A})...")
    try:
        result = sample(model_id_a, prompt_tokens, max_tokens=10)
        tokens = result.get("sequences", [{}])[0].get("tokens", [])[:5]
        print(f"    Generated tokens: {tokens}")
        sample_a_ok = True
    except Exception as e:
        print(f"    FAILED: {e}")
        sample_a_ok = False

    print(f"  Model B ({MODEL_B})...")
    try:
        result = sample(model_id_b, prompt_tokens, max_tokens=10)
        tokens = result.get("sequences", [{}])[0].get("tokens", [])[:5]
        print(f"    Generated tokens: {tokens}")
        sample_b_ok = True
    except Exception as e:
        print(f"    FAILED: {e}")
        sample_b_ok = False

    # Summary
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"vLLM actor for {MODEL_A}: {'FOUND' if has_actor_a else 'MISSING'} ({expected_actor_a})")
    print(f"vLLM actor for {MODEL_B}: {'FOUND' if has_actor_b else 'MISSING'} ({expected_actor_b})")
    print(f"Sampling Model A: {'OK' if sample_a_ok else 'FAILED'}")
    print(f"Sampling Model B: {'OK' if sample_b_ok else 'FAILED'}")

    all_passed = has_actor_a and has_actor_b and sample_a_ok and sample_b_ok
    if all_passed:
        print("\nSUCCESS: One vLLM per model architecture verified")
        return 0
    else:
        print("\nFAILURE: One vLLM per model not working correctly")
        return 1


if __name__ == "__main__":
    sys.exit(main())
