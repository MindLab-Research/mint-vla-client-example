#!/usr/bin/env python3
"""Simple test for MoE expert LoRA extraction (no tokenizer needed locally).

Verifies that MLP modules are included in LoRA state dict after MLP filter removal.
Uses raw token IDs for training data.
"""

import os
import sys
import time
import uuid

import requests

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")

# MoE model
MOE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


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


def create_session(base_model: str, lora_rank: int = 32, lr: float = 1e-4) -> tuple[str, str]:
    """Create training session."""
    session_id = f"moe_expert_lora_test_{uuid.uuid4().hex[:8]}"
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


def train_step(model_id: str, data: list, lr: float = 1e-4, loss_fn: str = "cross_entropy") -> dict:
    """Combined forward_backward + optim_step."""
    url = f"{BASE_URL}/api/v1/train_step"
    payload = {
        "model_id": model_id,
        "forward_backward_input": {"data": data, "loss_fn": loss_fn},
        "adam_params": {"learning_rate": lr, "beta1": 0.9, "beta2": 0.95, "eps": 1e-12},
    }
    resp = requests.post(url, json=payload, headers=get_headers(), timeout=120)
    resp.raise_for_status()
    return poll_future(resp.json().get("request_id"), timeout=300)


def save_weights(model_id: str, name: str = "test") -> dict:
    """Save weights for sampling."""
    url = f"{BASE_URL}/api/v1/save_weights"
    payload = {"model_id": model_id, "name": name}
    resp = requests.post(url, json=payload, headers=get_headers(), timeout=120)
    resp.raise_for_status()
    # MoE vLLM init takes ~2-3 minutes (model load + torch.compile + CUDA graph capture)
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
    """Prepare SFT data using raw token IDs (no tokenizer needed)."""
    # Use simple token patterns - actual content doesn't matter for testing LoRA extraction
    # Just need enough data to trigger training
    data = []
    for i in range(6):
        # Create synthetic token sequences
        input_tokens = list(range(100, 120))  # 20 tokens
        target_tokens = list(range(101, 121))  # Shifted by 1
        loss_mask = [0.0] * 10 + [1.0] * 10  # Only train on last 10 tokens
        data.append(make_sft_datum(input_tokens, target_tokens, loss_mask))
    return data


def check_resource_pool():
    """Check resource pool status."""
    resp = requests.get(f"{BASE_URL}/api/v1/resource_pool")
    return resp.json()


def kill_megatron_actor() -> bool:
    """Kill Megatron actor via HTTP API."""
    try:
        resp = requests.post(f"{BASE_URL}/api/v1/kill_megatron", headers=get_headers())
        result = resp.json()
        print(f"Kill Megatron: {result.get('message', 'N/A')}")
        return result.get("killed", False)
    except Exception as e:
        print(f"Kill Megatron error: {e}")
        return False


def main():
    print("=" * 70)
    print("MoE Expert LoRA Extraction Test")
    print("=" * 70)
    print(f"Model: {MOE_MODEL}")
    print()

    # Step 1: Check resource pool and clean up
    print("Step 1: Checking resource pool...")
    pool = check_resource_pool()
    print(f"Current GPUs used: {pool.get('total_gpus_used', 0)}")
    for actor in pool.get("actors", []):
        print(f"  - {actor['actor_name']}: {actor['num_gpus']} GPUs, idle={actor.get('idle', False)}")

    # Kill Megatron actor to get fresh state with new code
    print("\nKilling any existing Megatron actor...")
    kill_megatron_actor()

    time.sleep(2)
    pool = check_resource_pool()
    print(f"After cleanup: {pool.get('total_gpus_used', 0)} GPUs")

    # Step 2: Create session
    print(f"\nStep 2: Creating MoE training session...")
    t0 = time.time()
    try:
        session_id, model_id = create_session(MOE_MODEL, lora_rank=32, lr=1e-4)
        print(f"Session created in {time.time() - t0:.1f}s: {model_id}")
    except Exception as e:
        print(f"Session creation FAILED: {e}")
        return 1

    # Step 3: Train (3 iterations to trigger LoRA weight updates)
    print(f"\nStep 3: Training (3 iterations)...")
    data = prepare_data()
    losses = []

    for i in range(3):
        t0 = time.time()
        result = train_step(model_id, data, lr=1e-4)
        loss = result.get("metrics", {}).get("loss:mean", 0)
        losses.append(loss)
        print(f"  Iteration {i+1}: loss={loss:.4f}, time={time.time() - t0:.1f}s")

    # Step 4: Save weights and check MLP modules
    print(f"\nStep 4: Saving weights...")
    t0 = time.time()
    try:
        result = save_weights(model_id, name="moe_expert_test")
        print(f"Weights saved in {time.time() - t0:.1f}s")
        sampling_registered = result.get("sampling_registered", False)
        print(f"  Sampling registered: {sampling_registered}")
    except Exception as e:
        print(f"Save weights FAILED: {e}")
        return 1

    # Check state_dict_keys for MLP modules
    state_dict_keys = result.get("state_dict_keys", [])
    if not state_dict_keys:
        print("WARNING: No state_dict_keys returned!")
        return 1

    # Categorize keys
    mlp_keys = []
    attn_keys = []
    other_keys = []

    for key in state_dict_keys:
        key_lower = key.lower()
        if any(p in key_lower for p in ["mlp", "gate", "down_proj", "up_proj", "linear_fc1", "linear_fc2"]):
            mlp_keys.append(key)
        elif any(p in key_lower for p in ["attn", "q_proj", "k_proj", "v_proj", "o_proj", "linear_qkv", "linear_proj"]):
            attn_keys.append(key)
        else:
            other_keys.append(key)

    print(f"\nLoRA state_dict analysis:")
    print(f"  Total keys: {len(state_dict_keys)}")
    print(f"  MLP keys: {len(mlp_keys)}")
    print(f"  Attention keys: {len(attn_keys)}")
    print(f"  Other keys: {len(other_keys)}")

    if mlp_keys:
        print(f"\n  Sample MLP keys:")
        for k in mlp_keys[:5]:
            print(f"    - {k}")

    if attn_keys:
        print(f"\n  Sample attention keys:")
        for k in attn_keys[:5]:
            print(f"    - {k}")

    # Step 5: Sample (verifies vLLM can load expert LoRA)
    print(f"\nStep 5: Sampling from trained model...")
    # Use simple prompt tokens
    prompt_tokens = [100, 101, 102, 103, 104]

    try:
        t0 = time.time()
        result = sample(model_id, prompt_tokens, max_tokens=10, temperature=0.0)
        print(f"Sampling completed in {time.time() - t0:.1f}s")

        sequences = result.get("sequences", [])
        if sequences:
            tokens = sequences[0].get("tokens", [])
            print(f"  Generated tokens: {tokens[:10]}...")
    except Exception as e:
        print(f"Sampling FAILED: {e}")
        # Don't fail the test on sampling - vLLM expert LoRA is the question
        print("  (Sampling failure may indicate vLLM expert LoRA issue)")

    # Summary
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"Training: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"LoRA state_dict: {len(mlp_keys)} MLP, {len(attn_keys)} attention")

    if len(mlp_keys) > 0:
        print("\n✓ SUCCESS: MLP modules found in LoRA state_dict")
        print("  MLP filter removal is working correctly")
        return 0
    else:
        print("\n✗ FAILURE: No MLP modules in LoRA state_dict")
        print("  MLP filter may still be active or extraction failing")
        return 1


if __name__ == "__main__":
    sys.exit(main())
