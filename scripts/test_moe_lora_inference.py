#!/usr/bin/env python3
"""Test MoE LoRA inference with TP=4 configuration.

Verifies that vLLM can load FusedMoE LoRA weights when using
tensor parallelism (TP=4) instead of expert parallelism (DP=4).

vLLM 0.12.0 supports FusedMoE LoRA but NOT with Expert Parallelism.
"""

import os
import sys
import time
import uuid

import requests

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")

MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


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


def create_session(lora_rank: int = 32, lr: float = 1e-4) -> tuple[str, str]:
    """Create MoE training session."""
    session_id = f"moe_lora_test_{uuid.uuid4().hex[:8]}"
    url = f"{BASE_URL}/api/v1/create_model"
    payload = {
        "session_id": session_id,
        "model_seq_id": 1,
        "base_model": MODEL,
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
    """Single training step."""
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
    """Save weights for sampling (triggers vLLM LoRA loading)."""
    url = f"{BASE_URL}/api/v1/save_weights"
    payload = {"model_id": model_id, "name": name}
    resp = requests.post(url, json=payload, headers=get_headers(), timeout=300)
    resp.raise_for_status()
    # vLLM init + LoRA load can take 120-180s
    return poll_future(resp.json().get("request_id"), timeout=300)


def sample(model_id: str, prompt_tokens: list, max_tokens: int = 20, temperature: float = 0.0) -> dict:
    """Sample from model with LoRA."""
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
    """Prepare SFT data."""
    data = []
    for i in range(4):
        input_tokens = list(range(100, 120))
        target_tokens = list(range(101, 121))
        loss_mask = [0.0] * 10 + [1.0] * 10
        data.append(make_sft_datum(input_tokens, target_tokens, loss_mask))
    return data


def main():
    print("=" * 70)
    print("MoE LoRA Inference Test (TP=4, DP=1)")
    print("=" * 70)
    print(f"Model: {MODEL}")
    print(f"Config: TP=4, DP=1 (no Expert Parallelism)")
    print()

    # Step 1: Create session
    print("Step 1: Creating MoE training session...")
    t0 = time.time()
    try:
        session_id, model_id = create_session(lora_rank=32)
        print(f"  Created in {time.time() - t0:.1f}s: {model_id}")
    except Exception as e:
        print(f"  FAILED: {e}")
        return 1

    # Step 2: Train
    print("\nStep 2: Training (3 iterations)...")
    data = prepare_data()
    losses = []
    for i in range(3):
        result = train_step(model_id, data)
        loss = result.get("metrics", {}).get("loss:mean", 0)
        losses.append(loss)
        print(f"  Iter {i+1}: loss={loss:.4f}")

    # Step 3: Save weights (triggers vLLM with LoRA)
    print("\nStep 3: Saving weights (triggers vLLM LoRA loading)...")
    t0 = time.time()
    try:
        result = save_weights(model_id, name="moe_lora_test")
        save_time = time.time() - t0
        sampling_registered = result.get("sampling_registered", False)
        print(f"  Completed in {save_time:.1f}s")
        print(f"  sampling_registered: {sampling_registered}")

        if not sampling_registered:
            print("  WARNING: LoRA not registered for sampling!")
            print("  Checking for error details...")
            if "error" in result:
                print(f"  Error: {result['error']}")
    except Exception as e:
        print(f"  FAILED: {e}")
        return 1

    # Step 4: Sample
    print("\nStep 4: Sampling with LoRA...")
    prompt_tokens = [100, 101, 102, 103, 104]
    try:
        t0 = time.time()
        result = sample(model_id, prompt_tokens, max_tokens=10)
        sample_time = time.time() - t0
        tokens = result.get("sequences", [{}])[0].get("tokens", [])[:10]
        print(f"  Completed in {sample_time:.1f}s")
        print(f"  Generated tokens: {tokens}")
        sample_ok = True
    except Exception as e:
        print(f"  FAILED: {e}")
        sample_ok = False

    # Summary
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"Training: {losses[0]:.4f} -> {losses[-1]:.4f} ({(1 - losses[-1]/losses[0])*100:.1f}% reduction)")
    print(f"LoRA registered: {sampling_registered}")
    print(f"Sampling: {'OK' if sample_ok else 'FAILED'}")

    if sampling_registered and sample_ok:
        print("\nMoE LoRA inference with TP=4: WORKING")
        return 0
    else:
        print("\nMoE LoRA inference: NEEDS INVESTIGATION")
        return 1


if __name__ == "__main__":
    sys.exit(main())
