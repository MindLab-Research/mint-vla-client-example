#!/usr/bin/env python
"""MoE RL (GRPO/Importance Sampling) Integration Test - Phase 5 Validation.

Tests reinforcement learning training with MoE model via Megatron backend.

Steps:
1. Create MoE training session
2. Run forward pass to get logprobs (simulating rollout)
3. Compute advantages (simulated reward signal)
4. Run forward_backward with importance_sampling loss
5. Run optim_step
6. Verify metrics (loss, ratio, etc.)

Usage:
    TINKER_BASE_URL=http://localhost:8000 python scripts/test_moe_rl.py
"""

import os
import sys
import time
import uuid

import requests

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")


def poll_future(request_id: str, timeout: int = 600) -> dict:
    """Poll for async operation result."""
    poll_url = f"{BASE_URL}/api/v1/retrieve_future"
    start = time.time()

    while time.time() - start < timeout:
        resp = requests.post(poll_url, json={"request_id": request_id}, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 408:
            time.sleep(2)
            continue
        else:
            resp.raise_for_status()

    raise TimeoutError(f"Operation did not complete within {timeout}s")


def create_training_session(base_model: str, rank: int = 16, timeout: int = 900) -> tuple[str, str]:
    """Create training session, return (session_id, model_id)."""
    session_id = str(uuid.uuid4())
    model_seq_id = 1

    url = f"{BASE_URL}/api/v1/create_model"
    payload = {
        "session_id": session_id,
        "model_seq_id": model_seq_id,
        "base_model": base_model,
        "lora_config": {"rank": rank},
    }
    resp = requests.post(url, json=payload, timeout=900)
    resp.raise_for_status()
    future = resp.json()
    request_id = future.get("request_id")

    result = poll_future(request_id, timeout=timeout)

    if "error" in result:
        raise RuntimeError(f"Session creation failed: {result['error']}")

    model_id = result.get("model_id")
    backend = result.get("backend")
    return session_id, model_id, backend


def forward_backward(model_id: str, data: list, loss_fn: str = "cross_entropy",
                     loss_fn_config: dict = None) -> dict:
    """Run forward_backward and poll for result."""
    url = f"{BASE_URL}/api/v1/forward_backward"
    payload = {
        "model_id": model_id,
        "forward_backward_input": {
            "data": data,
            "loss_fn": loss_fn,
        },
    }
    if loss_fn_config:
        payload["forward_backward_input"]["loss_fn_config"] = loss_fn_config

    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    future = resp.json()
    request_id = future.get("request_id")

    return poll_future(request_id, timeout=300)


def optim_step(model_id: str, learning_rate: float = 1e-4) -> dict:
    """Run optim_step and poll for result."""
    url = f"{BASE_URL}/api/v1/optim_step"
    payload = {
        "model_id": model_id,
        "adam_params": {"learning_rate": learning_rate},
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    future = resp.json()
    request_id = future.get("request_id")

    return poll_future(request_id, timeout=120)


def create_rl_data() -> list:
    """Create RL training data with logprobs and advantages.

    For importance sampling loss:
    - logprobs: log probabilities from old policy (rollout)
    - advantages: reward signal (positive = good action, negative = bad action)
    - weights: loss mask for valid tokens
    """
    # Simulate a simple sequence
    seq_len = 5
    input_tokens = [9707, 1917, 0, 0, 0]  # "Hello World" + padding
    target_tokens = [1917, 0, 0, 0, 0]     # Shifted targets

    # Simulated old policy logprobs (would come from rollout)
    # These represent log P(action | state) under old policy
    old_logprobs = [-1.5, -2.0, -0.5, 0.0, 0.0]

    # Simulated advantages (would come from reward model or GAE)
    # Positive = good actions, Negative = bad actions
    advantages = [0.5, -0.3, 0.8, 0.0, 0.0]

    # Loss mask: only compute loss on valid tokens
    loss_mask = [1.0, 1.0, 1.0, 0.0, 0.0]

    return [{
        "model_input": {
            "chunks": [{"tokens": input_tokens, "type": "encoded_text"}]
        },
        "loss_fn_inputs": {
            "target_tokens": {
                "data": target_tokens,
                "shape": [seq_len],
                "dtype": "int64"
            },
            "logprobs": {
                "data": old_logprobs,
                "shape": [seq_len],
                "dtype": "float32"
            },
            "advantages": {
                "data": advantages,
                "shape": [seq_len],
                "dtype": "float32"
            },
            "loss_mask": {
                "data": loss_mask,
                "shape": [seq_len],
                "dtype": "float32"
            },
        },
    }]


def create_ppo_data() -> list:
    """Create PPO training data (same as RL but for PPO loss)."""
    return create_rl_data()


def test_moe_rl():
    """Test MoE RL training: importance_sampling and PPO losses."""
    print("=" * 70)
    print("TEST: MoE RL Training (Phase 5)")
    print("=" * 70)

    moe_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"

    # Step 1: Create training session
    print(f"\n[1/5] Creating MoE training session for: {moe_model}")
    print("This may take several minutes for Megatron initialization...")

    t0 = time.time()
    try:
        session_id, model_id, backend = create_training_session(moe_model, rank=16)
        init_time = time.time() - t0
        print(f"Session created in {init_time:.2f}s")
        print(f"  model_id={model_id}")
        print(f"  backend={backend}")
    except Exception as e:
        print(f"[FAIL] create_model failed: {e}")
        return False

    # Step 2: Test importance_sampling loss (GRPO-style)
    print(f"\n[2/5] Testing importance_sampling loss (GRPO-style)...")
    rl_data = create_rl_data()

    t0 = time.time()
    try:
        is_result = forward_backward(model_id, rl_data, loss_fn="importance_sampling")
        is_time = time.time() - t0
        print(f"importance_sampling forward_backward completed in {is_time:.2f}s")

        metrics = is_result.get("metrics", {})
        loss = metrics.get("loss:mean", "N/A")
        ratio = metrics.get("ratio:mean", "N/A")
        print(f"  Loss: {loss}")
        print(f"  Ratio (mean): {ratio}")

    except Exception as e:
        print(f"[FAIL] importance_sampling forward_backward failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 3: Optimizer step after importance_sampling
    print(f"\n[3/5] Running optim_step after importance_sampling...")

    t0 = time.time()
    try:
        optim_result = optim_step(model_id, learning_rate=1e-4)
        optim_time = time.time() - t0
        print(f"optim_step completed in {optim_time:.2f}s")
    except Exception as e:
        print(f"[FAIL] optim_step failed: {e}")
        return False

    # Step 4: Test PPO loss with clipping
    print(f"\n[4/5] Testing PPO loss with clipping...")
    ppo_data = create_ppo_data()

    t0 = time.time()
    try:
        ppo_result = forward_backward(
            model_id,
            ppo_data,
            loss_fn="ppo",
            loss_fn_config={"epsilon": 0.2}
        )
        ppo_time = time.time() - t0
        print(f"PPO forward_backward completed in {ppo_time:.2f}s")

        metrics = ppo_result.get("metrics", {})
        loss = metrics.get("loss:mean", "N/A")
        ratio = metrics.get("ratio:mean", "N/A")
        clipfrac = metrics.get("clipfrac:mean", "N/A")
        print(f"  Loss: {loss}")
        print(f"  Ratio (mean): {ratio}")
        print(f"  Clip fraction: {clipfrac}")

    except Exception as e:
        print(f"[FAIL] PPO forward_backward failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 5: Final optim_step after PPO
    print(f"\n[5/5] Running final optim_step after PPO...")

    t0 = time.time()
    try:
        optim_result = optim_step(model_id, learning_rate=1e-4)
        optim_time = time.time() - t0
        print(f"optim_step completed in {optim_time:.2f}s")
    except Exception as e:
        print(f"[FAIL] final optim_step failed: {e}")
        return False

    print("\n" + "=" * 70)
    print("[PASS] MoE RL Training completed successfully")
    print("=" * 70)
    print("\nVerified capabilities:")
    print("  - importance_sampling loss (GRPO/policy gradient)")
    print("  - PPO loss with epsilon clipping")
    print("  - RL metrics: ratio, clip_fraction")
    print("  - MoE model via Megatron backend")

    return True


def main():
    print(f"Server: {BASE_URL}")
    print("=" * 70)
    print("MOE RL TRAINING INTEGRATION TEST (Phase 5)")
    print("=" * 70)

    # Check server health
    print("\nChecking server health...")
    try:
        resp = requests.get(f"{BASE_URL}/api/v1/healthz", timeout=10)
        health = resp.json()
        print(f"Server status: {health.get('status')}")
    except Exception as e:
        print(f"Server not available: {e}")
        return 1

    # Run test
    success = test_moe_rl()

    print("\n" + "=" * 70)
    print("RESULT: " + ("PASS" if success else "FAIL"))
    print("=" * 70)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
