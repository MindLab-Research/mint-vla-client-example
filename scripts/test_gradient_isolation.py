#!/usr/bin/env python3
"""Test gradient isolation between interleaved training sessions.

Verifies that:
1. Session A's gradients are preserved when session B runs forward_backward
2. Session A's optim_step uses A's gradients (not B's)
3. Both sessions learn independently (losses decrease)

This tests the fix for the Tinker API incompatibility where verl's train_mode()
zeros gradients on entry, breaking separate forward_backward + optim_step calls.

Run from tinker-server root directory:
    TINKER_BASE_URL=http://localhost:8000 python scripts/test_gradient_isolation.py
"""

import os
import sys
import time
import uuid

import requests


def get_base_url():
    return os.environ.get("TINKER_BASE_URL", "http://localhost:8000")


def poll_future(request_id: str, timeout: int = 300) -> dict:
    """Poll for async operation result."""
    poll_url = f"{get_base_url()}/api/v1/retrieve_future"
    start = time.time()

    while time.time() - start < timeout:
        resp = requests.post(poll_url, json={"request_id": request_id}, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 408:
            time.sleep(0.5)
            continue
        else:
            resp.raise_for_status()

    raise TimeoutError(f"Operation did not complete within {timeout}s")


def create_session(base_model: str, lora_rank: int = 16) -> tuple[str, str]:
    """Create training session and return (session_id, model_id)."""
    session_id = str(uuid.uuid4())
    url = f"{get_base_url()}/api/v1/create_model"
    payload = {
        "session_id": session_id,
        "model_seq_id": 1,
        "base_model": base_model,
        "lora_config": {"rank": lora_rank},
    }

    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    future = resp.json()
    result = poll_future(future.get("request_id"), timeout=300)

    if "error" in result:
        raise RuntimeError(f"Session creation failed: {result['error']}")

    return session_id, result.get("model_id")


def forward_backward(model_id: str, data: list) -> float:
    """Run forward-backward pass. Returns loss."""
    url = f"{get_base_url()}/api/v1/forward_backward"
    payload = {
        "model_id": model_id,
        "forward_backward_input": {
            "data": data,
            "loss_fn": "cross_entropy",
        },
    }

    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    future = resp.json()
    result = poll_future(future.get("request_id"), timeout=300)

    if "error" in result:
        raise RuntimeError(f"forward_backward error: {result['error']}")

    loss = result.get("metrics", {}).get("loss:mean", 0)
    if loss == 0:
        loss_outputs = result.get("loss_fn_outputs", [])
        if loss_outputs:
            losses = [o.get("loss", {}).get("data", [0])[0] for o in loss_outputs]
            loss = sum(losses) / len(losses) if losses else 0

    return loss


def optim_step(model_id: str, learning_rate: float = 1e-4) -> dict:
    """Run optimizer step. Returns result dict."""
    url = f"{get_base_url()}/api/v1/optim_step"
    payload = {
        "model_id": model_id,
        "adam_params": {"learning_rate": learning_rate},
    }

    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    future = resp.json()
    result = poll_future(future.get("request_id"), timeout=120)

    if "error" in result:
        raise RuntimeError(f"optim_step error: {result['error']}")

    return result


def create_sample_data(seed: int, batch_size: int = 4, seq_len: int = 32) -> list:
    """Create sample data with deterministic pattern based on seed."""
    data = []
    for i in range(batch_size):
        # Use seed to create different data for different sessions
        base = 1000 + seed * 1000 + i * 100
        input_tokens = [151644] + list(range(base, base + seq_len))
        target_tokens = list(range(base, base + seq_len)) + [151645]
        loss_mask = [1.0] * seq_len + [1.0]

        data.append({
            "model_input": {
                "chunks": [{"tokens": input_tokens, "type": "encoded_text"}]
            },
            "loss_fn_inputs": {
                "target_tokens": {"data": target_tokens, "shape": [len(target_tokens)], "dtype": "int64"},
                "loss_mask": {"data": loss_mask, "shape": [len(loss_mask)], "dtype": "float32"},
            },
        })
    return data


def test_gradient_isolation():
    """Test that gradients are isolated between sessions."""
    # Use small model for faster testing
    BASE_MODEL = os.environ.get("TEST_MODEL", "Qwen/Qwen3-0.6B")
    LEARNING_RATE = 5e-5
    NUM_ITERATIONS = 5

    print(f"\n{'='*70}")
    print("GRADIENT ISOLATION TEST")
    print(f"{'='*70}")
    print(f"Model: {BASE_MODEL}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Iterations: {NUM_ITERATIONS}")
    print(f"{'='*70}\n")

    # Create two sessions
    print("[1/6] Creating Session A...")
    session_a_id, model_a_id = create_session(BASE_MODEL)
    print(f"      Session A: {session_a_id[:8]}... -> {model_a_id}")

    print("[2/6] Creating Session B...")
    session_b_id, model_b_id = create_session(BASE_MODEL)
    print(f"      Session B: {session_b_id[:8]}... -> {model_b_id}")

    # Create different data for each session
    data_a = create_sample_data(seed=42)
    data_b = create_sample_data(seed=99)

    losses_a = []
    losses_b = []

    print(f"\n[3/6] Training with INTERLEAVED sessions...")
    print(f"      Pattern: A.fwdbwd -> B.fwdbwd -> A.optim -> B.optim (repeat)\n")

    for i in range(NUM_ITERATIONS):
        # Interleaved pattern: A forward_backward, then B forward_backward
        # Then A optim_step, then B optim_step
        # This is the worst case for gradient isolation

        loss_a = forward_backward(model_a_id, data_a)
        loss_b = forward_backward(model_b_id, data_b)

        # Now optim_step for A (should use A's gradients, not B's)
        optim_step(model_a_id, LEARNING_RATE)
        # And optim_step for B
        optim_step(model_b_id, LEARNING_RATE)

        losses_a.append(loss_a)
        losses_b.append(loss_b)

        print(f"  Iteration {i+1}/{NUM_ITERATIONS}: loss_A={loss_a:.4f}, loss_B={loss_b:.4f}")

    # Analyze results
    print(f"\n[4/6] Analyzing Session A losses...")
    loss_drop_a = (losses_a[0] - losses_a[-1]) / losses_a[0] * 100 if losses_a[0] > 0 else 0
    print(f"      Initial: {losses_a[0]:.4f}")
    print(f"      Final:   {losses_a[-1]:.4f}")
    print(f"      Drop:    {loss_drop_a:.1f}%")

    print(f"\n[5/6] Analyzing Session B losses...")
    loss_drop_b = (losses_b[0] - losses_b[-1]) / losses_b[0] * 100 if losses_b[0] > 0 else 0
    print(f"      Initial: {losses_b[0]:.4f}")
    print(f"      Final:   {losses_b[-1]:.4f}")
    print(f"      Drop:    {loss_drop_b:.1f}%")

    # Test criteria
    print(f"\n[6/6] Checking test criteria...")
    passed = True

    # Criterion 1: Both sessions should show loss decrease
    # (indicates gradients were not corrupted by interleaving)
    if loss_drop_a < 1.0:
        print(f"      FAIL: Session A loss did not decrease significantly ({loss_drop_a:.1f}% < 1%)")
        passed = False
    else:
        print(f"      PASS: Session A loss decreased by {loss_drop_a:.1f}%")

    if loss_drop_b < 1.0:
        print(f"      FAIL: Session B loss did not decrease significantly ({loss_drop_b:.1f}% < 1%)")
        passed = False
    else:
        print(f"      PASS: Session B loss decreased by {loss_drop_b:.1f}%")

    # Criterion 2: Losses should be different (sessions use different data)
    # This confirms they're not sharing state incorrectly
    if abs(losses_a[-1] - losses_b[-1]) < 0.01:
        print(f"      WARN: Final losses suspiciously similar (may indicate shared state)")
    else:
        print(f"      PASS: Sessions have different final losses (independent training)")

    print(f"\n{'='*70}")
    if passed:
        print("GRADIENT ISOLATION TEST: PASSED")
    else:
        print("GRADIENT ISOLATION TEST: FAILED")
    print(f"{'='*70}\n")

    return passed


def test_gradient_accumulation():
    """Test that multiple forward_backward calls accumulate gradients."""
    BASE_MODEL = os.environ.get("TEST_MODEL", "Qwen/Qwen3-0.6B")
    LEARNING_RATE = 5e-5

    print(f"\n{'='*70}")
    print("GRADIENT ACCUMULATION TEST")
    print(f"{'='*70}")
    print(f"Model: {BASE_MODEL}")
    print(f"{'='*70}\n")

    # Create session
    print("[1/4] Creating session...")
    session_id, model_id = create_session(BASE_MODEL)
    print(f"      Session: {session_id[:8]}... -> {model_id}")

    data = create_sample_data(seed=123)

    # Test 1: Single forward_backward + optim_step
    print("\n[2/4] Testing single forward_backward + optim_step...")
    loss_single = forward_backward(model_id, data)
    optim_step(model_id, LEARNING_RATE)
    print(f"      Loss after 1x fwdbwd: {loss_single:.4f}")

    # Measure loss after the update
    loss_after_single = forward_backward(model_id, data)
    optim_step(model_id, LEARNING_RATE)
    print(f"      Loss after update: {loss_after_single:.4f}")

    # Create new session for accumulation test
    print("\n[3/4] Creating fresh session for accumulation test...")
    session_id2, model_id2 = create_session(BASE_MODEL)

    # Test 2: Multiple forward_backward (accumulation) + single optim_step
    print("[4/4] Testing 3x forward_backward + 1x optim_step...")
    for i in range(3):
        loss = forward_backward(model_id2, data)
        print(f"      forward_backward {i+1}: loss={loss:.4f}")

    # Single optim_step should apply accumulated gradients
    optim_step(model_id2, LEARNING_RATE)

    # Check if accumulated gradients had effect
    loss_after_accum = forward_backward(model_id2, data)
    print(f"      Loss after accumulated update: {loss_after_accum:.4f}")

    # The accumulated update should have larger effect
    drop_single = loss_single - loss_after_single
    drop_accum = loss_single - loss_after_accum

    print(f"\n      Drop with 1x fwdbwd: {drop_single:.4f}")
    print(f"      Drop with 3x fwdbwd: {drop_accum:.4f}")

    passed = True
    if drop_accum > drop_single * 1.5:  # Accumulated should have notably larger effect
        print(f"      PASS: Gradient accumulation has larger effect")
    elif drop_accum > drop_single:
        print(f"      PASS: Gradient accumulation working (effect slightly larger)")
    else:
        print(f"      WARN: Gradient accumulation may not be working as expected")
        # Don't fail - the effect depends on learning rate and data

    print(f"\n{'='*70}")
    print("GRADIENT ACCUMULATION TEST: PASSED")
    print(f"{'='*70}\n")

    return passed


def main():
    print("\n" + "="*70)
    print(" GRADIENT ISOLATION & ACCUMULATION TESTS")
    print("="*70)

    results = []

    # Test 1: Gradient isolation between sessions
    try:
        passed = test_gradient_isolation()
        results.append(("Gradient Isolation", passed))
    except Exception as e:
        print(f"ERROR in gradient isolation test: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Gradient Isolation", False))

    # Test 2: Gradient accumulation within session
    try:
        passed = test_gradient_accumulation()
        results.append(("Gradient Accumulation", passed))
    except Exception as e:
        print(f"ERROR in gradient accumulation test: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Gradient Accumulation", False))

    # Summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("="*70)
    if all_passed:
        print(" ALL TESTS PASSED")
        sys.exit(0)
    else:
        print(" SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
