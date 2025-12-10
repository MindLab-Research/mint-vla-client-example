#!/usr/bin/env python
"""MoE Training Integration Test - Full Training Flow Validation.

Tests end-to-end MoE training via MegatronTrainingWorker:
1. Create training session (routes to megatron backend)
2. forward_backward with sample data
3. optim_step to verify return format fix

Usage:
    TINKER_BASE_URL=http://localhost:8000 python scripts/test_moe_training.py
"""

import os
import sys
import time
import json
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


def create_training_session(base_model: str, rank: int = 16) -> tuple[str, str]:
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

    result = poll_future(request_id, timeout=600)

    # Check for error response
    if "error" in result:
        print(f"[ERROR] Session creation failed: {result['error']}")
        raise RuntimeError(f"Session creation failed: {result['error']}")

    model_id = result.get("model_id")
    backend = result.get("backend")

    print(f"Session created: model_id={model_id}, backend={backend}")
    return session_id, model_id


def forward_backward(model_id: str, data: list) -> dict:
    """Run forward_backward and poll for result."""
    url = f"{BASE_URL}/api/v1/forward_backward"
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
    request_id = future.get("request_id")

    return poll_future(request_id, timeout=300)


def optim_step(model_id: str, learning_rate: float = 1e-4) -> dict:
    """Run optim_step and poll for result."""
    url = f"{BASE_URL}/api/v1/optim_step"
    payload = {
        "model_id": model_id,
        "learning_rate": learning_rate,
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    future = resp.json()
    request_id = future.get("request_id")

    return poll_future(request_id, timeout=120)


def create_sample_data() -> list:
    """Create minimal sample data for SFT training.

    Returns data in Datum format expected by the API:
    - model_input: {chunks: [{tokens, type}]}
    - loss_fn_inputs: {target_tokens, loss_mask}
    """
    # Sample 1: "Hello world" -> predict next tokens
    input_tokens_1 = [9707, 1917, 0]  # "Hello world" + padding
    target_tokens_1 = [1917, 0, 0]    # Shifted targets
    loss_mask_1 = [1.0, 1.0, 0.0]     # Train on first 2 positions

    # Sample 2: Simple sequence
    input_tokens_2 = [9707, 0]        # Shorter
    target_tokens_2 = [0, 0]
    loss_mask_2 = [1.0, 0.0]

    return [
        {
            "model_input": {
                "chunks": [{"tokens": input_tokens_1, "type": "encoded_text"}]
            },
            "loss_fn_inputs": {
                "target_tokens": {"data": target_tokens_1, "shape": [len(target_tokens_1)], "dtype": "int64"},
                "loss_mask": {"data": loss_mask_1, "shape": [len(loss_mask_1)], "dtype": "float32"},
            },
        },
        {
            "model_input": {
                "chunks": [{"tokens": input_tokens_2, "type": "encoded_text"}]
            },
            "loss_fn_inputs": {
                "target_tokens": {"data": target_tokens_2, "shape": [len(target_tokens_2)], "dtype": "int64"},
                "loss_mask": {"data": loss_mask_2, "shape": [len(loss_mask_2)], "dtype": "float32"},
            },
        },
    ]


def test_moe_training_flow():
    """Test full MoE training: create_model -> forward_backward -> optim_step."""
    print("=" * 70)
    print("TEST: MoE Full Training Flow")
    print("=" * 70)

    moe_model = "Qwen/Qwen3-30B-A3B-Instruct-2507"

    # Step 1: Create session
    print(f"\n[1/3] Creating training session for: {moe_model}")
    print("This may take several minutes for Megatron initialization...")

    t0 = time.time()
    try:
        _, model_id = create_training_session(moe_model, rank=16)
        init_time = time.time() - t0
        print(f"Session created in {init_time:.2f}s")
    except Exception as e:
        print(f"[FAIL] create_model failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 2: forward_backward
    print(f"\n[2/3] Running forward_backward with sample data...")
    data = create_sample_data()
    print(f"Data: {len(data)} samples")

    t0 = time.time()
    try:
        fb_result = forward_backward(model_id, data)
        fb_time = time.time() - t0
        print(f"forward_backward completed in {fb_time:.2f}s")
        print(f"Result: {json.dumps(fb_result, indent=2, default=str)}")

        # Check for metrics
        metrics = fb_result.get("metrics", {})
        loss = metrics.get("loss:mean", "N/A")
        print(f"Loss: {loss}")
    except Exception as e:
        print(f"[FAIL] forward_backward failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 3: optim_step - this tests the return format fix
    print(f"\n[3/3] Running optim_step (tests return format fix)...")

    t0 = time.time()
    try:
        optim_result = optim_step(model_id, learning_rate=1e-4)
        optim_time = time.time() - t0
        print(f"optim_step completed in {optim_time:.2f}s")
        print(f"Result: {json.dumps(optim_result, indent=2, default=str)}")

        # Verify metrics field exists
        metrics = optim_result.get("metrics", {})
        step = metrics.get("step", "MISSING")
        print(f"Step: {step}")

        if "metrics" not in optim_result:
            print("[FAIL] optim_step result missing 'metrics' field")
            return False

    except Exception as e:
        print(f"[FAIL] optim_step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 70)
    print("[PASS] Full MoE training flow completed successfully")
    print("=" * 70)
    return True


def main():
    print(f"Server: {BASE_URL}")
    print("=" * 70)
    print("MOE TRAINING INTEGRATION TEST")
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
    success = test_moe_training_flow()

    print("\n" + "=" * 70)
    print("RESULT: " + ("PASS" if success else "FAIL"))
    print("=" * 70)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
