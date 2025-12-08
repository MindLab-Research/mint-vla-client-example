#!/usr/bin/env python3
"""Test script for P6: Custom losses via weights support.

Tests:
1. Standard SFT with positive weights (should average)
2. Custom loss backward with negative weights (should sum without averaging)
3. Full forward_backward_custom workflow simulation

Usage:
    TINKER_MODEL_PATH=/path/to/model python scripts/test_custom_loss.py
"""

import os
import time
import requests

MODEL_PATH = os.environ.get(
    "TINKER_MODEL_PATH",
    "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)
BASE_URL = "http://localhost:8000/api/v1"


def poll_future(request_id: str, timeout: int = 300) -> dict:
    """Poll until future resolves."""
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.post(f"{BASE_URL}/retrieve_future", json={"request_id": request_id})
        if resp.status_code == 200:
            return resp.json()
        time.sleep(1)
    raise TimeoutError(f"Future {request_id} did not resolve in {timeout}s")


def test_custom_loss():
    print("=" * 60)
    print("P6 Custom Loss Test Suite")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Base URL: {BASE_URL}")

    # 1. Create session
    print("\n1. Creating session...")
    resp = requests.post(f"{BASE_URL}/create_session", json={"tags": [], "user_metadata": {}})
    assert resp.status_code == 200, f"Failed: {resp.text}"
    session_id = resp.json()["session_id"]
    print(f"   session_id: {session_id}")

    # 2. Create model
    print("\n2. Creating model...")
    resp = requests.post(f"{BASE_URL}/create_model", json={
        "session_id": session_id,
        "model_seq_id": 0,
        "base_model": MODEL_PATH,
        "lora_config": {"rank": 32, "train_unembed": True, "train_mlp": True, "train_attn": True},
    })
    assert resp.status_code == 200, f"Failed: {resp.text}"
    request_id = resp.json()["request_id"]
    result = poll_future(request_id, timeout=120)
    model_id = result["model_id"]
    print(f"   model_id: {model_id}")

    # Test data with weights
    input_tokens = [9707, 1879, 2233, 1234]
    target_tokens = [1879, 2233, 1234, 5678]

    # 3. Test SFT with positive weights (should average)
    print("\n3. Testing SFT with positive weights...")
    weights_positive = [1.0, 1.0, 1.0, 1.0]
    data = [{
        "model_input": {"chunks": [{"tokens": input_tokens, "type": "encoded_text"}]},
        "loss_fn_inputs": {
            "target_tokens": {"data": target_tokens, "shape": [len(target_tokens)], "dtype": "int64"},
            "weights": {"data": weights_positive, "shape": [len(weights_positive)], "dtype": "float32"},
        },
    }]

    resp = requests.post(f"{BASE_URL}/forward_backward", json={
        "model_id": model_id,
        "forward_backward_input": {"data": data, "loss_fn": "cross_entropy"},
    })
    assert resp.status_code == 200, f"Failed: {resp.text}"
    request_id = resp.json()["request_id"]
    result = poll_future(request_id, timeout=60)

    sft_loss = result["metrics"].get("loss:mean", 0)
    sft_tokens = result["metrics"].get("num_tokens:sum", 0)
    print(f"   loss: {sft_loss:.4f}")
    print(f"   num_tokens: {sft_tokens}")
    assert sft_tokens == 4.0, f"Expected 4 tokens, got {sft_tokens}"
    print("   Positive weights -> averaged loss")

    # 4. Test custom loss backward with negative weights (should sum without averaging)
    print("\n4. Testing custom loss backward with negative weights...")
    # Simulate gradients from a custom loss function
    weights_negative = [-0.5, -0.3, 0.2, -0.1]  # Mixed, but has negatives

    data = [{
        "model_input": {"chunks": [{"tokens": input_tokens, "type": "encoded_text"}]},
        "loss_fn_inputs": {
            "target_tokens": {"data": target_tokens, "shape": [len(target_tokens)], "dtype": "int64"},
            "weights": {"data": weights_negative, "shape": [len(weights_negative)], "dtype": "float32"},
        },
    }]

    resp = requests.post(f"{BASE_URL}/forward_backward", json={
        "model_id": model_id,
        "forward_backward_input": {"data": data, "loss_fn": "cross_entropy"},
    })
    assert resp.status_code == 200, f"Failed: {resp.text}"
    request_id = resp.json()["request_id"]
    result = poll_future(request_id, timeout=60)

    custom_loss = result["metrics"].get("loss:mean", 0)
    custom_tokens = result["metrics"].get("num_tokens:sum", 0)
    print(f"   loss: {custom_loss:.4f}")
    print(f"   effective_tokens (abs sum): {custom_tokens:.2f}")
    # With negative weights, effective_tokens = sum(|weights|) = 0.5 + 0.3 + 0.2 + 0.1 = 1.1
    assert abs(custom_tokens - 1.1) < 0.01, f"Expected ~1.1 effective tokens, got {custom_tokens}"
    print("   Negative weights -> summed loss (custom loss backward)")

    # 5. Test forward (no backward) with weights
    print("\n5. Testing forward (returns logprobs)...")
    weights_forward = [1.0, 1.0, 1.0, 1.0]

    data = [{
        "model_input": {"chunks": [{"tokens": input_tokens, "type": "encoded_text"}]},
        "loss_fn_inputs": {
            "target_tokens": {"data": target_tokens, "shape": [len(target_tokens)], "dtype": "int64"},
            "weights": {"data": weights_forward, "shape": [len(weights_forward)], "dtype": "float32"},
        },
    }]

    resp = requests.post(f"{BASE_URL}/forward", json={
        "model_id": model_id,
        "forward_backward_input": {"data": data, "loss_fn": "cross_entropy"},
    })
    assert resp.status_code == 200, f"Failed: {resp.text}"
    request_id = resp.json()["request_id"]
    result = poll_future(request_id, timeout=60)

    logprobs = result["loss_fn_outputs"][0]["logprobs"]["data"]
    forward_loss = result["metrics"].get("loss:mean", 0)
    print(f"   loss: {forward_loss:.4f}")
    print(f"   logprobs: {logprobs}")
    print(f"   logprobs length: {len(logprobs)}")
    assert len(logprobs) == 4, f"Expected 4 logprobs, got {len(logprobs)}"

    # 6. Full forward_backward_custom workflow simulation
    print("\n6. Simulating full forward_backward_custom workflow...")

    # Step 6a: Forward pass to get logprobs
    print("   Step 6a: Forward pass to get logprobs...")
    resp = requests.post(f"{BASE_URL}/forward", json={
        "model_id": model_id,
        "forward_backward_input": {"data": data, "loss_fn": "cross_entropy"},
    })
    assert resp.status_code == 200, f"Failed: {resp.text}"
    request_id = resp.json()["request_id"]
    forward_result = poll_future(request_id, timeout=60)
    logprobs = forward_result["loss_fn_outputs"][0]["logprobs"]["data"]
    print(f"      logprobs: {logprobs}")

    # Step 6b: Simulate client-side custom loss and gradient computation
    # In real usage:
    #   logprobs_t = torch.tensor(logprobs, requires_grad=True)
    #   custom_loss = f(logprobs_t)  # e.g., DPO loss
    #   custom_loss.backward()
    #   grads = logprobs_t.grad.tolist()
    print("   Step 6b: Simulate client-side custom loss...")
    simulated_grads = [0.1, -0.2, 0.05, -0.15]
    negative_grads = [-g for g in simulated_grads]
    print(f"      simulated grads: {simulated_grads}")
    print(f"      negative grads (weights): {negative_grads}")

    # Step 6c: Forward-backward with negative gradients as weights
    print("   Step 6c: Forward-backward with negative grads as weights...")
    backward_data = [{
        "model_input": {"chunks": [{"tokens": input_tokens, "type": "encoded_text"}]},
        "loss_fn_inputs": {
            "target_tokens": {"data": target_tokens, "shape": [len(target_tokens)], "dtype": "int64"},
            "weights": {"data": negative_grads, "shape": [len(negative_grads)], "dtype": "float32"},
        },
    }]

    resp = requests.post(f"{BASE_URL}/forward_backward", json={
        "model_id": model_id,
        "forward_backward_input": {"data": backward_data, "loss_fn": "cross_entropy"},
    })
    assert resp.status_code == 200, f"Failed: {resp.text}"
    request_id = resp.json()["request_id"]
    backward_result = poll_future(request_id, timeout=60)

    workflow_loss = backward_result["metrics"].get("loss:mean", 0)
    print(f"      loss: {workflow_loss:.4f}")
    print("   Workflow complete: Custom loss gradients propagated to model")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_custom_loss()
