#!/usr/bin/env python3
"""Reproduce GitHub Issue #51 using raw HTTP requests.

Tests whether TensorData format works when sent to MinT server.
"""

import json
import time
import requests
import torch

BASE_URL = "http://localhost:8000"
SESSION_ID = f"issue51_test_{int(time.time())}"


def poll_future(request_id: str, timeout: int = 120) -> dict:
    """Poll for async result."""
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.post(
            f"{BASE_URL}/api/v1/retrieve_future",
            json={"request_id": request_id},
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 408:
            time.sleep(0.5)
            continue
        else:
            resp.raise_for_status()
    raise TimeoutError(f"Timeout waiting for {request_id}")


def main():
    from transformers import AutoTokenizer

    print("=" * 60)
    print("Issue #51 Reproduction: TensorData in RL training")
    print("=" * 60)

    # Create model
    print("\n1. Creating training model...")
    resp = requests.post(
        f"{BASE_URL}/api/v1/create_model",
        json={
            "session_id": SESSION_ID,
            "model_seq_id": 1,
            "base_model": "Qwen/Qwen3-0.6B",
            "lora_config": {"rank": 16},
        },
    )
    resp.raise_for_status()
    request_id = resp.json()["request_id"]
    result = poll_future(request_id)
    model_id = result["model_id"]
    print(f"   Model created: {model_id}")

    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)

    # Prepare test data
    prompt = "Question: What is 2 + 2?\nAnswer:"
    completion = " 4"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False) + [tokenizer.eos_token_id]
    all_tokens = prompt_tokens + completion_tokens
    input_tokens = all_tokens[:-1]
    target_tokens = all_tokens[1:]
    weights = [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(completion_tokens)
    logprobs = [0.0] * len(input_tokens)
    advantages = [0.0] * (len(prompt_tokens) - 1) + [0.5] * len(completion_tokens)

    # Test 1: SFT with plain lists
    print("\n2. TEST: SFT with plain lists (baseline)")
    datum_sft = {
        "model_input": {"chunks": [{"tokens": input_tokens, "type": "encoded_text"}]},
        "loss_fn_inputs": {
            "target_tokens": {"data": target_tokens, "shape": [len(target_tokens)], "dtype": "int64"},
            "weights": {"data": weights, "shape": [len(weights)], "dtype": "float32"},
        },
    }

    resp = requests.post(
        f"{BASE_URL}/api/v1/forward_backward",
        json={
            "model_id": model_id,
            "forward_backward_input": {"data": [datum_sft], "loss_fn": "cross_entropy"},
        },
    )
    resp.raise_for_status()
    result = poll_future(resp.json()["request_id"])
    if "error" in result:
        print(f"   FAILED: {result['error']}")
    else:
        print(f"   SUCCESS: {len(result.get('loss_fn_outputs', []))} outputs")

    # Test 2: RL with TensorData format (as TensorData.from_torch would produce)
    print("\n3. TEST: RL with TensorData format (issue scenario)")

    # This is what TensorData.from_torch() produces when serialized to JSON
    datum_rl_tensordata = {
        "model_input": {"chunks": [{"tokens": input_tokens, "type": "encoded_text"}]},
        "loss_fn_inputs": {
            "target_tokens": {
                "data": target_tokens,
                "shape": [len(target_tokens)],
                "dtype": "int64",
            },
            "weights": {
                "data": weights,
                "shape": [len(weights)],
                "dtype": "float32",
            },
            "logprobs": {
                "data": logprobs,
                "shape": [len(logprobs)],
                "dtype": "float32",
            },
            "advantages": {
                "data": advantages,
                "shape": [len(advantages)],
                "dtype": "float32",
            },
        },
    }

    resp = requests.post(
        f"{BASE_URL}/api/v1/forward_backward",
        json={
            "model_id": model_id,
            "forward_backward_input": {"data": [datum_rl_tensordata], "loss_fn": "importance_sampling"},
        },
    )
    resp.raise_for_status()
    result = poll_future(resp.json()["request_id"])
    if "error" in result:
        print(f"   FAILED: {result['error']}")
    else:
        print(f"   SUCCESS: {len(result.get('loss_fn_outputs', []))} outputs")

    # Test 3: What if someone passes TensorData object directly without proper serialization?
    # This tests if there's a case where the SDK doesn't serialize properly
    print("\n4. TEST: Malformed TensorData (missing 'data' key)")

    datum_malformed = {
        "model_input": {"chunks": [{"tokens": input_tokens, "type": "encoded_text"}]},
        "loss_fn_inputs": {
            "target_tokens": target_tokens,  # plain list, not TensorData format
            "weights": weights,
            "logprobs": logprobs,
            "advantages": advantages,
        },
    }

    resp = requests.post(
        f"{BASE_URL}/api/v1/forward_backward",
        json={
            "model_id": model_id,
            "forward_backward_input": {"data": [datum_malformed], "loss_fn": "importance_sampling"},
        },
    )
    resp.raise_for_status()
    result = poll_future(resp.json()["request_id"])
    if "error" in result:
        print(f"   FAILED: {result['error']}")
    else:
        print(f"   SUCCESS: {len(result.get('loss_fn_outputs', []))} outputs")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("If all tests pass, the issue claim is INCORRECT.")
    print("The server handles TensorData format correctly.")


if __name__ == "__main__":
    main()
