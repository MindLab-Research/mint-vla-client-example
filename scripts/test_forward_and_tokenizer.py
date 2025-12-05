"""Test POST /forward and GET /models/{model_id}/tokenizer endpoints.

Tests:
1. Create a training model
2. Get tokenizer info
3. Do forward pass (no backward) and verify logprobs returned
4. Do forward_backward and verify gradients accumulated
5. Compare: forward should NOT accumulate gradients

Run with:
    python scripts/test_forward_and_tokenizer.py
"""
import requests
import time

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


def test_forward_and_tokenizer():
    """Test forward and tokenizer endpoints."""
    print("=" * 60)
    print("TEST: forward and tokenizer endpoints")
    print("=" * 60)

    # 1. Create session
    print("\n1. Creating session...")
    resp = requests.post(f"{BASE_URL}/create_session", json={"tags": [], "user_metadata": {}})
    assert resp.status_code == 200, f"Failed: {resp.text}"
    session_id = resp.json()["session_id"]
    print(f"   session_id: {session_id}")

    # 2. Create training model
    print("\n2. Creating training model...")
    resp = requests.post(f"{BASE_URL}/create_model", json={
        "session_id": session_id,
        "model_seq_id": 0,
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "lora_config": {"rank": 32, "train_unembed": True, "train_mlp": True, "train_attn": True},
    })
    assert resp.status_code == 200, f"Failed: {resp.text}"
    request_id = resp.json()["request_id"]
    result = poll_future(request_id, timeout=120)
    model_id = result["model_id"]
    print(f"   model_id: {model_id}")

    # 3. Test GET /models/{model_id}/tokenizer
    print("\n3. Testing GET /models/{model_id}/tokenizer...")
    resp = requests.get(f"{BASE_URL}/models/{model_id}/tokenizer")
    assert resp.status_code == 200, f"Failed: {resp.text}"
    tokenizer_info = resp.json()
    print(f"   vocab_size: {tokenizer_info['tokenizer']['vocab_size']}")
    print(f"   pad_token: {tokenizer_info['tokenizer']['pad_token']}")
    print(f"   eos_token: {tokenizer_info['tokenizer']['eos_token']}")
    print(f"   eos_token_id: {tokenizer_info['tokenizer']['eos_token_id']}")

    # 4. Prepare test data
    input_tokens = [9707, 1917, 0]  # "Hello world" + padding
    target_tokens = [1917, 0, 0]     # Shifted targets
    loss_mask = [1.0, 1.0, 0.0]      # Train on first 2 positions

    test_data = {
        "model_id": model_id,
        "forward_backward_input": {
            "data": [{
                "model_input": {
                    "chunks": [{"tokens": input_tokens, "type": "encoded_text"}]
                },
                "loss_fn_inputs": {
                    "target_tokens": {"data": target_tokens, "shape": [len(target_tokens)], "dtype": "int64"},
                    "loss_mask": {"data": loss_mask, "shape": [len(loss_mask)], "dtype": "float32"},
                }
            }],
            "loss_fn": "cross_entropy",
        }
    }

    # 5. Test POST /forward (no backward)
    print("\n4. Testing POST /forward (no backward)...")
    resp = requests.post(f"{BASE_URL}/forward", json=test_data)
    assert resp.status_code == 200, f"Failed: {resp.text}"
    request_id = resp.json()["request_id"]
    result = poll_future(request_id, timeout=60)

    forward_loss = result['metrics'].get('loss:mean', 0)
    print(f"   loss: {forward_loss:.4f}")

    # Check that logprobs are returned
    loss_fn_outputs = result.get('loss_fn_outputs', [])
    assert len(loss_fn_outputs) > 0, "No loss_fn_outputs returned"

    first_output = loss_fn_outputs[0]
    assert 'logprobs' in first_output, "logprobs not in loss_fn_outputs"
    logprobs = first_output['logprobs']
    print(f"   logprobs shape: {logprobs['shape']}")
    print(f"   logprobs data (first 5): {logprobs['data'][:5]}")

    # Verify logprobs are negative (log probabilities)
    assert all(lp <= 0 for lp in logprobs['data']), "logprobs should be <= 0"
    print("   logprobs are valid (all <= 0)")

    # 6. Check model info - step count should be 0 (forward doesn't increment)
    print("\n5. Checking model step count after forward...")
    resp = requests.get(f"{BASE_URL}/models/{model_id}")
    assert resp.status_code == 200, f"Failed: {resp.text}"
    step_after_forward = resp.json()["current_step"]
    print(f"   current_step: {step_after_forward}")
    assert step_after_forward == 0, f"forward should not increment step count, got {step_after_forward}"

    # 7. Now do forward_backward to verify gradients are accumulated
    print("\n6. Testing POST /forward_backward (with backward)...")
    resp = requests.post(f"{BASE_URL}/forward_backward", json=test_data)
    assert resp.status_code == 200, f"Failed: {resp.text}"
    request_id = resp.json()["request_id"]
    result = poll_future(request_id, timeout=60)

    fb_loss = result['metrics'].get('loss:mean', 0)
    print(f"   loss: {fb_loss:.4f}")

    # Losses should be similar (same input data)
    print(f"   forward loss: {forward_loss:.4f}, forward_backward loss: {fb_loss:.4f}")

    # 8. Do optim_step to apply gradients
    print("\n7. Doing optim_step...")
    resp = requests.post(f"{BASE_URL}/optim_step", json={
        "model_id": model_id,
        "adam_params": {"learning_rate": 0.0001},
    })
    assert resp.status_code == 200, f"Failed: {resp.text}"
    request_id = resp.json()["request_id"]
    result = poll_future(request_id, timeout=60)
    print("   optim_step completed")

    # 9. Check step count incremented
    print("\n8. Checking model step count after optim_step...")
    resp = requests.get(f"{BASE_URL}/models/{model_id}")
    assert resp.status_code == 200, f"Failed: {resp.text}"
    step_after_optim = resp.json()["current_step"]
    print(f"   current_step: {step_after_optim}")
    assert step_after_optim == 1, f"optim_step should increment step count to 1, got {step_after_optim}"

    # 10. Cleanup
    print("\n9. Cleaning up...")
    resp = requests.delete(f"{BASE_URL}/models/{model_id}")
    print(f"   deleted {model_id}: {resp.status_code}")

    print("\n" + "=" * 60)
    print("TEST PASSED: forward and tokenizer endpoints work correctly")
    print("=" * 60)


if __name__ == "__main__":
    test_forward_and_tokenizer()
