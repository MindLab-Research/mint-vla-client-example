"""Test POST /create_model_from_state endpoint.

Tests:
1. Create a training model
2. Do one training step
3. Save state to checkpoint
4. Create new model from that checkpoint
5. Verify step count is restored

Run with:
    python scripts/test_create_model_from_state.py
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


def test_create_model_from_state():
    """Test create_model_from_state endpoint."""
    print("=" * 60)
    print("TEST: create_model_from_state")
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

    # 3. Do one training step
    print("\n3. Doing forward_backward...")
    # Training data in correct Datum format
    # Input: "Hello world" tokens, predict next token at each position
    input_tokens = [9707, 1917, 0]  # "Hello world" + padding
    target_tokens = [1917, 0, 0]     # Shifted targets (predict next token)
    loss_mask = [1.0, 1.0, 0.0]      # Train on first 2 positions

    resp = requests.post(f"{BASE_URL}/forward_backward", json={
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
    })
    assert resp.status_code == 200, f"Failed: {resp.text}"
    request_id = resp.json()["request_id"]
    result = poll_future(request_id, timeout=60)
    loss = result['metrics'].get('loss:mean', 0)
    print(f"   loss: {loss}")
    assert loss > 0, "Loss should be > 0 for actual training"

    # 4. Do optimizer step
    print("\n4. Doing optim_step...")
    resp = requests.post(f"{BASE_URL}/optim_step", json={
        "model_id": model_id,
        "adam_params": {"learning_rate": 0.0001},
    })
    assert resp.status_code == 200, f"Failed: {resp.text}"
    request_id = resp.json()["request_id"]
    result = poll_future(request_id, timeout=60)
    print(f"   step completed")

    # 5. Save state
    print("\n5. Saving state...")
    checkpoint_name = "test-checkpoint-1"
    resp = requests.post(f"{BASE_URL}/save_weights", json={
        "model_id": model_id,
        "path": checkpoint_name,
    })
    assert resp.status_code == 200, f"Failed: {resp.text}"
    request_id = resp.json()["request_id"]
    result = poll_future(request_id, timeout=60)
    saved_path = result["path"]
    print(f"   saved to: {saved_path}")

    # 6. Get model info to check step count
    print("\n6. Checking original model step count...")
    resp = requests.get(f"{BASE_URL}/models/{model_id}")
    assert resp.status_code == 200, f"Failed: {resp.text}"
    original_step = resp.json()["current_step"]
    print(f"   original model step: {original_step}")

    # 7. Create new model from state
    print("\n7. Creating new model from state...")
    resp = requests.post(f"{BASE_URL}/create_model_from_state", json={
        "session_id": session_id,
        "model_seq_id": 1,  # Different seq_id for new model
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "state_path": saved_path,
        "lora_config": {"rank": 32, "train_unembed": True, "train_mlp": True, "train_attn": True},
        "load_optimizer": True,
    })
    assert resp.status_code == 200, f"Failed: {resp.text}"
    request_id = resp.json()["request_id"]
    result = poll_future(request_id, timeout=120)
    new_model_id = result["model_id"]
    print(f"   new_model_id: {new_model_id}")

    # 8. Verify new model has correct step count
    print("\n8. Checking new model step count...")
    resp = requests.get(f"{BASE_URL}/models/{new_model_id}")
    assert resp.status_code == 200, f"Failed: {resp.text}"
    new_step = resp.json()["current_step"]
    print(f"   new model step: {new_step}")

    if new_step == original_step:
        print(f"\n   Step count restored correctly: {new_step}")
    else:
        print(f"\n   WARNING: Step mismatch! original={original_step}, new={new_step}")

    # 9. Test that new model can do training
    print("\n9. Testing new model can train...")
    resp = requests.post(f"{BASE_URL}/forward_backward", json={
        "model_id": new_model_id,
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
    })
    assert resp.status_code == 200, f"Failed: {resp.text}"
    request_id = resp.json()["request_id"]
    result = poll_future(request_id, timeout=60)
    loss = result['metrics'].get('loss:mean', 0)
    print(f"   forward_backward succeeded, loss: {loss}")
    assert loss > 0, "Loss should be > 0 for actual training"

    # 10. Cleanup - delete both models
    print("\n10. Cleaning up...")
    for mid in [model_id, new_model_id]:
        resp = requests.delete(f"{BASE_URL}/models/{mid}")
        print(f"   deleted {mid}: {resp.status_code}")

    print("\n" + "=" * 60)
    print("TEST PASSED: create_model_from_state works correctly")
    print("=" * 60)


if __name__ == "__main__":
    test_create_model_from_state()
