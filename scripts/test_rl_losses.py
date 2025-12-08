"""Test RL loss functions: importance_sampling and ppo.

Tests:
1. Create a training model
2. Test forward to get baseline logprobs
3. Test importance_sampling loss with fake old logprobs and advantages
4. Test ppo loss with clipping
5. Verify RL-specific metrics are returned

Run with:
    python scripts/test_rl_losses.py
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


def test_rl_losses():
    """Test importance_sampling and ppo loss functions."""
    print("=" * 60)
    print("TEST: RL loss functions (importance_sampling, ppo)")
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

    # 3. Test forward to get baseline logprobs
    print("\n3. Getting baseline logprobs with forward...")
    input_tokens = [9707, 1917, 0]  # "Hello world" + padding
    target_tokens = [1917, 0, 0]     # Shifted targets
    loss_mask = [1.0, 1.0, 0.0]      # Train on first 2 positions

    forward_data = {
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

    resp = requests.post(f"{BASE_URL}/forward", json=forward_data)
    assert resp.status_code == 200, f"Failed: {resp.text}"
    request_id = resp.json()["request_id"]
    result = poll_future(request_id, timeout=60)

    # Extract logprobs from forward pass
    old_logprobs = result["loss_fn_outputs"][0]["logprobs"]["data"]
    print(f"   old_logprobs: {old_logprobs}")

    # 4. Test importance_sampling loss
    print("\n4. Testing importance_sampling loss...")
    # Create fake advantages (positive = good action, negative = bad action)
    advantages = [1.0, -0.5, 0.0]  # Encourage first token, discourage second

    is_data = {
        "model_id": model_id,
        "forward_backward_input": {
            "data": [{
                "model_input": {
                    "chunks": [{"tokens": input_tokens, "type": "encoded_text"}]
                },
                "loss_fn_inputs": {
                    "target_tokens": {"data": target_tokens, "shape": [len(target_tokens)], "dtype": "int64"},
                    "loss_mask": {"data": loss_mask, "shape": [len(loss_mask)], "dtype": "float32"},
                    "logprobs": {"data": old_logprobs, "shape": [len(old_logprobs)], "dtype": "float32"},
                    "advantages": {"data": advantages, "shape": [len(advantages)], "dtype": "float32"},
                }
            }],
            "loss_fn": "importance_sampling",
        }
    }

    resp = requests.post(f"{BASE_URL}/forward_backward", json=is_data)
    assert resp.status_code == 200, f"Failed: {resp.text}"
    request_id = resp.json()["request_id"]
    result = poll_future(request_id, timeout=60)

    is_loss = result["metrics"].get("loss:mean", 0)
    is_ratio = result["metrics"].get("ratio:mean", 0)
    print(f"   loss: {is_loss:.4f}")
    print(f"   ratio:mean: {is_ratio:.4f}")
    print(f"   loss_fn_output_type: {result['loss_fn_output_type']}")

    # ratio should be ~1.0 since we're using the same policy's logprobs
    assert 0.9 < is_ratio < 1.1, f"ratio should be ~1.0 when using same policy, got {is_ratio}"
    print("   ratio is ~1.0 as expected (same policy)")

    # 5. Test PPO loss with default epsilon
    print("\n5. Testing ppo loss (default epsilon=0.2)...")
    ppo_data = {
        "model_id": model_id,
        "forward_backward_input": {
            "data": [{
                "model_input": {
                    "chunks": [{"tokens": input_tokens, "type": "encoded_text"}]
                },
                "loss_fn_inputs": {
                    "target_tokens": {"data": target_tokens, "shape": [len(target_tokens)], "dtype": "int64"},
                    "loss_mask": {"data": loss_mask, "shape": [len(loss_mask)], "dtype": "float32"},
                    "logprobs": {"data": old_logprobs, "shape": [len(old_logprobs)], "dtype": "float32"},
                    "advantages": {"data": advantages, "shape": [len(advantages)], "dtype": "float32"},
                }
            }],
            "loss_fn": "ppo",
        }
    }

    resp = requests.post(f"{BASE_URL}/forward_backward", json=ppo_data)
    assert resp.status_code == 200, f"Failed: {resp.text}"
    request_id = resp.json()["request_id"]
    result = poll_future(request_id, timeout=60)

    ppo_loss = result["metrics"].get("loss:mean", 0)
    ppo_ratio = result["metrics"].get("ratio:mean", 0)
    ppo_clipfrac = result["metrics"].get("clipfrac:mean", 0)
    print(f"   loss: {ppo_loss:.4f}")
    print(f"   ratio:mean: {ppo_ratio:.4f}")
    print(f"   clipfrac:mean: {ppo_clipfrac:.4f}")
    print(f"   loss_fn_output_type: {result['loss_fn_output_type']}")

    # clipfrac should be low since ratio ~1.0 is within [0.8, 1.2]
    assert ppo_clipfrac < 0.5, f"clipfrac should be low for ratio~1.0, got {ppo_clipfrac}"
    print("   clipfrac is low as expected (ratio within clip range)")

    # 6. Test PPO with custom epsilon
    print("\n6. Testing ppo loss with custom epsilon=0.1...")
    ppo_data["forward_backward_input"]["loss_fn_config"] = {"epsilon": 0.1}

    resp = requests.post(f"{BASE_URL}/forward_backward", json=ppo_data)
    assert resp.status_code == 200, f"Failed: {resp.text}"
    request_id = resp.json()["request_id"]
    result = poll_future(request_id, timeout=60)

    ppo_loss_tight = result["metrics"].get("loss:mean", 0)
    ppo_clipfrac_tight = result["metrics"].get("clipfrac:mean", 0)
    print(f"   loss: {ppo_loss_tight:.4f}")
    print(f"   clipfrac:mean: {ppo_clipfrac_tight:.4f}")

    # 7. Test with different old logprobs to trigger clipping
    print("\n7. Testing ppo with shifted old_logprobs to trigger clipping...")
    # Shift old logprobs significantly to create large ratio
    shifted_logprobs = [lp - 2.0 for lp in old_logprobs]  # exp(2) ~ 7.4 ratio

    ppo_data_shifted = {
        "model_id": model_id,
        "forward_backward_input": {
            "data": [{
                "model_input": {
                    "chunks": [{"tokens": input_tokens, "type": "encoded_text"}]
                },
                "loss_fn_inputs": {
                    "target_tokens": {"data": target_tokens, "shape": [len(target_tokens)], "dtype": "int64"},
                    "loss_mask": {"data": loss_mask, "shape": [len(loss_mask)], "dtype": "float32"},
                    "logprobs": {"data": shifted_logprobs, "shape": [len(shifted_logprobs)], "dtype": "float32"},
                    "advantages": {"data": advantages, "shape": [len(advantages)], "dtype": "float32"},
                }
            }],
            "loss_fn": "ppo",
        }
    }

    resp = requests.post(f"{BASE_URL}/forward_backward", json=ppo_data_shifted)
    assert resp.status_code == 200, f"Failed: {resp.text}"
    request_id = resp.json()["request_id"]
    result = poll_future(request_id, timeout=60)

    ppo_ratio_high = result["metrics"].get("ratio:mean", 0)
    ppo_clipfrac_high = result["metrics"].get("clipfrac:mean", 0)
    print(f"   ratio:mean: {ppo_ratio_high:.4f}")
    print(f"   clipfrac:mean: {ppo_clipfrac_high:.4f}")

    # With large ratio difference, clipfrac should be high
    assert ppo_clipfrac_high > 0.5, f"clipfrac should be high for large ratio, got {ppo_clipfrac_high}"
    print("   clipfrac is high as expected (ratio outside clip range)")

    # 8. Cleanup
    print("\n8. Cleaning up...")
    resp = requests.delete(f"{BASE_URL}/models/{model_id}")
    print(f"   deleted {model_id}: {resp.status_code}")

    print("\n" + "=" * 60)
    print("TEST PASSED: RL loss functions work correctly")
    print("=" * 60)


if __name__ == "__main__":
    test_rl_losses()
