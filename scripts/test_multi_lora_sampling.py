"""Test the multi-LoRA sampling migration.

Tests:
1. Create sampling session (no model_path) -> uses base model on multi-LoRA engine
2. Generate tokens using base model
3. Compute logprobs using base model

Run with:
    python scripts/test_multi_lora_sampling.py
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


def test_base_model_sampling():
    """Test create_sampling_session with base model (no LoRA)."""
    print("=" * 60)
    print("TEST: Base Model Sampling via Multi-LoRA Engine")
    print("=" * 60)

    # 1. Create session (required by create_sampling_session)
    print("\n1. Creating session...")
    resp = requests.post(f"{BASE_URL}/create_session", json={"tags": [], "user_metadata": {}})
    assert resp.status_code == 200, f"Failed: {resp.text}"
    session_id = resp.json()["session_id"]
    print(f"   session_id: {session_id}")

    # 2. Create sampling session without model_path (base model)
    print("\n2. Creating sampling session (base model)...")
    print("   (First call will initialize multi-LoRA engine ~60s)")
    start = time.time()
    resp = requests.post(f"{BASE_URL}/create_sampling_session", json={
        "session_id": session_id,
        # No model_path = base model
    })
    elapsed = time.time() - start
    assert resp.status_code == 200, f"Failed: {resp.text}"
    sampling_session_id = resp.json()["sampling_session_id"]
    print(f"   sampling_session_id: {sampling_session_id}")
    print(f"   Took {elapsed:.2f}s")

    # 3. Test sampling with base model
    print("\n3. Testing sampling (base model)...")
    # "Hello" token
    prompt_tokens = [9707]
    resp = requests.post(f"{BASE_URL}/asample", json={
        "sampling_session_id": sampling_session_id,
        "seq_id": 0,
        "num_samples": 1,
        "prompt": {"chunks": [{"tokens": prompt_tokens, "type": "encoded_text"}]},
        "sampling_params": {"max_tokens": 20, "temperature": 0.7}
    })
    assert resp.status_code == 200, f"Failed: {resp.text}"
    request_id = resp.json()["request_id"]
    print(f"   request_id: {request_id}")

    result = poll_future(request_id, timeout=60)
    print(f"   Generated {len(result['sequences'][0]['tokens'])} tokens")
    print(f"   Stop reason: {result['sequences'][0]['stop_reason']}")

    # 4. Test compute_logprobs with base model
    print("\n4. Testing compute_logprobs (base model)...")
    test_tokens = [9707, 1917]  # "Hello world" approximate
    resp = requests.post(f"{BASE_URL}/compute_logprobs", json={
        "sampling_session_id": sampling_session_id,
        "seq_id": 1,
        "sequence": {"chunks": [{"tokens": test_tokens, "type": "encoded_text"}]}
    })
    assert resp.status_code == 200, f"Failed: {resp.text}"
    request_id = resp.json()["request_id"]

    result = poll_future(request_id, timeout=60)
    logprobs = result["logprobs"]
    print(f"   Input: {len(test_tokens)} tokens")
    print(f"   Output: {len(logprobs)} logprobs")
    print(f"   Values: {logprobs}")
    assert len(logprobs) == len(test_tokens) - 1, "Length mismatch!"

    # 5. Test creating a second session (should be fast)
    print("\n5. Creating second sampling session (should be fast)...")
    start = time.time()
    resp = requests.post(f"{BASE_URL}/create_sampling_session", json={
        "session_id": session_id,
        # No model_path = base model
    })
    elapsed = time.time() - start
    assert resp.status_code == 200, f"Failed: {resp.text}"
    second_session_id = resp.json()["sampling_session_id"]
    print(f"   sampling_session_id: {second_session_id}")
    print(f"   Took {elapsed:.2f}s (should be <1s)")

    if elapsed < 5:
        print("   Fast session creation confirmed!")
    else:
        print("   WARNING: Second session took longer than expected")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    return sampling_session_id


if __name__ == "__main__":
    test_base_model_sampling()
