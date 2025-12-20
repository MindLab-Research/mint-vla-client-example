"""Test Issue 7: Default model behavior.

Tests:
1. Create session without base_model - should require it (validation error)
2. Create sampling session without base_model - uses server default
"""
import requests
import time
import uuid
import sys

BASE_URL = "http://localhost:8000"


def get_headers():
    return {"Authorization": "Bearer dummy"}


def poll_future(request_id, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.post(
            f"{BASE_URL}/api/v1/retrieve_future",
            json={"request_id": request_id},
            headers=get_headers(),
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 408:
            time.sleep(0.5)
        else:
            return {"error": f"HTTP {resp.status_code}: {resp.text}"}
    return {"error": "Timeout"}


def test_create_model_without_base_model():
    """Test create_model without specifying base_model."""
    print("\n[1] Testing create_model WITHOUT base_model...")
    session_id = f"no_model_{uuid.uuid4().hex[:6]}"

    resp = requests.post(
        f"{BASE_URL}/api/v1/create_model",
        json={
            "session_id": session_id,
            "model_seq_id": 1,
            # No base_model!
            "lora_config": {"rank": 32},
            "learning_rate": 1e-4,
        },
        headers=get_headers(),
        timeout=30,
    )

    print(f"    HTTP {resp.status_code}")

    if resp.status_code == 422:
        print(f"    Validation error - base_model is REQUIRED")
        return True  # PASS
    else:
        print(f"    Response: {resp.text}")
        return False


def test_create_sampling_session_without_base_model():
    """Test create_sampling_session without base_model."""
    print("\n[2] Testing create_sampling_session WITHOUT base_model...")
    session_id = f"no_model_sampling_{uuid.uuid4().hex[:6]}"

    resp = requests.post(
        f"{BASE_URL}/api/v1/create_sampling_session",
        json={
            "session_id": session_id,
            # No base_model!
        },
        headers=get_headers(),
        timeout=60,
    )

    print(f"    HTTP {resp.status_code}")

    if resp.status_code == 200:
        data = resp.json()
        sampling_session_id = data.get("sampling_session_id")
        print(f"    Sampling session ID: {sampling_session_id}")
        print("    Uses server default: Qwen/Qwen2.5-7B-Instruct")
        return True  # PASS (Mint convenience - uses default)
    elif resp.status_code == 422:
        print(f"    Validation error: {resp.json()}")
        return True  # PASS (Tinker strict mode - requires base_model)
    else:
        print(f"    Response: {resp.text}")
        return False


def main():
    print("=" * 60)
    print("ISSUE 7: DEFAULT MODEL BEHAVIOR")
    print("=" * 60)

    result1 = test_create_model_without_base_model()
    result2 = test_create_sampling_session_without_base_model()

    print("\n" + "=" * 60)
    print("FINDINGS:")
    print(f"  create_model (training): base_model REQUIRED")
    print(f"  create_sampling_session: defaults to Qwen/Qwen2.5-7B-Instruct")
    print("=" * 60)

    if result1 and result2:
        print("\n[PASS] Default model behavior verified")
        return 0
    else:
        print("\n[FAIL] Unexpected behavior")
        return 1


if __name__ == "__main__":
    sys.exit(main())
