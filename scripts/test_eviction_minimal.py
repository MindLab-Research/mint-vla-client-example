"""Minimal test for resource eviction.

This test verifies that:
1. Dense session creation works
2. MoE session creation triggers eviction check
3. Server doesn't hang indefinitely when resources are tight
"""

import requests
import time
import os
import subprocess

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
TIMEOUT = 60  # seconds

def poll_future(request_id: str, timeout: float = 300) -> dict:
    """Poll until future is ready."""
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
            time.sleep(1)
            continue
        else:
            raise RuntimeError(f"Unexpected status {resp.status_code}: {resp.text}")
    raise TimeoutError(f"Future {request_id} did not complete in {timeout}s")


def check_gpu_status() -> dict:
    """Check GPU availability via Ray."""
    result = subprocess.run([
        "ssh", "volcano", "python3", "-c",
        """import ray
ray.init(address='auto', ignore_reinit_error=True)
r = ray.available_resources()
t = ray.cluster_resources()
avail = int(r.get('GPU', 0))
total = int(t.get('GPU', 0))
print(f'{avail},{total}')
"""
    ], capture_output=True, text=True, timeout=30)
    # Parse last non-empty line
    lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip() and ',' in l]
    if not lines:
        raise RuntimeError(f"Could not parse GPU status: {result.stdout}")
    parts = lines[-1].split(',')
    return {"available": int(parts[0]), "total": int(parts[1])}


def create_dense_session(session_id: str = "test_dense_eviction") -> str:
    """Create a dense training session."""
    resp = requests.post(
        f"{BASE_URL}/api/v1/create_model",
        json={
            "session_id": session_id,
            "model_seq_id": 1,
            "base_model": "Qwen/Qwen2.5-7B-Instruct",
            "lora_config": {
                "rank": 16,
            },
        },
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    request_id = resp.json()["request_id"]
    result = poll_future(request_id, timeout=120)
    return result.get("model_id", session_id)


def create_moe_session(session_id: str = "test_moe_eviction") -> str:
    """Create a MoE training session."""
    resp = requests.post(
        f"{BASE_URL}/api/v1/create_model",
        json={
            "session_id": session_id,
            "model_seq_id": 1,
            "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "lora_config": {
                "rank": 16,
            },
        },
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    request_id = resp.json()["request_id"]
    result = poll_future(request_id, timeout=300)
    return result.get("model_id", session_id)


def main():
    print("=" * 60)
    print("Resource Eviction Test")
    print("=" * 60)

    # Check initial GPU status
    print("\n1. Checking initial GPU status...")
    try:
        gpu_status = check_gpu_status()
        print(f"   GPUs: {gpu_status['available']} / {gpu_status['total']}")
    except Exception as e:
        print(f"   Failed to check GPU status: {e}")
        gpu_status = None

    # Create dense session
    print("\n2. Creating dense training session...")
    try:
        dense_id = create_dense_session()
        print(f"   Dense session created: {dense_id}")
    except Exception as e:
        print(f"   Failed to create dense session: {e}")
        return

    # Check GPU status after dense
    print("\n3. Checking GPU status after dense session...")
    try:
        gpu_status = check_gpu_status()
        print(f"   GPUs: {gpu_status['available']} / {gpu_status['total']}")
    except Exception as e:
        print(f"   Failed to check GPU status: {e}")

    # Try to create MoE session (needs 8 GPUs)
    print("\n4. Creating MoE training session (needs 8 GPUs)...")
    print("   This will test if eviction is triggered when resources are tight.")
    try:
        moe_id = create_moe_session()
        print(f"   MoE session created: {moe_id}")
    except TimeoutError as e:
        print(f"   TIMEOUT: {e}")
        print("   This indicates eviction did not work - session hung waiting for resources.")
    except Exception as e:
        print(f"   ERROR: {e}")
        # Check if it's a resource error (expected when eviction kicks in but no idle actors)
        if "Insufficient GPUs" in str(e):
            print("   This is expected if there are no idle actors to evict.")

    # Final GPU status
    print("\n5. Final GPU status...")
    try:
        gpu_status = check_gpu_status()
        print(f"   GPUs: {gpu_status['available']} / {gpu_status['total']}")
    except Exception as e:
        print(f"   Failed to check GPU status: {e}")

    print("\n" + "=" * 60)
    print("Test complete. Check server logs for eviction messages:")
    print("  ssh volcano 'grep -i evict /tmp/tinker_server.log | tail -20'")
    print("=" * 60)


if __name__ == "__main__":
    main()
