"""Test Issue 14: Inferencer re-creation after kill.

Scenario:
1. Client starts sampling session
2. Does sampling successfully
3. vLLM inferencer is killed (simulating resource exhaustion or error)
4. Second sampling attempt should re-create the inferencer and succeed
"""
import requests
import time
import uuid
import sys

BASE_URL = "http://localhost:8000"
MODEL = "Qwen/Qwen2.5-7B-Instruct"


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


def sample(session_id, timeout=180):
    """Sample from a session."""
    resp = requests.post(
        f"{BASE_URL}/api/v1/asample",
        json={
            "sampling_session_id": session_id,
            "num_samples": 1,
            "prompt": {"chunks": [{"tokens": [1, 2, 3, 4, 5], "type": "encoded_text"}]},
            "sampling_params": {"max_tokens": 10, "temperature": 0.7},
        },
        headers=get_headers(),
        timeout=120,
    )
    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}: {resp.text}"}
    return poll_future(resp.json().get("request_id"), timeout=timeout)


def kill_vllm_for_model(model_name):
    """Kill the vLLM actor for a specific model via resource pool eviction."""
    # Get actor name for this model
    model_part = model_name.split("/")[-1].lower().replace(" ", "_")
    actor_name = f"tinker_vllm_{model_part}"

    # Use ray to kill directly since kill_vllm uses legacy name
    import subprocess
    kill_script = f'''
import ray
ray.init(address="auto", namespace="tinker", ignore_reinit_error=True)
try:
    actor = ray.get_actor("{actor_name}", namespace="tinker")
    ray.kill(actor)
    print("killed")
except ValueError:
    print("not_found")
'''
    result = subprocess.run(["python", "-c", kill_script], capture_output=True, text=True)
    return "killed" in result.stdout


def kill_vllm():
    """Kill the vLLM actor (legacy endpoint)."""
    resp = requests.post(
        f"{BASE_URL}/api/v1/kill_vllm",
        headers=get_headers(),
        timeout=30,
    )
    return resp.status_code == 200


def get_vllm_status():
    """Check vLLM actors via resource pool."""
    resp = requests.get(
        f"{BASE_URL}/api/v1/resource_pool",
        headers=get_headers(),
        timeout=10,
    )
    if resp.status_code == 200:
        data = resp.json()
        # Filter to vLLM actors
        vllm_actors = [a for a in data.get("actors", []) if "vllm" in a.get("actor_name", "")]
        return {"vllm_actors": len(vllm_actors), "actors": vllm_actors}
    return None


def main():
    print("=" * 60)
    print("ISSUE 14: INFERENCER RE-CREATION AFTER KILL")
    print("=" * 60)
    print(f"Model: {MODEL}")

    # Step 1: Create sampling session
    print("\n[1] Creating sampling session...")
    session_id = f"recovery_test_{uuid.uuid4().hex[:6]}"
    resp = requests.post(
        f"{BASE_URL}/api/v1/create_sampling_session",
        json={"session_id": session_id, "base_model": MODEL},
        headers=get_headers(),
        timeout=120,  # Increased timeout for vLLM init
    )
    if resp.status_code != 200:
        print(f"    FAILED: HTTP {resp.status_code}")
        return 1
    sampling_session_id = resp.json().get("sampling_session_id")
    print(f"    Session: {sampling_session_id}")

    # Step 2: First sampling (should trigger lazy vLLM init if not already up)
    print("\n[2] First sampling (may trigger vLLM init)...")
    start = time.time()
    result = sample(sampling_session_id)
    elapsed = time.time() - start
    if "error" in result:
        print(f"    FAILED: {result['error']}")
        return 1
    tokens = result.get("sequences", [{}])[0].get("tokens", [])
    print(f"    Generated {len(tokens)} tokens in {elapsed:.1f}s")

    # Check vLLM status
    status = get_vllm_status()
    print(f"    vLLM status: {status}")

    # Step 3: Kill vLLM actor for this specific model
    print("\n[3] Killing vLLM actor for this model...")
    if kill_vllm_for_model(MODEL):
        print("    vLLM actor killed")
    else:
        print("    WARNING: Could not kill vLLM (may already be dead)")

    # Verify it's dead
    time.sleep(2)
    status = get_vllm_status()
    print(f"    vLLM status after kill: {status}")

    # Step 4: Second sampling (should trigger re-creation, may take ~80s)
    print("\n[4] Second sampling (should re-create vLLM, may take ~80s)...")
    start = time.time()
    result = sample(sampling_session_id, timeout=300)  # 5 min for fresh vLLM init
    elapsed = time.time() - start

    if "error" in result:
        print(f"    FAILED: {result['error']}")
        print("\n[FAIL] Inferencer did not recover after kill")
        return 1

    tokens = result.get("sequences", [{}])[0].get("tokens", [])
    print(f"    Generated {len(tokens)} tokens in {elapsed:.1f}s")

    # Verify vLLM is back
    status = get_vllm_status()
    print(f"    vLLM status after recovery: {status}")

    print("\n" + "=" * 60)
    print("[PASS] Inferencer successfully re-created after kill")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
