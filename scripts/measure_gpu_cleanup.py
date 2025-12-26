#!/usr/bin/env python3
"""Measure GPU cleanup time after killing K2 Megatron actor.

This script:
1. Creates K2 training model (64 GPUs)
2. Waits for full initialization
3. Kills the Ray actor
4. Polls Ray resources AND nvidia-smi at intervals
5. Records time until CUDA memory is actually freed
"""
import os
import sys
import time
import json
import requests
import subprocess

BASE_URL = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("TINKER_API_KEY", "dummy")

def get_headers():
    return {"X-API-Key": API_KEY}

def poll_future(request_id, timeout=900):
    poll_url = f"{BASE_URL}/api/v1/retrieve_future"
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.post(poll_url, json={"request_id": request_id}, headers=get_headers(), timeout=300)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 408:
            time.sleep(2)
        else:
            raise RuntimeError(f"poll error: {resp.status_code} {resp.text}")
    raise RuntimeError(f"Timeout after {timeout}s")

def create_k2_model():
    """Create K2 training model and wait for initialization."""
    import uuid
    session_id = f"cleanup_test_{uuid.uuid4().hex[:8]}"
    model_seq_id = 1

    print("Creating K2 training model (64 GPUs, TP=8, EP=8)...")
    resp = requests.post(
        f"{BASE_URL}/api/v1/create_model",
        json={
            "session_id": session_id,
            "model_seq_id": model_seq_id,
            "base_model": "moonshotai/Kimi-K2-Thinking",
            "tensor_parallel_size": 8,
            "expert_parallel_size": 8,
            "moe_config": {"num_shared_experts": 1},
            "lora_config": {"rank": 32},
            "sequence_parallel": True,
        },
        headers=get_headers(),
        timeout=60,
    )
    resp.raise_for_status()
    request_id = resp.json()["request_id"]
    print(f"  request_id: {request_id}")

    result = poll_future(request_id, timeout=900)
    if "error" in result:
        raise RuntimeError(f"create_model failed: {result['error']}")

    model_id = result.get("model_id", f"{session_id}_{model_seq_id}")
    print(f"  model_id: {model_id}")
    return model_id

def get_ray_gpu_status():
    """Get Ray GPU availability via SSH."""
    cmd = """ssh volcano 'python3 << "PYEOF"
import ray
ray.init(address="auto", ignore_reinit_error=True)
r = ray.available_resources()
t = ray.cluster_resources()
gpu_key = "GPU"
avail = r.get(gpu_key, 0)
total = t.get(gpu_key, 0)
print(f"{avail:.0f}/{total:.0f}")
PYEOF' 2>/dev/null | tail -1"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip()

def get_nvidia_smi_memory():
    """Get actual GPU memory usage via nvidia-smi across all nodes."""
    # Check memory on first worker node (has GPUs)
    cmd = """ssh volcano 'ssh 192.168.32.92 "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits" 2>/dev/null | head -1'"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    try:
        return int(result.stdout.strip())
    except:
        return -1

def kill_k2_actor():
    """Kill the K2 Megatron actor."""
    cmd = """ssh volcano 'python3 << "PYEOF"
import ray
ray.init(address="auto", ignore_reinit_error=True)
try:
    actor = ray.get_actor("megatron_kimi_k2_thinking", namespace="tinker")
    ray.kill(actor, no_restart=True)
    print("killed")
except Exception as e:
    print(f"error: {e}")
PYEOF' 2>/dev/null | tail -1"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip()

def main():
    print("=" * 60)
    print("K2 GPU Cleanup Time Measurement")
    print("=" * 60)

    # Check initial state
    ray_status = get_ray_gpu_status()
    print(f"Initial Ray GPUs: {ray_status}")

    if not ray_status.startswith("64"):
        print("ERROR: Not all 64 GPUs available. Run cleanup first.")
        sys.exit(1)

    # Create K2 model
    start_time = time.time()
    model_id = create_k2_model()
    create_time = time.time() - start_time
    print(f"Model created in {create_time:.1f}s")

    # Verify GPUs are allocated
    ray_status = get_ray_gpu_status()
    print(f"After creation - Ray GPUs: {ray_status}")

    # Kill the actor
    print("\n" + "=" * 60)
    print("Killing K2 actor and measuring cleanup time...")
    print("=" * 60)

    kill_start = time.time()
    kill_result = kill_k2_actor()
    print(f"Kill result: {kill_result}")

    # Poll at intervals
    measurements = []
    for i in range(120):  # Poll for up to 2 minutes
        elapsed = time.time() - kill_start
        ray_gpus = get_ray_gpu_status()
        nvidia_mem = get_nvidia_smi_memory()

        measurements.append({
            "elapsed_sec": elapsed,
            "ray_gpus": ray_gpus,
            "nvidia_mem_mb": nvidia_mem,
        })

        print(f"[{elapsed:6.1f}s] Ray: {ray_gpus:>8}, nvidia-smi mem: {nvidia_mem:>6} MB")

        # Check if fully released
        if ray_gpus.startswith("64") and nvidia_mem < 1000:
            print(f"\nGPU cleanup complete at {elapsed:.1f}s")
            break

        time.sleep(1)

    # Save results
    results = {
        "create_time_sec": create_time,
        "measurements": measurements,
    }
    with open("/tmp/k2_cleanup_timing.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to /tmp/k2_cleanup_timing.json")

    # Summary
    ray_free_time = None
    cuda_free_time = None
    for m in measurements:
        if ray_free_time is None and m["ray_gpus"].startswith("64"):
            ray_free_time = m["elapsed_sec"]
        if cuda_free_time is None and m["nvidia_mem_mb"] < 1000:
            cuda_free_time = m["elapsed_sec"]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Time for Ray to report GPUs free: {ray_free_time:.1f}s" if ray_free_time else "Ray GPUs never freed")
    print(f"Time for CUDA memory release:     {cuda_free_time:.1f}s" if cuda_free_time else "CUDA memory never freed")
    if ray_free_time and cuda_free_time:
        gap = cuda_free_time - ray_free_time
        print(f"Gap (CUDA - Ray):                 {gap:.1f}s")

if __name__ == "__main__":
    main()
