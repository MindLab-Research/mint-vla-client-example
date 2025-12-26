#!/usr/bin/env python3
"""Measure GPU cleanup time after killing K2 Megatron actor.

Directly uses Ray on the server to avoid API complexities.
"""
import subprocess
import time
import json


def ssh_run(cmd):
    """Run command on volcano via SSH."""
    full_cmd = f"ssh volcano '{cmd}'"
    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip(), result.returncode


def get_ray_gpus():
    """Get Ray GPU availability."""
    cmd = """python3 << "PYEOF"
import ray
ray.init(address="auto", ignore_reinit_error=True)
r = ray.available_resources()
t = ray.cluster_resources()
gpu_key = "GPU"
print(f"{r.get(gpu_key, 0):.0f}/{t.get(gpu_key, 0):.0f}")
PYEOF"""
    out, _ = ssh_run(cmd)
    return out.split("\n")[-1]


def get_nvidia_smi_mem():
    """Get GPU memory on first worker node."""
    # Direct SSH to a worker node
    cmd = """ssh 192.168.32.92 'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1'"""
    out, _ = ssh_run(cmd)
    try:
        return int(out)
    except:
        return -1


def cleanup_gpus():
    """Remove all placement groups."""
    cmd = """python3 << "PYEOF"
import ray
from ray.util.placement_group import PlacementGroup
ray.init(address="auto", ignore_reinit_error=True)

# Kill actor
try:
    actor = ray.get_actor("megatron_kimi_k2_thinking", namespace="tinker")
    ray.kill(actor, no_restart=True)
    print("killed actor")
except:
    pass

# Remove PGs
pgs = ray.util.placement_group_table()
for pg_id, pg_info in pgs.items():
    if pg_info.get("state") != "REMOVED":
        bundles = pg_info.get("bundles", {})
        total_gpus = sum(b.get("GPU", 0) for b in bundles.values())
        if total_gpus > 0:
            try:
                pg = PlacementGroup(ray.PlacementGroupID.from_hex(pg_id))
                ray.util.remove_placement_group(pg)
                print(f"removed pg with {total_gpus} GPUs")
            except:
                pass

import time
time.sleep(2)

r = ray.available_resources()
gpu_key = "GPU"
print(f"available: {r.get(gpu_key, 0):.0f}")
PYEOF"""
    out, _ = ssh_run(cmd)
    return "available: 64" in out


def create_k2_model_sync():
    """Create K2 model synchronously via Ray."""
    cmd = '''python3 << "PYEOF"
import ray
import os
import sys
import time

# Set env
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/vePFS-Mindverse/share/huggingface"

ray.init(address="auto", ignore_reinit_error=True)

# Import after ray init
sys.path.insert(0, "/root/tinker_project/tinker-server")
from tinker_server.backend.megatron_distributed import MegatronActorPool

pool = MegatronActorPool()
pool.config.actor_eviction = False  # Disable eviction for test

print("Creating K2 actor...")
start = time.time()

actor = pool.get(
    model_path="moonshotai/Kimi-K2-Thinking",
    tensor_parallel_size=8,
    expert_parallel_size=8,
    moe_config={"num_shared_experts": 1},
    lora_config={"rank": 32},
    sequence_parallel=True,
    actor_name="megatron_kimi_k2_thinking",
)

elapsed = time.time() - start
print(f"Created in {elapsed:.1f}s")

# Verify actor is ready
diag = ray.get(actor.get_diagnostics.remote(), timeout=30)
print(f"Actor ready: {diag.get('status', 'unknown')}")
PYEOF'''
    print("Creating K2 model on server...")
    out, code = ssh_run(cmd)
    print(out)
    return code == 0


def kill_k2_actor():
    """Kill the K2 actor."""
    cmd = """python3 << "PYEOF"
import ray
ray.init(address="auto", ignore_reinit_error=True)
try:
    actor = ray.get_actor("megatron_kimi_k2_thinking", namespace="tinker")
    ray.kill(actor, no_restart=True)
    print("killed")
except Exception as e:
    print(f"error: {e}")
PYEOF"""
    out, _ = ssh_run(cmd)
    return "killed" in out


def main():
    print("=" * 60)
    print("K2 GPU Cleanup Time Measurement (Direct)")
    print("=" * 60)

    # Step 1: Cleanup
    print("\nStep 1: Cleanup existing resources...")
    if not cleanup_gpus():
        print("WARNING: Could not fully cleanup")
    time.sleep(3)

    # Step 2: Verify clean state
    ray_gpus = get_ray_gpus()
    print(f"Ray GPUs: {ray_gpus}")
    if not ray_gpus.startswith("64"):
        print("ERROR: Not all GPUs available")
        return

    # Step 3: Create K2 model
    print("\nStep 2: Create K2 model...")
    start = time.time()
    if not create_k2_model_sync():
        print("ERROR: Failed to create model")
        return
    create_time = time.time() - start
    print(f"Model creation took: {create_time:.1f}s")

    # Step 4: Verify GPUs allocated
    time.sleep(2)
    ray_gpus = get_ray_gpus()
    print(f"After creation - Ray GPUs: {ray_gpus}")

    # Step 5: Kill actor and measure
    print("\n" + "=" * 60)
    print("Step 3: Kill actor and measure cleanup time")
    print("=" * 60)

    kill_start = time.time()
    if not kill_k2_actor():
        print("WARNING: Kill may have failed")

    measurements = []
    for i in range(90):  # Poll for 90 seconds
        elapsed = time.time() - kill_start
        ray_gpus = get_ray_gpus()
        nvidia_mem = get_nvidia_smi_mem()

        measurements.append({
            "elapsed_sec": round(elapsed, 1),
            "ray_gpus": ray_gpus,
            "nvidia_mem_mb": nvidia_mem,
        })

        print(f"[{elapsed:5.1f}s] Ray: {ray_gpus:>8}, nvidia-smi: {nvidia_mem:>6} MB")

        # Check if done
        if ray_gpus.startswith("64") and nvidia_mem < 2000:
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

    # Summary
    ray_free_time = None
    cuda_free_time = None
    for m in measurements:
        if ray_free_time is None and m["ray_gpus"].startswith("64"):
            ray_free_time = m["elapsed_sec"]
        if cuda_free_time is None and m["nvidia_mem_mb"] > 0 and m["nvidia_mem_mb"] < 2000:
            cuda_free_time = m["elapsed_sec"]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model creation time:              {create_time:.1f}s")
    if ray_free_time is not None:
        print(f"Time for Ray to report GPUs free: {ray_free_time:.1f}s")
    else:
        print("Ray GPUs never freed within measurement window")
    if cuda_free_time is not None:
        print(f"Time for CUDA memory release:     {cuda_free_time:.1f}s")
    else:
        print("CUDA memory never freed within measurement window")
    if ray_free_time and cuda_free_time:
        gap = cuda_free_time - ray_free_time
        print(f"Gap (CUDA - Ray):                 {gap:.1f}s")


if __name__ == "__main__":
    main()
