#!/usr/bin/env python3
"""Minimal repro for multi-worker get_lora_state_dict timeout issue.

Self-contained test that:
1. Creates MegatronWorkerGroup (spawns workers)
2. Calls get_lora_state_dict to test LoRA extraction
3. Cleans up actors

Usage:
    ssh volcano 'cd /root/tinker_project/tinker-server && \
        HF_HUB_OFFLINE=1 HF_HOME=/vePFS-Mindverse/share/huggingface \
        python scripts/test_worker_response.py'
"""
import sys
import time

# Must set before importing ray/torch
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/vePFS-Mindverse/share/huggingface")

import ray

# Default model path on PFS - Qwen3 30B MoE
DEFAULT_MODEL = "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"


def main():
    print("=" * 60)
    print("Self-contained test: MegatronWorkerGroup get_lora_state_dict")
    print("=" * 60)

    # Connect to existing Ray cluster
    ray.init(address="auto", ignore_reinit_error=True, namespace="tinker")
    print(f"Connected to Ray cluster")

    # Import after ray.init
    from tinker_server.backend.megatron_distributed import (
        get_or_create_megatron_worker_group,
        kill_megatron_actor,
        DistributedConfig,
    )

    # Configuration
    base_model = DEFAULT_MODEL
    lora_rank = 8
    learning_rate = 1e-4
    # MoE requires multiple GPUs with expert parallelism
    # Replicate main server setting: TP=4, EP=2 (8 GPUs total)
    config = DistributedConfig(
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        expert_parallel_size=2,
    )

    print(f"\nConfig:")
    print(f"  base_model: {base_model}")
    print(f"  lora_rank: {lora_rank}")
    print(f"  world_size: {config.world_size}")

    # Kill any existing actor first
    print("\n--- Step 1: Clean up existing actor ---")
    if kill_megatron_actor():
        print("Killed existing actor, waiting 5s...")
        time.sleep(5)
    else:
        print("No existing actor")

    # Create fresh worker group
    print("\n--- Step 2: Create MegatronWorkerGroup ---")
    t0 = time.time()
    try:
        worker_group = get_or_create_megatron_worker_group(
            base_model=base_model,
            lora_rank=lora_rank,
            learning_rate=learning_rate,
            distributed_config=config,
        )
        print(f"Worker group created in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"FAILED to create worker group: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test get_lora_state_dict
    print("\n--- Step 3: Call get_lora_state_dict (60s timeout) ---")
    t0 = time.time()
    try:
        future = worker_group.get_lora_state_dict.remote()
        result = ray.get(future, timeout=60)
        elapsed = time.time() - t0
        print(f"get_lora_state_dict returned {len(result)} params in {elapsed:.2f}s")
        if result:
            sample_keys = list(result.keys())[:3]
            print(f"Sample keys: {sample_keys}")
    except ray.exceptions.GetTimeoutError:
        elapsed = time.time() - t0
        print(f"TIMEOUT after {elapsed:.1f}s - workers not responding")
        print("This indicates NCCL deadlock or worker hang")
        return 1
    except Exception as e:
        elapsed = time.time() - t0
        print(f"FAILED after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Cleanup
    print("\n--- Step 4: Cleanup ---")
    kill_megatron_actor()
    print("Actor killed")

    print("\n" + "=" * 60)
    print("Test completed")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
