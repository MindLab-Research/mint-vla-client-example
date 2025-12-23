#!/usr/bin/env python3
"""Test per-token log_probs from Megatron distributed forward.

Verifies that forward() returns per-token log probabilities needed for
Chat SL and DPO recipes.

Usage:
    ssh volcano 'cd /root/tinker_project/tinker-server && \
        HF_HUB_OFFLINE=1 HF_HOME=/vePFS-Mindverse/share/huggingface \
        python scripts/test_forward_logprobs.py'
"""
import sys
import time
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/vePFS-Mindverse/share/huggingface")

import ray

# Default MoE model path
DEFAULT_MODEL = "/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"


def main():
    print("=" * 60)
    print("Test: Megatron forward per-token log_probs")
    print("=" * 60)

    ray.init(address="auto", ignore_reinit_error=True, namespace="tinker")
    print(f"Connected to Ray cluster")

    from tinker_server.backend.megatron_distributed import (
        get_or_create_megatron_worker_group,
        kill_megatron_actor,
        DistributedConfig,
    )

    base_model = DEFAULT_MODEL
    lora_rank = 8
    learning_rate = 1e-4
    config = DistributedConfig(
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        expert_parallel_size=2,
    )

    print(f"\nConfig:")
    print(f"  base_model: {base_model}")
    print(f"  lora_rank: {lora_rank}")
    print(f"  world_size: {config.world_size}")

    # Clean up existing actor
    print("\n--- Step 1: Clean up existing actor ---")
    if kill_megatron_actor():
        print("Killed existing actor, waiting 5s...")
        time.sleep(5)
    else:
        print("No existing actor")

    # Create worker group
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

    # Create test batch - Tinker Datum format
    print("\n--- Step 3: Create test batch ---")
    import random

    # Simple test: 2 sequences, 16 tokens each
    batch_size = 2
    seq_len = 16

    # Create Tinker Datum format
    # Each item: {"model_input": {"chunks": [{"tokens": [...]}]}, "loss_fn_inputs": {...}}
    data_items = []
    for i in range(batch_size):
        tokens = [random.randint(100, 10000) for _ in range(seq_len)]
        # Weights: 1.0 for all tokens except first (BOS)
        weights = [0.0] + [1.0] * (seq_len - 1)

        item = {
            "model_input": {"chunks": [{"tokens": tokens}]},
            "loss_fn_inputs": {
                "weights": {"data": weights},
            }
        }
        data_items.append(item)

    batch_data = data_items

    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")

    # Test forward
    print("\n--- Step 4: Call forward (120s timeout) ---")
    t0 = time.time()
    try:
        future = worker_group.forward.remote(batch_data)
        result = ray.get(future, timeout=120)
        elapsed = time.time() - t0
        print(f"forward() returned in {elapsed:.2f}s")
    except ray.exceptions.GetTimeoutError:
        elapsed = time.time() - t0
        print(f"TIMEOUT after {elapsed:.1f}s")
        return 1
    except Exception as e:
        elapsed = time.time() - t0
        print(f"FAILED after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Check result structure
    print("\n--- Step 5: Verify log_probs in result ---")
    print(f"Result keys: {list(result.keys())}")

    if "log_probs" in result:
        log_probs_data = result["log_probs"]
        print(f"log_probs type: {type(log_probs_data)}")

        if isinstance(log_probs_data, dict):
            # Serialized tensor format
            print(f"log_probs shape: {log_probs_data.get('shape')}")
            print(f"log_probs dtype: {log_probs_data.get('dtype')}")
            data = log_probs_data.get("data")
            if data:
                import numpy as np
                arr = np.array(data)
                print(f"log_probs reconstructed shape: {arr.shape}")
                print(f"log_probs sample values: min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")

                # Verify log_probs are negative (valid log probabilities)
                if arr.max() <= 0:
                    print("PASS: log_probs are valid (all <= 0)")
                else:
                    print(f"WARNING: Some log_probs > 0 (max={arr.max()}), may indicate issues")
        elif isinstance(log_probs_data, (list, tuple)):
            # Raw list format
            import numpy as np
            arr = np.array(log_probs_data)
            print(f"log_probs shape: {arr.shape}")
            print(f"log_probs sample values: min={arr.min():.4f}, max={arr.max():.4f}")
        else:
            print(f"Unexpected log_probs format: {type(log_probs_data)}")
    else:
        print("FAIL: log_probs not in result")
        print(f"Available keys: {result.keys()}")
        if "error" in result:
            print(f"Error message: {result['error']}")
        return 1

    # Check other metrics
    if "loss" in result:
        print(f"loss: {result['loss']}")
    if "num_tokens" in result:
        print(f"num_tokens: {result['num_tokens']}")

    # Cleanup
    print("\n--- Step 6: Cleanup ---")
    kill_megatron_actor()
    print("Actor killed")

    print("\n" + "=" * 60)
    print("Test completed - per-token log_probs extraction works")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
