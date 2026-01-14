#!/usr/bin/env python3
"""Diagnose checkpoint loading by directly examining the checkpoint files.

This script bypasses Ray and examines checkpoint files directly to understand
what's being loaded and whether the values match expectations.
"""

import os
import torch

CHECKPOINT_PATH = "/vePFS-Mindverse/share/code/tinker-server/checkpoints/80e6bf97-56d9-4f3a-9872-7cc4b80a7379_0/debug_checkpoint_20260110_182006"


def main():
    print("=" * 70)
    print("CHECKPOINT DIAGNOSIS")
    print("=" * 70)
    print(f"Checkpoint path: {CHECKPOINT_PATH}")

    # List checkpoint files
    print("\n1. Checkpoint files:")
    files = sorted(os.listdir(CHECKPOINT_PATH))
    for f in files:
        path = os.path.join(CHECKPOINT_PATH, f)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  {f}: {size_mb:.2f} MB")

    # Load training metadata
    meta_path = os.path.join(CHECKPOINT_PATH, "training_meta.json")
    if os.path.exists(meta_path):
        import json
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"\n2. Training metadata:")
        for k, v in meta.items():
            print(f"  {k}: {v}")

    # Load rank 0 checkpoint
    print("\n3. Rank 0 checkpoint contents:")
    rank0_file = os.path.join(CHECKPOINT_PATH, "mp_rank_00_000_adapter.pt")
    if os.path.exists(rank0_file):
        ckpt = torch.load(rank0_file, map_location="cpu")
        print(f"  Top-level keys: {list(ckpt.keys())}")

        adapter_state = ckpt.get("adapter_state_dict", {})
        print(f"  adapter_state_dict: {len(adapter_state)} keys")

        # Sample some keys
        print(f"\n4. Sample adapter weights (rank 0):")
        for i, (key, val) in enumerate(list(adapter_state.items())[:5]):
            norm = val.float().norm().item()
            shape = list(val.shape)
            first5 = val.flatten()[:5].tolist()
            print(f"  {key}:")
            print(f"    shape={shape}, norm={norm:.6f}")
            print(f"    first5={first5}")

        # Check if any values are all zeros
        print(f"\n5. Check for zero-valued weights:")
        zero_count = 0
        nonzero_count = 0
        for key, val in adapter_state.items():
            if val.abs().max().item() < 1e-10:
                zero_count += 1
                if zero_count <= 3:
                    print(f"  ZERO: {key}")
            else:
                nonzero_count += 1
        print(f"  Total: {zero_count} zero, {nonzero_count} nonzero")

    # Load all rank files and compare
    print("\n6. Compare across ranks:")
    all_rank_keys = {}
    for rank in range(8):
        pattern = f"mp_rank_{rank:02d}_{rank:03d}_adapter.pt"
        rank_file = os.path.join(CHECKPOINT_PATH, pattern)
        if os.path.exists(rank_file):
            ckpt = torch.load(rank_file, map_location="cpu")
            adapter_state = ckpt.get("adapter_state_dict", {})
            all_rank_keys[rank] = set(adapter_state.keys())
            print(f"  Rank {rank}: {len(adapter_state)} keys")

    # Check if all ranks have same keys
    if len(all_rank_keys) > 1:
        first_keys = list(all_rank_keys.values())[0]
        all_same = all(keys == first_keys for keys in all_rank_keys.values())
        print(f"  All ranks have same keys: {all_same}")

    # Load safetensors file (vLLM format)
    print("\n7. Safetensors file (vLLM format):")
    st_file = os.path.join(CHECKPOINT_PATH, "adapter_model.safetensors")
    if os.path.exists(st_file):
        from safetensors import safe_open
        with safe_open(st_file, framework="pt", device="cpu") as f:
            st_keys = list(f.keys())
            print(f"  {len(st_keys)} keys")
            for key in st_keys[:5]:
                val = f.get_tensor(key)
                norm = val.float().norm().item()
                shape = list(val.shape)
                print(f"  {key}: shape={shape}, norm={norm:.6f}")

    # Compare Megatron vs safetensors weights
    print("\n8. Compare Megatron (rank 0) vs Safetensors weights:")
    if os.path.exists(rank0_file) and os.path.exists(st_file):
        megatron_ckpt = torch.load(rank0_file, map_location="cpu")
        megatron_state = megatron_ckpt.get("adapter_state_dict", {})

        from safetensors import safe_open
        with safe_open(st_file, framework="pt", device="cpu") as f:
            # Try to find matching keys
            # Megatron keys: decoder.layers.X.self_attention.linear_proj.adapter.linear_in.weight
            # Safetensors keys might be different

            # Get a sample Megatron key
            sample_meg_key = list(megatron_state.keys())[0]
            sample_st_key = list(f.keys())[0]
            print(f"  Sample Megatron key: {sample_meg_key}")
            print(f"  Sample Safetensors key: {sample_st_key}")

            # Print value comparison for a few keys
            for meg_key in list(megatron_state.keys())[:3]:
                meg_val = megatron_state[meg_key]
                print(f"\n  Megatron: {meg_key}")
                print(f"    shape={list(meg_val.shape)}, norm={meg_val.float().norm():.6f}")
                print(f"    first5={meg_val.flatten()[:5].tolist()}")


if __name__ == "__main__":
    main()
