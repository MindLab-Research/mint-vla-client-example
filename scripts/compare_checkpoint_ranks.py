#!/usr/bin/env python3
"""Compare keys across checkpoint ranks."""

import os
import torch

CHECKPOINT_PATH = "/vePFS-Mindverse/share/code/tinker-server/checkpoints/80e6bf97-56d9-4f3a-9872-7cc4b80a7379_0/debug_checkpoint_20260110_182006"

# Load rank 0 and rank 1 checkpoints
rank0_file = os.path.join(CHECKPOINT_PATH, "mp_rank_00_000_adapter.pt")
rank1_file = os.path.join(CHECKPOINT_PATH, "mp_rank_01_001_adapter.pt")

ckpt0 = torch.load(rank0_file, map_location="cpu")
ckpt1 = torch.load(rank1_file, map_location="cpu")

state0 = ckpt0.get("adapter_state_dict", {})
state1 = ckpt1.get("adapter_state_dict", {})

keys0 = set(state0.keys())
keys1 = set(state1.keys())

print(f"Rank 0 checkpoint: {len(keys0)} keys")
print(f"Rank 1 checkpoint: {len(keys1)} keys")
print(f"Keys in both: {len(keys0 & keys1)}")
print(f"Only in rank 0: {len(keys0 - keys1)}")
print(f"Only in rank 1: {len(keys1 - keys0)}")

# Check if they're identical
if keys0 == keys1:
    print("\nKeys are IDENTICAL across ranks")
    # Compare values for a sample key
    sample_key = list(keys0)[0]
    val0 = state0[sample_key]
    val1 = state1[sample_key]
    print(f"\nSample key: {sample_key}")
    print(f"  Rank 0: shape={list(val0.shape)}, norm={val0.float().norm():.6f}")
    print(f"  Rank 1: shape={list(val1.shape)}, norm={val1.float().norm():.6f}")

    if val0.shape == val1.shape:
        match = torch.allclose(val0.float(), val1.float(), atol=1e-4)
        print(f"  Values match: {match}")
    else:
        print(f"  Different shapes!")
else:
    print("\nKeys DIFFER across ranks")
    only0 = keys0 - keys1
    only1 = keys1 - keys0
    print(f"\nFirst 5 only in rank 0:")
    for k in list(only0)[:5]:
        print(f"  {k}")
    print(f"\nFirst 5 only in rank 1:")
    for k in list(only1)[:5]:
        print(f"  {k}")

# Count expert vs non-expert keys
expert_keys = [k for k in keys0 if ".experts." in k]
shared_expert_keys = [k for k in keys0 if ".shared_experts." in k]
attention_keys = [k for k in keys0 if "self_attention" in k]

print(f"\n" + "="*70)
print("Key breakdown (rank 0):")
print(f"  Attention (self_attention): {len(attention_keys)}")
print(f"  Shared experts: {len(shared_expert_keys)}")
print(f"  Routed experts: {len(expert_keys)}")
print(f"  Other: {len(keys0) - len(attention_keys) - len(shared_expert_keys) - len(expert_keys)}")
