#!/usr/bin/env python3
"""Check if expert LoRA weights differ across TP shards."""

import os
import torch

CHECKPOINT_PATH = "/vePFS-Mindverse/share/code/tinker-server/checkpoints/80e6bf97-56d9-4f3a-9872-7cc4b80a7379_0/debug_checkpoint_20260110_182006"

# Load all 8 checkpoint files
checkpoints = {}
for i in range(8):
    file = os.path.join(CHECKPOINT_PATH, f"mp_rank_{i:02d}_{i:03d}_adapter.pt")
    if os.path.exists(file):
        ckpt = torch.load(file, map_location="cpu")
        checkpoints[i] = ckpt.get("adapter_state_dict", {})
        print(f"Loaded rank {i}: {len(checkpoints[i])} keys")

# Find a routed expert key
sample_expert_key = None
sample_attn_key = None
for k in checkpoints[0].keys():
    if ".experts." in k and "shared" not in k and sample_expert_key is None:
        sample_expert_key = k
    if "self_attention" in k and sample_attn_key is None:
        sample_attn_key = k
    if sample_expert_key and sample_attn_key:
        break

print(f"\n" + "="*70)
print("ROUTED EXPERT LoRA across TP shards:")
print(f"Key: {sample_expert_key}")
for i in checkpoints:
    val = checkpoints[i].get(sample_expert_key)
    if val is not None:
        print(f"  Rank {i}: shape={list(val.shape)}, norm={val.float().norm():.6f}, first3={val.flatten()[:3].tolist()}")

print(f"\n" + "="*70)
print("ATTENTION LoRA across TP shards:")
print(f"Key: {sample_attn_key}")
for i in checkpoints:
    val = checkpoints[i].get(sample_attn_key)
    if val is not None:
        print(f"  Rank {i}: shape={list(val.shape)}, norm={val.float().norm():.6f}, first3={val.flatten()[:3].tolist()}")

# Check if routed expert weights are the same across ranks (would indicate no EP sharding)
print(f"\n" + "="*70)
print("Checking if values are identical across ranks...")
for key_type, key in [("Expert", sample_expert_key), ("Attention", sample_attn_key)]:
    val0 = checkpoints[0].get(key)
    all_same = True
    for i in range(1, 8):
        vali = checkpoints[i].get(key)
        if val0.shape != vali.shape:
            all_same = False
            print(f"  {key_type}: Shape differs between rank 0 and {i}")
            break
        if not torch.allclose(val0.float(), vali.float(), atol=1e-6):
            all_same = False
            print(f"  {key_type}: Values differ between rank 0 and {i} (max_diff={(val0 - vali).abs().max():.6f})")
            break
    if all_same:
        print(f"  {key_type}: IDENTICAL across all ranks")
