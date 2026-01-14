#!/usr/bin/env python3
"""Check checkpoint file mapping for each rank."""

import os

CHECKPOINT_PATH = "/vePFS-Mindverse/share/code/tinker-server/checkpoints/80e6bf97-56d9-4f3a-9872-7cc4b80a7379_0/debug_checkpoint_20260110_182006"

# List actual checkpoint files
print("Actual checkpoint files:")
files = [f for f in os.listdir(CHECKPOINT_PATH) if f.endswith('_adapter.pt')]
for f in sorted(files):
    print(f"  {f}")

print("\n" + "="*70)
print("Expected file for each rank (TP=8, EP=8):")

# Simulate what _get_rank_checkpoint_path would generate for each global rank
# With TP=8, EP=8, there are 64 global ranks
# Layout: global_rank = tp_rank * EP + ep_rank (or could be ep_rank * TP + tp_rank)
# Need to check which layout is used

for global_rank in range(64):
    # Assuming layout: global_rank = tp_rank * EP_size + ep_rank
    # i.e., tp_rank = global_rank % TP_size, ep_rank = global_rank // TP_size
    # Actually in Megatron, it's typically: tp_rank cycles fastest
    # global_rank = ep_rank * TP_size + tp_rank
    # tp_rank = global_rank % TP_size
    # ep_rank = global_rank // TP_size

    tp_rank = global_rank % 8  # TP_size = 8
    ep_rank = global_rank // 8  # EP_size = 8

    # Expected file from _get_rank_checkpoint_path
    expected_file = f"mp_rank_{tp_rank:02d}_{ep_rank:03d}_adapter.pt"
    expected_path = os.path.join(CHECKPOINT_PATH, expected_file)
    exists = os.path.exists(expected_path)

    if global_rank < 8 or (global_rank >= 56 and global_rank < 64):
        status = "EXISTS" if exists else "MISSING"
        print(f"  Rank {global_rank:2d} (TP={tp_rank}, EP={ep_rank}): {expected_file} - {status}")

print("\n" + "="*70)
print("What files would ranks 0-7 and 8-15 try to load?")
for global_rank in [0, 1, 2, 7, 8, 9, 16, 24, 32, 40, 48, 56]:
    tp_rank = global_rank % 8
    ep_rank = global_rank // 8
    expected_file = f"mp_rank_{tp_rank:02d}_{ep_rank:03d}_adapter.pt"
    expected_path = os.path.join(CHECKPOINT_PATH, expected_file)
    exists = os.path.exists(expected_path)
    print(f"  Rank {global_rank:2d} (TP={tp_rank}, EP={ep_rank}): {expected_file} - {'EXISTS' if exists else 'MISSING'}")
