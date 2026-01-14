#!/usr/bin/env python3
"""Verify that LoRA weights are actually applied during forward pass.

Check:
1. What are the fresh model's LoRA weights (should be zero or small init)
2. After loading checkpoint, what are the LoRA weights
3. During forward, are the LoRA adapters being called
"""

import os
import sys
import asyncio

# Add project root to path
sys.path.insert(0, "/home/yiwen/tinker_project/tinker-server")

import torch
import httpx

CHECKPOINT_PATH = "/vePFS-Mindverse/share/code/tinker-server/checkpoints/80e6bf97-56d9-4f3a-9872-7cc4b80a7379_0/debug_checkpoint_20260110_182006"

BASE_URL = "http://localhost:8000"
BASE_MODEL = "moonshotai/Moonlight-16B-A3B-Instruct"


async def get_fresh_model_lora_weights():
    """Get LoRA weight stats from fresh model via API."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        # Create fresh training session
        resp = await client.post(f"{BASE_URL}/api/v1/create_session", json={
            "model_name": BASE_MODEL,
            "lora_rank": 16,
            "max_seq_len": 8192,
        })
        resp.raise_for_status()
        data = resp.json()
        model_id = data["model_id"]
        print(f"Created fresh model: {model_id}")

        # Get weight stats via new endpoint
        resp = await client.post(f"{BASE_URL}/api/v1/debug/lora_weight_stats", json={
            "model_id": model_id,
        })
        if resp.status_code == 404:
            print("Debug endpoint not available, need to add it")
            return None
        resp.raise_for_status()
        return model_id, resp.json()


async def main():
    """Main diagnostic."""
    print("=" * 70)
    print("VERIFY LORA FORWARD USAGE")
    print("=" * 70)

    # First, let's directly examine the checkpoint
    from safetensors import safe_open

    st_file = os.path.join(CHECKPOINT_PATH, "adapter_model.safetensors")
    with safe_open(st_file, framework="pt", device="cpu") as f:
        st_keys = list(f.keys())
        print(f"\nCheckpoint has {len(st_keys)} LoRA parameters")

        # Get layer 0 o_proj LoRA A and B
        lora_a_key = "model.layers.0.self_attn.o_proj.lora_A.weight"
        lora_b_key = "model.layers.0.self_attn.o_proj.lora_B.weight"

        lora_a = f.get_tensor(lora_a_key)
        lora_b = f.get_tensor(lora_b_key)

        print(f"\nLayer 0 o_proj LoRA weights from checkpoint:")
        print(f"  lora_A: shape={list(lora_a.shape)}, norm={lora_a.float().norm():.6f}")
        print(f"    first5={lora_a.flatten()[:5].tolist()}")
        print(f"  lora_B: shape={list(lora_b.shape)}, norm={lora_b.float().norm():.6f}")
        print(f"    first5={lora_b.flatten()[:5].tolist()}")

        # LoRA output = (input @ A.T) @ B.T * (alpha / rank)
        # Or equivalently: input @ (A.T @ B.T) * (alpha / rank) = input @ (B @ A).T * scaling
        # The combined weight delta is: (B @ A) * scaling
        # Typically alpha = rank, so scaling = 1.0

        # Check the combined delta magnitude
        # lora_A: [rank, in_features] = [16, 2048]
        # lora_B: [out_features, rank] = [2048, 16]
        # delta = lora_B @ lora_A = [2048, 2048]

        print(f"\n  Computing LoRA delta (B @ A):")
        if lora_a.shape[0] == 16 and lora_b.shape[1] == 16:  # rank=16
            delta = lora_b.float() @ lora_a.float()
            print(f"    delta shape: {list(delta.shape)}")
            print(f"    delta norm: {delta.norm():.6f}")
            print(f"    delta mean: {delta.mean():.6f}")
            print(f"    delta max: {delta.max():.6f}")
            print(f"    delta min: {delta.min():.6f}")
        else:
            print(f"    Unexpected shapes: A={lora_a.shape}, B={lora_b.shape}")

    # Now check Megatron checkpoint
    meg_file = os.path.join(CHECKPOINT_PATH, "mp_rank_00_000_adapter.pt")
    meg_ckpt = torch.load(meg_file, map_location="cpu")
    meg_state = meg_ckpt.get("adapter_state_dict", {})

    # Find corresponding key
    meg_a_key = "decoder.layers.0.self_attention.linear_proj.adapter.linear_in.weight"
    meg_b_key = "decoder.layers.0.self_attention.linear_proj.adapter.linear_out.weight"

    print(f"\nMegatron rank 0 layer 0 o_proj LoRA (sharded):")
    meg_a = meg_state.get(meg_a_key)
    meg_b = meg_state.get(meg_b_key)

    if meg_a is not None:
        print(f"  linear_in (A): shape={list(meg_a.shape)}, norm={meg_a.float().norm():.6f}")
        print(f"    first5={meg_a.flatten()[:5].tolist()}")
    else:
        print(f"  linear_in (A): NOT FOUND")

    if meg_b is not None:
        print(f"  linear_out (B): shape={list(meg_b.shape)}, norm={meg_b.float().norm():.6f}")
        print(f"    first5={meg_b.flatten()[:5].tolist()}")
    else:
        print(f"  linear_out (B): NOT FOUND")

    # Compare the first shard of safetensors with Megatron rank 0
    # Safetensors lora_A: [16, 2048] -> should shard to [16, 256] per rank
    print(f"\nComparing first shard:")
    if meg_a is not None:
        shard_size = lora_a.shape[1] // 8  # TP=8
        st_shard = lora_a[:, :shard_size]
        print(f"  Safetensors shard [0:256]: shape={list(st_shard.shape)}, norm={st_shard.float().norm():.6f}")
        print(f"    first5={st_shard.flatten()[:5].tolist()}")
        print(f"  Megatron rank 0: shape={list(meg_a.shape)}, norm={meg_a.float().norm():.6f}")
        print(f"    first5={meg_a.flatten()[:5].tolist()}")

        if st_shard.shape == meg_a.shape:
            match = torch.allclose(st_shard.float(), meg_a.float(), atol=1e-4)
            diff = (st_shard.float() - meg_a.float()).abs().max()
            print(f"  Match: {match}, max_diff: {diff:.6f}")
        else:
            print(f"  Shape mismatch!")

    # Check lora_B sharding
    # Safetensors lora_B: [2048, 16] -> should shard to [256, 16] per rank
    print(f"\nComparing lora_B shard:")
    if meg_b is not None:
        shard_size = lora_b.shape[0] // 8  # TP=8
        st_shard_b = lora_b[:shard_size, :]
        print(f"  Safetensors shard [0:256]: shape={list(st_shard_b.shape)}, norm={st_shard_b.float().norm():.6f}")
        print(f"    first5={st_shard_b.flatten()[:5].tolist()}")
        print(f"  Megatron rank 0: shape={list(meg_b.shape)}, norm={meg_b.float().norm():.6f}")
        print(f"    first5={meg_b.flatten()[:5].tolist()}")

        if st_shard_b.shape == meg_b.shape:
            match = torch.allclose(st_shard_b.float(), meg_b.float(), atol=1e-4)
            diff = (st_shard_b.float() - meg_b.float()).abs().max()
            print(f"  Match: {match}, max_diff: {diff:.6f}")
        else:
            print(f"  Shape mismatch!")


if __name__ == "__main__":
    asyncio.run(main())
