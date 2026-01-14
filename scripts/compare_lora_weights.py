#!/usr/bin/env python3
"""Compare LoRA weights in Megatron actor vs PEFT export.

This identifies if there's a mismatch between the live weights and exported weights.

Run on volcano: python3 /root/tinker_project/tinker-server/scripts/compare_lora_weights.py
"""

import ray
import torch
import numpy as np

def main():
    print("Connecting to Ray...")
    ray.init(address="auto", ignore_reinit_error=True)

    # Find Megatron actor
    actors = ray.util.list_named_actors(all_namespaces=True)
    megatron_actors = [a for a in actors if 'megatron' in a['name'].lower()]

    if not megatron_actors:
        print("ERROR: No Megatron actor found")
        return

    print(f"Found Megatron actor: {megatron_actors[0]['name']}")
    megatron = ray.get_actor(megatron_actors[0]['name'], namespace=megatron_actors[0].get('namespace', 'tinker'))

    # Get PEFT state dict (this is what gets exported to vLLM)
    print("\nGetting PEFT state dict (exported weights)...")
    peft_state = ray.get(megatron.get_lora_state_dict.remote(use_per_expert_lora=False), timeout=120)
    print(f"PEFT state dict has {len(peft_state)} keys")

    # Print sample weights for layer 0 attention
    print("\n=== Sample PEFT weights (layer 0 q_proj) ===")
    for key in sorted(peft_state.keys()):
        if 'layers.0.' in key and 'q_proj' in key:
            tensor = peft_state[key]
            print(f"{key}:")
            print(f"  shape={tensor.shape}, dtype={tensor.dtype}")
            print(f"  norm={tensor.float().norm().item():.6f}")
            print(f"  mean={tensor.float().mean().item():.6f}, std={tensor.float().std().item():.6f}")
            print(f"  min={tensor.float().min().item():.6f}, max={tensor.float().max().item():.6f}")
            print(f"  first 5 values: {tensor.flatten()[:5].tolist()}")

    # Now let's look at MLP expert weights which are more likely to cause issues with MoE
    print("\n=== Sample PEFT weights (layer 1 expert 0 gate_proj) ===")
    for key in sorted(peft_state.keys()):
        if 'layers.1.' in key and 'experts.0.' in key and 'gate_proj' in key:
            tensor = peft_state[key]
            print(f"{key}:")
            print(f"  shape={tensor.shape}, dtype={tensor.dtype}")
            print(f"  norm={tensor.float().norm().item():.6f}")
            print(f"  first 5 values: {tensor.flatten()[:5].tolist()}")

    # Check if lora_A and lora_B have correct relationship
    # For a properly trained LoRA, the product A @ B should have meaningful structure
    print("\n=== Checking LoRA structure ===")
    for layer_key in ['layers.0.self_attn.q_proj']:
        a_key = f"model.{layer_key}.lora_A.weight"
        b_key = f"model.{layer_key}.lora_B.weight"
        if a_key in peft_state and b_key in peft_state:
            A = peft_state[a_key].float()  # [rank, in_features]
            B = peft_state[b_key].float()  # [out_features, rank]

            # LoRA output = x @ A.T @ B.T = x @ (B @ A).T
            # The effective weight delta is B @ A
            delta = B @ A  # [out_features, in_features]

            print(f"\n{layer_key}:")
            print(f"  A shape: {A.shape}, B shape: {B.shape}")
            print(f"  delta (B @ A) shape: {delta.shape}")
            print(f"  delta norm: {delta.norm().item():.6f}")
            print(f"  delta mean: {delta.mean().item():.6f}")
            print(f"  delta max abs: {delta.abs().max().item():.6f}")


if __name__ == "__main__":
    main()
