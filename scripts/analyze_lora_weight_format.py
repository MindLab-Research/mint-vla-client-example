#!/usr/bin/env python3
"""Check weight format: Compare exported weights with what Megatron/vLLM expect.

The issue: Same weights produce -33.9 nats in Megatron but -8.8 nats in vLLM at position 7.

This script checks:
1. Weight shapes in exported checkpoint
2. How Megatron loads and applies them (expected format)
3. How vLLM loads and applies them (expected format)
4. Look for transpose or scaling mismatches

"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from safetensors.torch import load_file

# Checkpoint path from previous investigation
CHECKPOINT_PATH = "/root/tinker_project/tinker-server/checkpoints/137ddf58-55bf-4c16-9d49-52f94b89276c_0/debug_checkpoint_20260111_083801"


def analyze_checkpoint():
    """Load and analyze the exported checkpoint."""

    adapter_path = Path(CHECKPOINT_PATH) / "adapter_model.safetensors"
    config_path = Path(CHECKPOINT_PATH) / "adapter_config.json"

    if not adapter_path.exists():
        print(f"Checkpoint not found: {adapter_path}")
        return

    # Load weights
    print("=" * 70)
    print("CHECKPOINT ANALYSIS")
    print("=" * 70)

    state_dict = load_file(str(adapter_path))
    with open(config_path) as f:
        config = json.load(f)

    print(f"\nConfig: rank={config.get('r')}, alpha={config.get('lora_alpha')}")
    print(f"Scale = alpha/r = {config.get('lora_alpha', 16)}/{config.get('r', 16)} = {config.get('lora_alpha', 16)/config.get('r', 16)}")

    print(f"\nTotal keys: {len(state_dict)}")

    # Categorize keys
    mla_keys = [k for k in state_dict if '.self_attn.' in k]
    mlp_keys = [k for k in state_dict if '.mlp.' in k]

    print(f"MLA (attention) keys: {len(mla_keys)}")
    print(f"MLP (MoE) keys: {len(mlp_keys)}")

    # Analyze MLP keys - this is where MoE LoRA lives
    print("\n" + "=" * 70)
    print("MLP/MoE LoRA WEIGHTS")
    print("=" * 70)

    # Get layer 1 MLP keys (first MoE layer)
    layer1_mlp = [k for k in mlp_keys if '.layers.1.' in k]
    print(f"\nLayer 1 MLP keys: {len(layer1_mlp)}")

    # Check for per-expert vs shared format
    expert_keys = [k for k in layer1_mlp if '.experts.' in k]
    shared_keys = [k for k in layer1_mlp if '.experts.' not in k and '.shared_expert.' not in k]

    print(f"  Per-expert keys: {len(expert_keys)}")
    print(f"  Shared/non-expert keys: {len(shared_keys)}")

    if expert_keys:
        # Per-expert format - check expert 0 vs expert 1
        expert_0_keys = [k for k in expert_keys if '.experts.0.' in k]
        expert_1_keys = [k for k in expert_keys if '.experts.1.' in k]

        print(f"\n  Expert 0 keys: {len(expert_0_keys)}")
        print(f"  Expert 1 keys: {len(expert_1_keys)}")

        # Check if expert 0 and 1 weights are identical (replicated)
        if expert_0_keys and expert_1_keys:
            print("\n  Checking if expert 0 and 1 weights are identical (replicated from shared):")
            for k0 in expert_0_keys[:3]:
                k1 = k0.replace('.experts.0.', '.experts.1.')
                if k1 in state_dict:
                    w0 = state_dict[k0]
                    w1 = state_dict[k1]
                    diff = (w0 - w1).abs().max().item()
                    print(f"    {k0.split('.')[-2:]}: diff={diff:.8f}, identical={diff < 1e-6}")

    # Show sample weight shapes
    print("\n" + "=" * 70)
    print("SAMPLE WEIGHT SHAPES")
    print("=" * 70)

    # lora_A and lora_B for different layer types
    sample_keys = {}
    for k in state_dict:
        if 'lora_A' in k or 'lora_B' in k:
            layer_type = 'mlp' if '.mlp.' in k else 'attn'
            ab = 'lora_A' if 'lora_A' in k else 'lora_B'
            key = f"{layer_type}_{ab}"
            if key not in sample_keys:
                sample_keys[key] = k

    print("\nSample shapes for each layer type:")
    for key_type, key in sample_keys.items():
        w = state_dict[key]
        print(f"  {key_type}: {key.split('model.layers.')[-1][:60]}")
        print(f"    shape={list(w.shape)}, dtype={w.dtype}")
        print(f"    norm={w.norm().item():.6f}, max={w.abs().max().item():.6f}")

    # Check gate_proj and up_proj specifically (fused linear_fc1)
    print("\n" + "=" * 70)
    print("GATE_PROJ AND UP_PROJ (FUSED FC1)")
    print("=" * 70)

    gate_keys = [k for k in state_dict if 'gate_proj' in k]
    up_keys = [k for k in state_dict if 'up_proj' in k]

    print(f"\ngate_proj keys: {len(gate_keys)}")
    print(f"up_proj keys: {len(up_keys)}")

    # Check layer 1 expert 0 gate and up proj
    for proj in ['gate_proj', 'up_proj']:
        keys = [k for k in state_dict if f'.layers.1.mlp.experts.0.{proj}.' in k]
        for k in keys[:2]:
            w = state_dict[k]
            print(f"\n{k.split('model.')[-1]}:")
            print(f"  shape={list(w.shape)}")
            print(f"  norm={w.norm().item():.6f}")

    # Compute statistics on how weights changed from initialization
    print("\n" + "=" * 70)
    print("WEIGHT CHANGE FROM INITIALIZATION")
    print("=" * 70)

    print("\nLoRA_B should be initialized to zero in fresh LoRA.")
    print("Non-zero LoRA_B indicates training has modified weights.")

    lora_b_nonzero = []
    lora_b_zero = []

    for k in state_dict:
        if 'lora_B' in k:
            w = state_dict[k]
            norm = w.norm().item()
            if norm > 1e-6:
                lora_b_nonzero.append((k, norm))
            else:
                lora_b_zero.append(k)

    print(f"\nLoRA_B tensors with non-zero norm: {len(lora_b_nonzero)}")
    print(f"LoRA_B tensors with zero norm: {len(lora_b_zero)}")

    if lora_b_nonzero:
        print("\nSample non-zero LoRA_B:")
        for k, norm in lora_b_nonzero[:5]:
            w = state_dict[k]
            print(f"  {k.split('model.')[-1][:60]}")
            print(f"    norm={norm:.6f}, shape={list(w.shape)}")

    return state_dict, config


def check_megatron_format():
    """Check how Megatron expects weights."""
    print("\n" + "=" * 70)
    print("MEGATRON WEIGHT FORMAT")
    print("=" * 70)

    print("""
Megatron uses:
- ParallelLinearAdapter for LoRA
- linear_in: ColumnParallelLinear (input -> rank)
- linear_out: RowParallelLinear (rank -> output)

For MoE layers:
- TEGroupedMLP wrapped with LoRALinear
- LoRA applied AFTER expert computation (shared across all experts)

Weight shapes expected:
- lora_A: [hidden_dim/TP, rank]  (ColumnParallel, sharded on output)
- lora_B: [rank, output_dim/TP]  (RowParallel, sharded on input)

Scaling: output = base + lora_B @ activation @ lora_A * (alpha/r)
    """)


def check_vllm_format():
    """Check how vLLM expects weights."""
    print("\n" + "=" * 70)
    print("VLLM WEIGHT FORMAT")
    print("=" * 70)

    print("""
vLLM uses:
- FusedMoEWithLoRA for MoE layers
- Per-expert LoRA weights indexed by expert_id

Weight shapes:
- w13_lora_a_stacked: [max_loras, num_experts, max_rank, hidden_dim]
- w13_lora_b_stacked: [max_loras, num_experts, intermediate*2, max_rank]

The kernel applies LoRA per-expert:
  lora_out = (x @ lora_A[expert_id].T) @ lora_B[expert_id].T * scale

Key difference: lora_A and lora_B may be TRANSPOSED compared to Megatron!
    """)


if __name__ == "__main__":
    # Run on volcano server
    state_dict, config = analyze_checkpoint()
    check_megatron_format()
    check_vllm_format()

    print("\n" + "=" * 70)
    print("KEY QUESTION")
    print("=" * 70)
    print("""
If Megatron expects: output = lora_B @ lora_A @ x
And vLLM expects:    output = x @ lora_A.T @ lora_B.T

Then with same exported weights, one system produces WRONG results!

NEXT STEP: Check the actual matrix multiplication order in both systems.
    """)
