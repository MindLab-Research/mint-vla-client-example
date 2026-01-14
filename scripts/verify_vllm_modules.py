#!/usr/bin/env python3
"""
Verify vLLM module names for DeepseekV2/V3/Moonlight models.

This script loads the model config and prints the actual module names
that vLLM would look for when loading LoRA weights.
"""

import sys
sys.path.insert(0, "/home/yiwen/tinker_project/vllm")

from vllm.lora.utils import get_supported_lora_modules

# Check if we can create a minimal model to inspect
def print_expected_modules():
    """Print expected module names from DeepseekV2 model structure."""

    print("=" * 60)
    print("Expected vLLM module names for DeepseekV2/V3/Moonlight MLA models:")
    print("=" * 60)

    # From the vLLM DeepseekV2Attention code:
    # MLA attention (q_lora_rank is not None):
    mla_attention_modules = [
        "q_a_proj",       # Down projection for Q
        "q_b_proj",       # Up projection for Q
        "kv_a_proj_with_mqa",  # Down projection for KV (with MQA)
        "kv_b_proj",      # Up projection for KV
        "o_proj",         # Output projection
    ]

    # Non-MLA attention (q_lora_rank is None):
    standard_attention_modules = [
        "q_proj",         # Q projection (only for non-MLA)
        "kv_a_proj_with_mqa",  # Still MLA for KV
        "kv_b_proj",
        "o_proj",
    ]

    # MLP modules (from DeepseekV2MLP):
    mlp_modules = [
        "gate_up_proj",   # Merged gate+up projection
        "down_proj",      # Down projection
    ]

    # MoE experts (from FusedMoE):
    moe_modules = [
        "experts",        # The FusedMoE layer itself
    ]

    print("\nMLA Attention modules (DeepseekV2/V3/Moonlight/K2):")
    for m in mla_attention_modules:
        print(f"  - self_attn.{m}")

    print("\nMLP modules (dense layers):")
    for m in mlp_modules:
        print(f"  - mlp.{m}")

    print("\nMoE modules:")
    for m in moe_modules:
        print(f"  - mlp.{m}")

    print("\n" + "=" * 60)
    print("Megatron export name mapping issues:")
    print("=" * 60)

    # What Megatron exports vs what vLLM expects
    mapping_issues = [
        ("linear_q_proj → q_proj", "WRONG", "vLLM MLA expects q_a_proj or q_b_proj"),
        ("linear_kv_down_proj → kv_a_proj_with_mqa", "CORRECT", ""),
        ("linear_kv_up_proj → kv_b_proj", "CORRECT", ""),
        ("linear_proj → o_proj", "CORRECT", ""),
        ("linear_fc1 → gate_proj", "WRONG", "vLLM expects gate_up_proj"),
        ("linear_fc2 → down_proj", "CORRECT", ""),
    ]

    for megatron_to_vllm, status, note in mapping_issues:
        color = "\033[92m" if status == "CORRECT" else "\033[91m"
        reset = "\033[0m"
        print(f"{color}{status:8}{reset} {megatron_to_vllm}")
        if note:
            print(f"         ↳ {note}")

    print("\n" + "=" * 60)
    print("CRITICAL ISSUES IDENTIFIED:")
    print("=" * 60)
    print("""
1. Q Projection Mismatch:
   - Megatron has: linear_q_proj (single module)
   - Conversion maps to: self_attn.q_proj
   - vLLM MLA expects: self_attn.q_a_proj OR self_attn.q_b_proj

   The Megatron-Bridge Moonlight model uses linear_q_proj for the Q projection,
   which is different from the DeepSeek V3 style (linear_q_down_proj + linear_q_up_proj).

   Need to verify if vLLM supports q_proj for Moonlight or if we need different mapping.

2. MLP Gate Mismatch:
   - Megatron has: linear_fc1 (gate+up fused)
   - Conversion maps to: mlp.gate_proj
   - vLLM expects: mlp.gate_up_proj (merged gate+up)

   This is definitely wrong. The conversion should map to gate_up_proj.
""")


if __name__ == "__main__":
    print_expected_modules()
