#!/usr/bin/env python3
"""Compare Megatron checkpoint weights vs safetensors weights.

Check if the same weights are in both files (just in different key formats).
"""

import os
import torch

CHECKPOINT_PATH = "/vePFS-Mindverse/share/code/tinker-server/checkpoints/80e6bf97-56d9-4f3a-9872-7cc4b80a7379_0/debug_checkpoint_20260110_182006"


def main():
    from safetensors import safe_open

    # Load Megatron rank 0 checkpoint
    meg_file = os.path.join(CHECKPOINT_PATH, "mp_rank_00_000_adapter.pt")
    meg_ckpt = torch.load(meg_file, map_location="cpu")
    meg_state = meg_ckpt.get("adapter_state_dict", {})

    # Load safetensors
    st_file = os.path.join(CHECKPOINT_PATH, "adapter_model.safetensors")
    st_state = {}
    with safe_open(st_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            st_state[key] = f.get_tensor(key)

    print("=" * 70)
    print("WEIGHT COMPARISON: Megatron vs Safetensors")
    print("=" * 70)
    print(f"Megatron (rank 0): {len(meg_state)} keys")
    print(f"Safetensors: {len(st_state)} keys")

    # Try to find matching weights by content
    print("\n" + "=" * 70)
    print("FINDING MATCHES BY VALUE")
    print("=" * 70)

    matched = 0
    meg_matched_keys = set()
    st_matched_keys = set()

    for meg_key, meg_val in meg_state.items():
        meg_flat = meg_val.flatten()
        for st_key, st_val in st_state.items():
            if st_key in st_matched_keys:
                continue
            st_flat = st_val.flatten()
            if meg_flat.shape == st_flat.shape:
                if torch.allclose(meg_flat.float(), st_flat.float(), atol=1e-4):
                    print(f"MATCH:")
                    print(f"  Megatron: {meg_key} shape={list(meg_val.shape)}")
                    print(f"  Safetens: {st_key} shape={list(st_val.shape)}")
                    matched += 1
                    meg_matched_keys.add(meg_key)
                    st_matched_keys.add(st_key)
                    break

        if matched >= 10:
            print("... (showing first 10 matches)")
            break

    print(f"\nTotal matches found: {matched}")

    # Check layer 0 specifically
    print("\n" + "=" * 70)
    print("LAYER 0 WEIGHTS COMPARISON")
    print("=" * 70)

    # Megatron layer 0 keys
    meg_layer0 = {k: v for k, v in meg_state.items() if "layers.0." in k}
    print(f"\nMegatron layer 0 keys: {len(meg_layer0)}")
    for key in list(meg_layer0.keys())[:5]:
        print(f"  {key}: shape={list(meg_layer0[key].shape)}")

    # Safetensors layer 0 keys
    st_layer0 = {k: v for k, v in st_state.items() if "layers.0." in k}
    print(f"\nSafetensors layer 0 keys: {len(st_layer0)}")
    for key in list(st_layer0.keys())[:5]:
        print(f"  {key}: shape={list(st_layer0[key].shape)}")

    # Compare self_attention.linear_proj (o_proj)
    print("\n" + "=" * 70)
    print("O_PROJ (LINEAR_PROJ) COMPARISON")
    print("=" * 70)

    # Megatron: decoder.layers.0.self_attention.linear_proj.adapter.linear_in.weight
    meg_oproj_in = meg_state.get("decoder.layers.0.self_attention.linear_proj.adapter.linear_in.weight")
    meg_oproj_out = meg_state.get("decoder.layers.0.self_attention.linear_proj.adapter.linear_out.weight")

    # Safetensors: model.layers.0.self_attn.o_proj.lora_A.weight / lora_B.weight
    st_oproj_a = st_state.get("model.layers.0.self_attn.o_proj.lora_A.weight")
    st_oproj_b = st_state.get("model.layers.0.self_attn.o_proj.lora_B.weight")

    if meg_oproj_in is not None:
        print(f"Megatron linear_proj.linear_in: shape={list(meg_oproj_in.shape)}")
        print(f"  first5={meg_oproj_in.flatten()[:5].tolist()}")
        print(f"  norm={meg_oproj_in.float().norm():.6f}")
    else:
        print("Megatron linear_proj.linear_in: NOT FOUND")

    if st_oproj_a is not None:
        print(f"Safetensors o_proj.lora_A: shape={list(st_oproj_a.shape)}")
        print(f"  first5={st_oproj_a.flatten()[:5].tolist()}")
        print(f"  norm={st_oproj_a.float().norm():.6f}")
    else:
        print("Safetensors o_proj.lora_A: NOT FOUND")

    if meg_oproj_out is not None:
        print(f"\nMegatron linear_proj.linear_out: shape={list(meg_oproj_out.shape)}")
        print(f"  first5={meg_oproj_out.flatten()[:5].tolist()}")
        print(f"  norm={meg_oproj_out.float().norm():.6f}")
    else:
        print("\nMegatron linear_proj.linear_out: NOT FOUND")

    if st_oproj_b is not None:
        print(f"Safetensors o_proj.lora_B: shape={list(st_oproj_b.shape)}")
        print(f"  first5={st_oproj_b.flatten()[:5].tolist()}")
        print(f"  norm={st_oproj_b.float().norm():.6f}")
    else:
        print("Safetensors o_proj.lora_B: NOT FOUND")

    # Check if they match
    if meg_oproj_in is not None and st_oproj_a is not None:
        # LoRA A is the "down" projection (rank reduction): [r, d] or [d, r]
        # LoRA B is the "up" projection: [d, r] or [r, d]
        # Megatron linear_in might be A, linear_out might be B
        match_a_in = torch.allclose(meg_oproj_in.float(), st_oproj_a.float(), atol=1e-4)
        match_a_in_t = torch.allclose(meg_oproj_in.float(), st_oproj_a.T.float(), atol=1e-4) if meg_oproj_in.shape == tuple(reversed(st_oproj_a.shape)) else False
        print(f"\nmeg_oproj_in == st_oproj_a: {match_a_in}")
        print(f"meg_oproj_in == st_oproj_a.T: {match_a_in_t}")


if __name__ == "__main__":
    main()
