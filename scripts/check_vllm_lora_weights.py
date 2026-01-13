#!/usr/bin/env python3
"""Check if vLLM has trained LoRA weights or fresh ones.

Run on volcano: python3 /root/tinker_project/tinker-server/scripts/check_vllm_lora_weights.py
"""

import ray
import asyncio
import os

async def main():
    print("Connecting to Ray...")
    ray.init(address="auto", ignore_reinit_error=True)

    # Find vLLM actor
    actors = ray.util.list_named_actors(all_namespaces=True)
    vllm_actors = [a for a in actors if 'vllm' in a['name'].lower()]

    if not vllm_actors:
        print("ERROR: No vLLM actor found")
        return

    vllm_actor = ray.get_actor(vllm_actors[0]['name'], namespace=vllm_actors[0].get('namespace', 'tinker'))

    # Get loaded LoRAs
    loaded_loras = ray.get(vllm_actor.list_loras.remote(), timeout=30)
    print(f"Loaded LoRAs: {loaded_loras}")

    if not loaded_loras:
        print("No LoRA loaded")
        return

    # Check _lora_paths attribute
    # We can't directly access _lora_paths, but the add_lora_with_id debug prints show the path
    # Let's check the temp directory for LoRA files

    print("\n" + "=" * 60)
    print("Checking LoRA temp directories...")
    print("=" * 60)

    # List tinker_lora temp directories
    import subprocess
    result = subprocess.run(
        ["ls", "-la", "/tmp"],
        capture_output=True,
        text=True
    )

    lora_dirs = [line for line in result.stdout.split('\n') if 'tinker_lora' in line]
    print(f"\nFound {len(lora_dirs)} tinker_lora directories:")
    for d in lora_dirs:
        print(d)

    # Find most recent
    lora_paths = []
    for f in os.listdir('/tmp'):
        if f.startswith('tinker_lora'):
            full_path = os.path.join('/tmp', f)
            if os.path.isdir(full_path):
                lora_paths.append(full_path)

    if not lora_paths:
        print("\nNo tinker_lora directories found!")
        return

    # Sort by modification time
    lora_paths.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    print("\n" + "=" * 60)
    print("Checking most recent LoRA directory...")
    print("=" * 60)

    from safetensors.torch import load_file
    import torch

    for lora_path in lora_paths[:3]:  # Check top 3 most recent
        print(f"\n{lora_path}:")
        safetensor_path = os.path.join(lora_path, "adapter_model.safetensors")

        if not os.path.exists(safetensor_path):
            print("  No adapter_model.safetensors found")
            continue

        state_dict = load_file(safetensor_path)
        print(f"  Keys: {len(state_dict)}")

        total_norm = 0.0
        nonzero_count = 0
        max_norm = 0.0
        max_key = None

        for k, v in state_dict.items():
            norm = float(v.norm().item()) if isinstance(v, torch.Tensor) else 0.0
            total_norm += norm
            if norm > 1e-8:
                nonzero_count += 1
            if norm > max_norm:
                max_norm = norm
                max_key = k

        print(f"  Non-zero tensors: {nonzero_count}/{len(state_dict)}")
        print(f"  Total L2 norm: {total_norm:.6f}")
        print(f"  Max norm: {max_norm:.6f} ({max_key})")

        # Check a few specific keys
        print("\n  Sample tensor norms:")
        for i, (k, v) in enumerate(list(state_dict.items())[:5]):
            norm = float(v.norm().item()) if isinstance(v, torch.Tensor) else 0.0
            print(f"    {k}: {norm:.6f}")


if __name__ == "__main__":
    asyncio.run(main())
