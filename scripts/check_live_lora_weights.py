#!/usr/bin/env python3
"""Check if Megatron actor's live LoRA weights match the exported PEFT.

Run on volcano: python3 /root/tinker_project/tinker-server/scripts/check_live_lora_weights.py
"""

import ray
import torch


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

    # Call a method to dump live LoRA weights from the model's named_parameters
    # This requires adding a method to the worker
    print("\nGetting live LoRA weights from model.named_parameters()...")

    live_weights = ray.get(megatron.get_live_lora_weights.remote(), timeout=60)

    if not live_weights:
        print("No live weights returned (method may not exist)")
        return

    print(f"\nLive LoRA weights ({len(live_weights)} tensors):")
    for name, (shape, norm, first5) in live_weights.items():
        if 'layers.0.' in name and ('q_proj' in name or 'gate_proj' in name):
            print(f"\n{name}:")
            print(f"  shape={shape}, norm={norm:.6f}")
            print(f"  first 5: {first5}")


if __name__ == "__main__":
    main()
