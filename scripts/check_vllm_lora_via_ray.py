#!/usr/bin/env python3
"""Check LoRA weights loaded in vLLM via Ray.

Run on volcano: python3 /root/tinker_project/tinker-server/scripts/check_vllm_lora_via_ray.py
"""

import ray


@ray.remote(num_cpus=1)
def check_lora_weights(lora_path: str) -> dict:
    """Check LoRA weights on worker node."""
    from safetensors.torch import load_file
    import torch
    import os

    result = {"path": lora_path, "exists": False}

    if not os.path.exists(lora_path):
        return result

    result["exists"] = True
    safetensor_path = os.path.join(lora_path, "adapter_model.safetensors")

    if not os.path.exists(safetensor_path):
        result["error"] = "No adapter_model.safetensors"
        return result

    state_dict = load_file(safetensor_path)
    result["num_keys"] = len(state_dict)

    total_norm = 0.0
    nonzero_count = 0
    norms = {}

    for k, v in state_dict.items():
        norm = float(v.norm().item()) if isinstance(v, torch.Tensor) else 0.0
        total_norm += norm
        if norm > 1e-8:
            nonzero_count += 1
        norms[k] = norm

    result["nonzero_count"] = nonzero_count
    result["total_norm"] = total_norm

    # Get top 5 by norm
    sorted_norms = sorted(norms.items(), key=lambda x: x[1], reverse=True)
    result["top_norms"] = dict(sorted_norms[:5])
    result["bottom_norms"] = dict(sorted_norms[-5:])

    return result


def main():
    print("Connecting to Ray...")
    ray.init(address="auto", ignore_reinit_error=True)

    # Known LoRA path from logs
    lora_path = "/tmp/tinker_lora_1_xdmukq90"

    # Schedule on node with IP 192.168.32.215
    resources = {"node:192.168.32.215": 0.01}

    print(f"\nChecking LoRA at {lora_path} on node 192.168.32.215...")
    result = ray.get(check_lora_weights.options(resources=resources).remote(lora_path))

    print(f"\n{'=' * 60}")
    print("LoRA Weight Check Results")
    print("=" * 60)

    if not result["exists"]:
        print(f"ERROR: Path {lora_path} does not exist")
        return

    if "error" in result:
        print(f"ERROR: {result['error']}")
        return

    print(f"Path: {result['path']}")
    print(f"Number of keys: {result['num_keys']}")
    print(f"Non-zero tensors: {result['nonzero_count']}/{result['num_keys']}")
    print(f"Total L2 norm: {result['total_norm']:.6f}")

    print("\nTop 5 tensors by norm:")
    for k, v in result["top_norms"].items():
        print(f"  {k}: {v:.6f}")

    print("\nBottom 5 tensors by norm:")
    for k, v in result["bottom_norms"].items():
        print(f"  {k}: {v:.6f}")

    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    if result["total_norm"] < 0.01:
        print("WARNING: Total norm is very small - this looks like FRESH/UNTRAINED LoRA!")
    elif result["total_norm"] < 1.0:
        print("INFO: Total norm is small - LoRA may have minimal training effect")
    else:
        print("INFO: Non-trivial total norm - LoRA appears to have trained weights")


if __name__ == "__main__":
    main()
