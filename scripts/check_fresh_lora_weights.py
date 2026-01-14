#!/usr/bin/env python3
"""Check if fresh LoRA actually has zero weights."""

import asyncio
import os

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker
import torch


async def main():
    print("=" * 80)
    print("FRESH LORA WEIGHT CHECK")
    print("=" * 80)

    model_name = "moonshotai/Moonlight-16B-A3B-Instruct"
    base_url = os.environ["TINKER_BASE_URL"]

    # Create training session (fresh LoRA)
    print("\n[1] Creating training session with fresh LoRA...")
    service_client = tinker.ServiceClient(base_url=base_url)
    training_client = await service_client.create_lora_training_client_async(model_name, rank=16)
    print(f"    model_id: {training_client.model_id}")

    # Export LoRA weights
    print("\n[2] Exporting LoRA weights...")
    import tempfile
    import httpx

    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            f"{base_url}/api/v1/save_lora_weights",
            json={"model_id": training_client.model_id}
        )
        result = response.json()

        if "future_id" in result:
            # Poll for completion
            future_id = result["future_id"]
            for _ in range(60):
                poll = await client.post(
                    f"{base_url}/api/v1/retrieve_future",
                    json={"future_id": future_id}
                )
                if poll.status_code == 200:
                    result = poll.json()
                    break
                await asyncio.sleep(1.0)

    print(f"    Export result: {result}")

    # Check the exported weights
    print("\n[3] Checking exported LoRA weight norms...")

    # Find the temp directory
    import glob
    temp_dirs = glob.glob("/tmp/tinker_lora_*")
    if not temp_dirs:
        print("    ERROR: No exported LoRA found!")
        return

    temp_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    adapter_path = os.path.join(temp_dirs[0], "adapter_model.safetensors")

    if not os.path.exists(adapter_path):
        # Try to get from the result
        if "path" in result:
            adapter_path = os.path.join(result["path"], "adapter_model.safetensors")

    print(f"    Checking: {adapter_path}")

    from safetensors import safe_open

    with safe_open(adapter_path, framework="pt") as f:
        keys = list(f.keys())
        print(f"    Total keys: {len(keys)}")

        # Check lora_A and lora_B norms
        lora_a_norms = []
        lora_b_norms = []

        for key in keys[:50]:  # Check first 50
            tensor = f.get_tensor(key)
            norm = tensor.norm().item()

            if "lora_A" in key or "lora_a" in key.lower() or ".linear_in." in key:
                lora_a_norms.append((key, norm, tuple(tensor.shape)))
            elif "lora_B" in key or "lora_b" in key.lower() or ".linear_out." in key:
                lora_b_norms.append((key, norm, tuple(tensor.shape)))

        print("\n    lora_A weights (should have non-zero norms for Kaiming init):")
        for key, norm, shape in lora_a_norms[:5]:
            print(f"      {key[-60:]}: norm={norm:.6f}, shape={shape}")

        print("\n    lora_B weights (should have ZERO norms for standard init):")
        for key, norm, shape in lora_b_norms[:5]:
            status = "OK (zero)" if norm < 1e-6 else "NON-ZERO!"
            print(f"      {key[-60:]}: norm={norm:.6f}, shape={shape} [{status}]")

        # Summary
        nonzero_b = sum(1 for _, n, _ in lora_b_norms if n > 1e-6)
        print(f"\n    Summary: {nonzero_b}/{len(lora_b_norms)} lora_B weights are non-zero")

        if nonzero_b > 0:
            print("\n    >>> WARNING: Fresh LoRA has non-zero lora_B weights!")
            print("    >>> This means fresh LoRA contributes to the output, explaining divergence.")
        else:
            print("\n    >>> Fresh LoRA has zero lora_B weights as expected.")
            print("    >>> Divergence must be from base model differences or computation.")


if __name__ == "__main__":
    asyncio.run(main())
