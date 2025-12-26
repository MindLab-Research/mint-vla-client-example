"""
Test official Tinker to determine MoE LoRA adapter structure (shared vs per-expert).

This script:
1. Creates a LoRA training client with an MoE model
2. Does minimal training (1 step)
3. Downloads the weights
4. Inspects the adapter structure
"""

import asyncio
import os
import tarfile
import tempfile
import urllib.request
from pathlib import Path

# Use official Tinker (no TINKER_BASE_URL override)
os.environ.pop("TINKER_BASE_URL", None)

import tinker
from tinker import types


async def main():
    # Use a small MoE model
    model_name = "Qwen/Qwen3-30B-A3B"  # MoE model

    print(f"Creating training client for MoE model: {model_name}")
    service_client = tinker.ServiceClient()

    # Create LoRA training client with default settings
    training_client = await service_client.create_lora_training_client_async(
        base_model=model_name,
        rank=32,
    )

    print("Training client created. Getting tokenizer...")
    tokenizer = training_client.get_tokenizer()

    # Create minimal training data
    prompt = "Hello world"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(" test", add_special_tokens=False)

    tokens = prompt_tokens + completion_tokens
    weights = [0] * len(prompt_tokens) + [1] * len(completion_tokens)

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]

    datum = types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
    )

    print("Running 1 training step...")
    fwd_bwd_future = await training_client.forward_backward_async([datum], "cross_entropy")
    optim_future = await training_client.optim_step_async(types.AdamParams(learning_rate=1e-4))

    await fwd_bwd_future
    await optim_future
    print("Training step complete.")

    # Save weights for download
    print("Saving weights for sampler...")
    save_result = await training_client.save_weights_for_sampler_async(name="moe_test")
    checkpoint_path = save_result.path
    print(f"Checkpoint path: {checkpoint_path}")

    # Download the checkpoint
    print("Getting download URL...")
    rc = service_client.create_rest_client()
    url_response = rc.get_checkpoint_archive_url_from_tinker_path(checkpoint_path).result()

    # Download and extract
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / "archive.tar"
        print(f"Downloading to {archive_path}...")
        urllib.request.urlretrieve(url_response.url, archive_path)

        # Extract
        extract_dir = Path(tmpdir) / "extracted"
        extract_dir.mkdir()
        print(f"Extracting to {extract_dir}...")
        with tarfile.open(archive_path) as tar:
            tar.extractall(extract_dir)

        # List all files
        print("\n=== Files in checkpoint ===")
        for f in sorted(extract_dir.rglob("*")):
            if f.is_file():
                print(f"  {f.relative_to(extract_dir)}")

        # Load and inspect adapter config and weights
        adapter_config_path = extract_dir / "adapter_config.json"
        if adapter_config_path.exists():
            print("\n=== adapter_config.json ===")
            print(adapter_config_path.read_text())

        # Look for safetensors files
        safetensor_files = list(extract_dir.glob("*.safetensors"))
        if safetensor_files:
            print("\n=== Adapter weight names ===")
            try:
                from safetensors import safe_open
                for sf_path in safetensor_files:
                    print(f"\nFile: {sf_path.name}")
                    with safe_open(sf_path, framework="pt") as f:
                        keys = list(f.keys())
                        # Focus on MoE-related keys
                        expert_keys = [k for k in keys if "expert" in k.lower() or "moe" in k.lower()]
                        if expert_keys:
                            print("Expert-related keys:")
                            for k in sorted(expert_keys)[:20]:  # First 20
                                tensor = f.get_tensor(k)
                                print(f"  {k}: {tensor.shape}")
                            if len(expert_keys) > 20:
                                print(f"  ... and {len(expert_keys) - 20} more expert keys")
                        else:
                            print("No expert-related keys found. All keys:")
                            for k in sorted(keys)[:30]:
                                tensor = f.get_tensor(k)
                                print(f"  {k}: {tensor.shape}")
                            if len(keys) > 30:
                                print(f"  ... and {len(keys) - 30} more keys")
            except ImportError:
                print("safetensors not installed, cannot inspect weights")

        # Also copy to a permanent location for further inspection
        output_dir = Path("/tmp/tinker_moe_lora_checkpoint")
        output_dir.mkdir(exist_ok=True)
        import shutil
        for f in extract_dir.iterdir():
            if f.is_file():
                shutil.copy(f, output_dir / f.name)
        print(f"\nCheckpoint copied to: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
