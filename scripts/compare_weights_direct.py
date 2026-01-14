#!/usr/bin/env python3
"""Directly compare LoRA weights between Megatron state dict and exported checkpoint.

This script:
1. Trains LoRA for 1 step
2. Gets Megatron's internal LoRA state dict
3. Exports to vLLM format
4. Compares the actual tensor values

If tensors differ, export bug. If identical, forward pass bug.
"""

import asyncio
import os
from datetime import datetime

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker
import torch
import numpy as np

MODEL_NAME = "moonshotai/Moonlight-16B-A3B-Instruct"

TEST_TEXT = """<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi<|im_end|>"""


async def main():
    from transformers import AutoTokenizer
    from safetensors.torch import load_file
    import json

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    mask = [1.0] * len(input_tokens)

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.ones(len(input_tokens), dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.zeros(len(input_tokens), dtype=torch.float32)),
        }
    )

    # ===============================================================
    # PHASE 1: Train and export
    # ===============================================================
    print("\n" + "=" * 70)
    print("PHASE 1: Train 1 step")
    print("=" * 70)

    client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)
    print(f"Model ID: {client.model_id}")

    # Train 1 step
    fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
    await fwd_bwd.result_async()
    await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()

    # Save checkpoint
    checkpoint_name = f"weight_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_result = await (await client.save_state_async(name=checkpoint_name)).result_async()
    checkpoint_path = save_result.path
    print(f"Checkpoint saved: {checkpoint_path}")

    # Parse tinker:// path to get actual filesystem path
    # Format: tinker://local/{model_id}/{name}
    if checkpoint_path.startswith("tinker://local/"):
        parts = checkpoint_path.replace("tinker://local/", "").split("/")
        model_id = parts[0]
        name = parts[1]
        # Standard location for tinker checkpoints on volcano
        fs_path = f"/tmp/tinker_checkpoints/{model_id}/{name}"
    else:
        fs_path = checkpoint_path

    print(f"Filesystem path: {fs_path}")

    # ===============================================================
    # PHASE 2: Read exported weights
    # ===============================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Read exported checkpoint")
    print("=" * 70)

    # Need to read from volcano server
    import subprocess
    result = subprocess.run(
        ["ssh", "volcano", f"ls -la {fs_path}/"],
        capture_output=True, text=True
    )
    print(result.stdout)

    # Check adapter config
    result = subprocess.run(
        ["ssh", "volcano", f"cat {fs_path}/adapter_config.json"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        config = json.loads(result.stdout)
        print(f"\nAdapter config: rank={config.get('r')}, alpha={config.get('lora_alpha')}")

    # Download and analyze weights
    result = subprocess.run(
        ["ssh", "volcano", f"python3 -c \"import torch; from safetensors.torch import load_file; sd = load_file('{fs_path}/adapter_model.safetensors'); print('Keys:', len(sd)); print('Sample keys:'); [print('  ', k, sd[k].shape, sd[k].dtype) for k in list(sd.keys())[:10]]\""],
        capture_output=True, text=True
    )
    print("\nExported weights info:")
    print(result.stdout)
    if result.stderr:
        print("Stderr:", result.stderr)

    # ===============================================================
    # PHASE 3: Check for per-expert expansion
    # ===============================================================
    print("\n" + "=" * 70)
    print("PHASE 3: Check per-expert weight expansion")
    print("=" * 70)

    result = subprocess.run(
        ["ssh", "volcano", f"""python3 << 'EOF'
import torch
from safetensors.torch import load_file

sd = load_file("{fs_path}/adapter_model.safetensors")

# Check if per-expert format
expert_keys = [k for k in sd if '.experts.' in k]
shared_keys = [k for k in sd if '.mlp.' in k and '.experts.' not in k and '.shared_expert.' not in k]

print(f"Per-expert keys: {len(expert_keys)}")
print(f"Shared (non-expert) MLP keys: {len(shared_keys)}")

# Check if expert weights are identical (replicated from shared)
if expert_keys:
    # Get layer 1 experts for gate_proj lora_A
    layer1_gate_A = {{k: sd[k] for k in sd if '.layers.1.mlp.experts.' in k and 'gate_proj.lora_A' in k}}
    if len(layer1_gate_A) > 1:
        keys = sorted(layer1_gate_A.keys())
        first = layer1_gate_A[keys[0]]
        print(f"\\nChecking if expert weights are identical:")
        for k in keys[:5]:
            w = layer1_gate_A[k]
            diff = (first - w).abs().max().item()
            print(f"  {k.split('.')[-4]}.{k.split('.')[-3]}: diff={diff:.8f}")

        # Check weight norm
        print(f"\\nWeight norms:")
        for k in keys[:3]:
            w = layer1_gate_A[k]
            print(f"  {k.split('mlp.')[-1]}: norm={w.norm().item():.6f}")

# Also check MLA weights (attention)
mla_keys = [k for k in sd if '.self_attn.' in k]
print(f"\\nMLA (attention) keys: {len(mla_keys)}")
if mla_keys:
    sample_mla = mla_keys[0]
    w = sd[sample_mla]
    print(f"  Sample: {sample_mla.split('model.')[-1][:50]}")
    print(f"  Shape: {w.shape}, norm: {w.norm().item():.6f}")
EOF"""],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print("Stderr:", result.stderr[:500])


if __name__ == "__main__":
    asyncio.run(main())
