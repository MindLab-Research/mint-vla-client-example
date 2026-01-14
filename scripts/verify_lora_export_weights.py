#!/usr/bin/env python3
"""Verify LoRA weight export from Megatron to vLLM.

This script:
1. Trains LoRA for a few steps
2. Exports the checkpoint
3. Loads the checkpoint and inspects the per-expert weights
4. Verifies if all experts have identical LoRA weights (as expected from replication)
"""

import asyncio
import os
import sys

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker
import torch
import numpy as np
from pathlib import Path

MODEL_NAME = "moonshotai/Moonlight-16B-A3B-Instruct"


async def main():
    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    test_text = "<|im_start|>user\nHello<|im_end|><|im_start|>assistant\nHi<|im_end|>"
    tokens = tokenizer.encode(test_text, add_special_tokens=False)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    mask = [1.0] * len(input_tokens)

    print(f"Sequence: {len(tokens)} tokens")

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    print("\n" + "=" * 70)
    print("PHASE 1: Create LoRA and train 5 steps")
    print("=" * 70)

    client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)
    print(f"Model ID: {client.model_id}")

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.ones(len(input_tokens), dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.zeros(len(input_tokens), dtype=torch.float32)),
        }
    )

    for step in range(5):
        fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
        await fwd_bwd.result_async()
        await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()
    print("  Training completed.")

    print("\n" + "=" * 70)
    print("PHASE 2: Export to vLLM and get checkpoint path")
    print("=" * 70)

    sampling_client = await client.save_weights_and_get_sampling_client_async()

    # Find the checkpoint path (ephemeral checkpoints are in /tmp/tinker-checkpoints)
    # Use the API to get the checkpoint info
    base_url = os.environ["TINKER_BASE_URL"]
    import httpx

    # Get model info to find checkpoint path
    async with httpx.AsyncClient(timeout=60.0) as http_client:
        response = await http_client.get(f"{base_url}/api/v1/models/{client.model_id}")
        if response.status_code == 200:
            model_info = response.json()
            print(f"  Model info: {model_info}")
        else:
            print(f"  Could not get model info: {response.status_code}")

    print("\n" + "=" * 70)
    print("PHASE 3: Inspect exported checkpoint")
    print("=" * 70)

    # List checkpoint directories
    checkpoint_base = Path("/tmp/tinker-checkpoints")
    if checkpoint_base.exists():
        # Find most recent checkpoint
        checkpoints = sorted(checkpoint_base.glob("*/adapter_model.safetensors"), key=lambda p: p.stat().st_mtime, reverse=True)
        if checkpoints:
            checkpoint_path = checkpoints[0].parent
            print(f"  Found checkpoint: {checkpoint_path}")

            # Load the checkpoint
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path / "adapter_model.safetensors")

            print(f"\n  Total keys: {len(state_dict)}")

            # Find MoE layer keys (layers 1+ are MoE in Moonlight)
            moe_keys = [k for k in state_dict.keys() if '.layers.1.' in k and '.mlp.' in k]
            attention_keys = [k for k in state_dict.keys() if '.layers.1.' in k and '.self_attn.' in k]
            layer0_keys = [k for k in state_dict.keys() if '.layers.0.' in k]

            print(f"  Layer 0 (dense) keys: {len(layer0_keys)}")
            print(f"  Layer 1 attention keys: {len(attention_keys)}")
            print(f"  Layer 1 MoE MLP keys: {len(moe_keys)}")

            # Check if we have per-expert keys
            expert_0_keys = [k for k in moe_keys if '.experts.0.' in k]
            expert_1_keys = [k for k in moe_keys if '.experts.1.' in k]
            shared_mlp_keys = [k for k in moe_keys if '.mlp.' in k and '.experts.' not in k and '.shared_expert.' not in k]

            print(f"\n  Per-expert keys (expert 0): {len(expert_0_keys)}")
            print(f"  Per-expert keys (expert 1): {len(expert_1_keys)}")
            print(f"  Shared MLP keys (non-expert): {len(shared_mlp_keys)}")

            if expert_0_keys and expert_1_keys:
                print("\n  Per-expert expansion verified!")

                # Check if expert 0 and expert 1 weights are identical
                for key0 in expert_0_keys[:5]:
                    key1 = key0.replace('.experts.0.', '.experts.1.')
                    if key1 in state_dict:
                        t0 = state_dict[key0]
                        t1 = state_dict[key1]
                        diff = (t0 - t1).abs().max().item()
                        print(f"    {key0[-50:]}: diff={diff:.6f}, shape={list(t0.shape)}")
            else:
                print("\n  WARNING: No per-expert keys found!")
                print(f"  Sample MoE keys: {moe_keys[:5]}")

            # Show sample key shapes
            print("\n  Sample key shapes:")
            for key in list(state_dict.keys())[:10]:
                print(f"    {key}: {list(state_dict[key].shape)}")
        else:
            print("  No checkpoints found!")
    else:
        print(f"  Checkpoint base directory not found: {checkpoint_base}")

    print("\n" + "=" * 70)
    print("PHASE 4: Compare logprobs")
    print("=" * 70)

    # Get Megatron logprobs
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    mega_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    # Get vLLM logprobs
    prompt = tinker.ModelInput.from_ints(tokens)
    vllm_lp = await sampling_client.compute_logprobs_async(prompt)

    print("\nPosition-by-position comparison (Megatron[i] vs vLLM[i+1]):")
    for pos in range(min(len(mega_lp), len(vllm_lp) - 1)):
        tgt = target_tokens[pos] if pos < len(target_tokens) else -1
        tgt_str = tokenizer.decode([tgt])[:8] if tgt >= 0 else "N/A"
        m = mega_lp[pos]
        v = vllm_lp[pos + 1] if pos + 1 < len(vllm_lp) and vllm_lp[pos + 1] is not None else float('nan')
        diff = m - v
        flag = " <-- MISMATCH" if abs(diff) > 1 else ""
        print(f"  pos {pos:2d}: M={m:8.4f}, V={v:8.4f}, diff={diff:+8.4f}, target='{tgt_str}'{flag}")


if __name__ == "__main__":
    asyncio.run(main())
