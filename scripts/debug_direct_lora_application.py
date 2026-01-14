#!/usr/bin/env python3
"""Direct comparison of LoRA application between Megatron and manual computation.

This script:
1. Trains Megatron for 10 steps
2. Gets the trained logits from Megatron at position 7
3. Exports PEFT weights
4. Loads PEFT weights manually and computes what output SHOULD be
5. Compares to see if there's a discrepancy in LoRA application

The goal is to determine if Megatron is applying LoRA weights incorrectly
during forward pass.
"""

import asyncio
import os
import subprocess
import sys

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker
import torch


def run_ssh(cmd, capture=True):
    result = subprocess.run(["ssh", "volcano", cmd], capture_output=capture, text=True)
    return result.stdout.strip() if capture else None


def kill_megatron_actors():
    cmd = '''python3 -c "
import ray
ray.init()
actors = [a for a in ray.util.list_named_actors(all_namespaces=True) if 'Megatron' in a['name'] or 'megatron' in a['name']]
for a in actors:
    try:
        handle = ray.get_actor(a['name'], namespace=a.get('namespace', None))
        ray.kill(handle)
    except: pass
print(f'Killed {len(actors)} actors')
"'''
    result = subprocess.run(["ssh", "volcano", cmd], capture_output=True, text=True)
    print(result.stdout.strip())


def dump_megatron_lora_weights_on_server():
    """Dump LoRA weights from Megatron actor on server side."""
    cmd = '''python3 << 'PYEOF'
import ray
import torch
import json

ray.init(address="auto", ignore_reinit_error=True)

# Find the Megatron actor
actors = [a for a in ray.util.list_named_actors(all_namespaces=True) if 'megatron' in a['name'].lower()]
if not actors:
    print("ERROR: No Megatron actor found")
    exit(1)

actor_info = actors[0]
print(f"Found actor: {actor_info['name']}")

# Get the actor handle
actor = ray.get_actor(actor_info['name'], namespace=actor_info.get('namespace', 'tinker'))

# Call get_lora_state_dict
state_dict = ray.get(actor.get_lora_state_dict.remote(use_per_expert_lora=False), timeout=60)

# Print summary of weights
print(f"\\nLoRA state dict has {len(state_dict)} keys")
for key, tensor in state_dict.items():
    print(f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}, norm={tensor.norm().item():.4f}")

# Check a specific attention layer
for key, tensor in state_dict.items():
    if 'layers.0.' in key and 'lora_A' in key:
        print(f"\\nLayer 0 attention lora_A sample: {key}")
        print(f"  First 5 values: {tensor.flatten()[:5].tolist()}")
        break
PYEOF
'''
    result = subprocess.run(["ssh", "volcano", cmd], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)


async def main():
    model_name = "moonshotai/Moonlight-16B-A3B-Instruct"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    test_text = """<|im_start|>user
Count down from 10 to 1, one number per line.<|im_end|>
<|im_start|>assistant
10
9
8
7
6
5
4
3
2
1<|im_end|>"""

    tokens = tokenizer.encode(test_text, add_special_tokens=False)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    mask = [1.0] * len(input_tokens)

    print(f"Position 7: target={target_tokens[7]} = '{tokenizer.decode([target_tokens[7]])}'")

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    # === PHASE 1: Kill existing and start fresh ===
    print("\n" + "="*60)
    print("PHASE 1: Start fresh Megatron actor")
    print("="*60)

    print("\nKilling existing Megatron actors...")
    kill_megatron_actors()
    await asyncio.sleep(5)

    print("\nCreating fresh Megatron client...")
    client = await service_client.create_lora_training_client_async(model_name, rank=16)

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.ones(len(input_tokens), dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.zeros(len(input_tokens), dtype=torch.float32)),
        }
    )

    # === PHASE 2: Get baseline ===
    print("\n" + "="*60)
    print("PHASE 2: Baseline (zero LoRA)")
    print("="*60)

    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    baseline_lp7 = result.loss_fn_outputs[0]["logprobs"].to_numpy()[7]
    print(f"Baseline logprob at pos7: {baseline_lp7:.4f}")

    # === PHASE 3: Train 10 steps ===
    print("\n" + "="*60)
    print("PHASE 3: Training 10 steps")
    print("="*60)

    for step in range(10):
        fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
        await fwd_bwd.result_async()
        await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()
        if step % 2 == 0:
            print(f"  Step {step+1}/10 done")

    # === PHASE 4: Get trained logprob from Megatron ===
    print("\n" + "="*60)
    print("PHASE 4: Megatron trained forward")
    print("="*60)

    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    trained_lp7 = result.loss_fn_outputs[0]["logprobs"].to_numpy()[7]
    print(f"Trained Megatron logprob at pos7: {trained_lp7:.4f}")

    # === PHASE 5: Dump LoRA weights from Megatron ===
    print("\n" + "="*60)
    print("PHASE 5: Dumping LoRA weights from Megatron actor")
    print("="*60)
    dump_megatron_lora_weights_on_server()

    # === PHASE 6: Export PEFT and get vLLM result ===
    print("\n" + "="*60)
    print("PHASE 6: Export PEFT and test vLLM")
    print("="*60)

    print("\nExporting PEFT weights...")
    sampling_client = await client.save_weights_and_get_sampling_client_async()

    # Get vLLM logprob
    sample_result = await sampling_client.sample_async(
        prompt=tinker.ModelInput.from_ints(input_tokens),
        num_samples=1,
        sampling_params=tinker.SamplingParams(max_tokens=1, temperature=0.0),
        include_prompt_logprobs=True,
    )
    vllm_lp7 = sample_result.prompt_logprobs[7] if sample_result.prompt_logprobs[7] is not None else 0.0
    print(f"vLLM logprob at pos7: {vllm_lp7:.4f}")

    # Find and print the PEFT export path
    peft_path = run_ssh("ls -td /tmp/lora_export_* 2>/dev/null | head -1")
    print(f"\nPEFT export path: {peft_path}")

    # List PEFT files
    peft_files = run_ssh(f"ls -la {peft_path}/ 2>/dev/null | head -10") if peft_path else ""
    print(f"PEFT files:\n{peft_files}")

    # === SUMMARY ===
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Baseline logprob:         {baseline_lp7:.4f}")
    print(f"Trained Megatron logprob: {trained_lp7:.4f}")
    print(f"vLLM (same PEFT) logprob: {vllm_lp7:.4f}")
    print()
    print(f"Megatron train effect: {trained_lp7 - baseline_lp7:.4f}")
    print(f"vLLM train effect:     {vllm_lp7 - baseline_lp7:.4f}")
    print()
    print(f"Train-vLLM mismatch:   {trained_lp7 - vllm_lp7:.4f}")
    print()
    if abs(trained_lp7 - vllm_lp7) > 1.0:
        print(">>> SIGNIFICANT MISMATCH - Megatron is applying LoRA differently than vLLM")
    else:
        print(">>> Results match within tolerance")


if __name__ == "__main__":
    asyncio.run(main())
