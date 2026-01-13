#!/usr/bin/env python3
"""Diagnose LoRA weight application during Megatron inference.

Compares:
1. LoRA weights in Megatron memory during inference
2. PEFT export weights saved to disk
3. vLLM logprobs with same PEFT weights

Goal: Identify where Megatron's LoRA application diverges from vLLM.
"""

import asyncio
import os
import subprocess

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker
import torch
import numpy as np


def run_ssh(cmd):
    result = subprocess.run(["ssh", "volcano", cmd], capture_output=True, text=True)
    return result.stdout.strip()


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


def dump_megatron_lora_weights():
    """Dump LoRA weights from Megatron actor to a file for comparison."""
    cmd = '''python3 << 'PYEOF'
import ray
import torch
import os

ray.init(address="auto", ignore_reinit_error=True)

# Find the Megatron actor
actors = [a for a in ray.util.list_named_actors(all_namespaces=True) if 'megatron' in a['name'].lower()]
if not actors:
    print("No Megatron actor found")
    exit(1)

print(f"Found actors: {actors}")
actor_info = actors[0]
actor = ray.get_actor(actor_info['name'], namespace=actor_info.get('namespace'))

# Get LoRA weights via actor method
# We need to add a method to dump weights, or access them via state_dict
import pickle

# Try to get the model and dump LoRA weights
@ray.remote
def get_lora_weights(actor):
    """Remote function to extract LoRA weights from actor."""
    # This won't work directly - we need to add a method to the worker
    pass

# Alternative: dump via model state_dict export
# The save_weights_async method already does this!
print("Use save_weights_and_get_sampling_client_async to get PEFT export")
print("Then compare PEFT weights vs what Megatron computes during forward")
PYEOF
'''
    result = subprocess.run(["ssh", "volcano", cmd], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr}")


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

    # Kill existing actors for clean state
    print("\nKilling existing Megatron actors...")
    kill_megatron_actors()
    await asyncio.sleep(5)

    # Create client and train
    print("\n=== Creating fresh Megatron client ===")
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

    # Train 10 steps
    print("\n=== Training 10 steps ===")
    for step in range(10):
        fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
        await fwd_bwd.result_async()
        await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()
    print("Training complete.")

    # === MEGATRON FORWARD ===
    print("\n=== Megatron forward (trained) ===")
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    megatron_lp7 = result.loss_fn_outputs[0]["logprobs"].to_numpy()[7]
    print(f"Megatron logprob at pos7: {megatron_lp7:.4f}")

    # === EXPORT PEFT WEIGHTS ===
    print("\n=== Exporting PEFT weights ===")
    sampling_client = await client.save_weights_and_get_sampling_client_async()

    # The PEFT weights are saved at a known location
    # Let's check the weights directory
    print("\n=== Checking PEFT export ===")
    peft_check = run_ssh("ls -la /tmp/lora_export_* 2>/dev/null | tail -5 || echo 'No export dirs'")
    print(peft_check)

    # Find the latest export and examine weights
    peft_examine = run_ssh('''
cd /tmp &&
latest=$(ls -td lora_export_* 2>/dev/null | head -1)
if [ -n "$latest" ]; then
    echo "Latest export: $latest"
    ls -la "$latest"
    python3 << PYEOF
import torch
import os
import glob

# Find adapter files
adapter_files = glob.glob("/tmp/lora_export_*/adapter_model.safetensors")
if adapter_files:
    from safetensors import safe_open
    with safe_open(adapter_files[-1], framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"\\nPEFT keys ({len(keys)} total):")
        for k in keys[:10]:
            tensor = f.get_tensor(k)
            print(f"  {k}: shape={tensor.shape}, norm={tensor.norm().item():.4f}")

        # Specifically look at layer 0 qkv LoRA
        for k in keys:
            if "layers.0" in k and "qkv" in k:
                tensor = f.get_tensor(k)
                print(f"\\nLayer 0 QKV: {k}")
                print(f"  shape={tensor.shape}, norm={tensor.norm().item():.4f}")
                print(f"  min={tensor.min().item():.4f}, max={tensor.max().item():.4f}")
                break
PYEOF
fi
''')
    print(peft_examine)

    # === vLLM FORWARD ===
    print("\n=== vLLM forward (same PEFT weights) ===")
    sample_result = await sampling_client.sample_async(
        prompt=tinker.ModelInput.from_ints(input_tokens),
        num_samples=1,
        sampling_params=tinker.SamplingParams(max_tokens=1, temperature=0.0),
        include_prompt_logprobs=True,
        topk_prompt_logprobs=20,
    )

    vllm_lp7 = sample_result.prompt_logprobs[7] if sample_result.prompt_logprobs[7] is not None else 0.0
    print(f"vLLM logprob at pos7: {vllm_lp7:.4f}")

    # === SECOND MEGATRON FORWARD ===
    # Key test: Does Megatron give the same result if we run forward again?
    print("\n=== Second Megatron forward (check consistency) ===")
    fwd2 = await client.forward_async([datum], loss_fn="importance_sampling")
    result2 = await fwd2.result_async()
    megatron_lp7_2 = result2.loss_fn_outputs[0]["logprobs"].to_numpy()[7]
    print(f"Megatron logprob at pos7 (run 2): {megatron_lp7_2:.4f}")
    print(f"Diff between runs: {abs(megatron_lp7 - megatron_lp7_2):.6f}")

    # === SUMMARY ===
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"Megatron logprob pos7 (run 1): {megatron_lp7:.4f}")
    print(f"Megatron logprob pos7 (run 2): {megatron_lp7_2:.4f}")
    print(f"vLLM logprob pos7:             {vllm_lp7:.4f}")
    print(f"Megatron vs vLLM diff:         {megatron_lp7 - vllm_lp7:.4f}")

    if sample_result.topk_prompt_logprobs and len(sample_result.topk_prompt_logprobs) > 7:
        topk = sample_result.topk_prompt_logprobs[7]
        if topk:
            print("\nvLLM top-5 at position 7:")
            for rank, (tok_id, logprob) in enumerate(topk[:5]):
                tok_str = tokenizer.decode([tok_id]).replace('\n', '\\n')
                marker = " <-- TARGET" if tok_id == target_tokens[7] else ""
                print(f"  {rank+1}. token={tok_id:6d} '{tok_str:15s}' logprob={logprob:.4f}{marker}")


if __name__ == "__main__":
    asyncio.run(main())
