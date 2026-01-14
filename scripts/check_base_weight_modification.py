#!/usr/bin/env python3
"""Check if base model weights are modified during LoRA training.

Hypothesis: Base weights should be frozen during LoRA training.
If they're being modified, this could explain the train-inference mismatch.
"""

import asyncio
import os
import subprocess

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker
import torch


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


def check_base_weight_modification():
    """Check base weight modification on the server."""
    cmd = '''python3 << 'PYEOF'
import ray
import torch

ray.init(address="auto", ignore_reinit_error=True)

# Find the Megatron actor
actors = [a for a in ray.util.list_named_actors(all_namespaces=True) if 'megatron' in a['name'].lower()]
if not actors:
    print("No Megatron actor found")
    exit(1)

print(f"Found actor: {actors[0]['name']}")

# We need to access the model's base weights
# This would require a remote method on the actor

# For now, let's check the state_dict for any changes
# by looking at the weight norms and requires_grad flags
print("Need to implement weight inspection on the actor side")
PYEOF
'''
    result = subprocess.run(["ssh", "volcano", cmd], capture_output=True, text=True)
    print(result.stdout)


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

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    # Kill existing actors
    print("Killing existing Megatron actors...")
    kill_megatron_actors()
    await asyncio.sleep(5)

    # Create client
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

    # Get baseline logprob
    print("\n=== BASELINE ===")
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    baseline_lp7 = result.loss_fn_outputs[0]["logprobs"].to_numpy()[7]
    print(f"Baseline logprob at pos7: {baseline_lp7:.4f}")

    # Check weights before training
    print("\n=== Checking weights before training ===")
    check_base_weight_modification()

    # Train 10 steps
    print("\n=== Training 10 steps ===")
    for step in range(10):
        fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
        await fwd_bwd.result_async()
        await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()

    # Check weights after training
    print("\n=== Checking weights after training ===")
    check_base_weight_modification()

    # Get trained logprob
    print("\n=== TRAINED ===")
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    trained_lp7 = result.loss_fn_outputs[0]["logprobs"].to_numpy()[7]
    print(f"Trained logprob at pos7: {trained_lp7:.4f}")

    print("\n=== SUMMARY ===")
    print(f"Baseline: {baseline_lp7:.4f}")
    print(f"Trained:  {trained_lp7:.4f}")
    print(f"Diff:     {trained_lp7 - baseline_lp7:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
