#!/usr/bin/env python3
"""Capture logits at position 7 for baseline vs trained LoRA."""

import asyncio
import os
import subprocess

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker
import torch

DIAG_LOG = "/vePFS-Mindverse/share/code/raw_logit_diag.log"


def run_ssh(cmd):
    result = subprocess.run(["ssh", "volcano", cmd], capture_output=True, text=True)
    return result.stdout.strip()


def clear_diag_log():
    run_ssh(f"rm -f {DIAG_LOG}")


def get_diag_log():
    return run_ssh(f"cat {DIAG_LOG} 2>/dev/null")


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

    # Kill existing actors
    print("\nKilling existing Megatron actors...")
    kill_megatron_actors()
    await asyncio.sleep(5)

    # Create client
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

    # === PHASE 1: BASELINE ===
    print("\n=== BASELINE (zero LoRA) ===")
    clear_diag_log()
    await asyncio.sleep(1)

    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    baseline_lp7 = result.loss_fn_outputs[0]["logprobs"].to_numpy()[7]
    print(f"Baseline logprob at pos7: {baseline_lp7:.4f}")

    await asyncio.sleep(1)
    baseline_log = get_diag_log()
    print("\nBaseline logits (tp=0 only):")
    for line in baseline_log.split('\n'):
        if 'tp=0' in line and 'TARGET_LOGIT' in line:
            print(f"  {line}")

    # === PHASE 2: TRAIN 10 STEPS ===
    print("\n=== Training 10 steps ===")
    for step in range(10):
        fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
        await fwd_bwd.result_async()
        await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()

    # === PHASE 3: TRAINED ===
    print("\n=== TRAINED (after 10 steps) ===")
    clear_diag_log()
    await asyncio.sleep(1)

    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    trained_lp7 = result.loss_fn_outputs[0]["logprobs"].to_numpy()[7]
    print(f"Trained logprob at pos7: {trained_lp7:.4f}")

    await asyncio.sleep(1)
    trained_log = get_diag_log()
    print("\nTrained logits (tp=0 only):")
    for line in trained_log.split('\n'):
        if 'tp=0' in line and 'TARGET_LOGIT' in line:
            print(f"  {line}")

    # === SUMMARY ===
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Baseline pos7 logprob: {baseline_lp7:.4f}")
    print(f"Trained pos7 logprob:  {trained_lp7:.4f}")
    print(f"Diff:                  {trained_lp7 - baseline_lp7:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
