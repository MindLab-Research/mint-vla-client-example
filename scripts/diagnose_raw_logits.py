#!/usr/bin/env python3
"""Check raw logits at specific positions in Megatron.

Adds server-side diagnostics to dump logits for problematic positions.
"""

import asyncio
import os
import subprocess
import re
import numpy as np

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

    print(f"Sequence length: {len(input_tokens)} tokens")

    # Print problematic positions
    problematic = [1, 2, 7, 9, 16, 17, 21, 22]
    print("\nProblematic positions:")
    for pos in problematic:
        if pos < len(target_tokens):
            tok = target_tokens[pos]
            tok_str = tokenizer.decode([tok]).replace('\n', '\\n')
            print(f"  pos {pos}: target={tok} '{tok_str}'")

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    # Kill existing actors
    print("\nKilling existing Megatron actors...")
    kill_megatron_actors()
    await asyncio.sleep(5)

    # Create client and train
    print("\n=== Creating client and training ===")
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

    for step in range(10):
        fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
        await fwd_bwd.result_async()
        await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()
    print("Training complete.")

    # Clear diagnostic log and run forward
    print("\n=== Running Megatron forward ===")
    clear_diag_log()
    await asyncio.sleep(1)

    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    megatron_logprobs = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    await asyncio.sleep(2)

    # Get diagnostic log
    print("\n=== Server-side diagnostics ===")
    diag = get_diag_log()
    print(diag[:3000] if len(diag) > 3000 else diag)

    # Print Megatron logprobs at problematic positions
    print("\n=== Megatron logprobs at problematic positions ===")
    for pos in problematic:
        if pos < len(megatron_logprobs):
            tok = target_tokens[pos]
            tok_str = tokenizer.decode([tok]).replace('\n', '\\n')
            print(f"  pos {pos}: target={tok} '{tok_str}' -> logprob={megatron_logprobs[pos]:.6f}")

    # Also check which positions have -0.0000 (very suspicious)
    print("\n=== Positions with logprob > -0.001 (too confident) ===")
    for pos in range(len(megatron_logprobs)):
        if megatron_logprobs[pos] > -0.001:
            tok = target_tokens[pos] if pos < len(target_tokens) else -1
            tok_str = tokenizer.decode([tok]).replace('\n', '\\n') if tok >= 0 else "?"
            print(f"  pos {pos}: target={tok} '{tok_str}' -> logprob={megatron_logprobs[pos]:.6f}")


if __name__ == "__main__":
    asyncio.run(main())
