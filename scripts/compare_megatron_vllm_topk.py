#!/usr/bin/env python3
"""Compare exact logits between Megatron and vLLM at position 7."""

import asyncio
import os
import subprocess

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker
import torch
import numpy as np


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

    # Also decode nearby tokens for context
    print("\nContext around position 7:")
    for i in range(5, 12):
        if i < len(target_tokens):
            print(f"  pos {i}: input={input_tokens[i]} '{tokenizer.decode([input_tokens[i]])}' -> target={target_tokens[i]} '{tokenizer.decode([target_tokens[i]])}'")

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    # Kill existing actors
    print("\nKilling existing Megatron actors...")
    kill_megatron_actors()
    await asyncio.sleep(5)

    # Create client and train
    print("\n=== Creating fresh Megatron client and training ===")
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
    for step in range(10):
        fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
        await fwd_bwd.result_async()
        await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()
    print("Training complete.")

    # === MEGATRON FORWARD ===
    print("\n=== Megatron forward (trained) ===")
    clear_diag_log()
    await asyncio.sleep(1)

    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    megatron_lp7 = result.loss_fn_outputs[0]["logprobs"].to_numpy()[7]
    print(f"Megatron logprob at pos7: {megatron_lp7:.4f}")

    await asyncio.sleep(1)
    meg_diag = get_diag_log()
    print("Megatron logits:")
    for line in meg_diag.split('\n'):
        if 'tp=0' in line and 'TARGET_LOGIT' in line and 'pos=7' in line:
            print(f"  {line}")

    # === vLLM FORWARD ===
    print("\n=== vLLM forward (trained weights via PEFT export) ===")
    sampling_client = await client.save_weights_and_get_sampling_client_async()

    # Get top-k logprobs at position 7
    sample_result = await sampling_client.sample_async(
        prompt=tinker.ModelInput.from_ints(input_tokens),
        num_samples=1,
        sampling_params=tinker.SamplingParams(max_tokens=1, temperature=0.0),
        include_prompt_logprobs=True,
        topk_prompt_logprobs=50,  # Get top-50 for position 7
    )

    vllm_lp7 = sample_result.prompt_logprobs[7] if sample_result.prompt_logprobs[7] is not None else 0.0
    print(f"vLLM logprob at pos7: {vllm_lp7:.4f}")

    # Print top-k at position 7
    if sample_result.topk_prompt_logprobs and len(sample_result.topk_prompt_logprobs) > 7:
        topk = sample_result.topk_prompt_logprobs[7]
        if topk:
            print("\nvLLM top-20 at position 7:")
            target_rank = None
            for rank, (tok_id, logprob) in enumerate(topk[:20]):
                tok_str = tokenizer.decode([tok_id]).replace('\n', '\\n')
                marker = " <-- TARGET" if tok_id == target_tokens[7] else ""
                if tok_id == target_tokens[7]:
                    target_rank = rank
                print(f"  {rank+1:2d}. token={tok_id:6d} '{tok_str:15s}' logprob={logprob:.4f}{marker}")

            if target_rank is not None:
                print(f"\nTarget token 3922 ('Count') rank in vLLM: {target_rank + 1}")

            # Check if Chinese token 6955 is in top-k
            chinese_token = 6955
            for rank, (tok_id, logprob) in enumerate(topk):
                if tok_id == chinese_token:
                    tok_str = tokenizer.decode([tok_id])
                    print(f"\nChinese token {chinese_token} ('{tok_str}') in vLLM: rank={rank+1}, logprob={logprob:.4f}")
                    break
            else:
                print(f"\nChinese token {chinese_token} ('我想') NOT in vLLM top-50!")

    # === SUMMARY ===
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"Megatron logprob pos7: {megatron_lp7:.4f}")
    print(f"vLLM logprob pos7:     {vllm_lp7:.4f}")
    print(f"Diff:                  {vllm_lp7 - megatron_lp7:.4f}")
    print()
    print("HYPOTHESIS:")
    print("If Megatron shows Chinese '我想' at high logit but vLLM doesn't,")
    print("the LoRA is being applied differently between the two systems.")


if __name__ == "__main__":
    asyncio.run(main())
