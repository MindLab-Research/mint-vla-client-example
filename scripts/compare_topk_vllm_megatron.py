#!/usr/bin/env python3
"""Compare top-K tokens between vLLM and Megatron.

This script:
1. Trains Megatron for 10 steps (captures raw logits via diagnostic log)
2. Exports PEFT to vLLM
3. Gets top-K logprobs from vLLM for the same input
4. Compares the two

Run locally: python scripts/compare_topk_vllm_megatron.py
"""

import asyncio
import os
import subprocess
import sys

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker
import torch


def run_ssh(cmd):
    result = subprocess.run(["ssh", "volcano", cmd], capture_output=True, text=True)
    return result.stdout.strip()


def get_megatron_diagnostic():
    """Get latest raw logits from Megatron diagnostic log."""
    log = run_ssh('tail -50 /vePFS-Mindverse/share/code/raw_logit_diag.log 2>/dev/null | grep "tp=0"')
    return log


def kill_megatron():
    cmd = '''python3 -c "
import ray
ray.init()
actors = [a for a in ray.util.list_named_actors(all_namespaces=True) if 'megatron' in a['name'].lower()]
for a in actors:
    try:
        handle = ray.get_actor(a['name'], namespace=a.get('namespace', None))
        ray.kill(handle)
    except: pass
print(f'Killed {len(actors)} Megatron actors')
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

    print("="*60)
    print("TOKEN ANALYSIS")
    print("="*60)
    for pos in [7, 8, 23]:
        print(f"pos={pos}: target={target_tokens[pos]} ('{tokenizer.decode([target_tokens[pos]])}')")

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    # Clear diagnostic log
    run_ssh('echo "" > /vePFS-Mindverse/share/code/raw_logit_diag.log')

    # === PHASE 1: Fresh Megatron baseline ===
    print("\n" + "="*60)
    print("PHASE 1: Kill existing, start fresh Megatron")
    print("="*60)
    kill_megatron()
    await asyncio.sleep(5)

    client = await service_client.create_lora_training_client_async(model_name, rank=16)

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.ones(len(input_tokens), dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.ones(len(input_tokens), dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.zeros(len(input_tokens), dtype=torch.float32)),
        }
    )

    # Baseline forward
    print("\nRunning baseline forward...")
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    baseline_lp7 = result.loss_fn_outputs[0]["logprobs"].to_numpy()[7]
    print(f"Baseline Megatron logprob at pos7: {baseline_lp7:.4f}")

    baseline_diag = get_megatron_diagnostic()
    print(f"\nBaseline Megatron raw logits (from diag log):\n{baseline_diag}")

    # === PHASE 2: Train 10 steps ===
    print("\n" + "="*60)
    print("PHASE 2: Train 10 steps")
    print("="*60)
    for step in range(10):
        fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
        await fwd_bwd.result_async()
        await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()
        if step % 3 == 0:
            print(f"  Step {step+1}/10")

    # Trained forward
    print("\nRunning trained forward...")
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    trained_lp7 = result.loss_fn_outputs[0]["logprobs"].to_numpy()[7]
    print(f"Trained Megatron logprob at pos7: {trained_lp7:.4f}")

    trained_diag = get_megatron_diagnostic()
    print(f"\nTrained Megatron raw logits (from diag log):\n{trained_diag}")

    # === PHASE 3: Export to vLLM and get top-K ===
    print("\n" + "="*60)
    print("PHASE 3: Export PEFT to vLLM, get top-K logprobs")
    print("="*60)
    sampling_client = await client.save_weights_and_get_sampling_client_async()

    # Request top-10 logprobs at each prompt position
    sample_result = await sampling_client.sample_async(
        prompt=tinker.ModelInput.from_ints(input_tokens),
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=1,
            temperature=0.0,
            topk_prompt_logprobs=10,  # Get top 10 at each position
        ),
        include_prompt_logprobs=True,
    )

    print(f"\nvLLM logprob at pos7: {sample_result.prompt_logprobs[7]:.4f}")

    # Check if we got top-k data
    if hasattr(sample_result, 'topk_prompt_logprobs') and sample_result.topk_prompt_logprobs:
        print("\nvLLM Top-K at position 7:")
        topk = sample_result.topk_prompt_logprobs[7]
        for i, (tok, lp) in enumerate(topk[:10]):
            tok_str = tokenizer.decode([tok]) if tok else "?"
            marker = " <-- TARGET" if tok == target_tokens[7] else ""
            print(f"  {i+1}. token={tok:5d} ({repr(tok_str):15s}): logprob={lp:.4f}{marker}")
    else:
        print("No top-K data available in response")
        print(f"Response attributes: {dir(sample_result)}")

    # === SUMMARY ===
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Position 7, target={target_tokens[7]} ('{tokenizer.decode([target_tokens[7]])}')")
    print(f"  Baseline Megatron: {baseline_lp7:.4f}")
    print(f"  Trained Megatron:  {trained_lp7:.4f}  (effect: {trained_lp7 - baseline_lp7:.4f})")
    print(f"  vLLM (same PEFT):  {sample_result.prompt_logprobs[7]:.4f}")
    print()
    if trained_lp7 < baseline_lp7:
        print(">>> BUG: Training DECREASED logprob in Megatron (opposite of expected)")
    if abs(trained_lp7 - sample_result.prompt_logprobs[7]) > 1.0:
        print(f">>> BUG: Megatron vs vLLM mismatch = {trained_lp7 - sample_result.prompt_logprobs[7]:.2f} nats")


if __name__ == "__main__":
    asyncio.run(main())
