#!/usr/bin/env python3
"""Test if fresh Megatron actor with PEFT weights matches vLLM.

This tests whether the issue is:
1. Specific to the TRAINED actor (state corruption)
2. Or inherent in Megatron LoRA application (weight application bug)

Flow:
1. Kill all Megatron actors
2. Create fresh Megatron client
3. Train 10 steps
4. Export PEFT weights
5. Get Megatron logprob (should be wrong: -17.76)
6. Get vLLM logprob (should be correct: -0.0067)
7. Kill Megatron actor
8. Create NEW fresh Megatron client
9. LOAD the same PEFT weights
10. Get Megatron logprob - does it match vLLM or trained actor?
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

    # === PHASE 1: Train and get trained actor's logprob ===
    print("\n" + "="*60)
    print("PHASE 1: Train a fresh actor and capture logprob")
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

    print("\nTraining 10 steps...")
    for step in range(10):
        fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
        await fwd_bwd.result_async()
        await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()

    print("\nGetting TRAINED actor logprob...")
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    trained_lp7 = result.loss_fn_outputs[0]["logprobs"].to_numpy()[7]
    print(f"TRAINED Megatron logprob at pos7: {trained_lp7:.4f}")

    # === PHASE 2: Export PEFT and test vLLM ===
    print("\n" + "="*60)
    print("PHASE 2: Export PEFT and test vLLM")
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

    # Find the PEFT export path
    peft_path = run_ssh("ls -td /tmp/lora_export_* 2>/dev/null | head -1")
    print(f"\nPEFT export path: {peft_path}")

    # === PHASE 3: Kill actor and reload with PEFT weights ===
    print("\n" + "="*60)
    print("PHASE 3: Fresh Megatron actor + LOAD PEFT weights")
    print("="*60)

    print("\nKilling trained Megatron actor...")
    kill_megatron_actors()
    await asyncio.sleep(5)

    print("\nCreating fresh Megatron client...")
    client2 = await service_client.create_lora_training_client_async(model_name, rank=16)

    print("\nBaseline (zero LoRA) logprob from fresh actor...")
    fwd_baseline = await client2.forward_async([datum], loss_fn="importance_sampling")
    result_baseline = await fwd_baseline.result_async()
    baseline_lp7 = result_baseline.loss_fn_outputs[0]["logprobs"].to_numpy()[7]
    print(f"FRESH BASELINE Megatron logprob at pos7: {baseline_lp7:.4f}")

    # TODO: Load PEFT weights into this fresh actor
    # This requires implementing load_weights in the API
    # For now, we can't directly test this

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"TRAINED actor logprob:    {trained_lp7:.4f}")
    print(f"vLLM (same PEFT) logprob: {vllm_lp7:.4f}")
    print(f"FRESH BASELINE logprob:   {baseline_lp7:.4f}")
    print()
    print("ANALYSIS:")
    print(f"  Train-vLLM diff: {trained_lp7 - vllm_lp7:.4f}")
    print()
    print("If trained actor shows -17.76 but vLLM shows -0.0067,")
    print("and a fresh actor with LOADED PEFT would match vLLM,")
    print("then the issue is STATE CORRUPTION in the trained actor.")
    print()
    print("If fresh actor with LOADED PEFT also shows -17.76,")
    print("then the issue is in Megatron's LoRA APPLICATION, not state.")


if __name__ == "__main__":
    asyncio.run(main())
