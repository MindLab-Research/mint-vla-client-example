#!/usr/bin/env python3
"""Test if degradation is in weights - verify with both Megatron reload AND vLLM."""

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

def get_diag_entries():
    output = run_ssh(f"cat {DIAG_LOG} 2>/dev/null")
    entries = []
    for line in output.split('\n'):
        if not line or 'RAW-LOGIT-PRE' not in line:
            continue
        match = re.search(r'target_logit=([-\d.]+).*max_token=(\d+)', line)
        if match:
            entries.append({
                'target_logit': float(match.group(1)),
                'max_token': int(match.group(2)),
            })
    return entries

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
    print(f"Position 7 target token: {target_tokens[7]} = '{tokenizer.decode([target_tokens[7]])}'")

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    # Kill existing actors for clean state
    print("\nKilling existing Megatron actors...")
    kill_megatron_actors()
    await asyncio.sleep(5)

    # PHASE 1: Baseline
    print("\n" + "="*70)
    print("PHASE 1: Baseline (fresh Megatron, zero LoRA)")
    print("="*70)

    clear_diag_log()
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

    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    baseline_logprobs = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    await asyncio.sleep(2)
    entries1 = get_diag_entries()

    print(f"Baseline pos7 logprob: {baseline_logprobs[7]:.4f}")
    if entries1:
        print(f"  Avg target_logit: {np.mean([e['target_logit'] for e in entries1]):.4f}")

    # PHASE 2: Train
    print("\n" + "="*70)
    print("PHASE 2: Training 10 steps")
    print("="*70)

    for step in range(10):
        fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
        await fwd_bwd.result_async()
        await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()
        print(f"  Step {step+1} complete")

    # PHASE 3: After training (Megatron)
    print("\n" + "="*70)
    print("PHASE 3: After training (same Megatron)")
    print("="*70)

    clear_diag_log()
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    trained_logprobs = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    await asyncio.sleep(2)
    entries3 = get_diag_entries()

    print(f"Trained pos7 logprob: {trained_logprobs[7]:.4f}")
    if entries3:
        print(f"  Avg target_logit: {np.mean([e['target_logit'] for e in entries3]):.4f}, max_token: {entries3[0]['max_token']}")

    # PHASE 4: Save weights and get sampling client (vLLM)
    print("\n" + "="*70)
    print("PHASE 4: Save weights and get vLLM sampling client")
    print("="*70)

    sampling_client = await client.save_weights_and_get_sampling_client_async()
    print(f"Got sampling client for vLLM")

    # PHASE 5: vLLM logprobs with trained weights
    print("\n" + "="*70)
    print("PHASE 5: vLLM logprobs (trained weights)")
    print("="*70)

    # For vLLM, we need to use the full sequence as prompt and get logprobs
    # Use asample with logprobs enabled, max_tokens=1 to just get the logprobs of input
    prompt_tokens = input_tokens  # All tokens except last

    sample_result = await sampling_client.sample_async(
        prompt=tinker.ModelInput.from_ints(prompt_tokens),
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=1,
            temperature=0.0,
        ),
        include_prompt_logprobs=True,
        topk_prompt_logprobs=5,
    )

    vllm_trained_logprobs = None
    if sample_result.prompt_logprobs:
        # prompt_logprobs is List[Optional[float]] - logprob of each prompt token
        vllm_trained_logprobs = np.array([lp if lp is not None else 0.0 for lp in sample_result.prompt_logprobs])
        print(f"vLLM (trained) pos7 logprob: {vllm_trained_logprobs[7]:.4f}")
    else:
        print("WARNING: No prompt_logprobs returned from vLLM")

    # PHASE 6: Fresh Megatron + loaded trained weights
    print("\n" + "="*70)
    print("PHASE 6: Fresh Megatron (zero LoRA) then load trained")
    print("="*70)

    # First save checkpoint for Megatron reload
    save_future = await client.save_state_async(name="test_weights_vs_state")
    save_result = await save_future.result_async()
    checkpoint_path = save_result.path
    print(f"Saved checkpoint to: {checkpoint_path}")

    # Convert to absolute path
    if checkpoint_path.startswith("tinker://local/"):
        path_part = checkpoint_path[len("tinker://local/"):]
        abs_checkpoint_uri = f"tinker://localhost/vePFS-Mindverse/share/code/tinker-server/checkpoints/{path_part}"
    else:
        abs_checkpoint_uri = checkpoint_path

    kill_megatron_actors()
    await asyncio.sleep(5)

    clear_diag_log()
    client2 = await service_client.create_lora_training_client_async(model_name, rank=16)

    # First check zero LoRA
    fwd = await client2.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    fresh_zero_logprobs = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())
    print(f"Fresh (zero) pos7 logprob: {fresh_zero_logprobs[7]:.4f}")

    # Load trained weights
    print(f"Loading from: {abs_checkpoint_uri}")
    load_future = await client2.load_state_async(abs_checkpoint_uri)
    await load_future.result_async()
    print("Loaded.")

    clear_diag_log()
    fwd = await client2.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    fresh_loaded_logprobs = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())
    print(f"Fresh (loaded) pos7 logprob: {fresh_loaded_logprobs[7]:.4f}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Phase':<45} {'Logprob':>10}")
    print("-"*60)
    print(f"{'1. Baseline (Megatron, zero LoRA)':<45} {baseline_logprobs[7]:>10.4f}")
    print(f"{'3. After training (Megatron, same actor)':<45} {trained_logprobs[7]:>10.4f}")
    if vllm_trained_logprobs is not None:
        print(f"{'5. vLLM (trained weights via PEFT export)':<45} {vllm_trained_logprobs[7]:>10.4f}")
    print(f"{'6a. Fresh Megatron (zero LoRA)':<45} {fresh_zero_logprobs[7]:>10.4f}")
    print(f"{'6b. Fresh Megatron (loaded trained)':<45} {fresh_loaded_logprobs[7]:>10.4f}")

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    megatron_match = abs(fresh_loaded_logprobs[7] - trained_logprobs[7]) < 0.01
    print(f"Megatron reload matches training: {megatron_match} (diff={abs(fresh_loaded_logprobs[7] - trained_logprobs[7]):.4f})")

    if vllm_trained_logprobs is not None:
        vllm_match = abs(vllm_trained_logprobs[7] - trained_logprobs[7]) < 0.5
        print(f"vLLM matches Megatron trained: {vllm_match} (diff={abs(vllm_trained_logprobs[7] - trained_logprobs[7]):.4f})")

        if vllm_match:
            print("\n>>> CONCLUSION: vLLM confirms degradation is in WEIGHTS (PEFT export correct)")
        else:
            print("\n>>> CONCLUSION: vLLM differs from Megatron - possible export issue")

if __name__ == "__main__":
    asyncio.run(main())
