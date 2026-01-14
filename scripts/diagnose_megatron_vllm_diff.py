#!/usr/bin/env python3
"""Diagnose Megatron vs vLLM logprob differences with detailed intermediate values.

Gathers evidence:
1. Top-k tokens at each position from both systems
2. Logprobs for all target tokens (pattern analysis)
3. Position-by-position comparison
"""

import asyncio
import os
import subprocess
import numpy as np

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

    print(f"Sequence length: {len(input_tokens)} tokens")
    print(f"\nTarget tokens:")
    for i, tok in enumerate(target_tokens[:30]):
        tok_str = tokenizer.decode([tok]).replace('\n', '\\n')
        print(f"  pos {i:2d}: {tok:5d} = '{tok_str}'")

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    # Kill existing actors for clean state
    print("\nKilling existing Megatron actors...")
    kill_megatron_actors()
    await asyncio.sleep(5)

    # Create training client and train
    print("\n" + "="*70)
    print("PHASE 1: Create client and train 10 steps")
    print("="*70)

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

    # Get Megatron logprobs
    print("\n" + "="*70)
    print("PHASE 2: Megatron forward (trained LoRA)")
    print("="*70)

    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    megatron_logprobs = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    print(f"Megatron logprobs shape: {megatron_logprobs.shape}")
    print(f"Megatron logprobs range: [{megatron_logprobs.min():.4f}, {megatron_logprobs.max():.4f}]")

    # Get vLLM logprobs with top-k
    print("\n" + "="*70)
    print("PHASE 3: vLLM forward (same trained weights)")
    print("="*70)

    sampling_client = await client.save_weights_and_get_sampling_client_async()

    sample_result = await sampling_client.sample_async(
        prompt=tinker.ModelInput.from_ints(input_tokens),
        num_samples=1,
        sampling_params=tinker.SamplingParams(max_tokens=1, temperature=0.0),
        include_prompt_logprobs=True,
        topk_prompt_logprobs=20,  # Get top-20 for comparison
    )

    vllm_logprobs = np.array([lp if lp is not None else 0.0 for lp in sample_result.prompt_logprobs])
    print(f"vLLM logprobs shape: {vllm_logprobs.shape}")
    print(f"vLLM logprobs range: [{vllm_logprobs.min():.4f}, {vllm_logprobs.max():.4f}]")

    # Position-by-position comparison
    print("\n" + "="*70)
    print("PHASE 4: Position-by-position comparison")
    print("="*70)

    print(f"\n{'Pos':>4} {'Target':>6} {'Token':>15} {'Megatron':>12} {'vLLM':>12} {'Diff':>12} {'Flag'}")
    print("-" * 80)

    problematic_positions = []
    for i in range(min(len(megatron_logprobs), len(vllm_logprobs), len(target_tokens))):
        meg_lp = megatron_logprobs[i]
        vllm_lp = vllm_logprobs[i]
        diff = meg_lp - vllm_lp
        tok = target_tokens[i]
        tok_str = tokenizer.decode([tok]).replace('\n', '\\n')[:12]

        flag = ""
        if abs(diff) > 1.0:
            flag = "*** LARGE DIFF"
            problematic_positions.append(i)
        elif abs(diff) > 0.1:
            flag = "* diff"

        print(f"{i:4d} {tok:6d} {tok_str:>15} {meg_lp:12.4f} {vllm_lp:12.4f} {diff:12.4f} {flag}")

    # Detailed analysis of problematic positions
    if problematic_positions and sample_result.topk_prompt_logprobs:
        print("\n" + "="*70)
        print("PHASE 5: Top-k analysis at problematic positions")
        print("="*70)

        for pos in problematic_positions[:5]:  # Analyze first 5 problematic positions
            print(f"\n--- Position {pos} ---")
            target_tok = target_tokens[pos]
            target_str = tokenizer.decode([target_tok]).replace('\n', '\\n')
            print(f"Target token: {target_tok} = '{target_str}'")
            print(f"Megatron logprob: {megatron_logprobs[pos]:.4f}")
            print(f"vLLM logprob: {vllm_logprobs[pos]:.4f}")

            if pos < len(sample_result.topk_prompt_logprobs) and sample_result.topk_prompt_logprobs[pos]:
                topk = sample_result.topk_prompt_logprobs[pos]
                print(f"\nvLLM top-20 tokens:")
                target_rank = None
                for rank, (tok_id, logprob) in enumerate(topk):
                    tok_str = tokenizer.decode([tok_id]).replace('\n', '\\n')
                    marker = " <-- TARGET" if tok_id == target_tok else ""
                    if tok_id == target_tok:
                        target_rank = rank
                    print(f"  {rank+1:2d}. {tok_id:6d} '{tok_str:15s}' logprob={logprob:.4f}{marker}")

                if target_rank is not None:
                    print(f"\nTarget token rank in vLLM: {target_rank + 1}")
                else:
                    print(f"\nTarget token NOT in vLLM top-20!")

    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    diffs = megatron_logprobs[:len(vllm_logprobs)] - vllm_logprobs[:len(megatron_logprobs)]
    print(f"Total positions: {len(diffs)}")
    print(f"Positions with |diff| > 1.0: {np.sum(np.abs(diffs) > 1.0)}")
    print(f"Positions with |diff| > 0.1: {np.sum(np.abs(diffs) > 0.1)}")
    print(f"Mean diff: {np.mean(diffs):.4f}")
    print(f"Std diff: {np.std(diffs):.4f}")
    print(f"Max diff: {np.max(diffs):.4f} at pos {np.argmax(diffs)}")
    print(f"Min diff: {np.min(diffs):.4f} at pos {np.argmin(diffs)}")

    # Check if there's a pattern (e.g., all positions wrong vs specific positions)
    print("\n" + "="*70)
    print("PATTERN ANALYSIS")
    print("="*70)

    # Group by token type
    newline_positions = [i for i, t in enumerate(target_tokens) if tokenizer.decode([t]) == '\n']
    number_positions = [i for i, t in enumerate(target_tokens) if tokenizer.decode([t]).strip().isdigit()]
    other_positions = [i for i in range(len(target_tokens)) if i not in newline_positions and i not in number_positions]

    def analyze_group(name, positions):
        if not positions:
            return
        group_diffs = [diffs[i] for i in positions if i < len(diffs)]
        if group_diffs:
            print(f"\n{name} (n={len(group_diffs)}):")
            print(f"  Mean diff: {np.mean(group_diffs):.4f}")
            print(f"  Large diffs (>1.0): {sum(1 for d in group_diffs if abs(d) > 1.0)}")

    analyze_group("Newline tokens", newline_positions)
    analyze_group("Number tokens", number_positions)
    analyze_group("Other tokens", other_positions)


if __name__ == "__main__":
    asyncio.run(main())
