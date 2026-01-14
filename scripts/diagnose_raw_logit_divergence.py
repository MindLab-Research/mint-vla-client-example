#!/usr/bin/env python3
"""Diagnose raw logit divergence between Megatron and vLLM.

The issue: After training, Megatron's forward pass produces different argmax tokens
than vLLM at certain positions, despite using the same LoRA weights.

Example from diagnostic logs:
- Position 23, target='<' (token 27)
- Megatron: target_logit=3.52, max_logit=58.50 at token 795='10'
- vLLM: token 27 is argmax with logprob ~-0.004

This script:
1. Loads the debug data from latest test run
2. Trains fresh LoRA for 10 steps
3. Runs forward() to get Megatron logprobs (uses eval_mode)
4. Exports to vLLM
5. Compares raw logits at problematic positions

The hypothesis: Something in Megatron's forward (eval_mode) produces different
logits than training (train_mode) and vLLM inference.
"""

import asyncio
import json
import os
import sys

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker
import torch
import numpy as np

MODEL_NAME = "moonshotai/Moonlight-16B-A3B-Instruct"

# Key positions to investigate (from previous runs)
INVESTIGATE_POSITIONS = [5, 14, 21, 23]


def find_latest_debug_file():
    """Find the most recent debug JSON file."""
    import glob
    files = glob.glob("/tmp/megatron_vllm_debug_*.json")
    if not files:
        return None
    return max(files, key=os.path.getmtime)


async def main():
    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Load latest debug data
    debug_file = find_latest_debug_file()
    if debug_file:
        print(f"Found debug file: {debug_file}")
        with open(debug_file, "r") as f:
            data = json.load(f)
        input_tokens = data["input_tokens"]
        target_tokens = data["target_tokens"]
        print(f"Using existing data: {len(input_tokens)} input tokens")
    else:
        # Create new test data
        print("No debug file found, creating new test data...")
        TEST_TEXT = """<|im_start|>user
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
        tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]

    mask = [1.0] * len(input_tokens)

    print(f"\nToken sequence at investigate positions:")
    for pos in INVESTIGATE_POSITIONS:
        if pos < len(target_tokens):
            target_str = tokenizer.decode([target_tokens[pos]])
            context = tokenizer.decode(input_tokens[max(0, pos-2):pos+1])
            print(f"  pos {pos}: target={target_tokens[pos]:6d} '{repr(target_str):10s}', context='{repr(context)}'")

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    # Create Megatron training client
    print("\n" + "=" * 70)
    print("Creating Megatron training client...")
    print("=" * 70)
    client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.ones(len(input_tokens), dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.zeros(len(input_tokens), dtype=torch.float32)),
        }
    )

    # FRESH forward (baseline)
    print("\n" + "=" * 70)
    print("FRESH Megatron forward (before training)")
    print("=" * 70)
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    fresh_logprobs = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    print("Fresh logprobs at investigate positions:")
    for pos in INVESTIGATE_POSITIONS:
        if pos < len(fresh_logprobs):
            target_str = tokenizer.decode([target_tokens[pos]])
            print(f"  pos {pos}: {fresh_logprobs[pos]:8.4f} (target='{target_str}')")

    # Train 10 steps with detailed logging
    print("\n" + "=" * 70)
    print("Training 10 steps (detailed logging EVERY step)...")
    print("=" * 70)
    for step in range(10):
        fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
        result = await fwd_bwd.result_async()

        # Get logprobs from forward_backward result
        step_logprobs = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

        # Log ALL investigate positions EVERY step
        print(f"  Step {step+1} (before optim_step):")
        for pos in INVESTIGATE_POSITIONS:
            if pos < len(step_logprobs):
                target_str = tokenizer.decode([target_tokens[pos]])
                print(f"    pos {pos}: {step_logprobs[pos]:8.4f} (target='{target_str}')")

        await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()
        print(f"    (optim_step done)")

    # TRAINED forward (eval_mode)
    print("\n" + "=" * 70)
    print("TRAINED Megatron forward (after training, uses eval_mode)")
    print("Check /vePFS-Mindverse/share/code/raw_logit_diag.log for raw logits")
    print("=" * 70)
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    trained_logprobs = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    print("Trained logprobs at investigate positions:")
    for pos in INVESTIGATE_POSITIONS:
        if pos < len(trained_logprobs):
            target_str = tokenizer.decode([target_tokens[pos]])
            fresh_lp = fresh_logprobs[pos] if pos < len(fresh_logprobs) else float('nan')
            delta = trained_logprobs[pos] - fresh_lp
            print(f"  pos {pos}: {trained_logprobs[pos]:8.4f} (fresh={fresh_lp:.4f}, delta={delta:+.4f}, target='{target_str}')")

    # Export to vLLM
    print("\n" + "=" * 70)
    print("Exporting to vLLM...")
    print("=" * 70)
    sampling_client = await client.save_weights_and_get_sampling_client_async()
    await asyncio.sleep(2)

    # Get vLLM logprobs
    sample_result = await sampling_client.sample_async(
        prompt=tinker.ModelInput.from_ints(input_tokens),
        num_samples=1,
        sampling_params=tinker.SamplingParams(max_tokens=1, temperature=0.0),
        include_prompt_logprobs=True,
    )

    if sample_result.prompt_logprobs:
        vllm_logprobs = np.array([lp if lp is not None else -100.0 for lp in sample_result.prompt_logprobs])

        print("vLLM trained logprobs at investigate positions (aligned: vLLM[i+1] vs Megatron[i]):")
        for pos in INVESTIGATE_POSITIONS:
            vllm_idx = pos + 1  # Alignment: Megatron[i] predicts same token as vLLM[i+1]
            if vllm_idx < len(vllm_logprobs) and pos < len(trained_logprobs):
                target_str = tokenizer.decode([target_tokens[pos]])
                meg_lp = trained_logprobs[pos]
                vllm_lp = vllm_logprobs[vllm_idx]
                diff = meg_lp - vllm_lp
                print(f"  pos {pos}: Megatron={meg_lp:8.4f}, vLLM[{vllm_idx}]={vllm_lp:8.4f}, diff={diff:+8.4f}, target='{target_str}'")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("Key insight from raw logit diagnostics:")
    print("  - Look at /vePFS-Mindverse/share/code/raw_logit_diag.log")
    print("  - Compare target_logit vs local_max at problematic positions")
    print("  - The argmax token ID tells us what Megatron THINKS is most likely")


if __name__ == "__main__":
    asyncio.run(main())
