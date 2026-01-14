#!/usr/bin/env python3
"""Deep trace: Compare hidden states and LoRA outputs at position 7 vs 14.

The trained checkpoint degrades position 7 (-8.2 → -26.2) but improves position 14 (-0.35 → 0.0).
This script traces:
1. The raw logits at both positions
2. The argmax token at both positions
3. Specific logit values for target tokens vs top tokens

Usage:
    TINKER_BASE_URL=http://localhost:8000 python scripts/trace_position_7_computation.py
"""

import asyncio
import os
import sys

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker
import torch
import numpy as np

MODEL_NAME = "moonshotai/Moonlight-16B-A3B-Instruct"

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


async def get_logprobs_with_debug(client, datum):
    """Get logprobs and any debug info from forward pass."""
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    logprobs = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    # Check for debug info in metrics
    debug_info = None
    if hasattr(result, 'metrics') and result.metrics:
        debug_info = result.metrics.get('_debug_logits', None)

    return logprobs, result, debug_info


async def main():
    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    mask = [1.0] * len(input_tokens)

    print(f"Sequence: {len(input_tokens)} tokens")

    # Show detailed token info at positions 7 and 14
    analyze_positions = [7, 14]
    print("\n" + "=" * 70)
    print("TOKENS AT POSITIONS TO ANALYZE")
    print("=" * 70)

    for pos in analyze_positions:
        # Input token at position pos
        input_tok = input_tokens[pos]
        input_str = tokenizer.decode([input_tok])
        # Target token = what should be predicted at position pos
        target_tok = target_tokens[pos]
        target_str = tokenizer.decode([target_tok])
        # Context: input tokens 0 to pos
        context = tokenizer.decode(input_tokens[:pos+1])

        print(f"\nPosition {pos}:")
        print(f"  Input token[{pos}]: {input_tok} ({repr(input_str)})")
        print(f"  Target token[{pos}] (=input[{pos+1}]): {target_tok} ({repr(target_str)})")
        print(f"  Context: {repr(context[:80])}...")

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.ones(len(input_tokens), dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.zeros(len(input_tokens), dtype=torch.float32)),
        }
    )

    # ===============================================================
    # PHASE 1: Fresh LoRA - get logprobs and debug info
    # ===============================================================
    print("\n" + "=" * 70)
    print("PHASE 1: Fresh LoRA logprobs")
    print("=" * 70)

    client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)
    print(f"Model ID: {client.model_id}")

    fresh_logprobs, fresh_result, fresh_debug = await get_logprobs_with_debug(client, datum)

    for pos in analyze_positions:
        target_tok = target_tokens[pos]
        target_str = tokenizer.decode([target_tok])
        print(f"\nPosition {pos} (target={repr(target_str)}):")
        print(f"  logprob = {fresh_logprobs[pos]:.6f}")
        print(f"  prob = {np.exp(fresh_logprobs[pos]):.6f} ({100*np.exp(fresh_logprobs[pos]):.4f}%)")

    if fresh_debug:
        print(f"\nDebug logits info: {fresh_debug}")

    # ===============================================================
    # PHASE 2: Train 10 steps and observe progression
    # ===============================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Train 10 steps")
    print("=" * 70)

    step_data = []
    for step in range(10):
        # Forward-backward
        fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
        await fwd_bwd.result_async()

        # Optim step
        await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()

        # Get logprobs after this step
        lp, _, dbg = await get_logprobs_with_debug(client, datum)

        step_data.append({
            'step': step + 1,
            'pos7_lp': lp[7],
            'pos14_lp': lp[14],
            'debug': dbg
        })

        print(f"Step {step+1}: pos7={lp[7]:.4f}, pos14={lp[14]:.4f}")

    # ===============================================================
    # PHASE 3: Analyze the degradation pattern
    # ===============================================================
    print("\n" + "=" * 70)
    print("PHASE 3: Degradation Analysis")
    print("=" * 70)

    # Get final trained logprobs
    trained_logprobs, trained_result, trained_debug = await get_logprobs_with_debug(client, datum)

    print("\nFinal trained logprobs at key positions:")
    for pos in analyze_positions:
        target_tok = target_tokens[pos]
        target_str = tokenizer.decode([target_tok])
        delta = trained_logprobs[pos] - fresh_logprobs[pos]
        print(f"\nPosition {pos} (target={repr(target_str)}):")
        print(f"  Fresh logprob:   {fresh_logprobs[pos]:.6f}")
        print(f"  Trained logprob: {trained_logprobs[pos]:.6f}")
        print(f"  Delta:           {delta:+.6f}")
        print(f"  Fresh prob:      {100*np.exp(fresh_logprobs[pos]):.4f}%")
        print(f"  Trained prob:    {100*np.exp(trained_logprobs[pos]):.4f}%")

    if trained_debug:
        print(f"\nDebug logits info after training: {trained_debug}")

    # ===============================================================
    # PHASE 4: Export to vLLM and get TOP-K
    # ===============================================================
    print("\n" + "=" * 70)
    print("PHASE 4: Export to vLLM and get logprobs")
    print("=" * 70)

    sampling_client = await client.save_weights_and_get_sampling_client_async()
    print("Exported to vLLM")
    await asyncio.sleep(2)

    # Get vLLM logprobs
    sample_result = await sampling_client.sample_async(
        prompt=tinker.ModelInput.from_ints(input_tokens),
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=1,
            temperature=0.0,
        ),
        include_prompt_logprobs=True,
    )

    if sample_result.prompt_logprobs:
        vllm_logprobs = np.array([lp if lp is not None else -100.0 for lp in sample_result.prompt_logprobs])

        print("\nvLLM logprobs at key positions (note: vLLM[pos+1] aligns with Megatron[pos]):")
        for pos in analyze_positions:
            target_tok = target_tokens[pos]
            target_str = tokenizer.decode([target_tok])
            vllm_idx = pos + 1
            mega_lp = trained_logprobs[pos]
            vllm_lp = vllm_logprobs[vllm_idx] if vllm_idx < len(vllm_logprobs) else float('nan')
            diff = mega_lp - vllm_lp
            print(f"\nPosition {pos} (target={repr(target_str)}):")
            print(f"  Megatron:    {mega_lp:.6f}")
            print(f"  vLLM[{vllm_idx}]:   {vllm_lp:.6f}")
            print(f"  Difference:  {diff:+.6f}")

    # ===============================================================
    # PHASE 5: Summary
    # ===============================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nStep-by-step progression:")
    print(f"{'Step':<6} {'Pos 7':<12} {'Pos 14':<12}")
    print("-" * 30)
    print(f"{'Fresh':<6} {fresh_logprobs[7]:<12.4f} {fresh_logprobs[14]:<12.4f}")
    for sd in step_data:
        print(f"{sd['step']:<6} {sd['pos7_lp']:<12.4f} {sd['pos14_lp']:<12.4f}")

    print(f"\nPosition 7 degraded by {trained_logprobs[7] - fresh_logprobs[7]:.2f} nats")
    print(f"Position 14 improved by {trained_logprobs[14] - fresh_logprobs[14]:+.2f} nats")

    # The key question: WHY does position 7 degrade while position 14 improves?
    print("\n" + "=" * 70)
    print("HYPOTHESIS")
    print("=" * 70)
    print("""
Position 7 predicts 'Count' (token 3922) - a rare token in the training data.
Position 14 predicts '1' (token 16) - appears multiple times in sequence.

The shared LoRA learns to boost '1' at positions 14, 21, 49 (where it's correct).
But this same LoRA is also applied at position 7, where it INCORRECTLY boosts
tokens that are good for other positions but not for position 7.

This is the "cross-position interference" from shared LoRA architecture.
vLLM doesn't show this because per-expert LoRA isolates the computations.
""")


if __name__ == "__main__":
    asyncio.run(main())
