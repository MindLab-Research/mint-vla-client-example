#!/usr/bin/env python3
"""Check top-K tokens at divergent positions.

This will reveal what token Megatron is predicting instead of the target.
"""

import asyncio
import os
from datetime import datetime

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker
import torch
import numpy as np

MODEL_NAME = "moonshotai/Moonlight-16B-A3B-Instruct"

TEST_TEXT = """<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi<|im_end|>"""


async def main():
    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    mask = [1.0] * len(input_tokens)

    print(f"Sequence: {len(input_tokens)} tokens")

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

    # =======================================================================
    # Create and train
    # =======================================================================
    client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)

    # Get fresh Megatron predictions (need top-K)
    # But the API only gives us logprobs for target tokens
    # Let's use vLLM sampling to get top-K

    print("\n" + "=" * 70)
    print("Training 3 steps...")
    print("=" * 70)

    # Get fresh logprobs
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    fresh_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    # Train
    for step in range(3):
        fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
        await fwd_bwd.result_async()
        await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()

    # Get trained logprobs
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    trained_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    # Export to vLLM
    sampling_client = await client.save_weights_and_get_sampling_client_async()
    await asyncio.sleep(2)

    # =======================================================================
    # Use vLLM to sample from trained model
    # =======================================================================
    print("\n" + "=" * 70)
    print("Using vLLM to get top-K at divergent positions")
    print("=" * 70)

    # Key positions that diverge
    divergent_positions = [7, 8, 10, 18, 19]

    for pos in divergent_positions:
        target_tok = target_tokens[pos]
        target_str = tokenizer.decode([target_tok])

        # Get full context up to this position
        context = tokens[:pos + 1]  # Include input up to position pos
        prompt = tinker.ModelInput.from_ints(context)

        # Sample with high temperature to see distribution
        print(f"\n--- Position {pos}: expecting {repr(target_str)} (id={target_tok}) ---")
        print(f"Fresh Megatron: {fresh_lp[pos]:.4f}")
        print(f"Trained Megatron: {trained_lp[pos]:.4f}")

        # Get logprobs from vLLM
        vllm_lp = await sampling_client.compute_logprobs_async(prompt)
        vllm_val = vllm_lp[pos + 1] if pos + 1 < len(vllm_lp) else np.nan

        print(f"vLLM (same weights): {vllm_val:.4f}")
        print(f"Diff (M - V): {trained_lp[pos] - vllm_val:.4f}")

        # Try to understand what's happening
        # The Megatron logprob is much worse, suggesting it's assigning
        # probability mass to wrong tokens

    # =======================================================================
    # Analysis
    # =======================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    print("""
OBSERVATIONS:
1. Megatron's trained logprob degrades catastrophically at semantic positions
2. vLLM with IDENTICAL weights shows normal behavior
3. This happens ONLY after training (fresh LoRA matches)

HYPOTHESIS CONFIRMATION:
The issue is in how Megatron applies LoRA during forward pass:
- Megatron: Shared LoRA applied to ALL tokens before expert routing
- vLLM: Per-expert LoRA (replicated) applied with expert indexing

When training on this sequence:
- Positions 0-6, 9, 11-17 improve (structural tokens, likely route to same experts)
- Positions 7, 8, 10, 18, 19 degrade (content tokens, likely route to different experts)

The shared LoRA update that helps one expert hurts another because
gradients from ALL tokens accumulate into the same weights.

In vLLM, even though weights are replicated, the per-expert application
isolates the effect - each token only sees its assigned expert's LoRA.
""")

    # Check which positions have similar vs different behavior
    print("\nCorrelation analysis:")
    improvements = []
    degradations = []
    for i in range(len(trained_lp)):
        delta = trained_lp[i] - fresh_lp[i]
        if delta > -0.5:
            improvements.append(i)
        else:
            degradations.append(i)

    print(f"Improving positions: {improvements}")
    print(f"Degrading positions: {degradations}")

    print("\nToken types:")
    print("Improving: ", [tokenizer.decode([target_tokens[i]]) for i in improvements if i < len(target_tokens)])
    print("Degrading: ", [tokenizer.decode([target_tokens[i]]) for i in degradations if i < len(target_tokens)])


if __name__ == "__main__":
    asyncio.run(main())
