#!/usr/bin/env python3
"""Trace MoE forward pass to understand divergence.

Key hypothesis to test:
1. Megatron: shared LoRA applied after token permutation, single LoRA for all experts
2. vLLM: per-expert LoRA with expert indexing

This script:
1. Trains 1 step to get divergent weights
2. Gets detailed output at divergent position from both systems
3. Compares top-K predictions to understand the difference

Focus: position 7 (known divergent position from previous experiments)
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


async def get_detailed_logprobs(service_client, tokenizer, tokens, train_steps=0, lr=1e-3):
    """Get logprobs with detailed info."""
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    mask = [1.0] * len(input_tokens)

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.ones(len(input_tokens), dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.zeros(len(input_tokens), dtype=torch.float32)),
        }
    )

    # Create fresh LoRA
    client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)

    # Get fresh Megatron logprobs
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    fresh_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    # Train if requested
    if train_steps > 0:
        for _ in range(train_steps):
            fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
            await fwd_bwd.result_async()
            await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=lr))).result_async()

    # Get trained Megatron logprobs
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    trained_mega_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    # Export to vLLM and get logprobs
    sampling_client = await client.save_weights_and_get_sampling_client_async()
    await asyncio.sleep(2)

    prompt = tinker.ModelInput.from_ints(tokens)
    vllm_lp = await sampling_client.compute_logprobs_async(prompt)

    return {
        'client': client,
        'sampling_client': sampling_client,
        'fresh_lp': fresh_lp,
        'trained_mega_lp': trained_mega_lp,
        'vllm_lp': vllm_lp,
        'target_tokens': target_tokens,
    }


async def main():
    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]

    print(f"Sequence: {len(input_tokens)} tokens")
    print(f"Tokens: {input_tokens}")
    print(f"Text: {[tokenizer.decode([t]) for t in input_tokens]}")

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    # =======================================================================
    # PHASE 1: Get detailed comparison for all positions
    # =======================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: Train 1 step and compare all positions")
    print("=" * 80)

    result = await get_detailed_logprobs(service_client, tokenizer, tokens, train_steps=1, lr=1e-3)

    # Print detailed comparison
    print(f"\n{'Pos':<5} {'Token':<12} {'Fresh':<10} {'M-Train':<10} {'V-Train':<10} {'M-V Diff':<10} {'M Delta':<10}")
    print("-" * 80)

    divergent_positions = []
    for i in range(len(result['trained_mega_lp'])):
        f = result['fresh_lp'][i]
        m = result['trained_mega_lp'][i]
        v = result['vllm_lp'][i + 1] if i + 1 < len(result['vllm_lp']) and result['vllm_lp'][i + 1] is not None else np.nan

        m_v_diff = m - v if not np.isnan(v) else np.nan
        m_delta = m - f

        tok = tokenizer.decode([result['target_tokens'][i]]) if i < len(result['target_tokens']) else "?"
        flag = ""
        if abs(m_v_diff) > 5:
            flag = " <-- DIVERGENT"
            divergent_positions.append(i)

        print(f"{i:<5} {repr(tok):<12} {f:<10.2f} {m:<10.2f} {v:<10.2f} {m_v_diff:<+10.2f} {m_delta:<+10.2f}{flag}")

    # =======================================================================
    # PHASE 2: Analyze divergent positions
    # =======================================================================
    if divergent_positions:
        print("\n" + "=" * 80)
        print(f"PHASE 2: Analyzing {len(divergent_positions)} divergent positions")
        print("=" * 80)

        for pos in divergent_positions[:3]:  # Analyze first 3
            tok_id = result['target_tokens'][pos]
            tok = tokenizer.decode([tok_id])

            print(f"\n--- Position {pos}: predicting {repr(tok)} (token_id={tok_id}) ---")
            print(f"Fresh logprob:    {result['fresh_lp'][pos]:.4f}")
            print(f"Megatron trained: {result['trained_mega_lp'][pos]:.4f}")
            print(f"vLLM trained:     {result['vllm_lp'][pos + 1]:.4f}")

            # The key insight: same weights, different results
            # What could cause this?
            # 1. Different token routing in MoE
            # 2. Different LoRA application order
            # 3. Different numerical precision

    # =======================================================================
    # PHASE 3: Test with more training steps
    # =======================================================================
    print("\n" + "=" * 80)
    print("PHASE 3: Training 5 steps to see divergence trend")
    print("=" * 80)

    result5 = await get_detailed_logprobs(service_client, tokenizer, tokens, train_steps=5, lr=1e-3)

    print(f"\n{'Pos':<5} {'Token':<12} {'Fresh':<10} {'M-Train':<10} {'V-Train':<10} {'M-V Diff':<10}")
    print("-" * 70)

    for i in range(len(result5['trained_mega_lp'])):
        m = result5['trained_mega_lp'][i]
        v = result5['vllm_lp'][i + 1] if i + 1 < len(result5['vllm_lp']) and result5['vllm_lp'][i + 1] is not None else np.nan
        f = result5['fresh_lp'][i]

        m_v_diff = m - v if not np.isnan(v) else np.nan

        tok = tokenizer.decode([result5['target_tokens'][i]]) if i < len(result5['target_tokens']) else "?"

        print(f"{i:<5} {repr(tok):<12} {f:<10.2f} {m:<10.2f} {v:<10.2f} {m_v_diff:<+10.2f}")

    # =======================================================================
    # SUMMARY
    # =======================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("""
KEY OBSERVATIONS TO MAKE:
1. Which positions diverge consistently?
2. Does divergence increase with more training?
3. Are the divergent tokens semantically related?

HYPOTHESES TO TEST:
A. Expert routing differs between systems
B. LoRA is applied at different points in the computation
C. Token permutation affects the shared vs per-expert LoRA differently
""")


if __name__ == "__main__":
    asyncio.run(main())
