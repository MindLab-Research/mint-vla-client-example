#!/usr/bin/env python3
"""Check if routing weight scaling causes the Megatron vs vLLM mismatch.

Hypothesis:
- Megatron: LoRA_delta added to MoE output WITHOUT routing weight scaling
- vLLM: LoRA_delta multiplied BY routing weights before adding

If routing weights don't sum to 1, this creates a scaling mismatch.

For DeepSeekV3/Moonlight MoE:
- Routing uses softmax over selected experts (not all experts)
- If top_k=6 and each token selects 6 experts, weights should sum to 1
- But if there's normalization differences, weights might not sum to 1

Usage:
    TINKER_BASE_URL=http://localhost:8000 python scripts/check_routing_weight_scaling.py
"""

import asyncio
import os

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker
import torch
import numpy as np

MODEL_NAME = "moonshotai/Moonlight-16B-A3B-Instruct"

# Simple test sequence
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
    print(f"Tokens: {input_tokens}")

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
    # PHASE 1: Fresh LoRA - both should match
    # ===============================================================
    print("\n" + "=" * 70)
    print("PHASE 1: Fresh LoRA comparison")
    print("=" * 70)

    client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)
    print(f"Model ID: {client.model_id}")

    # Get Megatron logprobs
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    fresh_mega_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    # Get vLLM logprobs
    sampling_client = await client.save_weights_and_get_sampling_client_async()
    await asyncio.sleep(2)

    prompt = tinker.ModelInput.from_ints(tokens)
    fresh_vllm_lp = await sampling_client.compute_logprobs_async(prompt)

    print("\nFresh LoRA logprobs comparison:")
    print(f"{'Pos':<6} {'Megatron':<12} {'vLLM':<12} {'Diff':<12} {'Token':<20}")
    print("-" * 62)
    for i in range(len(fresh_mega_lp)):
        m = fresh_mega_lp[i]
        v = fresh_vllm_lp[i + 1] if i + 1 < len(fresh_vllm_lp) and fresh_vllm_lp[i + 1] is not None else float('nan')
        diff = m - v if not np.isnan(v) else float('nan')
        tok_str = tokenizer.decode([target_tokens[i]]) if i < len(target_tokens) else "N/A"
        print(f"{i:<6} {m:<12.4f} {v:<12.4f} {diff:<+12.4f} {repr(tok_str):<20}")

    # ===============================================================
    # PHASE 2: Train 1 step - check scaling
    # ===============================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Train 1 step")
    print("=" * 70)

    fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
    await fwd_bwd.result_async()
    await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()

    # Get Megatron logprobs after training
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    trained_mega_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    # Export to vLLM
    trained_sampling_client = await client.save_weights_and_get_sampling_client_async()
    await asyncio.sleep(2)

    trained_vllm_lp = await trained_sampling_client.compute_logprobs_async(prompt)

    print("\nTrained LoRA logprobs comparison:")
    print(f"{'Pos':<6} {'Megatron':<12} {'vLLM':<12} {'Diff':<12} {'M-delta':<12} {'V-delta':<12}")
    print("-" * 74)
    for i in range(len(trained_mega_lp)):
        m = trained_mega_lp[i]
        v = trained_vllm_lp[i + 1] if i + 1 < len(trained_vllm_lp) and trained_vllm_lp[i + 1] is not None else float('nan')
        diff = m - v if not np.isnan(v) else float('nan')
        m_delta = m - fresh_mega_lp[i]
        v_delta = (v - fresh_vllm_lp[i + 1]) if not np.isnan(v) and fresh_vllm_lp[i + 1] is not None else float('nan')
        print(f"{i:<6} {m:<12.4f} {v:<12.4f} {diff:<+12.4f} {m_delta:<+12.4f} {v_delta:<+12.4f}")

    # ===============================================================
    # PHASE 3: Analysis
    # ===============================================================
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # If the difference is consistent across positions, it's a global scaling issue
    # If the difference varies by position, it's related to expert routing

    diffs = []
    for i in range(len(trained_mega_lp)):
        v = trained_vllm_lp[i + 1] if i + 1 < len(trained_vllm_lp) and trained_vllm_lp[i + 1] is not None else None
        if v is not None:
            diffs.append(trained_mega_lp[i] - v)

    diffs = np.array(diffs)
    print(f"\nMegatron - vLLM difference statistics:")
    print(f"  Mean: {np.mean(diffs):.4f}")
    print(f"  Std:  {np.std(diffs):.4f}")
    print(f"  Min:  {np.min(diffs):.4f}")
    print(f"  Max:  {np.max(diffs):.4f}")

    if np.std(diffs) < 0.1 and abs(np.mean(diffs)) > 0.5:
        print("\n** HYPOTHESIS: Constant scaling difference (routing weights issue) **")
    elif np.std(diffs) > 1.0:
        print("\n** HYPOTHESIS: Position-dependent difference (expert routing issue) **")
    else:
        print("\n** HYPOTHESIS: Small/negligible difference **")

    # Check if differences correlate with position in MoE vs dense layers
    print("\n" + "=" * 70)
    print("MoE vs Dense layer analysis")
    print("=" * 70)
    print("""
Moonlight architecture:
- Layer 0: Dense MLP (no expert routing)
- Layers 1-27: MoE layers (expert routing)

If the issue is in MoE LoRA, positions processed by later layers
(which go through more MoE layers) should show larger differences.
    """)


if __name__ == "__main__":
    asyncio.run(main())
