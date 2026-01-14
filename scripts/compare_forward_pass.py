#!/usr/bin/env python3
"""Compare forward pass outputs between Megatron and vLLM at tensor level.

This script:
1. Creates fresh Megatron LoRA
2. Exports to vLLM
3. Compares logits at all positions before and after training
4. Identifies exact positions/tokens where divergence occurs

Key insight: We need to compare the RAW LOGITS, not just final logprobs.
"""

import asyncio
import os
import json
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
    print(f"Tokens: {input_tokens}")
    print(f"Text: {[tokenizer.decode([t]) for t in input_tokens]}")

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
    # PHASE 1: Fresh LoRA - get baseline
    # ===============================================================
    print("\n" + "=" * 70)
    print("PHASE 1: Fresh LoRA baseline")
    print("=" * 70)

    client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)
    print(f"Model ID: {client.model_id}")

    # Get Megatron logprobs for all positions
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    fresh_mega_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    # Export to vLLM and get logprobs
    fresh_sampling_client = await client.save_weights_and_get_sampling_client_async()
    await asyncio.sleep(3)

    prompt = tinker.ModelInput.from_ints(tokens)
    fresh_vllm_lp = await fresh_sampling_client.compute_logprobs_async(prompt)

    print("\nFresh LoRA comparison (all positions):")
    print(f"{'Pos':<6} {'Token':<15} {'Megatron':<12} {'vLLM':<12} {'Diff':<12}")
    print("-" * 58)

    max_diff_fresh = 0
    for i in range(len(fresh_mega_lp)):
        m = fresh_mega_lp[i]
        # vLLM offset by 1
        v = fresh_vllm_lp[i + 1] if i + 1 < len(fresh_vllm_lp) and fresh_vllm_lp[i + 1] is not None else float('nan')
        diff = m - v if not np.isnan(v) else float('nan')
        max_diff_fresh = max(max_diff_fresh, abs(diff) if not np.isnan(diff) else 0)
        tok_str = repr(tokenizer.decode([target_tokens[i]]))[:12] if i < len(target_tokens) else "N/A"
        print(f"{i:<6} {tok_str:<15} {m:<12.4f} {v:<12.4f} {diff:<+12.4f}")

    print(f"\nMax diff (fresh): {max_diff_fresh:.4f}")

    # ===============================================================
    # PHASE 2: Train 1 step
    # ===============================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Train 1 step with lr=1e-3")
    print("=" * 70)

    fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
    await fwd_bwd.result_async()
    optim_result = await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()
    print(f"Optimizer step result: grad_norm={optim_result}")

    # Get trained Megatron logprobs
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    trained_mega_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    # Export trained weights to vLLM
    trained_sampling_client = await client.save_weights_and_get_sampling_client_async()
    await asyncio.sleep(3)

    trained_vllm_lp = await trained_sampling_client.compute_logprobs_async(prompt)

    print("\nTrained LoRA comparison (all positions):")
    print(f"{'Pos':<6} {'Token':<15} {'M-fresh':<10} {'M-train':<10} {'V-train':<10} {'M-V diff':<10} {'M delta':<10}")
    print("-" * 86)

    max_diff_trained = 0
    divergent_positions = []

    for i in range(len(trained_mega_lp)):
        m_f = fresh_mega_lp[i]
        m_t = trained_mega_lp[i]
        v_t = trained_vllm_lp[i + 1] if i + 1 < len(trained_vllm_lp) and trained_vllm_lp[i + 1] is not None else float('nan')
        diff = m_t - v_t if not np.isnan(v_t) else float('nan')
        m_delta = m_t - m_f

        max_diff_trained = max(max_diff_trained, abs(diff) if not np.isnan(diff) else 0)
        if abs(diff) > 5:  # significant divergence
            divergent_positions.append((i, diff, m_delta))

        tok_str = repr(tokenizer.decode([target_tokens[i]]))[:12] if i < len(target_tokens) else "N/A"
        flag = " <--" if abs(diff) > 5 else ""
        print(f"{i:<6} {tok_str:<15} {m_f:<10.2f} {m_t:<10.2f} {v_t:<10.2f} {diff:<+10.2f} {m_delta:<+10.2f}{flag}")

    print(f"\nMax diff (trained): {max_diff_trained:.4f}")

    # ===============================================================
    # PHASE 3: Analysis
    # ===============================================================
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    if divergent_positions:
        print(f"\nDivergent positions (diff > 5 nats): {len(divergent_positions)}")
        for pos, diff, m_delta in divergent_positions:
            tok_str = tokenizer.decode([target_tokens[pos]]) if pos < len(target_tokens) else "N/A"
            print(f"  pos {pos}: diff={diff:+.2f}, M_delta={m_delta:+.2f}, token={repr(tok_str)}")

        # Check pattern
        print("\nPattern analysis:")
        m_deltas = [m_delta for _, _, m_delta in divergent_positions]
        print(f"  Megatron deltas: mean={np.mean(m_deltas):.2f}, std={np.std(m_deltas):.2f}")

        # Key question: Does training DEGRADE Megatron while IMPROVING vLLM?
        megatron_degrades = sum(1 for _, diff, m_delta in divergent_positions if m_delta < -5)
        vllm_improves = sum(1 for _, diff, m_delta in divergent_positions if diff < -5)  # vLLM better than Megatron

        print(f"  Positions where Megatron degrades (delta < -5): {megatron_degrades}")
        print(f"  Positions where vLLM is better (diff < -5): {vllm_improves}")
    else:
        print("\nNo significant divergence found!")

    # Save checkpoint for further analysis
    checkpoint_name = f"forward_pass_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_result = await (await client.save_state_async(name=checkpoint_name)).result_async()
    print(f"\nCheckpoint saved: {save_result.path}")


if __name__ == "__main__":
    asyncio.run(main())
