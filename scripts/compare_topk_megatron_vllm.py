#!/usr/bin/env python3
"""Compare top-K logits between Megatron and vLLM at problematic positions.

Key question: Do Megatron and vLLM produce the same TOP tokens, just with different probabilities?
Or are the top tokens completely different?

If top tokens are the same: the issue is in scaling/normalization
If top tokens are different: the issue is in how LoRA modifies the logits

Usage:
    TINKER_BASE_URL=http://localhost:8000 python scripts/compare_topk_megatron_vllm.py
"""

import asyncio
import os
import sys
import json
from datetime import datetime

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


async def get_vllm_topk(sampling_client, input_tokens, k=10):
    """Get top-K logits from vLLM at each position."""
    result = await sampling_client.sample_async(
        prompt=tinker.ModelInput.from_ints(input_tokens),
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=1,
            temperature=1.0,  # Need temperature > 0 for logprobs
            logprobs=k,  # Get top-K logprobs
        ),
        include_prompt_logprobs=True,
    )
    return result


async def main():
    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    mask = [1.0] * len(input_tokens)

    print(f"Sequence: {len(input_tokens)} tokens")

    # Positions to analyze
    analyze_positions = [7, 14, 23]

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
    # PHASE 1: Fresh LoRA baseline
    # ===============================================================
    print("\n" + "=" * 70)
    print("PHASE 1: Create fresh LoRA and get baseline")
    print("=" * 70)

    client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)
    print(f"Model ID: {client.model_id}")

    # Get fresh Megatron logprobs
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    fresh_mega_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    # Get fresh vLLM logprobs
    fresh_sampling_client = await client.save_weights_and_get_sampling_client_async()
    await asyncio.sleep(2)
    fresh_vllm_result = await get_vllm_topk(fresh_sampling_client, input_tokens)

    print("\nFresh LoRA comparison at key positions:")
    for pos in analyze_positions:
        target_tok = target_tokens[pos]
        target_str = tokenizer.decode([target_tok])
        mega_lp = fresh_mega_lp[pos]
        vllm_idx = pos + 1  # Alignment offset
        vllm_lp = fresh_vllm_result.prompt_logprobs[vllm_idx] if vllm_idx < len(fresh_vllm_result.prompt_logprobs) else float('nan')
        diff = mega_lp - vllm_lp if vllm_lp is not None else float('nan')
        print(f"  pos {pos}: M={mega_lp:.4f}, V={vllm_lp:.4f}, diff={diff:+.4f}, target={repr(target_str)}")

    # ===============================================================
    # PHASE 2: Train 3 steps
    # ===============================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Train 3 steps")
    print("=" * 70)

    for step in range(3):
        fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
        await fwd_bwd.result_async()
        await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()

        # Get logprobs after this step
        fwd = await client.forward_async([datum], loss_fn="importance_sampling")
        result = await fwd.result_async()
        mega_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())
        print(f"Step {step+1}: pos7={mega_lp[7]:.4f}, pos14={mega_lp[14]:.4f}, pos23={mega_lp[23]:.4f}")

    # ===============================================================
    # PHASE 3: Export and compare top-K
    # ===============================================================
    print("\n" + "=" * 70)
    print("PHASE 3: Export trained weights and compare top-K")
    print("=" * 70)

    # Get trained Megatron logprobs
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    trained_mega_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    # Export to vLLM
    trained_sampling_client = await client.save_weights_and_get_sampling_client_async()
    await asyncio.sleep(2)

    # Get vLLM logprobs with top-K
    trained_vllm_result = await get_vllm_topk(trained_sampling_client, input_tokens, k=10)

    print("\nTrained comparison at key positions:")
    for pos in analyze_positions:
        target_tok = target_tokens[pos]
        target_str = tokenizer.decode([target_tok])
        mega_lp = trained_mega_lp[pos]
        vllm_idx = pos + 1
        vllm_lp = trained_vllm_result.prompt_logprobs[vllm_idx] if vllm_idx < len(trained_vllm_result.prompt_logprobs) else float('nan')
        diff = mega_lp - vllm_lp if vllm_lp is not None else float('nan')
        flag = " <-- MISMATCH" if abs(diff) > 5 else ""
        print(f"  pos {pos}: M={mega_lp:.4f}, V={vllm_lp:.4f}, diff={diff:+.4f}, target={repr(target_str)}{flag}")

    # ===============================================================
    # PHASE 4: Analyze top-K tokens
    # ===============================================================
    print("\n" + "=" * 70)
    print("PHASE 4: Analyze top-K tokens at position 7")
    print("=" * 70)

    # For detailed top-K analysis, we need to use a lower-level API
    # Let's use compute_logprobs which gives us all token logprobs

    # Get vLLM logprobs for all tokens
    prompt = tinker.ModelInput.from_ints(tokens)
    all_lp = await trained_sampling_client.compute_logprobs_async(prompt)

    print(f"\nvLLM prompt_logprobs at position 8 (predicting position 7's target):")
    if trained_vllm_result.prompt_logprobs_full:
        topk_at_8 = trained_vllm_result.prompt_logprobs_full.get(8, {})
        if topk_at_8:
            for tok_id, lp in sorted(topk_at_8.items(), key=lambda x: -x[1])[:10]:
                tok_str = tokenizer.decode([int(tok_id)])
                print(f"  token {tok_id:6d} ({repr(tok_str):10s}): {lp:.4f}")

    # Save checkpoint path for further analysis
    checkpoint_result = await (await client.save_state_async(
        path=f"/root/tinker_project/tinker-server/checkpoints/topk_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )).result_async()
    print(f"\nCheckpoint saved for analysis: {checkpoint_result}")

    # ===============================================================
    # SUMMARY
    # ===============================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nDelta from fresh to trained:")
    for pos in analyze_positions:
        fresh_m = fresh_mega_lp[pos]
        trained_m = trained_mega_lp[pos]
        delta_m = trained_m - fresh_m

        vllm_idx = pos + 1
        fresh_v = fresh_vllm_result.prompt_logprobs[vllm_idx] if vllm_idx < len(fresh_vllm_result.prompt_logprobs) else float('nan')
        trained_v = trained_vllm_result.prompt_logprobs[vllm_idx] if vllm_idx < len(trained_vllm_result.prompt_logprobs) else float('nan')
        delta_v = trained_v - fresh_v if not np.isnan(trained_v) else float('nan')

        print(f"  pos {pos}: Megatron {delta_m:+.2f}, vLLM {delta_v:+.2f}")

    print("""
KEY QUESTION: Why does training DEGRADE Megatron's logprob at position 7
while IMPROVING vLLM's logprob with the SAME weights?

If weights are identical and computation is identical, outputs must be identical.
Therefore, either:
1. The weights are NOT identical (export bug)
2. The computation is NOT identical (architectural difference)
3. There's hidden state affecting Megatron (not exported)
""")


if __name__ == "__main__":
    asyncio.run(main())
