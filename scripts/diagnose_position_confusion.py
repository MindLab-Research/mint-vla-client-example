#!/usr/bin/env python3
"""Diagnose position confusion between Megatron and vLLM.

Key hypothesis: Position 7 in Megatron sees different hidden states than position 8 in vLLM.

Test plan:
1. Get top-K logits from Megatron at position 7 after training
2. Get top-K logits from vLLM at position 8 after training
3. Compare: Are they predicting the same tokens?
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


async def main():
    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    mask = [1.0] * len(input_tokens)

    print(f"Sequence: {len(tokens)} tokens, input: {len(input_tokens)}")

    # Show key positions
    for pos in [6, 7, 8, 9, 14, 31, 49]:
        if pos < len(input_tokens):
            inp = tokenizer.decode([input_tokens[pos]])
            tgt = tokenizer.decode([target_tokens[pos]])
            print(f"  pos={pos:2d}: input={repr(inp):10s} -> target={repr(tgt):12s} (id={target_tokens[pos]})")

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    print("\n" + "=" * 70)
    print("PHASE 1: Create fresh LoRA and get baseline")
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

    # Get fresh baseline
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    fresh_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    print(f"\nFresh Megatron logprobs (pos 7, 8, 31, 49):")
    for pos in [7, 8, 31, 49]:
        if pos < len(fresh_lp):
            tgt = target_tokens[pos]
            tgt_str = tokenizer.decode([tgt])
            print(f"  pos={pos:2d}: lp={fresh_lp[pos]:.4f} (target={tgt}={repr(tgt_str)})")

    print("\n" + "=" * 70)
    print("PHASE 2: Train 10 steps")
    print("=" * 70)

    for step in range(10):
        fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
        await fwd_bwd.result_async()
        await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()

        if (step + 1) % 3 == 0 or step == 9:
            fwd = await client.forward_async([datum], loss_fn="importance_sampling")
            result = await fwd.result_async()
            train_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())
            print(f"  Step {step+1}: pos7={train_lp[7]:.4f}, pos8={train_lp[8]:.4f}, pos31={train_lp[31]:.4f}")

    # Final Megatron logprobs
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    mega_trained_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    print("\n" + "=" * 70)
    print("PHASE 3: Export to vLLM and compare")
    print("=" * 70)

    sampling_client = await client.save_weights_and_get_sampling_client_async()

    # vLLM expects full sequence
    prompt = tinker.ModelInput.from_ints(tokens)
    vllm_lp = await sampling_client.compute_logprobs_async(prompt)

    print("\n" + "=" * 100)
    print("COMPARISON: Megatron vs vLLM (aligned: Mega[i] vs vLLM[i+1])")
    print("=" * 100)
    print(f"{'pos':>4} {'target':>12} {'Mega Fresh':>12} {'Mega Train':>12} {'vLLM[i+1]':>12} {'M-V diff':>10}")
    print("-" * 70)

    for pos in [6, 7, 8, 9, 14, 31, 33, 49]:
        if pos < len(mega_trained_lp):
            vllm_pos = pos + 1
            tgt = target_tokens[pos]
            tgt_str = tokenizer.decode([tgt])[:8]
            mega_fresh_val = fresh_lp[pos]
            mega_train_val = mega_trained_lp[pos]
            vllm_val = vllm_lp[vllm_pos] if vllm_pos < len(vllm_lp) and vllm_lp[vllm_pos] is not None else float('nan')
            diff = mega_train_val - vllm_val if not np.isnan(vllm_val) else float('nan')

            flag = ""
            if abs(diff) > 5:
                flag = " ***"
            elif mega_train_val < -10 and vllm_val > -5:
                flag = " !!! MISMATCH"

            print(f"{pos:4d} {tgt_str:>12s} {mega_fresh_val:12.4f} {mega_train_val:12.4f} {vllm_val:12.4f} {diff:+10.4f}{flag}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Find positions where Megatron gets worse but vLLM improves
    bad_positions = []
    for pos in range(min(len(mega_trained_lp), len(vllm_lp) - 1)):
        mega_delta = mega_trained_lp[pos] - fresh_lp[pos]
        vllm_fresh_pos = pos + 1
        if vllm_fresh_pos < len(vllm_lp) and vllm_lp[vllm_fresh_pos] is not None:
            # Megatron got worse (more negative) but shouldn't have
            if mega_delta < -5:
                bad_positions.append((pos, mega_delta, vllm_lp[vllm_fresh_pos]))

    if bad_positions:
        print(f"\nPositions where Megatron degraded > 5 nats:")
        for pos, mega_delta, vllm_val in bad_positions[:10]:
            tgt = target_tokens[pos]
            tgt_str = tokenizer.decode([tgt])
            print(f"  pos={pos}: mega_delta={mega_delta:+.2f}, vllm_trained={vllm_val:.2f}, target={repr(tgt_str)}")


if __name__ == "__main__":
    asyncio.run(main())
