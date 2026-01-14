#!/usr/bin/env python3
"""Trace the exact difference between Megatron and vLLM LoRA application for MoE.

Key question: After training, why does Megatron predict token 16 at position 7
while vLLM correctly predicts token 3922?

Hypothesis: The shared LoRA in Megatron is applied uniformly to all tokens,
while vLLM per-expert LoRA separates the effect.
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

    print(f"Sequence: {len(tokens)} tokens")

    # Key positions
    pos7_target = target_tokens[7]  # Should be 3922 ('Count')
    pos14_target = target_tokens[14]  # Should be 16 ('1')
    pos49_target = target_tokens[49]  # Should be 16 ('1')

    print(f"\nKey positions:")
    print(f"  pos 7: target={pos7_target} ({repr(tokenizer.decode([pos7_target]))})")
    print(f"  pos 14: target={pos14_target} ({repr(tokenizer.decode([pos14_target]))})")
    print(f"  pos 49: target={pos49_target} ({repr(tokenizer.decode([pos49_target]))})")

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    print("\n" + "=" * 70)
    print("PHASE 1: Create LoRA and get FRESH baseline")
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

    # Get fresh Megatron logprobs
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    fresh_mega_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    # Get fresh vLLM logprobs (same checkpoint = base model)
    sampling_client = await client.save_weights_and_get_sampling_client_async()
    prompt = tinker.ModelInput.from_ints(tokens)
    fresh_vllm_lp = await sampling_client.compute_logprobs_async(prompt)

    print("\nFresh baseline comparison (Megatron[i] vs vLLM[i+1]):")
    for pos in [7, 14, 49]:
        mega_lp = fresh_mega_lp[pos]
        vllm_lp = fresh_vllm_lp[pos + 1] if pos + 1 < len(fresh_vllm_lp) and fresh_vllm_lp[pos + 1] is not None else float('nan')
        diff = mega_lp - vllm_lp
        tgt = target_tokens[pos]
        tgt_str = tokenizer.decode([tgt])
        print(f"  pos {pos}: Mega={mega_lp:.4f}, vLLM={vllm_lp:.4f}, diff={diff:+.4f} (target={repr(tgt_str)})")

    print("\n" + "=" * 70)
    print("PHASE 2: Train 10 steps")
    print("=" * 70)

    for step in range(10):
        fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
        await fwd_bwd.result_async()
        await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()

        if (step + 1) % 5 == 0:
            fwd = await client.forward_async([datum], loss_fn="importance_sampling")
            result = await fwd.result_async()
            train_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())
            print(f"  Step {step+1}: pos7={train_lp[7]:.4f}, pos14={train_lp[14]:.4f}, pos49={train_lp[49]:.4f}")

    print("\n" + "=" * 70)
    print("PHASE 3: Compare TRAINED Megatron vs vLLM")
    print("=" * 70)

    # Get trained Megatron logprobs
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    trained_mega_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    # Export and get trained vLLM logprobs
    sampling_client = await client.save_weights_and_get_sampling_client_async()
    trained_vllm_lp = await sampling_client.compute_logprobs_async(prompt)

    print("\nTrained comparison (Megatron[i] vs vLLM[i+1]):")
    print(f"{'pos':>4} {'target':>10} {'Mega Fresh':>12} {'Mega Train':>12} {'vLLM Train':>12} {'M-V diff':>10}")
    print("-" * 70)

    for pos in [7, 8, 9, 14, 31, 49]:
        tgt = target_tokens[pos]
        tgt_str = tokenizer.decode([tgt])[:8]
        mega_fresh = fresh_mega_lp[pos]
        mega_train = trained_mega_lp[pos]
        vllm_pos = pos + 1
        vllm_train = trained_vllm_lp[vllm_pos] if vllm_pos < len(trained_vllm_lp) and trained_vllm_lp[vllm_pos] is not None else float('nan')
        diff = mega_train - vllm_train

        flag = ""
        if abs(diff) > 5:
            flag = " <-- MISMATCH"

        print(f"{pos:4d} {tgt_str:>10s} {mega_fresh:12.4f} {mega_train:12.4f} {vllm_train:12.4f} {diff:+10.4f}{flag}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS: Why position 7 differs")
    print("=" * 70)

    mega_delta_7 = trained_mega_lp[7] - fresh_mega_lp[7]
    mega_delta_14 = trained_mega_lp[14] - fresh_mega_lp[14]
    mega_delta_49 = trained_mega_lp[49] - fresh_mega_lp[49]

    print(f"\nMegatron logprob changes:")
    print(f"  pos 7 (target='Count'): {mega_delta_7:+.4f}  {'DEGRADED' if mega_delta_7 < -1 else 'OK'}")
    print(f"  pos 14 (target='1'): {mega_delta_14:+.4f}  {'IMPROVED' if mega_delta_14 > 1 else 'OK'}")
    print(f"  pos 49 (target='1'): {mega_delta_49:+.4f}  {'IMPROVED' if mega_delta_49 > 1 else 'OK'}")

    if mega_delta_7 < -5 and mega_delta_14 > 1:
        print("\n** CROSS-POSITION INTERFERENCE DETECTED **")
        print("The shared LoRA learned to boost token 16 (for positions 14, 49)")
        print("but this bleeds into position 7, degrading predictions there.")

    vllm_delta_7 = trained_vllm_lp[8] - fresh_vllm_lp[8] if 8 < len(trained_vllm_lp) else float('nan')
    print(f"\nvLLM logprob change at pos 7 (vLLM[8]): {vllm_delta_7:+.4f}")

    if not np.isnan(vllm_delta_7) and vllm_delta_7 > -1:
        print("\nvLLM did NOT degrade at position 7.")
        print("This suggests the per-expert LoRA in vLLM prevents cross-position interference.")


if __name__ == "__main__":
    asyncio.run(main())
