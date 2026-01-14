#!/usr/bin/env python3
"""Compare Megatron vs vLLM after training.

Train for a few steps, then:
1. Check logprobs in Megatron
2. Save weights to vLLM
3. Check logprobs in vLLM via compute_logprobs
4. Compare position 7 between both
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

    # Key positions
    key_positions = [7, 23, 31, 49]
    print("\nKey positions:")
    for pos in key_positions:
        tgt_str = tokenizer.decode([target_tokens[pos]])
        print(f"  pos={pos:2d}: target={repr(tgt_str)}")

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    print("\n" + "=" * 70)
    print("Creating fresh training client...")
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
    print("\n" + "=" * 70)
    print("Fresh LoRA logprobs (Megatron):")
    print("=" * 70)

    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    fresh_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    for pos in key_positions:
        print(f"  pos={pos:2d}: {fresh_lp[pos]:8.4f}")

    # Also check vLLM with fresh weights
    print("\n" + "=" * 70)
    print("Fresh LoRA logprobs (vLLM via compute_logprobs):")
    print("=" * 70)

    sampling_client = await client.save_weights_and_get_sampling_client_async()
    # compute_logprobs uses the full sequence (input + last target)
    prompt = tinker.ModelInput.from_ints(tokens)  # Full sequence for logprobs
    vllm_fresh_lp = await sampling_client.compute_logprobs_async(prompt)

    for pos in key_positions:
        if pos < len(vllm_fresh_lp) and vllm_fresh_lp[pos] is not None:
            print(f"  pos={pos:2d}: {vllm_fresh_lp[pos]:8.4f}")
        else:
            print(f"  pos={pos:2d}: None")

    # Train 5 steps
    print("\n" + "=" * 70)
    print("Training 5 steps...")
    print("=" * 70)

    for step in range(5):
        fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
        await fwd_bwd.result_async()
        await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()
        print(f"  Step {step+1} done")

    # Check logprobs in Megatron after training
    print("\n" + "=" * 70)
    print("After 5 steps logprobs (Megatron - BEFORE save):")
    print("=" * 70)

    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    trained_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    for pos in key_positions:
        delta = trained_lp[pos] - fresh_lp[pos]
        print(f"  pos={pos:2d}: {trained_lp[pos]:8.4f} (delta={delta:+8.4f})")

    # Save weights to vLLM
    print("\n" + "=" * 70)
    print("Saving trained weights to vLLM...")
    print("=" * 70)

    sampling_client = await client.save_weights_and_get_sampling_client_async()
    print("  Weights saved and loaded into vLLM")

    # Check logprobs in vLLM via compute_logprobs
    print("\n" + "=" * 70)
    print("After 5 steps logprobs (vLLM via compute_logprobs):")
    print("=" * 70)

    prompt = tinker.ModelInput.from_ints(tokens)
    vllm_trained_lp = await sampling_client.compute_logprobs_async(prompt)

    print(f"\n{'pos':>4s} {'Megatron':>12s} {'vLLM':>12s} {'diff':>12s} {'Mega delta':>12s} {'vLLM delta':>12s}")
    print("-" * 70)
    for pos in key_positions:
        mega_val = trained_lp[pos]
        vllm_val = vllm_trained_lp[pos] if pos < len(vllm_trained_lp) and vllm_trained_lp[pos] is not None else float('nan')
        diff = mega_val - vllm_val if not np.isnan(vllm_val) else float('nan')
        mega_delta = mega_val - fresh_lp[pos]
        vllm_fresh = vllm_fresh_lp[pos] if pos < len(vllm_fresh_lp) and vllm_fresh_lp[pos] is not None else float('nan')
        vllm_delta = vllm_val - vllm_fresh if not np.isnan(vllm_val) and not np.isnan(vllm_fresh) else float('nan')
        print(f"{pos:4d} {mega_val:12.4f} {vllm_val:12.4f} {diff:+12.4f} {mega_delta:+12.4f} {vllm_delta:+12.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nKey question: Does position 7 get BETTER in vLLM but WORSE in Megatron?")
    print("If yes, the bug is in how Megatron applies LoRA, not in training.")
    print("If both get worse, the bug is in training itself (or expected gradient interference).")


if __name__ == "__main__":
    asyncio.run(main())
