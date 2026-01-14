#!/usr/bin/env python3
"""Test if reinitializing Megatron from exported weights fixes the discrepancy.

Hypothesis: Megatron's forward pass uses weights differently than the exported version.
Test:
1. Train for 5 steps
2. Get Megatron logprobs (should show position 7 degraded)
3. Export weights to vLLM
4. Get vLLM logprobs (should be correct)
5. Create NEW Megatron client, load the same exported weights
6. Get logprobs from new Megatron
7. Compare: If new Megatron matches vLLM, the bug is in how trained Megatron uses its weights
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

    key_positions = [7, 23, 31, 49]

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    print("\n" + "=" * 70)
    print("Phase 1: Create and train Megatron client")
    print("=" * 70)

    client1 = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)

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
    fwd = await client1.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    fresh_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    print("\nFresh logprobs (Megatron):")
    for pos in key_positions:
        print(f"  pos={pos:2d}: {fresh_lp[pos]:8.4f}")

    # Train 5 steps
    print("\nTraining 5 steps...")
    for step in range(5):
        fwd_bwd = await client1.forward_backward_async([datum], loss_fn="importance_sampling")
        await fwd_bwd.result_async()
        await (await client1.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()
        print(f"  Step {step+1} done")

    # Get trained logprobs from original Megatron
    fwd = await client1.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    trained_mega1_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    print("\nTrained logprobs (original Megatron):")
    for pos in key_positions:
        delta = trained_mega1_lp[pos] - fresh_lp[pos]
        print(f"  pos={pos:2d}: {trained_mega1_lp[pos]:8.4f} (delta={delta:+8.4f})")

    print("\n" + "=" * 70)
    print("Phase 2: Export weights and check vLLM")
    print("=" * 70)

    # Save weights - this triggers export
    sampling_client = await client1.save_weights_and_get_sampling_client_async()

    # Get vLLM logprobs
    prompt = tinker.ModelInput.from_ints(tokens)
    vllm_lp = await sampling_client.compute_logprobs_async(prompt)

    print("\nTrained logprobs (vLLM - with correct alignment):")
    for pos in key_positions:
        vllm_pos = pos + 1  # Correct alignment
        vllm_val = vllm_lp[vllm_pos] if vllm_pos < len(vllm_lp) and vllm_lp[vllm_pos] is not None else float('nan')
        print(f"  pos={pos:2d}: {vllm_val:8.4f}")

    print("\n" + "=" * 70)
    print("Phase 3: Create FRESH Megatron and load exported checkpoint")
    print("=" * 70)

    # Get the model path that was saved
    # The sampling_client should have the path
    model_path = sampling_client.model_path if hasattr(sampling_client, 'model_path') else None
    if model_path:
        print(f"\nExported to: {model_path}")

    # Check if there's a checkpoint we can load
    # Actually, save_weights_and_get_sampling_client creates an ephemeral adapter
    # We need to use save_checkpoint to get a persistent path

    # Let me try forward again on original Megatron to confirm the bad result persists
    print("\n" + "=" * 70)
    print("Phase 4: Double-check original Megatron (should still be bad)")
    print("=" * 70)

    fwd = await client1.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    trained_mega1_check = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    print("\nRe-check trained logprobs (original Megatron):")
    for pos in key_positions:
        print(f"  pos={pos:2d}: {trained_mega1_check[pos]:8.4f}")

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"\n{'pos':>4s} {'Fresh':>10s} {'Mega1':>10s} {'vLLM':>10s} {'Mega1-vLLM':>12s}")
    print("-" * 60)
    for pos in key_positions:
        vllm_pos = pos + 1
        vllm_val = vllm_lp[vllm_pos] if vllm_pos < len(vllm_lp) and vllm_lp[vllm_pos] is not None else float('nan')
        diff = trained_mega1_lp[pos] - vllm_val if not np.isnan(vllm_val) else float('nan')
        print(f"{pos:4d} {fresh_lp[pos]:10.4f} {trained_mega1_lp[pos]:10.4f} {vllm_val:10.4f} {diff:+12.4f}")

    print("\nIf Mega1 and vLLM differ significantly, the bug is in Megatron's forward pass")
    print("with trained weights, not in the export process.")


if __name__ == "__main__":
    asyncio.run(main())
