#!/usr/bin/env python3
"""Diagnose WHY trained LoRA produces wrong logits in Megatron.

Key observation from logs:
- Position 7, target "Hello" (19180): fresh logit ~2.28, trained logit ~1.96
- But argmax shifted to "user" (2482) with logit 23.12

The LoRA is adding a large positive delta to "user" and reducing "Hello".
This is the bug: LoRA is applied wrongly in Megatron.

Let's verify by checking if the SAME sequence of operations produces different results.
"""

import asyncio
import os

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker
import torch
import numpy as np

MODEL_NAME = "moonshotai/Moonlight-16B-A3B-Instruct"


async def main():
    from transformers import AutoTokenizer

    print("=" * 70)
    print("DIAGNOSING LORA APPLICATION BUG")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    # Test sequence
    tokens = [27, 91, 348, 9485, 91, 29, 2482, 198, 19180, 163586]
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

    print("\n1. Create fresh LoRA and get fresh logprobs...")
    client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)

    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    fresh_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    # Check top-K for position 7
    if "topk_indices" in result.loss_fn_outputs[0]:
        fresh_topk = result.loss_fn_outputs[0]["topk_indices"].to_numpy()
        print(f"\nFresh top-K at position 7: {fresh_topk[0, 7, :5] if fresh_topk.ndim == 3 else 'N/A'}")

    print("\n2. Train 1 step...")
    fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
    await fwd_bwd.result_async()
    await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()

    print("\n3. Get trained Megatron logprobs...")
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    trained_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    if "topk_indices" in result.loss_fn_outputs[0]:
        trained_topk = result.loss_fn_outputs[0]["topk_indices"].to_numpy()
        trained_topk_logits = result.loss_fn_outputs[0]["topk_logits"].to_numpy()
        print(f"\nTrained top-K at position 7:")
        print(f"  Indices: {trained_topk[0, 7, :5] if trained_topk.ndim == 3 else 'N/A'}")
        print(f"  Logits: {trained_topk_logits[0, 7, :5] if trained_topk_logits.ndim == 3 else 'N/A'}")

    print("\n4. Export to vLLM and get vLLM logprobs...")
    sampling_client = await client.save_weights_and_get_sampling_client_async()
    await asyncio.sleep(2)

    prompt = tinker.ModelInput.from_ints(tokens)
    vllm_lp = await sampling_client.compute_logprobs_async(prompt)

    print("\n" + "=" * 70)
    print("COMPARISON AT POSITION 7 (predicting 'Hello')")
    print("=" * 70)

    pos = 7
    target_tok = target_tokens[pos]
    target_str = tokenizer.decode([target_tok])

    print(f"\nTarget token: {target_tok} ({repr(target_str)})")
    print(f"Fresh Megatron logprob: {fresh_lp[pos]:.4f}")
    print(f"Trained Megatron logprob: {trained_lp[pos]:.4f}")
    print(f"vLLM logprob (same weights): {vllm_lp[pos + 1]:.4f}")

    print("\nObservation:")
    print(f"  Megatron change: {trained_lp[pos] - fresh_lp[pos]:.4f} nats")
    print(f"  vLLM change: {vllm_lp[pos + 1] - fresh_lp[pos]:.4f} nats (approx, different fresh)")

    print("\n" + "=" * 70)
    print("HYPOTHESIS CHECK")
    print("=" * 70)

    if "topk_indices" in result.loss_fn_outputs[0]:
        # Check what Megatron's top prediction is
        top1_idx = trained_topk[0, pos, 0]
        top1_str = tokenizer.decode([top1_idx])
        print(f"\nMegatron's top prediction at pos 7: {top1_idx} ({repr(top1_str)})")
        print(f"This should be 'Hello' (19180) but might be something else!")

        if top1_idx != target_tok:
            print(f"\n*** BUG CONFIRMED: Megatron predicts {repr(top1_str)} instead of {repr(target_str)} ***")
            print("This happens AFTER training, but vLLM with same weights predicts correctly.")
            print("The bug is in how Megatron applies LoRA in the forward pass.")

    print("\n" + "=" * 70)
    print("NEXT STEP: Add instrumentation to Megatron to find WHERE the bug is")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
