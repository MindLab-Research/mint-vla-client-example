#!/usr/bin/env python3
"""Focused investigation of position 23 divergence.

After 3 training steps:
- Megatron at pos23: -10.2516 nats
- vLLM at pos23: -0.2451 nats
- Same weights!

Question: What is the top-1 prediction at position 23 in Megatron vs vLLM?
"""

import asyncio
import os

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

    print(f"Sequence: {len(input_tokens)} tokens")
    print(f"\nTokens around position 23:")
    for i in range(20, min(28, len(input_tokens))):
        inp = tokenizer.decode([input_tokens[i]])
        tgt = tokenizer.decode([target_tokens[i]])
        print(f"  pos {i}: input={repr(inp)}, target={repr(tgt)} (id={target_tokens[i]})")

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

    # Create fresh LoRA
    print("\n1. Create fresh LoRA...")
    client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)

    # Train 3 steps (reproducing the issue)
    print("\n2. Training 3 steps to reproduce the issue...")
    for step in range(3):
        fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
        await fwd_bwd.result_async()
        await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()

        # Get logprobs at each step
        fwd = await client.forward_async([datum], loss_fn="importance_sampling")
        result = await fwd.result_async()
        lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())
        print(f"   Step {step+1}: pos23={lp[23]:.4f}")

    # Get final Megatron result with top-K
    print("\n3. Getting Megatron forward pass with top-K...")
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    mega_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    if "topk_indices" in result.loss_fn_outputs[0]:
        topk_idx = result.loss_fn_outputs[0]["topk_indices"].to_numpy()
        topk_logits = result.loss_fn_outputs[0]["topk_logits"].to_numpy()

        print(f"\nMegatron top-10 at position 23 (target: '<' = {target_tokens[23]}):")
        print(f"Target logprob: {mega_lp[23]:.4f}")
        for k in range(min(10, topk_idx.shape[-1])):
            idx = int(topk_idx[0, 23, k])
            logit = topk_logits[0, 23, k]
            tok = tokenizer.decode([idx])
            marker = " <-- TARGET" if idx == target_tokens[23] else ""
            print(f"  {k+1}. {repr(tok):15} (id={idx:6}) logit={logit:+.4f}{marker}")
    else:
        print("WARNING: No top-K in output!")

    # Export to vLLM
    print("\n4. Exporting to vLLM...")
    sampling_client = await client.save_weights_and_get_sampling_client_async()
    await asyncio.sleep(2)

    # Get vLLM top-K using sampling
    print("\n5. Getting vLLM prediction at position 23...")
    # Use sample with logprobs to get top tokens
    sample_result = await sampling_client.sample_async(
        prompt=tinker.ModelInput.from_ints(tokens),  # Full sequence
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=1,
            temperature=1.0,
            logprobs=10,  # Get top-10
        ),
        include_prompt_logprobs=True,
    )

    # The target token at position 23 is predicted from input tokens 0..23
    # So we need position 24 in prompt_logprobs (which includes position 0)
    target_pos = 24  # Position in prompt_logprobs for position 23's prediction

    print(f"\nvLLM prompt_logprobs for position 23 (target: '<'):")
    if hasattr(sample_result, 'prompt_logprobs') and len(sample_result.prompt_logprobs) > target_pos:
        vllm_lp = sample_result.prompt_logprobs[target_pos]
        print(f"   Target logprob: {vllm_lp:.4f}" if vllm_lp is not None else "   Target logprob: None")

    # Also get all logprobs
    all_lp = await sampling_client.compute_logprobs_async(tinker.ModelInput.from_ints(tokens))
    print(f"\nvLLM compute_logprobs at position 24: {all_lp[24] if len(all_lp) > 24 else 'N/A'}")

    # CRITICAL COMPARISON
    print("\n" + "=" * 60)
    print("CRITICAL COMPARISON AT POSITION 23")
    print("=" * 60)
    print(f"Target token: '<' (id={target_tokens[23]})")
    print(f"Megatron logprob: {mega_lp[23]:.4f} nats")
    vllm_lp_val = all_lp[24] if len(all_lp) > 24 else None
    print(f"vLLM logprob: {vllm_lp_val:.4f} nats" if vllm_lp_val is not None else "vLLM logprob: N/A")

    if vllm_lp_val is not None:
        diff = mega_lp[23] - vllm_lp_val
        print(f"Difference: {diff:.4f} nats")

        if abs(diff) > 5:
            print("\n*** CATASTROPHIC DIVERGENCE ***")
            print("Same weights produce wildly different results!")
            print("This is a bug in either:")
            print("  1. LoRA application in Megatron forward pass")
            print("  2. Expert routing/permutation handling")
            print("  3. Tensor parallelism sharding")


if __name__ == "__main__":
    asyncio.run(main())
