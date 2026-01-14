#!/usr/bin/env python3
"""Compare Megatron vs vLLM with CORRECT index alignment.

The key insight: vLLM logprobs[i+1] corresponds to Megatron logprobs[i]
because they use different indexing conventions.
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

    # Key positions (in Megatron indexing)
    key_positions = [7, 23, 31, 49]
    print("\nKey positions (Megatron indexing):")
    for pos in key_positions:
        inp_str = tokenizer.decode([input_tokens[pos]])
        tgt_str = tokenizer.decode([target_tokens[pos]])
        vllm_pos = pos + 1  # vLLM index
        print(f"  Mega pos={pos:2d}, vLLM pos={vllm_pos:2d}: input={repr(inp_str):8s} -> target={repr(tgt_str):12s}")

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

    # Get fresh Megatron logprobs
    print("\n" + "=" * 70)
    print("Fresh LoRA logprobs (Megatron):")
    print("=" * 70)

    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    mega_fresh = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    # Get fresh vLLM logprobs
    sampling_client = await client.save_weights_and_get_sampling_client_async()
    prompt = tinker.ModelInput.from_ints(tokens)
    vllm_fresh = await sampling_client.compute_logprobs_async(prompt)

    print("\nFresh logprobs comparison (with correct alignment):")
    print(f"{'Mega pos':>8s} {'vLLM pos':>8s} {'Mega lp':>10s} {'vLLM lp':>10s} {'diff':>10s} {'target':>12s}")
    print("-" * 70)
    for pos in key_positions:
        vllm_pos = pos + 1  # Correct alignment
        mega_val = mega_fresh[pos]
        vllm_val = vllm_fresh[vllm_pos] if vllm_pos < len(vllm_fresh) and vllm_fresh[vllm_pos] is not None else float('nan')
        diff = mega_val - vllm_val if not np.isnan(vllm_val) else float('nan')
        tgt_str = tokenizer.decode([target_tokens[pos]])[:10]
        print(f"{pos:8d} {vllm_pos:8d} {mega_val:10.4f} {vllm_val:10.4f} {diff:+10.4f} {repr(tgt_str):>12s}")

    # Train 5 steps
    print("\n" + "=" * 70)
    print("Training 5 steps...")
    print("=" * 70)

    for step in range(5):
        fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
        await fwd_bwd.result_async()
        await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()
        print(f"  Step {step+1} done")

    # Get trained Megatron logprobs
    print("\n" + "=" * 70)
    print("After 5 steps logprobs (Megatron):")
    print("=" * 70)

    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    mega_trained = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    # Get trained vLLM logprobs
    sampling_client = await client.save_weights_and_get_sampling_client_async()
    vllm_trained = await sampling_client.compute_logprobs_async(prompt)

    print("\nTrained logprobs comparison (with correct alignment):")
    print(f"{'Mega pos':>8s} {'vLLM pos':>8s} {'Mega lp':>10s} {'vLLM lp':>10s} {'Mega Δ':>10s} {'vLLM Δ':>10s} {'target':>12s}")
    print("-" * 90)
    for pos in key_positions:
        vllm_pos = pos + 1
        mega_val = mega_trained[pos]
        vllm_val = vllm_trained[vllm_pos] if vllm_pos < len(vllm_trained) and vllm_trained[vllm_pos] is not None else float('nan')
        mega_delta = mega_val - mega_fresh[pos]
        vllm_delta = (vllm_val - vllm_fresh[vllm_pos]) if not np.isnan(vllm_val) and vllm_fresh[vllm_pos] is not None else float('nan')
        tgt_str = tokenizer.decode([target_tokens[pos]])[:10]
        print(f"{pos:8d} {vllm_pos:8d} {mega_val:10.4f} {vllm_val:10.4f} {mega_delta:+10.4f} {vllm_delta:+10.4f} {repr(tgt_str):>12s}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nWith correct alignment (Mega[i] vs vLLM[i+1]):")
    print("- Fresh logprobs should match closely (within ~0.1 nat)")
    print("- Trained logprobs should BOTH improve OR BOTH degrade")
    print("\nIf Mega gets WORSE but vLLM gets BETTER with same weights,")
    print("then the bug is in Megatron's forward pass (expert routing?)")


if __name__ == "__main__":
    asyncio.run(main())
