#!/usr/bin/env python3
"""Verify that Megatron and vLLM receive identical token sequences.

The fresh logprob discrepancies suggest a possible sequence alignment issue.
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

    print("\nFull token sequence:")
    print("=" * 80)
    for i, (tok, tgt) in enumerate(zip(tokens, tokens[1:] + [None])):
        tok_str = tokenizer.decode([tok])
        tgt_str = tokenizer.decode([tgt]) if tgt else "N/A"
        print(f"pos={i:2d}: token={tok:6d} {repr(tok_str):15s} -> next={tgt if tgt else 'N/A':>6} {repr(tgt_str):15s}")

    print(f"\nTotal tokens: {len(tokens)}")
    print(f"Input tokens (for Megatron): {len(input_tokens)}")
    print(f"Full sequence (for vLLM compute_logprobs): {len(tokens)}")

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    # Create training client and get Megatron logprobs
    print("\n" + "=" * 80)
    print("Creating fresh training client...")
    print("=" * 80)

    client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.tensor([1.0] * len(input_tokens), dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.ones(len(input_tokens), dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.zeros(len(input_tokens), dtype=torch.float32)),
        }
    )

    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    mega_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    # Save weights to vLLM and get logprobs
    sampling_client = await client.save_weights_and_get_sampling_client_async()

    # CRITICAL: What sequence does vLLM receive?
    # compute_logprobs expects the FULL sequence (including the last token)
    # and returns logprobs for predicting each token given the prefix
    prompt = tinker.ModelInput.from_ints(tokens)
    vllm_lp = await sampling_client.compute_logprobs_async(prompt)

    print(f"\nMegatron logprobs: {len(mega_lp)} positions")
    print(f"vLLM logprobs: {len(vllm_lp)} positions")

    # Check alignment
    print("\n" + "=" * 80)
    print("Logprob comparison (first 30 positions):")
    print("=" * 80)
    print(f"{'pos':>4s} {'Mega':>10s} {'vLLM':>10s} {'diff':>10s} {'input':>10s} {'target':>12s}")
    print("-" * 70)

    for i in range(min(30, len(mega_lp), len(vllm_lp) if vllm_lp else 0)):
        mega_val = mega_lp[i]
        vllm_val = vllm_lp[i] if vllm_lp and i < len(vllm_lp) and vllm_lp[i] is not None else float('nan')
        diff = mega_val - vllm_val if not np.isnan(vllm_val) else float('nan')
        inp_str = tokenizer.decode([input_tokens[i]])[:8] if i < len(input_tokens) else "N/A"
        tgt_str = tokenizer.decode([target_tokens[i]])[:10] if i < len(target_tokens) else "N/A"
        print(f"{i:4d} {mega_val:10.4f} {vllm_val:10.4f} {diff:+10.4f} {repr(inp_str):>10s} {repr(tgt_str):>12s}")

    # Try with shifted indices
    print("\n" + "=" * 80)
    print("Checking if vLLM is shifted by 1:")
    print("=" * 80)
    print("Comparing mega_lp[i] vs vllm_lp[i+1]:")
    print(f"{'pos':>4s} {'Mega[i]':>10s} {'vLLM[i+1]':>10s} {'diff':>10s} {'target[i]':>12s}")
    print("-" * 60)

    for i in range(min(20, len(mega_lp))):
        if vllm_lp and i + 1 < len(vllm_lp) and vllm_lp[i + 1] is not None:
            mega_val = mega_lp[i]
            vllm_val = vllm_lp[i + 1]
            diff = mega_val - vllm_val
            tgt_str = tokenizer.decode([target_tokens[i]])[:10] if i < len(target_tokens) else "N/A"
            print(f"{i:4d} {mega_val:10.4f} {vllm_val:10.4f} {diff:+10.4f} {repr(tgt_str):>12s}")


if __name__ == "__main__":
    asyncio.run(main())
