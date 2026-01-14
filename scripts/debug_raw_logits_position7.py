#!/usr/bin/env python3
"""Debug raw logits at position 7 between Megatron and vLLM."""

import os
import sys

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import asyncio
import tinker
import torch
import numpy as np

MODEL_NAME = "moonshotai/Moonlight-16B-A3B-Instruct"


async def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Simple test sequence
    TEST_TEXT = "<|im_start|>user\nCount down from 10 to 1, one number per line.<|im_end|>\n<|im_start|>assistant\n"
    tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]

    print("=" * 70)
    print("RAW LOGITS DEBUG AT POSITION 7")
    print("=" * 70)

    print(f"\nToken sequence around position 7:")
    for i in range(5, 12):
        if i < len(input_tokens):
            tok_str = tokenizer.decode([input_tokens[i]])
            print(f"  input[{i}] = {input_tokens[i]:5d} ({repr(tok_str):12s})")

    print(f"\nTarget tokens (shifted by 1):")
    for i in range(5, 12):
        if i < len(target_tokens):
            tok_str = tokenizer.decode([target_tokens[i]])
            print(f"  target[{i}] = {target_tokens[i]:5d} ({repr(tok_str):12s})")

    # The target at position 7 is...
    target_7 = target_tokens[7]
    print(f"\n>>> Target at position 7: token_id={target_7}, string={repr(tokenizer.decode([target_7]))}")

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    print("\n" + "=" * 70)
    print("Creating fresh Megatron LoRA client...")
    print("=" * 70)
    client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)

    # Get Megatron logprobs for the target at position 7
    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.tensor([1.0] * len(input_tokens), dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.ones(len(input_tokens), dtype=torch.float32)),
        },
    )

    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    meg_logprobs = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    print(f"\nMegatron fresh logprobs around position 7:")
    for i in range(5, 12):
        if i < len(meg_logprobs):
            target_str = tokenizer.decode([target_tokens[i]])
            print(f"  pos={i}: logprob={meg_logprobs[i]:8.4f} (target={repr(target_str)})")

    # Export to vLLM and get logprobs
    print("\n" + "=" * 70)
    print("Exporting fresh weights to vLLM...")
    print("=" * 70)

    sampling_client = await client.save_weights_and_get_sampling_client_async()
    await asyncio.sleep(2)

    # Get vLLM logprobs with explicit logprobs_for parameter
    sample_result = await sampling_client.sample_async(
        prompt=tinker.ModelInput.from_ints(input_tokens),
        num_samples=1,
        sampling_params=tinker.SamplingParams(max_tokens=1, temperature=0.0, logprobs=1),
        include_prompt_logprobs=True,
    )

    if sample_result.prompt_logprobs:
        vllm_lp = sample_result.prompt_logprobs
        print(f"\nvLLM fresh prompt_logprobs around position 7:")
        for i in range(5, 12):
            if i < len(vllm_lp):
                # prompt_logprobs[i] is logprob for token input_tokens[i] given input_tokens[:i]
                actual_tok = input_tokens[i] if i < len(input_tokens) else -1
                actual_str = tokenizer.decode([actual_tok]) if actual_tok >= 0 else "?"
                print(f"  pos={i}: logprob={vllm_lp[i]:8.4f} (actual_token={repr(actual_str)})")
    else:
        print("vLLM didn't return prompt_logprobs")

    print("\n" + "=" * 70)
    print("CRITICAL COMPARISON")
    print("=" * 70)
    print(f"\nMegatron pos 7: logprob={meg_logprobs[7]:8.4f} for target token {target_tokens[7]} ({repr(tokenizer.decode([target_tokens[7]]))})")
    print(f"vLLM pos 7: logprob={vllm_lp[7]:8.4f} for actual token {input_tokens[7]} ({repr(tokenizer.decode([input_tokens[7]]))})")

    print("\n>>> These are computing DIFFERENT things!")
    print("    - Megatron pos 7: P(target[7] | input[:8]) = P('Count' | [<, |, im, _start, |, >, user, \\n])")
    print("    - vLLM pos 7: P(input[7] | input[:7]) = P('\\n' | [<, |, im, _start, |, >, user])")

    print("\n>>> To compare apples-to-apples, need to get vLLM logprob for 'Count' at position 7")
    print("    That would be vLLM pos 8's prompt_logprob, which predicts input[8]='Count'")

    # Actually get logprob for specific token from vLLM using logprobs_for
    print("\n" + "=" * 70)
    print("Getting vLLM logprob for specific target tokens...")
    print("=" * 70)

    # Try to get the raw logits for the target tokens
    # vLLM prompt_logprobs returns log P(token[i] | tokens[:i])
    # So prompt_logprobs[i] = log P(input_tokens[i] | input_tokens[:i])
    #
    # We want log P(target_tokens[i] | input_tokens[:i+1])
    # = log P(input_tokens[i+1] | input_tokens[:i+1])
    # = prompt_logprobs[i+1]

    print(f"\nAligned comparison (Megatron[i] vs vLLM[i+1]):")
    for i in range(5, 12):
        if i < len(meg_logprobs) and i+1 < len(vllm_lp):
            meg_lp = meg_logprobs[i]
            vllm_lp_aligned = vllm_lp[i+1]
            target_str = tokenizer.decode([target_tokens[i]])
            diff = abs(meg_lp - vllm_lp_aligned)
            marker = "***" if diff > 1.0 else ""
            print(f"  pos={i}: Meg={meg_lp:8.4f}, vLLM[{i+1}]={vllm_lp_aligned:8.4f}, diff={diff:6.4f} (target={repr(target_str)}) {marker}")

    print("\n>>> If aligned diffs are small, the fresh model is correct.")
    print(">>> The bug must be in how training updates Megatron vs vLLM.")


if __name__ == "__main__":
    asyncio.run(main())
