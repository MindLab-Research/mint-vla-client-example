#!/usr/bin/env python3
"""Diagnose label rolling behavior in verl.

Test hypothesis: external labels passed with key "label" get incorrectly rolled.
"""

import asyncio
import os
import torch

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")
os.environ.setdefault("TINKER_TELEMETRY", "0")

import tinker


async def main():
    model_name = "moonshotai/Moonlight-16B-A3B-Instruct"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    prompt = "<|im_start|>user\nWhat is 2+3?<|im_end|>\n<|im_start|>assistant\n"
    response = "5<|im_end|>"

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    full_tokens = tokenizer.encode(prompt + response, add_special_tokens=False)
    prompt_len = len(prompt_tokens)

    print(f"Full tokens: {len(full_tokens)}")
    print(f"Prompt len: {prompt_len}")
    print(f"Tokens: {full_tokens}")
    print(f"Decoded: {[tokenizer.decode([t]) for t in full_tokens]}")
    print()

    # Cookbook format: input = [:-1], target = [1:]
    input_tokens = full_tokens[:-1]  # 26 tokens (0-25)
    target_tokens = full_tokens[1:]  # 26 tokens (positions 1-26)

    # Mask: only compute loss on response tokens
    # input[24] (pos 24) predicts target[24] = token 25 = "5"
    # input[25] (pos 25) predicts target[25] = token 26 = "<|im_end|>"
    mask = [0.0] * (prompt_len - 1) + [1.0] * (len(input_tokens) - prompt_len + 1)

    print("=== Input/Target alignment ===")
    print(f"input_tokens len: {len(input_tokens)}")
    print(f"target_tokens len: {len(target_tokens)}")
    print(f"mask len: {len(mask)}")
    print()

    # Show key positions
    print("Key positions:")
    for pos in [23, 24, 25]:
        if pos < len(input_tokens):
            inp = tokenizer.decode([input_tokens[pos]])
            tgt = tokenizer.decode([target_tokens[pos]])
            m = mask[pos]
            print(f"  pos {pos}: input='{inp}' (id={input_tokens[pos]}), target='{tgt}' (id={target_tokens[pos]}), mask={m}")
    print()

    print("Expected behavior:")
    print("  - Position 24: logprob of '5' given context ending at 'assistant\\n'")
    print("  - Position 25: logprob of '<|im_end|>' given context ending at '5'")
    print()

    print("If verl incorrectly rolls labels:")
    print("  - label[25] = target_tokens[0] = '<' (first token, wrapped)")
    print("  - logprob at pos 25 would be P('<' | ...) instead of P('<|im_end|>' | ...)")
    print()

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])
    training_client = await service_client.create_lora_training_client_async(model_name, rank=16)

    # Get vLLM sampling client for comparison
    print("=== Getting vLLM sampling client ===")
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()

    # Get vLLM logprobs for the same sequence
    # Feed the full sequence as prompt, request 0 new tokens, get prompt logprobs
    print("=== vLLM logprobs for comparison ===")
    full_sequence = prompt + response
    vllm_resp = await sampling_client.sample_async(
        prompt=tinker.ModelInput.from_ints(full_tokens),
        num_samples=1,
        sampling_params=tinker.SamplingParams(max_tokens=1, temperature=1.0),
    )
    # vLLM prompt_logprobs gives logprobs for all prompt tokens
    # Note: prompt_logprobs[i] = log P(prompt[i] | prompt[0:i])
    if hasattr(vllm_resp.sequences[0], 'prompt_logprobs') and vllm_resp.sequences[0].prompt_logprobs:
        vllm_prompt_logprobs = list(vllm_resp.sequences[0].prompt_logprobs)
        print(f"vLLM prompt_logprobs len: {len(vllm_prompt_logprobs)}")
        print(f"vLLM pos 25: {vllm_prompt_logprobs[25]:.4f} (token='5')")
        print(f"vLLM pos 26: {vllm_prompt_logprobs[26]:.4f} (token='<|im_end|>')")
    else:
        print("vLLM prompt_logprobs not available, skipping comparison")
        vllm_prompt_logprobs = None
    print()

    # Test Megatron with current tinker format
    print("=== Megatron with cookbook format ===")
    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.zeros(len(input_tokens), dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.zeros(len(input_tokens), dtype=torch.float32)),
        }
    )

    fwd = await training_client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    megatron_logprobs = result.loss_fn_outputs[0]["logprobs"].to_torch().tolist()

    print(f"Megatron logprobs len: {len(megatron_logprobs)}")
    print(f"Megatron pos 24: {megatron_logprobs[24]:.4f} (expected: ~-0.003 for '5')")
    print(f"Megatron pos 25: {megatron_logprobs[25]:.4f} (expected: ~-6.2 for '<|im_end|>')")
    print()
    print("All Megatron logprobs:")
    for i, lp in enumerate(megatron_logprobs):
        tok = tokenizer.decode([target_tokens[i]]) if i < len(target_tokens) else "?"
        print(f"  pos {i}: {lp:.4f} (target='{tok}')")


if __name__ == "__main__":
    asyncio.run(main())
