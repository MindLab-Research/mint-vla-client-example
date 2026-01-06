#!/usr/bin/env python3
"""Megatron-only test for label roll fix. No vLLM needed."""

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

    # Simple test sequence
    full_tokens = tokenizer.encode("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello<|im_end|>", add_special_tokens=False)

    print(f"Tokens: {full_tokens}")
    print(f"Decoded: {[tokenizer.decode([t]) for t in full_tokens]}")
    print()

    # Cookbook format: input = [:-1], target = [1:]
    input_tokens = full_tokens[:-1]
    target_tokens = full_tokens[1:]

    # Mask all positions
    mask = [1.0] * len(input_tokens)

    print(f"input len: {len(input_tokens)}, target len: {len(target_tokens)}")
    print(f"Last 3 input: {[tokenizer.decode([t]) for t in input_tokens[-3:]]}")
    print(f"Last 3 target: {[tokenizer.decode([t]) for t in target_tokens[-3:]]}")
    print()

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])
    training_client = await service_client.create_lora_training_client_async(model_name, rank=16)

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.zeros(len(input_tokens), dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.zeros(len(input_tokens), dtype=torch.float32)),
        }
    )

    print("Calling Megatron forward...")
    fwd = await training_client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    logprobs = result.loss_fn_outputs[0]["logprobs"].to_torch().tolist()

    print(f"\nLogprobs (len={len(logprobs)}):")
    for i in range(len(logprobs)):
        inp = tokenizer.decode([input_tokens[i]])
        tgt = tokenizer.decode([target_tokens[i]])
        lp = logprobs[i]
        print(f"  {i}: input='{inp}' -> target='{tgt}' logprob={lp:.4f}")

    # Check last position specifically
    print(f"\n=== LAST POSITION CHECK ===")
    print(f"Last input: '{tokenizer.decode([input_tokens[-1]])}' (id={input_tokens[-1]})")
    print(f"Last target: '{tokenizer.decode([target_tokens[-1]])}' (id={target_tokens[-1]})")
    print(f"Last logprob: {logprobs[-1]:.4f}")
    print()

    # If roll bug exists, last position computes P(first_token) instead of P(last_target)
    first_token = full_tokens[0]
    print(f"First token (wrapped if bug): '{tokenizer.decode([first_token])}' (id={first_token})")

    if logprobs[-1] < -10:
        print("\nWARNING: Last logprob very negative - likely computing wrong token!")
    else:
        print("\nLast logprob looks reasonable.")


if __name__ == "__main__":
    asyncio.run(main())
