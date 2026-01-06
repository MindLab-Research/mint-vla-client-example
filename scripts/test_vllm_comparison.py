#!/usr/bin/env python3
"""Compare Megatron formats against vLLM ground truth."""

import asyncio
import os
import torch

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

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

    print(f"Full tokens: {len(full_tokens)}, Prompt len: {prompt_len}")
    print(f"Response tokens: {full_tokens[prompt_len:]} = {[tokenizer.decode([t]) for t in full_tokens[prompt_len:]]}")

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])
    training_client = await service_client.create_lora_training_client_async(model_name, rank=16)
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()

    # Get vLLM logprobs (ground truth)
    print("\n=== vLLM compute_logprobs (ground truth) ===")
    vllm_logprobs = await sampling_client.compute_logprobs_async(
        tinker.ModelInput.from_ints(full_tokens)
    )
    print(f"vLLM logprobs length: {len(vllm_logprobs)}")
    print(f"Position 25 (predicting '<|im_end|>'): {vllm_logprobs[25]:.4f}")
    print(f"Position 26 (predicting next): {vllm_logprobs[26] if len(vllm_logprobs) > 26 else 'N/A'}")

    # Original format (Megatron)
    print("\n=== Megatron Original Format (input=[:-1], target=[1:]) ===")
    input_orig = full_tokens[:-1]
    target_orig = full_tokens[1:]
    mask_orig = [0.0] * (prompt_len - 1) + [1.0] * (len(input_orig) - prompt_len + 1)

    datum_orig = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_orig),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.tensor(mask_orig, dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_orig, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.zeros(len(input_orig), dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.zeros(len(input_orig), dtype=torch.float32)),
        }
    )

    fwd = await training_client.forward_async([datum_orig], loss_fn="importance_sampling")
    result = await fwd.result_async()
    lp_orig = result.loss_fn_outputs[0]["logprobs"].to_torch().tolist()
    print(f"Megatron logprobs length: {len(lp_orig)}")
    print(f"Position 25 (predicting '<|im_end|>'): {lp_orig[25]:.4f}")

    # Full-sequence format (Megatron)
    print("\n=== Megatron Full-Sequence Format (input=full, target=shifted) ===")
    input_full = full_tokens
    target_full = full_tokens[1:] + [full_tokens[0]]
    base_mask = [0.0] * prompt_len + [1.0] * (len(full_tokens) - prompt_len)
    mask_full = base_mask[1:] + [0.0]

    datum_full = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_full),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.tensor(mask_full, dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_full, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.zeros(len(input_full), dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.zeros(len(input_full), dtype=torch.float32)),
        }
    )

    fwd = await training_client.forward_async([datum_full], loss_fn="importance_sampling")
    result = await fwd.result_async()
    lp_full = result.loss_fn_outputs[0]["logprobs"].to_torch().tolist()
    print(f"Megatron logprobs length: {len(lp_full)}")
    print(f"Position 25 (predicting '<|im_end|>'): {lp_full[25]:.4f}")

    # Summary
    print("\n=== SUMMARY: Logprob for '<|im_end|>' ===")
    print(f"vLLM:                  {vllm_logprobs[25]:.4f}")
    print(f"Megatron Original:     {lp_orig[25]:.4f}")
    print(f"Megatron Full-seq:     {lp_full[25]:.4f}")
    print(f"\nDiff (Orig vs vLLM):   {lp_orig[25] - vllm_logprobs[25]:.4f}")
    print(f"Diff (Full vs vLLM):   {lp_full[25] - vllm_logprobs[25]:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
