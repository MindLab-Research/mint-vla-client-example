#!/usr/bin/env python3
"""Run both formats multiple times to check for consistency."""

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

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])
    training_client = await service_client.create_lora_training_client_async(model_name, rank=16)

    print(f"Full tokens: {len(full_tokens)}, Prompt len: {prompt_len}")
    print(f"Position 25: target = '<|im_end|>' (token {full_tokens[-1]})")
    print()

    # Run each format 3 times
    for run in range(3):
        print(f"=== Run {run+1} ===")

        # Original format
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

        # Full-sequence format
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

        # Compare position 25 (predicting <|im_end|>)
        print(f"  Original pos 25: {lp_orig[25]:.4f}")
        print(f"  Full-seq pos 25: {lp_full[25]:.4f}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
