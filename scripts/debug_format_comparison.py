#!/usr/bin/env python3
"""Debug: print full logprobs array from both formats."""

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

    print(f"Prompt length: {prompt_len}")
    print(f"Full tokens: {len(full_tokens)}")
    print(f"Tokens: {full_tokens}")
    print(f"Decoded: {[tokenizer.decode([t]) for t in full_tokens]}")

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])
    training_client = await service_client.create_lora_training_client_async(model_name, rank=16)

    # ORIGINAL FORMAT
    print("\n" + "="*60)
    print("ORIGINAL FORMAT: input=[:-1], target=[1:]")
    print("="*60)

    input_orig = full_tokens[:-1]
    target_orig = full_tokens[1:]
    mask_orig = [0.0] * (prompt_len - 1) + [1.0] * (len(input_orig) - prompt_len + 1)

    print(f"input length: {len(input_orig)}")
    print(f"target length: {len(target_orig)}")
    print(f"mask length: {len(mask_orig)}")
    print(f"mask: {mask_orig}")

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

    print(f"\nLogprobs array length: {len(lp_orig)}")
    print(f"\nPosition -> (input_token, target_token, logprob, mask):")
    for i in range(len(input_orig)):
        inp_tok = tokenizer.decode([input_orig[i]])
        tgt_tok = tokenizer.decode([target_orig[i]])
        print(f"  {i:3d}: input={inp_tok:15s} target={tgt_tok:15s} logprob={lp_orig[i]:10.4f} mask={mask_orig[i]}")

    # FULL-SEQUENCE FORMAT
    print("\n" + "="*60)
    print("FULL-SEQUENCE FORMAT: input=full, target=shifted+dummy")
    print("="*60)

    input_full = full_tokens
    target_full = full_tokens[1:] + [full_tokens[0]]  # dummy at end
    base_mask = [0.0] * prompt_len + [1.0] * (len(full_tokens) - prompt_len)
    mask_full = base_mask[1:] + [0.0]

    print(f"input length: {len(input_full)}")
    print(f"target length: {len(target_full)}")
    print(f"mask length: {len(mask_full)}")
    print(f"mask: {mask_full}")

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

    print(f"\nLogprobs array length: {len(lp_full)}")
    print(f"\nPosition -> (input_token, target_token, logprob, mask):")
    for i in range(len(input_full)):
        inp_tok = tokenizer.decode([input_full[i]])
        tgt_tok = tokenizer.decode([target_full[i]])
        print(f"  {i:3d}: input={inp_tok:15s} target={tgt_tok:15s} logprob={lp_full[i]:10.4f} mask={mask_full[i]}")


if __name__ == "__main__":
    asyncio.run(main())
