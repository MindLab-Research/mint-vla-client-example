#!/usr/bin/env python3
"""Test if export step corrupts Megatron's internal state.

Hypothesis: save_weights_and_get_sampling_client_async() corrupts Megatron LoRA state.
"""

import asyncio
import os
import torch

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker


async def get_megatron_logprobs(training_client, tokens, prompt_len):
    """Get logprobs from Megatron for a sequence."""
    input_tokens = tokens
    target_tokens = tokens[1:] + [tokens[0]]
    base_mask = [0.0] * prompt_len + [1.0] * (len(tokens) - prompt_len)
    mask = base_mask[1:] + [0.0]

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.zeros(len(input_tokens), dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.zeros(len(input_tokens), dtype=torch.float32)),
        }
    )

    fwd_future = await training_client.forward_async([datum], loss_fn="importance_sampling")
    fwd_result = await fwd_future.result_async()
    return fwd_result.loss_fn_outputs[0]["logprobs"].to_torch().tolist()


async def do_sft_step(training_client, tokens, prompt_len, lr=5e-5):
    """Do one SFT step."""
    input_tokens = tokens
    target_tokens = tokens[1:] + [tokens[0]]
    base_mask = [0.0] * prompt_len + [1.0] * (len(tokens) - prompt_len)
    mask = base_mask[1:] + [0.0]
    base_advantages = [0.0] * prompt_len + [1.0] * (len(tokens) - prompt_len)
    advantages = base_advantages[1:] + [0.0]
    logprobs = [0.0] * len(input_tokens)

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.tensor(advantages, dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.tensor(logprobs, dtype=torch.float32)),
        }
    )

    fwd_bwd = await training_client.forward_backward_async([datum], loss_fn="importance_sampling")
    await fwd_bwd.result_async()
    optim = await training_client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=lr))
    await optim.result_async()


async def main():
    print("=" * 70)
    print("TEST: Does export corrupt Megatron's internal state?")
    print("=" * 70)

    model_name = "moonshotai/Moonlight-16B-A3B-Instruct"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Test sequence (same as v3 test)
    test_prompt = "<|im_start|>user\nWhat is 2 + 3?<|im_end|>\n<|im_start|>assistant\n"
    test_response = "5<|im_end|>"
    test_prompt_tokens = tokenizer.encode(test_prompt, add_special_tokens=False)
    test_full_tokens = tokenizer.encode(test_prompt + test_response, add_special_tokens=False)
    test_prompt_len = len(test_prompt_tokens)

    # Training sequences (same as v3 test)
    training_examples = [
        ("<|im_start|>user\nWhat is 1+1?<|im_end|>\n<|im_start|>assistant\n", "2<|im_end|>"),
        ("<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n", "4<|im_end|>"),
        ("<|im_start|>user\nWhat is 3+3?<|im_end|>\n<|im_start|>assistant\n", "6<|im_end|>"),
        ("<|im_start|>user\nWhat is 4+4?<|im_end|>\n<|im_start|>assistant\n", "8<|im_end|>"),
        ("<|im_start|>user\nWhat is 5+5?<|im_end|>\n<|im_start|>assistant\n", "10<|im_end|>"),
    ]

    print(f"\nTest sequence: {test_prompt!r} -> {test_response!r}")
    print(f"Test prompt length: {test_prompt_len}, Full length: {len(test_full_tokens)}")

    # Create training client
    print("\n[1] Creating training session...")
    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])
    training_client = await service_client.create_lora_training_client_async(model_name, rank=16)

    # Get logprobs with FRESH LoRA
    print("\n[2] Megatron logprobs with FRESH LoRA...")
    lp_fresh = await get_megatron_logprobs(training_client, test_full_tokens, test_prompt_len)
    lp_5_fresh = lp_fresh[test_prompt_len - 1]  # Position predicting '5'
    print(f"    Logprob for '5': {lp_5_fresh:.4f}")

    # Do SFT training
    print("\n[3] Doing 5 SFT steps on training examples...")
    for i, (prompt, response) in enumerate(training_examples):
        tokens = tokenizer.encode(prompt + response, add_special_tokens=False)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        await do_sft_step(training_client, tokens, prompt_len, lr=5e-5)
        print(f"    Step {i+1}/5 done")

    # Get logprobs AFTER training, BEFORE export
    print("\n[4] Megatron logprobs AFTER training, BEFORE export...")
    lp_after_train = await get_megatron_logprobs(training_client, test_full_tokens, test_prompt_len)
    lp_5_after_train = lp_after_train[test_prompt_len - 1]
    print(f"    Logprob for '5': {lp_5_after_train:.4f}")

    # Do the export step
    print("\n[5] Calling save_weights_and_get_sampling_client_async (export to vLLM)...")
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()
    print("    Export completed")

    # Get logprobs AFTER export
    print("\n[6] Megatron logprobs AFTER export...")
    lp_after_export = await get_megatron_logprobs(training_client, test_full_tokens, test_prompt_len)
    lp_5_after_export = lp_after_export[test_prompt_len - 1]
    print(f"    Logprob for '5': {lp_5_after_export:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Logprob for '5' at different stages")
    print("=" * 70)
    print(f"  Fresh LoRA:      {lp_5_fresh:.4f}")
    print(f"  After training:  {lp_5_after_train:.4f}")
    print(f"  After export:    {lp_5_after_export:.4f}")

    if abs(lp_5_after_train - lp_5_after_export) > 0.5:
        print(f"\n  BUG FOUND: Export changed Megatron logprobs by {lp_5_after_export - lp_5_after_train:.4f}")
    else:
        print(f"\n  Export did NOT corrupt Megatron state (diff: {lp_5_after_export - lp_5_after_train:.4f})")

    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
