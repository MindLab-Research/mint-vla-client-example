#!/usr/bin/env python3
"""Test vLLM vs Megatron on same sequence AFTER SFT training.

Uses the exact sequence from test_lora_application.py to verify
vLLM and Megatron produce matching logprobs after training.
"""

import asyncio
import os
import time

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker
import torch


async def do_training_step(training_client, tokenizer, prompt: str, response: str):
    """Do a single training step."""
    full_text = prompt + response
    tokens = tokenizer.encode(full_text, add_special_tokens=False)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

    prompt_len = len(prompt_tokens)
    mask = [0.0] * prompt_len + [1.0] * (len(tokens) - prompt_len)

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.ones(len(tokens), dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.zeros(len(tokens), dtype=torch.float32)),
        }
    )

    fwd_bwd_future = await training_client.forward_backward_async([datum], loss_fn="importance_sampling")
    await fwd_bwd_future.result_async()

    optim_future = await training_client.optim_step_async(
        adam_params=tinker.AdamParams(learning_rate=1e-4)
    )
    await optim_future.result_async()


async def main():
    print("=" * 70)
    print("TEST: vLLM vs Megatron after SFT (same sequence as lora_application)")
    print("=" * 70)

    model_name = "moonshotai/Moonlight-16B-A3B-Instruct"
    base_url = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Test sequence (same as test_lora_application.py)
    test_prompt = "<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n<|im_start|>assistant\n"
    test_response = "4<|im_end|>"
    test_tokens = tokenizer.encode(test_prompt + test_response, add_special_tokens=False)

    print(f"\nTest sequence: {len(test_tokens)} tokens")
    print(f"Decoded: {tokenizer.decode(test_tokens)!r}")

    # Create training session
    print("\n[1] Creating training session with fresh LoRA...")
    service_client = tinker.ServiceClient(base_url=base_url)
    training_client = await service_client.create_lora_training_client_async(model_name, rank=16)

    # Train 5 steps
    print("\n[2] Training 5 steps...")
    training_examples = [
        ("<|im_start|>user\nWhat is 1+1?<|im_end|>\n<|im_start|>assistant\n", "2<|im_end|>"),
        ("<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n", "4<|im_end|>"),
        ("<|im_start|>user\nWhat is 3+3?<|im_end|>\n<|im_start|>assistant\n", "6<|im_end|>"),
        ("<|im_start|>user\nWhat is 4+4?<|im_end|>\n<|im_start|>assistant\n", "8<|im_end|>"),
        ("<|im_start|>user\nWhat is 5+5?<|im_end|>\n<|im_start|>assistant\n", "10<|im_end|>"),
    ]

    for i, (prompt, response) in enumerate(training_examples):
        await do_training_step(training_client, tokenizer, prompt, response)
        print(f"    Step {i+1}/5 completed")

    # Export to vLLM
    print("\n[3] Exporting trained weights to vLLM...")
    t0 = time.time()
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()
    print(f"    Export completed in {time.time()-t0:.1f}s")

    # Get vLLM logprobs via compute_logprobs
    print("\n[4] Getting vLLM logprobs...")
    vllm_logprobs = await sampling_client.compute_logprobs_async(
        tinker.ModelInput.from_ints(test_tokens)
    )
    print(f"    Got {len(vllm_logprobs)} logprobs")

    # Get Megatron logprobs
    print("\n[5] Getting Megatron logprobs...")
    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(test_tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.ones(len(test_tokens), dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(test_tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.zeros(len(test_tokens), dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.zeros(len(test_tokens), dtype=torch.float32)),
        }
    )
    fwd_future = await training_client.forward_async([datum], loss_fn="importance_sampling")
    fwd_result = await fwd_future.result_async()
    megatron_logprobs = fwd_result.loss_fn_outputs[0]["logprobs"].to_torch().tolist()
    print(f"    Got {len(megatron_logprobs)} logprobs")

    # Compare (with label shift: vLLM[i] corresponds to Megatron[i-1])
    print("\n" + "=" * 70)
    print("COMPARISON: vLLM vs Megatron (after SFT training)")
    print("=" * 70)

    print(f"\n{'pos':>4} | {'token':>8} | {'text':>12} | {'vLLM':>10} | {'Megatron':>10} | {'diff':>10}")
    print("-" * 70)

    diffs = []
    prompt_len = len(tokenizer.encode(test_prompt, add_special_tokens=False))

    for i in range(len(test_tokens)):
        tok = test_tokens[i]
        text = tokenizer.decode([tok])

        v = vllm_logprobs[i] if i < len(vllm_logprobs) else float('nan')
        # Label shift: Megatron logprob at position i-1 corresponds to token i
        m_idx = i - 1
        m = megatron_logprobs[m_idx] if 0 <= m_idx < len(megatron_logprobs) else float('nan')
        d = v - m if not (isinstance(m, float) and m != m) else float('nan')

        if not (isinstance(d, float) and d != d):
            diffs.append(d)

        marker = " *** " if abs(d) > 1 else ""
        in_prompt = "P" if i < prompt_len else "R"
        print(f"{i:>4}{in_prompt}| {tok:>8} | {repr(text):>12} | {v:>10.4f} | {m:>10.4f} | {d:>+10.4f}{marker}")

    # Summary
    if diffs:
        mean_diff = sum(abs(d) for d in diffs) / len(diffs)
        max_diff = max(abs(d) for d in diffs)
        above_1 = sum(1 for d in diffs if abs(d) > 1)

        print("\n" + "=" * 70)
        print(f"Mean |diff|: {mean_diff:.4f}")
        print(f"Max |diff|: {max_diff:.4f}")
        print(f"Tokens with |diff| > 1: {above_1}/{len(diffs)}")
        print("=" * 70)

        if max_diff < 0.5:
            print("\nCONCLUSION: vLLM and Megatron MATCH after training")
        elif above_1 > 0:
            print("\nCONCLUSION: vLLM and Megatron DIVERGE after training")
        else:
            print("\nCONCLUSION: Minor differences, likely numerical precision")


if __name__ == "__main__":
    asyncio.run(main())
