#!/usr/bin/env python3
"""Test Megatron SFT consistency: logprobs should increase for trained tokens.

No vLLM involved. Pure Megatron self-consistency check.
"""

import asyncio
import os
import torch

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker


async def get_megatron_logprobs(training_client, tokens, prompt_len):
    """Get logprobs from Megatron for a sequence."""
    # Full-sequence format: input=[t0..tN], target=[t1..tN, dummy], mask last=0
    input_tokens = tokens
    target_tokens = tokens[1:] + [tokens[0]]  # shifted, dummy at end

    base_mask = [0.0] * prompt_len + [1.0] * (len(tokens) - prompt_len)
    mask = base_mask[1:] + [0.0]  # dummy position masked out

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

    # For SFT: advantages=1 for response tokens, logprobs=0 (no importance weighting)
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
    print("MEGATRON SFT CONSISTENCY TEST")
    print("If SFT works, logprobs for target tokens should INCREASE")
    print("=" * 70)

    model_name = "moonshotai/Moonlight-16B-A3B-Instruct"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Training sequence
    prompt = "<|im_start|>user\nWhat is 2+3?<|im_end|>\n<|im_start|>assistant\n"
    response = "5<|im_end|>"

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    full_tokens = tokenizer.encode(prompt + response, add_special_tokens=False)
    prompt_len = len(prompt_tokens)

    print(f"\nPrompt: {prompt!r}")
    print(f"Response: {response!r}")
    print(f"Prompt length: {prompt_len}, Full length: {len(full_tokens)}")

    # Create training client
    print("\n[1] Creating training session...")
    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])
    training_client = await service_client.create_lora_training_client_async(model_name, rank=16)

    # Get logprobs BEFORE training
    print("\n[2] Getting Megatron logprobs BEFORE training...")
    logprobs_before = await get_megatron_logprobs(training_client, full_tokens, prompt_len)

    # Extract response token logprobs
    # Position i in logprobs corresponds to predicting token i+1 (due to shifted targets)
    # Response starts at prompt_len, so we track positions prompt_len-1 onwards
    response_logprobs_before = []
    print("\n  Response token logprobs BEFORE:")
    for i in range(len(full_tokens) - prompt_len):
        # Position that predicts this response token
        pos = prompt_len - 1 + i  # Position predicting response token i
        target_tok = full_tokens[prompt_len + i]  # The response token being predicted
        if pos < len(logprobs_before):
            lp = logprobs_before[pos]
            response_logprobs_before.append((pos, target_tok, lp))
            print(f"    pos {pos} predicts token {target_tok} ({tokenizer.decode([target_tok])!r}) -> logprob {lp:.4f}")

    # Do SFT steps
    num_steps = 10
    print(f"\n[3] Doing {num_steps} SFT steps...")
    for i in range(num_steps):
        await do_sft_step(training_client, full_tokens, prompt_len, lr=1e-4)
        if (i + 1) % 5 == 0:
            print(f"    Step {i+1}/{num_steps} done")

    # Get logprobs AFTER training
    print("\n[4] Getting Megatron logprobs AFTER training...")
    logprobs_after = await get_megatron_logprobs(training_client, full_tokens, prompt_len)

    response_logprobs_after = []
    print("\n  Response token logprobs AFTER:")
    for i in range(len(full_tokens) - prompt_len):
        pos = prompt_len - 1 + i
        target_tok = full_tokens[prompt_len + i]
        if pos < len(logprobs_after):
            lp = logprobs_after[pos]
            response_logprobs_after.append((pos, target_tok, lp))
            print(f"    pos {pos} predicts token {target_tok} ({tokenizer.decode([target_tok])!r}) -> logprob {lp:.4f}")

    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON: BEFORE vs AFTER")
    print("=" * 70)
    print(f"\n  {'pos':>4} | {'token':>12} | {'BEFORE':>10} | {'AFTER':>10} | {'CHANGE':>10}")
    print(f"  {'-'*60}")

    total_change = 0
    for i in range(min(len(response_logprobs_before), len(response_logprobs_after))):
        pos, tok, before = response_logprobs_before[i]
        _, _, after = response_logprobs_after[i]
        change = after - before
        total_change += change

        # Positive change = logprob increased = better
        indicator = "+" if change > 0 else ""
        print(f"  {pos:>4} | {tokenizer.decode([tok]):>12} | {before:>10.4f} | {after:>10.4f} | {indicator}{change:>9.4f}")

    n = min(len(response_logprobs_before), len(response_logprobs_after))
    print(f"\n  Mean logprob change: {total_change / n if n > 0 else 0:+.4f}")

    if total_change > 0:
        print("\n  RESULT: Logprobs INCREASED after SFT (expected behavior)")
    else:
        print("\n  RESULT: Logprobs DECREASED after SFT (BUG - SFT should increase logprobs)")

    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
