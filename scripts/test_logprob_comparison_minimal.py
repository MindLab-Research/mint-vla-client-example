#!/usr/bin/env python3
"""Minimal test comparing vLLM and Megatron logprobs for SAME input.

Purpose: Isolate whether the logprob discrepancy exists in the forward pass
itself, independent of training effects.

Test procedure:
1. Create training session with fresh LoRA (lora_B=zeros -> zero contribution)
2. Get a sampling client for that session (exports zero-contrib LoRA to vLLM)
3. Generate a sequence with vLLM, record logprobs
4. Feed exact same sequence to Megatron forward (no training), record logprobs
5. Compare token-by-token

Expected: With zero-contribution LoRA, both should give ~same logprobs.
If they differ significantly -> bug in forward pass (not training).
"""

import asyncio
import math
import os
import sys
import time

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker


async def main():
    print("=" * 70)
    print("MINIMAL LOGPROB COMPARISON: vLLM vs Megatron")
    print("=" * 70)

    model_name = "moonshotai/Moonlight-16B-A3B-Instruct"
    base_url = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")

    # Load tokenizer locally
    print("\n[1] Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print(f"    Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    # Create tinker client
    print("\n[2] Creating training session with fresh LoRA...")
    service_client = tinker.ServiceClient(base_url=base_url)

    t0 = time.time()
    training_client = await service_client.create_lora_training_client_async(
        model_name, rank=16
    )
    print(f"    Training client created in {time.time()-t0:.1f}s")

    # Get sampling client (this exports fresh LoRA to vLLM)
    print("\n[3] Getting sampling client (exports fresh zero-contrib LoRA)...")
    t0 = time.time()
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()
    print(f"    Sampling client ready in {time.time()-t0:.1f}s")

    # Test prompt - simple arithmetic
    prompt = """<|im_start|>user
What is 2 + 3?<|im_end|>
<|im_start|>assistant
"""

    # Tokenize prompt
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    model_input = tinker.ModelInput.from_ints(prompt_tokens)

    print(f"\n[4] Sampling with vLLM...")
    print(f"    Prompt tokens: {len(prompt_tokens)}")
    t0 = time.time()

    # Sample with correct API
    resp = await sampling_client.sample_async(
        prompt=model_input,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=100,
            temperature=0.7,
            seed=42,
        ),
    )
    print(f"    Sampled in {time.time()-t0:.1f}s")

    seq = resp.sequences[0]
    generated_tokens = seq.tokens
    vllm_logprobs = seq.logprobs

    if vllm_logprobs is None:
        print("    ERROR: No logprobs returned from vLLM")
        return

    generated_text = tokenizer.decode(generated_tokens)
    print(f"\n    Generated text ({len(generated_text)} chars):")
    print(f"    {repr(generated_text[:200])}")
    print(f"\n    vLLM produced {len(generated_tokens)} tokens with logprobs")

    # Construct full sequence for Megatron
    full_tokens = prompt_tokens + list(generated_tokens)
    prompt_len = len(prompt_tokens)

    print(f"\n[5] Preparing Megatron forward pass...")
    print(f"    Full sequence: {len(full_tokens)} tokens")
    print(f"    Prompt length: {prompt_len} tokens")
    print(f"    Response length: {len(generated_tokens)} tokens")

    # Create training batch for forward-only pass
    import torch

    # SFT format: input = full[:-1], target = full[1:]
    # This means:
    #   - input[i] predicts target[i] = full[i+1]
    #   - At position i, logprob = log P(full[i+1] | full[0:i+1])
    #
    # vLLM logprobs[j] = log P(generated[j] | prompt + generated[0:j])
    #                  = log P(full[prompt_len + j] | full[0:prompt_len + j])
    #
    # To compare: Megatron position i predicts full[i+1]
    # To predict full[prompt_len + j], we need i+1 = prompt_len + j, so i = prompt_len + j - 1
    input_tokens = full_tokens[:-1]  # N-1 tokens
    target_tokens = full_tokens[1:]  # N-1 tokens, shifted

    # Mask: only care about response tokens (positions prompt_len-1 onwards in input array)
    # Position prompt_len-1 in input predicts full[prompt_len] = first generated token
    mask = [0.0] * (prompt_len - 1) + [1.0] * (len(input_tokens) - prompt_len + 1)

    # Pad vllm logprobs to match input sequence length
    # vllm_logprobs[j] corresponds to Megatron position prompt_len + j - 1
    vllm_logprobs_padded = [0.0] * len(input_tokens)
    for j, lp in enumerate(vllm_logprobs):
        megatron_pos = prompt_len + j - 1
        if 0 <= megatron_pos < len(vllm_logprobs_padded):
            vllm_logprobs_padded[megatron_pos] = lp

    print(f"    SFT format: input_tokens={len(input_tokens)}, target_tokens={len(target_tokens)}")

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),  # Use input_tokens, not full_tokens
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            # Dummy advantages - we just want logprobs, not training
            "advantages": tinker.TensorData.from_torch(torch.zeros(len(input_tokens), dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.tensor(vllm_logprobs_padded, dtype=torch.float32)),
        }
    )

    print(f"\n[6] Running Megatron forward pass (via forward with no update)...")
    t0 = time.time()

    # Use forward to compute logprobs (not forward_backward to avoid backward pass)
    try:
        fwd_future = await training_client.forward_async(
            [datum],
            loss_fn="importance_sampling",
        )
        fwd_bwd_result = await fwd_future.result_async()
        print(f"    Forward pass completed in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"    ERROR in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return

    # Extract Megatron logprobs
    if not fwd_bwd_result.loss_fn_outputs:
        print("    ERROR: No loss_fn_outputs returned")
        return

    megatron_output = fwd_bwd_result.loss_fn_outputs[0]
    if "logprobs" not in megatron_output:
        print(f"    ERROR: No logprobs in output. Keys: {list(megatron_output.keys())}")
        return

    megatron_logprobs_tensor = megatron_output["logprobs"]
    megatron_logprobs = megatron_logprobs_tensor.to_torch().tolist()
    print(f"    Megatron produced {len(megatron_logprobs)} logprobs")

    # Compare logprobs token-by-token
    print("\n" + "=" * 70)
    print("COMPARISON: vLLM vs Megatron logprobs for response tokens")
    print("=" * 70)

    # vLLM logprobs are for generated tokens (starting after prompt)
    # Megatron logprobs need alignment based on label shift convention
    #
    # From metrics.py docstring:
    # - sampling_logprobs[i] = log P(token[i] | context up to i-1) (vLLM convention)
    # - training_logprobs[i] = log P(token[i+1] | context up to i) (Megatron with label shift)
    #
    # So to compare token[i]'s probability:
    # - vLLM: sampling_logprobs[i]
    # - Megatron: training_logprobs[i-1]

    print(f"\n{'pos':>4} | {'token':>8} | {'text':>15} | {'vLLM':>10} | {'Megatron':>10} | {'diff':>10} | {'ratio':>10}")
    print("-" * 90)

    diffs = []
    for i in range(min(30, len(generated_tokens))):
        vllm_idx = i  # vLLM index in generated_tokens
        # Megatron index: need to account for prompt offset and label shift
        # vLLM's logprobs[i] is for generated token at position prompt_len + i
        # Megatron's logprobs[prompt_len + i - 1] should give P(token[prompt_len + i])
        megatron_idx = prompt_len + i - 1  # -1 for label shift

        if megatron_idx < 0 or megatron_idx >= len(megatron_logprobs):
            continue

        tok = generated_tokens[vllm_idx]
        text = tokenizer.decode([tok])
        v_lp = vllm_logprobs[vllm_idx]
        m_lp = megatron_logprobs[megatron_idx]
        diff = v_lp - m_lp
        diffs.append(diff)

        # Calculate probability ratio
        ratio = math.exp(diff) if abs(diff) < 50 else float('inf')

        print(f"{prompt_len + i:>4} | {tok:>8} | {repr(text):>15} | {v_lp:>10.4f} | {m_lp:>10.4f} | {diff:>+10.4f} | {ratio:>10.2f}x")

    if diffs:
        print("\n" + "-" * 90)
        print(f"SUMMARY (with label shift):")
        print(f"  Mean diff: {sum(diffs)/len(diffs):.4f}")
        print(f"  Max diff: {max(abs(d) for d in diffs):.4f}")
        print(f"  Tokens compared: {len(diffs)}")

        # Count how many tokens have >1 nat difference
        above_1 = sum(1 for d in diffs if abs(d) > 1)
        above_5 = sum(1 for d in diffs if abs(d) > 5)
        above_10 = sum(1 for d in diffs if abs(d) > 10)
        print(f"  Tokens with |diff| > 1: {above_1} ({100*above_1/len(diffs):.1f}%)")
        print(f"  Tokens with |diff| > 5: {above_5} ({100*above_5/len(diffs):.1f}%)")
        print(f"  Tokens with |diff| > 10: {above_10} ({100*above_10/len(diffs):.1f}%)")

    # Also try NO SHIFT alignment
    print("\n" + "=" * 70)
    print("ALTERNATE ALIGNMENT (no shift): vLLM[i] vs Megatron[prompt_len + i]")
    print("=" * 70)

    print(f"\n{'pos':>4} | {'token':>8} | {'text':>15} | {'vLLM':>10} | {'Megatron':>10} | {'diff':>10}")
    print("-" * 80)

    diffs_noshift = []
    for i in range(min(30, len(generated_tokens))):
        megatron_idx = prompt_len + i  # No shift

        if megatron_idx >= len(megatron_logprobs):
            continue

        tok = generated_tokens[i]
        text = tokenizer.decode([tok])
        v_lp = vllm_logprobs[i]
        m_lp = megatron_logprobs[megatron_idx]
        diff = v_lp - m_lp
        diffs_noshift.append(diff)

        print(f"{prompt_len + i:>4} | {tok:>8} | {repr(text):>15} | {v_lp:>10.4f} | {m_lp:>10.4f} | {diff:>+10.4f}")

    if diffs_noshift:
        print(f"\nNO-SHIFT SUMMARY: Mean diff = {sum(diffs_noshift)/len(diffs_noshift):.4f}")

    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    if diffs:
        mean_diff = sum(diffs) / len(diffs)
        mean_diff_noshift = sum(diffs_noshift) / len(diffs_noshift) if diffs_noshift else 999

        # Which alignment is better?
        if abs(mean_diff) < abs(mean_diff_noshift):
            print(f"  LABEL SHIFT alignment is better (mean diff: {mean_diff:.4f})")
        else:
            print(f"  NO SHIFT alignment is better (mean diff: {mean_diff_noshift:.4f})")

        best_mean = min(abs(mean_diff), abs(mean_diff_noshift))
        if best_mean < 0.5:
            print("  GOOD: Logprobs are roughly aligned (best mean diff < 0.5)")
        elif best_mean < 2.0:
            print("  WARNING: Some misalignment (best mean diff < 2.0)")
        else:
            print("  BUG: Significant misalignment detected!")
            print("  This indicates vLLM and Megatron produce different distributions")
            print("  for the SAME input with SAME (zero-contrib) LoRA.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
