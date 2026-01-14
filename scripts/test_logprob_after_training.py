#!/usr/bin/env python3
"""Test logprob alignment BEFORE and AFTER training steps.

Purpose: Isolate whether the logprob discrepancy emerges from:
1. LoRA weight export to vLLM (exported weights aren't correct)
2. Megatron forward pass with non-zero LoRA (computation differs from vLLM)

Test procedure:
1. Create training session with fresh LoRA
2. Verify logprobs match (baseline)
3. Do N training steps on some data
4. Export weights to vLLM again
5. Compare vLLM and Megatron logprobs with trained LoRA
6. Also compute logprobs via vLLM compute_logprobs_async for comparison
"""

import asyncio
import math
import os
import sys
import time

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker
import torch


async def compare_logprobs(
    sampling_client: tinker.SamplingClient,
    training_client: tinker.TrainingClient,
    tokenizer,
    label: str,
):
    """Compare vLLM and Megatron logprobs for a test sequence."""

    # Test prompt
    prompt = """<|im_start|>user
What is 2 + 3?<|im_end|>
<|im_start|>assistant
"""

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    model_input = tinker.ModelInput.from_ints(prompt_tokens)

    # Sample with vLLM
    resp = await sampling_client.sample_async(
        prompt=model_input,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=50,
            temperature=0.7,
            seed=42,
        ),
    )

    seq = resp.sequences[0]
    generated_tokens = list(seq.tokens)
    vllm_sample_logprobs = list(seq.logprobs) if seq.logprobs else []

    if not vllm_sample_logprobs:
        print(f"  [{label}] ERROR: No logprobs from vLLM sampling")
        return None

    # Also get vLLM logprobs via compute_logprobs_async
    full_tokens = prompt_tokens + generated_tokens
    full_input = tinker.ModelInput.from_ints(full_tokens)
    vllm_compute_logprobs = await sampling_client.compute_logprobs_async(full_input)

    # Create Datum for Megatron forward using SFT format
    # Input: [t0, ..., t_{N-1}], Target: [t1, ..., t_N] (last token not in input!)
    prompt_len = len(prompt_tokens)

    # SFT format: input excludes last token, target is shifted by 1
    input_tokens = full_tokens[:-1]  # [t0, ..., t_{N-1}]
    target_tokens = full_tokens[1:]  # [t1, ..., t_N] - includes last token!

    # Mask: 0 for prompt (excluding last prompt token), 1 for response tokens
    # Position i in input predicts target[i] = full[i+1]
    # Prompt ends at position prompt_len-1 in input, which predicts full[prompt_len] = first response token
    mask = [0.0] * (prompt_len - 1) + [1.0] * (len(input_tokens) - prompt_len + 1)

    # Pad vllm logprobs to match input array
    # vLLM logprobs[j] is for generated token j, which is at full position prompt_len + j
    # In SFT format, position (prompt_len + j - 1) in input predicts full[prompt_len + j]
    vllm_logprobs_padded = [0.0] * len(input_tokens)
    for j, lp in enumerate(vllm_sample_logprobs):
        megatron_pos = prompt_len + j - 1
        if 0 <= megatron_pos < len(vllm_logprobs_padded):
            vllm_logprobs_padded[megatron_pos] = lp

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.zeros(len(input_tokens), dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.tensor(vllm_logprobs_padded, dtype=torch.float32)),
        }
    )

    # Megatron forward
    fwd_future = await training_client.forward_async([datum], loss_fn="importance_sampling")
    fwd_result = await fwd_future.result_async()

    if not fwd_result.loss_fn_outputs:
        print(f"  [{label}] ERROR: No loss_fn_outputs")
        return None

    megatron_logprobs = fwd_result.loss_fn_outputs[0]["logprobs"].to_torch().tolist()

    # Compare logprobs
    print(f"\n  [{label}] Comparing logprobs:")
    print(f"  Generated: {tokenizer.decode(generated_tokens)[:80]!r}")

    # vLLM sample vs Megatron (with label shift)
    diffs_shift = []
    diffs_noshift = []
    diffs_vllm_compute = []  # vLLM compute_logprobs vs vLLM sample
    diffs_vllm_compute_megatron = []  # vLLM compute_logprobs vs Megatron

    print(f"\n  {'pos':>4} | {'tok':>8} | {'txt':>10} | {'vSample':>10} | {'vCompute':>10} | {'Megatron':>10} | {'s-m diff':>10}")
    print(f"  {'-'*80}")

    for i in range(min(20, len(generated_tokens))):
        tok = generated_tokens[i]
        text = tokenizer.decode([tok])

        v_sample = vllm_sample_logprobs[i]
        # compute_logprobs returns logprob for each position; for token at prompt_len+i,
        # the logprob is at index prompt_len+i (but note: position 0 has no logprob, so offset by 1)
        v_compute_idx = prompt_len + i
        v_compute = vllm_compute_logprobs[v_compute_idx] if v_compute_idx < len(vllm_compute_logprobs) else float('nan')

        # Megatron with label shift
        m_idx_shift = prompt_len + i - 1
        m_shift = megatron_logprobs[m_idx_shift] if 0 <= m_idx_shift < len(megatron_logprobs) else float('nan')

        # Megatron without shift
        m_idx_noshift = prompt_len + i
        m_noshift = megatron_logprobs[m_idx_noshift] if m_idx_noshift < len(megatron_logprobs) else float('nan')

        diff_shift = v_sample - m_shift if not (math.isnan(m_shift)) else float('nan')
        diff_noshift = v_sample - m_noshift if not (math.isnan(m_noshift)) else float('nan')
        diff_vc = v_sample - v_compute if not (math.isnan(v_compute)) else float('nan')
        diff_vcm = v_compute - m_shift if not (math.isnan(v_compute) or math.isnan(m_shift)) else float('nan')

        if not math.isnan(diff_shift):
            diffs_shift.append(diff_shift)
        if not math.isnan(diff_noshift):
            diffs_noshift.append(diff_noshift)
        if not math.isnan(diff_vc):
            diffs_vllm_compute.append(diff_vc)
        if not math.isnan(diff_vcm):
            diffs_vllm_compute_megatron.append(diff_vcm)

        print(f"  {prompt_len+i:>4} | {tok:>8} | {repr(text):>10} | {v_sample:>10.4f} | {v_compute:>10.4f} | {m_shift:>10.4f} | {diff_shift:>+10.4f}")

    # Summary
    print(f"\n  Summary for [{label}]:")
    if diffs_shift:
        mean_shift = sum(diffs_shift) / len(diffs_shift)
        max_shift = max(abs(d) for d in diffs_shift)
        above_1 = sum(1 for d in diffs_shift if abs(d) > 1)
        print(f"    vLLM_sample vs Megatron (shift): mean={mean_shift:.4f}, max={max_shift:.4f}, |diff|>1: {above_1}/{len(diffs_shift)}")

    if diffs_vllm_compute:
        mean_vc = sum(diffs_vllm_compute) / len(diffs_vllm_compute)
        print(f"    vLLM_sample vs vLLM_compute: mean={mean_vc:.4f}")

    if diffs_vllm_compute_megatron:
        mean_vcm = sum(diffs_vllm_compute_megatron) / len(diffs_vllm_compute_megatron)
        max_vcm = max(abs(d) for d in diffs_vllm_compute_megatron)
        above_1_vcm = sum(1 for d in diffs_vllm_compute_megatron if abs(d) > 1)
        print(f"    vLLM_compute vs Megatron (shift): mean={mean_vcm:.4f}, max={max_vcm:.4f}, |diff|>1: {above_1_vcm}/{len(diffs_vllm_compute_megatron)}")

    return {
        "mean_diff_shift": sum(diffs_shift) / len(diffs_shift) if diffs_shift else float('nan'),
        "max_diff_shift": max(abs(d) for d in diffs_shift) if diffs_shift else float('nan'),
        "above_1": sum(1 for d in diffs_shift if abs(d) > 1) if diffs_shift else 0,
        "total": len(diffs_shift),
    }


async def do_training_step(training_client: tinker.TrainingClient, tokenizer, prompt: str, response: str):
    """Do a single training step using SFT format.

    SFT format:
    - input_tokens: full sequence excluding last token [t0, ..., t_{N-1}]
    - target_tokens: shifted by 1 [t1, ..., t_N] - includes last token not in input!
    - mask: 0 for prompt, 1 for response tokens
    """

    full_text = prompt + response
    tokens = tokenizer.encode(full_text, add_special_tokens=False)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

    prompt_len = len(prompt_tokens)

    # SFT format
    input_tokens = tokens[:-1]  # Exclude last token
    target_tokens = tokens[1:]  # Shift by 1, includes last token

    # Mask: 0 for prompt (up to prompt_len-1), 1 for response
    mask = [0.0] * (prompt_len - 1) + [1.0] * (len(input_tokens) - prompt_len + 1)

    # Advantages: 1 for response tokens (for SFT, uniform weight)
    advantages = [0.0] * (prompt_len - 1) + [1.0] * (len(input_tokens) - prompt_len + 1)

    # Logprobs: dummy zeros (will be computed by forward pass)
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

    # Forward backward
    fwd_bwd_future = await training_client.forward_backward_async([datum], loss_fn="importance_sampling")
    await fwd_bwd_future.result_async()

    # Optim step
    optim_future = await training_client.optim_step_async(
        adam_params=tinker.AdamParams(learning_rate=5e-5)
    )
    await optim_future.result_async()


async def main():
    print("=" * 70)
    print("LOGPROB COMPARISON: BEFORE vs AFTER TRAINING")
    print("=" * 70)

    model_name = "moonshotai/Moonlight-16B-A3B-Instruct"
    base_url = os.environ.get("TINKER_BASE_URL", "http://localhost:8000")

    # Load tokenizer
    print("\n[1] Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Create clients
    print("\n[2] Creating training session with fresh LoRA...")
    service_client = tinker.ServiceClient(base_url=base_url)
    training_client = await service_client.create_lora_training_client_async(model_name, rank=16)

    # Get sampling client with fresh weights
    print("\n[3] Getting sampling client (fresh LoRA)...")
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()

    # Test BEFORE training
    print("\n" + "=" * 70)
    print("[PHASE 1] Testing with FRESH LoRA (before any training)")
    print("=" * 70)
    result_before = await compare_logprobs(sampling_client, training_client, tokenizer, "BEFORE")

    # Do training steps
    print("\n" + "=" * 70)
    print("[PHASE 2] Doing 5 training steps...")
    print("=" * 70)

    training_examples = [
        ("<|im_start|>user\nWhat is 1+1?<|im_end|>\n<|im_start|>assistant\n", "2<|im_end|>"),
        ("<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n", "4<|im_end|>"),
        ("<|im_start|>user\nWhat is 3+3?<|im_end|>\n<|im_start|>assistant\n", "6<|im_end|>"),
        ("<|im_start|>user\nWhat is 4+4?<|im_end|>\n<|im_start|>assistant\n", "8<|im_end|>"),
        ("<|im_start|>user\nWhat is 5+5?<|im_end|>\n<|im_start|>assistant\n", "10<|im_end|>"),
    ]

    for i, (prompt, response) in enumerate(training_examples):
        t0 = time.time()
        await do_training_step(training_client, tokenizer, prompt, response)
        print(f"  Step {i+1}/5 completed in {time.time()-t0:.1f}s")

    # Get NEW sampling client with trained weights
    print("\n[4] Exporting trained weights to vLLM...")
    t0 = time.time()
    sampling_client_after = await training_client.save_weights_and_get_sampling_client_async()
    print(f"    Export completed in {time.time()-t0:.1f}s")

    # Test AFTER training
    print("\n" + "=" * 70)
    print("[PHASE 3] Testing with TRAINED LoRA")
    print("=" * 70)
    result_after = await compare_logprobs(sampling_client_after, training_client, tokenizer, "AFTER")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\nBEFORE training:")
    if result_before:
        print(f"  Mean diff: {result_before['mean_diff_shift']:.4f}")
        print(f"  Max diff: {result_before['max_diff_shift']:.4f}")
        print(f"  Tokens with |diff| > 1: {result_before['above_1']}/{result_before['total']}")

    print(f"\nAFTER training:")
    if result_after:
        print(f"  Mean diff: {result_after['mean_diff_shift']:.4f}")
        print(f"  Max diff: {result_after['max_diff_shift']:.4f}")
        print(f"  Tokens with |diff| > 1: {result_after['above_1']}/{result_after['total']}")

    print("\n" + "=" * 70)
    if result_before and result_after:
        if result_after['max_diff_shift'] > 1 and result_before['max_diff_shift'] < 0.5:
            print("CONCLUSION: Bug emerges AFTER training")
            print("  - Fresh LoRA: logprobs match")
            print("  - Trained LoRA: logprobs diverge")
            print("  - Root cause likely in: LoRA export OR Megatron forward with non-zero LoRA")
        elif result_after['max_diff_shift'] < 0.5:
            print("CONCLUSION: Logprobs still match after training (GOOD)")
            print("  - Issue may be elsewhere (e.g., specific token types, longer sequences)")
        else:
            print("CONCLUSION: Need more investigation")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
