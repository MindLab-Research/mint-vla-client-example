#!/usr/bin/env python3
"""Debug logprob alignment between vLLM and Megatron.

Prints detailed token-by-token comparison to identify alignment issues.
"""

import asyncio
import os

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker
import torch


async def main():
    print("=" * 80)
    print("LOGPROB ALIGNMENT DIAGNOSTIC")
    print("=" * 80)

    model_name = "moonshotai/Moonlight-16B-A3B-Instruct"

    # Load tokenizer
    print("\n[1] Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Test sequence
    test_prompt = "<|im_start|>user\nWhat is 2+3?<|im_end|>\n<|im_start|>assistant\n"
    test_response = "5<|im_end|>"
    full_text = test_prompt + test_response

    prompt_tokens = tokenizer.encode(test_prompt, add_special_tokens=False)
    full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
    prompt_len = len(prompt_tokens)
    response_len = len(full_tokens) - prompt_len

    print(f"\n[2] Token Analysis:")
    print(f"    Prompt: {repr(test_prompt[:50])}...")
    print(f"    Response: {repr(test_response)}")
    print(f"    Prompt tokens: {prompt_len}")
    print(f"    Response tokens: {response_len}")
    print(f"    Total tokens: {len(full_tokens)}")
    print(f"\n    Full token IDs: {full_tokens}")

    print("\n    Token breakdown:")
    for i, tid in enumerate(full_tokens):
        text = tokenizer.decode([tid])
        marker = " <-- prompt end" if i == prompt_len - 1 else ""
        marker = " <-- first response" if i == prompt_len else marker
        print(f"      [{i:2}] {tid:>8} = {repr(text):>20}{marker}")

    # Create training session
    print("\n[3] Creating training session with fresh LoRA...")
    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])
    training_client = await service_client.create_lora_training_client_async(model_name, rank=16)
    print(f"    model_id: {training_client.model_id}")

    # Export fresh LoRA
    print("\n[4] Exporting fresh LoRA to vLLM...")
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()

    # Get vLLM logprobs
    print("\n[5] Getting vLLM logprobs...")
    model_input = tinker.ModelInput.from_ints(full_tokens)
    vllm_logprobs_raw = await sampling_client.compute_logprobs_async(model_input)
    vllm_logprobs = list(vllm_logprobs_raw)

    print(f"\n    vLLM raw logprobs length: {len(vllm_logprobs)}")
    print(f"    vLLM logprobs semantics: logprobs[i] = log P(token[i+1] | token[0:i+1])")
    print(f"\n    vLLM logprobs by position:")
    for i, lp in enumerate(vllm_logprobs):
        next_token = full_tokens[i + 1] if i + 1 < len(full_tokens) else None
        next_text = tokenizer.decode([next_token]) if next_token else "N/A"
        marker = ""
        if i == prompt_len - 2:
            marker = " <-- last prompt token prediction"
        if i == prompt_len - 1:
            marker = " <-- FIRST response token prediction"
        if i == prompt_len:
            marker = " <-- SECOND response token prediction"
        print(f"      logprobs[{i:2}] = {lp:>10.4f}  (predicts token[{i+1}] = {repr(next_text):>15}){marker}")

    # Get Megatron logprobs
    print("\n[6] Getting Megatron logprobs...")
    input_tokens = full_tokens[:-1]
    target_tokens = full_tokens[1:]
    mask = [0.0] * (prompt_len - 1) + [1.0] * (len(input_tokens) - prompt_len + 1)

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

    if not fwd_result.loss_fn_outputs:
        print("    ERROR: No loss_fn_outputs!")
        return

    megatron_logprobs = fwd_result.loss_fn_outputs[0]["logprobs"].to_torch().tolist()

    print(f"\n    Megatron input_tokens length: {len(input_tokens)}")
    print(f"    Megatron target_tokens length: {len(target_tokens)}")
    print(f"    Megatron logprobs length: {len(megatron_logprobs)}")
    print(f"    Megatron semantics: logprobs[i] = log P(target[i] | input[0:i+1])")
    print(f"\n    Megatron logprobs by position:")
    for i, lp in enumerate(megatron_logprobs):
        target_token = target_tokens[i]
        target_text = tokenizer.decode([target_token])
        marker = ""
        if i == prompt_len - 2:
            marker = " <-- last prompt token prediction"
        if i == prompt_len - 1:
            marker = " <-- FIRST response token prediction"
        if i == prompt_len:
            marker = " <-- SECOND response token prediction"
        print(f"      logprobs[{i:2}] = {lp:>10.4f}  (predicts target[{i}] = {repr(target_text):>15}){marker}")

    # Compare aligned logprobs
    print("\n" + "=" * 80)
    print("ALIGNED COMPARISON")
    print("=" * 80)
    print("\n    For response tokens:")
    print(f"    {'Token':>8} | {'Text':>15} | {'vLLM idx':>8} | {'vLLM lp':>10} | {'Mega idx':>8} | {'Mega lp':>10} | {'Diff':>10}")
    print("    " + "-" * 95)

    for resp_idx in range(response_len):
        token_pos = prompt_len + resp_idx
        token_id = full_tokens[token_pos]
        token_text = tokenizer.decode([token_id])

        # vLLM: prompt_logprobs[0] = 0.0 placeholder, prompt_logprobs[i] = logprob of token[i]
        # So for token at position token_pos, use prompt_logprobs[token_pos]
        vllm_idx = token_pos
        vllm_lp = vllm_logprobs[vllm_idx] if vllm_idx < len(vllm_logprobs) else float('nan')

        # Megatron: logprobs[i] = logprob of token[i+1] given token[0:i+1]
        # For token at position token_pos, we need logprobs[token_pos - 1]
        mega_idx = token_pos - 1
        mega_lp = megatron_logprobs[mega_idx] if mega_idx < len(megatron_logprobs) else float('nan')

        diff = vllm_lp - mega_lp if not (vllm_lp != vllm_lp or mega_lp != mega_lp) else float('nan')
        flag = " ***" if abs(diff) > 1.0 else ""

        print(f"    {token_id:>8} | {repr(token_text):>15} | {vllm_idx:>8} | {vllm_lp:>10.4f} | {mega_idx:>8} | {mega_lp:>10.4f} | {diff:>+10.4f}{flag}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
