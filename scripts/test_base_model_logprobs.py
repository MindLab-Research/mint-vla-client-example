#!/usr/bin/env python3
"""Test BASE MODEL logprobs between vLLM and Megatron (no LoRA).

This isolates whether the divergence is in:
1. Base model differences (would show even without LoRA)
2. LoRA export/application (would only show with LoRA)
"""

import asyncio
import os

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import httpx
import torch


async def get_megatron_base_logprobs(base_url: str, model_id: str, tokens: list[int], prompt_len: int):
    """Get Megatron logprobs via forward endpoint (no LoRA applied)."""
    import tinker

    # Create a fresh training client to get base model logprobs
    # (fresh LoRA has zero contribution)
    service_client = tinker.ServiceClient(base_url=base_url)
    training_client = await service_client.create_lora_training_client_async(
        "moonshotai/Moonlight-16B-A3B-Instruct", rank=16
    )

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
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
        return None

    logprobs = fwd_result.loss_fn_outputs[0]["logprobs"].to_torch().tolist()
    return logprobs, training_client


async def get_vllm_base_logprobs(base_url: str, tokens: list[int]):
    """Get vLLM BASE MODEL logprobs (no LoRA)."""
    async with httpx.AsyncClient(timeout=300.0) as client:
        # Call compute_logprobs without a session (uses base model)
        response = await client.post(
            f"{base_url}/api/v1/compute_logprobs",
            json={
                "model_name": "moonshotai/Moonlight-16B-A3B-Instruct",
                "input_ids": tokens,
            }
        )

        if response.status_code != 200:
            raise Exception(f"Failed: {response.text}")

        result = response.json()

        # Poll for result
        future_id = result["future_id"]
        for _ in range(60):
            poll_response = await client.post(
                f"{base_url}/api/v1/retrieve_future",
                json={"future_id": future_id}
            )
            if poll_response.status_code == 200:
                return poll_response.json()["logprobs"]
            await asyncio.sleep(1.0)

        raise Exception("Timeout")


async def main():
    print("=" * 80)
    print("BASE MODEL LOGPROB COMPARISON (No LoRA)")
    print("=" * 80)

    model_name = "moonshotai/Moonlight-16B-A3B-Instruct"
    base_url = os.environ["TINKER_BASE_URL"]

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Test sequence
    test_prompt = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
    test_response = "Hi"
    full_text = test_prompt + test_response

    prompt_tokens = tokenizer.encode(test_prompt, add_special_tokens=False)
    full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
    prompt_len = len(prompt_tokens)

    print(f"\nTest sequence: {repr(test_prompt[:40])}... + {repr(test_response)}")
    print(f"Prompt tokens: {prompt_len}, Total: {len(full_tokens)}")

    # Get Megatron base logprobs (uses fresh LoRA = zero contribution)
    print("\n[1] Getting Megatron BASE MODEL logprobs (fresh LoRA = zero weights)...")
    megatron_logprobs, training_client = await get_megatron_base_logprobs(
        base_url, None, full_tokens, prompt_len
    )
    print(f"    Got {len(megatron_logprobs)} logprobs")

    # Export fresh LoRA to vLLM and get logprobs
    print("\n[2] Exporting fresh LoRA to vLLM and getting logprobs...")
    import tinker
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()

    model_input = tinker.ModelInput.from_ints(full_tokens)
    vllm_logprobs_raw = await sampling_client.compute_logprobs_async(model_input)
    vllm_logprobs = list(vllm_logprobs_raw)
    print(f"    Got {len(vllm_logprobs)} logprobs")

    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    print(f"\n{'Pos':>4} | {'Token ID':>8} | {'Text':>15} | {'vLLM':>12} | {'Megatron':>12} | {'Diff':>10}")
    print("-" * 75)

    # Compare response tokens only
    for i in range(response_len := len(full_tokens) - prompt_len):
        token_pos = prompt_len + i
        token_id = full_tokens[token_pos]
        token_text = tokenizer.decode([token_id])

        # vLLM index
        vllm_idx = token_pos - 1
        vllm_lp = vllm_logprobs[vllm_idx] if vllm_idx < len(vllm_logprobs) else float('nan')

        # Megatron index
        mega_idx = token_pos - 1
        mega_lp = megatron_logprobs[mega_idx] if mega_idx < len(megatron_logprobs) else float('nan')

        diff = vllm_lp - mega_lp if not (vllm_lp != vllm_lp or mega_lp != mega_lp) else float('nan')
        flag = " ***" if abs(diff) > 1.0 else ""

        print(f"{token_pos:>4} | {token_id:>8} | {repr(token_text):>15} | {vllm_lp:>12.4f} | {mega_lp:>12.4f} | {diff:>+10.4f}{flag}")

    # Also compare some prompt tokens
    print("\nPrompt token comparison (last 5):")
    for i in range(max(0, prompt_len - 5), prompt_len):
        token_id = full_tokens[i]
        token_text = tokenizer.decode([token_id])

        vllm_idx = i - 1
        mega_idx = i - 1

        vllm_lp = vllm_logprobs[vllm_idx] if vllm_idx >= 0 and vllm_idx < len(vllm_logprobs) else float('nan')
        mega_lp = megatron_logprobs[mega_idx] if mega_idx >= 0 and mega_idx < len(megatron_logprobs) else float('nan')
        diff = vllm_lp - mega_lp if not (vllm_lp != vllm_lp or mega_lp != mega_lp) else float('nan')

        print(f"{i:>4} | {token_id:>8} | {repr(token_text):>15} | {vllm_lp:>12.4f} | {mega_lp:>12.4f} | {diff:>+10.4f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
