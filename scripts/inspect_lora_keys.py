#!/usr/bin/env python3
"""Inspect the full structure of exported LoRA state dict."""

import asyncio
import os

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker

MODEL_NAME = "moonshotai/Moonlight-16B-A3B-Instruct"


async def main():
    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    print("Creating LoRA client...")
    client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)

    print("Saving weights to get sampling client...")
    sampling_client = await client.save_weights_and_get_sampling_client_async()

    # Get the raw state dict by examining the client
    print(f"Session ID: {sampling_client.session_id}")

    # We need to trace what gets exported. Let me check via API
    # For now, let's just print some logprobs to confirm the client works
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    test_text = "Hello world"
    tokens = tokenizer.encode(test_text, add_special_tokens=False)
    prompt = tinker.ModelInput.from_ints(tokens)

    lp = await sampling_client.compute_logprobs_async(prompt)
    print(f"Logprobs: {lp}")

    print("\nTo see full keys, need to modify server code or use Ray inspect")


if __name__ == "__main__":
    asyncio.run(main())
