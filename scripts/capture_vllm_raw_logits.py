#!/usr/bin/env python3
"""Capture vLLM raw logits for comparison with Megatron.

This script:
1. Creates a vLLM sampling session with trained LoRA checkpoint
2. Creates trigger file on PFS to enable raw logits dump
3. Runs inference with the same input as Megatron
4. Saves raw logits for comparison
"""

import asyncio
import os
import sys
import time
import uuid
import httpx

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker
import torch
import numpy as np

MODEL_NAME = "moonshotai/Moonlight-16B-A3B-Instruct"
CHECKPOINT_PATH = "/vePFS-Mindverse/share/code/tinker-server/checkpoints/80e6bf97-56d9-4f3a-9872-7cc4b80a7379_0/debug_checkpoint_20260110_182006"
TRIGGER_PATH = "/vePFS-Mindverse/share/code/vllm_prompt_logits_trigger"
OUTPUT_PATH = "/vePFS-Mindverse/share/code/vllm_prompt_raw_logits.pt"

TEST_TEXT = """<|im_start|>user
Count down from 10 to 1, one number per line.<|im_end|>
<|im_start|>assistant
10
9
8
7
6
5
4
3
2
1<|im_end|>"""


async def main():
    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]

    print(f"Sequence: {len(input_tokens)} input tokens")
    print(f"Input tokens: {input_tokens}")

    # Show token sequence
    print("\nToken sequence:")
    for i in range(min(20, len(tokens))):
        print(f"  pos={i:2d}: {tokens[i]:6d} ({repr(tokenizer.decode([tokens[i]])):15s})")
    if len(tokens) > 20:
        print("  ...")

    base_url = os.environ["TINKER_BASE_URL"]

    # Create sampling session with trained checkpoint via REST API
    print("\n" + "=" * 70)
    print("Creating vLLM sampling session with trained checkpoint")
    print("=" * 70)
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    session_id = str(uuid.uuid4())
    async with httpx.AsyncClient(timeout=600.0) as client:
        # Create session
        resp = await client.post(
            f"{base_url}/api/v1/create_sampling_session",
            json={
                "session_id": session_id,
                "base_model": MODEL_NAME,
                "model_path": f"file://{CHECKPOINT_PATH}",
                "lora_rank": 16,
            }
        )
        if resp.status_code != 200:
            print(f"Failed to create session: {resp.status_code} {resp.text}")
            return

        result = resp.json()
        sampling_session_id = result["sampling_session_id"]
        print(f"Sampling session created: {sampling_session_id}")

        # Create trigger file to enable raw logits dump (on remote server via SSH)
        print(f"\nCreating trigger file at {TRIGGER_PATH}")
        import subprocess
        subprocess.run(["ssh", "volcano", f"echo '{OUTPUT_PATH}' > {TRIGGER_PATH}"], check=True)

        # Run inference - this should trigger the logits dump
        print("\n" + "=" * 70)
        print("Running inference to capture raw logits")
        print("=" * 70)

        # Submit sample request
        resp = await client.post(
            f"{base_url}/api/v1/asample",
            json={
                "sampling_session_id": sampling_session_id,
                "prompt": {
                    "chunks": [{"type": "encoded_text", "tokens": input_tokens}]
                },
                "num_samples": 1,
                "sampling_params": {
                    "max_tokens": 1,
                    "temperature": 0.0,
                },
                "include_prompt_logprobs": True,
            }
        )
        if resp.status_code != 200:
            print(f"Failed to submit sample: {resp.status_code} {resp.text}")
            return

        request_id = resp.json()["request_id"]
        print(f"Sample request submitted: {request_id}")

        # Poll for result
        for _ in range(120):
            resp = await client.post(
                f"{base_url}/api/v1/retrieve_future",
                json={"request_id": request_id}
            )
            if resp.status_code == 200:
                result = resp.json()
                print("Inference complete")
                break
            elif resp.status_code == 408:
                await asyncio.sleep(1)
            else:
                print(f"Unexpected response: {resp.status_code} {resp.text}")
                return
        else:
            print("Timeout waiting for inference result")
            return

    # Check if logits were dumped (on remote server)
    time.sleep(3)  # Wait for file write

    import subprocess
    result_check = subprocess.run(["ssh", "volcano", f"test -f {OUTPUT_PATH} && echo 'exists'"], capture_output=True, text=True)

    if result_check.stdout.strip() == "exists":
        print(f"\nRaw logits dumped to {OUTPUT_PATH}")

        # Copy file locally for analysis
        local_output = "/tmp/vllm_prompt_raw_logits.pt"
        subprocess.run(["scp", f"volcano:{OUTPUT_PATH}", local_output], check=True)

        data = torch.load(local_output, map_location="cpu")
        print(f"  Keys: {data.keys()}")
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

        # Compare with Megatron (also need to copy)
        local_megatron = "/tmp/megatron_logits.pt"
        subprocess.run(["scp", "volcano:/vePFS-Mindverse/share/code/logits_processor_input.pt", local_megatron], check=True)
        megatron_data = torch.load(local_megatron, map_location="cpu")
        print(f"\nMegatron logits shape: {megatron_data['logits'].shape}")
        print(f"vLLM logits shape: {data['raw_logits'].shape}")
    else:
        print(f"\nWARNING: Raw logits file not found at {OUTPUT_PATH}")
        print("The dump may have failed. Check if trigger file was consumed.")
        trigger_check = subprocess.run(["ssh", "volcano", f"test -f {TRIGGER_PATH} && echo 'exists'"], capture_output=True, text=True)
        if trigger_check.stdout.strip() == "exists":
            print(f"  Trigger file still exists - dump was NOT triggered")
        else:
            print(f"  Trigger file was consumed - dump may have failed during write")

    # Also show prompt logprobs from API for comparison
    if "prompt_logprobs" in result:
        logprobs = result["prompt_logprobs"]
        print(f"\nvLLM logprobs from API (first 20 positions):")
        for i in range(min(20, len(logprobs))):
            if i < len(target_tokens) and logprobs[i] is not None:
                target_str = tokenizer.decode([target_tokens[i]])
                print(f"  pos={i:2d}: {logprobs[i]:8.4f} (target={repr(target_str)})")


if __name__ == "__main__":
    asyncio.run(main())
