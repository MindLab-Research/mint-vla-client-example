#!/usr/bin/env python3
"""Dump raw logits from vLLM for comparison with Megatron.

Usage (on volcano):
    python scripts/dump_vllm_raw_logits.py
"""
import time
import requests
import ray

MODEL_NAME = "moonshotai/Moonlight-16B-A3B-Instruct"
CHECKPOINT_PATH = "/vePFS-Mindverse/share/code/tinker-server/checkpoints/80e6bf97-56d9-4f3a-9872-7cc4b80a7379_0/debug_checkpoint_20260110_182006/"
DUMP_PATH = "/vePFS-Mindverse/share/code/vllm_raw_logits.pt"
BASE_URL = "http://localhost:8000"

# Same test text as Megatron comparison
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


def main():
    from transformers import AutoTokenizer

    print("=" * 60)
    print("vLLM Raw Logits Dump")
    print("=" * 60)

    # Tokenize
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)
    input_tokens = tokens[:-1]
    print(f"Input tokens ({len(input_tokens)}): {input_tokens}")

    # Create sampling session via HTTP
    print(f"\nCreating sampling session via HTTP...")
    resp = requests.post(
        f"{BASE_URL}/api/v1/create_sampling_session",
        json={"base_model": MODEL_NAME, "session_id": "dump_session"},
    )
    print(f"Response: {resp.status_code} - {resp.text[:200]}")

    # Wait for vLLM actor
    print("\nWaiting for vLLM actor...")
    ray.init(address="auto", namespace="tinker", ignore_reinit_error=True)

    for i in range(60):
        actors = [a for a in ray.util.list_named_actors(all_namespaces=True) if "vllm" in a["name"]]
        if actors:
            print(f"vLLM actor found after {i*2}s: {actors[0]['name']}")
            break
        time.sleep(2)
    else:
        print("ERROR: vLLM actor not found after 120s")
        return

    actor = ray.get_actor(actors[0]["name"], namespace="tinker")

    # Load checkpoint
    print(f"\nLoading checkpoint: {CHECKPOINT_PATH}")
    load_result = ray.get(actor.add_lora_from_path.remote(
        lora_int_id=1,
        lora_path=CHECKPOINT_PATH,
        lora_name="debug_checkpoint",
    ))
    print(f"Load result: {load_result}")

    # Dump raw logits
    print(f"\nDumping raw logits to: {DUMP_PATH}")
    dump_result = ray.get(actor.dump_raw_logits.remote(
        prompt_ids=input_tokens,
        request_id="vllm_dump_001",
        dump_path=DUMP_PATH,
    ))
    print(f"Dump result: {dump_result}")

    # Verify
    import os
    if os.path.exists(DUMP_PATH):
        import torch
        data = torch.load(DUMP_PATH)
        print(f"\nDump verified: {len(data)} entries")
        if data:
            print(f"First entry shape: {data[0]['shape']}")
    else:
        print(f"\nWARNING: Dump file not created")


if __name__ == "__main__":
    main()
