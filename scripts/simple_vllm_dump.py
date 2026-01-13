#!/usr/bin/env python3
"""Simple script to trigger vLLM forward pass and dump raw logits."""

import asyncio
import os

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker

MODEL_NAME = "moonshotai/Moonlight-16B-A3B-Instruct"
CHECKPOINT_PATH = "/vePFS-Mindverse/share/code/tinker-server/checkpoints/80e6bf97-56d9-4f3a-9872-7cc4b80a7379_0/debug_checkpoint_20260110_182006/"

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

    print(f"Sequence length: {len(input_tokens)} input tokens")
    print(f"First 10 tokens: {input_tokens[:10]}")

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    # Create Megatron client with trained checkpoint
    print("Creating Megatron client and loading trained checkpoint...")
    meg_client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)
    await (await meg_client.load_state_async(CHECKPOINT_PATH)).result_async()
    print("Checkpoint loaded into Megatron")

    # Export to vLLM
    print("\nExporting to vLLM...")
    vllm_client = await meg_client.save_weights_and_get_sampling_client_async()
    await asyncio.sleep(2)
    print("vLLM client ready")

    # Get prompt logprobs from vLLM (this triggers the dump)
    print("\nRunning vLLM forward pass with prompt_logprobs=5...")
    try:
        result = await vllm_client.sample_async(
            prompt=tinker.ModelInput.from_ints(input_tokens),
            num_samples=1,
            sampling_params=tinker.SamplingParams(max_tokens=1, temperature=0.0),
            topk_prompt_logprobs=5,
        )
        print(f"Result type: {type(result)}")
        if hasattr(result, 'prompt_logprobs'):
            print(f"Got {len(result.prompt_logprobs)} prompt logprobs")
    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()

    print("\nDone! Check for dump file at /vePFS-Mindverse/share/code/vllm_raw_logits.pt")


if __name__ == "__main__":
    asyncio.run(main())
