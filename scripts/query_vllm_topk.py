#!/usr/bin/env python3
"""Query vLLM directly via Ray to get top-K tokens at each position.

Run on volcano: python3 /root/tinker_project/tinker-server/scripts/query_vllm_topk.py
"""

import ray
import asyncio

MODEL_NAME = "moonshotai/Moonlight-16B-A3B-Instruct"

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


async def query_vllm_topk():
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]

    print(f"\nInput: {len(input_tokens)} tokens")
    for pos in [7, 8, 23]:
        print(f"Position {pos}: target={target_tokens[pos]} ('{tokenizer.decode([target_tokens[pos]])}')")

    # Find vLLM actor
    actors = ray.util.list_named_actors(all_namespaces=True)
    vllm_actors = [a for a in actors if 'vllm' in a['name'].lower()]

    if not vllm_actors:
        print("ERROR: No vLLM actor found")
        return

    print(f"\nFound vLLM actor: {vllm_actors[0]['name']}")

    # Access the vLLM engine directly through the TinkerVLLMWorker
    # The worker wraps a vLLM async engine
    vllm_worker = ray.get_actor(vllm_actors[0]['name'], namespace=vllm_actors[0].get('namespace', 'tinker'))

    # Use prompt_logprobs=10 to get top-10 at each position
    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        prompt_logprobs=10,  # Get top-10 logprobs at each prompt position
    )

    prompt = TokensPrompt(prompt_token_ids=input_tokens)

    print(f"\nCalling vLLM generate with prompt_logprobs=10...")

    # The TinkerVLLMWorker.generate method signature from verl_inference.py:
    # async def generate(self, prompt_ids, request_id, max_tokens, temperature, ...)
    # But it doesn't pass prompt_logprobs through. Need to call the engine directly.

    # Try using generate_with_lora which might have more options
    try:
        result = await vllm_worker.generate_with_lora.remote(
            prompt_ids=input_tokens,
            request_id="test_topk_001",
            sampling_params=sampling_params,
        )
        print(f"generate_with_lora result: {result}")
    except Exception as e:
        print(f"generate_with_lora failed: {e}")

    # Alternative: check if we can access the raw engine
    # The engine is self.engine in TinkerVLLMWorker, which is a VLLMEngine
    # But it's not directly exposed as a Ray method

    print("\n\nFalling back to checking what the worker exposes...")

    # Let's at least get the logprobs we can
    try:
        logprobs = await vllm_worker.compute_prompt_logprobs.remote(
            prompt_ids=input_tokens,
            request_id="test_logprobs_002",
        )
        print(f"\ncompute_prompt_logprobs result (len={len(logprobs)}):")
        for pos in [7, 8, 23]:
            if pos < len(logprobs):
                print(f"  pos={pos}: logprob={logprobs[pos]:.4f}")
    except Exception as e:
        print(f"compute_prompt_logprobs failed: {e}")


def main():
    print("Connecting to Ray...")
    ray.init(address="auto", ignore_reinit_error=True)

    asyncio.get_event_loop().run_until_complete(query_vllm_topk())


if __name__ == "__main__":
    main()
