#!/usr/bin/env python3
"""Query vLLM directly via Ray to get raw logits and top-K tokens.

Run on volcano: python3 /root/tinker_project/tinker-server/scripts/query_vllm_raw_logits.py
"""

import ray
import torch

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


def main():
    from transformers import AutoTokenizer

    print("Connecting to Ray...")
    ray.init(address="auto", ignore_reinit_error=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]

    print(f"\nInput: {len(input_tokens)} tokens")
    print(f"Position 7: target={target_tokens[7]} ('{tokenizer.decode([target_tokens[7]])}')")

    # Find vLLM actor
    actors = ray.util.list_named_actors(all_namespaces=True)
    vllm_actors = [a for a in actors if 'vllm' in a['name'].lower()]

    if not vllm_actors:
        print("ERROR: No vLLM actor found")
        return

    print(f"\nFound vLLM actor: {vllm_actors[0]['name']}")
    vllm_actor = ray.get_actor(vllm_actors[0]['name'], namespace=vllm_actors[0].get('namespace', 'tinker'))

    # Check what methods are available
    print("\nvLLM actor methods related to logprobs/logits:")
    try:
        # Try to get the engine's generate method with logprobs
        # vLLM engines typically have methods like generate, encode, etc.

        # First, let's see what the actor exposes
        import inspect
        # Can't easily inspect remote actor methods, so let's try known methods

        # Try compute_logprobs which should exist
        print("Calling compute_logprobs...")
        logprobs_result = ray.get(vllm_actor.compute_logprobs.remote(
            prompt_ids=input_tokens,
            request_id="test_logprobs_001",
        ), timeout=60)

        print(f"\ncompute_logprobs result type: {type(logprobs_result)}")
        if isinstance(logprobs_result, list):
            print(f"Length: {len(logprobs_result)}")
            if len(logprobs_result) > 7:
                print(f"logprobs[7] = {logprobs_result[7]:.4f}")
        else:
            print(f"Result: {logprobs_result}")

    except Exception as e:
        print(f"compute_logprobs failed: {e}")

    # Try to find a method that returns raw logits
    print("\n\nTrying to get raw logits...")
    try:
        # vLLM v1 has different API - check for generate with special params
        from vllm import SamplingParams as VLLMSamplingParams

        # Generate with logprobs=10 to get top-10 at each position
        sampling_params = VLLMSamplingParams(
            max_tokens=1,
            temperature=0.0,
            logprobs=10,  # Return top-10 logprobs for generated token
            prompt_logprobs=10,  # Return top-10 logprobs for each prompt token
        )

        print(f"Calling generate with prompt_logprobs=10...")
        result = ray.get(vllm_actor.generate.remote(
            prompt_ids=input_tokens,
            request_id="test_topk_001",
            max_tokens=1,
            temperature=0.0,
            # Can't easily pass prompt_logprobs through the tinker interface
        ), timeout=60)

        print(f"generate result type: {type(result)}")
        print(f"Result: {result}")

    except Exception as e:
        print(f"generate with logprobs failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
