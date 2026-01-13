#!/usr/bin/env python3
"""Verify sample API logprobs vs compute_prompt_topk_with_lora.

The sample API reportedly gave -0.005 for position 7 target.
compute_prompt_topk_with_lora gives -8.09.

This script tests via direct Ray calls.
"""

import asyncio
import ray

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


async def main():
    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]

    print(f"\nFull sequence: {len(tokens)} tokens")
    print(f"Position 7 target: {target_tokens[7]} ('{tokenizer.decode([target_tokens[7]])}')")

    ray.init(address="auto", ignore_reinit_error=True)

    actors = ray.util.list_named_actors(all_namespaces=True)
    vllm_actors = [a for a in actors if 'vllm' in a['name'].lower()]

    if not vllm_actors:
        print("ERROR: No vLLM actor")
        return

    vllm_actor = ray.get_actor(vllm_actors[0]['name'], namespace=vllm_actors[0].get('namespace', 'tinker'))

    loaded_loras = ray.get(vllm_actor.list_loras.remote(), timeout=30)
    print(f"Loaded LoRAs: {loaded_loras}")

    if not loaded_loras:
        print("No LoRA loaded!")
        return

    lora_id = list(loaded_loras)[0]

    # Test 1: compute_prompt_topk_with_lora
    print("\n" + "=" * 60)
    print("Test 1: compute_prompt_topk_with_lora")
    print("=" * 60)

    topk_result = ray.get(vllm_actor.compute_prompt_topk_with_lora.remote(
        prompt_ids=input_tokens,
        request_id="test_topk_001",
        lora_int_id=lora_id,
        k=10,
    ), timeout=120)

    pos7 = topk_result[7]
    sorted_topk = sorted(pos7.items(), key=lambda x: x[1], reverse=True)

    print(f"Top-5 at position 7:")
    for i, (tok_id, lp) in enumerate(sorted_topk[:5]):
        marker = " <-- TARGET" if tok_id == target_tokens[7] else ""
        print(f"  {i+1}. {tok_id} ('{tokenizer.decode([tok_id])}'): {lp:.4f}{marker}")

    if target_tokens[7] in pos7:
        print(f"\nTarget logprob: {pos7[target_tokens[7]]:.4f}")

    # Test 2: compute_prompt_logprobs_with_lora
    print("\n" + "=" * 60)
    print("Test 2: compute_prompt_logprobs_with_lora")
    print("=" * 60)

    logprobs = ray.get(vllm_actor.compute_prompt_logprobs_with_lora.remote(
        prompt_ids=input_tokens,
        request_id="test_lp_001",
        lora_int_id=lora_id,
    ), timeout=120)

    print(f"Position 7 logprob: {logprobs[7]:.4f}")

    # Test 3: generate_with_lora - generate 1 token from prompt prefix
    print("\n" + "=" * 60)
    print("Test 3: generate_with_lora (greedy, 1 token)")
    print("=" * 60)

    # Use prompt up to and including position 7
    prompt_prefix = input_tokens[:8]
    print(f"Prompt: {len(prompt_prefix)} tokens (up to position 7)")
    print(f"Expected next: {target_tokens[7]} ('{tokenizer.decode([target_tokens[7]])}')")

    gen_result = ray.get(vllm_actor.generate_with_lora.remote(
        prompt_ids=prompt_prefix,
        request_id="test_gen_001",
        lora_int_id=lora_id,
        max_tokens=1,
        temperature=0.0,  # Greedy
        logprobs=True,
    ), timeout=120)

    gen_token = gen_result['token_ids'][0] if gen_result['token_ids'] else None
    gen_lp = gen_result['logprobs'][0] if gen_result['logprobs'] else None

    print(f"Generated: {gen_token} ('{tokenizer.decode([gen_token])}')")
    print(f"Generated logprob: {gen_lp:.4f}")
    print(f"Match target? {gen_token == target_tokens[7]}")

    # Test 4: Generate with temperature=1.0 to see full distribution
    print("\n" + "=" * 60)
    print("Test 4: generate_with_lora (temp=1.0, 1 token)")
    print("=" * 60)

    gen_result2 = ray.get(vllm_actor.generate_with_lora.remote(
        prompt_ids=prompt_prefix,
        request_id="test_gen_002",
        lora_int_id=lora_id,
        max_tokens=1,
        temperature=1.0,
        logprobs=True,
    ), timeout=120)

    gen_token2 = gen_result2['token_ids'][0] if gen_result2['token_ids'] else None
    gen_lp2 = gen_result2['logprobs'][0] if gen_result2['logprobs'] else None

    print(f"Generated: {gen_token2} ('{tokenizer.decode([gen_token2])}')")
    print(f"Generated logprob: {gen_lp2:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"compute_prompt_topk_with_lora pos7: {pos7.get(target_tokens[7], 'N/A'):.4f}")
    print(f"compute_prompt_logprobs_with_lora pos7: {logprobs[7]:.4f}")
    print(f"generate_with_lora greedy: token={gen_token}, logprob={gen_lp:.4f}")

    if abs(pos7.get(target_tokens[7], -100) - logprobs[7]) > 0.01:
        print("\nWARNING: topk and logprobs methods DISAGREE!")


if __name__ == "__main__":
    asyncio.run(main())
