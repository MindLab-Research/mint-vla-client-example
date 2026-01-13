#!/usr/bin/env python3
"""Check vLLM logprobs at ASSISTANT RESPONSE positions, not user message."""

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

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]

    print("Assistant response positions (32+):")
    print("=" * 60)
    for i in range(30, min(52, len(tokens))):
        tok_str = tokenizer.decode([tokens[i]])
        print(f"  pos={i:2d}: token={tokens[i]:6d} ({repr(tok_str):15s})")

    # Key positions to check:
    # pos 31 -> target is '10' (first response content)
    # pos 32 -> target is '\n'
    # pos 33 -> target is '9'
    # etc.

    positions_to_check = [31, 32, 33, 34, 35]  # First few assistant response positions

    print("\nConnecting to Ray...")
    ray.init(address="auto", ignore_reinit_error=True)

    actors = ray.util.list_named_actors(all_namespaces=True)
    vllm_actors = [a for a in actors if 'vllm' in a['name'].lower()]

    if not vllm_actors:
        print("ERROR: No vLLM actor")
        return

    vllm_actor = ray.get_actor(vllm_actors[0]['name'], namespace=vllm_actors[0].get('namespace', 'tinker'))

    loaded_loras = ray.get(vllm_actor.list_loras.remote(), timeout=30)
    print(f"Loaded LoRAs: {loaded_loras}")

    lora_id = list(loaded_loras)[0]

    # Get LoRA top-K
    print("\n" + "=" * 60)
    print("vLLM with LoRA - ASSISTANT RESPONSE positions")
    print("=" * 60)

    topk_result = ray.get(vllm_actor.compute_prompt_topk_with_lora.remote(
        prompt_ids=input_tokens,
        request_id="asst_topk_001",
        lora_int_id=lora_id,
        k=10,
    ), timeout=120)

    for pos in positions_to_check:
        if pos >= len(topk_result):
            continue

        target = target_tokens[pos]
        target_str = tokenizer.decode([target])
        pos_topk = topk_result[pos]

        sorted_topk = sorted(pos_topk.items(), key=lambda x: x[1], reverse=True)
        argmax_id, argmax_lp = sorted_topk[0]
        argmax_str = tokenizer.decode([argmax_id])

        target_lp = pos_topk.get(target, -100.0)
        is_argmax = (argmax_id == target)

        print(f"\npos={pos}: target={target} ('{target_str}')")
        print(f"  Argmax: {argmax_id} ('{argmax_str}') logprob={argmax_lp:.4f}")
        print(f"  Target logprob: {target_lp:.4f}")
        print(f"  Target is argmax: {is_argmax}")

    # Get base model logprobs for comparison
    print("\n" + "=" * 60)
    print("BASE MODEL (no LoRA) - ASSISTANT RESPONSE positions")
    print("=" * 60)

    base_logprobs = ray.get(vllm_actor.compute_prompt_logprobs_base.remote(
        prompt_ids=input_tokens,
        request_id="asst_base_001",
    ), timeout=120)

    lora_logprobs = ray.get(vllm_actor.compute_prompt_logprobs_with_lora.remote(
        prompt_ids=input_tokens,
        request_id="asst_lora_001",
        lora_int_id=lora_id,
    ), timeout=120)

    print(f"\n{'Pos':<5} {'Target':<10} {'Base LP':<12} {'LoRA LP':<12} {'Diff':<10}")
    print("-" * 50)

    for pos in positions_to_check:
        if pos >= len(base_logprobs):
            continue
        target = target_tokens[pos]
        target_str = tokenizer.decode([target])[:8]
        base_lp = base_logprobs[pos]
        lora_lp = lora_logprobs[pos]
        diff = lora_lp - base_lp

        print(f"{pos:<5} {target_str:<10} {base_lp:<12.4f} {lora_lp:<12.4f} {diff:<+10.4f}")


if __name__ == "__main__":
    asyncio.run(main())
