#!/usr/bin/env python3
"""Check multiple positions in vLLM to compare with Megatron.

Position 7: target 3922 'Count' - Megatron says argmax is space (220)
Position 8: target 2291 - Megatron says target IS argmax
Position 23: target 27 - Megatron says target IS argmax

Run on volcano: python3 /root/tinker_project/tinker-server/scripts/check_vllm_positions.py
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


async def main():
    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]

    print(f"\nTotal tokens: {len(tokens)}")
    print(f"Input tokens: {len(input_tokens)}")

    # Print token sequence around positions of interest
    print("\n" + "=" * 60)
    print("Token sequence:")
    print("=" * 60)
    for i in range(min(30, len(tokens))):
        tok_str = tokenizer.decode([tokens[i]])
        print(f"  pos={i}: token={tokens[i]:6d} ({repr(tok_str):15s})")

    print("\nConnecting to Ray...")
    ray.init(address="auto", ignore_reinit_error=True)

    # Find vLLM actor
    actors = ray.util.list_named_actors(all_namespaces=True)
    vllm_actors = [a for a in actors if 'vllm' in a['name'].lower()]

    if not vllm_actors:
        print("ERROR: No vLLM actor found")
        return

    vllm_actor = ray.get_actor(vllm_actors[0]['name'], namespace=vllm_actors[0].get('namespace', 'tinker'))

    # Get loaded LoRAs
    loaded_loras = ray.get(vllm_actor.list_loras.remote(), timeout=30)
    print(f"\nLoaded LoRAs: {loaded_loras}")

    if not loaded_loras:
        print("ERROR: No LoRA loaded")
        return

    lora_id = list(loaded_loras)[0]

    # Get full top-K for all positions
    print("\n" + "=" * 60)
    print("vLLM Top-K at multiple positions (with LoRA)")
    print("=" * 60)

    topk_result = ray.get(vllm_actor.compute_prompt_topk_with_lora.remote(
        prompt_ids=input_tokens,
        request_id="positions_check_001",
        lora_int_id=lora_id,
        k=10,
    ), timeout=120)

    # Check positions 7, 8, 23
    positions_to_check = [7, 8, 22]  # Note: topk_result[i] predicts target_tokens[i]

    for pos in positions_to_check:
        if pos >= len(topk_result):
            print(f"\nPosition {pos}: not available")
            continue

        pos_topk = topk_result[pos]
        target = target_tokens[pos]
        target_str = tokenizer.decode([target])

        print(f"\n--- Position {pos} (predicting target={target} '{target_str}') ---")

        sorted_topk = sorted(pos_topk.items(), key=lambda x: x[1], reverse=True)
        for rank, (tok_id, logprob) in enumerate(sorted_topk[:5], 1):
            tok_str = tokenizer.decode([tok_id])
            marker = " <-- TARGET" if tok_id == target else ""
            print(f"  {rank}. token={tok_id:6d} ({repr(tok_str):15s}): logprob={logprob:8.4f}{marker}")

        if target in pos_topk:
            print(f"  Target logprob: {pos_topk[target]:.4f}")
        else:
            print(f"  WARNING: Target not in top-{len(pos_topk)}")

    # Also check base model for comparison
    print("\n" + "=" * 60)
    print("Base model (no LoRA) logprobs")
    print("=" * 60)

    base_logprobs = ray.get(vllm_actor.compute_prompt_logprobs_base.remote(
        prompt_ids=input_tokens,
        request_id="positions_base_001",
    ), timeout=120)

    for pos in positions_to_check:
        if pos >= len(base_logprobs):
            continue
        target = target_tokens[pos]
        target_str = tokenizer.decode([target])
        print(f"Position {pos} (target={target} '{target_str}'): base logprob = {base_logprobs[pos]:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
