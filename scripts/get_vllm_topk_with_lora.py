#!/usr/bin/env python3
"""Get vLLM top-K with LoRA loaded.

Run on volcano: python3 /root/tinker_project/tinker-server/scripts/get_vllm_topk_with_lora.py
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

    print(f"\nPosition 7: target={target_tokens[7]} ('{tokenizer.decode([target_tokens[7]])}')")

    print("\nConnecting to Ray...")
    ray.init(address="auto", ignore_reinit_error=True)

    # Find vLLM actor
    actors = ray.util.list_named_actors(all_namespaces=True)
    vllm_actors = [a for a in actors if 'vllm' in a['name'].lower()]

    if not vllm_actors:
        print("ERROR: No vLLM actor found")
        return

    vllm_actor = ray.get_actor(vllm_actors[0]['name'], namespace=vllm_actors[0].get('namespace', 'tinker'))

    # Check if LoRA is loaded
    try:
        loras = await vllm_actor.engine.list_loras.remote()
        print(f"\nLoaded LoRAs: {loras}")
    except Exception as e:
        print(f"Could not list LoRAs: {e}")

    # Call compute_prompt_topk
    print("\n" + "=" * 60)
    print("vLLM Top-K at position 7")
    print("=" * 60)

    try:
        topk_result = ray.get(vllm_actor.compute_prompt_topk.remote(
            prompt_ids=input_tokens,
            request_id="topk_lora_001",
            k=10,
        ), timeout=120)

        if len(topk_result) > 7:
            pos7_topk = topk_result[7]
            print(f"\nTarget token: {target_tokens[7]} ('{tokenizer.decode([target_tokens[7]])}')")
            print()

            # Sort by logprob descending
            sorted_topk = sorted(pos7_topk.items(), key=lambda x: x[1], reverse=True)
            for rank, (tok_id, logprob) in enumerate(sorted_topk[:10], 1):
                tok_str = tokenizer.decode([tok_id])
                marker = " <-- TARGET" if tok_id == target_tokens[7] else ""
                print(f"  {rank:2d}. token={tok_id:6d} ({repr(tok_str):15s}): logprob={logprob:8.4f}{marker}")

            # Check target
            if target_tokens[7] in pos7_topk:
                print(f"\n  Target token logprob: {pos7_topk[target_tokens[7]]:.4f}")
            else:
                print(f"\n  WARNING: Target NOT in top-{len(pos7_topk)}")

    except Exception as e:
        print(f"compute_prompt_topk failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
