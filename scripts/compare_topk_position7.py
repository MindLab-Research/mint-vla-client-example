#!/usr/bin/env python3
"""Compare top-K tokens between vLLM and Megatron at position 7.

Run on volcano: python3 /root/tinker_project/tinker-server/scripts/compare_topk_position7.py
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

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]

    print(f"\nInput: {len(input_tokens)} tokens")
    print(f"Position 7: target={target_tokens[7]} ('{tokenizer.decode([target_tokens[7]])}')")

    print("\nConnecting to Ray...")
    ray.init(address="auto", ignore_reinit_error=True)

    # Find vLLM actor
    actors = ray.util.list_named_actors(all_namespaces=True)
    vllm_actors = [a for a in actors if 'vllm' in a['name'].lower()]

    if not vllm_actors:
        print("ERROR: No vLLM actor found")
        return

    print(f"Found vLLM actor: {vllm_actors[0]['name']}")
    vllm_actor = ray.get_actor(vllm_actors[0]['name'], namespace=vllm_actors[0].get('namespace', 'tinker'))

    # Call compute_prompt_topk
    print("\n" + "=" * 60)
    print("vLLM Top-K at position 7")
    print("=" * 60)

    try:
        import asyncio

        async def get_topk():
            return await vllm_actor.compute_prompt_topk.remote(
                prompt_ids=input_tokens,
                request_id="topk_pos7_001",
                k=10,
            )

        topk_result = ray.get(vllm_actor.compute_prompt_topk.remote(
            prompt_ids=input_tokens,
            request_id="topk_pos7_001",
            k=10,
        ), timeout=120)

        print(f"Returned {len(topk_result)} positions")

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

            # Check if target is in top-K
            if target_tokens[7] in pos7_topk:
                print(f"\n  Target token logprob: {pos7_topk[target_tokens[7]]:.4f}")
            else:
                print(f"\n  WARNING: Target token NOT in top-{len(pos7_topk)}")
        else:
            print(f"ERROR: Position 7 not available")

    except Exception as e:
        print(f"compute_prompt_topk failed: {e}")
        import traceback
        traceback.print_exc()

    # Also check Megatron diagnostic log
    print("\n" + "=" * 60)
    print("Megatron Diagnostic (from log)")
    print("=" * 60)

    try:
        with open("/vePFS-Mindverse/share/code/raw_logit_diag.log", "r") as f:
            lines = f.readlines()
            pos7_lines = [l for l in lines if "pos=7" in l]
            if pos7_lines:
                print("Latest pos=7 entries:")
                for line in pos7_lines[-5:]:
                    print(line.strip())
            else:
                print("No pos=7 entries in diagnostic log")
    except FileNotFoundError:
        print("Diagnostic log not found")


if __name__ == "__main__":
    main()
