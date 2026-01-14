#!/usr/bin/env python3
"""Diagnose discrepancy between vLLM sample API and compute_prompt_topk_with_lora.

Sample API gives -0.005 but compute_prompt_topk_with_lora gives -8.09 for same LoRA.
This script investigates why.

Run on volcano: python3 /root/tinker_project/tinker-server/scripts/diagnose_vllm_lora_discrepancy.py
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

    # Check loaded LoRAs
    print("\n" + "=" * 60)
    print("Checking loaded LoRAs...")
    print("=" * 60)

    try:
        loaded_loras = ray.get(vllm_actor.list_loras.remote(), timeout=30)
        print(f"Loaded LoRAs: {loaded_loras}")
    except Exception as e:
        print(f"Could not list LoRAs: {e}")
        loaded_loras = set()

    if not loaded_loras:
        print("ERROR: No LoRA loaded in vLLM")
        return

    lora_id = list(loaded_loras)[0]
    print(f"Using lora_id={lora_id}")

    # Method 1: compute_prompt_topk_with_lora
    print("\n" + "=" * 60)
    print("Method 1: compute_prompt_topk_with_lora")
    print("=" * 60)

    try:
        topk_result = ray.get(vllm_actor.compute_prompt_topk_with_lora.remote(
            prompt_ids=input_tokens,
            request_id="diag_topk_001",
            lora_int_id=lora_id,
            k=10,
        ), timeout=120)

        if len(topk_result) > 7:
            pos7_topk = topk_result[7]
            print(f"\nTarget token: {target_tokens[7]} ('{tokenizer.decode([target_tokens[7]])}')")
            print()

            sorted_topk = sorted(pos7_topk.items(), key=lambda x: x[1], reverse=True)
            for rank, (tok_id, logprob) in enumerate(sorted_topk[:10], 1):
                tok_str = tokenizer.decode([tok_id])
                marker = " <-- TARGET" if tok_id == target_tokens[7] else ""
                print(f"  {rank:2d}. token={tok_id:6d} ({repr(tok_str):15s}): logprob={logprob:8.4f}{marker}")

            if target_tokens[7] in pos7_topk:
                print(f"\n  Target token logprob: {pos7_topk[target_tokens[7]]:.4f}")
            else:
                print(f"\n  WARNING: Target NOT in top-{len(pos7_topk)}")

    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()

    # Method 2: compute_prompt_logprobs_with_lora (scalar logprob)
    print("\n" + "=" * 60)
    print("Method 2: compute_prompt_logprobs_with_lora")
    print("=" * 60)

    try:
        logprobs_result = ray.get(vllm_actor.compute_prompt_logprobs_with_lora.remote(
            prompt_ids=input_tokens,
            request_id="diag_logprobs_001",
            lora_int_id=lora_id,
        ), timeout=120)

        if len(logprobs_result) > 7:
            print(f"\nPosition 7 logprob: {logprobs_result[7]:.4f}")
            print(f"(This is P(target_tokens[7] | input_tokens[0:8]))")

    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()

    # Method 3: generate_with_lora (single token generation to get logprob)
    print("\n" + "=" * 60)
    print("Method 3: generate_with_lora (1 token)")
    print("=" * 60)

    # Use only the first 8 tokens as prompt (positions 0-7)
    # So we're asking: given tokens[0:8], what's the logprob of generating tokens[8]?
    prompt_for_gen = input_tokens[:8]  # tokens at positions 0-7

    try:
        gen_result = ray.get(vllm_actor.generate_with_lora.remote(
            prompt_ids=prompt_for_gen,
            request_id="diag_gen_001",
            lora_int_id=lora_id,
            max_tokens=1,
            temperature=1.0,
            logprobs=True,
        ), timeout=120)

        print(f"\nGenerated token: {gen_result['token_ids']}")
        if gen_result['logprobs']:
            print(f"Generated token logprob: {gen_result['logprobs'][0]:.4f}")
        print(f"Expected token at position 8: {target_tokens[7]} ('{tokenizer.decode([target_tokens[7]])}')")

    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()

    # Method 4: Compare base model vs LoRA
    print("\n" + "=" * 60)
    print("Method 4: Base model (no LoRA) comparison")
    print("=" * 60)

    try:
        base_logprobs = ray.get(vllm_actor.compute_prompt_logprobs_base.remote(
            prompt_ids=input_tokens,
            request_id="diag_base_001",
        ), timeout=120)

        if len(base_logprobs) > 7:
            print(f"\nBase model position 7 logprob: {base_logprobs[7]:.4f}")

    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Expected: All methods with same LoRA should give similar logprobs.
If they differ significantly, there's a bug in LoRA application.

Potential causes of discrepancy:
1. Different LoRA paths being used
2. LoRA not applied during prompt_logprobs computation
3. _lora_paths dict missing entry for this lora_id
4. Engine internal state mismatch
""")


if __name__ == "__main__":
    asyncio.run(main())
