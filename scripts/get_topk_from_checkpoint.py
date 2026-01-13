#!/usr/bin/env python3
"""Get top-K tokens from vLLM using saved checkpoint."""

import asyncio
import json
import os

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker

MODEL_NAME = "moonshotai/Moonlight-16B-A3B-Instruct"
CHECKPOINT_PATH = "tinker://localhost/vePFS-Mindverse/share/code/tinker-server/checkpoints/d01393b8-7dcb-47b1-a95c-477e81b22498_0/debug_checkpoint"


async def main():
    from transformers import AutoTokenizer

    # Load saved data
    print("Loading saved data...")
    with open("/tmp/debug_checkpoint_data.json", "r") as f:
        data = json.load(f)

    input_tokens = data["input_tokens"]
    target_tokens = data["target_tokens"]
    corrupted_positions = data["corrupted_positions"]
    megatron_trained = data["megatron_trained"]
    vllm_trained = data["vllm_trained"]

    print(f"Input tokens: {len(input_tokens)}")
    print(f"Corrupted positions: {corrupted_positions}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    # Create sampling session using checkpoint
    print("\nCreating sampling session from checkpoint...")
    sampling_client = await service_client.create_sampling_client_async(
        base_model=MODEL_NAME,
        model_path=CHECKPOINT_PATH,
    )
    print(f"Sampling session created")

    # Wait for vLLM to load
    await asyncio.sleep(3)

    # Query with top-K
    print("\nQuerying vLLM for top-K...")
    sample_result = await sampling_client.sample_async(
        prompt=tinker.ModelInput.from_ints(input_tokens),
        num_samples=1,
        sampling_params=tinker.SamplingParams(max_tokens=1, temperature=0.0),
        include_prompt_logprobs=True,
        topk_prompt_logprobs=10,
    )

    vllm_logprobs = sample_result.prompt_logprobs
    vllm_topk = sample_result.topk_prompt_logprobs

    print(f"Got {len(vllm_logprobs) if vllm_logprobs else 0} logprobs")
    print(f"Got {len(vllm_topk) if vllm_topk else 0} topk entries")

    if not vllm_topk:
        print("ERROR: topk_prompt_logprobs is empty!")
        return

    # Analyze corrupted positions
    print("\n" + "=" * 100)
    print("TOP-K ANALYSIS AT CORRUPTED POSITIONS")
    print("=" * 100)

    for pos in corrupted_positions:
        if pos >= len(target_tokens):
            continue

        target = target_tokens[pos]
        target_str = tokenizer.decode([target])

        meg_t = megatron_trained[pos] if pos < len(megatron_trained) else float('nan')
        vllm_t = vllm_trained[pos] if pos < len(vllm_trained) else float('nan')

        print(f"\n{'='*80}")
        print(f"POSITION {pos}: target={target} {repr(target_str)}")
        print(f"  Megatron trained logprob: {meg_t:.4f}")
        print(f"  vLLM trained logprob:     {vllm_t:.4f}")
        print(f"  Diff (Meg - vLLM):        {meg_t - vllm_t:+.4f}")

        # Show context
        ctx_start = max(0, pos - 2)
        ctx_end = min(len(input_tokens), pos + 3)
        ctx_tokens = input_tokens[ctx_start:ctx_end]
        ctx_str = tokenizer.decode(ctx_tokens)
        print(f"  Context: ...{repr(ctx_str)}...")

        # vLLM top-K
        if pos < len(vllm_topk) and vllm_topk[pos]:
            pos_topk = vllm_topk[pos]
            sorted_topk = sorted(pos_topk.items(), key=lambda x: x[1], reverse=True)

            print(f"\n  vLLM Top-10 at position {pos}:")
            for rank, (tok_id, lp) in enumerate(sorted_topk[:10], 1):
                tok_id = int(tok_id)  # JSON keys are strings
                tok_str = tokenizer.decode([tok_id])
                marker = " <-- TARGET" if tok_id == target else ""
                print(f"    {rank:2d}. {tok_id:6d} {repr(tok_str):15s}: {lp:8.4f}{marker}")

            target_str_key = str(target)
            if target_str_key in pos_topk:
                rank = [int(t) for t, _ in sorted_topk].index(target) + 1
                print(f"\n  Target {repr(target_str)} in vLLM top-10: YES, rank={rank}, logprob={pos_topk[target_str_key]:.4f}")
            else:
                print(f"\n  Target {repr(target_str)} in vLLM top-10: NO")

            # Show what vLLM thinks is #1
            argmax_tok, argmax_lp = sorted_topk[0]
            argmax_tok = int(argmax_tok)
            argmax_str = tokenizer.decode([argmax_tok])
            print(f"  vLLM argmax: {argmax_tok} {repr(argmax_str)} (logprob={argmax_lp:.4f})")
        else:
            print(f"\n  vLLM top-K: not available for this position")


if __name__ == "__main__":
    asyncio.run(main())
