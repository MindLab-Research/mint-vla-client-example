#!/usr/bin/env python3
"""Query top-K at corrupted positions using saved checkpoint data.

Uses the data saved by train_and_save_checkpoint.py
"""

import json
import numpy as np

MODEL_NAME = "moonshotai/Moonlight-16B-A3B-Instruct"


def main():
    from transformers import AutoTokenizer

    # Load saved data
    print("Loading saved data...")
    with open("/tmp/debug_checkpoint_data.json", "r") as f:
        data = json.load(f)

    checkpoint_path = data["checkpoint_path"]
    corrupted_positions = data["corrupted_positions"]
    megatron_fresh = np.array(data["megatron_fresh"])
    megatron_trained = np.array(data["megatron_trained"])
    vllm_trained = np.array(data["vllm_trained"])
    vllm_topk = data["vllm_topk"]
    target_tokens = data["target_tokens"]
    input_tokens = data["input_tokens"]

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Corrupted positions: {corrupted_positions}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Analyze each corrupted position
    print("\n" + "=" * 100)
    print("TOP-K ANALYSIS AT CORRUPTED POSITIONS")
    print("=" * 100)

    for pos in corrupted_positions:
        if pos >= len(target_tokens):
            continue

        target = target_tokens[pos]
        target_str = tokenizer.decode([target])

        mf = megatron_fresh[pos]
        mt = megatron_trained[pos]
        vt = vllm_trained[pos] if pos < len(vllm_trained) else float('nan')

        print(f"\n{'='*80}")
        print(f"POSITION {pos}: target={target} {repr(target_str)}")
        print(f"  Megatron fresh logprob:   {mf:.4f}")
        print(f"  Megatron trained logprob: {mt:.4f}")
        print(f"  vLLM trained logprob:     {vt:.4f}")
        print(f"  Diff (Meg - vLLM):        {mt - vt:+.4f}")

        # Show context
        ctx_start = max(0, pos - 2)
        ctx_end = min(len(input_tokens), pos + 3)
        ctx_tokens = input_tokens[ctx_start:ctx_end]
        ctx_str = tokenizer.decode(ctx_tokens)
        print(f"  Context: ...{repr(ctx_str)}...")

        # vLLM top-K
        if vllm_topk and pos < len(vllm_topk) and vllm_topk[pos]:
            pos_topk = {tok: lp for tok, lp in vllm_topk[pos]}
            sorted_topk = sorted(pos_topk.items(), key=lambda x: x[1], reverse=True)

            print(f"\n  vLLM Top-10 at position {pos}:")
            for rank, (tok_id, lp) in enumerate(sorted_topk[:10], 1):
                tok_str = tokenizer.decode([tok_id])
                marker = " <-- TARGET" if tok_id == target else ""
                print(f"    {rank:2d}. {tok_id:6d} {repr(tok_str):15s}: {lp:8.4f}{marker}")

            if target in pos_topk:
                rank = [t for t, _ in sorted_topk].index(target) + 1
                print(f"\n  Target {repr(target_str)} in vLLM top-10: YES, rank={rank}, logprob={pos_topk[target]:.4f}")
            else:
                print(f"\n  Target {repr(target_str)} in vLLM top-10: NO")

            # Show what vLLM thinks is #1
            argmax_tok, argmax_lp = sorted_topk[0]
            argmax_str = tokenizer.decode([argmax_tok])
            print(f"  vLLM argmax: {argmax_tok} {repr(argmax_str)} (logprob={argmax_lp:.4f})")
        else:
            print(f"\n  vLLM top-K: not available for this position")


if __name__ == "__main__":
    main()
