#!/usr/bin/env python3
"""Get top-K tokens and raw logits at corrupted positions from both Megatron and vLLM."""

import asyncio
import os

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker
import torch
import numpy as np

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

# Corrupted positions from the comparison (Megatron trained shows very negative values)
CORRUPTED_POSITIONS = [10, 13, 17, 21, 22, 23, 25, 30]


async def main():
    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]

    print(f"Sequence: {len(input_tokens)} tokens")

    # Show what the corrupted positions are
    print("\n" + "=" * 70)
    print("CORRUPTED POSITIONS INFO")
    print("=" * 70)
    for pos in CORRUPTED_POSITIONS:
        input_tok = tokens[pos]
        target_tok = target_tokens[pos]
        print(f"pos {pos:2d}: input={input_tok:6d} {repr(tokenizer.decode([input_tok])):15s} -> target={target_tok:6d} {repr(tokenizer.decode([target_tok])):15s}")

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    # Create training client and train
    print("\n" + "=" * 70)
    print("PHASE 1: Create and train Megatron")
    print("=" * 70)

    client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)

    mask = [1.0] * len(input_tokens)
    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.ones(len(input_tokens), dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.zeros(len(input_tokens), dtype=torch.float32)),
        }
    )

    # Train 10 steps
    for step in range(10):
        fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
        await fwd_bwd.result_async()
        await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()

    print("Training complete")

    # Get Megatron logprobs
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    megatron_logprobs = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    print("\n" + "=" * 70)
    print("PHASE 2: Export to vLLM and get top-K")
    print("=" * 70)

    sampling_client = await client.save_weights_and_get_sampling_client_async()
    print("Exported to vLLM")
    await asyncio.sleep(2)

    # Get vLLM logprobs with top-K
    sample_result = await sampling_client.sample_async(
        prompt=tinker.ModelInput.from_ints(input_tokens),
        num_samples=1,
        sampling_params=tinker.SamplingParams(max_tokens=1, temperature=0.0),
        include_prompt_logprobs=True,
        topk_prompt_logprobs=10,
    )

    vllm_logprobs = np.array([lp if lp is not None else -100.0 for lp in sample_result.prompt_logprobs])
    vllm_topk = sample_result.topk_prompt_logprobs

    print("\n" + "=" * 100)
    print("COMPARISON AT CORRUPTED POSITIONS")
    print("=" * 100)

    for pos in CORRUPTED_POSITIONS:
        if pos >= len(megatron_logprobs):
            continue

        target = target_tokens[pos]
        target_str = tokenizer.decode([target])

        meg_lp = megatron_logprobs[pos]
        vllm_lp = vllm_logprobs[pos] if pos < len(vllm_logprobs) else float('nan')

        print(f"\n{'='*80}")
        print(f"POSITION {pos}: target={target} {repr(target_str)}")
        print(f"  Megatron logprob: {meg_lp:.4f}")
        print(f"  vLLM logprob:     {vllm_lp:.4f}")
        print(f"  Diff:             {meg_lp - vllm_lp:+.4f}")

        # vLLM top-K from sample API
        if vllm_topk and pos < len(vllm_topk) and vllm_topk[pos]:
            pos_topk = {tok: lp for tok, lp in vllm_topk[pos]}
            sorted_topk = sorted(pos_topk.items(), key=lambda x: x[1], reverse=True)

            print(f"\n  vLLM Top-10 at pos {pos}:")
            for rank, (tok_id, lp) in enumerate(sorted_topk[:10], 1):
                tok_str = tokenizer.decode([tok_id])
                marker = " <-- TARGET" if tok_id == target else ""
                print(f"    {rank:2d}. {tok_id:6d} {repr(tok_str):15s}: {lp:8.4f}{marker}")

            # Check if target is in top-K
            if target in pos_topk:
                print(f"\n  Target in vLLM top-10: YES, logprob={pos_topk[target]:.4f}")
            else:
                print(f"\n  Target in vLLM top-10: NO")
        else:
            print(f"\n  vLLM top-K: not available")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Pos':<4} {'Target':<15} {'Meg LP':<12} {'vLLM LP':<12} {'Diff':<12}")
    print("-" * 60)
    for pos in CORRUPTED_POSITIONS:
        if pos >= len(megatron_logprobs):
            continue
        target = target_tokens[pos]
        target_str = repr(tokenizer.decode([target]))[:12]
        meg_lp = megatron_logprobs[pos]
        vllm_lp = vllm_logprobs[pos] if pos < len(vllm_logprobs) else float('nan')
        diff = meg_lp - vllm_lp
        print(f"{pos:<4} {target_str:<15} {meg_lp:<12.4f} {vllm_lp:<12.4f} {diff:<+12.4f}")


if __name__ == "__main__":
    asyncio.run(main())
