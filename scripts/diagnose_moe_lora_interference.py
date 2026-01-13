#!/usr/bin/env python3
"""Diagnose MoE LoRA interference between positions.

Hypothesis: Position 7 gets wrong predictions because:
1. Shared LoRA in Megatron applies same weights to all tokens after expert dispatch
2. Training on positions 14, 49 (target=16) creates strong bias for token 16
3. This bias affects position 7 even though its target is 3922

Test plan:
1. Get expert routing for each position (which experts are selected)
2. Check if positions 7, 14, 49 share experts
3. Print top-K predictions at position 7 before and after training
"""

import asyncio
import os
import sys

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


async def main():
    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    mask = [1.0] * len(input_tokens)

    print(f"Sequence: {len(tokens)} tokens, input: {len(input_tokens)}")

    # Key positions to analyze
    KEY_POSITIONS = [7, 14, 49]  # 7: target=Count, 14&49: target=1

    print("\n" + "=" * 70)
    print("TOKEN ANALYSIS")
    print("=" * 70)
    for pos in KEY_POSITIONS:
        if pos < len(input_tokens):
            inp = tokenizer.decode([input_tokens[pos]])
            tgt = tokenizer.decode([target_tokens[pos]])
            print(f"  pos={pos:2d}: input={repr(inp):10s} -> target={repr(tgt):12s} (id={target_tokens[pos]})")

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    print("\n" + "=" * 70)
    print("PHASE 1: Create LoRA and get fresh predictions")
    print("=" * 70)

    client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.ones(len(input_tokens), dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.zeros(len(input_tokens), dtype=torch.float32)),
        }
    )

    # Get fresh baseline with return_logits=True to analyze top-K
    fwd = await client.forward_async([datum], loss_fn="importance_sampling", return_logits=True)
    result = await fwd.result_async()

    fresh_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    print("\nFresh Megatron logprobs at key positions:")
    for pos in KEY_POSITIONS:
        if pos < len(fresh_lp):
            tgt = target_tokens[pos]
            tgt_str = tokenizer.decode([tgt])
            print(f"  pos={pos:2d}: lp={fresh_lp[pos]:8.4f} (target={tgt}={repr(tgt_str)})")

    # Check if logits are available
    if "logits" in result.loss_fn_outputs[0]:
        logits = result.loss_fn_outputs[0]["logits"].to_numpy()
        print(f"\nLogits shape: {logits.shape}")

        # Show top-5 at position 7
        print("\nTop-5 predictions at position 7 (FRESH):")
        pos7_logits = logits[7]
        top5_idx = np.argsort(pos7_logits)[-5:][::-1]
        for i, idx in enumerate(top5_idx):
            tok_str = tokenizer.decode([idx])
            print(f"  {i+1}. token={idx:6d} ({repr(tok_str):12s}): logit={pos7_logits[idx]:.4f}")

        print(f"\n  Target token 3922 ('Count'): logit={pos7_logits[3922]:.4f}")
        print(f"  Token 16 ('1'): logit={pos7_logits[16]:.4f}")

    print("\n" + "=" * 70)
    print("PHASE 2: Train 10 steps with verbose logging")
    print("=" * 70)

    for step in range(10):
        fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
        await fwd_bwd.result_async()
        await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()

        if (step + 1) % 5 == 0 or step == 0:
            fwd = await client.forward_async([datum], loss_fn="importance_sampling", return_logits=True)
            result = await fwd.result_async()
            train_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

            logit_info = ""
            if "logits" in result.loss_fn_outputs[0]:
                logits = result.loss_fn_outputs[0]["logits"].to_numpy()
                pos7_logits = logits[7]
                top_idx = np.argmax(pos7_logits)
                top_str = tokenizer.decode([top_idx])
                target_logit = pos7_logits[3922]
                top_logit = pos7_logits[top_idx]
                logit_info = f", pos7_top={top_idx}({repr(top_str)}):{top_logit:.2f}, target:{target_logit:.2f}"

            print(f"  Step {step+1}: pos7={train_lp[7]:.4f}, pos14={train_lp[14]:.4f}, pos49={train_lp[49]:.4f}{logit_info}")

    # Final analysis
    print("\n" + "=" * 70)
    print("PHASE 3: Final analysis")
    print("=" * 70)

    fwd = await client.forward_async([datum], loss_fn="importance_sampling", return_logits=True)
    result = await fwd.result_async()
    trained_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    print("\nFinal Megatron logprobs at key positions:")
    for pos in KEY_POSITIONS:
        if pos < len(trained_lp):
            tgt = target_tokens[pos]
            tgt_str = tokenizer.decode([tgt])
            delta = trained_lp[pos] - fresh_lp[pos]
            print(f"  pos={pos:2d}: lp={trained_lp[pos]:8.4f} (delta={delta:+8.4f}, target={tgt}={repr(tgt_str)})")

    if "logits" in result.loss_fn_outputs[0]:
        logits = result.loss_fn_outputs[0]["logits"].to_numpy()

        print("\nTop-5 predictions at position 7 (AFTER TRAINING):")
        pos7_logits = logits[7]
        top5_idx = np.argsort(pos7_logits)[-5:][::-1]
        for i, idx in enumerate(top5_idx):
            tok_str = tokenizer.decode([idx])
            print(f"  {i+1}. token={idx:6d} ({repr(tok_str):12s}): logit={pos7_logits[idx]:.4f}")

        print(f"\n  Target token 3922 ('Count'): logit={pos7_logits[3922]:.4f}")
        print(f"  Token 16 ('1'): logit={pos7_logits[16]:.4f}")

        # Check other key positions
        print("\nTop-1 predictions at other positions:")
        for pos in [14, 49]:
            if pos < len(logits):
                pos_logits = logits[pos]
                top_idx = np.argmax(pos_logits)
                top_str = tokenizer.decode([top_idx])
                tgt = target_tokens[pos]
                tgt_logit = pos_logits[tgt]
                print(f"  pos={pos:2d}: top={top_idx:6d} ({repr(top_str):12s}), target {tgt} logit={tgt_logit:.4f}")

    # Compare with vLLM
    print("\n" + "=" * 70)
    print("PHASE 4: Export to vLLM and compare")
    print("=" * 70)

    sampling_client = await client.save_weights_and_get_sampling_client_async()

    # vLLM expects full sequence for logprob computation
    prompt = tinker.ModelInput.from_ints(tokens)
    vllm_lp = await sampling_client.compute_logprobs_async(prompt)

    print("\nvLLM logprobs at key positions (aligned: vLLM[i+1] vs Megatron[i]):")
    for pos in KEY_POSITIONS:
        vllm_pos = pos + 1  # Alignment correction
        if vllm_pos < len(vllm_lp) and vllm_lp[vllm_pos] is not None:
            tgt = target_tokens[pos]
            tgt_str = tokenizer.decode([tgt])
            diff = trained_lp[pos] - vllm_lp[vllm_pos]
            print(f"  pos={pos:2d}: Megatron={trained_lp[pos]:8.4f}, vLLM={vllm_lp[vllm_pos]:8.4f}, diff={diff:+8.4f} (target={repr(tgt_str)})")


if __name__ == "__main__":
    asyncio.run(main())
