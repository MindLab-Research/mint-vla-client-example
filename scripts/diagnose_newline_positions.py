#!/usr/bin/env python3
"""Diagnose what happens to ALL newline positions during training.

Hypothesis: The 10 positions where '\n' should predict digits overwhelm
position 7 where '\n' should predict 'Count'.
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

    # Find all newline positions
    newline_positions = []
    for i, (inp, tgt) in enumerate(zip(input_tokens, target_tokens)):
        if tokenizer.decode([inp]) == '\n':
            tgt_str = tokenizer.decode([tgt])
            newline_positions.append((i, tgt, tgt_str))

    print(f"\nFound {len(newline_positions)} newline positions:")
    for pos, tgt_id, tgt_str in newline_positions:
        print(f"  pos={pos:2d}: target={tgt_id:6d} = {repr(tgt_str)}")

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    print("\n" + "=" * 70)
    print("Creating fresh training client...")
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

    # Step 1: Fresh forward
    print("\n" + "=" * 70)
    print("STEP 1: Fresh LoRA forward")
    print("=" * 70)

    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    fresh_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    print("\nLogprobs at ALL newline positions (fresh):")
    for pos, tgt_id, tgt_str in newline_positions:
        print(f"  pos={pos:2d}: logprob={fresh_lp[pos]:8.4f} (target={repr(tgt_str):10s})")

    # Step 2: Forward_backward + optim_step (one training step)
    print("\n" + "=" * 70)
    print("STEP 2: Train one step")
    print("=" * 70)

    fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
    await fwd_bwd.result_async()
    await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()

    # Step 3: Forward after one update
    print("\n" + "=" * 70)
    print("STEP 3: Forward after one training step")
    print("=" * 70)

    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    step1_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    print("\nLogprobs at ALL newline positions (after 1 step):")
    print(f"{'pos':>4s} {'fresh':>10s} {'step1':>10s} {'delta':>10s} {'direction':>10s} {'target':>12s}")
    print("-" * 70)

    for pos, tgt_id, tgt_str in newline_positions:
        delta = step1_lp[pos] - fresh_lp[pos]
        direction = "BETTER" if delta > 0 else "WORSE" if delta < 0 else "SAME"
        print(f"{pos:4d} {fresh_lp[pos]:10.4f} {step1_lp[pos]:10.4f} {delta:+10.4f} {direction:>10s} {repr(tgt_str):>12s}")

    # Continue training for a few more steps
    print("\n" + "=" * 70)
    print("STEP 4: Continue training 9 more steps")
    print("=" * 70)

    for step in range(9):
        fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
        await fwd_bwd.result_async()
        await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()

    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    step10_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    print("\nLogprobs at ALL newline positions (after 10 steps):")
    print(f"{'pos':>4s} {'fresh':>10s} {'step10':>10s} {'delta':>10s} {'direction':>10s} {'target':>12s}")
    print("-" * 70)

    for pos, tgt_id, tgt_str in newline_positions:
        delta = step10_lp[pos] - fresh_lp[pos]
        direction = "BETTER" if delta > 0 else "WORSE" if delta < 0 else "SAME"
        print(f"{pos:4d} {fresh_lp[pos]:10.4f} {step10_lp[pos]:10.4f} {delta:+10.4f} {direction:>10s} {repr(tgt_str):>12s}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Count targets by type
    digit_positions = [p for p, _, s in newline_positions if len(s.strip()) == 1 and s.strip().isdigit()]
    non_digit_positions = [p for p, _, s in newline_positions if not (len(s.strip()) == 1 and s.strip().isdigit())]

    print(f"\nDigit targets ({len(digit_positions)} positions): {digit_positions}")
    print(f"Non-digit targets ({len(non_digit_positions)} positions): {non_digit_positions}")

    digit_deltas = [step10_lp[p] - fresh_lp[p] for p in digit_positions]
    non_digit_deltas = [step10_lp[p] - fresh_lp[p] for p in non_digit_positions]

    print(f"\nAverage delta for digit targets: {np.mean(digit_deltas):+.4f}")
    print(f"Average delta for non-digit targets: {np.mean(non_digit_deltas):+.4f}")

    if np.mean(digit_deltas) > 0 and np.mean(non_digit_deltas) < 0:
        print("\nHYPOTHESIS CONFIRMED: Digit positions get BETTER, non-digit positions get WORSE")
        print("The majority gradient (10 digit positions) overwhelms the minority (2 non-digit positions)")
    else:
        print("\nHypothesis NOT confirmed. Need further investigation.")


if __name__ == "__main__":
    asyncio.run(main())
