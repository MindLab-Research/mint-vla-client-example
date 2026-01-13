#!/usr/bin/env python3
"""Diagnose what happens during a single training step.

Trace:
1. Fresh LoRA forward - get baseline logits for position 7
2. Forward_backward - get gradients
3. Optim_step - update weights
4. Forward again - see how logits changed

Goal: Understand why position 7 gets WORSE while positions 8, 23 get BETTER.
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

    print(f"Sequence: {len(input_tokens)} tokens")
    print()

    # Show position 7 details
    print("Position 7 details:")
    print(f"  Input:  token {input_tokens[7]:5d} = {repr(tokenizer.decode([input_tokens[7]]))}")
    print(f"  Target: token {target_tokens[7]:5d} = {repr(tokenizer.decode([target_tokens[7]]))}")
    print()
    print("Position 8 details:")
    print(f"  Input:  token {input_tokens[8]:5d} = {repr(tokenizer.decode([input_tokens[8]]))}")
    print(f"  Target: token {target_tokens[8]:5d} = {repr(tokenizer.decode([target_tokens[8]]))}")
    print()

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    # Create training client
    print("=" * 70)
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
    print()
    print("=" * 70)
    print("STEP 1: Fresh LoRA forward")
    print("=" * 70)

    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    fresh_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    print("Logprobs at key positions:")
    for pos in [7, 8, 23]:
        target_str = tokenizer.decode([target_tokens[pos]])
        print(f"  pos={pos:2d}: logprob={fresh_lp[pos]:8.4f} (target={repr(target_str)})")

    # Step 2: Forward_backward (one step)
    print()
    print("=" * 70)
    print("STEP 2: Forward_backward (computes loss + gradients)")
    print("=" * 70)

    fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
    result2 = await fwd_bwd.result_async()

    # Extract metrics
    loss = result2.metrics.get("loss:mean", 0.0)
    ratio = result2.metrics.get("ratio:mean", 0.0)
    print(f"  Loss: {loss:.4f}")
    print(f"  Ratio: {ratio:.4f}")

    # Get logprobs from forward_backward
    fwdbwd_lp = np.array(result2.loss_fn_outputs[0]["logprobs"].to_numpy())
    print("Logprobs (should match fresh, since weights not updated yet):")
    for pos in [7, 8, 23]:
        diff = fwdbwd_lp[pos] - fresh_lp[pos]
        print(f"  pos={pos:2d}: logprob={fwdbwd_lp[pos]:8.4f} (diff from fresh: {diff:+.6f})")

    # Step 3: Optim step
    print()
    print("=" * 70)
    print("STEP 3: Optimizer step (updates weights)")
    print("=" * 70)

    await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()
    print("  Weights updated with lr=1e-3")

    # Step 4: Forward after one update
    print()
    print("=" * 70)
    print("STEP 4: Forward after one training step")
    print("=" * 70)

    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result3 = await fwd.result_async()
    step1_lp = np.array(result3.loss_fn_outputs[0]["logprobs"].to_numpy())

    print("Logprobs after one step:")
    for pos in [7, 8, 23]:
        target_str = tokenizer.decode([target_tokens[pos]])[:8]
        delta = step1_lp[pos] - fresh_lp[pos]
        direction = "BETTER" if delta > 0 else "WORSE" if delta < 0 else "SAME"
        print(f"  pos={pos:2d}: {step1_lp[pos]:8.4f} (delta={delta:+.4f}, {direction}, target={repr(target_str)})")

    # Step 5: Continue training a few more steps
    print()
    print("=" * 70)
    print("STEP 5: Continue training 4 more steps")
    print("=" * 70)

    for step in range(4):
        fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
        await fwd_bwd.result_async()
        await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()

        fwd = await client.forward_async([datum], loss_fn="importance_sampling")
        result = await fwd.result_async()
        step_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

        delta7 = step_lp[7] - fresh_lp[7]
        delta8 = step_lp[8] - fresh_lp[8]
        delta23 = step_lp[23] - fresh_lp[23]

        print(f"  Step {step+2}: pos7={step_lp[7]:7.3f} (Δ={delta7:+.3f}), "
              f"pos8={step_lp[8]:7.3f} (Δ={delta8:+.3f}), "
              f"pos23={step_lp[23]:7.3f} (Δ={delta23:+.3f})")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("If position 7 gets WORSE while 8/23 get BETTER, the bug is confirmed.")
    print("The question is: why does training push position 7 in the wrong direction?")
    print()
    print("Possible causes:")
    print("1. MoE routing: Position 7 uses different experts than 8/23")
    print("2. Gradient interference: Other positions' gradients overwhelm position 7")
    print("3. Label/target issue: Something wrong with how position 7's target is handled")


if __name__ == "__main__":
    asyncio.run(main())
