#!/usr/bin/env python3
"""Debug: test if train_mode/eval_mode transition affects forward pass.

Hypothesis: The transition between train_mode() and eval_mode() contexts
causes some state to persist that corrupts the forward pass.

Test plan:
1. Fresh model forward (baseline)
2. Enter train_mode, do nothing, exit train_mode
3. Forward again (check if corrupted just by mode transition)
4. Enter train_mode, do one forward_backward, exit
5. Forward again (check if training corrupted it)
"""

import os
import sys

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import asyncio
import tinker
import torch
import numpy as np

MODEL_NAME = "moonshotai/Moonlight-16B-A3B-Instruct"

TEST_TEXT = "<|im_start|>user\nCount down from 10 to 1, one number per line.<|im_end|>\n<|im_start|>assistant\n10\n9\n8\n7\n6\n5\n4\n3\n2\n1<|im_end|>"


async def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    mask = [1.0] * len(input_tokens)

    print("=" * 70)
    print("HYPOTHESIS TEST: Mode transition causing forward corruption?")
    print("=" * 70)
    print()

    # Print position 7 details
    print(f"Position 7: input=token {input_tokens[7]} ({repr(tokenizer.decode([input_tokens[7]]))})")
    print(f"            target=token {target_tokens[7]} ({repr(tokenizer.decode([target_tokens[7]]))})")
    print()

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])
    client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.ones(len(input_tokens), dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.zeros(len(input_tokens), dtype=torch.float32)),
        },
    )

    # Test 1: Fresh forward (baseline)
    print("Test 1: Fresh model forward")
    fwd1 = await client.forward_async([datum], loss_fn="importance_sampling")
    res1 = await fwd1.result_async()
    lp1 = np.array(res1.loss_fn_outputs[0]["logprobs"].to_numpy())
    print(f"  Position 7 logprob: {lp1[7]:.4f}")
    print(f"  Position 8 logprob: {lp1[8]:.4f}")
    print(f"  Position 23 logprob: {lp1[23]:.4f}")

    # Test 2: Forward again (check consistency of eval mode)
    print("\nTest 2: Second forward (no mode change)")
    fwd2 = await client.forward_async([datum], loss_fn="importance_sampling")
    res2 = await fwd2.result_async()
    lp2 = np.array(res2.loss_fn_outputs[0]["logprobs"].to_numpy())
    diff2 = lp2[7] - lp1[7]
    print(f"  Position 7 logprob: {lp2[7]:.4f} (diff from test 1: {diff2:+.6f})")

    # Test 3: Do forward_backward but with zero advantages (no gradient impact)
    # This exercises train_mode but shouldn't actually change weights
    print("\nTest 3: forward_backward with zero advantages (train_mode exercise)")
    zero_adv_datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.zeros(len(input_tokens), dtype=torch.float32)),  # ZERO
            "logprobs": tinker.TensorData.from_torch(torch.zeros(len(input_tokens), dtype=torch.float32)),
        },
    )
    fwd_bwd = await client.forward_backward_async([zero_adv_datum], loss_fn="importance_sampling")
    await fwd_bwd.result_async()
    # Skip optimizer step - we're just testing mode transition

    # Test 4: Forward after train_mode (but no actual weight change)
    print("\nTest 4: Forward after train_mode (no optim step)")
    fwd4 = await client.forward_async([datum], loss_fn="importance_sampling")
    res4 = await fwd4.result_async()
    lp4 = np.array(res4.loss_fn_outputs[0]["logprobs"].to_numpy())
    diff4 = lp4[7] - lp1[7]
    print(f"  Position 7 logprob: {lp4[7]:.4f} (diff from test 1: {diff4:+.6f})")

    if abs(diff4) > 0.1:
        print(f"  >>> BUG: Mode transition alone corrupted forward! (diff={diff4:.4f})")

    # Test 5: Now do actual training
    print("\nTest 5: ONE training step (forward_backward + optim_step)")
    fwd_bwd_real = await client.forward_backward_async([datum], loss_fn="importance_sampling")
    await fwd_bwd_real.result_async()
    await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()
    print("  Training step complete")

    # Test 6: Forward after training
    print("\nTest 6: Forward after training")
    fwd6 = await client.forward_async([datum], loss_fn="importance_sampling")
    res6 = await fwd6.result_async()
    lp6 = np.array(res6.loss_fn_outputs[0]["logprobs"].to_numpy())
    diff6 = lp6[7] - lp1[7]
    print(f"  Position 7 logprob: {lp6[7]:.4f} (diff from fresh: {diff6:+.4f})")
    print(f"  Position 8 logprob: {lp6[8]:.4f}")
    print(f"  Position 23 logprob: {lp6[23]:.4f}")

    if lp6[7] < lp1[7] - 5:
        print(f"  >>> BUG: Training made logprob WORSE (expected improvement)")
        print(f"  >>> This is the bug we're investigating!")

    # Test 7: Multiple forwards after training (check stability)
    print("\nTest 7: Forward stability after training")
    for i in range(3):
        fwd = await client.forward_async([datum], loss_fn="importance_sampling")
        res = await fwd.result_async()
        lp = np.array(res.loss_fn_outputs[0]["logprobs"].to_numpy())
        diff = lp[7] - lp6[7]
        print(f"  Forward {i+1}: pos 7 = {lp[7]:.4f} (diff from test 6: {diff:+.6f})")

    # Test 8: Compare with vLLM
    print("\nTest 8: vLLM comparison")
    sampling = await client.save_weights_and_get_sampling_client_async()
    await asyncio.sleep(2)
    vllm_res = await sampling.sample_async(
        prompt=tinker.ModelInput.from_ints(input_tokens),
        num_samples=1,
        sampling_params=tinker.SamplingParams(max_tokens=1, temperature=0.0),
        include_prompt_logprobs=True,
    )
    vllm_lp = vllm_res.prompt_logprobs
    # vLLM prompt_logprobs[i] = P(token[i] | token[0:i])
    # Megatron position 7 = P(target[7] | input[0:7]) = P(token[8] | token[0:7])
    # So vLLM position 8 aligns with Megatron position 7
    print(f"  vLLM position 8 (aligns with Megatron pos 7): {vllm_lp[8]:.4f}")
    print(f"  vLLM position 9: {vllm_lp[9]:.4f}")
    print(f"  vLLM position 24: {vllm_lp[24]:.4f}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Fresh Megatron pos 7:        {lp1[7]:.4f}")
    print(f"After mode transition pos 7: {lp4[7]:.4f} (should match fresh)")
    print(f"After training pos 7:        {lp6[7]:.4f}")
    print(f"vLLM pos 8 (aligned):        {vllm_lp[8]:.4f}")
    print()

    if abs(lp4[7] - lp1[7]) > 0.5:
        print("FINDING: Mode transition alone corrupts forward pass!")
        print("         BUG is in train_mode/eval_mode handling, not in training.")
    elif lp6[7] < lp1[7] - 5 and vllm_lp[8] > lp1[7] - 1:
        print("FINDING: Training corrupts Megatron forward, but vLLM is fine.")
        print("         Megatron-vLLM gap:", lp6[7] - vllm_lp[8], "nats")
        print("         BUG is in Megatron forward with trained LoRA.")
    else:
        print("No obvious pattern detected. Need more investigation.")


if __name__ == "__main__":
    asyncio.run(main())
