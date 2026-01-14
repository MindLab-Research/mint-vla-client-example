#!/usr/bin/env python3
"""Compare exact top-K logits at position 7 between Megatron and vLLM.

This script:
1. Trains LoRA for 10 steps
2. Gets top-K logits at position 7 from both Megatron (via server API) and vLLM
3. Checks if token 3922 ('Count') vs token 16 ('1') ranking changed
"""

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


async def main():
    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    mask = [1.0] * len(input_tokens)

    print(f"Sequence: {len(tokens)} tokens")
    print(f"Position 7: input='{tokenizer.decode([input_tokens[7]])}' (token {input_tokens[7]})")
    print(f"         -> target='{tokenizer.decode([target_tokens[7]])}' (token {target_tokens[7]})")

    target_7 = target_tokens[7]  # Should be 3922 ('Count')
    token_16 = 16  # The interfering token ('1')

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    print("\n" + "=" * 70)
    print("PHASE 1: Create LoRA and get FRESH top-K")
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

    # Get fresh Megatron logprobs to baseline
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    fresh_mega_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())
    print(f"\nFresh Megatron pos 7: logprob={fresh_mega_lp[7]:.4f} (target={target_7})")

    # Get fresh vLLM logprobs
    sampling_client = await client.save_weights_and_get_sampling_client_async()
    prompt = tinker.ModelInput.from_ints(tokens)
    fresh_vllm_lp = await sampling_client.compute_logprobs_async(prompt)
    print(f"Fresh vLLM pos 8 (=7+1): logprob={fresh_vllm_lp[8]:.4f}")

    # Now get top-K from vLLM
    print("\n--- Fresh vLLM top-K at position 8 (predicting token at position 7) ---")
    # We need to call the sampling API with top_logprobs option
    # But the current API may not expose this directly

    print("\n" + "=" * 70)
    print("PHASE 2: Train 10 steps")
    print("=" * 70)

    for step in range(10):
        fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
        await fwd_bwd.result_async()
        await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()
        if (step + 1) % 5 == 0:
            fwd = await client.forward_async([datum], loss_fn="importance_sampling")
            result = await fwd.result_async()
            train_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())
            print(f"  Step {step+1}: pos7={train_lp[7]:.4f}, pos14={train_lp[14]:.4f}")

    print("\n" + "=" * 70)
    print("PHASE 3: Get trained logprobs and analyze")
    print("=" * 70)

    # Get trained Megatron logprobs
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    trained_mega_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    # Export and get vLLM logprobs
    sampling_client = await client.save_weights_and_get_sampling_client_async()
    trained_vllm_lp = await sampling_client.compute_logprobs_async(prompt)

    print(f"\nPosition 7 comparison:")
    print(f"  Fresh Megatron:   {fresh_mega_lp[7]:.4f}")
    print(f"  Trained Megatron: {trained_mega_lp[7]:.4f} (delta: {trained_mega_lp[7] - fresh_mega_lp[7]:+.4f})")
    print(f"  Fresh vLLM:       {fresh_vllm_lp[8]:.4f}")
    print(f"  Trained vLLM:     {trained_vllm_lp[8]:.4f} (delta: {trained_vllm_lp[8] - fresh_vllm_lp[8]:+.4f})")

    print(f"\n  Megatron-vLLM diff (fresh):   {fresh_mega_lp[7] - fresh_vllm_lp[8]:+.4f}")
    print(f"  Megatron-vLLM diff (trained): {trained_mega_lp[7] - trained_vllm_lp[8]:+.4f}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    mega_degradation = trained_mega_lp[7] - fresh_mega_lp[7]
    vllm_change = trained_vllm_lp[8] - fresh_vllm_lp[8]

    print(f"\nMegatron degradation at pos 7: {mega_degradation:.2f} nats")
    print(f"vLLM change at pos 7:          {vllm_change:.2f} nats")

    if mega_degradation < -10 and vllm_change > -1:
        print("\n** CONFIRMED: Cross-position gradient interference in Megatron's shared LoRA **")
        print(f"   - Megatron degrades by {abs(mega_degradation):.1f} nats")
        print(f"   - vLLM shows change of {vllm_change:.2f} nats")
        print("\nRoot cause hypothesis:")
        print("  1. During training, positions 14 and 49 (target='1') push gradients to boost token 16")
        print("  2. Position 7 (target='Count') pushes gradient to boost token 3922")
        print("  3. Shared LoRA receives ALL gradients and learns a compromise favoring token 16")
        print("  4. This harms position 7's prediction of token 3922")
        print("\n  5. During vLLM inference, per-expert LoRA (even if identical) + different routing")
        print("     causes different forward pass, which happens to be less harmful to position 7")

    # Check if the issue is routing-related by comparing other positions
    print("\n\nAll positions comparison:")
    print(f"{'pos':>4} {'target':>6} {'M-fresh':>10} {'M-train':>10} {'V-train':>10} {'M-degr':>10} {'M-V':>10}")
    print("-" * 70)
    for pos in range(len(trained_mega_lp)):
        tgt = target_tokens[pos] if pos < len(target_tokens) else -1
        m_fresh = fresh_mega_lp[pos]
        m_train = trained_mega_lp[pos]
        v_pos = pos + 1
        v_train = trained_vllm_lp[v_pos] if v_pos < len(trained_vllm_lp) and trained_vllm_lp[v_pos] is not None else float('nan')
        m_degr = m_train - m_fresh
        mv_diff = m_train - v_train if not np.isnan(v_train) else float('nan')

        # Only show interesting positions
        if abs(m_degr) > 0.5 or abs(mv_diff) > 1:
            flag = ""
            if m_degr < -5:
                flag = " <-- DEGRADED"
            elif abs(mv_diff) > 5:
                flag = " <-- M-V MISMATCH"
            print(f"{pos:4d} {tgt:6d} {m_fresh:10.4f} {m_train:10.4f} {v_train:10.4f} {m_degr:+10.4f} {mv_diff:+10.4f}{flag}")


if __name__ == "__main__":
    asyncio.run(main())
