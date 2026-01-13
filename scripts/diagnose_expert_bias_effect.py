#!/usr/bin/env python3
"""Diagnose expert_bias effect on Megatron vs vLLM logprob mismatch.

Hypothesis: The expert_bias accumulated during training causes different routing
in Megatron vs vLLM, which explains the logprob divergence.

Test:
1. Train LoRA for 10 steps
2. Compare logprobs WITH expert_bias (Megatron default)
3. Reset expert_bias and compare logprobs again
4. If reset brings Megatron closer to vLLM, expert_bias is a contributing factor
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

    print(f"Sequence: {len(tokens)} tokens")

    # Key positions to analyze
    positions = [7, 14, 31, 49]
    print("\nKey positions:")
    for pos in positions:
        tgt = target_tokens[pos]
        tgt_str = tokenizer.decode([tgt])
        print(f"  pos {pos}: target={tgt} ({repr(tgt_str)})")

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    print("\n" + "=" * 70)
    print("PHASE 1: Create LoRA and train 10 steps")
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

    for step in range(10):
        fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
        await fwd_bwd.result_async()
        await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()
        if (step + 1) % 5 == 0:
            print(f"  Completed step {step + 1}")

    print("\n" + "=" * 70)
    print("PHASE 2: Get logprobs WITH expert_bias (as accumulated during training)")
    print("=" * 70)

    # Get Megatron logprobs with expert_bias
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    mega_with_bias = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    print("\nMegatron WITH expert_bias:")
    for pos in positions:
        tgt = target_tokens[pos]
        tgt_str = tokenizer.decode([tgt])[:8]
        print(f"  pos {pos}: logprob={mega_with_bias[pos]:.4f} (target={repr(tgt_str)})")

    print("\n" + "=" * 70)
    print("PHASE 3: Reset expert_bias and get logprobs again")
    print("=" * 70)

    # Reset expert_bias via the new API endpoint
    base_url = os.environ["TINKER_BASE_URL"]
    import httpx
    async with httpx.AsyncClient() as http_client:
        try:
            # Get the model_id from the training client
            model_id = client.model_id

            response = await http_client.post(
                f"{base_url}/api/v1/reset_expert_bias",
                json={"model_id": model_id},
                timeout=30.0
            )
            if response.status_code == 200:
                reset_result = response.json()
                print(f"  Reset expert_bias: {reset_result}")
            else:
                print(f"  Reset API failed: status={response.status_code}, body={response.text}")
        except Exception as e:
            print(f"  Could not reset expert_bias: {e}")

    # Get Megatron logprobs after reset (if reset worked)
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    mega_after_reset = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    print("\nMegatron after reset attempt:")
    for pos in positions:
        tgt = target_tokens[pos]
        tgt_str = tokenizer.decode([tgt])[:8]
        diff = mega_after_reset[pos] - mega_with_bias[pos]
        print(f"  pos {pos}: logprob={mega_after_reset[pos]:.4f} (diff from before: {diff:+.4f})")

    print("\n" + "=" * 70)
    print("PHASE 4: Export to vLLM and compare")
    print("=" * 70)

    # Export to vLLM (which has zero expert_bias)
    sampling_client = await client.save_weights_and_get_sampling_client_async()
    prompt = tinker.ModelInput.from_ints(tokens)
    vllm_lp = await sampling_client.compute_logprobs_async(prompt)

    print("\nFinal comparison (Megatron[pos] vs vLLM[pos+1]):")
    print(f"{'pos':>4} {'target':>8} {'Mega(bias)':>12} {'Mega(reset)':>12} {'vLLM':>12} {'M(b)-V':>10} {'M(r)-V':>10}")
    print("-" * 80)

    for pos in positions:
        tgt = target_tokens[pos]
        tgt_str = tokenizer.decode([tgt])[:8]
        mega_b = mega_with_bias[pos]
        mega_r = mega_after_reset[pos]
        vllm_pos = pos + 1
        vllm = vllm_lp[vllm_pos] if vllm_pos < len(vllm_lp) and vllm_lp[vllm_pos] is not None else float('nan')
        diff_b = mega_b - vllm
        diff_r = mega_r - vllm

        flag = ""
        if abs(diff_r) < abs(diff_b) * 0.5:
            flag = " <-- RESET HELPS"
        elif abs(diff_b) > 5:
            flag = " <-- MISMATCH"

        print(f"{pos:4d} {tgt_str:>8s} {mega_b:12.4f} {mega_r:12.4f} {vllm:12.4f} {diff_b:+10.4f} {diff_r:+10.4f}{flag}")

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Check if reset helped
    total_diff_with_bias = sum(abs(mega_with_bias[p] - (vllm_lp[p+1] if p+1 < len(vllm_lp) and vllm_lp[p+1] else mega_with_bias[p])) for p in positions)
    total_diff_after_reset = sum(abs(mega_after_reset[p] - (vllm_lp[p+1] if p+1 < len(vllm_lp) and vllm_lp[p+1] else mega_after_reset[p])) for p in positions)

    print(f"\nTotal |diff| with bias:  {total_diff_with_bias:.2f}")
    print(f"Total |diff| after reset: {total_diff_after_reset:.2f}")

    if total_diff_after_reset < total_diff_with_bias * 0.7:
        print("\n** expert_bias is a MAJOR contributor to the mismatch **")
    elif total_diff_after_reset < total_diff_with_bias * 0.9:
        print("\n** expert_bias is a MINOR contributor **")
    else:
        print("\n** expert_bias is NOT the main cause - shared LoRA interference is the issue **")


if __name__ == "__main__":
    asyncio.run(main())
