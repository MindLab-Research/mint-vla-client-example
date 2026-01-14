#!/usr/bin/env python3
"""Compare Megatron vs vLLM logprobs at the SAME position with SAME weights.

The bug: Same trained LoRA weights produce different logprobs.
- vLLM position 7: -8.09 (argmax='请')
- Megatron position 7: -21.27 (argmax=space with logit 21.75)

This script:
1. Trains LoRA in Megatron
2. Gets Megatron logprobs at ALL positions
3. Exports to vLLM
4. Gets vLLM logprobs at ALL positions
5. Compares side-by-side
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


async def get_vllm_logprobs(sampling_client, input_tokens, tokenizer):
    """Get vLLM logprobs via tinker sampling API."""
    # Use tinker's include_prompt_logprobs feature
    # NOTE: topk_prompt_logprobs disabled due to SDK validation error
    sample_result = await sampling_client.sample_async(
        prompt=tinker.ModelInput.from_ints(input_tokens),
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=1,
            temperature=0.0,
        ),
        include_prompt_logprobs=True,
        # topk_prompt_logprobs=10,  # Disabled: SDK type validation fails on dict format
    )

    if not sample_result.prompt_logprobs:
        print("ERROR: No prompt_logprobs returned")
        return None, None

    logprobs = np.array([lp if lp is not None else -100.0 for lp in sample_result.prompt_logprobs])

    # Top-K disabled for now
    topk = None

    return logprobs, topk


async def main():
    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    mask = [1.0] * len(input_tokens)  # Train on ALL positions

    print(f"Sequence: {len(input_tokens)} tokens")
    print(f"Training on ALL positions (mask=1.0 everywhere)")

    # Show token sequence
    print("\nToken sequence:")
    for i in range(min(15, len(tokens))):
        print(f"  pos={i:2d}: {tokens[i]:6d} ({repr(tokenizer.decode([tokens[i]])):15s})")
    print("  ...")

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    # Create training client
    print("\n" + "=" * 70)
    print("PHASE 1: Create Megatron training client")
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

    # Baseline - Megatron fresh
    print("\n" + "=" * 70)
    print("PHASE 2: Baseline (fresh LoRA) - Megatron")
    print("=" * 70)

    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    baseline_logprobs = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    print(f"Megatron baseline logprobs (first 10 positions):")
    for i in range(min(10, len(baseline_logprobs))):
        target_str = tokenizer.decode([target_tokens[i]])
        print(f"  pos={i:2d}: {baseline_logprobs[i]:8.4f} (target={repr(target_str)})")

    # Baseline - vLLM fresh (export fresh LoRA to vLLM)
    print("\n" + "=" * 70)
    print("PHASE 2b: Baseline (fresh LoRA) - vLLM")
    print("=" * 70)

    fresh_sampling_client = await client.save_weights_and_get_sampling_client_async()
    print("Exported fresh weights to vLLM")
    await asyncio.sleep(2)

    vllm_fresh, _ = await get_vllm_logprobs(fresh_sampling_client, input_tokens, tokenizer)
    if vllm_fresh is not None:
        print(f"vLLM fresh logprobs (first 10):")
        for i in range(min(10, len(vllm_fresh))):
            target_str = tokenizer.decode([target_tokens[i]])
            print(f"  pos={i:2d}: {vllm_fresh[i]:8.4f} (target={repr(target_str)})")

    # Train
    print("\n" + "=" * 70)
    print("PHASE 3: Train 10 steps (lr=1e-3)")
    print("=" * 70)

    for step in range(10):
        fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
        await fwd_bwd.result_async()
        await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()

        if step % 3 == 0 or step == 9:
            fwd = await client.forward_async([datum], loss_fn="importance_sampling")
            result = await fwd.result_async()
            step_logprobs = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())
            print(f"  Step {step+1}: pos7={step_logprobs[7]:.4f}, pos31={step_logprobs[31]:.4f}")

    # After training - Megatron
    print("\n" + "=" * 70)
    print("PHASE 4: Megatron logprobs after training")
    print("=" * 70)

    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    megatron_trained = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    print(f"Megatron trained logprobs:")
    for i in range(min(10, len(megatron_trained))):
        target_str = tokenizer.decode([target_tokens[i]])
        delta = megatron_trained[i] - baseline_logprobs[i]
        print(f"  pos={i:2d}: {megatron_trained[i]:8.4f} (delta={delta:+.4f}, target={repr(target_str)})")

    # Export trained weights to vLLM
    print("\n" + "=" * 70)
    print("PHASE 5: Export trained weights to vLLM + Save full checkpoint")
    print("=" * 70)

    # Save full checkpoint (BOTH Megatron format AND PEFT format)
    # This enables loading back into fresh Megatron for debugging
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = f"debug_checkpoint_{timestamp}"
    print(f"Saving full checkpoint as: {checkpoint_name}")
    save_result = await (await client.save_state_async(name=checkpoint_name)).result_async()
    checkpoint_path = save_result.path
    print(f"Checkpoint saved to: {checkpoint_path}")

    # Also export to vLLM for sampling
    sampling_client = await client.save_weights_and_get_sampling_client_async()
    print("Exported to vLLM")

    await asyncio.sleep(2)  # Wait for vLLM to load

    vllm_trained, vllm_topk = await get_vllm_logprobs(sampling_client, input_tokens, tokenizer)

    if vllm_trained is None:
        print("Failed to get vLLM logprobs")
        return

    print(f"vLLM trained logprobs (first 10):")
    for i in range(min(10, len(vllm_trained))):
        target_str = tokenizer.decode([target_tokens[i]])
        print(f"  pos={i:2d}: {vllm_trained[i]:8.4f} (target={repr(target_str)})")

    # Side-by-side comparison WITH CORRECT ALIGNMENT
    # Megatron logprobs[i] = P(target_tokens[i] = input_tokens[i+1] | input_tokens[0:i+1])
    # vLLM prompt_logprobs[i] = P(input_tokens[i] | input_tokens[0:i]) for i > 0
    # So Megatron[i] predicts same token as vLLM[i+1] - need to shift vLLM by +1
    print("\n" + "=" * 110)
    print("COMPARISON: Megatron vs vLLM (WITH CORRECT ALIGNMENT: Megatron[i] vs vLLM[i+1])")
    print("=" * 110)
    print("Note: Both systems predict target_tokens[i] = input_tokens[i+1]")

    print(f"\n{'Pos':<4} {'Target':<10} {'Meg Fresh':<10} {'Meg Train':<10} {'vLLM F[i+1]':<12} {'vLLM T[i+1]':<12} {'Meg-vLLM':<10}")
    print("-" * 110)

    problematic_positions = []
    for i in range(min(45, len(megatron_trained))):
        target_str = tokenizer.decode([target_tokens[i]])[:8]
        meg_f = baseline_logprobs[i]
        meg_t = megatron_trained[i]
        # CORRECT ALIGNMENT: vLLM[i+1] predicts same token as Megatron[i]
        vllm_idx = i + 1
        vllm_f = vllm_fresh[vllm_idx] if vllm_fresh is not None and vllm_idx < len(vllm_fresh) else float('nan')
        vllm_t = vllm_trained[vllm_idx] if vllm_idx < len(vllm_trained) else float('nan')

        m_diff = meg_t - vllm_t if not np.isnan(vllm_t) else float('nan')

        status = ""
        if not np.isnan(m_diff) and abs(m_diff) > 1.0:
            status = "***"
            problematic_positions.append(i)

        vllm_f_str = f"{vllm_f:<12.4f}" if not np.isnan(vllm_f) else "N/A         "
        vllm_t_str = f"{vllm_t:<12.4f}" if not np.isnan(vllm_t) else "N/A         "
        print(f"{i:<4} {target_str:<10} {meg_f:<10.4f} {meg_t:<10.4f} {vllm_f_str} {vllm_t_str} {m_diff:<+10.4f} {status}")

    # Show top-K at problematic positions (with correct alignment)
    if problematic_positions and vllm_topk:
        print("\n" + "=" * 70)
        print("TOP-K AT PROBLEMATIC POSITIONS (vLLM[i+1] for alignment)")
        print("=" * 70)

        for pos in problematic_positions[:3]:  # Show first 3
            vllm_idx = pos + 1  # Correct alignment
            print(f"\nPosition {pos} (target={target_tokens[pos]} '{tokenizer.decode([target_tokens[pos]])}'):")
            print(f"  Megatron fresh: {baseline_logprobs[pos]:.4f}")
            print(f"  Megatron trained: {megatron_trained[pos]:.4f}")
            if vllm_fresh is not None and vllm_idx < len(vllm_fresh):
                print(f"  vLLM fresh[{vllm_idx}]: {vllm_fresh[vllm_idx]:.4f}")
            if vllm_idx < len(vllm_trained):
                print(f"  vLLM trained[{vllm_idx}]: {vllm_trained[vllm_idx]:.4f}")

            if vllm_idx < len(vllm_topk):
                print(f"  vLLM trained top-5 at position {vllm_idx}:")
                sorted_topk = sorted(vllm_topk[vllm_idx].items(), key=lambda x: x[1], reverse=True)
                for rank, (tok_id, lp) in enumerate(sorted_topk[:5], 1):
                    tok_str = tokenizer.decode([tok_id])
                    marker = " <-- TARGET" if tok_id == target_tokens[pos] else ""
                    print(f"    {rank}. {tok_id:6d} ({repr(tok_str):12s}): {lp:.4f}{marker}")

    # Summary with CORRECT ALIGNMENT
    print("\n" + "=" * 70)
    print("SUMMARY (with correct alignment: Megatron[i] vs vLLM[i+1])")
    print("=" * 70)

    if vllm_trained is not None:
        # Aligned comparison: megatron[i] vs vllm[i+1]
        aligned_len = min(len(megatron_trained), len(vllm_trained) - 1)
        aligned_diffs = megatron_trained[:aligned_len] - vllm_trained[1:aligned_len+1]
        print(f"Max |Megatron trained - vLLM trained| (aligned): {np.max(np.abs(aligned_diffs)):.4f}")
        print(f"Mean |Megatron trained - vLLM trained| (aligned): {np.mean(np.abs(aligned_diffs)):.4f}")
    if vllm_fresh is not None:
        aligned_len = min(len(baseline_logprobs), len(vllm_fresh) - 1)
        aligned_fresh_diffs = baseline_logprobs[:aligned_len] - vllm_fresh[1:aligned_len+1]
        print(f"Max |Megatron fresh - vLLM fresh| (aligned): {np.max(np.abs(aligned_fresh_diffs)):.4f}")
        print(f"Mean |Megatron fresh - vLLM fresh| (aligned): {np.mean(np.abs(aligned_fresh_diffs)):.4f}")
    print(f"Positions with |trained diff| > 1.0 (aligned): {len(problematic_positions)}")

    # Save all debug data to JSON file
    import json
    debug_data = {
        "checkpoint_path": checkpoint_path,
        "timestamp": timestamp,
        "model_name": MODEL_NAME,
        "input_tokens": input_tokens,
        "target_tokens": target_tokens,
        "megatron_fresh": baseline_logprobs.tolist(),
        "megatron_trained": megatron_trained.tolist(),
        "vllm_fresh": vllm_fresh.tolist() if vllm_fresh is not None else [],
        "vllm_trained": vllm_trained.tolist() if vllm_trained is not None else [],
        "problematic_positions": problematic_positions,
    }
    debug_file = f"/tmp/megatron_vllm_debug_{timestamp}.json"
    with open(debug_file, "w") as f:
        json.dump(debug_data, f, indent=2)
    print(f"\nDebug data saved to: {debug_file}")

    print("\n" + "=" * 70)
    print("CHECKPOINT PATH (save this for later debugging):")
    print(f"  {checkpoint_path}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
