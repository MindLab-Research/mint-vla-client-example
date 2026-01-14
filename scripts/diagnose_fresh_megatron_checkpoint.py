#!/usr/bin/env python3
"""Diagnose: Load trained checkpoint into FRESH Megatron and trace computation.

Key finding from previous investigation:
- Training produces garbage logprobs at certain positions (e.g., -46 to -51 nats)
- vLLM with same exported weights shows normal behavior (-8 nats)
- Loading checkpoint into FRESH Megatron still shows garbage

This script:
1. Creates a FRESH Megatron actor (reinit_lora_weights)
2. Gets baseline logprobs (should be ~-8 for position 7)
3. Loads the trained checkpoint
4. Gets logprobs again (if garbage, we've reproduced the issue)
5. Traces the LoRA weight values and application

Usage:
    TINKER_BASE_URL=http://localhost:8000 python scripts/diagnose_fresh_megatron_checkpoint.py /path/to/checkpoint
"""

import asyncio
import os
import sys
import json

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


async def get_megatron_logprobs(client, datum):
    """Get logprobs from Megatron training client."""
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    return np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())


async def main():
    from transformers import AutoTokenizer

    # Check for checkpoint path argument
    if len(sys.argv) < 2:
        print("Usage: python diagnose_fresh_megatron_checkpoint.py <checkpoint_path>")
        print("\nTo get a checkpoint, run test_megatron_vllm_logprob_mismatch.py first")
        print("It will print a CHECKPOINT PATH at the end.")
        return

    checkpoint_path = sys.argv[1]
    print(f"Checkpoint path: {checkpoint_path}")

    # Check if checkpoint exists
    if not checkpoint_path.startswith("tinker://"):
        # Local path check
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: Checkpoint not found: {checkpoint_path}")
            return

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    mask = [1.0] * len(input_tokens)

    print(f"Sequence: {len(input_tokens)} tokens")

    # Key positions to analyze
    key_positions = [7, 14, 23, 31, 49]
    print("\nKey positions to analyze:")
    for pos in key_positions:
        if pos < len(target_tokens):
            tgt = target_tokens[pos]
            tgt_str = tokenizer.decode([tgt])
            print(f"  pos {pos}: target={tgt} ({repr(tgt_str)})")

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.ones(len(input_tokens), dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.zeros(len(input_tokens), dtype=torch.float32)),
        }
    )

    # ===============================================================
    # PHASE 1: Create FRESH Megatron actor
    # ===============================================================
    print("\n" + "=" * 70)
    print("PHASE 1: Create FRESH Megatron actor")
    print("=" * 70)

    client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)
    print(f"Model ID: {client.model_id}")

    # Get baseline logprobs (fresh LoRA)
    print("\nGetting baseline logprobs (fresh LoRA)...")
    fresh_logprobs = await get_megatron_logprobs(client, datum)

    print("\nFresh LoRA logprobs at key positions:")
    for pos in key_positions:
        if pos < len(fresh_logprobs):
            tgt_str = tokenizer.decode([target_tokens[pos]])[:8]
            print(f"  pos {pos}: {fresh_logprobs[pos]:8.4f} (target={repr(tgt_str)})")

    # ===============================================================
    # PHASE 2: Load trained checkpoint
    # ===============================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Load trained checkpoint into SAME actor")
    print("=" * 70)

    # Load the checkpoint
    load_result = await (await client.load_state_async(path=checkpoint_path)).result_async()
    print(f"Loaded checkpoint: {checkpoint_path}")

    # Get logprobs after loading
    print("\nGetting logprobs after loading checkpoint...")
    loaded_logprobs = await get_megatron_logprobs(client, datum)

    print("\nLoaded checkpoint logprobs at key positions:")
    for pos in key_positions:
        if pos < len(loaded_logprobs):
            tgt_str = tokenizer.decode([target_tokens[pos]])[:8]
            delta = loaded_logprobs[pos] - fresh_logprobs[pos]
            flag = " <-- GARBAGE" if loaded_logprobs[pos] < -30 else ""
            print(f"  pos {pos}: {loaded_logprobs[pos]:8.4f} (delta={delta:+.4f}, target={repr(tgt_str)}){flag}")

    # ===============================================================
    # PHASE 3: Create ANOTHER fresh Megatron actor
    # ===============================================================
    print("\n" + "=" * 70)
    print("PHASE 3: Create ANOTHER FRESH Megatron actor")
    print("=" * 70)

    # Note: create_lora_training_client_async creates a new session
    # which will reinitialize LoRA weights
    client2 = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)
    print(f"New Model ID: {client2.model_id}")

    # Get fresh logprobs from new actor
    print("\nGetting logprobs from NEW fresh actor...")
    fresh2_logprobs = await get_megatron_logprobs(client2, datum)

    print("\nNew fresh actor logprobs at key positions:")
    for pos in key_positions:
        if pos < len(fresh2_logprobs):
            tgt_str = tokenizer.decode([target_tokens[pos]])[:8]
            diff_from_first = fresh2_logprobs[pos] - fresh_logprobs[pos]
            print(f"  pos {pos}: {fresh2_logprobs[pos]:8.4f} (diff from first fresh={diff_from_first:+.4f})")

    # ===============================================================
    # PHASE 4: Load checkpoint into NEW actor
    # ===============================================================
    print("\n" + "=" * 70)
    print("PHASE 4: Load checkpoint into NEW actor")
    print("=" * 70)

    load_result2 = await (await client2.load_state_async(path=checkpoint_path)).result_async()
    print(f"Loaded checkpoint into new actor: {checkpoint_path}")

    # Get logprobs
    loaded2_logprobs = await get_megatron_logprobs(client2, datum)

    print("\nLoaded checkpoint (new actor) logprobs at key positions:")
    for pos in key_positions:
        if pos < len(loaded2_logprobs):
            tgt_str = tokenizer.decode([target_tokens[pos]])[:8]
            delta = loaded2_logprobs[pos] - fresh2_logprobs[pos]
            diff_actors = loaded2_logprobs[pos] - loaded_logprobs[pos]
            flag = " <-- GARBAGE" if loaded2_logprobs[pos] < -30 else ""
            print(f"  pos {pos}: {loaded2_logprobs[pos]:8.4f} (delta={delta:+.4f}, diff_actors={diff_actors:+.4f}){flag}")

    # ===============================================================
    # PHASE 5: Export to vLLM and compare
    # ===============================================================
    print("\n" + "=" * 70)
    print("PHASE 5: Export to vLLM")
    print("=" * 70)

    sampling_client = await client2.save_weights_and_get_sampling_client_async()
    print("Exported to vLLM")
    await asyncio.sleep(2)

    # Get vLLM logprobs
    sample_result = await sampling_client.sample_async(
        prompt=tinker.ModelInput.from_ints(input_tokens),
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=1,
            temperature=0.0,
        ),
        include_prompt_logprobs=True,
    )

    if sample_result.prompt_logprobs:
        vllm_logprobs = np.array([lp if lp is not None else -100.0 for lp in sample_result.prompt_logprobs])

        print("\nvLLM logprobs at key positions (aligned: vLLM[pos+1] vs Megatron[pos]):")
        for pos in key_positions:
            if pos + 1 < len(vllm_logprobs):
                tgt_str = tokenizer.decode([target_tokens[pos]])[:8]
                vllm_lp = vllm_logprobs[pos + 1]
                mega_lp = loaded2_logprobs[pos]
                diff = mega_lp - vllm_lp
                flag = " <-- MISMATCH" if abs(diff) > 5 else ""
                print(f"  pos {pos}: Megatron={mega_lp:8.4f}, vLLM[{pos+1}]={vllm_lp:8.4f}, diff={diff:+.4f}{flag}")
    else:
        print("Failed to get vLLM logprobs")

    # ===============================================================
    # SUMMARY
    # ===============================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Check if issue reproduces
    garbage_positions = [pos for pos in key_positions if pos < len(loaded2_logprobs) and loaded2_logprobs[pos] < -30]

    if garbage_positions:
        print(f"\n*** ISSUE REPRODUCED: Garbage logprobs at positions {garbage_positions} ***")
        print("\nThis confirms the issue is in how Megatron applies the trained LoRA weights,")
        print("NOT in accumulated training state (optimizer, expert_bias, etc.)")
        print("\nThe trained weights work correctly in vLLM but produce garbage in Megatron.")
        print("This suggests a fundamental difference in how LoRA is applied.")
    else:
        print("\nNo garbage logprobs detected. Issue may not reproduce with this checkpoint.")

    print(f"\n{'Position':<10} {'Fresh':<12} {'Loaded(A1)':<12} {'Fresh(A2)':<12} {'Loaded(A2)':<12}")
    print("-" * 60)
    for pos in key_positions:
        if pos < len(loaded2_logprobs):
            print(f"{pos:<10} {fresh_logprobs[pos]:<12.4f} {loaded_logprobs[pos]:<12.4f} {fresh2_logprobs[pos]:<12.4f} {loaded2_logprobs[pos]:<12.4f}")


if __name__ == "__main__":
    asyncio.run(main())
