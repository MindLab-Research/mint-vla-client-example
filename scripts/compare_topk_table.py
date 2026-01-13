#!/usr/bin/env python3
"""Generate comparison table: top-k tokens from vLLM vs Megatron (fresh vs trained).

Checkpoint: /vePFS-Mindverse/share/code/tinker-server/checkpoints/80e6bf97-56d9-4f3a-9872-7cc4b80a7379_0/debug_checkpoint_20260110_182006/
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
CHECKPOINT_PATH = "/vePFS-Mindverse/share/code/tinker-server/checkpoints/80e6bf97-56d9-4f3a-9872-7cc4b80a7379_0/debug_checkpoint_20260110_182006/"

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


async def get_megatron_topk(client, input_tokens, target_tokens, tokenizer, k=5):
    """Get top-k tokens from Megatron at each position via raw logits."""
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

    # Forward pass to get logprobs of target tokens
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    target_logprobs = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    # Also get raw logits via debug API if available
    # For now, return target logprobs and position info
    return target_logprobs


async def get_vllm_topk_via_raw_logits(sampling_client, input_tokens, tokenizer, k=5):
    """Get top-k tokens from vLLM via sampling with logprobs."""
    # vLLM returns prompt_logprobs for each position
    sample_result = await sampling_client.sample_async(
        prompt=tinker.ModelInput.from_ints(input_tokens),
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=1,
            temperature=0.0,
        ),
        include_prompt_logprobs=True,
    )

    logprobs = [lp if lp is not None else -100.0 for lp in sample_result.prompt_logprobs]
    return np.array(logprobs)


async def main():
    from transformers import AutoTokenizer

    print("=" * 80)
    print("TOP-K COMPARISON TABLE: vLLM vs Megatron (Fresh vs Trained Checkpoint)")
    print("=" * 80)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]

    print(f"Sequence length: {len(input_tokens)} input tokens")
    print(f"Text:\n{TEST_TEXT[:200]}...")
    print()

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    # ==========================================================================
    # COLUMN 1: Megatron Fresh
    # ==========================================================================
    print("=" * 60)
    print("COLUMN 1: Megatron (Fresh LoRA)")
    print("=" * 60)

    meg_fresh_client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)
    meg_fresh_logprobs = await get_megatron_topk(meg_fresh_client, input_tokens, target_tokens, tokenizer)
    print(f"Got {len(meg_fresh_logprobs)} logprobs from Megatron fresh")

    # ==========================================================================
    # COLUMN 2: vLLM Fresh (export fresh LoRA)
    # ==========================================================================
    print("\n" + "=" * 60)
    print("COLUMN 2: vLLM (Fresh LoRA)")
    print("=" * 60)

    vllm_fresh_client = await meg_fresh_client.save_weights_and_get_sampling_client_async()
    await asyncio.sleep(2)
    vllm_fresh_logprobs = await get_vllm_topk_via_raw_logits(vllm_fresh_client, input_tokens, tokenizer)
    print(f"Got {len(vllm_fresh_logprobs)} logprobs from vLLM fresh")

    # ==========================================================================
    # COLUMN 3: Megatron Trained (load checkpoint)
    # ==========================================================================
    print("\n" + "=" * 60)
    print("COLUMN 3: Megatron (Trained Checkpoint)")
    print("=" * 60)

    # Create fresh client, then load checkpoint
    meg_trained_client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)
    print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
    await (await meg_trained_client.load_state_async(CHECKPOINT_PATH)).result_async()
    print("Checkpoint loaded")
    meg_trained_logprobs = await get_megatron_topk(meg_trained_client, input_tokens, target_tokens, tokenizer)
    print(f"Got {len(meg_trained_logprobs)} logprobs from Megatron trained")

    # ==========================================================================
    # COLUMN 4: vLLM Trained (export from loaded checkpoint)
    # ==========================================================================
    print("\n" + "=" * 60)
    print("COLUMN 4: vLLM (Trained Checkpoint)")
    print("=" * 60)

    vllm_trained_client = await meg_trained_client.save_weights_and_get_sampling_client_async()
    await asyncio.sleep(2)
    vllm_trained_logprobs = await get_vllm_topk_via_raw_logits(vllm_trained_client, input_tokens, tokenizer)
    print(f"Got {len(vllm_trained_logprobs)} logprobs from vLLM trained")

    # ==========================================================================
    # COMPARISON TABLE
    # ==========================================================================
    print("\n" + "=" * 120)
    print("COMPARISON TABLE: Target Token Log Probabilities")
    print("=" * 120)
    print("Note: vLLM[i+1] corresponds to Megatron[i] (predicting input_tokens[i+1])")
    print()

    # Header
    print(f"{'Pos':<4} {'Target Token':<15} {'Meg Fresh':<12} {'Meg Train':<12} {'vLLM Fresh':<12} {'vLLM Train':<12} {'M-V Fresh':<12} {'M-V Train':<12}")
    print("-" * 120)

    problematic = []
    for i in range(len(meg_fresh_logprobs)):
        target_tok = target_tokens[i]
        target_str = tokenizer.decode([target_tok])[:12]

        meg_f = meg_fresh_logprobs[i]
        meg_t = meg_trained_logprobs[i]

        # Alignment: vLLM[i+1] = Megatron[i]
        vllm_idx = i + 1
        vllm_f = vllm_fresh_logprobs[vllm_idx] if vllm_idx < len(vllm_fresh_logprobs) else float('nan')
        vllm_t = vllm_trained_logprobs[vllm_idx] if vllm_idx < len(vllm_trained_logprobs) else float('nan')

        diff_fresh = meg_f - vllm_f if not np.isnan(vllm_f) else float('nan')
        diff_train = meg_t - vllm_t if not np.isnan(vllm_t) else float('nan')

        marker = ""
        if not np.isnan(diff_train) and abs(diff_train) > 1.0:
            marker = " ***"
            problematic.append(i)

        print(f"{i:<4} {target_str:<15} {meg_f:<12.4f} {meg_t:<12.4f} {vllm_f:<12.4f} {vllm_t:<12.4f} {diff_fresh:<+12.4f} {diff_train:<+12.4f}{marker}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    aligned_len = min(len(meg_fresh_logprobs), len(vllm_fresh_logprobs) - 1)
    fresh_diffs = meg_fresh_logprobs[:aligned_len] - vllm_fresh_logprobs[1:aligned_len+1]
    train_diffs = meg_trained_logprobs[:aligned_len] - vllm_trained_logprobs[1:aligned_len+1]

    print(f"Fresh:   max|M-V|={np.max(np.abs(fresh_diffs)):.4f}, mean|M-V|={np.mean(np.abs(fresh_diffs)):.4f}")
    print(f"Trained: max|M-V|={np.max(np.abs(train_diffs)):.4f}, mean|M-V|={np.mean(np.abs(train_diffs)):.4f}")
    print(f"Positions with |trained diff| > 1.0: {len(problematic)}")

    if problematic:
        print(f"\nProblematic positions: {problematic}")


if __name__ == "__main__":
    asyncio.run(main())
