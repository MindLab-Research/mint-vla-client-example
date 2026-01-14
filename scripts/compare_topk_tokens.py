#!/usr/bin/env python3
"""Get actual TOP-K tokens from Megatron vs vLLM at each position."""

import asyncio
import os
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
    """Get top-k tokens from Megatron via forward pass."""
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

    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()

    output = result.loss_fn_outputs[0]
    target_logprobs = np.array(output["logprobs"].to_numpy())

    # Check for topk_indices and topk_logits
    topk_per_position = []
    if "topk_indices" in output and "topk_logits" in output:
        topk_indices_data = output["topk_indices"]
        topk_logits_data = output["topk_logits"]

        # Handle TensorData format
        if hasattr(topk_indices_data, "data"):
            topk_indices = topk_indices_data.data
            topk_logits = topk_logits_data.data
        else:
            topk_indices = topk_indices_data
            topk_logits = topk_logits_data

        # topk_indices/topk_logits: list of [seq_len][k]
        for pos in range(len(target_logprobs)):
            if pos < len(topk_indices):
                pos_indices = topk_indices[pos]
                pos_logits = topk_logits[pos]
                topk_per_position.append(list(zip(pos_indices[:k], pos_logits[:k])))
            else:
                topk_per_position.append([])
    else:
        print(f"  topk not found in output. Keys: {list(output.keys())}")

    return target_logprobs, topk_per_position


async def get_vllm_topk(sampling_client, input_tokens, tokenizer, k=5):
    """Get top-k tokens from vLLM."""
    # Note: topk_prompt_logprobs causes SDK validation error, so we skip it
    sample_result = await sampling_client.sample_async(
        prompt=tinker.ModelInput.from_ints(input_tokens),
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=1,
            temperature=0.0,
            logprobs=k,
        ),
        include_prompt_logprobs=True,
        # topk_prompt_logprobs=k,  # Causes SDK validation error
    )

    target_logprobs = [lp if lp is not None else -100.0 for lp in sample_result.prompt_logprobs]
    # topk not available without topk_prompt_logprobs
    return np.array(target_logprobs), []


def format_topk(topk_list, tokenizer, k=5):
    """Format top-k list as string."""
    if not topk_list:
        return "N/A"
    lines = []
    for i, (tok_id, lp) in enumerate(topk_list[:k]):
        tok_str = repr(tokenizer.decode([tok_id]))[:12]
        lines.append(f"{tok_id:6d} {tok_str:12s} {lp:8.2f}")
    return " | ".join(lines)


async def main():
    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]

    print(f"Sequence length: {len(input_tokens)} tokens")

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    # Create clients
    print("\nCreating clients...")
    meg_fresh_client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)

    meg_trained_client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)
    await (await meg_trained_client.load_state_async(CHECKPOINT_PATH)).result_async()
    print("  Checkpoint loaded")

    vllm_fresh_client = await meg_fresh_client.save_weights_and_get_sampling_client_async()
    await asyncio.sleep(2)

    vllm_trained_client = await meg_trained_client.save_weights_and_get_sampling_client_async()
    await asyncio.sleep(2)

    # Get data
    print("\nGetting Megatron fresh...")
    meg_fresh_lp, meg_fresh_topk = await get_megatron_topk(meg_fresh_client, input_tokens, target_tokens, tokenizer)

    print("Getting Megatron trained...")
    meg_train_lp, meg_train_topk = await get_megatron_topk(meg_trained_client, input_tokens, target_tokens, tokenizer)

    print("Getting vLLM fresh...")
    vllm_fresh_lp, vllm_fresh_topk = await get_vllm_topk(vllm_fresh_client, input_tokens, tokenizer)

    print("Getting vLLM trained...")
    vllm_train_lp, vllm_train_topk = await get_vllm_topk(vllm_trained_client, input_tokens, tokenizer)

    # Print comparison table
    print("\n" + "=" * 160)
    print("TOP-K TOKEN COMPARISON (positions with |Meg_Train - vLLM_Train| > 1.0)")
    print("=" * 160)

    for pos in range(len(input_tokens)):
        target_tok = target_tokens[pos]
        target_str = repr(tokenizer.decode([target_tok]))[:10]

        vllm_idx = pos + 1  # Alignment

        meg_f = meg_fresh_lp[pos]
        meg_t = meg_train_lp[pos]
        vllm_f = vllm_fresh_lp[vllm_idx] if vllm_idx < len(vllm_fresh_lp) else float('nan')
        vllm_t = vllm_train_lp[vllm_idx] if vllm_idx < len(vllm_train_lp) else float('nan')

        diff = abs(meg_t - vllm_t) if not np.isnan(vllm_t) else 0

        if diff > 1.0 or pos < 5:  # Show problematic + first 5
            print(f"\n--- Position {pos}: Target={target_tok} {target_str} ---")
            print(f"Target logprob: Meg_F={meg_f:.2f}, Meg_T={meg_t:.2f}, vLLM_F={vllm_f:.2f}, vLLM_T={vllm_t:.2f}, |diff|={diff:.2f}")

            print(f"\nMegatron Fresh top-5:")
            if meg_fresh_topk and pos < len(meg_fresh_topk) and meg_fresh_topk[pos]:
                for i, (tid, lp) in enumerate(meg_fresh_topk[pos][:5]):
                    marker = " <-- TARGET" if tid == target_tok else ""
                    print(f"  {i+1}. {tid:6d} {repr(tokenizer.decode([tid])):15s} {lp:8.2f}{marker}")
            else:
                print("  (not available)")

            print(f"\nMegatron Trained top-5:")
            if meg_train_topk and pos < len(meg_train_topk) and meg_train_topk[pos]:
                for i, (tid, lp) in enumerate(meg_train_topk[pos][:5]):
                    marker = " <-- TARGET" if tid == target_tok else ""
                    print(f"  {i+1}. {tid:6d} {repr(tokenizer.decode([tid])):15s} {lp:8.2f}{marker}")
            else:
                print("  (not available)")

            print(f"\nvLLM Fresh top-5 (pos {vllm_idx}):")
            if vllm_fresh_topk and vllm_idx < len(vllm_fresh_topk) and vllm_fresh_topk[vllm_idx]:
                for i, (tid, lp) in enumerate(vllm_fresh_topk[vllm_idx][:5]):
                    marker = " <-- TARGET" if tid == target_tok else ""
                    print(f"  {i+1}. {tid:6d} {repr(tokenizer.decode([tid])):15s} {lp:8.2f}{marker}")
            else:
                print("  (not available)")

            print(f"\nvLLM Trained top-5 (pos {vllm_idx}):")
            if vllm_train_topk and vllm_idx < len(vllm_train_topk) and vllm_train_topk[vllm_idx]:
                for i, (tid, lp) in enumerate(vllm_train_topk[vllm_idx][:5]):
                    marker = " <-- TARGET" if tid == target_tok else ""
                    print(f"  {i+1}. {tid:6d} {repr(tokenizer.decode([tid])):15s} {lp:8.2f}{marker}")
            else:
                print("  (not available)")


if __name__ == "__main__":
    asyncio.run(main())
