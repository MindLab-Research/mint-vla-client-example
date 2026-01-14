#!/usr/bin/env python3
"""Train LoRA and save checkpoint to a KNOWN path for reuse.

Checkpoint will be saved to: /tmp/debug_lora_checkpoint/

After running this, use query_checkpoint_topk.py to investigate without retraining.
"""

import asyncio
import os

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker
import torch
import numpy as np

MODEL_NAME = "moonshotai/Moonlight-16B-A3B-Instruct"
CHECKPOINT_PATH = "/tmp/debug_lora_checkpoint"

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

    print(f"Sequence: {len(input_tokens)} tokens")
    print(f"Checkpoint will be saved to: {CHECKPOINT_PATH}")

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    # Create training client
    print("\n" + "=" * 70)
    print("Creating Megatron training client")
    print("=" * 70)

    client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)

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

    # Get fresh logprobs
    print("\n" + "=" * 70)
    print("Getting fresh Megatron logprobs")
    print("=" * 70)
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    fresh_logprobs = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())
    print(f"Fresh logprobs sample: pos7={fresh_logprobs[7]:.4f}, pos10={fresh_logprobs[10]:.4f}")

    # Train 10 steps
    print("\n" + "=" * 70)
    print("Training 10 steps (lr=1e-3)")
    print("=" * 70)

    for step in range(10):
        fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
        await fwd_bwd.result_async()
        await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()

        if step % 3 == 0 or step == 9:
            fwd = await client.forward_async([datum], loss_fn="importance_sampling")
            result = await fwd.result_async()
            step_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())
            print(f"  Step {step+1}: pos7={step_lp[7]:.4f}, pos10={step_lp[10]:.4f}")

    # Get trained logprobs
    print("\n" + "=" * 70)
    print("Getting trained Megatron logprobs")
    print("=" * 70)
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    trained_logprobs = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    # Save checkpoint to known path
    print("\n" + "=" * 70)
    print("Saving checkpoint")
    print("=" * 70)

    # Use save_weights_for_sampler which returns a tinker:// URI
    save_future = await client.save_weights_for_sampler_async(name="debug_checkpoint")
    save_result = await save_future.result_async()
    checkpoint_path = save_result.path
    print(f"Checkpoint saved to: {checkpoint_path}")

    # Also export to vLLM for comparison
    print("\nExporting to vLLM...")
    sampling_client = await client.save_weights_and_get_sampling_client_async(name="debug_vllm")
    await asyncio.sleep(2)

    # Get vLLM logprobs with TOP-K
    sample_result = await sampling_client.sample_async(
        prompt=tinker.ModelInput.from_ints(input_tokens),
        num_samples=1,
        sampling_params=tinker.SamplingParams(max_tokens=1, temperature=0.0),
        include_prompt_logprobs=True,
        topk_prompt_logprobs=10,
    )
    vllm_logprobs = np.array([lp if lp is not None else -100.0 for lp in sample_result.prompt_logprobs])

    # Convert top-K to serializable format
    vllm_topk_data = []
    if sample_result.topk_prompt_logprobs:
        for pos_topk in sample_result.topk_prompt_logprobs:
            if pos_topk:
                vllm_topk_data.append([(tok, lp) for tok, lp in pos_topk])
            else:
                vllm_topk_data.append(None)

    # Save comparison data
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Pos':<4} {'Target':<15} {'Meg Fresh':<12} {'Meg Train':<12} {'vLLM Train':<12} {'M-V Diff':<12}")
    print("-" * 70)

    corrupted = []
    for i in range(min(35, len(trained_logprobs))):
        target_str = repr(tokenizer.decode([target_tokens[i]]))[:12]
        mf = fresh_logprobs[i]
        mt = trained_logprobs[i]
        vt = vllm_logprobs[i] if i < len(vllm_logprobs) else float('nan')
        diff = mt - vt if not np.isnan(vt) else float('nan')

        marker = ""
        if not np.isnan(diff) and abs(diff) > 1.0:
            marker = " ***"
            corrupted.append(i)

        print(f"{i:<4} {target_str:<15} {mf:<12.4f} {mt:<12.4f} {vt:<12.4f} {diff:<+12.4f}{marker}")

    print(f"\nCorrupted positions (|diff| > 1.0): {corrupted}")
    print(f"\nCheckpoint URI: {checkpoint_path}")

    # Save data for later analysis
    import json
    data = {
        "checkpoint_path": checkpoint_path,
        "corrupted_positions": corrupted,
        "megatron_fresh": fresh_logprobs.tolist(),
        "megatron_trained": trained_logprobs.tolist(),
        "vllm_trained": vllm_logprobs.tolist(),
        "vllm_topk": vllm_topk_data,
        "target_tokens": target_tokens,
        "input_tokens": input_tokens,
    }
    with open("/tmp/debug_checkpoint_data.json", "w") as f:
        json.dump(data, f)
    print(f"Data saved to: /tmp/debug_checkpoint_data.json")


if __name__ == "__main__":
    asyncio.run(main())
