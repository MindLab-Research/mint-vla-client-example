#!/usr/bin/env python3
"""Controlled experiment for KL divergence investigation.

Run 10 training steps and log per-token logprobs for analysis.
Compare vLLM and Megatron logprobs on the SAME training data.
"""

import asyncio
import json
import os
import sys
from datetime import datetime

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker
import torch
import numpy as np
from transformers import AutoTokenizer


async def main():
    model_name = "moonshotai/Moonlight-16B-A3B-Instruct"
    num_steps = 10

    # Get experiment config from env
    hollowman = os.environ.get("USE_HOLLOWMAN_MBRIDGE", "false").lower() in ("true", "1", "yes")
    mbridge_export = os.environ.get("USE_MBRIDGE_LORA_EXPORT", "false").lower() in ("true", "1", "yes")

    exp_name = f"hollowman={hollowman}_mbridge_export={mbridge_export}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"/tmp/kl_experiment_{exp_name}_{timestamp}.jsonl"

    print(f"=" * 60)
    print(f"CONTROLLED KL EXPERIMENT")
    print(f"=" * 60)
    print(f"USE_HOLLOWMAN_MBRIDGE: {hollowman}")
    print(f"USE_MBRIDGE_LORA_EXPORT: {mbridge_export}")
    print(f"Log file: {log_file}")
    print(f"=" * 60)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Create training session
    print("\nCreating training session...")
    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])
    training_client = await service_client.create_lora_training_client_async(model_name, rank=16)

    # Training data - use this for BOTH training and comparison
    prompt = "<|im_start|>user\nCalculate 7+8<|im_end|>\n<|im_start|>assistant\n"
    response = "The answer is 15.<|im_end|>"
    full_text = prompt + response
    tokens = tokenizer.encode(full_text, add_special_tokens=False)
    prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))

    input_tokens = tokens[:-1]  # All but last token
    target_tokens = tokens[1:]  # All but first token (shifted by 1)
    mask = [0.0] * (prompt_len - 1) + [1.0] * (len(input_tokens) - prompt_len + 1)

    # Create datum for training
    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.ones(len(input_tokens), dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.zeros(len(input_tokens), dtype=torch.float32)),
        }
    )

    # Create input for vLLM logprob comparison (same as training input)
    vllm_input = tinker.ModelInput.from_ints(input_tokens)

    results = []
    action_start = prompt_len - 1

    print(f"\nTraining sequence length: {len(input_tokens)} tokens")
    print(f"Action tokens start at position: {action_start}")
    print(f"\nTokens:")
    for i, tok in enumerate(input_tokens):
        marker = "<-- action start" if i == action_start else ""
        print(f"  {i:3d}: {tok:6d} = {repr(tokenizer.decode([tok]))} {marker}")

    for step in range(num_steps):
        print(f"\n{'='*60}")
        print(f"STEP {step}")
        print(f"{'='*60}")

        # Forward-backward to get Megatron logprobs
        fwd_bwd = await training_client.forward_backward_async([datum], loss_fn="importance_sampling")
        fwd_bwd_result = await fwd_bwd.result_async()
        megatron_lp = fwd_bwd_result.loss_fn_outputs[0]["logprobs"].to_numpy()

        # Optim step
        optim = await training_client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=5e-5))
        await optim.result_async()

        # Export to vLLM and get logprobs on SAME input
        sampling_client = await training_client.save_weights_and_get_sampling_client_async()
        vllm_lp = await sampling_client.compute_logprobs_async(vllm_input)
        vllm_lp = np.array(vllm_lp)

        # Compute per-token comparison for action tokens
        print(f"\nPer-token logprobs (action tokens):")
        print(f"{'pos':>4} | {'token':>8} | {'decoded':>15} | {'vLLM':>12} | {'Megatron':>12} | {'diff':>12}")
        print("-" * 80)

        diffs = []
        for i in range(action_start, len(input_tokens)):
            token_id = input_tokens[i]
            decoded = tokenizer.decode([token_id])
            v_lp = vllm_lp[i] if i < len(vllm_lp) else float('nan')
            m_lp = megatron_lp[i] if i < len(megatron_lp) else float('nan')
            diff = v_lp - m_lp
            diffs.append(diff)
            print(f"{i:4d} | {token_id:8d} | {repr(decoded):>15} | {v_lp:12.6f} | {m_lp:12.6f} | {diff:+12.6f}")

        kl_mean = np.mean(diffs)
        kl_max = np.max(np.abs(diffs))
        print(f"\nKL (mean): {kl_mean:.6f}")
        print(f"KL (max abs diff): {kl_max:.6f}")

        # Store result
        step_result = {
            "step": step,
            "config": {
                "hollowman": hollowman,
                "mbridge_export": mbridge_export,
            },
            "input_tokens": input_tokens,
            "vllm_logprobs": vllm_lp.tolist(),
            "megatron_logprobs": megatron_lp.tolist(),
            "kl_mean": float(kl_mean),
            "kl_max_abs": float(kl_max),
            "per_token_diffs": diffs,
        }
        results.append(step_result)

        # Save incrementally
        with open(log_file, "a") as f:
            f.write(json.dumps(step_result) + "\n")

    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Results saved to: {log_file}")
    print(f"{'=' * 60}")

    # Summary
    print("\nKL progression:")
    print(f"{'Step':>4} | {'KL Mean':>12} | {'KL Max Abs':>12}")
    print("-" * 35)
    for r in results:
        print(f"{r['step']:4d} | {r['kl_mean']:12.6f} | {r['kl_max_abs']:12.6f}")


if __name__ == "__main__":
    asyncio.run(main())
