#!/usr/bin/env python3
"""Deep analysis of the Megatron vs vLLM divergence.

This script investigates the exact mechanism causing divergence.
"""

import asyncio
import os
from datetime import datetime

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker
import torch
import numpy as np

MODEL_NAME = "moonshotai/Moonlight-16B-A3B-Instruct"

TEST_TEXT = """<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi<|im_end|>"""


async def main():
    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    mask = [1.0] * len(input_tokens)

    print(f"Sequence: {len(input_tokens)} tokens")
    print(f"Full token sequence: {tokens}")

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

    # =======================================================================
    # EXPERIMENT: Train on only structural tokens
    # =======================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT: Compare training progression")
    print("=" * 70)

    for step in range(10):
        client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)

        # Train for 'step' steps
        for _ in range(step):
            fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
            await fwd_bwd.result_async()
            await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()

        # Get Megatron logprobs
        fwd = await client.forward_async([datum], loss_fn="importance_sampling")
        result = await fwd.result_async()
        mega_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

        # Export to vLLM
        sampling_client = await client.save_weights_and_get_sampling_client_async()
        await asyncio.sleep(1)

        # Get vLLM logprobs (use full sequence)
        prompt = tinker.ModelInput.from_ints(tokens)
        vllm_lp_list = await sampling_client.compute_logprobs_async(prompt)

        # Key positions to track
        improving = [0, 1, 5, 9, 16]  # structural
        degrading = [7, 8, 10, 18, 19]  # content

        # Print summary
        improve_mega = np.mean([mega_lp[i] for i in improving if i < len(mega_lp)])
        degrade_mega = np.mean([mega_lp[i] for i in degrading if i < len(mega_lp)])

        improve_vllm = []
        degrade_vllm = []
        for i in improving:
            if i + 1 < len(vllm_lp_list) and vllm_lp_list[i + 1] is not None:
                improve_vllm.append(vllm_lp_list[i + 1])
        for i in degrading:
            if i + 1 < len(vllm_lp_list) and vllm_lp_list[i + 1] is not None:
                degrade_vllm.append(vllm_lp_list[i + 1])

        improve_vllm_mean = np.mean(improve_vllm) if improve_vllm else np.nan
        degrade_vllm_mean = np.mean(degrade_vllm) if degrade_vllm else np.nan

        print(f"Step {step}: Megatron(improving)={improve_mega:.2f}, Megatron(degrading)={degrade_mega:.2f}, "
              f"vLLM(improving)={improve_vllm_mean:.2f}, vLLM(degrading)={degrade_vllm_mean:.2f}")

    # =======================================================================
    # SINGLE STEP DETAILED ANALYSIS
    # =======================================================================
    print("\n" + "=" * 70)
    print("SINGLE STEP DETAILED ANALYSIS")
    print("=" * 70)

    client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)

    # Fresh
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    fresh_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    sampling_client_fresh = await client.save_weights_and_get_sampling_client_async()
    await asyncio.sleep(1)
    prompt = tinker.ModelInput.from_ints(tokens)
    vllm_fresh = await sampling_client_fresh.compute_logprobs_async(prompt)

    print("\nFresh LoRA (before training):")
    print(f"{'Pos':<5} {'Token':<15} {'Megatron':<12} {'vLLM':<12} {'Diff':<10}")
    print("-" * 60)
    for i in range(len(fresh_lp)):
        m = fresh_lp[i]
        v = vllm_fresh[i + 1] if i + 1 < len(vllm_fresh) and vllm_fresh[i + 1] is not None else np.nan
        diff = m - v if not np.isnan(v) else np.nan
        tok = tokenizer.decode([target_tokens[i]]) if i < len(target_tokens) else "?"
        print(f"{i:<5} {repr(tok):<15} {m:<12.4f} {v:<12.4f} {diff:<+10.4f}")

    # Train 1 step
    fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
    await fwd_bwd.result_async()
    await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()

    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    trained_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    sampling_client_trained = await client.save_weights_and_get_sampling_client_async()
    await asyncio.sleep(1)
    vllm_trained = await sampling_client_trained.compute_logprobs_async(prompt)

    print("\nAfter 1 training step:")
    print(f"{'Pos':<5} {'Token':<15} {'Megatron':<12} {'vLLM':<12} {'Diff':<10} {'M-change':<10}")
    print("-" * 75)
    for i in range(len(trained_lp)):
        m = trained_lp[i]
        v = vllm_trained[i + 1] if i + 1 < len(vllm_trained) and vllm_trained[i + 1] is not None else np.nan
        diff = m - v if not np.isnan(v) else np.nan
        m_change = m - fresh_lp[i]
        tok = tokenizer.decode([target_tokens[i]]) if i < len(target_tokens) else "?"
        flag = " ***" if abs(diff) > 5 else ""
        print(f"{i:<5} {repr(tok):<15} {m:<12.4f} {v:<12.4f} {diff:<+10.4f} {m_change:<+10.4f}{flag}")


if __name__ == "__main__":
    asyncio.run(main())
