#!/usr/bin/env python3
"""Debug the exact point of divergence between Megatron and vLLM.

Hypothesis: The issue is in how shared LoRA vs per-expert LoRA affects different tokens.

This script:
1. Checks which positions route to which experts (if we can get that info)
2. Looks at the pattern of divergent vs non-divergent positions
3. Tests if the issue correlates with token semantics or position
"""

import asyncio
import os
import json
from datetime import datetime

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker
import torch
import numpy as np

MODEL_NAME = "moonshotai/Moonlight-16B-A3B-Instruct"

# Test sequences to understand the pattern
TEST_SEQUENCES = {
    "simple": """<|im_start|>user
Count: 1 2 3<|im_end|>
<|im_start|>assistant
4<|im_end|>""",

    "repetitive": """<|im_start|>user
A A A A<|im_end|>
<|im_start|>assistant
A<|im_end|>""",

    "diverse": """<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
4<|im_end|>""",
}


async def test_sequence(service_client, tokenizer, name, text, train_steps=1):
    """Test a single sequence."""
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"{'='*70}")

    tokens = tokenizer.encode(text, add_special_tokens=False)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    mask = [1.0] * len(input_tokens)

    print(f"Sequence length: {len(input_tokens)}")

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "mask": tinker.TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "advantages": tinker.TensorData.from_torch(torch.ones(len(input_tokens), dtype=torch.float32)),
            "logprobs": tinker.TensorData.from_torch(torch.zeros(len(input_tokens), dtype=torch.float32)),
        }
    )

    # Create fresh LoRA
    client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)

    # Get fresh logprobs
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    fresh_mega_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    # Train
    for step in range(train_steps):
        fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
        await fwd_bwd.result_async()
        await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()

    # Get trained logprobs
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    trained_mega_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    # Export to vLLM
    sampling_client = await client.save_weights_and_get_sampling_client_async()
    await asyncio.sleep(3)

    prompt = tinker.ModelInput.from_ints(tokens)
    trained_vllm_lp = await sampling_client.compute_logprobs_async(prompt)

    # Analyze
    degraded = []
    improved = []

    for i in range(len(trained_mega_lp)):
        m_delta = trained_mega_lp[i] - fresh_mega_lp[i]
        v_lp = trained_vllm_lp[i + 1] if i + 1 < len(trained_vllm_lp) and trained_vllm_lp[i + 1] is not None else np.nan
        diff = trained_mega_lp[i] - v_lp if not np.isnan(v_lp) else np.nan

        tok = tokenizer.decode([target_tokens[i]]) if i < len(target_tokens) else "?"

        if m_delta < -3:  # Megatron degraded
            degraded.append({
                'pos': i,
                'token': tok,
                'm_delta': m_delta,
                'm_fresh': fresh_mega_lp[i],
                'm_train': trained_mega_lp[i],
                'v_train': v_lp,
                'diff': diff
            })
        elif m_delta > 0.5:  # Megatron improved
            improved.append({
                'pos': i,
                'token': tok,
                'm_delta': m_delta,
                'm_fresh': fresh_mega_lp[i],
                'm_train': trained_mega_lp[i],
                'v_train': v_lp,
                'diff': diff
            })

    print(f"\nDegraded positions (M_delta < -3): {len(degraded)}")
    for d in degraded[:5]:
        print(f"  pos {d['pos']}: {repr(d['token']):<15} M: {d['m_fresh']:.2f} -> {d['m_train']:.2f} (delta={d['m_delta']:+.2f}), V: {d['v_train']:.2f}, diff={d['diff']:+.2f}")

    print(f"\nImproved positions (M_delta > 0.5): {len(improved)}")
    for d in improved[:5]:
        print(f"  pos {d['pos']}: {repr(d['token']):<15} M: {d['m_fresh']:.2f} -> {d['m_train']:.2f} (delta={d['m_delta']:+.2f}), V: {d['v_train']:.2f}, diff={d['diff']:+.2f}")

    return {
        'name': name,
        'degraded': degraded,
        'improved': improved,
        'total_positions': len(trained_mega_lp)
    }


async def main():
    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    results = []
    for name, text in TEST_SEQUENCES.items():
        result = await test_sequence(service_client, tokenizer, name, text)
        results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for r in results:
        print(f"\n{r['name']}: {len(r['degraded'])} degraded, {len(r['improved'])} improved out of {r['total_positions']} positions")

        # Analyze token patterns
        if r['degraded']:
            degraded_tokens = [d['token'] for d in r['degraded']]
            print(f"  Degraded tokens: {degraded_tokens}")

    print("\n" + "=" * 70)
    print("KEY QUESTION")
    print("=" * 70)
    print("""
If the weights are identical between Megatron and vLLM:
- Why does Megatron show degradation at certain positions?
- Why does vLLM show improvement/stability at the same positions?

This cannot be explained by weight differences - it must be architectural.
The divergence happens in the FORWARD PASS with identical weights.
""")


if __name__ == "__main__":
    asyncio.run(main())
