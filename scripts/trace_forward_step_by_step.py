#!/usr/bin/env python3
"""Trace EXACT computation in Megatron forward pass to find the bug.

The bug: Same weights produce correct results in vLLM but wrong results in Megatron.
This means the forward pass computation is different.

We need to find WHERE in the computation the divergence begins.
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
    """
    INVESTIGATION PLAN:

    1. Get fresh LoRA checkpoint (should match between systems)
    2. Get trained checkpoint (shows divergence)
    3. For trained checkpoint, trace:
       a. What are the actual LoRA weight values?
       b. What is the base model output at divergent positions?
       c. What is the LoRA delta at divergent positions?
       d. Are they being combined correctly?

    KEY HYPOTHESIS TO TEST:
    The LoRA delta might be correct, but applied to wrong positions due to
    token permutation in MoE.
    """

    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]

    print(f"Full sequence: {tokens}")
    print(f"Input tokens: {input_tokens}")
    print(f"Target tokens: {target_tokens}")

    # Identify divergent positions
    improving_positions = [0, 1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16, 17]
    degrading_positions = [7, 8, 10, 18, 19]

    print(f"\nImproving positions: {improving_positions}")
    print(f"Degrading positions: {degrading_positions}")

    print("\n" + "=" * 70)
    print("QUESTION: What makes degrading positions special?")
    print("=" * 70)

    print("\nDegrading position tokens:")
    for pos in degrading_positions:
        input_tok = input_tokens[pos] if pos < len(input_tokens) else "N/A"
        target_tok = target_tokens[pos] if pos < len(target_tokens) else "N/A"
        input_str = tokenizer.decode([input_tok]) if isinstance(input_tok, int) else "N/A"
        target_str = tokenizer.decode([target_tok]) if isinstance(target_tok, int) else "N/A"
        print(f"  pos {pos}: input={repr(input_str)} -> target={repr(target_str)}")

    print("\nImproving position tokens:")
    for pos in improving_positions[:5]:  # First 5 for brevity
        input_tok = input_tokens[pos] if pos < len(input_tokens) else "N/A"
        target_tok = target_tokens[pos] if pos < len(target_tokens) else "N/A"
        input_str = tokenizer.decode([input_tok]) if isinstance(input_tok, int) else "N/A"
        target_str = tokenizer.decode([target_tok]) if isinstance(target_tok, int) else "N/A"
        print(f"  pos {pos}: input={repr(input_str)} -> target={repr(target_str)}")

    print("\n" + "=" * 70)
    print("PATTERN ANALYSIS")
    print("=" * 70)

    # Check if degrading positions have any pattern
    degrading_input_tokens = [input_tokens[p] for p in degrading_positions if p < len(input_tokens)]
    improving_input_tokens = [input_tokens[p] for p in improving_positions if p < len(input_tokens)]

    print(f"\nDegrading position input token IDs: {degrading_input_tokens}")
    print(f"Improving position input token IDs (first 10): {improving_input_tokens[:10]}")

    # Are there duplicate tokens?
    degrading_set = set(degrading_input_tokens)
    improving_set = set(improving_input_tokens)
    overlap = degrading_set & improving_set

    print(f"\nUnique degrading input tokens: {degrading_set}")
    print(f"Unique improving input tokens: {improving_set}")
    print(f"Overlap: {overlap}")

    if overlap:
        print("\n*** IMPORTANT: Same token appears in BOTH improving and degrading positions! ***")
        for tok in overlap:
            deg_pos = [p for p in degrading_positions if p < len(input_tokens) and input_tokens[p] == tok]
            imp_pos = [p for p in improving_positions if p < len(input_tokens) and input_tokens[p] == tok]
            print(f"  Token {tok} ({repr(tokenizer.decode([tok]))}): degrades at {deg_pos}, improves at {imp_pos}")

    print("\n" + "=" * 70)
    print("HYPOTHESIS: Position-dependent bug")
    print("=" * 70)
    print("""
If the same token degrades at some positions but improves at others,
the bug is NOT about the token itself but about the POSITION.

This suggests:
1. Token permutation in MoE is involved
2. Or there's an indexing bug in LoRA application
3. Or the expert assignment affects the computation

Next step: Need to check if the LoRA delta is being added to the
correct position in the output tensor.
""")

    # Run actual test to confirm the pattern
    print("\n" + "=" * 70)
    print("VERIFICATION: Training and checking the pattern")
    print("=" * 70)

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

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

    client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)

    # Fresh
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    fresh_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    # Get top-K from fresh forward
    if "topk_indices" in result.loss_fn_outputs[0]:
        fresh_topk = result.loss_fn_outputs[0]["topk_indices"].to_numpy()
        print(f"\nFresh top-K shape: {fresh_topk.shape}")

    # Train 1 step
    fwd_bwd = await client.forward_backward_async([datum], loss_fn="importance_sampling")
    await fwd_bwd.result_async()
    await (await client.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()

    # Trained
    fwd = await client.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    trained_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    if "topk_indices" in result.loss_fn_outputs[0]:
        trained_topk = result.loss_fn_outputs[0]["topk_indices"].to_numpy()
        print(f"Trained top-K shape: {trained_topk.shape}")

    # Analyze
    print("\n" + "=" * 70)
    print("RESULT ANALYSIS")
    print("=" * 70)

    print(f"\n{'Pos':<5} {'Input':<12} {'Target':<12} {'Fresh LP':<12} {'Trained LP':<12} {'Delta':<12}")
    print("-" * 75)

    for i in range(len(trained_lp)):
        input_str = tokenizer.decode([input_tokens[i]]) if i < len(input_tokens) else "?"
        target_str = tokenizer.decode([target_tokens[i]]) if i < len(target_tokens) else "?"
        delta = trained_lp[i] - fresh_lp[i]

        # Flag
        flag = ""
        if i in degrading_positions:
            flag = " <-- DEGRADES"
        elif i in improving_positions:
            flag = " (improves)"

        print(f"{i:<5} {repr(input_str):<12} {repr(target_str):<12} {fresh_lp[i]:<12.4f} {trained_lp[i]:<12.4f} {delta:<+12.4f}{flag}")

    # Check if specific token appears in both categories
    print("\n" + "=" * 70)
    print("CRITICAL TEST: Same token in different categories")
    print("=" * 70)

    # Token 198 is '\n' - appears at multiple positions
    newline_token = 198
    newline_positions = [i for i, t in enumerate(input_tokens) if t == newline_token]
    print(f"\nNewline (token {newline_token}) appears at positions: {newline_positions}")
    for pos in newline_positions:
        cat = "DEGRADES" if pos in degrading_positions else "improves"
        if pos < len(trained_lp):
            print(f"  pos {pos}: {cat}, fresh={fresh_lp[pos]:.4f}, trained={trained_lp[pos]:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
