#!/usr/bin/env python3
"""Test: Does reloading checkpoint into fresh Megatron reproduce the issue?

This tests whether the garbage logprobs are in the WEIGHTS or in hidden STATE.

1. Train LoRA for 1 step
2. Save checkpoint
3. Create NEW fresh Megatron actor
4. Load checkpoint into fresh actor
5. Compare logprobs

If fresh actor + loaded checkpoint shows garbage: issue is in WEIGHTS
If fresh actor + loaded checkpoint shows correct: issue is in hidden STATE
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

    # Key positions to track
    key_positions = [7, 8, 10, 18, 19]

    # ===============================================================
    # PHASE 1: Create and train
    # ===============================================================
    print("\n" + "=" * 70)
    print("PHASE 1: Create first actor and train 1 step")
    print("=" * 70)

    client1 = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)
    print(f"Actor 1 Model ID: {client1.model_id}")

    # Fresh logprobs
    fwd = await client1.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    fresh_lp = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    print("\nFresh logprobs at key positions:")
    for pos in key_positions:
        print(f"  pos {pos}: {fresh_lp[pos]:.4f}")

    # Train 1 step
    fwd_bwd = await client1.forward_backward_async([datum], loss_fn="importance_sampling")
    await fwd_bwd.result_async()
    await (await client1.optim_step_async(adam_params=tinker.AdamParams(learning_rate=1e-3))).result_async()

    # Trained logprobs in original actor
    fwd = await client1.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    trained_lp_actor1 = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    print("\nTrained logprobs (original actor) at key positions:")
    for pos in key_positions:
        delta = trained_lp_actor1[pos] - fresh_lp[pos]
        print(f"  pos {pos}: {trained_lp_actor1[pos]:.4f} (delta={delta:+.2f})")

    # ===============================================================
    # PHASE 2: Save checkpoint
    # ===============================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Save checkpoint")
    print("=" * 70)

    checkpoint_name = f"hidden_state_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_result = await (await client1.save_state_async(name=checkpoint_name)).result_async()
    checkpoint_path = save_result.path
    print(f"Saved checkpoint: {checkpoint_path}")

    # ===============================================================
    # PHASE 3: Create fresh actor and load checkpoint
    # ===============================================================
    print("\n" + "=" * 70)
    print("PHASE 3: Create FRESH actor and load checkpoint")
    print("=" * 70)

    client2 = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)
    print(f"Actor 2 Model ID: {client2.model_id}")

    # Fresh logprobs from new actor
    fwd = await client2.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    fresh_lp_actor2 = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    print("\nFresh logprobs (new actor) at key positions:")
    for pos in key_positions:
        diff_from_actor1 = fresh_lp_actor2[pos] - fresh_lp[pos]
        print(f"  pos {pos}: {fresh_lp_actor2[pos]:.4f} (diff from actor1={diff_from_actor1:+.4f})")

    # Load checkpoint
    load_result = await (await client2.load_state_async(path=checkpoint_path)).result_async()
    print(f"\nLoaded checkpoint into fresh actor")

    # Logprobs after loading
    fwd = await client2.forward_async([datum], loss_fn="importance_sampling")
    result = await fwd.result_async()
    loaded_lp_actor2 = np.array(result.loss_fn_outputs[0]["logprobs"].to_numpy())

    print("\nLogprobs (fresh actor + loaded checkpoint) at key positions:")
    for pos in key_positions:
        delta_from_fresh = loaded_lp_actor2[pos] - fresh_lp_actor2[pos]
        diff_from_trained = loaded_lp_actor2[pos] - trained_lp_actor1[pos]
        print(f"  pos {pos}: {loaded_lp_actor2[pos]:.4f} (delta from fresh={delta_from_fresh:+.2f}, diff from trained actor1={diff_from_trained:+.4f})")

    # ===============================================================
    # PHASE 4: Export to vLLM
    # ===============================================================
    print("\n" + "=" * 70)
    print("PHASE 4: Export to vLLM")
    print("=" * 70)

    sampling_client = await client2.save_weights_and_get_sampling_client_async()
    await asyncio.sleep(2)

    prompt = tinker.ModelInput.from_ints(tokens)
    vllm_lp = await sampling_client.compute_logprobs_async(prompt)

    print("\nvLLM logprobs at key positions:")
    for pos in key_positions:
        vllm_val = vllm_lp[pos + 1] if pos + 1 < len(vllm_lp) and vllm_lp[pos + 1] is not None else float('nan')
        diff_from_mega = loaded_lp_actor2[pos] - vllm_val if not np.isnan(vllm_val) else float('nan')
        print(f"  pos {pos}: vLLM={vllm_val:.4f}, Megatron={loaded_lp_actor2[pos]:.4f}, diff={diff_from_mega:+.2f}")

    # ===============================================================
    # SUMMARY
    # ===============================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Position':<10} {'Fresh':<12} {'Trained(A1)':<14} {'Fresh(A2)':<12} {'Loaded(A2)':<14} {'vLLM':<12}")
    print("-" * 74)
    for pos in key_positions:
        f = fresh_lp[pos]
        t1 = trained_lp_actor1[pos]
        f2 = fresh_lp_actor2[pos]
        l2 = loaded_lp_actor2[pos]
        v = vllm_lp[pos + 1] if pos + 1 < len(vllm_lp) and vllm_lp[pos + 1] is not None else float('nan')
        print(f"{pos:<10} {f:<12.4f} {t1:<14.4f} {f2:<12.4f} {l2:<14.4f} {v:<12.4f}")

    # Analyze
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    actor1_vs_actor2_loaded = np.max([abs(trained_lp_actor1[p] - loaded_lp_actor2[p]) for p in key_positions])
    mega_vs_vllm = np.max([abs(loaded_lp_actor2[p] - (vllm_lp[p+1] if p+1 < len(vllm_lp) and vllm_lp[p+1] is not None else loaded_lp_actor2[p])) for p in key_positions])

    if actor1_vs_actor2_loaded < 0.1:
        print(f"\nActor1 trained ≈ Actor2 loaded (max diff = {actor1_vs_actor2_loaded:.4f})")
        print("→ Issue is in WEIGHTS, not hidden state")
    else:
        print(f"\nActor1 trained ≠ Actor2 loaded (max diff = {actor1_vs_actor2_loaded:.4f})")
        print("→ Issue might be in hidden STATE not exported")

    if mega_vs_vllm > 5:
        print(f"\nMegatron ≠ vLLM (max diff = {mega_vs_vllm:.2f})")
        print("→ Same weights produce different results in different systems")
        print("→ This confirms ARCHITECTURAL difference in how LoRA is applied")


if __name__ == "__main__":
    asyncio.run(main())
