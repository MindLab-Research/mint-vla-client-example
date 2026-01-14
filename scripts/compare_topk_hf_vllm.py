#!/usr/bin/env python3
"""Compare top-K tokens from HuggingFace (with LoRA) vs vLLM (with LoRA).

Uses the checkpoint to load LoRA weights into both systems and compare top-K.
This bypasses Megatron to directly compare weight application.

Usage:
    python scripts/compare_topk_hf_vllm.py

Requires:
    - Saved debug data from test_megatron_vllm_logprob_mismatch.py
    - SSH tunnel to volcano for API access
"""

import asyncio
import json
import os
import sys

import numpy as np
import torch

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

MODEL_NAME = "moonshotai/Moonlight-16B-A3B-Instruct"

# Find the latest debug data
DEBUG_DIR = "/tmp"
CHECKPOINT_PATH = None


def find_latest_debug_file():
    """Find the most recent debug JSON file."""
    import glob
    files = glob.glob(f"{DEBUG_DIR}/megatron_vllm_debug_*.json")
    if not files:
        return None
    return max(files, key=os.path.getmtime)


async def get_vllm_topk_direct(input_tokens: list[int], checkpoint_path: str, k: int = 10):
    """Get vLLM top-K by calling API directly (bypassing SDK type validation).

    First tries to use the checkpoint path. If that fails (tinker:// paths not supported),
    falls back to using the training session's model_id which should already be registered.
    """
    import httpx

    base_url = os.environ["TINKER_BASE_URL"]

    async with httpx.AsyncClient(timeout=300.0) as client:
        # First try: Use existing model_id from training session
        # The model_id is embedded in the checkpoint path
        # e.g., tinker://local/80e6bf97-56d9-4f3a-9872-7cc4b80a7379_0/debug_checkpoint_20260110_182006
        model_id = None
        if "tinker://local/" in checkpoint_path:
            parts = checkpoint_path.replace("tinker://local/", "").split("/")
            if parts:
                model_id = parts[0]  # e.g., "80e6bf97-56d9-4f3a-9872-7cc4b80a7379_0"
                print(f"Extracted model_id: {model_id}")

        # Try using model_id directly (already registered from save_weights)
        if model_id:
            print(f"Trying to use existing registered model_id: {model_id}")
            resp = await client.post(f"{base_url}/api/v1/asample", json={
                "model_id": model_id,
                "num_samples": 1,
                "prompt": {"chunks": [{"tokens": input_tokens}]},
                "sampling_params": {"max_tokens": 1, "temperature": 0.0},
                "include_prompt_logprobs": True,
                "topk_prompt_logprobs": k,
            })
            if resp.status_code == 200:
                request_id = resp.json()["request_id"]
                print(f"Request submitted: {request_id}")

                # Poll for result
                for _ in range(60):
                    resp = await client.post(f"{base_url}/api/v1/retrieve_future", json={
                        "request_id": request_id
                    })
                    if resp.status_code == 200:
                        break
                    await asyncio.sleep(1)
                else:
                    print("Timeout waiting for result")
                    return None, None

                result = resp.json()
                logprobs = result.get("prompt_logprobs", [])
                topk = result.get("topk_prompt_logprobs", [])
                return logprobs, topk
            else:
                print(f"model_id request failed: {resp.text[:200]}")

        # Fallback: Create new session
        print("Creating new sampling session...")
        resp = await client.post(f"{base_url}/api/v1/create_session", json={})
        session_id = resp.json()["session_id"]

        # Resolve checkpoint path for local filesystem
        resolved_path = checkpoint_path
        if checkpoint_path.startswith("tinker://local/"):
            resolved_path = "/vePFS-Mindverse/share/code/tinker-server/checkpoints/" + checkpoint_path.replace("tinker://local/", "")
        elif checkpoint_path.startswith("tinker://localhost"):
            resolved_path = checkpoint_path.replace("tinker://localhost", "")

        print(f"Resolved path: {resolved_path}")

        resp = await client.post(f"{base_url}/api/v1/create_sampling_session", json={
            "session_id": session_id,
            "base_model": MODEL_NAME,
            "model_path": resolved_path,  # Use resolved filesystem path
        })
        resp_data = resp.json()
        print(f"create_sampling_session response: {resp_data}")

        if "detail" in resp_data:
            print(f"Error creating sampling session: {resp_data['detail']}")
            return None, None

        sampling_session_id = resp_data.get("sampling_session_id") or resp_data.get("model_id") or session_id

        # Wait for vLLM
        await asyncio.sleep(5)

        # Sample with top-K
        resp = await client.post(f"{base_url}/api/v1/asample", json={
            "sampling_session_id": sampling_session_id,
            "num_samples": 1,
            "prompt": {"chunks": [{"tokens": input_tokens}]},
            "sampling_params": {"max_tokens": 1, "temperature": 0.0},
            "include_prompt_logprobs": True,
            "topk_prompt_logprobs": k,
        })
        if resp.status_code != 200:
            print(f"Error in asample: {resp.text}")
            return None, None
        request_id = resp.json()["request_id"]

        # Poll for result
        for _ in range(60):
            resp = await client.post(f"{base_url}/api/v1/retrieve_future", json={
                "request_id": request_id
            })
            if resp.status_code == 200:
                break
            await asyncio.sleep(1)
        else:
            print("Timeout waiting for result")
            return None, None

        result = resp.json()
        logprobs = result.get("prompt_logprobs", [])
        topk = result.get("topk_prompt_logprobs", [])

        return logprobs, topk


def get_hf_topk(input_tokens: list[int], checkpoint_path: str, k: int = 10):
    """Get top-K from HuggingFace model with LoRA loaded.

    Note: This requires significant GPU memory for Moonlight-16B.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("Loading HuggingFace model (this may take a while)...")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA adapter
    if checkpoint_path:
        # Resolve checkpoint path
        if checkpoint_path.startswith("tinker://local/"):
            checkpoint_path = checkpoint_path.replace(
                "tinker://local/",
                "/vePFS-Mindverse/share/code/tinker-server/checkpoints/"
            )
        elif checkpoint_path.startswith("tinker://localhost"):
            checkpoint_path = checkpoint_path.replace("tinker://localhost", "")

        print(f"Loading LoRA from: {checkpoint_path}")
        model = PeftModel.from_pretrained(model, checkpoint_path)

    model.eval()

    # Run forward pass
    input_ids = torch.tensor([input_tokens], dtype=torch.long, device=model.device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    # Get top-K for each position
    # logits[0, i, :] gives logits for predicting token at position i+1
    seq_len = logits.shape[1]
    topk_results = []
    logprobs_results = []

    for i in range(seq_len - 1):
        # Get top-K tokens and their logprobs
        pos_logits = logits[0, i, :]  # [vocab_size]
        log_probs = torch.log_softmax(pos_logits, dim=-1)

        topk_vals, topk_idx = torch.topk(log_probs, k)

        topk_dict = {int(idx): float(val) for idx, val in zip(topk_idx, topk_vals)}
        topk_results.append(topk_dict)

        # Get logprob of the actual next token
        if i + 1 < len(input_tokens):
            next_token = input_tokens[i + 1]
            logprobs_results.append(float(log_probs[next_token]))

    return logprobs_results, topk_results


async def main():
    from transformers import AutoTokenizer

    # Find latest debug file
    debug_file = find_latest_debug_file()
    if debug_file is None:
        print("No debug file found. Run test_megatron_vllm_logprob_mismatch.py first.")
        return

    print(f"Loading debug data from: {debug_file}")
    with open(debug_file, "r") as f:
        data = json.load(f)

    checkpoint_path = data["checkpoint_path"]
    input_tokens = data["input_tokens"]
    target_tokens = data["target_tokens"]
    problematic_positions = data["problematic_positions"]
    megatron_trained = data["megatron_trained"]

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Input tokens: {len(input_tokens)}")
    print(f"Problematic positions: {problematic_positions}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Get vLLM top-K
    print("\n" + "=" * 70)
    print("Getting vLLM top-K (bypassing SDK)")
    print("=" * 70)
    vllm_logprobs, vllm_topk = await get_vllm_topk_direct(input_tokens, checkpoint_path)

    if vllm_topk is None:
        print("Failed to get vLLM top-K")
        return

    print(f"Got {len(vllm_topk)} top-K entries from vLLM")

    # Compare at problematic positions
    # Remember: megatron[i] predicts target_tokens[i] = input_tokens[i+1]
    # vLLM topk[i+1] predicts input_tokens[i+1]
    print("\n" + "=" * 100)
    print("TOP-K COMPARISON AT PROBLEMATIC POSITIONS")
    print("Alignment: Megatron[i] and vLLM[i+1] both predict target_tokens[i]")
    print("=" * 100)

    for pos in problematic_positions[:5]:  # First 5 problematic positions
        if pos >= len(target_tokens):
            continue

        target = target_tokens[pos]
        target_str = tokenizer.decode([target])

        vllm_idx = pos + 1  # Correct alignment
        meg_logprob = megatron_trained[pos]

        print(f"\n{'='*80}")
        print(f"Position {pos}: target={target} '{target_str}'")
        print(f"  Megatron trained logprob: {meg_logprob:.4f}")

        if vllm_idx < len(vllm_topk) and vllm_topk[vllm_idx]:
            pos_topk = vllm_topk[vllm_idx]
            sorted_topk = sorted(pos_topk.items(), key=lambda x: float(x[1]), reverse=True)

            print(f"\n  vLLM Top-10 at position {vllm_idx}:")
            for rank, (tok_id_str, lp) in enumerate(sorted_topk[:10], 1):
                tok_id = int(tok_id_str)
                tok_str = tokenizer.decode([tok_id])
                marker = " <-- TARGET" if tok_id == target else ""
                print(f"    {rank:2d}. {tok_id:6d} ({repr(tok_str):15s}): {float(lp):8.4f}{marker}")

            target_key = str(target)
            if target_key in pos_topk:
                vllm_target_logprob = float(pos_topk[target_key])
                print(f"\n  Target '{target_str}' in vLLM top-10: YES, logprob={vllm_target_logprob:.4f}")
                print(f"  Diff (Megatron - vLLM): {meg_logprob - vllm_target_logprob:+.4f}")
            else:
                print(f"\n  Target '{target_str}' NOT in vLLM top-10")
                # Get vLLM's actual logprob for target if available in prompt_logprobs
                if vllm_logprobs and vllm_idx < len(vllm_logprobs):
                    vllm_target_logprob = vllm_logprobs[vllm_idx]
                    print(f"  vLLM target logprob (from prompt_logprobs): {vllm_target_logprob:.4f}")
                    print(f"  Diff (Megatron - vLLM): {meg_logprob - vllm_target_logprob:+.4f}")
        else:
            print(f"\n  vLLM top-K not available for position {vllm_idx}")


if __name__ == "__main__":
    asyncio.run(main())
