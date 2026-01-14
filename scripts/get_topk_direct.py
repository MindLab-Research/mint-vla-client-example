#!/usr/bin/env python3
"""Get top-K tokens by calling API directly (bypassing SDK type validation)."""

import asyncio
import json
import httpx

MODEL_NAME = "moonshotai/Moonlight-16B-A3B-Instruct"
CHECKPOINT_PATH = "tinker://localhost/vePFS-Mindverse/share/code/tinker-server/checkpoints/d01393b8-7dcb-47b1-a95c-477e81b22498_0/debug_checkpoint"
BASE_URL = "http://localhost:8000"


async def main():
    from transformers import AutoTokenizer

    # Load saved data
    print("Loading saved data...")
    with open("/tmp/debug_checkpoint_data.json", "r") as f:
        data = json.load(f)

    input_tokens = data["input_tokens"]
    target_tokens = data["target_tokens"]
    corrupted_positions = data["corrupted_positions"]
    megatron_trained = data["megatron_trained"]
    vllm_trained = data["vllm_trained"]

    print(f"Input tokens: {len(input_tokens)}")
    print(f"Corrupted positions: {corrupted_positions}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    async with httpx.AsyncClient(timeout=300.0) as client:
        # Create session
        print("\nCreating session...")
        resp = await client.post(f"{BASE_URL}/api/v1/create_session", json={})
        session_id = resp.json()["session_id"]
        print(f"Session: {session_id}")

        # Create sampling session
        print("Creating sampling session...")
        resp = await client.post(f"{BASE_URL}/api/v1/create_sampling_session", json={
            "session_id": session_id,
            "base_model": MODEL_NAME,
            "model_path": CHECKPOINT_PATH,
        })
        sampling_session_id = resp.json()["sampling_session_id"]
        print(f"Sampling session: {sampling_session_id}")

        # Wait for vLLM
        print("Waiting for vLLM...")
        await asyncio.sleep(5)

        # Sample with top-K
        print("Querying with top-K...")
        resp = await client.post(f"{BASE_URL}/api/v1/asample", json={
            "sampling_session_id": sampling_session_id,
            "num_samples": 1,
            "prompt": {"chunks": [{"tokens": input_tokens}]},
            "sampling_params": {"max_tokens": 1, "temperature": 0.0},
            "include_prompt_logprobs": True,
            "topk_prompt_logprobs": 10,
        })
        print(f"Response status: {resp.status_code}")
        print(f"Response: {resp.text[:500]}")
        if resp.status_code != 200:
            print("Error in asample")
            return
        request_id = resp.json()["request_id"]
        print(f"Request ID: {request_id}")

        # Poll for result
        print("Polling for result...")
        for _ in range(60):
            resp = await client.post(f"{BASE_URL}/api/v1/retrieve_future", json={
                "request_id": request_id
            })
            if resp.status_code == 200:
                break
            await asyncio.sleep(1)
        else:
            print("Timeout waiting for result")
            return

        result = resp.json()
        vllm_topk = result.get("topk_prompt_logprobs", [])

        if not vllm_topk:
            print("ERROR: No top-K data returned")
            return

        print(f"Got {len(vllm_topk)} top-K entries")

    # Analyze corrupted positions
    print("\n" + "=" * 100)
    print("TOP-K ANALYSIS AT CORRUPTED POSITIONS")
    print("=" * 100)

    for pos in corrupted_positions:
        if pos >= len(target_tokens):
            continue

        target = target_tokens[pos]
        target_str = tokenizer.decode([target])

        meg_t = megatron_trained[pos] if pos < len(megatron_trained) else float('nan')
        vllm_t = vllm_trained[pos] if pos < len(vllm_trained) else float('nan')

        print(f"\n{'='*80}")
        print(f"POSITION {pos}: target={target} {repr(target_str)}")
        print(f"  Megatron trained logprob: {meg_t:.4f}")
        print(f"  vLLM trained logprob:     {vllm_t:.4f}")
        print(f"  Diff (Meg - vLLM):        {meg_t - vllm_t:+.4f}")

        # Show context
        ctx_start = max(0, pos - 2)
        ctx_end = min(len(input_tokens), pos + 3)
        ctx_tokens = input_tokens[ctx_start:ctx_end]
        ctx_str = tokenizer.decode(ctx_tokens)
        print(f"  Context: ...{repr(ctx_str)}...")

        # vLLM top-K
        if pos < len(vllm_topk) and vllm_topk[pos]:
            pos_topk = vllm_topk[pos]
            # Keys are strings in JSON
            sorted_topk = sorted(pos_topk.items(), key=lambda x: x[1], reverse=True)

            print(f"\n  vLLM Top-10 at position {pos}:")
            for rank, (tok_id_str, lp) in enumerate(sorted_topk[:10], 1):
                tok_id = int(tok_id_str)
                tok_str = tokenizer.decode([tok_id])
                marker = " <-- TARGET" if tok_id == target else ""
                print(f"    {rank:2d}. {tok_id:6d} {repr(tok_str):15s}: {lp:8.4f}{marker}")

            target_key = str(target)
            if target_key in pos_topk:
                rank = [int(t) for t, _ in sorted_topk].index(target) + 1
                print(f"\n  Target {repr(target_str)} in vLLM top-10: YES, rank={rank}, logprob={pos_topk[target_key]:.4f}")
            else:
                print(f"\n  Target {repr(target_str)} in vLLM top-10: NO")

            # Show what vLLM thinks is #1
            argmax_tok_str, argmax_lp = sorted_topk[0]
            argmax_tok = int(argmax_tok_str)
            argmax_str = tokenizer.decode([argmax_tok])
            print(f"  vLLM argmax: {argmax_tok} {repr(argmax_str)} (logprob={argmax_lp:.4f})")
        else:
            print(f"\n  vLLM top-K: not available for this position")


if __name__ == "__main__":
    asyncio.run(main())
