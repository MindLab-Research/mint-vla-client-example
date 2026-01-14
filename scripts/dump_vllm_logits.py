#!/usr/bin/env python3
"""Dump raw logits from vLLM with trained LoRA for comparison with Megatron."""

import asyncio
import os
import json
import torch

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "dummy")

import tinker

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


async def main():
    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)
    input_tokens = tokens[:-1]  # 51 tokens
    target_tokens = tokens[1:]  # 51 tokens (shifted)

    print(f"Sequence length: {len(input_tokens)} input tokens")
    print(f"Input tokens: {input_tokens}")
    print(f"Target tokens: {target_tokens}")
    print()

    service_client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])

    # Create Megatron client with trained checkpoint
    print("Creating Megatron client and loading trained checkpoint...")
    meg_client = await service_client.create_lora_training_client_async(MODEL_NAME, rank=16)
    await (await meg_client.load_state_async(CHECKPOINT_PATH)).result_async()
    print("Checkpoint loaded into Megatron")

    # Export to vLLM
    print("\nExporting to vLLM...")
    vllm_client = await meg_client.save_weights_and_get_sampling_client_async()
    await asyncio.sleep(2)
    print("vLLM client ready")

    # Get prompt logprobs from vLLM
    print("\nRunning vLLM forward pass with prompt_logprobs...")
    sample_result = await vllm_client.sample_async(
        prompt=tinker.ModelInput.from_ints(input_tokens),
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=1,
            temperature=0.0,
        ),
        topk_prompt_logprobs=10,  # Get top-10 at each position
    )

    # Dump the results
    output = {
        "input_tokens": input_tokens,
        "target_tokens": target_tokens,
        "prompt_logprobs": sample_result.prompt_logprobs,
        "prompt_logprobs_full": sample_result.prompt_logprobs_full,
    }

    # Save to file
    output_path = "/vePFS-Mindverse/share/code/vllm_logits_dump.json"

    # Convert to serializable format
    serializable = {
        "input_tokens": input_tokens,
        "target_tokens": target_tokens,
        "positions": [],
    }

    if sample_result.prompt_logprobs_full:
        for pos, full_lp in enumerate(sample_result.prompt_logprobs_full):
            target = target_tokens[pos] if pos < len(target_tokens) else -1
            target_lp = sample_result.prompt_logprobs[pos] if pos < len(sample_result.prompt_logprobs) else None

            pos_data = {
                "pos": pos,
                "target": target,
                "target_lp": target_lp,
                "topk": [],
            }

            if full_lp:
                # full_lp is dict {token_id: logprob}
                sorted_items = sorted(full_lp.items(), key=lambda x: -x[1])
                for tok_id, logprob in sorted_items[:10]:
                    pos_data["topk"].append({"tok": tok_id, "logit": logprob})

            serializable["positions"].append(pos_data)

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nSaved vLLM logits to {output_path}")

    # Print comparison at problematic positions
    print("\n" + "=" * 60)
    print("vLLM top-K at problematic positions:")
    print("=" * 60)
    problematic = [5, 14, 21, 23, 29]
    for pos in problematic:
        if pos < len(serializable["positions"]):
            p = serializable["positions"][pos]
            target_tok = p["target"]
            target_lp = p["target_lp"]
            topk = p["topk"]

            target_str = tokenizer.decode([target_tok]) if target_tok >= 0 else "N/A"
            print(f"\nPosition {pos} (target={target_tok} '{target_str}', lp={target_lp:.4f if target_lp else 'N/A'}):")

            for i, t in enumerate(topk[:5]):
                tok_str = tokenizer.decode([t["tok"]])
                marker = " <-- TARGET" if t["tok"] == target_tok else ""
                print(f"  {i+1}. token={t['tok']:6d} '{tok_str:15s}' logit={t['logit']:.4f}{marker}")


if __name__ == "__main__":
    asyncio.run(main())
