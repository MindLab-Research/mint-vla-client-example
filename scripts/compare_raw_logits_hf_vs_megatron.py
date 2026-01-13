#!/usr/bin/env python3
"""Compare raw logits between HuggingFace and Megatron.

This script runs ON THE SERVER (volcano) to:
1. Show token analysis for the test sequence
2. Get raw logits from HuggingFace base model (ground truth)
3. Compare with Megatron's diagnostic output

Run on volcano: python3 /root/tinker_project/tinker-server/scripts/compare_raw_logits_hf_vs_megatron.py
"""

import torch

MODEL_NAME = "moonshotai/Moonlight-16B-A3B-Instruct"

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


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]

    print(f"\n{'='*60}")
    print("TOKEN ANALYSIS")
    print(f"{'='*60}")
    print(f"Total tokens: {len(tokens)}, Input: {len(input_tokens)}")

    # Show positions of interest
    for pos in [7, 8, 23]:
        if pos < len(target_tokens):
            print(f"\nPosition {pos}:")
            print(f"  Input token {input_tokens[pos]}: {repr(tokenizer.decode([input_tokens[pos]]))}")
            print(f"  Target token {target_tokens[pos]}: {repr(tokenizer.decode([target_tokens[pos]]))}")

    print(f"\n{'='*60}")
    print("FULL SEQUENCE (first 30 positions)")
    print(f"{'='*60}")
    for i in range(min(30, len(target_tokens))):
        inp_tok = input_tokens[i]
        tgt_tok = target_tokens[i]
        print(f"pos={i:2d}: input={inp_tok:5d} ({repr(tokenizer.decode([inp_tok])):12s}) -> target={tgt_tok:5d} ({repr(tokenizer.decode([tgt_tok]))})")

    print(f"\n{'='*60}")
    print("LOADING HF MODEL (base, no LoRA)")
    print(f"{'='*60}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    print(f"\n{'='*60}")
    print("RUNNING HF INFERENCE")
    print(f"{'='*60}")
    input_tensor = torch.tensor([input_tokens], device=model.device)

    with torch.no_grad():
        outputs = model(input_tensor, output_hidden_states=False)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    print(f"Logits shape: {logits.shape}")

    print(f"\n{'='*60}")
    print("RAW LOGITS AT KEY POSITIONS (HF Base Model)")
    print(f"{'='*60}")
    for pos in [7, 8, 23]:
        if pos < logits.shape[1]:
            pos_logits = logits[0, pos, :].float()
            target_tok = target_tokens[pos]
            target_logit = pos_logits[target_tok].item()
            max_logit = pos_logits.max().item()
            argmax_tok = pos_logits.argmax().item()

            # Get top 10 tokens
            top10_vals, top10_idx = pos_logits.topk(10)

            print(f"\nPosition {pos} (target={target_tok} '{tokenizer.decode([target_tok])}'):")
            print(f"  TARGET_LOGIT = {target_logit:.2f}")
            print(f"  MAX_LOGIT = {max_logit:.2f} at token {argmax_tok} ('{tokenizer.decode([argmax_tok])}')")
            print(f"  Top 10 tokens:")
            for i, (val, idx) in enumerate(zip(top10_vals.tolist(), top10_idx.tolist())):
                marker = " <-- TARGET" if idx == target_tok else ""
                print(f"    {i+1:2d}. token {idx:5d} ({repr(tokenizer.decode([idx])):15s}): logit={val:.2f}{marker}")

    print(f"\n{'='*60}")
    print("MEGATRON DIAGNOSTIC (from raw_logit_diag.log)")
    print(f"{'='*60}")
    print("After training (WRONG - target logit decreased):")
    print("  pos=7:  TARGET_LOGIT=3.61,  MAX=20.50 at token 6955 (Chinese '我想')")
    print("  pos=8:  TARGET_LOGIT=12.25, MAX=20.38 at token 276")
    print("  pos=23: TARGET_LOGIT=27.50, MAX=27.50 at token 27 (correct)")
    print()
    print("Baseline (before training, should match HF base):")
    print("  pos=7:  TARGET_LOGIT=8.81,  MAX=14.00 at token 40")

    # Also check token 6955
    tok_6955 = 6955
    print(f"\n{'='*60}")
    print(f"SUSPICIOUS TOKEN ANALYSIS: token {tok_6955}")
    print(f"{'='*60}")
    print(f"Token {tok_6955} decodes to: {repr(tokenizer.decode([tok_6955]))}")
    for pos in [7, 8, 23]:
        if pos < logits.shape[1]:
            pos_logits = logits[0, pos, :].float()
            logit_6955 = pos_logits[tok_6955].item()
            print(f"  At position {pos}: logit={logit_6955:.2f}")


if __name__ == "__main__":
    main()
