#!/usr/bin/env python3
"""Compare vLLM and Megatron top-k predictions at problematic positions."""

import json
import re
import torch
from transformers import AutoTokenizer

MODEL_NAME = "moonshotai/Moonlight-16B-A3B-Instruct"

# Sequence info
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

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)
input_tokens = tokens[:-1]  # 51 tokens
target_tokens = tokens[1:]   # 51 tokens (shifted)

print("Token analysis:")
print("=" * 70)
for i in range(min(30, len(input_tokens))):
    inp_tok = input_tokens[i]
    tgt_tok = target_tokens[i]
    inp_str = tokenizer.decode([inp_tok])
    tgt_str = tokenizer.decode([tgt_tok])
    print(f"Pos {i:2d}: Input: {inp_tok:6d} '{inp_str:15s}' -> Target: {tgt_tok:6d} '{tgt_str}'")

print("\n\nMegatron dump analysis:")
print("=" * 70)

# Load Megatron dump
meg_data = torch.load("/vePFS-Mindverse/share/code/logits_processor_input.pt", map_location="cpu")
meg_logits = meg_data["logits"]  # [1, 56, 20480]
meg_labels = meg_data["external_label"]  # [1, 56]

vocab_shard = 20480  # tp_rank=0 shard

problematic_positions = [5, 14, 21, 23, 29]

print("\nMegatron predictions at problematic positions:")
for pos in problematic_positions:
    logit_slice = meg_logits[0, pos, :]
    argmax_local = logit_slice.argmax().item()
    max_logit = logit_slice.max().item()

    target_tok = target_tokens[pos] if pos < len(target_tokens) else -1
    target_in_shard = target_tok < vocab_shard

    target_logit = meg_logits[0, pos, target_tok].item() if target_in_shard else None

    argmax_str = tokenizer.decode([argmax_local])
    target_str = tokenizer.decode([target_tok]) if target_tok >= 0 else "N/A"

    print(f"\nPosition {pos}:")
    print(f"  Target: {target_tok} '{target_str}'")
    print(f"  Megatron argmax: {argmax_local} '{argmax_str}' (logit={max_logit:.2f})")
    if target_in_shard:
        gap = max_logit - target_logit
        print(f"  Target logit: {target_logit:.2f} (gap={gap:.2f})")

# Top-5 for each problematic position
print("\n\nMegatron top-5 at each position:")
for pos in problematic_positions:
    logit_slice = meg_logits[0, pos, :]
    top5_values, top5_indices = torch.topk(logit_slice, 5)

    target_tok = target_tokens[pos]
    target_str = tokenizer.decode([target_tok])

    print(f"\nPosition {pos} (target={target_tok} '{target_str}'):")
    for i, (val, idx) in enumerate(zip(top5_values.tolist(), top5_indices.tolist())):
        tok_str = tokenizer.decode([idx])
        marker = " <-- TARGET" if idx == target_tok else ""
        print(f"  {i+1}. token={idx:6d} '{tok_str:15s}' logit={val:8.2f}{marker}")

# Load Megatron top-k JSON (from previous analysis)
print("\n\nvLLM vs Megatron comparison at positions 12 (should match) vs 14 (diverges):")
print("Position 12 = '10' (correct), Position 14 = '1' (Megatron predicts 9)")

# vLLM at position 12: {'795': -0.12042810022830...}
# vLLM at position 14: {'220': -0.00092332641361...}

print("\nvLLM position 12: token 795 ('10') has logprob -0.12 (prob=88.7%)")
print("Megatron position 12: argmax=795 ('10') - MATCHES")

print("\nvLLM position 14: token 220 (' ') has logprob -0.0009 (prob=99.9%)")
print("Megatron position 14: argmax=24 ('9') - DIVERGES")
print("  Target at position 14 is token 16 ('1')")

# More detailed analysis
print("\n\n" + "=" * 70)
print("Key insight: Check if vLLM and Megatron are looking at the same positions")
print("=" * 70)

print("\nSequence structure:")
print("Position 14 in sequence: input[14] -> output[14] predicts input[15]")
print(f"input[14] = {input_tokens[14]} ('{tokenizer.decode([input_tokens[14]])}')")
print(f"target[14] = {target_tokens[14]} ('{tokenizer.decode([target_tokens[14]])}')")

# Check all tokens around position 14
print("\nContext around position 14:")
for i in range(10, 20):
    inp = input_tokens[i]
    tgt = target_tokens[i]
    print(f"  Pos {i}: input={inp:6d} '{tokenizer.decode([inp]):10s}' -> target={tgt:6d} '{tokenizer.decode([tgt])}'")
