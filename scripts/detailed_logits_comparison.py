#!/usr/bin/env python3
"""Detailed position-by-position logits comparison between Megatron and vLLM.

Focus on identifying:
1. Which positions have largest differences
2. What are the top-k tokens at each position
3. Are differences correlated with specific token types (numbers, special tokens, etc.)
"""

import subprocess
import torch
import numpy as np
from transformers import AutoTokenizer

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("moonshotai/Moonlight-16B-A3B-Instruct", trust_remote_code=True)

# Load dumps
meg_data = torch.load("/tmp/megatron_logits.pt", map_location="cpu")
vllm_data = torch.load("/tmp/vllm_logits.pt", map_location="cpu")

meg_logits = meg_data["logits"].squeeze(0).float()  # [56, 20480]
vllm_logits = vllm_data["raw_logits"].float()  # [50, 163840]

# Get target tokens
target_tokens = vllm_data.get("tgt_token_ids", None)

# Constants
shard_vocab = 20480
full_vocab = 163840

print(f"Megatron logits: {meg_logits.shape}")
print(f"vLLM logits: {vllm_logits.shape}")

# The test sequence
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

tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)
input_tokens = tokens[:-1]
target_tokens_list = tokens[1:]

print(f"\nToken sequence length: {len(tokens)}")

# Compare position by position
print("\n" + "="*120)
print("POSITION-BY-POSITION COMPARISON")
print("="*120)

# Only compare positions that exist in both (and vocab range 0-20479)
common_pos = min(meg_logits.shape[0], vllm_logits.shape[0])
vllm_shard0 = vllm_logits[:common_pos, :shard_vocab]

print(f"\n{'Pos':<4} {'Target':<12} {'Meg argmax':<12} {'vLLM argmax':<12} {'Target logit diff':<16} {'Argmax match?':<12}")
print("-" * 120)

position_diffs = []
for pos in range(common_pos):
    if pos >= len(target_tokens_list):
        break

    target_tok = target_tokens_list[pos]
    target_str = repr(tokenizer.decode([target_tok]))[:10]

    meg_row = meg_logits[pos]
    vllm_row = vllm_shard0[pos]

    meg_argmax = meg_row.argmax().item()
    vllm_argmax = vllm_row.argmax().item()

    # Target token logit diff (if target is in shard 0)
    if target_tok < shard_vocab:
        target_diff = meg_row[target_tok].item() - vllm_row[target_tok].item()
    else:
        target_diff = float('nan')

    # Max diff at this position
    diff_row = meg_row - vllm_row
    max_diff = diff_row.abs().max().item()

    argmax_match = "Yes" if meg_argmax == vllm_argmax else f"No ({meg_argmax} vs {vllm_argmax})"

    meg_argmax_str = repr(tokenizer.decode([meg_argmax]))[:8]
    vllm_argmax_str = repr(tokenizer.decode([vllm_argmax]))[:8]

    position_diffs.append((pos, max_diff, target_diff))

    # Only print positions with significant differences
    if abs(target_diff) > 5 or max_diff > 20 or meg_argmax != vllm_argmax:
        print(f"{pos:<4} {target_str:<12} {meg_argmax_str:<12} {vllm_argmax_str:<12} {target_diff:< 16.4f} {argmax_match:<12}")

# Sort by max diff to find most problematic positions
print("\n" + "="*120)
print("TOP 10 POSITIONS BY MAXIMUM DIFFERENCE")
print("="*120)

position_diffs.sort(key=lambda x: x[1], reverse=True)
for pos, max_diff, target_diff in position_diffs[:10]:
    target_tok = target_tokens_list[pos]
    target_str = repr(tokenizer.decode([target_tok]))[:10]

    meg_row = meg_logits[pos]
    vllm_row = vllm_shard0[pos]

    meg_argmax = meg_row.argmax().item()
    vllm_argmax = vllm_row.argmax().item()

    print(f"\nPosition {pos} (target={target_str}, max_diff={max_diff:.2f}):")

    # Top-5 for Megatron
    print(f"  Megatron top-5:")
    meg_topk = torch.topk(meg_row, 5)
    for rank, (logit, idx) in enumerate(zip(meg_topk.values.tolist(), meg_topk.indices.tolist())):
        tok_str = repr(tokenizer.decode([idx]))[:12]
        marker = " <-- TARGET" if idx == target_tok else ""
        print(f"    {rank+1}. {idx:6d} {tok_str:<14}: {logit:.4f}{marker}")

    # Top-5 for vLLM
    print(f"  vLLM top-5:")
    vllm_topk = torch.topk(vllm_row, 5)
    for rank, (logit, idx) in enumerate(zip(vllm_topk.values.tolist(), vllm_topk.indices.tolist())):
        tok_str = repr(tokenizer.decode([idx]))[:12]
        marker = " <-- TARGET" if idx == target_tok else ""
        print(f"    {rank+1}. {idx:6d} {tok_str:<14}: {logit:.4f}{marker}")

# Check for specific token patterns
print("\n" + "="*120)
print("TOKEN-SPECIFIC ANALYSIS")
print("="*120)

# Check if certain tokens consistently have higher/lower logits in one system
interesting_tokens = [795, 24, 16, 2482, 220, 198, 348, 91]  # "10", "9", "1", "user", space, newline, "<", "|"

print(f"\n{'Token':<8} {'Name':<15} {'Meg mean':<12} {'vLLM mean':<12} {'Ratio':<10} {'Corr':<10}")
print("-" * 80)

for tok in interesting_tokens:
    if tok >= shard_vocab:
        continue

    meg_tok = meg_logits[:common_pos, tok]
    vllm_tok = vllm_shard0[:common_pos, tok]

    meg_mean = meg_tok.mean().item()
    vllm_mean = vllm_tok.mean().item()
    ratio = meg_mean / vllm_mean if abs(vllm_mean) > 0.001 else float('inf')

    # Correlation
    corr = np.corrcoef(meg_tok.numpy(), vllm_tok.numpy())[0, 1]

    tok_name = repr(tokenizer.decode([tok]))[:14]
    print(f"{tok:<8} {tok_name:<15} {meg_mean:<12.4f} {vllm_mean:<12.4f} {ratio:<10.4f} {corr:<10.4f}")

# Check position-token correlation
print("\n" + "="*120)
print("POSITIONS WHERE TOKEN '10' (795) IS ANOMALOUSLY HIGH IN MEGATRON")
print("="*120)

tok_795_meg = meg_logits[:common_pos, 795]
tok_795_vllm = vllm_shard0[:common_pos, 795]

for pos in range(common_pos):
    meg_val = tok_795_meg[pos].item()
    vllm_val = tok_795_vllm[pos].item()
    diff = meg_val - vllm_val

    if diff > 20:  # More than 20 logit difference
        target = target_tokens_list[pos] if pos < len(target_tokens_list) else -1
        target_str = repr(tokenizer.decode([target]))[:10] if target >= 0 else "N/A"
        print(f"Position {pos}: Megatron={meg_val:.2f}, vLLM={vllm_val:.2f}, diff={diff:.2f}, target={target_str}")

        # Check if this is where "10" should actually appear
        is_10_target = target == 795
        print(f"  Is '10' the target token? {is_10_target}")
