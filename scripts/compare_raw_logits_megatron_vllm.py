#!/usr/bin/env python3
"""Compare raw logits between Megatron and vLLM dumps.

Megatron dump: /vePFS-Mindverse/share/code/logits_processor_input.pt
- Shape [1, 56, 20480] - sharded vocab (1/8 of 163840 due to TP=8)

vLLM dump: /vePFS-Mindverse/share/code/vllm_prompt_raw_logits.pt
- Shape [50, 163840] - full vocab

Analysis:
1. Determine which vocab shard (TP rank) the Megatron dump contains
2. Compare logits at problematic positions
"""

import subprocess
import torch
import numpy as np

# Copy files locally
print("Copying dump files from volcano...")
subprocess.run(["scp", "volcano:/vePFS-Mindverse/share/code/logits_processor_input.pt", "/tmp/megatron_logits.pt"], check=True)
subprocess.run(["scp", "volcano:/vePFS-Mindverse/share/code/vllm_prompt_raw_logits.pt", "/tmp/vllm_logits.pt"], check=True)

# Load dumps
meg_data = torch.load("/tmp/megatron_logits.pt", map_location="cpu")
vllm_data = torch.load("/tmp/vllm_logits.pt", map_location="cpu")

meg_logits = meg_data["logits"]  # [1, 56, 20480]
vllm_logits = vllm_data["raw_logits"]  # [50, 163840]

print(f"Megatron logits: shape={meg_logits.shape}, dtype={meg_logits.dtype}")
print(f"vLLM logits: shape={vllm_logits.shape}, dtype={vllm_logits.dtype}")

# Vocab info
full_vocab = 163840
shard_vocab = 20480
n_shards = full_vocab // shard_vocab  # 8 shards

print(f"\nVocab: {full_vocab} full, {shard_vocab} per shard, {n_shards} shards")

# Check Megatron data for additional info
print(f"\nMegatron dump keys: {meg_data.keys()}")
for k, v in meg_data.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    elif isinstance(v, (list, tuple)) and len(v) < 100:
        print(f"  {k}: {v}")

print(f"\nvLLM dump keys: {vllm_data.keys()}")
for k, v in vllm_data.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    elif isinstance(v, (list, tuple)) and len(v) < 100:
        print(f"  {k}: {v}")

# Target tokens for comparison (from test sequence)
# The problematic positions from previous analysis: 5, 14, 21, 23
# Target tokens at these positions need to be within shard range for comparison

# First, let's understand the position mapping
# Megatron: [1, 56, 20480] - 56 positions
# vLLM: [50, 163840] - 50 positions
# vLLM prompt_logprobs returns len(input)-1 values (predicting next token)
# So vLLM[i] predicts input_tokens[i+1]

# Let's check which tokens are in which shard
target_tokens = vllm_data.get("tgt_token_ids", None)
if target_tokens is not None:
    print(f"\nTarget tokens from vLLM dump: {target_tokens.tolist()}")
    for i, tok in enumerate(target_tokens.tolist()):
        shard = tok // shard_vocab
        offset = tok % shard_vocab
        print(f"  pos {i}: token {tok} -> shard {shard}, offset {offset}")

# Check which shard Megatron dump is from by analyzing argmax patterns
print("\n" + "="*80)
print("SHARD ANALYSIS: Determining which TP rank the Megatron dump is from")
print("="*80)

# For each position, find argmax in Megatron
meg_logits_2d = meg_logits.squeeze(0)  # [56, 20480]

# For positions where we know the target token, check if it's in range
# If target token 2482 (user) is at position 5, and Megatron shows high logit at offset 2482,
# then Megatron is shard 0 (tokens 0-20479)

# Let's examine specific positions
print("\nPosition-by-position analysis:")
print(f"{'Pos':<4} {'Meg argmax':<12} {'Meg max logit':<14} {'vLLM argmax':<12} {'vLLM max logit':<14}")
print("-" * 70)

# Align positions:
# Megatron has 56 positions, vLLM has 50 positions
# Assuming same input, check if positions align

# Convert to float32 for comparison
meg_f32 = meg_logits_2d.float()
vllm_f32 = vllm_logits.float()

for pos in range(min(20, vllm_f32.shape[0], meg_f32.shape[0])):
    meg_row = meg_f32[pos]
    vllm_row = vllm_f32[pos]

    meg_argmax = meg_row.argmax().item()
    meg_max = meg_row.max().item()

    vllm_argmax = vllm_row.argmax().item()
    vllm_max = vllm_row.max().item()

    # Check which shard vLLM argmax would be in
    vllm_shard = vllm_argmax // shard_vocab
    vllm_offset = vllm_argmax % shard_vocab

    print(f"{pos:<4} {meg_argmax:<12} {meg_max:<14.4f} {vllm_argmax:<12} ({vllm_shard}:{vllm_offset}) {vllm_max:<14.4f}")

# Key analysis: Compare raw logits at specific vocab indices
print("\n" + "="*80)
print("RAW LOGITS COMPARISON AT SPECIFIC TOKENS")
print("="*80)

# Known important tokens (must be in shard 0, i.e., < 20480)
important_tokens = {
    "user": 2482,      # pos 5 target
    "1": 16,           # pos 14, 15, 50
    "10": 795,         # pos 12, 32
    "9": 24,           # pos 34
    "assistant": 78191 // 1,  # Need to check actual token ID
    "space": 220,
    "newline": 198,
}

# Get actual token IDs from tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("moonshotai/Moonlight-16B-A3B-Instruct", trust_remote_code=True)

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

print(f"\nFull token sequence ({len(tokens)} tokens):")
for i, tok in enumerate(tokens[:60]):
    tok_str = repr(tokenizer.decode([tok]))
    shard = tok // shard_vocab
    offset = tok % shard_vocab
    marker = ""
    if i > 0:
        # This position predicts token at position i
        # vLLM[i-1] and Megatron[i-1] predict this token
        pass
    print(f"  pos={i:2d}: token={tok:6d} shard={shard} offset={offset:5d} {tok_str:15s}")

# Compare logits for tokens in shard 0 (offset 0-20479)
print("\n" + "="*80)
print("LOGIT VALUES FOR IN-RANGE TOKENS (shard 0)")
print("="*80)

# Problematic positions from earlier: 5, 14, 21, 23
# At these positions, Megatron predicts wrong tokens
problematic_positions = [5, 14, 21, 23]

for pos in problematic_positions:
    if pos >= len(target_tokens_list):
        continue

    target_tok = target_tokens_list[pos]
    target_shard = target_tok // shard_vocab
    target_offset = target_tok % shard_vocab
    target_str = repr(tokenizer.decode([target_tok]))

    print(f"\nPosition {pos}: target={target_tok} ({target_str}), shard={target_shard}")

    # Get Megatron logit for this position
    if pos < meg_f32.shape[0]:
        meg_row = meg_f32[pos]
        meg_argmax = meg_row.argmax().item()
        meg_max = meg_row.max().item()

        if target_shard == 0 and target_offset < shard_vocab:
            # Target is in this shard
            meg_target_logit = meg_row[target_offset].item()
            print(f"  Megatron: target_logit={meg_target_logit:.4f}, argmax={meg_argmax} ({meg_row[meg_argmax].item():.4f})")
            print(f"  Megatron top-5:")
            topk = torch.topk(meg_row, 5)
            for rank, (logit, idx) in enumerate(zip(topk.values.tolist(), topk.indices.tolist())):
                # This idx is an offset within the shard
                # For shard 0, actual token = idx
                tok_str = repr(tokenizer.decode([idx]))
                marker = " <-- TARGET" if idx == target_tok else ""
                print(f"    {rank+1}. token={idx:6d} ({tok_str:15s}): logit={logit:.4f}{marker}")
        else:
            print(f"  Megatron: Target token {target_tok} is in shard {target_shard}, not in this dump (shard 0?)")
            print(f"  Megatron top-5 (shard 0 only):")
            topk = torch.topk(meg_row, 5)
            for rank, (logit, idx) in enumerate(zip(topk.values.tolist(), topk.indices.tolist())):
                tok_str = repr(tokenizer.decode([idx]))
                print(f"    {rank+1}. token={idx:6d} ({tok_str:15s}): logit={logit:.4f}")

    # Get vLLM logit for this position
    if pos < vllm_f32.shape[0]:
        vllm_row = vllm_f32[pos]
        vllm_argmax = vllm_row.argmax().item()
        vllm_max = vllm_row.max().item()
        vllm_target_logit = vllm_row[target_tok].item()

        print(f"  vLLM: target_logit={vllm_target_logit:.4f}, argmax={vllm_argmax} ({vllm_row[vllm_argmax].item():.4f})")
        print(f"  vLLM top-5:")
        topk = torch.topk(vllm_row, 5)
        for rank, (logit, idx) in enumerate(zip(topk.values.tolist(), topk.indices.tolist())):
            tok_str = repr(tokenizer.decode([idx]))
            marker = " <-- TARGET" if idx == target_tok else ""
            print(f"    {rank+1}. token={idx:6d} ({tok_str:15s}): logit={logit:.4f}{marker}")

# Direct logit comparison for shard 0 slice
print("\n" + "="*80)
print("DIRECT COMPARISON: Megatron vs vLLM[:, :20480] (shard 0 slice)")
print("="*80)

# Compare first 20480 tokens of vLLM with Megatron (assuming Megatron is shard 0)
vllm_shard0 = vllm_f32[:, :shard_vocab]  # [50, 20480]

print(f"vLLM shard 0 slice: {vllm_shard0.shape}")
print(f"Megatron: {meg_f32.shape}")

# For positions that exist in both
common_pos = min(vllm_shard0.shape[0], meg_f32.shape[0])
print(f"Common positions: {common_pos}")

# Statistics
diff = meg_f32[:common_pos] - vllm_shard0[:common_pos]
print(f"\nLogit difference statistics (Megatron - vLLM shard0):")
print(f"  Mean: {diff.mean().item():.4f}")
print(f"  Std: {diff.std().item():.4f}")
print(f"  Max: {diff.max().item():.4f}")
print(f"  Min: {diff.min().item():.4f}")
print(f"  Abs mean: {diff.abs().mean().item():.4f}")

# Per-position statistics
print(f"\nPer-position max |diff|:")
for pos in range(min(20, common_pos)):
    row_diff = diff[pos]
    max_diff = row_diff.abs().max().item()
    max_diff_idx = row_diff.abs().argmax().item()
    print(f"  pos {pos:2d}: max |diff|={max_diff:.4f} at token {max_diff_idx}")

# Check if there's a constant offset (which would indicate different TP ranks)
print("\n" + "="*80)
print("CHECKING FOR TP RANK OFFSET")
print("="*80)

# If Megatron is from rank R, its token indices are [R*20480, (R+1)*20480)
# Let's check correlation with each possible vLLM shard

best_corr = -1
best_rank = 0

for rank in range(8):
    start = rank * shard_vocab
    end = start + shard_vocab
    vllm_slice = vllm_f32[:common_pos, start:end]

    # Flatten and compute correlation
    meg_flat = meg_f32[:common_pos].flatten()
    vllm_flat = vllm_slice.flatten()

    # Normalize
    meg_norm = (meg_flat - meg_flat.mean()) / meg_flat.std()
    vllm_norm = (vllm_flat - vllm_flat.mean()) / vllm_flat.std()

    corr = (meg_norm * vllm_norm).mean().item()

    print(f"Rank {rank} (tokens {start}-{end-1}): correlation = {corr:.6f}")

    if corr > best_corr:
        best_corr = corr
        best_rank = rank

print(f"\nBest match: TP rank {best_rank} with correlation {best_corr:.6f}")

# If best rank is 0, analyze the differences more closely
if best_rank == 0:
    print("\n" + "="*80)
    print("DETAILED DIFFERENCE ANALYSIS (Megatron is TP rank 0)")
    print("="*80)

    vllm_rank0 = vllm_f32[:common_pos, :shard_vocab]
    diff = meg_f32[:common_pos] - vllm_rank0

    # Find positions with largest differences
    pos_max_diff = diff.abs().max(dim=1).values  # Max diff per position
    worst_positions = pos_max_diff.topk(10).indices.tolist()

    print(f"Positions with largest logit differences:")
    for pos in worst_positions:
        max_diff = diff[pos].abs().max().item()
        max_diff_idx = diff[pos].abs().argmax().item()
        meg_val = meg_f32[pos, max_diff_idx].item()
        vllm_val = vllm_rank0[pos, max_diff_idx].item()
        tok_str = repr(tokenizer.decode([max_diff_idx]))
        print(f"  pos {pos:2d}: max |diff|={max_diff:.4f} at token {max_diff_idx:6d} ({tok_str})")
        print(f"          Megatron={meg_val:.4f}, vLLM={vllm_val:.4f}")
