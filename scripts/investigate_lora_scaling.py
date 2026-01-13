#!/usr/bin/env python3
"""Investigate why Megatron shows amplified LoRA effect vs vLLM.

Key finding: Token 795 ("10") shows wildly different logits:
- Megatron: 53-58 at many positions (wrong positions!)
- vLLM: 11-26 at same positions

Hypothesis: LoRA weights are being scaled differently or applied at different points.
"""

import subprocess
import torch
import numpy as np
from pathlib import Path

CHECKPOINT = "/vePFS-Mindverse/share/code/tinker-server/checkpoints/80e6bf97-56d9-4f3a-9872-7cc4b80a7379_0/debug_checkpoint_20260110_182006"

# Copy PEFT checkpoint to local (files are in the checkpoint root, not peft_format subdir)
print("Copying PEFT checkpoint...")
subprocess.run(["mkdir", "-p", "/tmp/peft_checkpoint"], check=True)
subprocess.run(["scp", f"volcano:{CHECKPOINT}/adapter_config.json", "/tmp/peft_checkpoint/"], check=True)
subprocess.run(["scp", f"volcano:{CHECKPOINT}/adapter_model.safetensors", "/tmp/peft_checkpoint/"], check=True)

# Load PEFT adapter weights
peft_path = Path("/tmp/peft_checkpoint")
adapter_files = list(peft_path.glob("*.safetensors"))
print(f"Found adapter files: {adapter_files}")

if adapter_files:
    from safetensors import safe_open
    with safe_open(adapter_files[0], framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"\nPEFT adapter keys ({len(keys)}):")
        for k in sorted(keys)[:30]:
            tensor = f.get_tensor(k)
            print(f"  {k}: shape={tensor.shape}, dtype={tensor.dtype}, norm={tensor.float().norm().item():.6f}")
        if len(keys) > 30:
            print(f"  ... and {len(keys) - 30} more")

        # Check output layer LoRA specifically
        output_keys = [k for k in keys if "lm_head" in k.lower() or "embed_out" in k.lower() or "output" in k.lower()]
        if output_keys:
            print(f"\nOutput layer LoRA keys: {output_keys}")
            for k in output_keys:
                tensor = f.get_tensor(k)
                print(f"  {k}: shape={tensor.shape}, norm={tensor.float().norm().item():.6f}")
                if tensor.numel() < 100:
                    print(f"    values: {tensor.flatten().tolist()}")

# Also check the Megatron format weights (mp_rank_* files in same directory)
print("\n" + "="*80)
print("Checking Megatron format checkpoint")
print("="*80)

subprocess.run(["mkdir", "-p", "/tmp/megatron_checkpoint"], check=True)
# Copy just one rank file for analysis
subprocess.run(["scp", f"volcano:{CHECKPOINT}/mp_rank_00_000_adapter.pt", "/tmp/megatron_checkpoint/"], check=True)

meg_path = Path("/tmp/megatron_checkpoint")
meg_files = list(meg_path.glob("*.pt"))
print(f"Found Megatron files: {meg_files}")

for mf in meg_files[:3]:
    print(f"\n--- {mf.name} ---")
    data = torch.load(mf, map_location="cpu")
    if isinstance(data, dict):
        for k, v in list(data.items())[:20]:
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}, norm={v.float().norm().item():.6f}")
            else:
                print(f"  {k}: {type(v)}")
    elif isinstance(data, torch.Tensor):
        print(f"  tensor: shape={data.shape}, dtype={data.dtype}")

# Now let's check the actual token 795 embedding difference
print("\n" + "="*80)
print("Investigating token 795 ('10') specifically")
print("="*80)

# Load the model's output embedding from base model
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import gc

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("moonshotai/Moonlight-16B-A3B-Instruct", trust_remote_code=True)

# We can't load the full model, but we can check the embedding dimensions
config = AutoConfig.from_pretrained("moonshotai/Moonlight-16B-A3B-Instruct", trust_remote_code=True)
print(f"Model config:")
print(f"  vocab_size: {config.vocab_size}")
print(f"  hidden_size: {config.hidden_size}")

# Check if LoRA modifies the output layer (lm_head)
print("\n" + "="*80)
print("Checking LoRA rank and scaling")
print("="*80)

# Read adapter_config.json
config_file = peft_path / "adapter_config.json"
if config_file.exists():
    import json
    with open(config_file) as f:
        adapter_config = json.load(f)
    print(f"Adapter config: {json.dumps(adapter_config, indent=2)}")

# The key question: How is the LoRA applied in Megatron vs vLLM?
# Let's trace the actual hidden state values at token 795

# Check if there's any weight difference at token 795 index
print("\n" + "="*80)
print("Checking weight differences for token 795")
print("="*80)

# Token 795 is in shard 0, offset 795
# In vLLM with full vocab, it's at index 795
# In Megatron with TP=8, rank 0 has vocab [0:20480], so token 795 is at local index 795

# The issue might be in how the hidden states are transformed to logits
# Let me check if there are any lm_head LoRA weights
if adapter_files:
    with safe_open(adapter_files[0], framework="pt", device="cpu") as f:
        # Look for lm_head or output projection weights
        lm_head_keys = [k for k in f.keys() if any(x in k.lower() for x in ["lm_head", "embed_out", "output", "wte", "wpe"])]
        print(f"lm_head related keys: {lm_head_keys}")

        # Check each LoRA A/B pair
        lora_a_keys = sorted([k for k in f.keys() if "lora_A" in k])
        lora_b_keys = sorted([k for k in f.keys() if "lora_B" in k])

        print(f"\nLoRA A matrices ({len(lora_a_keys)}):")
        for k in lora_a_keys[:10]:
            t = f.get_tensor(k)
            print(f"  {k}: {t.shape}")

        print(f"\nLoRA B matrices ({len(lora_b_keys)}):")
        for k in lora_b_keys[:10]:
            t = f.get_tensor(k)
            print(f"  {k}: {t.shape}")

# Compare actual raw logit dumps more carefully
print("\n" + "="*80)
print("Detailed raw logits analysis for token 795")
print("="*80)

# Load dumps
meg_data = torch.load("/tmp/megatron_logits.pt", map_location="cpu")
vllm_data = torch.load("/tmp/vllm_logits.pt", map_location="cpu")

meg_logits = meg_data["logits"].squeeze(0).float()  # [56, 20480]
vllm_logits = vllm_data["raw_logits"].float()  # [50, 163840]

# Token 795 logits across all positions
tok_795_meg = meg_logits[:, 795]  # [56]
tok_795_vllm = vllm_logits[:, 795]  # [50]

print(f"Token 795 ('10') logits across positions:")
print(f"{'Pos':<4} {'Megatron':<12} {'vLLM':<12} {'Diff':<12}")
print("-" * 44)
for pos in range(min(50, len(tok_795_meg))):
    meg_val = tok_795_meg[pos].item()
    vllm_val = tok_795_vllm[pos].item() if pos < len(tok_795_vllm) else float('nan')
    diff = meg_val - vllm_val if not np.isnan(vllm_val) else float('nan')
    marker = " ***" if abs(diff) > 20 else ""
    print(f"{pos:<4} {meg_val:<12.4f} {vllm_val:<12.4f} {diff:<+12.4f}{marker}")

# Check correlation and scale difference
common_len = min(len(tok_795_meg), len(tok_795_vllm))
meg_795 = tok_795_meg[:common_len]
vllm_795 = tok_795_vllm[:common_len]

corr = np.corrcoef(meg_795.numpy(), vllm_795.numpy())[0, 1]
scale = (meg_795.mean() / vllm_795.mean()).item()

print(f"\nToken 795 statistics:")
print(f"  Correlation: {corr:.4f}")
print(f"  Mean scale (Megatron/vLLM): {scale:.4f}")
print(f"  Megatron mean: {meg_795.mean().item():.4f}")
print(f"  vLLM mean: {vllm_795.mean().item():.4f}")

# Check if the pattern is consistent across other tokens
print("\n" + "="*80)
print("Scale comparison across multiple tokens")
print("="*80)

sample_tokens = [16, 24, 91, 198, 220, 348, 658, 795, 1101, 2291, 2343, 2470, 2482, 3922, 9485]
print(f"{'Token':<8} {'Meg mean':<12} {'vLLM mean':<12} {'Scale':<12} {'Corr':<12}")
print("-" * 56)

for tok in sample_tokens:
    meg_tok = meg_logits[:common_len, tok]
    vllm_tok = vllm_logits[:common_len, tok]
    scale = (meg_tok.mean() / vllm_tok.mean()).item() if vllm_tok.mean() != 0 else float('inf')
    corr = np.corrcoef(meg_tok.numpy(), vllm_tok.numpy())[0, 1]
    tok_str = repr(tokenizer.decode([tok]))[:6]
    print(f"{tok:<5} {tok_str:<6} {meg_tok.mean().item():<12.4f} {vllm_tok.mean().item():<12.4f} {scale:<12.4f} {corr:<12.4f}")
