#!/usr/bin/env python3
"""Profile K2 memory usage to determine max context length.

This script calculates theoretical memory requirements and can run
actual profiling to find the exact limits.
"""

import argparse
import math

# K2-Thinking model specs
# Ground truth from config.json:
# - Quantization: compressed-tensors, INT4 (num_bits=4, group_size=32)
# - INT4 targets: Linear layers NOT in ignore list
# - BF16 (ignore list): lm_head, self_attn.*, shared_experts.*, mlp.gate/up/down_proj
# - Disk size: 554 GB for 62 shards (~0.53 bytes/param = INT4)
# - GPU size: 67 GiB × 16 = 1072 GiB (~1.03 bytes/param after decompression)
MODEL_SPECS = {
    "total_params": 1.04e12,  # 1.04T
    "active_params": 32e9,    # 32B active per token
    "num_experts": 384,
    "experts_per_token": 9,   # 8 routed + 1 shared
    "num_layers": 61,
    "hidden_size": 7168,
    "moe_intermediate_size": 2048,  # per expert
    "num_attention_heads": 64,
    "num_kv_heads": 8,
    "head_dim": 128,  # MLA uses kv_lora_rank for compression
    "kv_lora_rank": 512,  # MLA compressed KV
    "vocab_size": 163840,
    # Memory reality (measured):
    "disk_bytes_per_param": 0.53,  # INT4 packed on disk
    "gpu_bytes_per_param": 1.03,   # Decompressed for compute
}

def calculate_model_memory(tp: int = 16, quantization: str = "int4"):
    """Calculate model weight memory per GPU.

    Uses measured GPU memory (1.03 bytes/param after decompression),
    not theoretical INT4 (0.5 bytes/param).
    """
    total_params = MODEL_SPECS["total_params"]

    # Use measured GPU bytes per param (includes decompression overhead)
    bytes_per_param = MODEL_SPECS["gpu_bytes_per_param"]  # 1.03 measured

    # Total model size in GPU memory
    model_bytes = total_params * bytes_per_param

    # Per-GPU after tensor parallelism
    per_gpu_bytes = model_bytes / tp

    return per_gpu_bytes / (1024**3)  # GiB


def calculate_kv_cache_memory(
    max_model_len: int,
    batch_size: int = 1,
    tp: int = 16,
):
    """Calculate KV cache memory per GPU for MLA model.

    MLA uses compressed KV with kv_lora_rank instead of full head_dim.
    """
    num_layers = MODEL_SPECS["num_layers"]
    kv_lora_rank = MODEL_SPECS["kv_lora_rank"]

    # MLA KV cache: 2 (K+V) * layers * kv_lora_rank * 2 bytes (bf16)
    # Per token, per GPU (after TP sharding)
    bytes_per_token = 2 * num_layers * kv_lora_rank * 2 / tp

    total_bytes = bytes_per_token * max_model_len * batch_size

    return total_bytes / (1024**3)  # GiB


def calculate_lora_memory(
    lora_rank: int = 32,
    target_modules: int = 7,  # q,k,v,o,gate,up,down
    tp: int = 16,
):
    """Calculate LoRA adapter memory per GPU."""
    hidden_size = MODEL_SPECS["hidden_size"]
    num_layers = MODEL_SPECS["num_layers"]
    num_experts = MODEL_SPECS["num_experts"]

    # Attention LoRA (shared across experts)
    attn_lora_params = 4 * (hidden_size * lora_rank * 2)  # q,k,v,o

    # MLP LoRA (per expert for MoE)
    mlp_lora_params = 3 * (MODEL_SPECS["moe_intermediate_size"] * lora_rank * 2)  # gate,up,down
    mlp_lora_total = mlp_lora_params * num_experts

    # Total per layer
    per_layer = attn_lora_params + mlp_lora_total
    total_params = per_layer * num_layers

    # BF16 weights
    total_bytes = total_params * 2

    # Per GPU after TP
    per_gpu_bytes = total_bytes / tp

    return per_gpu_bytes / (1024**3)  # GiB


def calculate_activation_memory(
    seq_len: int,
    batch_size: int = 1,
    tp: int = 16,
):
    """Estimate activation memory during forward pass."""
    hidden_size = MODEL_SPECS["hidden_size"]
    num_layers = MODEL_SPECS["num_layers"]

    # Rough estimate: hidden states + attention scores + intermediate
    # Per layer: ~4 * hidden_size * seq_len * batch_size * 2 bytes
    per_layer = 4 * hidden_size * seq_len * batch_size * 2
    total = per_layer * num_layers / tp

    return total / (1024**3)  # GiB


def profile_config(
    max_model_len: int,
    lora_rank: int,
    tp: int = 16,
    gpu_memory_gib: float = 80.0,
    utilization: float = 0.90,
):
    """Profile a specific configuration and report memory breakdown."""
    available = gpu_memory_gib * utilization

    model = calculate_model_memory(tp=tp, quantization="int4")
    kv_cache = calculate_kv_cache_memory(max_model_len=max_model_len, tp=tp)
    lora = calculate_lora_memory(lora_rank=lora_rank, tp=tp)
    activation = calculate_activation_memory(seq_len=max_model_len, tp=tp)
    overhead = 5.0  # vLLM runtime, CUDA context, etc.

    total = model + kv_cache + lora + activation + overhead

    print(f"\n{'='*60}")
    print(f"K2-Thinking Memory Profile")
    print(f"  max_model_len={max_model_len}, lora_rank={lora_rank}, TP={tp}")
    print(f"{'='*60}")
    print(f"\nMemory Breakdown (per GPU):")
    print(f"  Model weights (INT4):     {model:6.2f} GiB")
    print(f"  KV cache (MLA):           {kv_cache:6.2f} GiB")
    print(f"  LoRA buffers:             {lora:6.2f} GiB")
    print(f"  Activations (estimate):   {activation:6.2f} GiB")
    print(f"  Overhead (vLLM/CUDA):     {overhead:6.2f} GiB")
    print(f"  {'-'*35}")
    print(f"  TOTAL:                    {total:6.2f} GiB")
    print(f"\nGPU Capacity: {gpu_memory_gib:.1f} GiB @ {utilization*100:.0f}% = {available:.1f} GiB available")

    if total <= available:
        headroom = available - total
        print(f"✓ FITS with {headroom:.1f} GiB headroom")
        return True, headroom
    else:
        deficit = total - available
        print(f"✗ EXCEEDS by {deficit:.1f} GiB")

        # Suggest fixes
        print(f"\nOptimization options:")
        if kv_cache > 5:
            reduced_len = int(max_model_len * (available - total + kv_cache) / kv_cache * 0.8)
            print(f"  1. Reduce max_model_len to ~{reduced_len} (saves {kv_cache * 0.5:.1f} GiB)")
        if lora > 1:
            print(f"  2. Reduce lora_rank to {lora_rank // 2} (saves {lora * 0.5:.1f} GiB)")
        print(f"  3. Increase TP from {tp} to {tp * 2} (halves per-GPU memory)")

        return False, -deficit


def find_max_context(lora_rank: int, tp: int = 16, target_headroom: float = 5.0):
    """Binary search to find maximum context length that fits."""
    low, high = 1024, 262144
    best = low

    while low <= high:
        mid = (low + high) // 2
        model = calculate_model_memory(tp=tp)
        kv = calculate_kv_cache_memory(max_model_len=mid, tp=tp)
        lora = calculate_lora_memory(lora_rank=lora_rank, tp=tp)
        activation = calculate_activation_memory(seq_len=mid, tp=tp)
        overhead = 5.0

        total = model + kv + lora + activation + overhead
        available = 80.0 * 0.90

        if total + target_headroom <= available:
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    return best


def profile_training(
    batch_tokens: int,
    lora_rank: int = 32,
    tp: int = 8,
    ep: int = 8,
    cp: int = 1,
    gpu_memory_gib: float = 80.0,
    utilization: float = 0.95,
):
    """Profile training memory per GPU (Megatron backend).

    Training uses TP=8, EP=8, CP=1 = 64 GPUs (or with MoE Parallel Folding).
    Each GPU holds: model shard + LoRA + gradients + optimizer + activations.

    With MoE Parallel Folding (EP > 1 and CP > 1):
    - world_size = TP * max(EP, CP)
    - Activations, dispatcher, spike memory sharded by CP
    """
    available = gpu_memory_gib * utilization

    # MoE Parallel Folding: when EP > 1 and CP > 1
    if ep > 1 and cp > 1:
        world_size = tp * max(ep, cp)
    else:
        world_size = tp * ep * cp

    # Model weights per GPU (distributed across TP and EP)
    # With EP=8: 384 experts / 8 = 48 experts per EP rank
    # With TP=8: each expert's weights split across 8 GPUs
    total_params = MODEL_SPECS["total_params"]
    # Training uses BF16 (2 bytes) not INT4
    model_bytes = total_params * 2 / world_size
    model_gib = model_bytes / (1024**3)

    # LoRA parameters (trainable)
    # Only attention LoRA for MoE (MLP experts not supported in vLLM inference)
    hidden = MODEL_SPECS["hidden_size"]
    num_layers = MODEL_SPECS["num_layers"]
    # q, k, v, o projections
    lora_params = 4 * (hidden * lora_rank * 2) * num_layers
    lora_bytes = lora_params * 2 / tp  # BF16, split by TP
    lora_gib = lora_bytes / (1024**3)

    # Gradients (same size as trainable params)
    grad_gib = lora_gib  # Only LoRA params have gradients

    # Optimizer states (Adam: 2x for momentum + variance)
    optim_gib = lora_gib * 2

    # Activations per token (rough estimate)
    # With MoE recompute enabled, only need to store attention activations
    # With CP > 1, each GPU only processes batch_tokens / cp
    hidden = MODEL_SPECS["hidden_size"]
    effective_tokens = batch_tokens // cp if cp > 1 else batch_tokens
    # Per layer: attention output + some intermediates
    bytes_per_token_per_layer = hidden * 4 * 2  # Conservative estimate, BF16
    activation_bytes = bytes_per_token_per_layer * num_layers * effective_tokens / tp
    activation_gib = activation_bytes / (1024**3)

    # Expert bias gradient all-reduce buffer (the OOM location!)
    # 384 experts × 61 layers × sizeof(float32) gradient
    num_experts = MODEL_SPECS["num_experts"]
    expert_bias_bytes = num_experts * num_layers * 4  # float32
    expert_bias_gib = expert_bias_bytes / (1024**3)

    # NCCL buffers for all-reduce
    nccl_buffer_gib = 2.0  # Rough estimate

    overhead = 3.0  # CUDA context, misc

    total = model_gib + lora_gib + grad_gib + optim_gib + activation_gib + expert_bias_gib + nccl_buffer_gib + overhead

    print(f"\n{'='*60}")
    print(f"K2-Thinking TRAINING Memory Profile")
    print(f"  batch_tokens={batch_tokens}, lora_rank={lora_rank}")
    print(f"  TP={tp}, EP={ep}, CP={cp}")
    if ep > 1 and cp > 1:
        print(f"  MoE Parallel Folding: world_size = TP * max(EP, CP) = {world_size}")
    else:
        print(f"  world_size = TP * EP * CP = {world_size} GPUs")
    print(f"{'='*60}")
    print(f"\nMemory Breakdown (per GPU):")
    print(f"  Model weights (BF16):     {model_gib:6.2f} GiB")
    print(f"  LoRA params:              {lora_gib:6.2f} GiB")
    print(f"  Gradients:                {grad_gib:6.2f} GiB")
    print(f"  Optimizer (Adam 2x):      {optim_gib:6.2f} GiB")
    print(f"  Activations:              {activation_gib:6.2f} GiB")
    print(f"  Expert bias buffer:       {expert_bias_gib:6.2f} GiB")
    print(f"  NCCL buffers:             {nccl_buffer_gib:6.2f} GiB")
    print(f"  Overhead (CUDA):          {overhead:6.2f} GiB")
    print(f"  {'-'*35}")
    print(f"  TOTAL:                    {total:6.2f} GiB")
    print(f"\nGPU Capacity: {gpu_memory_gib:.1f} GiB @ {utilization*100:.0f}% = {available:.1f} GiB available")

    if total <= available:
        headroom = available - total
        print(f"FITS with {headroom:.1f} GiB headroom")
        return True, headroom
    else:
        deficit = total - available
        print(f"EXCEEDS by {deficit:.1f} GiB")
        # Suggest max batch tokens
        activation_per_token = bytes_per_token_per_layer * num_layers / tp / (1024**3)
        max_tokens = int((available - (total - activation_gib)) / activation_per_token)
        print(f"\nMax batch_tokens for this config: ~{max_tokens}")
        return False, -deficit


def main():
    parser = argparse.ArgumentParser(description="Profile K2 memory usage")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--tp", type=int, default=16)
    parser.add_argument("--find-max", action="store_true", help="Find max context length")
    parser.add_argument("--training", action="store_true", help="Profile training memory")
    parser.add_argument("--batch-tokens", type=int, default=4096, help="Total tokens in training batch")
    parser.add_argument("--train-tp", type=int, default=8)
    parser.add_argument("--train-ep", type=int, default=8)
    parser.add_argument("--train-cp", type=int, default=1, help="Context parallelism (1 or 2)")
    args = parser.parse_args()

    if args.training:
        profile_training(
            batch_tokens=args.batch_tokens,
            lora_rank=args.lora_rank,
            tp=args.train_tp,
            ep=args.train_ep,
            cp=args.train_cp,
        )
    elif args.find_max:
        for rank in [8, 16, 32]:
            max_ctx = find_max_context(lora_rank=rank, tp=args.tp)
            print(f"lora_rank={rank}: max_model_len ≈ {max_ctx}")
    else:
        profile_config(
            max_model_len=args.max_model_len,
            lora_rank=args.lora_rank,
            tp=args.tp,
        )


if __name__ == "__main__":
    main()
