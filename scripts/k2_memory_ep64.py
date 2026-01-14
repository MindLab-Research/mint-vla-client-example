#!/usr/bin/env python3
"""
K2 Memory Projections for TP=8, EP=64, ETP=1 Configuration

Goal: Predict memory usage for the new parallelism configuration where:
- TP=8: Tensor parallelism for attention (shards across 8 GPUs within each TP group)
- EP=64: Expert parallelism (384 experts / 64 = 6 experts per GPU)
- ETP=1: Expert tensor parallelism = 1 (each expert NOT split)
- CP=1: No context parallelism

With MoE Parallel Folding using ETP:
- world_size = EP = 64 GPUs
- TP groups are subgroups of 8 within the 64 GPUs (8 TP groups total)
- Each GPU has 6 experts (same as current TP=8, EP=8 config!)
- Attention weights sharded by TP=8

Key insight: Memory should be SIMILAR to TP=8, EP=8 because:
- experts_per_gpu = 384 / 64 = 6 (same)
- attention sharding = TP = 8 (same)

The difference is in communication patterns, not memory.
"""

from dataclasses import dataclass


@dataclass
class K2Config:
    """K2-Thinking architecture parameters."""
    hidden_size: int = 7168
    num_layers: int = 61
    num_attention_heads: int = 64

    # MLA parameters
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    # MoE parameters
    n_routed_experts: int = 384
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 2048
    n_shared_experts: int = 1
    shared_expert_intermediate_size: int = 2048

    # Dense FFN (layer 0)
    intermediate_size: int = 18432
    num_dense_layers: int = 1

    vocab_size: int = 163840

    # Training uses BF16
    bytes_per_param: int = 2


@dataclass
class TrainingConfig:
    """Training parallelism settings."""
    tp: int = 8      # Tensor parallel for attention
    ep: int = 64     # Expert parallel
    etp: int = 1     # Expert tensor parallel (1 = experts not split)
    pp: int = 1      # Pipeline parallel
    cp: int = 1      # Context parallel
    lora_rank: int = 16
    moe_recompute: bool = True

    @property
    def world_size(self) -> int:
        """World size with MoE Parallel Folding.

        MoE Parallel Folding is used when EP > TP and ETP < TP:
        - EP determines expert distribution (experts don't span multiple EP ranks)
        - TP is a subgroup within EP ranks for attention
        - world_size = EP

        Without folding (EP <= TP or ETP = TP):
        - Traditional orthogonal: world_size = TP * EP
        """
        if self.ep > self.tp and self.etp < self.tp:
            # MoE Parallel Folding: experts distributed across EP GPUs
            # TP is a subgroup for attention operations
            return self.ep * self.pp * self.cp
        elif self.ep > 1 and self.cp > 1:
            # CP/EP folding
            return self.tp * self.pp * max(self.ep, self.cp)
        else:
            # Traditional: TP and EP are orthogonal dimensions
            return self.tp * self.ep * self.pp * self.cp

    @property
    def experts_per_gpu(self) -> int:
        """Number of experts per GPU."""
        return 384 // self.world_size


def bytes_to_gib(b: int) -> float:
    return b / (1024 ** 3)


def calculate_memory(cfg: K2Config, train: TrainingConfig, seq_len: int) -> dict:
    """Calculate memory breakdown for given sequence length."""
    results = {}
    bf16 = cfg.bytes_per_param

    # === Model Parameters ===

    # Embedding (sharded by TP)
    embedding_bytes = (cfg.vocab_size * cfg.hidden_size // train.tp) * bf16

    # MLA Attention per layer (sharded by TP)
    qk_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim  # 192
    w_dq = cfg.hidden_size * cfg.q_lora_rank
    w_uq = cfg.q_lora_rank * cfg.num_attention_heads * qk_head_dim
    w_dkv = cfg.hidden_size * cfg.kv_lora_rank
    w_uk = cfg.kv_lora_rank * cfg.num_attention_heads * qk_head_dim
    w_uv = cfg.kv_lora_rank * cfg.num_attention_heads * cfg.v_head_dim
    w_o = cfg.num_attention_heads * cfg.v_head_dim * cfg.hidden_size
    attn_params_per_layer = w_dq + w_uq + w_dkv + w_uk + w_uv + w_o
    attn_bytes_per_layer = (attn_params_per_layer // train.tp) * bf16

    # Expert FFN per GPU
    # With EP=64: 384/64 = 6 experts per GPU
    # Each expert: 3 * hidden * moe_intermediate (SwiGLU)
    expert_params = 3 * cfg.hidden_size * cfg.moe_intermediate_size
    expert_bytes_per_gpu = train.experts_per_gpu * expert_params * bf16

    # Shared expert (sharded by TP)
    shared_expert_params = 3 * cfg.hidden_size * cfg.shared_expert_intermediate_size
    shared_expert_bytes = (shared_expert_params // train.tp) * bf16

    # Router (small, replicated)
    router_bytes = cfg.hidden_size * cfg.n_routed_experts * bf16

    # LayerNorm (small, replicated)
    layernorm_bytes = 2 * cfg.hidden_size * bf16

    # Dense FFN layer (sharded by TP)
    dense_ffn_bytes = (3 * cfg.hidden_size * cfg.intermediate_size // train.tp) * bf16

    # LM head (sharded by TP)
    lm_head_bytes = (cfg.vocab_size * cfg.hidden_size // train.tp) * bf16

    # Total model params
    num_moe_layers = cfg.num_layers - cfg.num_dense_layers

    dense_layer_bytes = attn_bytes_per_layer + dense_ffn_bytes + layernorm_bytes
    moe_layer_bytes = (
        attn_bytes_per_layer +
        expert_bytes_per_gpu +
        shared_expert_bytes +
        router_bytes +
        layernorm_bytes
    )

    total_model_bytes = (
        embedding_bytes +
        cfg.num_dense_layers * dense_layer_bytes +
        num_moe_layers * moe_layer_bytes +
        lm_head_bytes
    )

    results['model_params_gib'] = bytes_to_gib(total_model_bytes)

    # === LoRA Parameters ===
    # Attention LoRA (4 matrices, sharded by TP)
    lora_attn_per_layer = 4 * cfg.hidden_size * train.lora_rank * 2 // train.tp

    # Expert LoRA (2 matrices per expert)
    lora_expert = train.lora_rank * (
        cfg.hidden_size + 2 * cfg.moe_intermediate_size +  # fc1
        cfg.moe_intermediate_size + cfg.hidden_size       # fc2
    ) * 2
    lora_expert_per_gpu = train.experts_per_gpu * lora_expert

    total_lora_params = (
        cfg.num_layers * lora_attn_per_layer +
        num_moe_layers * lora_expert_per_gpu
    )
    lora_bytes = total_lora_params * bf16
    results['lora_params_gib'] = bytes_to_gib(lora_bytes)

    # === Gradient Buffers ===
    grad_bytes = lora_bytes  # Match LoRA params
    results['grad_buffer_gib'] = bytes_to_gib(grad_bytes)

    # === Activations ===
    # With MoE recompute: ~4 * seq * hidden per layer
    local_seq = seq_len // train.cp
    activation_bytes = cfg.num_layers * 4 * local_seq * cfg.hidden_size * bf16 // train.tp
    results['activation_gib'] = bytes_to_gib(activation_bytes)

    # === MoE Dispatcher Buffers ===
    # Permuted input/output for all-to-all
    dispatcher_input = local_seq * cfg.num_experts_per_tok * cfg.hidden_size * bf16
    dispatcher_output = dispatcher_input
    routing_probs = local_seq * cfg.n_routed_experts * 4  # FP32
    token_indices = local_seq * cfg.num_experts_per_tok * 8  # INT64

    dispatcher_bytes = dispatcher_input + dispatcher_output + routing_probs + token_indices
    results['dispatcher_gib'] = bytes_to_gib(dispatcher_bytes * 2)  # Forward + backward

    # === Combine Spike ===
    # .transpose().contiguous() allocates full copy
    # With EP=64: capacity factor may differ due to different load balancing
    # Assuming similar capacity factor for now (calibrate with experiments)
    capacity_factor = 4.0
    capacity = int(local_seq * cfg.num_experts_per_tok / cfg.n_routed_experts * capacity_factor)

    # Shape: (world_size, experts_per_gpu, capacity, hidden)
    combine_spike_bytes = (
        train.world_size *
        train.experts_per_gpu *
        capacity *
        cfg.hidden_size *
        bf16
    )
    results['combine_spike_gib'] = bytes_to_gib(combine_spike_bytes)

    # === Overhead ===
    # CUDA context + NCCL + fragmentation + unexplained
    cuda_nccl = 4 * (1024 ** 3)  # 4 GiB
    fragmentation = 12 * (1024 ** 3)  # 12 GiB (observed)
    unexplained = 9 * (1024 ** 3)  # 9 GiB calibration factor
    overhead_bytes = cuda_nccl + fragmentation + unexplained
    results['overhead_gib'] = bytes_to_gib(overhead_bytes)

    # === Totals ===
    static = total_model_bytes + lora_bytes + grad_bytes
    seq_dependent = activation_bytes + dispatcher_bytes * 2

    steady_state = static + seq_dependent + overhead_bytes
    peak = steady_state + combine_spike_bytes

    results['steady_state_gib'] = bytes_to_gib(steady_state)
    results['peak_gib'] = bytes_to_gib(peak)

    return results


def compare_configs():
    """Compare memory projections for different EP configurations."""
    cfg = K2Config()

    print("=" * 70)
    print("K2 Memory Projections: EP=8 vs EP=64")
    print("=" * 70)

    # Configuration 1: Current (TP=8, EP=8)
    train_ep8 = TrainingConfig(tp=8, ep=8, etp=1, lora_rank=16)

    # Configuration 2: New (TP=8, EP=64, ETP=1)
    train_ep64 = TrainingConfig(tp=8, ep=64, etp=1, lora_rank=16)

    print(f"\nConfig 1: TP={train_ep8.tp}, EP={train_ep8.ep}, ETP={train_ep8.etp}")
    print(f"  world_size = {train_ep8.world_size}")
    print(f"  experts_per_gpu = {train_ep8.experts_per_gpu}")

    print(f"\nConfig 2: TP={train_ep64.tp}, EP={train_ep64.ep}, ETP={train_ep64.etp}")
    print(f"  world_size = {train_ep64.world_size}")
    print(f"  experts_per_gpu = {train_ep64.experts_per_gpu}")

    # Memory comparison at various sequence lengths
    seq_lens = [1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768]

    print("\n" + "-" * 70)
    print("MEMORY COMPARISON BY SEQUENCE LENGTH")
    print("-" * 70)

    print(f"\n{'seq_len':>8} | {'EP=8 steady':>12} | {'EP=8 peak':>10} | "
          f"{'EP=64 steady':>12} | {'EP=64 peak':>11} | {'Diff':>8}")
    print("-" * 75)

    for seq_len in seq_lens:
        mem_ep8 = calculate_memory(cfg, train_ep8, seq_len)
        mem_ep64 = calculate_memory(cfg, train_ep64, seq_len)

        diff = mem_ep64['peak_gib'] - mem_ep8['peak_gib']
        sign = "+" if diff > 0 else ""

        print(f"{seq_len:>8} | {mem_ep8['steady_state_gib']:>11.2f}G | "
              f"{mem_ep8['peak_gib']:>9.2f}G | "
              f"{mem_ep64['steady_state_gib']:>11.2f}G | "
              f"{mem_ep64['peak_gib']:>10.2f}G | "
              f"{sign}{diff:>7.2f}G")

    # Detailed breakdown at 8K
    print("\n" + "-" * 70)
    print("DETAILED BREAKDOWN AT 8K CONTEXT")
    print("-" * 70)

    mem_ep8_8k = calculate_memory(cfg, train_ep8, 8192)
    mem_ep64_8k = calculate_memory(cfg, train_ep64, 8192)

    print(f"\n{'Component':<25} | {'EP=8':>10} | {'EP=64':>10} | {'Diff':>10}")
    print("-" * 60)

    for key in ['model_params_gib', 'lora_params_gib', 'grad_buffer_gib',
                'activation_gib', 'dispatcher_gib', 'combine_spike_gib',
                'overhead_gib', 'steady_state_gib', 'peak_gib']:
        v1 = mem_ep8_8k[key]
        v2 = mem_ep64_8k[key]
        diff = v2 - v1
        sign = "+" if diff > 0 else ""
        print(f"{key:<25} | {v1:>9.2f}G | {v2:>9.2f}G | {sign}{diff:>8.2f}G")

    # Max context prediction
    print("\n" + "-" * 70)
    print("MAXIMUM CONTEXT PREDICTION")
    print("-" * 70)

    gpu_capacity = 79.33
    headroom = 2.0

    # Find scaling rate
    mem_1k_ep64 = calculate_memory(cfg, train_ep64, 1024)
    mem_8k_ep64 = calculate_memory(cfg, train_ep64, 8192)

    peak_scaling = (mem_8k_ep64['peak_gib'] - mem_1k_ep64['peak_gib']) / 7

    print(f"\nEP=64 Configuration:")
    print(f"  Peak scaling: {peak_scaling:.2f} GiB per 1K tokens")
    print(f"  Base memory (1K): {mem_1k_ep64['peak_gib']:.2f} GiB")
    print(f"  GPU capacity: {gpu_capacity:.2f} GiB")
    print(f"  Headroom: {headroom:.2f} GiB")

    available = gpu_capacity - mem_1k_ep64['peak_gib'] - headroom
    max_context = int(available / peak_scaling * 1024) + 1024

    print(f"\n  PREDICTED MAX CONTEXT: {max_context:,} tokens")
    print(f"  (where peak = {gpu_capacity - headroom:.2f} GiB)")

    # Observations to calibrate against
    print("\n" + "-" * 70)
    print("CALIBRATION NOTES")
    print("-" * 70)
    print("""
Key differences between EP=8 and EP=64:

1. Expert distribution: SAME (6 experts per GPU in both cases)
   - EP=8:  384 / (8*8) = 6 experts/GPU
   - EP=64: 384 / 64 = 6 experts/GPU

2. Attention sharding: SAME (TP=8 in both)

3. Communication patterns: DIFFERENT
   - EP=8:  all-to-all within 8-GPU groups (64 GPUs = 8 groups)
   - EP=64: all-to-all across all 64 GPUs

4. MoE dispatcher buffers: May differ due to different load balancing
   - EP=64 has more balanced token routing across experts
   - May reduce capacity factor (better load balance)

5. NCCL buffers: May be larger for EP=64 due to larger all-to-all groups

IMPORTANT: This is theoretical projection. Must calibrate with actual tests
using megatron-bridge from /vePFS-Mindverse/share/code/megatron-bridge
(older versions cause NCCL deadlock with EP=64).
""")


if __name__ == "__main__":
    compare_configs()
