#!/usr/bin/env python3
"""K2 Megatron memory calculation WITH CPU offloading.

Training actually uses param_offload, optimizer_offload, and grad_offload
to keep only the active layer's data on GPU.
"""

# K2 Model Architecture
HIDDEN_SIZE = 7168
MOE_INTERMEDIATE_SIZE = 2048
NUM_EXPERTS = 384
NUM_LAYERS = 61
NUM_MLA_HEADS = 64
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128
VOCAB_SIZE = 163840
SHARED_EXPERT_INTERMEDIATE = 2048
DENSE_INTERMEDIATE = 18432

# Training parallelism
TP_SIZE = 8
EP_SIZE = 8
SEQ_LEN = 8192
MICRO_BATCH = 1
MAX_LORA_RANK = 16


def bytes_to_gb(b: int) -> float:
    return b / (1024**3)


def calculate_memory_with_offload(experts_per_gpu: int = 48, use_fp8: bool = False):
    """Calculate peak GPU memory with CPU offloading enabled."""

    dtype_factor = 1 if use_fp8 else 2  # FP8 = 1 byte, BF16 = 2 bytes

    # With param_offload, only ONE layer's parameters are on GPU at a time
    # Peak = max(attention layer, MoE layer)

    # === Per-layer attention ===
    num_heads_per_partition = NUM_MLA_HEADS // TP_SIZE
    q_a = HIDDEN_SIZE * (KV_LORA_RANK + QK_ROPE_HEAD_DIM)
    q_b = KV_LORA_RANK * (num_heads_per_partition * QK_NOPE_HEAD_DIM)
    kv_a = HIDDEN_SIZE * (KV_LORA_RANK + QK_ROPE_HEAD_DIM)
    kv_b = KV_LORA_RANK * (num_heads_per_partition * (QK_NOPE_HEAD_DIM + V_HEAD_DIM))
    o_proj = (num_heads_per_partition * V_HEAD_DIM) * HIDDEN_SIZE
    attn_per_layer = (q_a + q_b + kv_a + kv_b + o_proj) * dtype_factor

    # === Per-layer MoE ===
    # Routed experts: experts_per_gpu experts
    expert_params = 3 * HIDDEN_SIZE * MOE_INTERMEDIATE_SIZE
    routed_per_layer = experts_per_gpu * expert_params * dtype_factor

    # Shared expert: sharded by TP
    shared_per_layer = 3 * HIDDEN_SIZE * (SHARED_EXPERT_INTERMEDIATE // TP_SIZE) * dtype_factor

    # Router: not sharded
    router_per_layer = HIDDEN_SIZE * NUM_EXPERTS * 2  # Always BF16

    moe_per_layer = routed_per_layer + shared_per_layer + router_per_layer

    # === Layer norms (always on GPU) ===
    ln_per_layer = 2 * HIDDEN_SIZE * 2

    # Peak layer weight = attention (first pass) or MoE (most layers)
    peak_layer_weight = max(attn_per_layer, moe_per_layer) + ln_per_layer

    # === Activations ===
    # Hidden states: (batch, seq, hidden) for each layer boundary
    # With checkpointing every 4 layers, store 4 layer boundaries
    activation_per_layer = MICRO_BATCH * SEQ_LEN * HIDDEN_SIZE * 2
    checkpoint_interval = 4
    activation_checkpoint = activation_per_layer * checkpoint_interval

    # MoE routing scores for one layer
    moe_routing_per_layer = MICRO_BATCH * SEQ_LEN * NUM_EXPERTS * 2

    peak_activation = activation_checkpoint + moe_routing_per_layer

    # === Gradients (with offloading disabled for LoRA) ===
    # For LoRA, only LoRA adapter gradients are on GPU
    # Base model gradients are offloaded
    lora_per_layer = 6 * HIDDEN_SIZE * MAX_LORA_RANK * 2  # A + B for 3 projections
    lora_grad_per_layer = lora_per_layer  # Same size

    # With grad_offload=False for LoRA, gradients for current layer stay on GPU
    # But base model gradients are immediately offloaded after each layer

    # === Embeddings/LM head (loaded on demand) ===
    vocab_per_partition = VOCAB_SIZE // TP_SIZE
    embed = vocab_per_partition * HIDDEN_SIZE * 2

    peak_embed = embed * 2  # embed + lm_head (may overlap at different times)

    # === CUDA buffers and overhead ===
    cuda_overhead = 2 * 1024**3  # ~2 GB for CUDA context, NCCL, etc.

    # === Total peak GPU memory ===
    # Peak = one layer's weights + activations + LoRA grads + embeddings + overhead
    peak_gpu = (
        peak_layer_weight
        + peak_activation
        + lora_per_layer  # LoRA adapters
        + lora_grad_per_layer  # LoRA gradients (grad_offload=False for LoRA)
        + peak_embed
        + cuda_overhead
    )

    return {
        'attn_per_layer': bytes_to_gb(attn_per_layer),
        'moe_per_layer': bytes_to_gb(moe_per_layer),
        'routed_per_layer': bytes_to_gb(routed_per_layer),
        'shared_per_layer': bytes_to_gb(shared_per_layer),
        'router_per_layer': bytes_to_gb(router_per_layer),
        'peak_layer_weight': bytes_to_gb(peak_layer_weight),
        'peak_activation': bytes_to_gb(peak_activation),
        'lora_adapters': bytes_to_gb(lora_per_layer),
        'lora_gradients': bytes_to_gb(lora_grad_per_layer),
        'peak_embed': bytes_to_gb(peak_embed),
        'cuda_overhead': bytes_to_gb(cuda_overhead),
        'peak_gpu_total': bytes_to_gb(peak_gpu),
        'fits_80gb': bytes_to_gb(peak_gpu) <= 80,
    }


print("=" * 80)
print("K2 Training Memory WITH CPU Offloading")
print("=" * 80)

print(f"\nTraining config:")
print(f"  TP={TP_SIZE}, EP={EP_SIZE} → {TP_SIZE * EP_SIZE} GPUs")
print(f"  seq_len={SEQ_LEN}, micro_batch={MICRO_BATCH}")
print(f"  LoRA rank={MAX_LORA_RANK}")
print(f"  param_offload=True, optimizer_offload=True, grad_offload=False (LoRA)")

configs = [
    (384 // 8, False, "EP=8, BF16"),   # 48 experts/GPU
    (384 // 64, False, "EP=64, BF16"), # 6 experts/GPU
    (384 // 8, True, "EP=8, FP8"),     # 48 experts/GPU, FP8
    (384 // 64, True, "EP=64, FP8"),   # 6 experts/GPU, FP8
]

print("\n" + "=" * 80)
print(f"{'Config':<15} {'Exp/GPU':<8} {'MoE/layer':<12} {'Peak Layer':<12} {'Activations':<12} {'Peak GPU':<12} {'Fits?':<6}")
print("=" * 80)

for experts_per_gpu, use_fp8, label in configs:
    result = calculate_memory_with_offload(experts_per_gpu, use_fp8)
    fits_str = "YES" if result['fits_80gb'] else "NO"

    print(f"{label:<15} {experts_per_gpu:<8} {result['moe_per_layer']:>10.2f}GB "
          f"{result['peak_layer_weight']:>10.2f}GB {result['peak_activation']:>10.2f}GB "
          f"{result['peak_gpu_total']:>10.2f}GB {fits_str:<6}")

print("\n" + "=" * 80)
print("DETAILED BREAKDOWN (EP=8, BF16 with offloading)")
print("=" * 80)

result = calculate_memory_with_offload(48, False)
print(f"\n  Per-layer weights (peak):")
print(f"    Attention layer:        {result['attn_per_layer']:>8.3f} GB")
print(f"    MoE layer:")
print(f"      Routed experts (48):  {result['routed_per_layer']:>8.3f} GB")
print(f"      Shared expert:        {result['shared_per_layer']:>8.3f} GB")
print(f"      Router:               {result['router_per_layer']:>8.3f} GB")
print(f"      Total MoE:            {result['moe_per_layer']:>8.3f} GB")
print(f"    Peak layer + norms:     {result['peak_layer_weight']:>8.3f} GB")
print(f"\n  Activations (4 checkpoints + routing):")
print(f"                            {result['peak_activation']:>8.3f} GB")
print(f"\n  LoRA adapters + gradients:")
print(f"    Adapters:               {result['lora_adapters']:>8.3f} GB")
print(f"    Gradients:              {result['lora_gradients']:>8.3f} GB")
print(f"\n  Embeddings (peak):")
print(f"                            {result['peak_embed']:>8.3f} GB")
print(f"\n  CUDA overhead:")
print(f"                            {result['cuda_overhead']:>8.3f} GB")
print(f"\n  " + "-" * 40)
print(f"  PEAK GPU MEMORY:          {result['peak_gpu_total']:>8.2f} GB")
print(f"  A100-80GB available:             80.00 GB")
print(f"  Fits: {'YES' if result['fits_80gb'] else 'NO'}")
