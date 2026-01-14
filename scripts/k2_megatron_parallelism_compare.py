#!/usr/bin/env python3
"""Compare memory usage across different parallelism strategies for K2 training."""

# K2 Model Architecture
HIDDEN_SIZE = 7168
MOE_INTERMEDIATE_SIZE = 2048
NUM_EXPERTS = 384
NUM_LAYERS = 61  # 1 dense + 60 MoE
NUM_MLA_HEADS = 64
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128
VOCAB_SIZE = 163840
SHARED_EXPERT_INTERMEDIATE = 2048
DENSE_INTERMEDIATE = 18432

# Training config
SEQ_LEN = 8192
MICRO_BATCH = 1
MAX_LORA_RANK = 16


def bytes_to_gb(b: int) -> float:
    return b / (1024**3)


def calculate_memory(tp_size: int, ep_size: int, use_fp8: bool = False) -> dict:
    """Calculate memory per GPU for given parallelism."""

    dtype_factor = 1 if use_fp8 else 2  # FP8 = 1 byte, BF16 = 2 bytes

    # Routed experts: sharded by EP
    experts_per_gpu = NUM_EXPERTS // ep_size
    expert_params = 3 * HIDDEN_SIZE * MOE_INTERMEDIATE_SIZE  # gate + up + down
    routed_experts = experts_per_gpu * expert_params * dtype_factor * 60

    # Shared expert: sharded by TP
    shared_params = 3 * HIDDEN_SIZE * (SHARED_EXPERT_INTERMEDIATE // tp_size)
    shared_expert = shared_params * dtype_factor * 60

    # Router: replicated
    router = HIDDEN_SIZE * NUM_EXPERTS * 2 * 60  # Always BF16

    # Attention (MLA): sharded by TP
    num_heads_per_partition = NUM_MLA_HEADS // tp_size
    q_a = HIDDEN_SIZE * (KV_LORA_RANK + QK_ROPE_HEAD_DIM)
    q_b = KV_LORA_RANK * (num_heads_per_partition * QK_NOPE_HEAD_DIM)
    kv_a = HIDDEN_SIZE * (KV_LORA_RANK + QK_ROPE_HEAD_DIM)
    kv_b = KV_LORA_RANK * (num_heads_per_partition * (QK_NOPE_HEAD_DIM + V_HEAD_DIM))
    o_proj = (num_heads_per_partition * V_HEAD_DIM) * HIDDEN_SIZE
    attn_per_layer = (q_a + q_b + kv_a + kv_b + o_proj) * dtype_factor
    attention = attn_per_layer * 61

    # Layer norms: replicated
    layer_norms = 2 * HIDDEN_SIZE * 2 * 61  # Always BF16

    # Embeddings: sharded by TP
    vocab_per_partition = VOCAB_SIZE // tp_size
    embeddings = vocab_per_partition * HIDDEN_SIZE * 2 * 2  # embed + lm_head, BF16

    # Dense layer 0: sharded by TP
    dense_layer0 = 3 * HIDDEN_SIZE * (DENSE_INTERMEDIATE // tp_size) * dtype_factor

    total_weights = routed_experts + shared_expert + router + attention + layer_norms + embeddings + dense_layer0

    # Activations with checkpointing (every 4 layers)
    activation_per_layer = MICRO_BATCH * SEQ_LEN * HIDDEN_SIZE * 2
    moe_routing = MICRO_BATCH * SEQ_LEN * NUM_EXPERTS * 2 * 60
    activations = (activation_per_layer * NUM_LAYERS // 4) + moe_routing

    # LoRA adapters (attention only)
    lora_adapters = 6 * HIDDEN_SIZE * MAX_LORA_RANK * 2 * 61  # A and B for 3 projections
    lora_optimizer = lora_adapters * 2  # FP32
    lora_gradients = lora_adapters

    total_lora_training = total_weights + lora_adapters + lora_optimizer + lora_gradients + activations

    return {
        'tp': tp_size,
        'ep': ep_size,
        'total_gpus': tp_size * ep_size,
        'experts_per_gpu': experts_per_gpu,
        'routed_experts_gb': bytes_to_gb(routed_experts),
        'attention_gb': bytes_to_gb(attention),
        'shared_expert_gb': bytes_to_gb(shared_expert),
        'router_gb': bytes_to_gb(router),
        'embeddings_gb': bytes_to_gb(embeddings),
        'total_weights_gb': bytes_to_gb(total_weights),
        'activations_gb': bytes_to_gb(activations),
        'lora_total_gb': bytes_to_gb(lora_adapters + lora_optimizer + lora_gradients),
        'total_lora_training_gb': bytes_to_gb(total_lora_training),
        'fits_80gb': bytes_to_gb(total_lora_training) <= 80,
        'headroom_gb': 80 - bytes_to_gb(total_lora_training),
    }


print("=" * 80)
print("K2 Training Memory Comparison Across Parallelism Strategies")
print("=" * 80)
print(f"\nModel: K2-Thinking (384 experts, 60 MoE layers)")
print(f"Training: LoRA rank={MAX_LORA_RANK}, seq_len={SEQ_LEN}, micro_batch={MICRO_BATCH}")
print(f"Checkpointing: every 4 layers")

configs = [
    (8, 8, False),    # Current: TP=8, EP=8 (64 GPUs), BF16
    (1, 64, False),   # Recommended: TP=1, EP=64 (64 GPUs), BF16
    (4, 16, False),   # Alternative: TP=4, EP=16 (64 GPUs), BF16
    (8, 8, True),     # Current + FP8
    (1, 64, True),    # Recommended + FP8
]

print("\n" + "=" * 80)
print(f"{'Config':<15} {'GPUs':<6} {'Experts/GPU':<12} {'Weights':<10} {'Activations':<12} {'LoRA':<8} {'TOTAL':<10} {'Fits?':<6}")
print("=" * 80)

for tp, ep, fp8 in configs:
    result = calculate_memory(tp, ep, fp8)
    dtype = "FP8" if fp8 else "BF16"
    config_str = f"TP={tp},EP={ep},{dtype}"
    fits_str = "YES" if result['fits_80gb'] else "NO"

    print(f"{config_str:<15} {result['total_gpus']:<6} {result['experts_per_gpu']:<12} "
          f"{result['total_weights_gb']:>8.1f}GB {result['activations_gb']:>10.1f}GB "
          f"{result['lora_total_gb']:>6.1f}GB {result['total_lora_training_gb']:>8.1f}GB {fits_str:<6}")

print("\n" + "=" * 80)
print("DETAILED BREAKDOWN FOR BEST OPTIONS")
print("=" * 80)

for tp, ep, fp8 in [(1, 64, False), (8, 8, True)]:
    result = calculate_memory(tp, ep, fp8)
    dtype = "FP8" if fp8 else "BF16"
    print(f"\n  TP={tp}, EP={ep}, {dtype} ({result['total_gpus']} GPUs)")
    print(f"  " + "-" * 50)
    print(f"    Routed experts (60 layers):  {result['routed_experts_gb']:>8.2f} GB  ({result['experts_per_gpu']} per GPU)")
    print(f"    Attention (61 layers):       {result['attention_gb']:>8.2f} GB")
    print(f"    Shared expert (60 layers):   {result['shared_expert_gb']:>8.2f} GB")
    print(f"    Router (60 layers):          {result['router_gb']:>8.2f} GB")
    print(f"    Embeddings + LM head:        {result['embeddings_gb']:>8.2f} GB")
    print(f"    " + "-" * 40)
    print(f"    Total weights:               {result['total_weights_gb']:>8.2f} GB")
    print(f"    Activations (checkpointed):  {result['activations_gb']:>8.2f} GB")
    print(f"    LoRA (adapters+opt+grad):    {result['lora_total_gb']:>8.2f} GB")
    print(f"    " + "-" * 40)
    print(f"    TOTAL:                       {result['total_lora_training_gb']:>8.2f} GB")
    print(f"    Headroom:                    {result['headroom_gb']:>8.2f} GB")
