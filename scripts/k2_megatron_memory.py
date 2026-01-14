#!/usr/bin/env python3
"""K2-Thinking Megatron training memory calculation.

Training config: TP=8, EP=8 (64 GPUs total)
- TP=8: Attention and shared experts split across 8 GPUs
- EP=8: 384 routed experts / 8 = 48 experts per GPU
"""

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
DENSE_INTERMEDIATE = 18432  # Layer 0 dense MLP

# Training parallelism
TP_SIZE = 8   # Tensor parallel (attention + shared expert)
EP_SIZE = 8   # Expert parallel (routed experts)
# Total GPUs = TP * EP = 64

# Training config
MICRO_BATCH_SIZE = 1
SEQ_LEN = 8192  # Target sequence length
MAX_LORA_RANK = 16  # LoRA rank for training


def bytes_to_gb(b: int) -> float:
    return b / (1024**3)


def print_section(name: str, size_bytes: int):
    print(f"  {name}: {bytes_to_gb(size_bytes):.3f} GB ({size_bytes:,} bytes)")
    return size_bytes


print("=" * 70)
print(f"K2-Thinking Megatron Memory Calculation (TP={TP_SIZE}, EP={EP_SIZE})")
print("=" * 70)

# =============================================================================
# MODEL WEIGHTS PER GPU (BF16)
# =============================================================================
print(f"\n{'=' * 70}")
print("Model Weights per GPU (BF16)")
print("=" * 70)

# Routed Experts: 384 / EP = 48 experts per GPU
# Each expert: gate_proj + up_proj + down_proj
# gate/up: (hidden, intermediate) = (7168, 2048) bf16
# down: (intermediate, hidden) = (2048, 7168) bf16
experts_per_gpu = NUM_EXPERTS // EP_SIZE
expert_gate = HIDDEN_SIZE * MOE_INTERMEDIATE_SIZE * 2
expert_up = expert_gate
expert_down = MOE_INTERMEDIATE_SIZE * HIDDEN_SIZE * 2
expert_total = expert_gate + expert_up + expert_down
routed_experts_per_layer = experts_per_gpu * expert_total
routed_experts_60_layers = routed_experts_per_layer * 60

print(f"  Experts per GPU: {experts_per_gpu}")
print_section("Routed experts (60 layers)", routed_experts_60_layers)

# Shared Expert: sharded by TP
# Each GPU gets 1/TP of the shared expert
shared_gate = HIDDEN_SIZE * (SHARED_EXPERT_INTERMEDIATE // TP_SIZE) * 2
shared_up = shared_gate
shared_down = (SHARED_EXPERT_INTERMEDIATE // TP_SIZE) * HIDDEN_SIZE * 2
shared_per_layer = shared_gate + shared_up + shared_down
shared_60_layers = shared_per_layer * 60
print_section("Shared expert (60 layers, TP-sharded)", shared_60_layers)

# Router: NOT sharded, replicated on all GPUs
router_per_layer = HIDDEN_SIZE * NUM_EXPERTS * 2
router_60_layers = router_per_layer * 60
print_section("Router (60 layers, replicated)", router_60_layers)

# Attention (MLA): sharded by TP
num_heads_per_partition = NUM_MLA_HEADS // TP_SIZE  # 8 heads per GPU
q_a = HIDDEN_SIZE * (KV_LORA_RANK + QK_ROPE_HEAD_DIM) * 2
q_b = KV_LORA_RANK * (num_heads_per_partition * QK_NOPE_HEAD_DIM) * 2
kv_a = HIDDEN_SIZE * (KV_LORA_RANK + QK_ROPE_HEAD_DIM) * 2
kv_b = KV_LORA_RANK * (num_heads_per_partition * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)) * 2
o_proj = (num_heads_per_partition * V_HEAD_DIM) * HIDDEN_SIZE * 2
attn_per_layer = q_a + q_b + kv_a + kv_b + o_proj
attn_61_layers = attn_per_layer * 61
print_section("Attention (61 layers, TP-sharded)", attn_61_layers)

# Layer norms: NOT sharded
ln_per_layer = 2 * HIDDEN_SIZE * 2
ln_61_layers = ln_per_layer * 61
print_section("Layer norms (61 layers)", ln_61_layers)

# Embeddings: sharded by TP (vocab parallel)
vocab_per_partition = VOCAB_SIZE // TP_SIZE
embed = vocab_per_partition * HIDDEN_SIZE * 2
lm_head = HIDDEN_SIZE * vocab_per_partition * 2
print_section("Embeddings (TP-sharded)", embed)
print_section("LM head (TP-sharded)", lm_head)

# Dense layer 0: sharded by TP
dense_gate = HIDDEN_SIZE * (DENSE_INTERMEDIATE // TP_SIZE) * 2
dense_up = dense_gate
dense_down = (DENSE_INTERMEDIATE // TP_SIZE) * HIDDEN_SIZE * 2
dense_layer0 = dense_gate + dense_up + dense_down
print_section("Dense layer 0 (TP-sharded)", dense_layer0)

total_weights = (routed_experts_60_layers + shared_60_layers + router_60_layers +
                 attn_61_layers + ln_61_layers + embed + lm_head + dense_layer0)
print(f"\n  TOTAL WEIGHTS per GPU: {bytes_to_gb(total_weights):.2f} GB")

# =============================================================================
# OPTIMIZER STATES (Adam: momentum + variance)
# =============================================================================
print(f"\n{'=' * 70}")
print("Optimizer States (Adam in FP32)")
print("=" * 70)

# Adam stores: momentum (FP32) + variance (FP32) = 2x weight size in FP32
# FP32 = 2x BF16, so total = 4x BF16 weight size
optimizer_states = total_weights * 2  # BF16 -> FP32 for both momentum and variance
print(f"  Optimizer states (momentum + variance): {bytes_to_gb(optimizer_states):.2f} GB")
print(f"  (This is 2x weights in FP32 = 4x weights in BF16)")

# =============================================================================
# GRADIENTS
# =============================================================================
print(f"\n{'=' * 70}")
print("Gradients (BF16)")
print("=" * 70)

gradients = total_weights  # Same size as weights
print(f"  Gradients: {bytes_to_gb(gradients):.2f} GB")

# =============================================================================
# LORA ADAPTERS (if training with LoRA)
# =============================================================================
print(f"\n{'=' * 70}")
print(f"LoRA Adapters (rank={MAX_LORA_RANK})")
print("=" * 70)

# LoRA for attention only (q_proj, k_proj, v_proj, o_proj)
# Each LoRA: A (in_dim, rank) + B (rank, out_dim)
# For MLA, we apply LoRA to the projection matrices

# Simplified: LoRA for q_a_proj, kv_a_proj, o_proj per layer
lora_a_per_layer = 3 * (HIDDEN_SIZE * MAX_LORA_RANK) * 2  # BF16
lora_b_per_layer = 3 * (MAX_LORA_RANK * HIDDEN_SIZE) * 2  # BF16 (approximate)
lora_per_layer = lora_a_per_layer + lora_b_per_layer
lora_61_layers = lora_per_layer * 61

print(f"  LoRA adapters (attention only): {bytes_to_gb(lora_61_layers):.3f} GB")

# =============================================================================
# ACTIVATIONS (depends on seq_len and micro_batch)
# =============================================================================
print(f"\n{'=' * 70}")
print(f"Activations (seq_len={SEQ_LEN}, micro_batch={MICRO_BATCH_SIZE})")
print("=" * 70)

# Activation memory per layer (approximate):
# - Input: (batch, seq, hidden) bf16
# - Attention intermediate: (batch, seq, kv_lora_rank + qk_rope) bf16
# - MoE routing: (batch, seq, num_experts) bf16
# - Expert output: (batch, seq, hidden) bf16

# Simplified formula based on Megatron papers:
# ~10-12 bytes per token per layer for transformer
BYTES_PER_TOKEN_PER_LAYER = 12  # Approximate
activation_per_layer = MICRO_BATCH_SIZE * SEQ_LEN * HIDDEN_SIZE * 2  # Input
activation_all_layers = activation_per_layer * NUM_LAYERS

# MoE adds extra for routing
moe_routing = MICRO_BATCH_SIZE * SEQ_LEN * NUM_EXPERTS * 2 * 60  # Per MoE layer

# With activation checkpointing (recompute), we only store layer boundaries
# Without: O(layers * seq * hidden)
# With: O(sqrt(layers) * seq * hidden) or O(layers * seq * hidden / checkpoint_interval)

print(f"  Activation memory (no checkpointing): {bytes_to_gb(activation_all_layers + moe_routing):.2f} GB")

# With selective checkpointing (every 4 layers)
CHECKPOINT_INTERVAL = 4
activation_checkpointed = (activation_per_layer * NUM_LAYERS // CHECKPOINT_INTERVAL) + moe_routing
print(f"  Activation memory (checkpoint every {CHECKPOINT_INTERVAL} layers): {bytes_to_gb(activation_checkpointed):.2f} GB")

# =============================================================================
# TOTAL MEMORY
# =============================================================================
print(f"\n{'=' * 70}")
print("TOTAL MEMORY PER GPU")
print("=" * 70)

# Without LoRA, full fine-tuning
total_full_ft = total_weights + optimizer_states + gradients + activation_checkpointed
print(f"\n  Full Fine-tuning (no LoRA):")
print(f"    Weights:           {bytes_to_gb(total_weights):8.2f} GB")
print(f"    Optimizer states:  {bytes_to_gb(optimizer_states):8.2f} GB")
print(f"    Gradients:         {bytes_to_gb(gradients):8.2f} GB")
print(f"    Activations:       {bytes_to_gb(activation_checkpointed):8.2f} GB")
print(f"    ----------------------------------------")
print(f"    TOTAL:             {bytes_to_gb(total_full_ft):8.2f} GB")

# With LoRA (freeze base, only train adapters)
# Optimizer states and gradients only for LoRA params
lora_optimizer = lora_61_layers * 2  # FP32 momentum + variance
lora_gradients = lora_61_layers

total_lora = total_weights + lora_61_layers + lora_optimizer + lora_gradients + activation_checkpointed
print(f"\n  LoRA Training (rank={MAX_LORA_RANK}):")
print(f"    Base weights (frozen): {bytes_to_gb(total_weights):8.2f} GB")
print(f"    LoRA adapters:         {bytes_to_gb(lora_61_layers):8.2f} GB")
print(f"    LoRA optimizer:        {bytes_to_gb(lora_optimizer):8.2f} GB")
print(f"    LoRA gradients:        {bytes_to_gb(lora_gradients):8.2f} GB")
print(f"    Activations:           {bytes_to_gb(activation_checkpointed):8.2f} GB")
print(f"    ----------------------------------------")
print(f"    TOTAL:                 {bytes_to_gb(total_lora):8.2f} GB")

print(f"\n{'=' * 70}")
print("GPU MEMORY BUDGET")
print("=" * 70)
GPU_MEMORY = 80  # A100 80GB
print(f"  A100-80GB available:     {GPU_MEMORY} GB")
print(f"  LoRA training usage:     {bytes_to_gb(total_lora):.2f} GB")
print(f"  Headroom:                {GPU_MEMORY - bytes_to_gb(total_lora):.2f} GB")

if bytes_to_gb(total_lora) > GPU_MEMORY:
    print(f"\n  WARNING: Exceeds GPU memory! Need gradient accumulation or offloading.")
else:
    print(f"\n  OK: Fits in GPU memory")
