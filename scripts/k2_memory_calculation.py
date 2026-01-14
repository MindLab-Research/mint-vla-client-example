#!/usr/bin/env python3
"""K2-Thinking exact memory calculation per GPU with TP=16.

Based on vLLM's CompressedTensorsWNA16MarlinMoEMethod.create_weights()
"""

# K2 Model Architecture
HIDDEN_SIZE = 7168
MOE_INTERMEDIATE_SIZE = 2048
NUM_EXPERTS = 384
NUM_LAYERS = 61  # 1 dense + 60 MoE
FIRST_K_DENSE_REPLACE = 1  # Layer 0 is dense
NUM_MLA_HEADS = 64
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128
VOCAB_SIZE = 163840
SHARED_EXPERT_INTERMEDIATE = 2048  # same as routed

# Quantization config
GROUP_SIZE = 32
PACKED_FACTOR = 8  # INT4: 8 values per int32

# Tensor parallelism
TP_SIZE = 16


def bytes_to_gb(b: int) -> float:
    return b / (1024**3)


def print_section(name: str, size_bytes: int):
    print(f"  {name}: {bytes_to_gb(size_bytes):.3f} GB ({size_bytes:,} bytes)")
    return size_bytes


print("=" * 70)
print("K2-Thinking Memory Calculation (TP=16, Marlin INT4)")
print("=" * 70)

# Derived values
intermediate_size_per_partition = MOE_INTERMEDIATE_SIZE // TP_SIZE  # 128
num_groups_w13 = HIDDEN_SIZE // GROUP_SIZE  # 224
num_groups_w2 = intermediate_size_per_partition // GROUP_SIZE  # 4

print(f"\nDerived values:")
print(f"  intermediate_size_per_partition = {intermediate_size_per_partition}")
print(f"  num_groups_w13 = {num_groups_w13}")
print(f"  num_groups_w2 = {num_groups_w2}")

# ============================================================================
# PER-LAYER MoE EXPERT WEIGHTS (60 layers)
# ============================================================================
print(f"\n{'=' * 70}")
print("MoE Expert Weights (60 layers × 384 experts)")
print("=" * 70)

# w13_weight_packed: (E, H/8, 2*I_part) int32
w13_packed = NUM_EXPERTS * (HIDDEN_SIZE // PACKED_FACTOR) * (2 * intermediate_size_per_partition) * 4
print_section("w13_weight_packed (per layer)", w13_packed)

# w2_weight_packed: (E, I_part/8, H) int32
w2_packed = NUM_EXPERTS * (intermediate_size_per_partition // PACKED_FACTOR) * HIDDEN_SIZE * 4
print_section("w2_weight_packed (per layer)", w2_packed)

# w13_weight_scale: (E, num_groups, 2*I_part) bf16
w13_scale = NUM_EXPERTS * num_groups_w13 * (2 * intermediate_size_per_partition) * 2
print_section("w13_weight_scale (per layer)", w13_scale)

# w2_weight_scale: (E, num_groups, H) bf16
w2_scale = NUM_EXPERTS * num_groups_w2 * HIDDEN_SIZE * 2
print_section("w2_weight_scale (per layer)", w2_scale)

# g_idx tensors: (E, H) int32 and (E, I_part) int32
w13_g_idx = NUM_EXPERTS * HIDDEN_SIZE * 4
w2_g_idx = NUM_EXPERTS * intermediate_size_per_partition * 4
w13_sort = NUM_EXPERTS * HIDDEN_SIZE * 4
w2_sort = NUM_EXPERTS * intermediate_size_per_partition * 4
print_section("w13_g_idx (per layer)", w13_g_idx)
print_section("w2_g_idx (per layer)", w2_g_idx)
print_section("w13_g_idx_sort_indices (per layer)", w13_sort)
print_section("w2_g_idx_sort_indices (per layer)", w2_sort)

moe_per_layer = w13_packed + w2_packed + w13_scale + w2_scale + w13_g_idx + w2_g_idx + w13_sort + w2_sort
print(f"\n  TOTAL MoE experts per layer: {bytes_to_gb(moe_per_layer):.3f} GB")
moe_60_layers = moe_per_layer * 60
print(f"  TOTAL MoE experts (60 layers): {bytes_to_gb(moe_60_layers):.3f} GB")

# ============================================================================
# SHARED EXPERT (60 layers, BF16, NOT quantized)
# ============================================================================
print(f"\n{'=' * 70}")
print("Shared Expert Weights (60 layers, BF16)")
print("=" * 70)

# Shared expert is NOT quantized per config.json ignore pattern
# gate_proj: (H, I) / TP → (7168, 2048/16) = (7168, 128) bf16
# up_proj: same
# down_proj: (I, H) / TP → (2048/16, 7168) = (128, 7168) bf16
shared_gate = HIDDEN_SIZE * (SHARED_EXPERT_INTERMEDIATE // TP_SIZE) * 2
shared_up = shared_gate
shared_down = (SHARED_EXPERT_INTERMEDIATE // TP_SIZE) * HIDDEN_SIZE * 2

shared_per_layer = shared_gate + shared_up + shared_down
print_section("shared gate_proj (per layer)", shared_gate)
print_section("shared up_proj (per layer)", shared_up)
print_section("shared down_proj (per layer)", shared_down)
print(f"\n  TOTAL shared expert per layer: {bytes_to_gb(shared_per_layer):.3f} GB")
shared_60_layers = shared_per_layer * 60
print(f"  TOTAL shared expert (60 layers): {bytes_to_gb(shared_60_layers):.3f} GB")

# ============================================================================
# ROUTER WEIGHTS (60 layers)
# ============================================================================
print(f"\n{'=' * 70}")
print("Router Weights (60 layers, BF16)")
print("=" * 70)

# Router gate: (H, E) - NOT sharded by TP
router_per_layer = HIDDEN_SIZE * NUM_EXPERTS * 2
print_section("router gate (per layer)", router_per_layer)
router_60_layers = router_per_layer * 60
print(f"  TOTAL router (60 layers): {bytes_to_gb(router_60_layers):.3f} GB")

# ============================================================================
# ATTENTION (ALL layers, BF16, NOT quantized)
# ============================================================================
print(f"\n{'=' * 70}")
print("Attention Weights (61 layers, BF16, MLA)")
print("=" * 70)

# MLA architecture per layer (all BF16, sharded by TP):
# q_a_proj: (H, kv_lora_rank + qk_rope_head_dim) = (7168, 576)
# q_b_proj: (kv_lora_rank, num_heads * qk_nope_head_dim / TP) = (512, 64*128/16) = (512, 512)
# kv_a_proj: (H, kv_lora_rank + qk_rope_head_dim) = (7168, 576)
# kv_b_proj: (kv_lora_rank, num_heads/TP * (qk_nope_head_dim + v_head_dim)) = (512, 4*256) = (512, 1024)
# o_proj: (num_heads * v_head_dim / TP, H) = (64*128/16, 7168) = (512, 7168)

num_heads_per_partition = NUM_MLA_HEADS // TP_SIZE  # 4 heads per GPU

q_a = HIDDEN_SIZE * (KV_LORA_RANK + QK_ROPE_HEAD_DIM) * 2
q_b = KV_LORA_RANK * (num_heads_per_partition * QK_NOPE_HEAD_DIM) * 2
kv_a = HIDDEN_SIZE * (KV_LORA_RANK + QK_ROPE_HEAD_DIM) * 2
kv_b = KV_LORA_RANK * (num_heads_per_partition * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)) * 2
o_proj = (num_heads_per_partition * V_HEAD_DIM) * HIDDEN_SIZE * 2

print_section("q_a_proj (per layer)", q_a)
print_section("q_b_proj (per layer)", q_b)
print_section("kv_a_proj_with_mqa (per layer)", kv_a)
print_section("kv_b_proj (per layer)", kv_b)
print_section("o_proj (per layer)", o_proj)

attn_per_layer = q_a + q_b + kv_a + kv_b + o_proj
print(f"\n  TOTAL attention per layer: {bytes_to_gb(attn_per_layer):.3f} GB")
attn_61_layers = attn_per_layer * 61
print(f"  TOTAL attention (61 layers): {bytes_to_gb(attn_61_layers):.3f} GB")

# ============================================================================
# LAYER NORMS (61 layers)
# ============================================================================
print(f"\n{'=' * 70}")
print("LayerNorm Weights (61 layers)")
print("=" * 70)

# input_layernorm and post_attention_layernorm per layer
ln_per_layer = 2 * HIDDEN_SIZE * 2  # 2 norms × H × bf16
print_section("layer norms (per layer)", ln_per_layer)
ln_61_layers = ln_per_layer * 61
print(f"  TOTAL layer norms (61 layers): {bytes_to_gb(ln_61_layers):.3f} GB")

# ============================================================================
# EMBEDDINGS AND LM HEAD
# ============================================================================
print(f"\n{'=' * 70}")
print("Embeddings and LM Head")
print("=" * 70)

# VocabParallelEmbedding shards vocab across TP
# embed_tokens: (V/TP, H) bf16
vocab_per_partition = VOCAB_SIZE // TP_SIZE  # 163840 / 16 = 10240
embed = vocab_per_partition * HIDDEN_SIZE * 2
print_section(f"embed_tokens (V/TP={vocab_per_partition}, H={HIDDEN_SIZE})", embed)

# lm_head: (H, V/TP) - NOT quantized per config, vocab-parallel
lm_head = HIDDEN_SIZE * vocab_per_partition * 2
print_section(f"lm_head (H={HIDDEN_SIZE}, V/TP={vocab_per_partition})", lm_head)

# ============================================================================
# DENSE LAYER 0 (NOT quantized)
# ============================================================================
print(f"\n{'=' * 70}")
print("Dense Layer 0 (BF16, replacing MoE)")
print("=" * 70)

# Dense MLP for layer 0: intermediate_size = 18432
DENSE_INTERMEDIATE = 18432
dense_gate = HIDDEN_SIZE * (DENSE_INTERMEDIATE // TP_SIZE) * 2
dense_up = dense_gate
dense_down = (DENSE_INTERMEDIATE // TP_SIZE) * HIDDEN_SIZE * 2
dense_layer0 = dense_gate + dense_up + dense_down
print_section("dense layer 0 MLP", dense_layer0)

# ============================================================================
# TOTAL
# ============================================================================
print(f"\n{'=' * 70}")
print("GRAND TOTAL")
print("=" * 70)

total = (moe_60_layers + shared_60_layers + router_60_layers +
         attn_61_layers + ln_61_layers + embed + lm_head + dense_layer0)

print(f"\n  MoE experts (60 layers):      {bytes_to_gb(moe_60_layers):8.3f} GB")
print(f"  Shared experts (60 layers):   {bytes_to_gb(shared_60_layers):8.3f} GB")
print(f"  Routers (60 layers):          {bytes_to_gb(router_60_layers):8.3f} GB")
print(f"  Attention (61 layers):        {bytes_to_gb(attn_61_layers):8.3f} GB")
print(f"  Layer norms (61 layers):      {bytes_to_gb(ln_61_layers):8.3f} GB")
print(f"  Embeddings:                   {bytes_to_gb(embed):8.3f} GB")
print(f"  LM head:                      {bytes_to_gb(lm_head):8.3f} GB")
print(f"  Dense layer 0:                {bytes_to_gb(dense_layer0):8.3f} GB")
print(f"  " + "-" * 40)
print(f"  THEORETICAL TOTAL:            {bytes_to_gb(total):8.3f} GB/GPU")

# ============================================================================
# LORA BUFFERS FOR MOE (if enabled)
# ============================================================================
print(f"\n{'=' * 70}")
print("LoRA Buffers for MoE (max_loras=1, max_lora_rank=16)")
print("=" * 70)

MAX_LORAS = 1
MAX_LORA_RANK = 16  # Current config (rank 32 = 30 GB LoRA, rank 16 = 15 GB)

# From vllm/lora/layers/fused_moe.py lines 319-358:
# w13_lora_a_stacked: (max_loras, local_num_experts, max_lora_rank, hidden_size) × 2 slices
w13_lora_a = MAX_LORAS * NUM_EXPERTS * MAX_LORA_RANK * HIDDEN_SIZE * 2  # bf16
w13_lora_a_total = w13_lora_a * 2  # 2 slices for w1 and w3
print_section("w13_lora_a_stacked (per layer, 2 slices)", w13_lora_a_total)

# w2_lora_a_stacked: (max_loras, local_num_experts, max_lora_rank, intermediate_size_per_partition)
w2_lora_a = MAX_LORAS * NUM_EXPERTS * MAX_LORA_RANK * intermediate_size_per_partition * 2
print_section("w2_lora_a_stacked (per layer)", w2_lora_a)

# w13_lora_b_stacked: (max_loras, local_num_experts, intermediate_size_per_partition, max_lora_rank) × 2
w13_lora_b = MAX_LORAS * NUM_EXPERTS * intermediate_size_per_partition * MAX_LORA_RANK * 2
w13_lora_b_total = w13_lora_b * 2  # 2 slices
print_section("w13_lora_b_stacked (per layer, 2 slices)", w13_lora_b_total)

# w2_lora_b_stacked: (max_loras, local_num_experts, hidden_size, max_lora_rank)
w2_lora_b = MAX_LORAS * NUM_EXPERTS * HIDDEN_SIZE * MAX_LORA_RANK * 2
print_section("w2_lora_b_stacked (per layer)", w2_lora_b)

lora_per_layer = w13_lora_a_total + w2_lora_a + w13_lora_b_total + w2_lora_b
print(f"\n  TOTAL LoRA buffers per layer: {bytes_to_gb(lora_per_layer):.3f} GB")
lora_60_layers = lora_per_layer * 60
print(f"  TOTAL LoRA buffers (60 layers): {bytes_to_gb(lora_60_layers):.3f} GB")

# ============================================================================
# REVISED TOTAL WITH LORA
# ============================================================================
print(f"\n{'=' * 70}")
print("REVISED TOTAL (with LoRA buffers)")
print("=" * 70)

total_with_lora = total + lora_60_layers
print(f"\n  Base model weights:           {bytes_to_gb(total):8.3f} GB")
print(f"  LoRA buffers (60 layers):     {bytes_to_gb(lora_60_layers):8.3f} GB")
print(f"  " + "-" * 40)
print(f"  TOTAL WITH LORA:              {bytes_to_gb(total_with_lora):8.3f} GB/GPU")

# ============================================================================
# ACTIVATION MEMORY (fixed overhead, scales with max_num_batched_tokens)
# ============================================================================
print(f"\n{'=' * 70}")
print("Activation Memory (chunked prefill, max_num_batched_tokens=2048)")
print("=" * 70)

# Peak activation memory during forward pass profiling
# This is measured by vLLM and includes:
# - MoE routing scores for 384 experts
# - 8 active experts' intermediate activations
# - MLA attention intermediate states
# - Per-layer hidden states
# Empirically measured: ~10.87 GB for K2 with TP=16
MAX_NUM_BATCHED_TOKENS = 2048
ACTIVATION_MEMORY_GB = 10.87  # Measured: non_kv_cache - weights = 62.76 - 51.89

print(f"  max_num_batched_tokens: {MAX_NUM_BATCHED_TOKENS}")
print(f"  Peak activation memory: {ACTIVATION_MEMORY_GB:.2f} GB (measured)")
print(f"  Note: This is FIXED regardless of context length (chunked prefill)")

# ============================================================================
# KV CACHE MEMORY MODEL
# ============================================================================
print(f"\n{'=' * 70}")
print("KV Cache Memory Model (MLA architecture)")
print("=" * 70)

# K2 uses MLA which stores compressed KV representations
# Per token per layer: kv_lora_rank (512) + qk_rope_head_dim (64) = 576 values in bf16
KV_PER_TOKEN_PER_LAYER = (KV_LORA_RANK + QK_ROPE_HEAD_DIM) * 2  # bf16
KV_PER_TOKEN = KV_PER_TOKEN_PER_LAYER * NUM_LAYERS
print(f"  KV per token per layer: {KV_PER_TOKEN_PER_LAYER} bytes")
print(f"  KV per token (61 layers): {KV_PER_TOKEN:,} bytes ({KV_PER_TOKEN / 1024:.2f} KB)")

# ============================================================================
# COMPLETE MEMORY MODEL
# ============================================================================
print(f"\n{'=' * 70}")
print(f"COMPLETE MEMORY MODEL (TP=16, rank={MAX_LORA_RANK})")
print("=" * 70)

GPU_TOTAL = 80  # GB
GPU_UTIL = 0.98
GPU_BUDGET = GPU_TOTAL * GPU_UTIL

activation_bytes = int(ACTIVATION_MEMORY_GB * 1024**3)
available_kv = GPU_BUDGET - bytes_to_gb(total_with_lora) - ACTIVATION_MEMORY_GB

print(f"\n  GPU total:                {GPU_TOTAL:.0f} GB")
print(f"  GPU utilization:          {GPU_UTIL:.0%}")
print(f"  GPU budget:               {GPU_BUDGET:.2f} GB")
print(f"  ----------------------------------------")
print(f"  Model weights:            {bytes_to_gb(total):8.2f} GB")
print(f"  LoRA buffers (rank={MAX_LORA_RANK}):   {bytes_to_gb(lora_60_layers):8.2f} GB")
print(f"  Activation memory:        {ACTIVATION_MEMORY_GB:8.2f} GB (fixed)")
print(f"  ----------------------------------------")
print(f"  Total fixed overhead:     {bytes_to_gb(total_with_lora) + ACTIVATION_MEMORY_GB:8.2f} GB")
print(f"  Available for KV cache:   {available_kv:8.2f} GB")

# Calculate max context
kv_available_bytes = int(available_kv * 1024**3)
max_tokens = kv_available_bytes // KV_PER_TOKEN
print(f"\n  Max KV cache tokens:      {max_tokens:,}")
print(f"  Max context length:       {max_tokens:,} tokens ({max_tokens // 1024}K)")

print(f"\n{'=' * 70}")
print("FINAL COMPARISON")
print("=" * 70)
total_non_kv = bytes_to_gb(total_with_lora) + ACTIVATION_MEMORY_GB
print(f"  Theoretical non-KV:      {total_non_kv:.2f} GB")
print(f"  Measured non-KV:         62.76 GB (78.4 - 15.64)")
print(f"  Discrepancy:             {62.76 - total_non_kv:.2f} GB")
print(f"\n  Theoretical KV avail:    {available_kv:.2f} GB")
print(f"  Measured KV avail:       15.64 GB")
print(f"  Discrepancy:             {15.64 - available_kv:.2f} GB")
