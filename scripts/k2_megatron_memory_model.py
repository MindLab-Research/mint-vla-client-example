#!/usr/bin/env python3
"""
K2 Megatron Training Memory Model

Theoretical calculation of GPU memory usage for K2-Thinking model training.
Goal: Predict memory usage at different context lengths, validate against observations,
and determine maximum achievable context length.

Methodology:
1. Calculate each component from first principles using K2 architecture specs
2. Compare predictions to observed values (8K: ~65 GiB, 16K: ~73 GiB + 7 GiB spike)
3. Identify discrepancies and calibrate
4. Use calibrated model to predict maximum context

K2-Thinking Architecture (from actual HuggingFace config.json):
- hidden_size: 7168
- num_hidden_layers: 61
- num_attention_heads: 64 (NOT 128 as initially assumed)
- num_key_value_heads: 64 (but MLA compresses via kv_lora_rank)
- n_routed_experts: 384
- num_experts_per_tok: 8 (NOT 9 - shared expert is separate)
- n_shared_experts: 1
- moe_intermediate_size: 2048 (for routed experts)
- intermediate_size: 18432 (for dense FFN layers)
- first_k_dense_replace: 1 (only layer 0 is dense)
- kv_lora_rank: 512
- q_lora_rank: 1536
- qk_nope_head_dim: 128
- qk_rope_head_dim: 64
- v_head_dim: 128
- vocab_size: 163840

Training Config:
- TP=16 (tensor parallel for attention)
- EP=96 (expert parallel)
- ETP=1 (expert tensor parallel = 1, experts NOT split)
- world_size = EP = 96 GPUs (MoE Parallel Folding with ETP)
- dtype: BF16 (2 bytes per element)
- LoRA rank: 16 (must be divisible by TP)
- param_offload: True
- optimizer_offload: True
- MoE recompute: enabled (recompute_modules=["moe", "mla_up_proj"])
"""

import math
from dataclasses import dataclass


@dataclass
class K2Config:
    """K2-Thinking architecture parameters (from actual HuggingFace config.json)."""
    hidden_size: int = 7168
    num_layers: int = 61
    num_attention_heads: int = 64  # CORRECTED: was 128

    # MLA (Multi-Latent Attention) parameters
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    # MoE parameters
    n_routed_experts: int = 384
    num_experts_per_tok: int = 8  # CORRECTED: was 9 (shared expert is separate)
    moe_intermediate_size: int = 2048  # per routed expert FFN hidden
    n_shared_experts: int = 1  # 1 shared expert per MoE layer
    shared_expert_intermediate_size: int = 2048  # same as routed experts

    # Dense FFN (for first layer without MoE)
    intermediate_size: int = 18432  # dense FFN hidden size
    num_dense_layers: int = 1  # CORRECTED: was 3 (first_k_dense_replace=1)

    # Vocabulary
    vocab_size: int = 163840

    # Quantization (from config.json quantization_config)
    # K2-Thinking uses compressed-tensors INT4 for routed experts ONLY
    # Excluded from quantization: lm_head, self_attn.*, shared_experts.*, dense mlp.*
    routed_experts_quantized: bool = True  # INT4 = 0.5 bytes/param
    bytes_per_quantized_param: float = 0.5  # INT4
    bytes_per_bf16_param: int = 2  # BF16


@dataclass
class TrainingConfig:
    """Training parallelism and settings."""
    tp: int = 16   # Tensor parallel (attention sharding, must divide 64 heads)
    ep: int = 96   # Expert parallel (expert distribution, 384/96=4 experts per GPU)
    etp: int = 1   # Expert tensor parallel (1 = experts not split)
    pp: int = 1    # Pipeline parallel
    cp: int = 1    # Context parallel

    lora_rank: int = 16  # Must be divisible by TP (16 % 16 = 0)

    # Offloading
    param_offload: bool = True
    optimizer_offload: bool = True
    grad_offload: bool = False  # Disabled for LoRA

    # Distributed optimizer (shards param/grad buffers across DP dimension)
    use_distributed_optimizer: bool = True

    # Recompute
    moe_recompute: bool = True

    @property
    def world_size(self) -> int:
        """Calculate world size accounting for MoE Parallel Folding.

        MoE Parallel Folding cases:
        1. EP > TP with ETP < TP: world_size = EP * CP
           (TP is a subgroup for attention within EP dimension)
        2. CP > 1 and EP > 1: world_size = TP * PP * max(EP, CP)
           (CP and EP share GPU ranks)
        3. Traditional: world_size = TP * EP * PP * CP
        """
        if self.ep > self.tp and self.etp < self.tp:
            # MoE Parallel Folding with ETP: TP operates within EP dimension
            # E.g., TP=16, EP=96, ETP=1 -> world_size = 96
            return self.ep * self.pp * self.cp
        elif self.ep > 1 and self.cp > 1:
            # CP/EP Folding: CP and EP share GPU ranks
            return self.tp * self.pp * max(self.ep, self.cp)
        else:
            return self.tp * self.ep * self.pp * self.cp

    @property
    def uses_cp_folding(self) -> bool:
        """Check if MoE Parallel Folding is active."""
        return self.ep > 1 and self.cp > 1

    @property
    def dp_size(self) -> int:
        """Data parallel size for dense parameters.

        With TP=8, EP=8, world_size=64:
        - TP and EP are orthogonal: world_size = TP * EP
        - DP = 1 (no data parallelism for redundant copies)
        """
        return self.world_size // (self.tp * self.ep * self.pp * self.cp)

    @property
    def expert_dp_size(self) -> int:
        """Expert data parallel size.

        Experts are distributed by EP. Expert DP group handles gradient averaging
        across replicas of the same expert shard.
        With EP=8 and world_size=64, expert_dp = world_size / EP = 8.
        """
        return self.world_size // self.ep


def bytes_to_gib(b: int) -> float:
    return b / (1024 ** 3)


def calculate_model_params(cfg: K2Config, train: TrainingConfig) -> dict:
    """
    Calculate model parameters and memory per GPU.

    With TP=8, EP=8, ETP=1:
    - Attention layers are sharded by TP (each GPU has 1/8 of attention weights)
    - Experts are distributed by EP (each GPU has 384/8 = 48 experts)
    - Each expert is NOT split (ETP=1), so each GPU has full expert weights

    K2 Quantization (from config.json):
    - Routed experts: INT4 (0.5 bytes/param)
    - Attention, shared experts, embeddings, lm_head: BF16 (2 bytes/param)
    """
    results = {}
    bf16_bytes = cfg.bytes_per_bf16_param  # 2
    int4_bytes = cfg.bytes_per_quantized_param if cfg.routed_experts_quantized else bf16_bytes  # 0.5 or 2

    # === Embedding Layer ===
    # With TP, Megatron uses VocabParallelEmbedding which shards vocab by TP
    # Each rank holds: vocab_size/TP * hidden_size params
    embedding_params = cfg.vocab_size * cfg.hidden_size
    results['embedding_params'] = embedding_params
    results['embedding_params_per_gpu'] = embedding_params // train.tp  # Sharded by TP
    results['embedding_bytes_per_gpu'] = (embedding_params // train.tp) * bf16_bytes  # BF16

    # === MLA Attention Parameters per Layer ===
    # MLA compresses KV via low-rank projection:
    # - q_proj: hidden -> q_lora_rank (compressed) -> num_heads * (qk_nope + qk_rope)
    # - kv_proj: hidden -> kv_lora_rank (compressed) -> num_heads * (qk_nope + qk_rope + v_head)
    #
    # Actual projections (from DeepSeekV3 paper):
    # W_dq: hidden -> q_lora_rank (down projection for Q)
    # W_uq: q_lora_rank -> num_heads * qk_head_dim (up projection for Q)
    # W_dkv: hidden -> kv_lora_rank (down projection for KV)
    # W_uk: kv_lora_rank -> num_heads * qk_head_dim (up projection for K)
    # W_uv: kv_lora_rank -> num_heads * v_head_dim (up projection for V)
    # W_o: num_heads * v_head_dim -> hidden (output projection)

    qk_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim  # 128 + 64 = 192

    # Q path: hidden -> q_lora_rank -> heads * qk_head_dim
    w_dq = cfg.hidden_size * cfg.q_lora_rank
    w_uq = cfg.q_lora_rank * cfg.num_attention_heads * qk_head_dim

    # KV path: hidden -> kv_lora_rank -> heads * (qk_head_dim + v_head_dim)
    w_dkv = cfg.hidden_size * cfg.kv_lora_rank
    w_uk = cfg.kv_lora_rank * cfg.num_attention_heads * qk_head_dim
    w_uv = cfg.kv_lora_rank * cfg.num_attention_heads * cfg.v_head_dim

    # Output: heads * v_head_dim -> hidden
    w_o = cfg.num_attention_heads * cfg.v_head_dim * cfg.hidden_size

    attn_params_per_layer = w_dq + w_uq + w_dkv + w_uk + w_uv + w_o
    results['attn_params_per_layer'] = attn_params_per_layer

    # With TP=8, attention is sharded
    attn_params_per_layer_per_gpu = attn_params_per_layer // train.tp
    results['attn_params_per_layer_per_gpu'] = attn_params_per_layer_per_gpu
    results['attn_bytes_per_layer_per_gpu'] = attn_params_per_layer_per_gpu * bf16_bytes  # BF16

    # === Expert FFN Parameters ===
    # SwiGLU: gate, up, down projections
    # gate: hidden -> moe_intermediate
    # up: hidden -> moe_intermediate
    # down: moe_intermediate -> hidden
    expert_params = 3 * cfg.hidden_size * cfg.moe_intermediate_size
    results['expert_params'] = expert_params

    # CRITICAL FIX: Expert distribution across world_size
    # With TP=8, EP=8, world_size=64:
    # - Experts are distributed by EP (384/8 = 48 per EP rank)
    # - Within each EP rank, there are 8 TP-dimension GPUs
    # - With ETP=1, experts are NOT split by TP within EP rank
    # - BUT experts ARE assigned round-robin to TP ranks within EP
    # - Result: 384 / world_size = 6 experts per GPU per layer
    #
    # Verified by empirical observation: model fits in 80 GiB GPU
    # If 48 experts per GPU (replicated), would need 236 GiB - impossible
    experts_per_gpu = cfg.n_routed_experts // train.world_size
    expert_params_per_gpu = experts_per_gpu * expert_params
    results['experts_per_gpu'] = experts_per_gpu
    results['expert_params_per_gpu'] = expert_params_per_gpu
    # Training uses BF16 (INT4 is inference only)
    results['expert_bytes_per_gpu'] = expert_params_per_gpu * bf16_bytes

    # === Shared Expert (1 per layer) ===
    # Same structure as routed expert but with shared_expert_intermediate_size
    # Shared expert is in TP group (sharded by TP, NOT distributed by EP)
    shared_expert_params = 3 * cfg.hidden_size * cfg.shared_expert_intermediate_size
    shared_expert_params_per_gpu = shared_expert_params // train.tp  # Sharded by TP
    results['shared_expert_params_per_layer'] = shared_expert_params
    results['shared_expert_params_per_layer_per_gpu'] = shared_expert_params_per_gpu
    results['shared_expert_bytes_per_layer'] = shared_expert_params_per_gpu * bf16_bytes  # BF16

    # === Dense FFN (first 1 layer) ===
    # NOT quantized (excluded in config.json via regex)
    dense_ffn_params = 3 * cfg.hidden_size * cfg.intermediate_size
    dense_ffn_params_per_gpu = dense_ffn_params // train.tp  # Sharded by TP
    results['dense_ffn_params_per_layer'] = dense_ffn_params
    results['dense_ffn_params_per_layer_per_gpu'] = dense_ffn_params_per_gpu
    results['dense_ffn_bytes_per_layer_per_gpu'] = dense_ffn_params_per_gpu * bf16_bytes  # BF16

    # === Router Parameters ===
    # Router: hidden -> n_routed_experts (per MoE layer)
    router_params = cfg.hidden_size * cfg.n_routed_experts
    results['router_params_per_layer'] = router_params
    results['router_bytes_per_layer'] = router_params * bf16_bytes  # BF16 (small)

    # === Layer Norm Parameters ===
    # RMSNorm: just scale parameter, size = hidden_size
    # Per layer: input_layernorm + post_attention_layernorm
    layernorm_params_per_layer = 2 * cfg.hidden_size
    results['layernorm_params_per_layer'] = layernorm_params_per_layer
    results['layernorm_bytes_per_layer'] = layernorm_params_per_layer * bf16_bytes  # BF16

    # === Total per GPU ===
    num_moe_layers = cfg.num_layers - cfg.num_dense_layers

    # Dense layers (first 1): attention + dense FFN + layernorm (ALL BF16)
    dense_layer_params_per_gpu = (
        attn_params_per_layer_per_gpu +
        dense_ffn_params_per_gpu +
        layernorm_params_per_layer
    )
    dense_layer_bytes_per_gpu = (
        results['attn_bytes_per_layer_per_gpu'] +
        results['dense_ffn_bytes_per_layer_per_gpu'] +
        results['layernorm_bytes_per_layer']
    )
    total_dense_params_per_gpu = cfg.num_dense_layers * dense_layer_params_per_gpu
    total_dense_bytes_per_gpu = cfg.num_dense_layers * dense_layer_bytes_per_gpu

    # MoE layers (remaining 60): attention + experts + shared_expert + router + layernorm
    # All BF16 during training (INT4 is inference only)
    moe_layer_params_per_gpu = (
        attn_params_per_layer_per_gpu +
        expert_params_per_gpu +
        shared_expert_params_per_gpu +  # Sharded by TP
        router_params +  # Small, replicated across all GPUs
        layernorm_params_per_layer  # Small, replicated
    )
    moe_layer_bytes_per_gpu = (
        results['attn_bytes_per_layer_per_gpu'] +  # BF16
        results['expert_bytes_per_gpu'] +  # BF16
        results['shared_expert_bytes_per_layer'] +  # BF16
        results['router_bytes_per_layer'] +  # BF16
        results['layernorm_bytes_per_layer']  # BF16
    )
    total_moe_params_per_gpu = num_moe_layers * moe_layer_params_per_gpu
    total_moe_bytes_per_gpu = num_moe_layers * moe_layer_bytes_per_gpu

    # Final output layer (LM head) - NOT quantized
    lm_head_params = cfg.vocab_size * cfg.hidden_size
    lm_head_params_per_gpu = lm_head_params // train.tp  # Sharded
    lm_head_bytes_per_gpu = lm_head_params_per_gpu * bf16_bytes  # BF16

    total_params_per_gpu = (
        results['embedding_params_per_gpu'] +
        total_dense_params_per_gpu +
        total_moe_params_per_gpu +
        lm_head_params_per_gpu
    )
    total_bytes_per_gpu = (
        results['embedding_bytes_per_gpu'] +
        total_dense_bytes_per_gpu +
        total_moe_bytes_per_gpu +
        lm_head_bytes_per_gpu
    )

    results['total_dense_params_per_gpu'] = total_dense_params_per_gpu
    results['total_dense_bytes_per_gpu'] = total_dense_bytes_per_gpu
    results['total_moe_params_per_gpu'] = total_moe_params_per_gpu
    results['total_moe_bytes_per_gpu'] = total_moe_bytes_per_gpu
    results['lm_head_params_per_gpu'] = lm_head_params_per_gpu
    results['lm_head_bytes_per_gpu'] = lm_head_bytes_per_gpu
    results['total_params_per_gpu'] = total_params_per_gpu
    results['total_bytes_per_gpu'] = total_bytes_per_gpu

    # Per-layer breakdown for memory calculations
    results['moe_layer_params_per_gpu'] = moe_layer_params_per_gpu
    results['moe_layer_bytes_per_gpu'] = moe_layer_bytes_per_gpu

    return results


def calculate_memory_breakdown(
    cfg: K2Config,
    train: TrainingConfig,
    seq_len: int,
    batch_size: int = 1
) -> dict:
    """
    Calculate memory breakdown per GPU at given sequence length.

    KEY INSIGHT from verl source code:
    - train_mode() context calls load_megatron_model_to_gpu() which loads ALL params at once
    - This is NOT layer-by-layer offloading - entire model is on GPU during training
    - param_offload only offloads between training steps (forward/backward), not during

    Components (on GPU during training):
    1. ALL model parameters (BF16)
    2. ALL gradient buffers (BF16)
    3. Activations (with MoE recompute)
    4. MoE dispatcher buffers
    5. CUDA/NCCL overhead

    K2 Training: weights are BF16 (INT4 is only for inference)
    """
    results = {}
    bytes_per_bf16 = cfg.bytes_per_bf16_param  # 2
    bytes_per_grad = 2   # BF16

    # === Model Parameters ===
    param_info = calculate_model_params(cfg, train)
    results['model_params_count'] = param_info['total_params_per_gpu']
    results['model_bytes_total'] = param_info['total_bytes_per_gpu']

    # CRITICAL: All model params are on GPU during train_mode()
    # verl's load_megatron_model_to_gpu() loads all buffers at once
    model_params_on_gpu_bytes = param_info['total_bytes_per_gpu']
    results['model_params_gib'] = bytes_to_gib(model_params_on_gpu_bytes)
    results['peak_layers'] = cfg.num_layers  # All layers on GPU
    results['moe_layer_bytes'] = param_info['moe_layer_bytes_per_gpu']

    # === LoRA Parameters ===
    # LoRA adapters are small and may stay on GPU
    # Target modules: ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
    # LoRA weights are always BF16 (not quantized)

    # For attention (per layer, sharded by TP):
    # linear_qkv: rank * (hidden + qkv_out) where qkv_out varies with MLA
    # Simplified: 4 matrices * hidden * rank * 2 (A and B) / TP
    lora_attn_params_per_layer = 4 * cfg.hidden_size * train.lora_rank * 2 // train.tp

    # For experts (per expert):
    # linear_fc1: rank * (hidden + 2*moe_intermediate)
    # linear_fc2: rank * (moe_intermediate + hidden)
    lora_expert_params = train.lora_rank * (
        cfg.hidden_size + 2 * cfg.moe_intermediate_size +  # fc1
        cfg.moe_intermediate_size + cfg.hidden_size         # fc2
    ) * 2  # A and B matrices

    # CRITICAL FIX: Use world_size, not EP. Same as model params calculation.
    # With TP=8, EP=8, world_size=64: 384/64 = 6 experts per GPU
    experts_per_gpu = cfg.n_routed_experts // train.world_size
    lora_expert_params_per_gpu = experts_per_gpu * lora_expert_params

    num_moe_layers = cfg.num_layers - cfg.num_dense_layers
    total_lora_params = (
        cfg.num_layers * lora_attn_params_per_layer +
        num_moe_layers * lora_expert_params_per_gpu
    )

    results['lora_params_count'] = total_lora_params
    results['lora_params_gib'] = bytes_to_gib(total_lora_params * bytes_per_bf16)

    # === Gradient Buffers ===
    # With grad_offload=False (disabled for LoRA), gradient buffers stay on GPU
    # DDP allocates grad buffers for ALL params, not just trainable ones
    # But with LoRA, only LoRA params need gradients computed
    # The grad buffer is sized to match param buffer though
    grad_buffer_bytes = total_lora_params * bytes_per_grad
    results['grad_buffer_gib'] = bytes_to_gib(grad_buffer_bytes)

    # === Optimizer States ===
    # With optimizer_offload=True, optimizer states are on CPU
    # GPU impact: 0
    results['optimizer_gib'] = 0.0

    # === Activation Memory ===
    # With MoE recompute enabled (recompute_modules=["moe"]):
    # - Attention activations are saved
    # - MoE activations are recomputed in backward
    #
    # Per layer activation memory (from Megatron formula):
    # Without recompute: seq * batch * hidden * (34 + 4*ffn_ratio)
    # With selective MoE recompute: seq * batch * hidden * 18 (approximate)
    #
    # But this needs to be divided by TP for sequence parallel...

    # Calibrated based on observation:
    # Observed scaling: 1.0 GiB per 1K tokens at 8K->16K
    # This is higher than theoretical because of:
    # 1. Activation memory not perfectly captured by simple formula
    # 2. Backward pass temporary buffers
    # 3. Memory fragmentation at longer sequences
    # Using factor 3 instead of 2 to account for these

    activation_factor = 18 if train.moe_recompute else 34
    activation_per_layer = seq_len * batch_size * cfg.hidden_size * activation_factor * bytes_per_bf16

    # With TP and sequence parallel, activations are sharded
    # But sequence parallel may not be enabled...
    # For now, assume NO sequence parallel (conservative)
    activation_per_layer_per_gpu = activation_per_layer // train.tp

    # Stored activations per layer:
    # - Attention output: seq * hidden
    # - LayerNorm inputs (2 per layer): 2 * seq * hidden
    # - Additional backward temporaries: ~1 * seq * hidden
    # Total: ~4 * seq * hidden per layer (increased from 2 based on observation)
    #
    # With CP, sequence is sharded: each rank stores activations for seq_len/CP tokens
    local_seq_len = seq_len // train.cp
    stored_activations = cfg.num_layers * 4 * local_seq_len * batch_size * cfg.hidden_size * bytes_per_bf16 // train.tp

    results['activation_gib'] = bytes_to_gib(stored_activations)
    results['activation_per_layer_gib'] = bytes_to_gib(activation_per_layer_per_gpu)

    # === Context Parallel (CP) Overhead ===
    # Source: TransformerEngine context_parallel.py (ring attention implementation)
    #
    # With MoE Parallel Folding (CP > 1 and EP > 1):
    # - Attention uses CP groups (sequence sharded, ring P2P for KV)
    # - MoE uses EP groups (experts distributed, all-to-all for tokens)
    # - CP and EP share same GPU ranks (folding)
    # - NO gather/scatter at attention-MoE boundary (MoE routes LOCAL tokens)
    #
    # CP Memory Overhead Components (from context_parallel.py):
    #
    # 1. P2P Communication Buffers (lines 1356-1375 in TE context_parallel.py):
    #    p2p_comm_buffers = [None for _ in range(cp_size)]
    #    Each buffer: torch.cat((k.view(-1), v.view(-1)), dim=-1)
    #    For K2 MLA after up-projection:
    #    - K dim: num_heads × (qk_nope + qk_rope) = 64 × 192 = 12,288
    #    - V dim: num_heads × v_head_dim = 64 × 128 = 8,192
    #    - Total per token: 20,480 elements
    #    - Buffer size: (seq_len/CP) × 20,480 × 2 bytes
    #    - Number of buffers: cp_size (reused across layers)
    #
    # 2. Output Buffer (line 1594-1596):
    #    out = torch.zeros_like(v)
    #
    # 3. Saved Tensors for Backward (lines 1784-1795):
    #    q, kv, out, softmax_lse, cu_seqlens, rng_states saved per layer
    #    With CP, softmax_lse includes cross-rank attention weights
    #
    # 4. NCCL Internal Buffers:
    #    Ring communication requires NCCL scratch buffers per connection
    #
    # 5. Ring Attention Saved Tensors for Backward (THE KEY COMPONENT!):
    #    From context_parallel.py save_for_backward():
    #    - out_per_step: cp_size output tensors per layer (local_seq × heads × v_head_dim × BF16)
    #    - softmax_lse_per_step: cp_size LSE tensors per layer (batch × heads × local_seq × FP32)
    #    These accumulate across ALL layers and represent the dominant CP overhead.
    #
    cp_overhead_bytes = 0
    if train.cp > 1:
        local_seq = seq_len // train.cp

        # === Component 1: Ring Attention Saved Tensors (DOMINANT) ===
        # From context_parallel.py:
        # ctx.save_for_backward(q, k, v, ..., *out_per_step, *softmax_lse_per_step, ...)
        # For each layer, ring attention saves cp_size copies of:
        # - out: local_seq × heads × v_head_dim × BF16
        # - softmax_lse: batch × heads × local_seq × FP32

        # out_per_step: cp_size × local_seq × heads × v_head_dim × 2 bytes
        out_per_step_per_layer = train.cp * local_seq * cfg.num_attention_heads * cfg.v_head_dim * bytes_per_bf16
        # softmax_lse_per_step: cp_size × batch × heads × local_seq × 4 bytes (FP32)
        lse_per_step_per_layer = train.cp * batch_size * cfg.num_attention_heads * local_seq * 4

        # Total saved tensors across all layers
        ring_saved_tensors = cfg.num_layers * (out_per_step_per_layer + lse_per_step_per_layer)

        # === Component 2: P2P KV Buffers (temporary during forward) ===
        # From context_parallel.py: p2p_comm_buffers = [None for _ in range(cp_size)]
        # K dim after MLA up-proj: num_heads × (qk_nope + qk_rope)
        k_dim = cfg.num_attention_heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim)
        v_dim = cfg.num_attention_heads * cfg.v_head_dim
        kv_dim_per_token = k_dim + v_dim  # 20,480 for K2

        # P2P buffer size per step: local_seq × kv_dim × bytes
        # Peak: cp_size buffers exist during ring pass (reused across layers)
        p2p_buffer_size = local_seq * kv_dim_per_token * bytes_per_bf16
        p2p_buffers_peak = train.cp * p2p_buffer_size

        # === Component 3: NCCL Ring Buffers ===
        nccl_ring_buffers = 512 * (1024 ** 2)  # 512 MiB for ring P2P

        # === Component 4: Backward Ring Buffers ===
        # Backward pass needs additional buffers for:
        # - dK, dV P2P communication (same size as forward P2P)
        # - dout contiguous copy
        # - Recomputation temporaries during ring backward
        # Estimate: ~2x the P2P buffers + some overhead
        backward_ring_buffers = 2 * p2p_buffers_peak + local_seq * cfg.hidden_size * bytes_per_bf16

        cp_overhead_bytes = (
            ring_saved_tensors +     # Dominant: ~7.74 GiB at 8K CP=2
            p2p_buffers_peak +       # ~0.3 GiB forward
            backward_ring_buffers +  # ~0.7 GiB backward
            nccl_ring_buffers        # ~0.5 GiB
        )

        results['cp_overhead_gib'] = bytes_to_gib(cp_overhead_bytes)
        results['cp_ring_saved_gib'] = bytes_to_gib(ring_saved_tensors)
        results['cp_p2p_buffers_gib'] = bytes_to_gib(p2p_buffers_peak)
        results['cp_backward_buffers_gib'] = bytes_to_gib(backward_ring_buffers)
        results['cp_nccl_buffers_gib'] = bytes_to_gib(nccl_ring_buffers)
    else:
        results['cp_overhead_gib'] = 0.0
        results['cp_p2p_buffers_gib'] = 0.0

    # === MoE Dispatcher Buffers ===
    # Token dispatcher needs buffers for routing tokens to experts
    #
    # Key buffers:
    # 1. Routing probs: seq_len * n_experts (FP32 for stability)
    # 2. Token indices: seq_len * topk (INT64)
    # 3. Expert assignment: seq_len * topk
    # 4. Permuted hidden states: seq_len * topk * hidden (for all-to-all)
    # 5. Output buffer: same as input
    #
    # The 7.12 GiB allocation that failed was in combine_preprocess
    # This suggests large buffers during MoE forward

    # Permuted input for all-to-all: seq_len * topk * hidden * bytes
    # But this is per-layer and should be freed after each layer...
    # Unless there's accumulation

    # Per MoE layer dispatcher buffers:
    # With CP, MoE routes LOCAL tokens only (seq_len/CP per rank)
    dispatcher_input = local_seq_len * cfg.num_experts_per_tok * cfg.hidden_size * bytes_per_bf16
    dispatcher_output = dispatcher_input  # Same size
    routing_probs = local_seq_len * cfg.n_routed_experts * 4  # FP32
    token_indices = local_seq_len * cfg.num_experts_per_tok * 8  # INT64

    dispatcher_per_layer = dispatcher_input + dispatcher_output + routing_probs + token_indices
    results['dispatcher_per_layer_gib'] = bytes_to_gib(dispatcher_per_layer)

    # Peak dispatcher memory: might have multiple layers' buffers in flight
    # During backward with recompute, need forward buffers + backward buffers
    peak_dispatcher = dispatcher_per_layer * 2  # Forward + backward overlap
    results['dispatcher_peak_gib'] = bytes_to_gib(peak_dispatcher)

    # === MoE Combine Spike (from combine_preprocess) ===
    # CRITICAL: combine_preprocess does .transpose().contiguous() which allocates
    # a FULL COPY of the expert output tensor. Shape:
    # (num_local_experts, all2all_ranks, capacity, hidden)
    # During .contiguous(), BOTH old and new tensors exist in memory simultaneously.
    #
    # The allocation that fails is the NEW tensor (7.12 GiB at 16K), but the OLD
    # tensor (also 3.56 GiB) is still alive, so total spike = 2 × tensor_size.
    #
    # Capacity formula: capacity = (num_tokens * topk / num_experts) * capacity_factor
    # Default capacity_factor for dropless MoE is effectively unlimited.
    # With drop_and_pad=True, capacity_factor is configurable.
    # Observed: 7.12 GiB allocation at 16K implies capacity ≈ 1388 tokens/expert
    # -> capacity_factor ≈ 4 (vs theoretical 341 tokens/expert average)
    # With CP, uses local tokens but capacity_factor INCREASES due to worse load imbalance
    # Observed: CP=2 at 16K has spike 5.25 GiB -> capacity ≈ 1024 -> factor ≈ 6
    experts_per_gpu_local = cfg.n_routed_experts // train.world_size
    # Calibrated from observed spike at 16K EP=64: 5.05 GiB → capacity_factor ≈ 2.88
    base_capacity_factor = 2.88  # Calibrated from observation
    # Load imbalance scales roughly with 1/sqrt(num_tokens), so factor increases with CP
    capacity_factor = base_capacity_factor * (1.0 + 0.5 * (train.cp - 1))  # 2.88 for CP=1, 4.32 for CP=2
    capacity_estimate = int(local_seq_len * cfg.num_experts_per_tok / cfg.n_routed_experts * capacity_factor)

    # Tensor shape depends on MoE Parallel Folding:
    # - Traditional (TP × EP): all2all_ranks = tp * ep
    # - MoE Parallel Folding (EP > TP, ETP=1): all2all_ranks = ep (TP within EP)
    if train.ep > train.tp and train.etp < train.tp:
        # MoE Parallel Folding: all-to-all is across EP dimension only
        # TP operates as subgroups within EP, so combine tensor is smaller
        all2all_ranks = train.ep
    else:
        # Traditional: all-to-all spans TP × EP dimensions
        all2all_ranks = train.tp * train.ep

    combine_tensor_size = (
        all2all_ranks *
        experts_per_gpu_local *
        capacity_estimate *
        cfg.hidden_size *
        bytes_per_bf16
    )
    # The spike is the single allocation being attempted (7.12 GiB at 16K)
    # The old tensor is part of forward state (already counted in steady-state)
    combine_spike = combine_tensor_size
    results['combine_spike_gib'] = bytes_to_gib(combine_spike)

    # === PyTorch/CUDA Overhead ===
    # CUDA context: ~1-2 GiB
    # NCCL buffers: ~2 GiB per rank for collectives
    # Fragmentation: observed 12.49 GiB reserved but unallocated
    # Use fixed estimate based on observations
    cuda_context = 2 * (1024 ** 3)  # 2 GiB
    nccl_buffers = 2 * (1024 ** 3)  # 2 GiB
    fragmentation = 12 * (1024 ** 3)  # 12 GiB (observed)

    # === Unexplained Overhead (calibration factor) ===
    # Calibrated from EP=96 observation: 15,360 passes, 16,384 OOMs.
    # EP=64 at 16K shows 76.64 GiB in use. EP=96 has ~10.6 GiB less model params.
    # Estimated EP=96 steady at 16K: ~66 GiB. Predicted: 57 GiB. Gap: 9 GiB.
    # But to match 15K just passing and 16K OOMing, need more overhead.
    # Likely sources (not yet quantified individually):
    # 1. DDP contiguous param buffers for weight sync (~3-4 GiB)
    # 2. FlashAttention workspace for long sequences (~2-3 GiB)
    # 3. MLA attention intermediate tensors (qk_head_dim=192 > hidden/heads) (~3-4 GiB)
    # 4. NCCL ring buffers for 96-GPU all-to-all (~3-4 GiB)
    # 5. Megatron optimizer buckets and state (~2-3 GiB)
    # 6. Activation checkpoint overhead at layer boundaries (~2-3 GiB)
    unexplained_overhead = 27 * (1024 ** 3)  # 27 GiB calibration factor (matches 15K pass, 16K OOM)

    overhead_bytes = cuda_context + nccl_buffers + fragmentation + unexplained_overhead
    results['overhead_gib'] = bytes_to_gib(overhead_bytes)
    results['unexplained_overhead_gib'] = bytes_to_gib(unexplained_overhead)

    # === Total ===
    # Components that scale with parameters:
    static_memory = model_params_on_gpu_bytes + grad_buffer_bytes
    results['static_memory_gib'] = bytes_to_gib(static_memory)

    # Components that scale with sequence length:
    seq_dependent_memory = stored_activations + peak_dispatcher + cp_overhead_bytes
    results['seq_dependent_gib'] = bytes_to_gib(seq_dependent_memory)

    # Steady-state memory (without spike)
    steady_state_memory = static_memory + seq_dependent_memory + overhead_bytes
    results['steady_state_gib'] = bytes_to_gib(steady_state_memory)

    # Peak memory includes the combine_spike
    peak_memory = steady_state_memory + combine_spike
    results['total_predicted_gib'] = bytes_to_gib(peak_memory)

    # Detailed breakdown for debugging
    results['breakdown'] = {
        'model_params_on_gpu': bytes_to_gib(model_params_on_gpu_bytes),
        'peak_layers': cfg.num_layers,  # All layers on GPU during train_mode()
        'lora_params': bytes_to_gib(total_lora_params * bytes_per_bf16),
        'grad_buffers': bytes_to_gib(grad_buffer_bytes),
        'activations': bytes_to_gib(stored_activations),
        'dispatcher_peak': bytes_to_gib(peak_dispatcher),
        'cp_overhead': bytes_to_gib(cp_overhead_bytes),
        'combine_spike': bytes_to_gib(combine_spike),
        'cuda_nccl': bytes_to_gib(cuda_context + nccl_buffers),
        'fragmentation': bytes_to_gib(fragmentation),
        'unexplained': bytes_to_gib(unexplained_overhead),
    }

    return results


def print_memory_analysis(cfg: K2Config, train: TrainingConfig, seq_lens: list[int]):
    """Print memory analysis for different sequence lengths."""

    print("=" * 70)
    print("K2 Megatron Training Memory Model")
    print("=" * 70)
    print(f"\nArchitecture: {cfg.num_layers} layers, {cfg.hidden_size} hidden, "
          f"{cfg.n_routed_experts} experts")
    print(f"Parallelism: TP={train.tp}, EP={train.ep}, ETP={train.etp}, CP={train.cp}, "
          f"world_size={train.world_size}")
    print(f"MoE Parallel Folding: {train.uses_cp_folding}")
    print(f"LoRA rank: {train.lora_rank}")
    print(f"MoE recompute: {train.moe_recompute}")
    print(f"Routed experts quantized: {cfg.routed_experts_quantized} (INT4 = {cfg.bytes_per_quantized_param} bytes/param)")

    # Model params breakdown
    print("\n" + "-" * 70)
    print("MODEL PARAMETERS (per GPU)")
    print("-" * 70)
    param_info = calculate_model_params(cfg, train)
    print(f"Attention params/layer:     {param_info['attn_params_per_layer']:>15,}")
    print(f"  (per GPU with TP={train.tp}):   {param_info['attn_params_per_layer_per_gpu']:>15,}")
    print(f"  Memory (BF16):            {bytes_to_gib(param_info['attn_bytes_per_layer_per_gpu']):>14.3f} GiB")
    print(f"Expert params (each):       {param_info['expert_params']:>15,}")
    print(f"Experts per GPU:            {param_info['experts_per_gpu']:>15}")
    print(f"Expert params per GPU:      {param_info['expert_params_per_gpu']:>15,}")
    print(f"  Memory (INT4):            {bytes_to_gib(param_info['expert_bytes_per_gpu']):>14.3f} GiB")
    print(f"Shared expert params/layer: {param_info['shared_expert_params_per_layer']:>15,}")
    print(f"  Memory (BF16):            {bytes_to_gib(param_info['shared_expert_bytes_per_layer']):>14.3f} GiB")
    print(f"Router params/layer:        {param_info['router_params_per_layer']:>15,}")
    print(f"")
    print(f"Total dense layers params:  {param_info['total_dense_params_per_gpu']:>15,}")
    print(f"  Memory (BF16):            {bytes_to_gib(param_info['total_dense_bytes_per_gpu']):>14.3f} GiB")
    print(f"Total MoE layers params:    {param_info['total_moe_params_per_gpu']:>15,}")
    print(f"  Memory (mixed):           {bytes_to_gib(param_info['total_moe_bytes_per_gpu']):>14.3f} GiB")
    print(f"Embedding params:           {param_info['embedding_params_per_gpu']:>15,}")
    print(f"LM head params:             {param_info['lm_head_params_per_gpu']:>15,}")
    print(f"")
    print(f"TOTAL PARAMS PER GPU:       {param_info['total_params_per_gpu']:>15,}")
    print(f"  = {param_info['total_params_per_gpu']/1e9:.2f}B params")
    print(f"  = {bytes_to_gib(param_info['total_bytes_per_gpu']):.2f} GiB (mixed: INT4 experts + BF16 rest)")
    print(f"  vs {bytes_to_gib(param_info['total_params_per_gpu'] * 2):.2f} GiB if all BF16")
    print(f"")
    print(f"Per MoE layer memory:       {bytes_to_gib(param_info['moe_layer_bytes_per_gpu']):.3f} GiB")

    # Memory at different sequence lengths
    print("\n" + "-" * 70)
    print("MEMORY BREAKDOWN BY SEQUENCE LENGTH")
    print("-" * 70)

    if train.cp > 1:
        # Extended table with CP overhead column
        print(f"\n{'seq_len':>8} | {'model':>7} | {'activ':>7} | {'disp':>7} | "
              f"{'cp_oh':>7} | {'spike':>7} | {'ovhd':>7} | {'STEADY':>7} | {'PEAK':>7}")
        print("-" * 90)

        for seq_len in seq_lens:
            mem = calculate_memory_breakdown(cfg, train, seq_len)
            print(f"{seq_len:>8} | {mem['model_params_gib']:>6.2f}G | "
                  f"{mem['activation_gib']:>6.2f}G | {mem['dispatcher_peak_gib']:>6.2f}G | "
                  f"{mem['cp_overhead_gib']:>6.2f}G | "
                  f"{mem['combine_spike_gib']:>6.2f}G | "
                  f"{mem['overhead_gib']:>6.2f}G | {mem['steady_state_gib']:>6.2f}G | "
                  f"{mem['total_predicted_gib']:>6.2f}G")
    else:
        # Standard table without CP overhead
        print(f"\n{'seq_len':>8} | {'model':>8} | {'grads':>8} | {'activ':>8} | "
              f"{'disp':>8} | {'spike':>8} | {'overhead':>8} | {'STEADY':>8} | {'PEAK':>8}")
        print("-" * 95)

        for seq_len in seq_lens:
            mem = calculate_memory_breakdown(cfg, train, seq_len)
            print(f"{seq_len:>8} | {mem['model_params_gib']:>7.2f}G | "
                  f"{mem['grad_buffer_gib']:>7.2f}G | "
                  f"{mem['activation_gib']:>7.2f}G | {mem['dispatcher_peak_gib']:>7.2f}G | "
                  f"{mem['combine_spike_gib']:>7.2f}G | "
                  f"{mem['overhead_gib']:>7.2f}G | {mem['steady_state_gib']:>7.2f}G | "
                  f"{mem['total_predicted_gib']:>7.2f}G")

    # Validation against observations
    print("\n" + "-" * 70)
    print("VALIDATION AGAINST OBSERVATIONS")
    print("-" * 70)
    print("\nObserved (from calibration tests):")
    print("  EP=96, TP=16, rank=16 configuration:")
    print("  - 15,360 tokens: PASS (max working)")
    print("  - 16,384 tokens: OOM")
    print("  GPU capacity: 79.33 GiB")
    print("\n  EP=64, TP=8 reference (detailed OOM at 16K):")
    print("  - In use: 76.64 GiB")
    print("  - Spike attempted: 5.05 GiB")

    mem_15k = calculate_memory_breakdown(cfg, train, 15360)
    mem_16k = calculate_memory_breakdown(cfg, train, 16384)

    print(f"\nPredicted (EP={train.ep}, TP={train.tp}):")
    print(f"  15K context: {mem_15k['steady_state_gib']:.2f} GiB steady + {mem_15k['combine_spike_gib']:.2f} GiB spike = {mem_15k['total_predicted_gib']:.2f} GiB peak")
    print(f"  16K context: {mem_16k['steady_state_gib']:.2f} GiB steady + {mem_16k['combine_spike_gib']:.2f} GiB spike = {mem_16k['total_predicted_gib']:.2f} GiB peak")

    print(f"\nValidation:")
    print(f"  15K peak ({mem_15k['total_predicted_gib']:.2f} GiB) < 79.33 GiB → {'PASS' if mem_15k['total_predicted_gib'] < 79.33 else 'FAIL'} (expected: PASS)")
    print(f"  16K peak ({mem_16k['total_predicted_gib']:.2f} GiB) > 79.33 GiB → {'FAIL' if mem_16k['total_predicted_gib'] > 79.33 else 'PASS'} (expected: FAIL)")

    print(f"\nSpike validation (vs EP=64 at 16K: 5.05 GiB observed):")
    obs_spike_16k = 5.05
    print(f"  16K spike: predicted {mem_16k['combine_spike_gib']:.2f} vs observed {obs_spike_16k:.2f} "
          f"(diff: {mem_16k['combine_spike_gib'] - obs_spike_16k:+.2f} GiB)")

    # Detailed breakdown for 15K case
    print("\n" + "-" * 70)
    print("DETAILED BREAKDOWN (15K context)")
    print("-" * 70)
    bd = mem_15k['breakdown']
    print(f"  Model params on GPU ({bd['peak_layers']} layers): {bd['model_params_on_gpu']:>7.2f} GiB")
    print(f"  LoRA params:                    {bd['lora_params']:>8.2f} GiB")
    print(f"  Gradient buffers:               {bd['grad_buffers']:>8.2f} GiB")
    print(f"  Activations (selective recomp): {bd['activations']:>8.2f} GiB")
    print(f"  Dispatcher peak:                {bd['dispatcher_peak']:>8.2f} GiB")
    print(f"  CUDA/NCCL overhead:             {bd['cuda_nccl']:>8.2f} GiB")
    print(f"  Fragmentation (observed):       {bd['fragmentation']:>8.2f} GiB")
    print(f"  Unexplained (calibration):      {bd['unexplained']:>8.2f} GiB")
    print(f"  ─────────────────────────────────────────────")
    print(f"  STEADY-STATE:                   {mem_15k['steady_state_gib']:>8.2f} GiB")
    print(f"  + Combine spike:                {bd['combine_spike']:>8.2f} GiB")
    print(f"  = PEAK:                         {mem_15k['total_predicted_gib']:>8.2f} GiB")

    # Memory scaling analysis
    print("\n" + "-" * 70)
    print("MEMORY SCALING ANALYSIS")
    print("-" * 70)
    mem_1k = calculate_memory_breakdown(cfg, train, 1024)
    mem_8k = calculate_memory_breakdown(cfg, train, 8192)

    # Calculate scaling rates (based on 1K to 16K range)
    steady_scaling_per_1k = (mem_16k['steady_state_gib'] - mem_1k['steady_state_gib']) / 15
    spike_scaling_per_1k = (mem_16k['combine_spike_gib'] - mem_1k['combine_spike_gib']) / 15
    peak_scaling_per_1k = (mem_16k['total_predicted_gib'] - mem_1k['total_predicted_gib']) / 15

    print(f"  Steady-state scaling:           {steady_scaling_per_1k:.2f} GiB per 1K tokens")
    print(f"  Spike scaling:                  {spike_scaling_per_1k:.2f} GiB per 1K tokens")
    print(f"  Peak scaling:                   {peak_scaling_per_1k:.2f} GiB per 1K tokens")

    # Maximum context calculation (must fit PEAK memory in GPU)
    gpu_capacity = 79.33
    base_memory = mem_1k['total_predicted_gib']
    available_for_seq = gpu_capacity - base_memory
    theoretical_max_tokens = int(available_for_seq / peak_scaling_per_1k * 1024)

    print(f"\n  GPU capacity:                   {gpu_capacity:.2f} GiB")
    print(f"  Base memory (1K context):       {base_memory:.2f} GiB")
    print(f"  Available for scaling:          {available_for_seq:.2f} GiB")
    print(f"\n  THEORETICAL MAX CONTEXT:        {theoretical_max_tokens:,} tokens")
    print(f"  (where peak memory = {gpu_capacity:.2f} GiB)")

    # Find safe max where peak < capacity with headroom
    headroom = 2.0  # 2 GiB safety margin
    safe_available = gpu_capacity - base_memory - headroom
    safe_max_tokens = int(safe_available / peak_scaling_per_1k * 1024)
    print(f"\n  SAFE MAX CONTEXT:               {safe_max_tokens:,} tokens")
    print(f"  (with {headroom:.1f} GiB headroom)")

    # === MODEL-BASED MAX CONTEXT ===
    print("\n" + "-" * 70)
    print("CALIBRATED MAX CONTEXT (EP=96, TP=16, rank=16)")
    print("-" * 70)
    print(f"\n  Max working (observed):         15,360 tokens")
    print(f"  Max working (predicted):        {theoretical_max_tokens:,} tokens")
    print(f"  Safe max (with 2 GiB headroom): {safe_max_tokens:,} tokens")


if __name__ == "__main__":
    cfg = K2Config()
    # IMPORTANT: During Megatron training, weights are BF16, not INT4
    # INT4 quantization (compressed-tensors) is only for inference
    cfg.routed_experts_quantized = False  # Training uses BF16

    seq_lens = [1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768]

    # === CP=1 Analysis (baseline) ===
    print("\n" + "=" * 70)
    print("CONFIGURATION 1: CP=1 (baseline)")
    print("=" * 70)
    train_cp1 = TrainingConfig(lora_rank=16, cp=1)  # Using lora_rank=16 as specified
    print_memory_analysis(cfg, train_cp1, seq_lens)

    # === CP=2 Analysis (MoE Parallel Folding) ===
    print("\n\n" + "=" * 70)
    print("CONFIGURATION 2: CP=2 (MoE Parallel Folding)")
    print("=" * 70)
    train_cp2 = TrainingConfig(lora_rank=16, cp=2)  # Using lora_rank=16 as specified
    print_memory_analysis(cfg, train_cp2, seq_lens)

    # === Direct Comparison at 8K ===
    print("\n\n" + "=" * 70)
    print("DIRECT COMPARISON: CP=1 vs CP=2 at 8K context")
    print("=" * 70)
    mem_cp1_8k = calculate_memory_breakdown(cfg, train_cp1, 8192)
    mem_cp2_8k = calculate_memory_breakdown(cfg, train_cp2, 8192)

    print("\nComponent breakdown at 8K tokens:")
    print(f"{'Component':<25} | {'CP=1':>10} | {'CP=2':>10} | {'Diff':>10}")
    print("-" * 60)
    for key in ['model_params_gib', 'activation_gib', 'dispatcher_peak_gib',
                'cp_overhead_gib', 'combine_spike_gib', 'overhead_gib',
                'steady_state_gib', 'total_predicted_gib']:
        v1 = mem_cp1_8k.get(key, 0)
        v2 = mem_cp2_8k.get(key, 0)
        diff = v2 - v1
        sign = "+" if diff > 0 else ""
        print(f"{key:<25} | {v1:>9.2f}G | {v2:>9.2f}G | {sign}{diff:>8.2f}G")

    print("\n" + "-" * 60)
    print("Observations from testing:")
    print("  CP=1 at 8K: Works (steady-state ~65 GiB)")
    print("  CP=2 at 8K: OOM (74.33 GiB in use + 5.25 GiB spike = 79.58 GiB)")
    print("  GPU capacity: 79.33 GiB")
    print(f"\nPrediction accuracy:")
    obs_cp1_8k = 65.0
    obs_cp2_8k = 74.33
    print(f"  CP=1: predicted {mem_cp1_8k['steady_state_gib']:.2f} GiB vs observed {obs_cp1_8k:.2f} GiB "
          f"(diff: {mem_cp1_8k['steady_state_gib'] - obs_cp1_8k:+.2f} GiB)")
    print(f"  CP=2: predicted {mem_cp2_8k['steady_state_gib']:.2f} GiB vs observed {obs_cp2_8k:.2f} GiB "
          f"(diff: {mem_cp2_8k['steady_state_gib'] - obs_cp2_8k:+.2f} GiB)")

