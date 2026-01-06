"""Monkey-patches for verl to fix MLA attention backend.

Problem: TransformerEngine attention backends have strict requirements:
- FlashAttention 2: requires head_dim_qk == head_dim_v
- FlashAttention 3: requires sm90+ (cluster has sm80 A800s)
- FusedAttention: for MLA requires qkv_layout_group = "hd_hd_hd" (verl uses thd)
- UnfusedAttention: disabled for thd format

MLA models (Moonlight-16B-A3B, DeepSeekV3, Kimi-K2) have:
- head_dim_qk = 192 (qk_nope=128 + qk_rope=64)
- head_dim_v = 128

Solution: Pad value tensor from 128→192 before attention, enabling FlashAttention 2.
Then unpad output from 192→128 after attention.
"""

import logging

logger = logging.getLogger(__name__)


def _is_mla_model(hf_config) -> bool:
    """Check if model uses Multi-Latent Attention (MLA).

    MLA models have different head dimensions for Q/K vs V.
    DeepSeekV3/Moonlight uses:
    - qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    - v_head_dim (different from qk_head_dim)
    """
    # Check for DeepSeekV3-style MLA config
    if hasattr(hf_config, 'qk_nope_head_dim') and hasattr(hf_config, 'v_head_dim'):
        qk_head_dim = getattr(hf_config, 'qk_nope_head_dim', 0) + getattr(hf_config, 'qk_rope_head_dim', 0)
        v_head_dim = getattr(hf_config, 'v_head_dim', 0)
        if qk_head_dim > 0 and v_head_dim > 0 and qk_head_dim != v_head_dim:
            return True

    # Check model_type
    model_type = getattr(hf_config, 'model_type', '')
    if model_type in ('deepseek_v3', 'deepseek_v2'):
        return True

    # Check architectures
    architectures = getattr(hf_config, 'architectures', [])
    if any('DeepseekV3' in arch or 'DeepseekV2' in arch for arch in architectures):
        return True

    return False


def apply_verl_patches():
    """Apply monkey-patches to verl for MLA attention backend fix."""
    try:
        from verl.workers.engine.megatron.transformer_impl import MegatronEngine
    except ImportError:
        logger.warning("verl.workers.engine.megatron.transformer_impl not found, skipping patches")
        return

    # Store original method
    original_build_tf_config = MegatronEngine._build_tf_config

    def patched_build_tf_config(self):
        """Patched _build_tf_config that uses fused attention for MLA models."""
        from verl.utils.megatron_utils import mapping_string_to_attn_backend
        from verl.utils.torch_dtypes import PrecisionType

        self.param_dtype = PrecisionType.to_dtype(self.engine_config.dtype)
        self.dtype = PrecisionType.to_dtype(self.param_dtype)

        override_transformer_config = mapping_string_to_attn_backend({**self.engine_config.override_transformer_config})

        self.provider = None
        self.vanilla_bridge = self.engine_config.vanilla_mbridge
        if self.vanilla_bridge:
            from verl.models.mcore.mbridge import AutoBridge

            bridge = AutoBridge.from_config(self.model_config.hf_config, dtype=self.param_dtype)
            bridge.set_extra_args(**override_transformer_config)
            tf_config = bridge.config
            tf_config.fp16 = self.param_dtype == torch.float16
            tf_config.bf16 = self.param_dtype == torch.bfloat16
        else:
            from verl.models.mcore.bridge import AutoBridge

            # Use Megatron-Bridge to convert HF config to Megatron config
            bridge = AutoBridge.from_hf_pretrained(
                self.model_config.local_path, trust_remote_code=self.model_config.trust_remote_code
            )
            # Get Megatron provider and configure it
            provider = bridge.to_megatron_provider(load_weights=False)

            # In case of invalid overrides, we need to make sure some critical params are set correctly
            provider.params_dtype = self.param_dtype

            # Pass distributed info
            provider.tensor_model_parallel_size = self.engine_config.tensor_model_parallel_size
            provider.pipeline_model_parallel_size = self.engine_config.pipeline_model_parallel_size
            provider.expert_model_parallel_size = self.engine_config.expert_model_parallel_size
            provider.expert_tensor_parallel_size = self.engine_config.expert_tensor_parallel_size
            provider.virtual_pipeline_model_parallel_size = self.engine_config.virtual_pipeline_model_parallel_size
            provider.context_parallel_size = self.engine_config.context_parallel_size
            provider.sequence_parallel = self.engine_config.sequence_parallel

            # PATCH: Use flash attention for MLA models (value padding makes it compatible)
            # Previously tried fused, but TE disables it for thd format with MLA
            # Value padding (128→192) makes head_dim_qk == head_dim_v, enabling FA2
            from megatron.core.transformer.enums import AttnBackend

            is_mla = _is_mla_model(self.model_config.hf_config)
            if is_mla:
                provider.attention_backend = AttnBackend.flash
                print(f"[VERL_PATCH] MLA model detected, using AttnBackend.flash (with value padding)")
                print(f"[VERL_PATCH] model_type: {getattr(self.model_config.hf_config, 'model_type', 'unknown')}")
                print(f"[VERL_PATCH] qk_nope_head_dim: {getattr(self.model_config.hf_config, 'qk_nope_head_dim', 'N/A')}")
                print(f"[VERL_PATCH] qk_rope_head_dim: {getattr(self.model_config.hf_config, 'qk_rope_head_dim', 'N/A')}")
                print(f"[VERL_PATCH] v_head_dim: {getattr(self.model_config.hf_config, 'v_head_dim', 'N/A')}")
            else:
                provider.attention_backend = AttnBackend.flash
                print(f"[VERL_PATCH] Non-MLA model, using AttnBackend.flash")

            provider.variable_seq_lengths = True
            provider.moe_token_dispatcher_type = "alltoall"
            provider.moe_router_load_balancing_type = "none"

            # Apply transformer config overrides
            for key, value in override_transformer_config.items():
                setattr(provider, key, value)

            provider.finalize()
            self.provider = provider
            tf_config = None  # Will be set after model creation
        self.bridge = bridge

        from verl.models.mcore import get_mcore_weight_converter
        if not self.bridge:
            self.weight_converter = get_mcore_weight_converter(self.model_config.hf_config, self.dtype)

        import torch.distributed
        if torch.distributed.get_rank() == 0:
            if tf_config is not None:
                print(f"TF config: {tf_config}")
        self.tf_config = tf_config

        from verl.workers.config.megatron_peft import get_peft_cls

        self.peft_cls = get_peft_cls(
            model_config=self.model_config, bridge=self.bridge, provider=self.provider, dtype=self.param_dtype
        )

    # Need torch import for the patched method
    import torch
    patched_build_tf_config.__globals__['torch'] = torch

    # Apply the patch
    MegatronEngine._build_tf_config = patched_build_tf_config
    print("[VERL_PATCH] Applied verl MLA attention backend patch")
    logger.info("Applied verl MLA attention backend patch")

    # Apply MLA value padding patch (enables FlashAttention 2 by making head_dim_qk == head_dim_v)
    _apply_mla_value_padding_patch()


def _apply_mla_value_padding_patch():
    """Patch MultiLatentAttention to pad value tensor for FlashAttention 2 compatibility.

    FlashAttention 2 requires head_dim_qk == head_dim_v.
    MLA has head_dim_qk=192, head_dim_v=128.

    This patch:
    1. Pads value tensor from 128→192 (with zeros) before core_attention
    2. Slices output from 192→128 after core_attention

    The padding is transparent to the rest of the model since:
    - Attention computes softmax(Q @ K^T / sqrt(d)) @ V
    - Zero-padded dimensions in V produce zero contributions to output
    - We slice those dimensions off before the output projection
    """
    import torch

    try:
        from megatron.core.transformer.multi_latent_attention import MultiLatentAttention
    except ImportError:
        logger.warning("megatron.core.transformer.multi_latent_attention not found, skipping MLA value padding patch")
        return

    # Store original forward method
    original_forward = MultiLatentAttention.forward

    def patched_forward(self, *args, **kwargs):
        """Patched forward with value padding for FlashAttention 2."""
        # Check if MLA with mismatched head dimensions
        q_head_dim = getattr(self, 'q_head_dim', None)
        v_head_dim = getattr(self.config, 'v_head_dim', None)

        if q_head_dim is None or v_head_dim is None or q_head_dim == v_head_dim:
            # Not MLA or dimensions already match, use original
            return original_forward(self, *args, **kwargs)

        # MLA with head_dim_qk != head_dim_v - need to patch
        pad_size = q_head_dim - v_head_dim  # 192 - 128 = 64

        # Get the core_attention module and save its original forward
        core_attn_module = self.core_attention
        original_core_attn_forward = core_attn_module.forward

        # Also save and temporarily modify TE's expected head_dim_v
        # TransformerEngine's DotProductAttention asserts value.shape[-1] == hidden_size_per_attention_head_v
        original_v_head_dim = getattr(core_attn_module, 'hidden_size_per_attention_head_v', None)

        def padded_core_attention_forward(query, key, value, attention_mask, **kw):
            """Wrap core_attention.forward to pad value and unpad output."""
            # Pad value from [*, head_dim_v] to [*, head_dim_qk]
            # value shape: thd format = [t, n, head_dim_v] or [s, b, n, head_dim_v]
            num_heads = None
            if value is not None:
                num_heads = value.shape[-2]  # Get num_heads before padding
                value_padded = torch.nn.functional.pad(value, (0, pad_size), mode='constant', value=0)
            else:
                value_padded = value

            # Call original forward directly (not the Module, to avoid recursion)
            output = original_core_attn_forward(query, key, value_padded, attention_mask, **kw)

            # Unpad output from [*, head_dim_qk] to [*, head_dim_v]
            # Core attention may flatten heads into last dimension:
            #   input value: [seq, num_heads, head_dim_v] -> [seq, num_heads, head_dim_qk]
            #   output: [seq, num_heads * head_dim_qk] (flattened)
            # We need to reshape, slice, and reshape back
            if output is not None:
                if output.shape[-1] == q_head_dim:
                    # Output preserves head dimension: [*, num_heads, head_dim_qk]
                    output = output[..., :v_head_dim]
                elif num_heads is not None and output.shape[-1] == num_heads * q_head_dim:
                    # Output has flattened heads: [seq, num_heads * head_dim_qk]
                    # Reshape to [seq, num_heads, head_dim_qk], slice, then reshape back
                    seq_len = output.shape[0]
                    output = output.view(seq_len, num_heads, q_head_dim)
                    output = output[..., :v_head_dim].contiguous()
                    output = output.view(seq_len, num_heads * v_head_dim)

            return output

        # Temporarily replace core_attention.forward method and expected head_dim
        core_attn_module.forward = padded_core_attention_forward
        if original_v_head_dim is not None:
            core_attn_module.hidden_size_per_attention_head_v = q_head_dim  # Change 128 -> 192
        try:
            result = original_forward(self, *args, **kwargs)
        finally:
            # Restore original forward method and head_dim
            core_attn_module.forward = original_core_attn_forward
            if original_v_head_dim is not None:
                core_attn_module.hidden_size_per_attention_head_v = original_v_head_dim

        return result

    # Apply the patch
    MultiLatentAttention.forward = patched_forward
    print("[VERL_PATCH] Applied MLA value padding patch for FlashAttention 2 compatibility")
    logger.info("Applied MLA value padding patch (head_dim_v 128→192 padding)")
