# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MLA value padding patch for THD format
# This enables Flash Attention 2 to work with MLA on sm80 (A100/A800) GPUs
# by padding value tensor to match query head dimension.
#
# Reference: ms-swift/swift/megatron/init.py

import logging

logger = logging.getLogger(__name__)

_mla_forward_patched = False


def apply_mla_forward_patch():
    """Patch MLASelfAttention.forward to support THD format with MLA on sm80 GPUs.

    The issue: Flash Attention 2 requires head_dim_qk == head_dim_v, but MLA has
    head_dim_qk=192 (qk_nope=128 + qk_rope=64) and head_dim_v=128.

    Solution: Pad value tensor to match query dimension before attention,
    then slice off the padding after.
    """
    global _mla_forward_patched
    if _mla_forward_patched:
        return

    try:
        import torch
        import torch.nn.functional as F
        from megatron.core.transformer.multi_latent_attention import (
            MLASelfAttention,
            deprecate_inference_params,
        )
    except ImportError:
        logger.warning("Could not import MLASelfAttention, MLA patch not applied")
        return

    # Save original forward
    original_forward = MLASelfAttention.forward

    def patched_forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        position_ids=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
    ):
        """Forward pass for multi-latent attention with THD format value padding."""
        assert rotary_pos_emb is None, "Rotary position embeddings should not be passed into MLA."
        assert attention_bias is None, "Attention bias should not be passed into MLA."
        assert (
            rotary_pos_cos is None and rotary_pos_sin is None
        ), "MLA does not support Flash Decoding"
        assert not rotary_pos_cos_sin, "Flash-infer rope has not been tested with MLA."
        assert not (
            self.training and getattr(self, 'cache_mla_latents', False)
        ), "cache_mla_latents conflicts with training."

        inference_context = deprecate_inference_params(inference_context, inference_params)

        if hasattr(self.config, 'cache_mla_latents') and self.config.cache_mla_latents:
            self.prepare_for_absorption()

        # Get Q, K, V tensors
        query, key, value = self.get_query_key_value_tensors(
            hidden_states,
            key_value_states,
            position_ids,
            packed_seq_params,
            inference_context=inference_context,
        )

        # Adjust for inference
        result = self._adjust_key_value_for_inference(
            inference_context, query, key, value, rotary_pos_emb=None
        )
        # Handle both 5-tuple and 6-tuple returns (mcore version differences)
        if len(result) == 6:
            query, key, value, _, attn_mask_type, block_table = result
        else:
            query, key, value, _, attn_mask_type = result
            block_table = None

        query = query.contiguous()
        key = key.contiguous()

        if value is not None:
            value = value.contiguous()

        # ========================================
        # MLA value padding for THD format (sm80)
        # ========================================
        thd_format = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        v_dim = value.shape[-1] if value is not None else 0
        q_dim = query.shape[-1]
        value_was_padded = False

        if thd_format and value is not None and q_dim != v_dim:
            # Pad value to match query dimension for FA2 compatibility
            pad_size = q_dim - v_dim
            value = F.pad(value, [0, pad_size])
            # Update attention head dimension
            self.core_attention.hidden_size_per_attention_head_v = value.shape[-1]
            value_was_padded = True
            logger.debug(f"MLA: Padded value from {v_dim} to {q_dim} for THD format")

        # Core attention computation
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query, key, value, attention_mask, packed_seq_params=packed_seq_params
            )
        else:
            if inference_context is None or inference_context.is_static_batching():
                core_attn_out = self.core_attention(
                    query,
                    key,
                    value,
                    attention_mask,
                    packed_seq_params=packed_seq_params,
                    attn_mask_type=attn_mask_type,
                )
            elif getattr(self, 'cache_mla_latents', False):
                # Dynamic batching attention kernel
                cu_query_lengths, max_seqlen_q = inference_context.cu_query_lengths()
                cu_kv_lengths, kv_lengths, max_seqlen_k = inference_context.cu_kv_lengths()
                core_attn_out = self.flash_decode_and_prefill(
                    query, key, value,
                    max_seqlen_q, max_seqlen_k,
                    cu_query_lengths, cu_kv_lengths,
                )
            else:
                core_attn_out = self.core_attention(
                    query, key, value,
                    attention_mask,
                    packed_seq_params=packed_seq_params,
                    attn_mask_type=attn_mask_type,
                )

        # ========================================
        # Remove padding from output
        # ========================================
        if thd_format:
            if core_attn_out.ndim == 2:
                # (t*n, h) -> (t, n, h)
                core_attn_out = core_attn_out.reshape(
                    *core_attn_out.shape[:-1], -1, value.shape[-1] if not value_was_padded else q_dim
                )
            if value_was_padded:
                # Slice off the padding
                core_attn_out = core_attn_out[..., :v_dim]
                # Restore original dimension
                self.core_attention.hidden_size_per_attention_head_v = v_dim
            # Reshape: (t, np, hn) -> (t, b=1, h=np*hn)
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        if hasattr(self, 'recompute_up_proj') and self.recompute_up_proj:
            if hasattr(self, 'qkv_up_checkpoint') and self.qkv_up_checkpoint is not None:
                self.qkv_up_checkpoint.discard_output_and_register_recompute(core_attn_out)
                self.qkv_up_checkpoint = None

        # Output projection
        output, bias = self.linear_proj(core_attn_out)
        return output, bias

    MLASelfAttention.forward = patched_forward
    _mla_forward_patched = True
    logger.info("Applied MLA value padding patch for THD format (sm80 FA2 compatibility)")


def apply_mla_patches():
    """Apply all MLA-related patches."""
    apply_mla_forward_patch()
