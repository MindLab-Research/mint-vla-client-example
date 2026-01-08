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


def _apply_external_label_patch():
    """Patch MegatronEngineWithLMHead.forward_step to use external labels when provided.

    Problem: verl's forward_step creates labels from input_ids.clone(), then rolls them:
        label = input_ids.clone()  # [t0, t1, ..., t_{N-1}]
        # After roll in model_forward.py: [t1, t2, ..., t_{N-1}, t0]
        # Position N-1 gets label t0 (WRONG - should be t_N)

    This causes wrong logprob at the last position when using standard SFT format:
        input = full_sequence[:-1]   # [t0, ..., t_{N-1}]
        target = full_sequence[1:]   # [t1, ..., t_N]

    The key insight: external labels ALREADY contain the correct target t_N at position N-1.
    We must NOT roll them - they're already correctly aligned.

    Solution: When external labels are provided, use key "external_label" instead of "label".
    model_forward.py only rolls when key == "label", so external labels won't be rolled.
    Update logits_processor to accept the label via **kwargs to handle both keys.
    """
    import torch

    try:
        from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead
    except ImportError:
        logger.warning("MegatronEngineWithLMHead not found, skipping external label patch")
        return

    original_forward_step = MegatronEngineWithLMHead.forward_step

    def patched_forward_step(self, batch_iter, model, postprocess_micro_batch_func):
        """Patched forward_step that uses external labels when provided."""
        from functools import partial
        from tensordict import TensorDict
        from verl.utils.megatron_utils import get_device_id
        from verl.workers.engine.megatron.transformer_impl import tu, DatasetPadMode, extract_multi_modal_inputs
        import verl.utils.torch_functional as verl_F
        from verl.utils.megatron.tensor_parallel import vocab_parallel_entropy
        from verl.utils.megatron.tensor_parallel import vocab_parallel_log_probs_from_logits

        batch: TensorDict = next(batch_iter)
        batch = batch.to(get_device_id())

        use_fused_kernels = tu.get_non_tensor_data(batch, key="use_fused_kernels", default=False)
        calculate_entropy = tu.get_non_tensor_data(batch, key="calculate_entropy", default=False)
        pad_mode = tu.get_non_tensor_data(batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        temperature = batch["temperature"]
        model_inputs = self.prepare_model_inputs(batch)
        input_ids = model_inputs["input_ids"]
        multi_modal_inputs = model_inputs["multi_modal_inputs"]

        if not isinstance(temperature, torch.Tensor):
            temperature = torch.tensor([temperature] * input_ids.shape[0], device=input_ids.device)

        assert temperature.shape[0] == input_ids.shape[0]
        temperature = verl_F.expand_as_nested(temperature, input_ids)

        # PATCH: Check for external labels (key "target")
        # External labels are already correctly shifted: target[i] = full_sequence[i+1]
        # Critically: target[N-1] = t_N (the LAST token, not in input_ids!)
        external_label = batch.get("target", None)
        use_external_label = tu.get_non_tensor_data(batch, key="use_external_label", default=False)

        if external_label is not None and use_external_label:
            # Use external labels - already correctly shifted, must NOT be rolled
            label = external_label
            label_key = "external_label"  # Key != "label" so model_forward.py won't roll
        else:
            # Original behavior: clone input_ids, will be rolled by model_forward.py
            if pad_mode == DatasetPadMode.NO_PADDING:
                label = input_ids.clone()
            else:
                raise NotImplementedError(f"Pad mode {pad_mode} is not supported for megatron engine")
            label_key = "label"  # Will trigger roll in model_forward.py

        from verl.models.mcore import get_mcore_forward_no_padding_fn

        if use_fused_kernels:
            raise NotImplementedError("Fused kernels are not supported for megatron engine")

        forward_fn = get_mcore_forward_no_padding_fn(self.model_config.hf_config)

        # Capture actual sequence lengths from input_ids (nested tensor) for padding mask
        # input_ids.offsets() gives cumulative lengths, diff() gives per-sequence lengths
        actual_seq_lens = input_ids.offsets().diff().tolist()

        def logits_processor(logits, temperature, **label_kwargs):
            """Process logits to compute log_probs.

            Accepts label via **kwargs to handle both "label" and "external_label" keys.
            model_forward.py passes logits_processor_args as **kwargs, so the key name
            determines the parameter name.

            CRITICAL FIX: Mask padded positions to avoid garbage logits corrupting log_probs.
            Padded positions have uninitialized/garbage logits that produce -2.2B log_probs.
            """
            # Get label from either key
            label = label_kwargs.get("label") or label_kwargs.get("external_label")
            if label is None:
                raise ValueError(f"No label found in kwargs: {label_kwargs.keys()}")

            assert logits.shape[:2] == label.shape[:2], f"Shape mismatch: logits={logits.shape}, label={label.shape}"
            temperature[temperature <= 0] = 1e-8
            assert torch.all(temperature > 0).item(), f"temperature must be positive. Got {temperature}"
            logits.div_(temperature.unsqueeze(dim=-1))
            ret = {}
            if calculate_entropy:
                logits_bak = logits.clone()
                entropy = vocab_parallel_entropy(logits)
                ret["entropy"] = entropy
            else:
                logits_bak = logits

            # Compute log_probs via cross-entropy
            log_probs = vocab_parallel_log_probs_from_logits(logits_bak, label)

            # CRITICAL FIX: Mask padded positions
            # Padded positions have garbage logits (uninitialized GPU memory) that produce
            # catastrophic log_probs like -2.2 billion. Set them to 0.0 (neutral value).
            #
            # The data flow is:
            # 1. preprocess_thd_no_padding pads sequence to align_size (TP * CP * 2)
            # 2. Model forward produces logits for ALL positions (valid + padding)
            # 3. Logits at padding positions are garbage (not computed by model)
            # 4. Cross-entropy on garbage produces garbage log_probs
            # 5. postprocess_thd_no_padding extracts only valid positions AFTER damage is done
            #
            # Fix: Zero out log_probs at padded positions before they propagate further.

            from megatron.core import parallel_state as mpu
            tp_size = mpu.get_tensor_model_parallel_world_size()
            cp_size = mpu.get_context_parallel_world_size()
            align_size = tp_size * cp_size * 2 if cp_size > 1 else tp_size

            # Build valid position mask based on actual vs padded lengths
            # log_probs shape: [1, padded_total_len] in THD format
            padded_total_len = log_probs.shape[1]
            valid_mask = torch.zeros(padded_total_len, dtype=torch.bool, device=log_probs.device)

            # Calculate cumulative padded offsets
            cu_padded = 0
            for i, actual_len in enumerate(actual_seq_lens):
                pad_size = (align_size - actual_len % align_size) % align_size
                padded_len = actual_len + pad_size
                # Mark valid positions (actual_len positions starting at cu_padded)
                valid_mask[cu_padded : cu_padded + actual_len] = True
                cu_padded += padded_len // cp_size  # Account for CP splitting

            # Count how many positions we're masking
            n_valid = valid_mask.sum().item()
            n_padded = padded_total_len - n_valid
            if n_padded > 0:
                # Zero out padded positions (they have garbage log_probs)
                log_probs[0, ~valid_mask] = 0.0

            ret["log_probs"] = log_probs
            return ret

        # PATCH: Use dynamic key for label
        logits_processor_args = {label_key: label, "temperature": temperature}

        output = forward_fn(
            model,
            input_ids,
            multi_modal_inputs,
            logits_processor=logits_processor,
            logits_processor_args=logits_processor_args,
            vision_model=hasattr(self.model_config.hf_config, "vision_config"),
            pad_token_id=self.model_config.tokenizer.pad_token_id,
            data_format="thd" if self.engine_config.use_remove_padding else "bshd",
        )

        return output, partial(postprocess_micro_batch_func, data=batch)

    MegatronEngineWithLMHead.forward_step = patched_forward_step
    print("[VERL_PATCH] Applied external label patch for MegatronEngineWithLMHead.forward_step")
    logger.info("Applied external label patch (fixes last-token logprob issue)")


def apply_verl_patches():
    """Apply monkey-patches to verl for MLA attention backend fix and external labels support."""
    try:
        from verl.workers.engine.megatron.transformer_impl import MegatronEngine
    except ImportError:
        logger.warning("verl.workers.engine.megatron.transformer_impl not found, skipping patches")
        return

    # Apply external label patch first (fixes last-token logprob issue)
    _apply_external_label_patch()

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
