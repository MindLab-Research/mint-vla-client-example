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
import torch

logger = logging.getLogger(__name__)


def vocab_parallel_topk(logits: torch.Tensor, k: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute top-K across vocab-parallel sharded logits.

    Args:
        logits: Tensor of shape (batch, seq_len, vocab_shard_size) - sharded across TP ranks
        k: Number of top tokens to return

    Returns:
        Tuple of (top_k_indices, top_k_logits) both of shape (batch, seq_len, k)
        Indices are global token IDs (accounting for vocab sharding)
    """
    from megatron.core import parallel_state as mpu
    import torch.distributed as dist

    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_size = mpu.get_tensor_model_parallel_world_size()
    tp_group = mpu.get_tensor_model_parallel_group()

    batch_size, seq_len, vocab_shard_size = logits.shape

    # Step 1: Find local top-K on this rank
    local_topk_logits, local_topk_indices = torch.topk(logits, k=k, dim=-1)  # (batch, seq, k)

    # Convert local indices to global token IDs
    global_topk_indices = local_topk_indices + tp_rank * vocab_shard_size

    if tp_size == 1:
        return global_topk_indices, local_topk_logits

    # Step 2: Gather all local top-Ks to all ranks
    # Each rank sends its top-K logits and indices
    gathered_logits = [torch.zeros_like(local_topk_logits) for _ in range(tp_size)]
    gathered_indices = [torch.zeros_like(global_topk_indices) for _ in range(tp_size)]

    dist.all_gather(gathered_logits, local_topk_logits, group=tp_group)
    dist.all_gather(gathered_indices, global_topk_indices, group=tp_group)

    # Step 3: Concatenate and find global top-K
    # Shape: (batch, seq, tp_size * k)
    all_logits = torch.cat(gathered_logits, dim=-1)
    all_indices = torch.cat(gathered_indices, dim=-1)

    # Find top-K among all candidates
    final_topk_logits, merge_indices = torch.topk(all_logits, k=k, dim=-1)

    # Gather corresponding global token indices
    final_topk_indices = torch.gather(all_indices, dim=-1, index=merge_indices)

    return final_topk_indices, final_topk_logits


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

        # DIAGNOSTIC: Print input_ids details before forward
        from megatron.core import parallel_state as mpu
        tp_rank = mpu.get_tensor_model_parallel_rank()
        if tp_rank == 0:
            input_flat = input_ids.values().tolist() if hasattr(input_ids, 'values') else []
            print(f"[PRE-FORWARD] input_ids nested tensor: shape={input_ids.shape}, actual_seq_lens={actual_seq_lens}")
            print(f"[PRE-FORWARD] input_ids.values() len={len(input_flat)}")

        def logits_processor(logits, temperature, **label_kwargs):
            """Process logits to compute log_probs."""
            import torch
            # DEBUG: Dump EVERYTHING
            from megatron.core import parallel_state as mpu
            tp_rank = mpu.get_tensor_model_parallel_rank()
            if tp_rank == 0:
                torch.save({
                    "logits": logits.cpu(),
                    "temperature": temperature.cpu() if hasattr(temperature, 'cpu') else temperature,
                    "label_kwargs_keys": list(label_kwargs.keys()),
                    "external_label": label_kwargs.get("external_label").cpu() if label_kwargs.get("external_label") is not None else None,
                    "label": label_kwargs.get("label").cpu() if label_kwargs.get("label") is not None else None,
                }, "/vePFS-Mindverse/share/code/logits_processor_input.pt")

            # Get label from either key
            # NOTE: Can't use `or` here because tensors don't support boolean evaluation
            label = label_kwargs.get("label")
            if label is None:
                label = label_kwargs.get("external_label")
            if label is None:
                raise ValueError(f"No label found in kwargs: {label_kwargs.keys()}")

            assert logits.shape[:2] == label.shape[:2], f"Shape mismatch: logits={logits.shape}, label={label.shape}"

            # DEBUG: Log label AFTER THD preprocessing (this is what model_forward.py produced)
            from megatron.core import parallel_state as mpu
            tp_rank = mpu.get_tensor_model_parallel_rank()
            if tp_rank == 0:
                import json
                label_flat = label[0].tolist() if label.dim() > 1 else label.tolist()
                debug_post = {
                    "logits_shape": str(logits.shape),
                    "label_shape_post_thd": str(label.shape),
                    "label_first_20_post_thd": label_flat[:20],
                    "label_last_10_post_thd": label_flat[-10:],
                    "label_len_post_thd": len(label_flat),
                }
                with open("/vePFS-Mindverse/share/code/model_forward_post_thd.json", "w") as f:
                    json.dump(debug_post, f, indent=2)
                print(f"[DEBUG-POST-THD] label_shape={label.shape}, logits_shape={logits.shape}")

            # DIAGNOSTIC: Print input_ids vs target alignment and logits analysis
            from megatron.core import parallel_state as mpu
            tp_rank = mpu.get_tensor_model_parallel_rank()
            if tp_rank == 0:
                # input_ids is captured from outer scope (NestedTensor in THD format)
                # For THD format with batch=1, values() gives the flattened token sequence
                input_flat = input_ids.values().tolist() if hasattr(input_ids, 'values') else input_ids[0].tolist()
                label_flat = label[0].tolist() if label.dim() > 1 else label.tolist()

                n_print = min(30, len(input_flat), len(label_flat))
                print(f"[INPUT-TARGET-ALIGN] input_ids len={len(input_flat)}, label len={len(label_flat)}, logits shape={logits.shape}")

                # Check actual logits argmax at each position to understand model predictions
                vocab_size = logits.shape[-1]
                print(f"[LOGITS-ARGMAX] Showing argmax predictions at key positions:")
                for pos in [5, 7, 8, 14, 21, 23, 32, 34]:
                    if pos < logits.shape[1]:
                        local_argmax = logits[0, pos, :].argmax().item()
                        local_max = logits[0, pos, :].max().item()
                        target_tok = label_flat[pos] if pos < len(label_flat) else -1
                        # Check if target is in this TP shard
                        target_local_idx = target_tok if target_tok < vocab_size else -1
                        target_logit = logits[0, pos, target_local_idx].item() if target_local_idx >= 0 else float('nan')
                        print(f"  pos={pos}: argmax={local_argmax} (max={local_max:.2f}), target={target_tok} (logit={target_logit:.2f})")

                # Print actual token sequence for debugging
                print(f"[TOKEN-SEQ] First 40 tokens: {input_flat[:40]}")
                print(f"[TOKEN-SEQ] Last 15 tokens: {input_flat[-15:]}")

            # DIAGNOSTIC: Print overall logit range and check for abnormally small values
            from megatron.core import parallel_state as mpu
            tp_rank = mpu.get_tensor_model_parallel_rank()
            overall_min = logits.min().item()
            overall_max = logits.max().item()
            # Print for positions 7 and 23 if they exist
            diag_parts = [f"tp={tp_rank}, overall=[{overall_min:.2f}, {overall_max:.2f}]"]
            for pos in [7, 23]:
                if logits.shape[1] > pos:
                    pos_min = logits[0, pos, :].min().item()
                    pos_max = logits[0, pos, :].max().item()
                    diag_parts.append(f"pos{pos}=[{pos_min:.2f}, {pos_max:.2f}]")
            print(f"[DIAG] {', '.join(diag_parts)}")
            # Alert if range is abnormally small (< 1.0)
            if overall_max - overall_min < 1.0:
                print(f"[DIAG-ALERT] ABNORMAL: logit range {overall_max - overall_min:.4f} < 1.0!")

            # DIAGNOSTIC: Print logits BEFORE temperature scaling for padded positions
            from megatron.core import parallel_state as mpu
            tp_rank = mpu.get_tensor_model_parallel_rank()
            if tp_rank == 0:
                # Check a few positions around the padding boundary (seq_len=51, padded to 56)
                for pos in [50, 51, 52, 53]:
                    if logits.shape[1] > pos:
                        raw_min = logits[0, pos, :].min().item()
                        raw_max = logits[0, pos, :].max().item()
                        temp_val = temperature[0, pos].item() if temperature.dim() > 1 else temperature[pos].item()
                        print(f"[DIAG-RAW] pos={pos}, raw_logits=[{raw_min:.4f}, {raw_max:.4f}], temp={temp_val:.6f}")

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

            # DIAGNOSTIC: Write RAW logits to file for clean capture
            import time
            import os
            from megatron.core import parallel_state as mpu
            tp_rank = mpu.get_tensor_model_parallel_rank()
            ep_rank = mpu.get_expert_model_parallel_rank() if hasattr(mpu, 'get_expert_model_parallel_rank') else 0
            vocab_size = logits_bak.shape[-1]
            call_id = f"{time.time():.6f}"

            # Log multiple positions to see pattern - show ALL tp_rank data
            from megatron.core import parallel_state as mpu
            tp_rank = mpu.get_tensor_model_parallel_rank()
            tp_size = mpu.get_tensor_model_parallel_world_size()
            with open("/vePFS-Mindverse/share/code/raw_logit_diag.log", "a") as f:
                for pos in [7, 8, 23]:
                    if pos < min(logits_bak.shape[1], label.shape[1]):
                        target_tok = label[0, pos].item()
                        local_max = logits_bak[0, pos, :].max().item()
                        local_argmax = logits_bak[0, pos, :].argmax().item()
                        # Convert local argmax to global token id
                        global_argmax = local_argmax + tp_rank * vocab_size
                        # Get target logit if target is in this TP rank's shard
                        shard_start = tp_rank * vocab_size
                        shard_end = shard_start + vocab_size
                        if shard_start <= target_tok < shard_end:
                            target_local_idx = target_tok - shard_start
                            target_logit = logits_bak[0, pos, target_local_idx].item()
                            f.write(f"[LOGIT] tp={tp_rank}, pos={pos}, target={target_tok}, TARGET_LOGIT={target_logit:.2f}, local_max={local_max:.2f}, local_argmax={local_argmax}, global_argmax={global_argmax}\n")
                        else:
                            f.write(f"[LOGIT] tp={tp_rank}, pos={pos}, target={target_tok}, local_max={local_max:.2f}, local_argmax={local_argmax}, global_argmax={global_argmax}\n")

            # Compute log_probs via cross-entropy
            log_probs = vocab_parallel_log_probs_from_logits(logits_bak, label)

            # DIAGNOSTIC: Print logits AFTER cross-entropy to see if softmax was applied in-place
            if tp_rank == 0 and logits_bak.shape[1] > 7 and label.shape[1] > 7:
                target7 = label[0, 7].item()
                if target7 < vocab_size:
                    post_logit7 = logits_bak[0, 7, target7].item()
                    post_min7 = logits_bak[0, 7, :].min().item()
                    post_max7 = logits_bak[0, 7, :].max().item()
                    lp7 = log_probs[0, 7].item()
                    print(f"[RAW-LOGIT-POST] id={call_id}, ep={ep_rank}, pos=7, logit={post_logit7:.4f}, range=[{post_min7:.2f},{post_max7:.2f}], lp={lp7:.4f}")

            # Also catch any lp < -10
            if tp_rank == 0:
                for pos in range(min(log_probs.shape[1], label.shape[1], 60)):
                    lp = log_probs[0, pos].item()
                    if lp < -10:
                        target_token = label[0, pos].item()
                        if target_token < vocab_size:
                            raw_logit = logits_bak[0, pos, target_token].item()
                            raw_max = logits_bak[0, pos, :].max().item()
                            print(f"[BAD-LP] pos={pos}, target={target_token}, logit={raw_logit:.2f}, max={raw_max:.2f}, lp={lp:.4f}")

            # DIAGNOSTIC: Print RAW LOGIT values for target tokens at BAD positions
            from megatron.core import parallel_state as mpu
            tp_rank = mpu.get_tensor_model_parallel_rank()
            tp_size = mpu.get_tensor_model_parallel_world_size()
            vocab_size = logits_bak.shape[-1]  # This is the SHARD size

            # Find positions with extremely negative log_probs (the -2.2B issue)
            for pos in range(min(log_probs.shape[1], 60)):
                lp_val = log_probs[0, pos].item()
                if lp_val < -1e6:  # Catastrophic logprob
                    target_token = label[0, pos].item() if pos < label.shape[1] else -1
                    shard_start = tp_rank * vocab_size
                    shard_end = shard_start + vocab_size

                    if shard_start <= target_token < shard_end:
                        local_idx = target_token - shard_start
                        target_logit = logits_bak[0, pos, local_idx].item()
                        logit_min = logits_bak[0, pos, :].min().item()
                        logit_max = logits_bak[0, pos, :].max().item()
                        print(f"[DIAG-BAD] pos={pos}, target={target_token}, logit={target_logit:.6f}, range=[{logit_min:.2f}, {logit_max:.2f}], lp={lp_val:.2e}")

            # CRITICAL FIX: Mask padded positions FIRST
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

            # DIAGNOSTIC: Print range BEFORE masking
            from megatron.core import parallel_state as mpu
            tp_rank = mpu.get_tensor_model_parallel_rank()
            if tp_rank == 0 and n_padded > 0:
                pre_mask_min = log_probs.min().item()
                pre_mask_max = log_probs.max().item()
                print(f"[DIAG-MASK] BEFORE: n_valid={n_valid}, n_padded={n_padded}, range=[{pre_mask_min:.2e}, {pre_mask_max:.4f}]")

            if n_padded > 0:
                # Zero out padded positions (they have garbage log_probs)
                log_probs[0, ~valid_mask] = 0.0

                # DIAGNOSTIC: Print range AFTER masking
                if tp_rank == 0:
                    post_mask_min = log_probs[0, valid_mask].min().item() if valid_mask.sum() > 0 else 0.0
                    post_mask_max = log_probs[0, valid_mask].max().item() if valid_mask.sum() > 0 else 0.0
                    print(f"[DIAG-MASK] AFTER: valid_range=[{post_mask_min:.4f}, {post_mask_max:.4f}]")

            # DIAGNOSTIC: Check for catastrophic logprobs at valid positions
            valid_log_probs = log_probs[0, valid_mask]
            min_valid_lp = valid_log_probs.min().item() if valid_log_probs.numel() > 0 else 0.0

            if min_valid_lp < -1e9:
                print(f"[DIAG-PADDING] PADDING MASK FAILED! min_valid_lp={min_valid_lp:.2e}, n_valid={n_valid}, n_padded={n_padded}")

            ret["log_probs"] = log_probs

            # Top-k tracking disabled - was causing symbolic shape comparison crash
            # Re-enable by uncommenting vocab_parallel_topk call if needed for KL debugging

            return ret

        # PATCH: Use dynamic key for label
        logits_processor_args = {label_key: label, "temperature": temperature}

        # DEBUG: Log actual inputs before model forward
        from megatron.core import parallel_state as mpu
        tp_rank = mpu.get_tensor_model_parallel_rank()
        if tp_rank == 0:
            import json
            # Extract values from NestedTensor
            input_vals = input_ids.values().tolist() if hasattr(input_ids, 'values') else input_ids[0].tolist()
            label_vals = label.values().tolist() if hasattr(label, 'values') else label[0].tolist()
            debug_data = {
                "input_ids_shape": str(input_ids.shape),
                "label_shape": str(label.shape),
                "input_ids_first_20": input_vals[:20],
                "input_ids_last_10": input_vals[-10:],
                "label_first_20": label_vals[:20],
                "label_last_10": label_vals[-10:],
                "label_key": label_key,
                "input_len": len(input_vals),
                "label_len": len(label_vals),
            }
            with open("/vePFS-Mindverse/share/code/model_forward_input.json", "w") as f:
                json.dump(debug_data, f, indent=2)
            print(f"[DEBUG-FORWARD] Wrote input debug to model_forward_input.json")

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


def _enable_megatron_determinism(seed: int = 42):
    """Enable full determinism for Megatron/TransformerEngine.

    This fixes non-deterministic forward passes that cause train-inference logprob mismatch.
    Without this, consecutive forward passes with identical inputs can differ by ~0.46 nats.

    Must be called BEFORE any Megatron/TE code is initialized.
    """
    import os
    import random
    import numpy as np
    import torch
    import socket

    # Set environment variables for deterministic execution
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # Required for CUDA determinism
    os.environ["NCCL_DETERMINISTIC"] = "1"
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"  # Critical for FlashAttention

    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Enable PyTorch deterministic algorithms
    # warn_only=True because some ops (like cumsum) don't have deterministic implementations
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Disable cuDNN benchmark for determinism (benchmark mode tests multiple algorithms)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Verify the settings were applied
    is_deterministic = torch.are_deterministic_algorithms_enabled()
    hostname = socket.gethostname()
    pid = os.getpid()

    # Write status to a file for verification
    status_file = f"/tmp/determinism_status_{hostname}_{pid}.txt"
    with open(status_file, "w") as f:
        f.write(f"hostname={hostname}\n")
        f.write(f"pid={pid}\n")
        f.write(f"are_deterministic_algorithms_enabled={is_deterministic}\n")
        f.write(f"cudnn.deterministic={torch.backends.cudnn.deterministic}\n")
        f.write(f"cudnn.benchmark={torch.backends.cudnn.benchmark}\n")
        f.write(f"FLASH_ATTENTION_DETERMINISTIC={os.environ.get('FLASH_ATTENTION_DETERMINISTIC')}\n")
        f.write(f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}\n")
        f.write(f"NCCL_DETERMINISTIC={os.environ.get('NCCL_DETERMINISTIC')}\n")

    print(f"[VERL_PATCH] Enabled full determinism mode (seed={seed}) on {hostname}")
    print(f"[VERL_PATCH]   torch.are_deterministic_algorithms_enabled() = {is_deterministic}")
    print(f"[VERL_PATCH]   torch.backends.cudnn.deterministic = {torch.backends.cudnn.deterministic}")
    print(f"[VERL_PATCH]   FLASH_ATTENTION_DETERMINISTIC = {os.environ.get('FLASH_ATTENTION_DETERMINISTIC')}")
    print(f"[VERL_PATCH]   Status written to {status_file}")
    logger.info(f"Enabled full determinism mode (FLASH_ATTENTION_DETERMINISTIC=1, cudnn.deterministic=True)")


def apply_verl_patches():
    """Apply monkey-patches to verl for MLA attention backend fix and external labels support."""
    # Note: _enable_megatron_determinism() is now called separately BEFORE this,
    # in megatron_distributed.py, to ensure it runs before any Megatron imports.

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

            # PATCH: MLA models require special handling: force flash attention + value padding.
            # Non-MLA models should keep Megatron's default attention backend selection.
            from megatron.core.transformer.enums import AttnBackend

            is_mla = _is_mla_model(self.model_config.hf_config)
            if is_mla:
                provider.attention_backend = AttnBackend.flash
                print(f"[VERL_PATCH] MLA model detected, using AttnBackend.flash (with value padding)")
                print(f"[VERL_PATCH] model_type: {getattr(self.model_config.hf_config, 'model_type', 'unknown')}")
                print(f"[VERL_PATCH] qk_nope_head_dim: {getattr(self.model_config.hf_config, 'qk_nope_head_dim', 'N/A')}")
                print(f"[VERL_PATCH] qk_rope_head_dim: {getattr(self.model_config.hf_config, 'qk_rope_head_dim', 'N/A')}")
                print(f"[VERL_PATCH] v_head_dim: {getattr(self.model_config.hf_config, 'v_head_dim', 'N/A')}")

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

    # Apply prepare_model_outputs patch to pass through all keys (including topk)
    _apply_prepare_model_outputs_patch()


def _apply_prepare_model_outputs_patch():
    """Patch MegatronEngineWithLMHead.prepare_model_outputs to pass through all keys.

    Original only extracts 'log_probs' and 'entropy', ignoring other keys like
    'topk_indices' and 'topk_logits' computed by logits_processor.
    """
    try:
        from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead
    except ImportError:
        logger.warning("MegatronEngineWithLMHead not found, skipping prepare_model_outputs patch")
        return

    original_prepare_model_outputs = MegatronEngineWithLMHead.prepare_model_outputs

    def patched_prepare_model_outputs(self, output: dict, data):
        """Patched to pass through all keys from logits_processor output."""
        # DEBUG: Verify this patched version is being called - write to PFS shared log
        import time
        with open("/vePFS-Mindverse/share/code/raw_logit_diag.log", "a") as f:
            f.write(f"[PATCHED_PREPARE_MODEL_OUTPUTS] Called with output keys: {list(output.keys()) if isinstance(output, dict) else type(output)}\n")

        # Start with log_probs (required)
        log_prob = output.get("log_probs")
        if log_prob is None:
            # Fall back to original method
            with open("/vePFS-Mindverse/share/code/raw_logit_diag.log", "a") as f:
                f.write("[PATCHED_PREPARE_MODEL_OUTPUTS] log_probs is None, falling back to original\n")
            return original_prepare_model_outputs(self, output, data)

        model_output = {"log_probs": log_prob}

        # Pass through ALL other keys (including topk_indices, topk_logits, entropy, etc.)
        for key, value in output.items():
            if key != "log_probs":  # Already added
                model_output[key] = value
                if key in ("topk_indices", "topk_logits"):
                    with open("/vePFS-Mindverse/share/code/raw_logit_diag.log", "a") as f:
                        f.write(f"[PATCHED_PREPARE_MODEL_OUTPUTS] Added {key}: shape={value.shape if hasattr(value, 'shape') else 'N/A'}\n")

        with open("/vePFS-Mindverse/share/code/raw_logit_diag.log", "a") as f:
            f.write(f"[PATCHED_PREPARE_MODEL_OUTPUTS] Returning model_output with keys: {list(model_output.keys())}\n")
        return model_output

    MegatronEngineWithLMHead.prepare_model_outputs = patched_prepare_model_outputs
    print("[VERL_PATCH] Applied prepare_model_outputs patch (passes through all keys including topk)")
    logger.info("Applied prepare_model_outputs patch (passes through all keys)")


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
