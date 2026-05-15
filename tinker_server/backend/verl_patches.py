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
from functools import wraps

import torch

logger = logging.getLogger(__name__)


def _apply_megatron_router_expert_bias_no_stack_patch() -> None:
    """Patch Megatron router expert-bias update to avoid stacking (reduce peak CUDA allocations).

    Megatron's `_update_router_expert_bias` stacks per-layer `local_tokens_per_expert` and
    `expert_bias` into new tensors, then calls `get_updated_expert_bias(...)` which performs a
    CUDA all-reduce. For very tight memory regimes (e.g. K2 at 32k context with all-layer LoRA),
    the extra temporary allocations can trigger OOM.

    This patch updates expert_bias per module in-place:
      - all-reduce module.local_tokens_per_expert directly (no stack allocations)
      - apply the same sign(offset) update rule to module.expert_bias
    """
    try:
        import importlib

        # NOTE: `import megatron.core.distributed.finalize_model_grads` can resolve to the
        # exported *function* `finalize_model_grads` from `megatron.core.distributed.__init__`,
        # not the module `.../finalize_model_grads.py`. Use importlib to force module import.
        fmg = importlib.import_module("megatron.core.distributed.finalize_model_grads")
    except Exception as e:
        logger.warning(
            "Could not import megatron.core.distributed.finalize_model_grads; skipping expert_bias no-stack patch: %s",
            e,
        )
        return

    if getattr(fmg, "_tinker_router_expert_bias_no_stack_patched", False):
        return

    if not hasattr(fmg, "_update_router_expert_bias"):
        logger.warning(
            "Megatron finalize_model_grads has no _update_router_expert_bias; skipping expert_bias no-stack patch"
        )
        return

    original = fmg._update_router_expert_bias

    def patched(model, config):
        if not getattr(config, "moe_router_enable_expert_bias", False):
            return original(model, config)

        # K2 at full-scale (TP=64, EP=64, CP=2) runs in an extremely tight VRAM regime
        # at 32k context with all-layer LoRA. Even small NCCL collectives inside
        # `_update_router_expert_bias` have been observed to OOM due to NCCL internal
        # allocations. Expert-bias updates are an auxiliary load-balancing tweak;
        # disable them for this specific K2 parallelism profile to avoid hard failure.
        if (
            getattr(config, "tensor_model_parallel_size", None) == 64
            and getattr(config, "expert_model_parallel_size", None) == 64
            and getattr(config, "context_parallel_size", None) == 2
            and getattr(config, "expert_tensor_parallel_size", 1) == 1
        ):
            if not getattr(patched, "_tinker_k2_expert_bias_disabled_logged", False):
                print(
                    "[VERL_PATCH] Disabled Megatron _update_router_expert_bias for K2 (TP=64 EP=64 CP=2) to avoid NCCL OOM"
                )
                patched._tinker_k2_expert_bias_disabled_logged = True  # type: ignore[attr-defined]

            import torch

            with torch.no_grad():
                for model_chunk in model:
                    for module in fmg.get_attr_wrapped_model(model_chunk, "modules")():
                        if hasattr(module, "expert_bias") and hasattr(module, "local_tokens_per_expert"):
                            module.local_tokens_per_expert.zero_()
            return

        tokens_per_expert = []
        expert_bias = []
        for model_chunk in model:
            for module in fmg.get_attr_wrapped_model(model_chunk, "modules")():
                if hasattr(module, "expert_bias"):
                    tokens_per_expert.append(module.local_tokens_per_expert)
                    expert_bias.append(module.expert_bias)

        if len(expert_bias) == 0:
            return

        import torch
        import torch.distributed as dist
        from megatron.core import parallel_state

        group = parallel_state.get_tensor_and_data_parallel_group(with_context_parallel=True)
        rate = getattr(config, "moe_router_bias_update_rate")

        with torch.no_grad():
            for t, b in zip(tokens_per_expert, expert_bias):
                dist.all_reduce(t, group=group)
                avg = t.sum(dim=-1, keepdim=True) / t.shape[-1]
                b.add_(torch.sign(avg - t) * rate)
                t.zero_()

    fmg._update_router_expert_bias = patched  # type: ignore[attr-defined]
    fmg._tinker_router_expert_bias_no_stack_patched = True  # type: ignore[attr-defined]
    print("[VERL_PATCH] Patched Megatron _update_router_expert_bias (no stacking, in-place update)")


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
        from verl.workers.engine.megatron.transformer_impl import tu, DatasetPadMode
        import verl.utils.torch_functional as verl_F
        from verl.utils.megatron.tensor_parallel import vocab_parallel_entropy
        from verl.utils.megatron.tensor_parallel import vocab_parallel_log_probs_from_logits

        batch: TensorDict = next(batch_iter)
        batch = batch.to(get_device_id())

        use_fused_kernels = tu.get_non_tensor_data(batch, key="use_fused_kernels", default=False)
        calculate_entropy = tu.get_non_tensor_data(batch, key="calculate_entropy", default=False)
        return_vocab_parallel_logits = tu.get_non_tensor_data(
            batch, key="return_vocab_parallel_logits", default=False
        )
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

        # Capture per-sample lengths up front so we can derive response-token logprobs
        # directly from the packed dense tensor before nested postprocess runs.
        # On the THD/remove-padding path we also use these lengths to zero out
        # alignment-only padded slots. On the BSHD path we must not apply the
        # THD packed valid_mask because postprocess already strips padding per row.
        use_remove_padding = bool(self.engine_config.use_remove_padding)
        if input_ids.is_nested:
            actual_seq_lens = input_ids.offsets().diff().tolist()
        else:
            actual_seq_lens = batch["attention_mask"].sum(dim=1).tolist()
        prompt_tensor = batch["prompts"]
        if getattr(prompt_tensor, "is_nested", False):
            prompt_lens = prompt_tensor.offsets().diff().tolist()
        else:
            prompt_lens = batch["attention_mask"][:, : prompt_tensor.shape[1]].sum(dim=1).tolist()
        response_mask = batch.get("response_mask")
        responses = batch.get("responses")
        response_lens = response_mask.sum(dim=1).tolist() if response_mask is not None else []
        max_response_len = responses.shape[1] if responses is not None else 0
        response_log_probs_dense = None
        packed_log_probs = None
        packed_vocab_parallel_logits = None
        import os
        debug_enabled = os.environ.get("MINT_VERL_DIAGNOSTICS", "0") == "1"

        def logits_processor(logits, temperature, **label_kwargs):
            """Process logits to compute log_probs."""
            nonlocal response_log_probs_dense, packed_log_probs, packed_vocab_parallel_logits
            import torch
            import torch.nn.functional as F
            from megatron.core import parallel_state as mpu

            # Get label from either key
            # NOTE: Can't use `or` here because tensors don't support boolean evaluation
            label = label_kwargs.get("label")
            if label is None:
                label = label_kwargs.get("external_label")
            if label is None:
                raise ValueError(f"No label found in kwargs: {label_kwargs.keys()}")

            assert logits.shape[:2] == label.shape[:2], f"Shape mismatch: logits={logits.shape}, label={label.shape}"

            temperature[temperature <= 0] = 1e-8
            assert torch.all(temperature > 0).item(), f"temperature must be positive. Got {temperature}"

            logits.div_(temperature.unsqueeze(dim=-1).to(logits.dtype))
            ret = {}
            if calculate_entropy:
                logits_bak = logits.clone()
                entropy = vocab_parallel_entropy(logits)
                ret["entropy"] = entropy
            else:
                logits_bak = logits

            if return_vocab_parallel_logits:
                ret["vocab_parallel_logits"] = logits_bak

            # Compute log_probs via cross-entropy
            log_probs = vocab_parallel_log_probs_from_logits(logits_bak, label)
            tp_size = mpu.get_tensor_model_parallel_world_size()
            cp_size = mpu.get_context_parallel_world_size()
            align_size = tp_size * cp_size * 2 if cp_size > 1 else tp_size
            valid_mask = None

            if use_remove_padding:
                # THD/remove-padding emits a single packed row before postprocess; zero out
                # alignment-only padded slots so downstream metrics/loss ignore them.
                padded_total_len = log_probs.shape[1]
                valid_mask = torch.zeros(padded_total_len, dtype=torch.bool, device=log_probs.device)

                cu_padded = 0
                for actual_len in actual_seq_lens:
                    pad_size = (align_size - actual_len % align_size) % align_size
                    padded_len = actual_len + pad_size
                    valid_mask[cu_padded : cu_padded + actual_len] = True
                    cu_padded += padded_len // cp_size

                n_valid = valid_mask.sum().item()
                n_padded = padded_total_len - n_valid
                if debug_enabled:
                    logger.warning(
                        "[VERL_DIAG] THD packed mask: use_remove_padding=%s logits_shape=%s "
                        "seq_lens=%s padded_total_len=%s n_valid=%s n_padded=%s",
                        use_remove_padding,
                        tuple(log_probs.shape),
                        actual_seq_lens,
                        padded_total_len,
                        n_valid,
                        n_padded,
                    )

                if n_padded > 0:
                    log_probs[0, ~valid_mask] = 0.0
            elif debug_enabled:
                logger.warning(
                    "[VERL_DIAG] BSHD path: skip packed valid_mask use_remove_padding=%s "
                    "logits_shape=%s seq_lens=%s",
                    use_remove_padding,
                    tuple(log_probs.shape),
                    actual_seq_lens,
                )

            ret["log_probs"] = log_probs
            if log_probs.dim() == 1:
                packed_log_probs = log_probs
            elif log_probs.dim() == 2 and log_probs.shape[0] == len(actual_seq_lens):
                packed_chunks = []
                for row, actual_len in zip(log_probs, actual_seq_lens, strict=True):
                    packed_chunks.append(row[: int(actual_len)])
                if packed_chunks:
                    packed_log_probs = torch.cat(packed_chunks, dim=0)
            elif use_remove_padding and log_probs.dim() == 2 and log_probs.shape[0] == 1:
                packed_chunks = []
                packed_start = 0
                for actual_len in actual_seq_lens:
                    actual_len_i = int(actual_len)
                    pad_size = (align_size - actual_len_i % align_size) % align_size
                    packed_chunks.append(log_probs[0, packed_start : packed_start + actual_len_i])
                    packed_start += actual_len_i + pad_size
                if packed_chunks:
                    packed_log_probs = torch.cat(packed_chunks, dim=0)
            if return_vocab_parallel_logits and logits_bak.dim() == 3:
                if logits_bak.shape[0] == len(actual_seq_lens):
                    packed_logits_chunks = []
                    for row, actual_len in zip(logits_bak, actual_seq_lens, strict=True):
                        packed_logits_chunks.append(row[: int(actual_len), :])
                    if packed_logits_chunks:
                        packed_vocab_parallel_logits = torch.cat(packed_logits_chunks, dim=0)
                elif use_remove_padding and valid_mask is not None and logits_bak.shape[0] == 1:
                    packed_vocab_parallel_logits = logits_bak[0, valid_mask, :]
            elif return_vocab_parallel_logits and logits_bak.dim() == 2:
                packed_vocab_parallel_logits = logits_bak
            if return_vocab_parallel_logits and packed_vocab_parallel_logits is None:
                logger.info(
                    "packed vocab logits missing logits_shape=%s actual_seq_lens=%s valid_mask_numel=%s valid_tokens=%s response_lens=%s",
                    list(logits_bak.shape),
                    [int(x) for x in actual_seq_lens],
                    int(valid_mask.numel()) if valid_mask is not None else None,
                    int(valid_mask.sum().item()) if valid_mask is not None else None,
                    [int(x) for x in response_lens],
                )
            if cp_size == 1 and response_lens:
                response_rows = []
                row_debug = []
                per_row_layout = (
                    log_probs.dim() == 2
                    and log_probs.shape[0] == len(response_lens)
                )
                packed_layout = use_remove_padding and log_probs.dim() == 2 and log_probs.shape[0] == 1
                packed_start = 0
                for row_idx, (prompt_len, response_len, actual_len) in enumerate(
                    zip(prompt_lens, response_lens, actual_seq_lens, strict=True)
                ):
                    prompt_len_i = int(prompt_len)
                    response_len_i = int(response_len)
                    actual_len_i = int(actual_len)
                    start = max(prompt_len_i - 1, 0)
                    end = start + response_len_i
                    if per_row_layout:
                        row = log_probs[row_idx, start:end]
                    elif packed_layout:
                        row = log_probs[0, packed_start + start : packed_start + end]
                    else:
                        continue
                    row_debug.append(
                        {
                            "row_idx": row_idx,
                            "actual_len": actual_len_i,
                            "prompt_len": prompt_len_i,
                            "response_len": response_len_i,
                            "per_row_layout": per_row_layout,
                            "packed_layout": packed_layout,
                            "log_probs_shape": list(log_probs.shape),
                            "packed_start": packed_start,
                            "start": start,
                            "end": end,
                            "row_numel": int(row.numel()),
                        }
                    )
                    if max_response_len > response_len_i:
                        row = F.pad(row, (0, max_response_len - response_len_i))
                    response_rows.append(row)
                    if packed_layout:
                        pad_size = (align_size - actual_len_i % align_size) % align_size
                        packed_start += actual_len_i + pad_size
                row_lengths = [int(r.numel()) for r in response_rows]
                if len(set(row_lengths)) > 1:
                    raise RuntimeError(
                        f"packed response_log_probs row mismatch row_lengths={row_lengths} row_debug={row_debug}"
                    )
                if response_rows:
                    response_log_probs_dense = torch.stack(response_rows, dim=0)
            return ret

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

        if isinstance(output, dict):
            if response_log_probs_dense is not None:
                output["response_log_probs"] = response_log_probs_dense
            if packed_log_probs is not None:
                output["packed_log_probs"] = packed_log_probs
            if packed_vocab_parallel_logits is not None:
                output["packed_vocab_parallel_logits"] = packed_vocab_parallel_logits

        return output, partial(postprocess_micro_batch_func, data=batch)

    MegatronEngineWithLMHead.forward_step = patched_forward_step
    print("[VERL_PATCH] Applied external label patch for MegatronEngineWithLMHead.forward_step")
    logger.info("Applied external label patch (fixes last-token logprob issue)")

    try:
        from verl.workers.engine import utils as verl_engine_utils
        from verl.workers.engine.megatron import transformer_impl as verl_transformer_impl
    except Exception as e:
        logger.warning(
            "Could not import verl.workers.engine.utils for packed logprob postprocess patch: %s",
            e,
        )
        return

    if not getattr(verl_engine_utils, "_tinker_packed_logprob_postprocess_patched", False):
        original_postprocess_batch_func = verl_engine_utils.postprocess_batch_func

        @wraps(original_postprocess_batch_func)
        def patched_postprocess_batch_func(output_lst, indices, data):
            packed_values = []
            packed_vocab_logits = []
            cleaned_output_lst = []
            for micro_output in output_lst:
                if not isinstance(micro_output, dict):
                    cleaned_output_lst.append(micro_output)
                    continue

                micro_output_copy = dict(micro_output)
                micro_model_output = micro_output_copy.get("model_output")
                if isinstance(micro_model_output, dict):
                    micro_model_output_copy = dict(micro_model_output)
                    value = micro_model_output_copy.pop("packed_log_probs", None)
                    if value is not None:
                        packed_values.append(value.reshape(-1))
                    packed_logits = micro_model_output_copy.pop("packed_vocab_parallel_logits", None)
                    if packed_logits is not None:
                        packed_vocab_logits.append(packed_logits)
                    micro_output_copy["model_output"] = micro_model_output_copy
                cleaned_output_lst.append(micro_output_copy)

            output = original_postprocess_batch_func(cleaned_output_lst, indices, data)
            if not isinstance(output, dict):
                return output
            model_output = output.get("model_output")
            if not isinstance(model_output, dict):
                return output

            if packed_values:
                model_output["packed_log_probs"] = torch.cat(packed_values, dim=0)
            if packed_vocab_logits:
                model_output["packed_vocab_parallel_logits"] = torch.cat(packed_vocab_logits, dim=0)
            return output

        verl_engine_utils.postprocess_batch_func = patched_postprocess_batch_func
        verl_transformer_impl.postprocess_batch_func = patched_postprocess_batch_func
        verl_engine_utils._tinker_packed_logprob_postprocess_patched = True  # type: ignore[attr-defined]
        logger.info("Applied packed logprob postprocess patch")


def _apply_rope_thd_cp_len_clamp_patch() -> None:
    """Patch Megatron RoPE THD CP slicing when freqs is only max_seq_len long.

    megatron/core/models/common/embeddings/rope_utils.py uses:
        full_seqlen = cp_size * x.size(0)

    When CP=2 but freqs.size(0) == x.size(0) (max_seq_len tensor, not "2*max_seq_len"),
    the backward slice becomes empty and _get_thd_freqs_on_this_cp_rank returns half-length freqs,
    triggering:
        RuntimeError: size of tensor a (30016) must match size of tensor b (15008)
    """
    try:
        from megatron.core.models.common.embeddings import rope_utils
    except Exception as e:
        logger.warning(f"Could not import megatron rope_utils; skipping RoPE CP patch: {type(e).__name__}: {e}")
        return

    if getattr(rope_utils, "_tinker_rope_thd_cp_full_seqlen_clamp_patched", False):
        return

    original = rope_utils._get_thd_freqs_on_this_cp_rank

    def patched(cp_rank: int, cp_size: int, x: torch.Tensor, freqs: torch.Tensor, offset: int = 0) -> torch.Tensor:
        if cp_size <= 1:
            return original(cp_rank, cp_size, x, freqs, offset)

        cp_seg = x.size(0) // 2
        full_seqlen = cp_size * x.size(0)

        max_full = max(0, freqs.size(0) - offset)
        if full_seqlen > max_full:
            full_seqlen = max_full

        return torch.cat(
            [
                freqs[offset + cp_rank * cp_seg : offset + (cp_rank + 1) * cp_seg],
                freqs[
                    offset
                    + full_seqlen
                    - (cp_rank + 1) * cp_seg : offset
                    + full_seqlen
                    - cp_rank * cp_seg
                ],
            ]
        )

    rope_utils._get_thd_freqs_on_this_cp_rank = patched  # type: ignore[attr-defined]
    rope_utils._tinker_rope_thd_cp_full_seqlen_clamp_patched = True  # type: ignore[attr-defined]
    print("[VERL_PATCH] Patched rope_utils._get_thd_freqs_on_this_cp_rank (clamp full_seqlen)")


def _apply_yarn_rope_cp_seq_len_align_patch() -> None:
    """Patch YarnRotaryEmbedding to build CP-aligned caches.

    Megatron's `get_pos_emb_on_this_cp_rank()` reshapes the sequence dimension into
    `2 * cp_size` partitions:

        pos_emb.view(..., 2 * cp_size, -1, ...)

    This requires the cached sequence length to be divisible by `2 * cp_size`.
    For CP=3, the common 4096 cache length fails (4096 % 6 != 0), causing:

        RuntimeError: shape '[6, -1, ...]' is invalid for input of size ...

    Aligning the cache length upward is safe because the first `seq_len` positions
    are unchanged (arange-based construction); we only compute extra tail positions.
    """
    try:
        from megatron.core.models.common.embeddings import yarn_rotary_pos_embedding
    except Exception as e:
        logger.warning(
            f"Could not import megatron yarn_rotary_pos_embedding; skipping Yarn RoPE CP align patch: "
            f"{type(e).__name__}: {e}"
        )
        return

    if getattr(yarn_rotary_pos_embedding, "_tinker_yarn_rope_cp_seq_len_align_patched", False):
        return

    original = yarn_rotary_pos_embedding.YarnRotaryEmbedding._set_cos_sin_cache

    def patched(self, seq_len, offset, dtype, packed_seq=False):
        cp_group = getattr(self, "cp_group", None)
        if cp_group is not None and cp_group.size() > 1 and not packed_seq:
            align = 2 * int(cp_group.size())
            rem = int(seq_len) % align
            if rem:
                seq_len = int(seq_len) + (align - rem)
        return original(self, seq_len, offset, dtype, packed_seq)

    yarn_rotary_pos_embedding.YarnRotaryEmbedding._set_cos_sin_cache = patched  # type: ignore[assignment]
    yarn_rotary_pos_embedding._tinker_yarn_rope_cp_seq_len_align_patched = True  # type: ignore[attr-defined]
    print("[VERL_PATCH] Patched YarnRotaryEmbedding._set_cos_sin_cache (CP-aligned cache len)")


def _apply_preprocess_thd_no_padding_cp_short_seq_patch() -> None:
    """Patch verl preprocess_thd_no_padding for CP when input sequences are not explicitly padded.

    Observed failure mode (CP>1):
        RuntimeError: The expanded size ... must match ... (e.g. 64 vs 50)

    Root cause: verl computes per-sample padded lengths for CP alignment but slices from the
    original jagged/nested `input_ids[i]` without clamping, assuming it is already padded.
    """
    try:
        from verl.models.mcore import util as verl_mcore_util
    except Exception as e:
        logger.warning(f"Could not import verl.models.mcore.util; skipping CP short-seq patch: {type(e).__name__}: {e}")
        return

    if getattr(verl_mcore_util, "_tinker_preprocess_thd_no_padding_cp_short_seq_patched", False):
        return

    import torch
    from megatron.core import parallel_state as mpu
    from megatron.core.packed_seq_params import PackedSeqParams

    def patched_preprocess_thd_no_padding(
        input_ids: torch.Tensor, pre_process: bool = True, need_roll: bool = False
    ) -> tuple[torch.Tensor, PackedSeqParams]:
        batch_size = input_ids.shape[0]

        tp_size = mpu.get_tensor_model_parallel_world_size()
        cp_size = mpu.get_context_parallel_world_size()
        cp_rank = mpu.get_context_parallel_rank()
        align_size = tp_size * cp_size * 2 if cp_size > 1 else tp_size
        seqlens_in_batch = input_ids.offsets().diff()

        pad_size = (align_size - seqlens_in_batch % align_size) % align_size
        seqlens_in_batch_padded = seqlens_in_batch + pad_size

        cu_seqlens_padded = torch.zeros(batch_size + 1, dtype=torch.int32, device=input_ids.device)
        cu_seqlens_padded[1:] = torch.cumsum(seqlens_in_batch_padded, dim=0)

        seqlens_in_batch_cpu: list[int] = seqlens_in_batch.tolist()
        seqlens_in_batch_padded_cpu: list[int] = seqlens_in_batch_padded.tolist()
        cu_seqlens_padded_cpu: list[int] = cu_seqlens_padded.tolist()

        max_seqlen_in_batch = max(seqlens_in_batch_padded_cpu)

        shape = list(input_ids.shape[1:])
        shape[0] = sum(seqlens_in_batch_padded_cpu) // cp_size
        if pre_process:
            input_ids_rmpad = torch.zeros(shape, dtype=input_ids.dtype, device=input_ids.device)
            if need_roll:
                saved_roll_dict: dict[int, torch.Tensor] = {}

            for i in range(batch_size):
                if cp_size <= 1:
                    seqlen = seqlens_in_batch_cpu[i]
                    start_idx = cu_seqlens_padded_cpu[i]
                    input_ids_rmpad[start_idx : start_idx + seqlen] = input_ids[i]
                    continue

                seqlen_padded_i = seqlens_in_batch_padded_cpu[i]
                seqlen = seqlen_padded_i // cp_size
                half_seqlen = seqlen // 2
                start_idx = cu_seqlens_padded_cpu[i] // cp_size

                d = input_ids[i]
                # NOTE: d is often a jagged/nested sequence of the *valid* length, not explicitly padded.
                # Clamp the first-chunk slice to avoid shape mismatch on assignment.
                chunk_start = half_seqlen * cp_rank
                chunk_end = min(half_seqlen * (cp_rank + 1), d.shape[0])
                chunk_len = max(0, chunk_end - chunk_start)
                if chunk_len > 0:
                    input_ids_rmpad[start_idx : start_idx + chunk_len] = d[chunk_start : chunk_start + chunk_len]

                remain_start = seqlen_padded_i - half_seqlen * (cp_rank + 1)
                remain_end = seqlen_padded_i - half_seqlen * cp_rank
                remain_end = min(remain_end, d.shape[0])
                remain_len = max(0, remain_end - remain_start)
                if remain_len > 0:
                    input_ids_rmpad[start_idx + half_seqlen : start_idx + half_seqlen + remain_len] = d[
                        remain_start:remain_end
                    ]

                if need_roll:
                    idx = (cp_rank + 1) * half_seqlen
                    if idx < d.shape[0]:
                        saved_roll_dict[start_idx + half_seqlen - 1] = d[idx]
                    if remain_len > 0:
                        k = start_idx + half_seqlen + remain_len - 1
                        if remain_end == d.shape[0]:
                            saved_roll_dict[k] = d[0]
                        elif remain_end < d.shape[0]:
                            saved_roll_dict[k] = d[remain_end]

            if need_roll:
                input_ids_rmpad = torch.roll(input_ids_rmpad, shifts=-1, dims=0)
                for k, v in saved_roll_dict.items():
                    input_ids_rmpad[k] = v

        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens_padded,
            max_seqlen_q=max_seqlen_in_batch,
            cu_seqlens_kv=cu_seqlens_padded,
            max_seqlen_kv=max_seqlen_in_batch,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
        )
        if pre_process:
            return input_ids_rmpad.unsqueeze(0), packed_seq_params
        return input_ids, packed_seq_params

    # Patch the canonical definition.
    verl_mcore_util.preprocess_thd_no_padding = patched_preprocess_thd_no_padding  # type: ignore[assignment]

    # Patch any modules that imported the symbol by value (common pattern: `from ...util import preprocess_thd_no_padding`).
    try:
        from verl.models.mcore import model_forward as verl_mcore_model_forward
    except Exception:
        verl_mcore_model_forward = None
    if verl_mcore_model_forward is not None and hasattr(verl_mcore_model_forward, "preprocess_thd_no_padding"):
        verl_mcore_model_forward.preprocess_thd_no_padding = patched_preprocess_thd_no_padding  # type: ignore[assignment]
    verl_mcore_util._tinker_preprocess_thd_no_padding_cp_short_seq_patched = True  # type: ignore[attr-defined]
    print("[VERL_PATCH] Patched verl preprocess_thd_no_padding (CP short-seq clamp)")


def _apply_te_triton_get_int_dtype_patch() -> None:
    """Patch TransformerEngine MoE permutation kernels for Triton compatibility.

    Seen on Volcano A800 workers during MoE token dispatch:
        RuntimeError: Unsupported function referenced: <function get_int_dtype ...>

    Source: transformer_engine/pytorch/triton/permutation.py uses:
        idtype = core.get_int_dtype(...)
    inside @triton.jit code. Newer Triton rejects that python-level helper.

    Fix: replace transformer_engine.pytorch.triton.permutation._compare_and_swap with an
    equivalent implementation that hardcodes idtype=tl.int32 (the MoE routing map is int32).
    """
    try:
        import transformer_engine.pytorch.triton.permutation as te_perm
    except Exception as e:
        logger.warning(f"Could not import transformer_engine triton permutation; skipping patch: {type(e).__name__}: {e}")
        return

    if getattr(te_perm, "_tinker_te_triton_get_int_dtype_patched", False):
        return

    try:
        import triton
        import triton.language as tl
    except Exception as e:
        logger.warning(f"Could not import triton; skipping TE permutation patch: {type(e).__name__}: {e}")
        return

    @triton.jit
    def _compare_and_swap(x, indices, flip, i: tl.constexpr, n_dims: tl.constexpr):
        n_outer: tl.constexpr = x.numel >> n_dims
        shape: tl.constexpr = [n_outer * (2**i), 2, 2 ** (n_dims - i - 1)]
        y = tl.reshape(x, shape)
        z = tl.reshape(indices, shape)

        mask = tl.arange(0, 2)[None, :, None]

        l_value = tl.reshape(tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape), x.shape).to(x.dtype)
        r_value = tl.reshape(tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape), x.shape).to(x.dtype)

        l_indice = tl.reshape(tl.broadcast_to(tl.sum(z * (1 - mask), 1)[:, None, :], shape), x.shape)
        r_indice = tl.reshape(tl.broadcast_to(tl.sum(z * mask, 1)[:, None, :], shape), x.shape)

        idtype = tl.int32

        il_value = l_value.to(idtype, bitcast=True)
        ir_value = r_value.to(idtype, bitcast=True)
        ix = x.to(idtype, bitcast=True)

        flag1 = tl.where(((l_value > r_value) ^ flip) != 0, il_value ^ ir_value, tl.zeros_like(ix))
        ret = ix ^ flag1
        flag2 = tl.where(((l_value > r_value) ^ flip) != 0, l_indice ^ r_indice, tl.zeros_like(ix))
        ind = indices ^ flag2

        return ret.to(x.dtype, bitcast=True), ind

    te_perm._compare_and_swap = _compare_and_swap  # type: ignore[assignment]
    te_perm._tinker_te_triton_get_int_dtype_patched = True  # type: ignore[attr-defined]
    print("[VERL_PATCH] Patched transformer_engine triton permutation _compare_and_swap (no core.get_int_dtype)")


def _apply_megatron_checkpoint_cleanup_patch() -> None:
    """Patch Megatron checkpoint backward to clear recompute tensors after grads are extracted."""
    try:
        from megatron.core.tensor_parallel import random as tp_random
    except Exception as e:
        logger.warning(
            "Could not import megatron.core.tensor_parallel.random; skipping checkpoint cleanup patch: %s",
            e,
        )
        return

    if getattr(tp_random, "_tinker_checkpoint_cleanup_patched", False):
        return

    def _clear_many(*values) -> None:
        try:
            from transformer_engine.pytorch.utils import clear_tensor_data
        except Exception:
            clear_tensor_data = None
        for value in values:
            if isinstance(value, (list, tuple)):
                _clear_many(*value)
                continue
            if not torch.is_tensor(value):
                continue
            try:
                if clear_tensor_data is not None:
                    clear_tensor_data(value)
                else:
                    value.data = torch.empty(0, dtype=value.dtype, device=value.device)
            except Exception:
                pass

    original_checkpoint_backward = tp_random.CheckpointFunction.backward

    @staticmethod
    @wraps(original_checkpoint_backward)
    def patched_checkpoint_backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), please use .backward() if possible"
            )

        inputs = ctx.saved_tensors
        if ctx.distribute_saved_activations:
            tp_random.safely_set_viewless_tensor_data(
                inputs[0],
                tp_random.gather_split_1d_tensor(inputs[0].data).view(ctx.input_0_shape),
            )

        detached_inputs = ()
        outputs = ()
        try:
            with tp_random._fork_rng():
                tp_random._set_all_rng_states(*ctx.rng_states)
                detached_inputs = tp_random.detach_variable(inputs)
                with torch.enable_grad():
                    outputs = ctx.run_function(*detached_inputs)

            if isinstance(outputs, torch.Tensor):
                outputs = (outputs,)

            # Preserve upstream backward semantics exactly, then clean up afterward.
            # PR 370's first version added extra filtering here, which changed the
            # effective backward graph before any cleanup happened.
            outputs, args = zip(
                *filter(lambda x: torch.is_tensor(x[0]) and x[0].requires_grad, zip(outputs, args))
            )
            try:
                torch.autograd.backward(outputs, args)
            except Exception:
                def _describe(value):
                    if isinstance(value, (list, tuple)):
                        return [_describe(v) for v in value]
                    if not torch.is_tensor(value):
                        return {"type": type(value).__name__}
                    return {
                        "shape": list(value.shape),
                        "numel": int(value.numel()),
                        "dtype": str(value.dtype),
                        "device": str(value.device),
                        "requires_grad": bool(value.requires_grad),
                        "grad_fn": type(value.grad_fn).__name__ if value.grad_fn is not None else None,
                    }

                logger.error(
                    "CheckpointFunction.backward exception diag inputs=%s detached_inputs=%s outputs=%s args=%s",
                    _describe(inputs),
                    _describe(detached_inputs),
                    _describe(outputs),
                    _describe(args),
                    exc_info=True,
                )
                raise
            grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else None for inp in detached_inputs)
            return (None, None) + grads
        finally:
            # Upstream keeps ctx.saved_tensors intact until this autograd frame
            # fully unwinds. Clearing the original saved inputs here can leave
            # later gradient edges observing zero-sized storage. Only clear the
            # detached recompute copies, which are local to this backward.
            _clear_many(*detached_inputs)
            for value in detached_inputs:
                if torch.is_tensor(value):
                    try:
                        value.grad = None
                    except Exception:
                        pass
            for attr, value in (
                ("run_function", None),
                ("rng_states", None),
                ("input_0_shape", None),
                ("distribute_saved_activations", False),
            ):
                try:
                    setattr(ctx, attr, value)
                except Exception:
                    pass

    tp_random.CheckpointFunction.backward = patched_checkpoint_backward

    original_without_output_backward = tp_random.CheckpointWithoutOutputFunction.backward

    @staticmethod
    @wraps(original_without_output_backward)
    def patched_without_output_backward(ctx, *args):
        try:
            return original_without_output_backward(ctx, *args)
        except Exception:
            def _describe(value):
                if isinstance(value, (list, tuple)):
                    return [_describe(v) for v in value]
                if value is None:
                    return None
                if not torch.is_tensor(value):
                    return {"type": type(value).__name__}
                return {
                    "shape": list(value.shape),
                    "numel": int(value.numel()),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                    "requires_grad": bool(value.requires_grad),
                    "grad_fn": type(value.grad_fn).__name__ if value.grad_fn is not None else None,
                }

            logger.error(
                "CheckpointWithoutOutputFunction.backward exception diag saved_tensors=%s inputs=%s outputs=%s args=%s",
                _describe(getattr(ctx, "saved_tensors", ()) or ()),
                _describe(getattr(ctx, "inputs", ()) or ()),
                _describe(getattr(ctx, "outputs", ()) or ()),
                _describe(args),
                exc_info=True,
            )
            raise
        finally:
            # Upstream releases these references by nulling ctx fields after the
            # local backward returns. Do not eagerly zero their storage inside
            # the autograd frame.
            for attr, value in (
                ("inputs", None),
                ("fp8_recipe", None),
            ):
                try:
                    setattr(ctx, attr, value)
                except Exception:
                    pass

    tp_random.CheckpointWithoutOutputFunction.backward = patched_without_output_backward
    tp_random._tinker_checkpoint_cleanup_patched = True  # type: ignore[attr-defined]
    logger.info("Applied Megatron checkpoint backward cleanup patch")


def _apply_megatron_backward_shape_diag_patch() -> None:
    """Log exact tensor metadata when Megatron backward_step throws."""
    try:
        from megatron.core.pipeline_parallel import schedules
    except Exception as e:
        logger.warning(
            "Could not import megatron.core.pipeline_parallel.schedules; skipping backward-step diag patch: %s",
            e,
        )
        return

    if getattr(schedules, "_tinker_backward_shape_diag_patched", False):
        return

    original_backward_step = schedules.backward_step

    def _describe(value):
        if isinstance(value, list):
            return [_describe(v) for v in value]
        if value is None:
            return None
        if not torch.is_tensor(value):
            return {"type": type(value).__name__}
        return {
            "shape": list(value.shape),
            "numel": int(value.numel()),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "requires_grad": bool(value.requires_grad),
            "grad_fn": type(value.grad_fn).__name__ if value.grad_fn is not None else None,
            "next_functions": [
                type(fn[0]).__name__ if fn and fn[0] is not None else None
                for fn in (getattr(value.grad_fn, "next_functions", ()) or ())[:8]
            ],
        }

    @wraps(original_backward_step)
    def patched_backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config):
        try:
            with torch.autograd.set_detect_anomaly(True):
                return original_backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)
        except Exception as e:
            input_desc = _describe(input_tensor)
            output_desc = _describe(output_tensor)
            output_grad_desc = _describe(output_tensor_grad)
            logger.error(
                "Megatron backward_step exception diag model_type=%s input=%s output=%s output_grad=%s",
                model_type,
                input_desc,
                output_desc,
                output_grad_desc,
                exc_info=True,
            )
            raise RuntimeError(
                "Megatron backward_step exception diag "
                f"model_type={model_type} input={input_desc} output={output_desc} "
                f"output_grad={output_grad_desc} original={type(e).__name__}: {e}"
            ) from e

    schedules.backward_step = patched_backward_step
    schedules._tinker_backward_shape_diag_patched = True  # type: ignore[attr-defined]
    logger.info("Applied Megatron backward-step shape diag patch")


def _apply_nested_no_padding_slice_patch() -> None:
    """Avoid NestedGetValuesBackward on PPO/value response slicing."""
    try:
        from verl.workers.utils import losses as verl_losses
        from verl.workers.utils import padding as verl_padding
    except Exception as e:
        logger.warning(
            "Could not import verl loss/padding helpers; skipping nested slice patch: %s",
            e,
        )
        return

    if getattr(verl_losses, "_tinker_nested_no_padding_slice_patched", False):
        return

    def _slice_from_padded(tensor: torch.Tensor, data, *, max_response_len_default: int | None = None):
        prompt_ids = data["prompts"]
        response_ids = data["responses"]
        attention_mask = data["attention_mask"]

        if tensor.is_nested and prompt_ids.is_nested:
            prompt_lens = prompt_ids.offsets().diff()
            response_lens = response_ids.offsets().diff()
            if max_response_len_default is None:
                max_response_len = int(response_lens.max().item()) if response_lens.numel() else 0
            else:
                max_response_len = int(max_response_len_default)
            rows = []
            for row_idx, (prompt_len, response_len) in enumerate(zip(prompt_lens, response_lens, strict=True)):
                prompt_len_i = int(prompt_len.item()) if hasattr(prompt_len, "item") else int(prompt_len)
                response_len_i = int(response_len.item()) if hasattr(response_len, "item") else int(response_len)
                start = max(prompt_len_i - 1, 0)
                end = start + response_len_i
                row_values = tensor[row_idx]
                row = row_values[start:end]
                pad_size = max_response_len - response_len_i
                if pad_size > 0:
                    row = F.pad(row, (0, pad_size))
                rows.append(row)
            if not rows:
                return torch.empty((0, 0), dtype=tensor.dtype, device=tensor.device)
            return torch.stack(rows, dim=0)

        if prompt_ids.is_nested:
            prompt_lens = prompt_ids.offsets().diff()
            response_lens = response_ids.offsets().diff()
            if max_response_len_default is None:
                max_response_len = int(response_lens.max().item()) if response_lens.numel() else 0
            else:
                max_response_len = int(max_response_len_default)
        else:
            assert not attention_mask.is_nested
            prompt_lens = attention_mask[:, : prompt_ids.shape[1]].sum(dim=1)
            response_lens = attention_mask[:, prompt_ids.shape[1] :].sum(dim=1)
            max_response_len = response_ids.shape[1]

        padded = tensor.to_padded_tensor(0.0) if tensor.is_nested else tensor
        rows = []
        for row_idx, (prompt_len, response_len) in enumerate(zip(prompt_lens, response_lens, strict=True)):
            prompt_len_i = int(prompt_len.item()) if hasattr(prompt_len, "item") else int(prompt_len)
            response_len_i = int(response_len.item()) if hasattr(response_len, "item") else int(response_len)
            start = max(prompt_len_i - 1, 0)
            end = start + response_len_i
            row = padded[row_idx, start:end]
            pad_size = max_response_len - response_len_i
            if pad_size > 0:
                row = F.pad(row, (0, pad_size))
            rows.append(row)
        if not rows:
            return torch.empty((0, 0), dtype=padded.dtype, device=padded.device)
        return torch.stack(rows, dim=0)

    def patched_slice_response_from_unpad_output(tensor: torch.Tensor, data) -> torch.Tensor:
        return _slice_from_padded(tensor, data)

    def patched_no_padding_2_padding(tensor: torch.Tensor, data) -> torch.Tensor:
        max_response_len = tu.get_non_tensor_data(data=data, key="max_response_len", default=-1)
        return _slice_from_padded(tensor, data, max_response_len_default=max_response_len)

    verl_losses._slice_response_from_unpad_output = patched_slice_response_from_unpad_output
    verl_padding.no_padding_2_padding = patched_no_padding_2_padding
    verl_losses.no_padding_2_padding = patched_no_padding_2_padding
    verl_losses._tinker_nested_no_padding_slice_patched = True  # type: ignore[attr-defined]
    logger.info("Applied nested no-padding response slice patch")


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

    # TransformerEngine uses `importlib.metadata.version("flash-attn")` which returns a version
    # string with a local suffix (e.g. "2.8.3+cu129torch2.9"). It compares that strictly against
    # max_version="2.8.3" and disables FlashAttention, leaving `flash_attn_func=None` and later
    # crashing with "TypeError: 'NoneType' object is not callable".
    #
    # Patch importlib.metadata.version early, before any TransformerEngine attention modules import it.
    try:
        import importlib.metadata as _imd

        _orig_version = getattr(_imd, "version", None)
        if callable(_orig_version) and not getattr(_orig_version, "_tinker_flash_attn_version_patch", False):
            def _patched_version(dist_name: str) -> str:
                v = _orig_version(dist_name)
                if dist_name in ("flash-attn", "flash-attn-3") and isinstance(v, str) and "+" in v:
                    return v.split("+", 1)[0]
                return v

            _patched_version._tinker_flash_attn_version_patch = True  # type: ignore[attr-defined]
            _imd.version = _patched_version  # type: ignore[assignment]
    except Exception as e:
        logger.warning(f"Failed to patch importlib.metadata.version for flash-attn: {e}")

    if os.environ.get("NVTE_FLASH_ATTN", "1") != "0":
        # TransformerEngine disables FlashAttention when the installed flash-attn distribution
        # version contains a local suffix (e.g. "2.8.3+cu129torch2.9") and is compared strictly
        # against max_version="2.8.3". That forces a fallback to O(seq^2) attention and OOMs at
        # long context.
        #
        # Force-enable FlashAttention by normalizing TE's cached version object before Megatron
        # imports initialize TE attention modules.
        try:
            from transformer_engine.pytorch.attention.dot_product_attention.utils import FlashAttentionUtils, PkgVersion
            te_ver = str(getattr(FlashAttentionUtils, "version", ""))
            if "+" in te_ver:
                base = te_ver.split("+", 1)[0]
                FlashAttentionUtils.version = PkgVersion(base)
                print(f"[VERL_PATCH] Normalized TransformerEngine FlashAttentionUtils.version {te_ver} -> {base}")
            FlashAttentionUtils.set_flash_attention_version()
        except Exception as e:
            logger.warning(f"Failed to force-enable TransformerEngine FlashAttention: {e}")

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
    logger.info("Enabled full determinism mode (FLASH_ATTENTION_DETERMINISTIC=1, cudnn.deterministic=True)")


def apply_verl_patches():
    """Apply monkey-patches to verl for MLA attention backend fix and external labels support."""
    # Note: _enable_megatron_determinism() is now called separately BEFORE this,
    # in megatron_distributed.py, to ensure it runs before any Megatron imports.

    try:
        from verl.workers.engine.megatron.transformer_impl import MegatronEngine
    except ImportError:
        logger.warning("verl.workers.engine.megatron.transformer_impl not found, skipping patches")
        return

    _apply_megatron_router_expert_bias_no_stack_patch()
    _apply_te_triton_get_int_dtype_patch()
    _apply_megatron_backward_shape_diag_patch()
    _apply_nested_no_padding_slice_patch()
    _apply_megatron_checkpoint_cleanup_patch()
    # Apply external label patch first (fixes last-token logprob issue)
    _apply_external_label_patch()
    _apply_rope_thd_cp_len_clamp_patch()
    _apply_yarn_rope_cp_seq_len_align_patch()
    _apply_preprocess_thd_no_padding_cp_short_seq_patch()

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
                print("[VERL_PATCH] MLA model detected, using AttnBackend.flash (with value padding)")
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

    # Override stale Verl MLA QKV patch with logic matching the current Megatron runtime.
    _apply_mla_get_query_key_value_patch()

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
        # Start with log_probs (required)
        log_prob = output.get("log_probs")
        if log_prob is None:
            return original_prepare_model_outputs(self, output, data)

        model_output = {"log_probs": log_prob}

        # Pass through ALL other keys (including topk_indices, topk_logits, entropy, etc.)
        for key, value in output.items():
            if key != "log_probs":  # Already added
                model_output[key] = value
        return model_output

    MegatronEngineWithLMHead.prepare_model_outputs = patched_prepare_model_outputs
    print("[VERL_PATCH] Applied prepare_model_outputs patch (passes through all keys including topk)")
    logger.info("Applied prepare_model_outputs patch (passes through all keys)")


def _apply_mla_get_query_key_value_patch():
    import torch
    from megatron.core import parallel_state, tensor_parallel

    try:
        from megatron.core.transformer.multi_latent_attention import (
            MLASelfAttention,
            apply_rotary_pos_emb,
            deprecate_inference_params,
            fused_apply_mla_rope_for_kv,
            fused_apply_mla_rope_for_q,
            gather_from_sequence_parallel_region,
            gather_from_tensor_model_parallel_region,
            scatter_to_sequence_parallel_region,
        )
    except ImportError:
        logger.warning("MLASelfAttention not found, skipping MLA QKV patch")
        return

    original = MLASelfAttention.get_query_key_value_tensors
    if getattr(original, "_tinker_mla_qkv_patch", False):
        return

    def patch_get_query_key_value_tensors(
        self,
        hidden_states,
        key_value_states=None,
        position_ids=None,
        packed_seq_params=None,
        inference_context=None,
        *,
        inference_params=None,
    ):
        assert hidden_states.ndim == 3, f"hidden_states should be 3D, [s, b, n*h], got {hidden_states.ndim}D"
        if packed_seq_params is not None:
            assert (
                packed_seq_params.local_cp_size is None
            ), "hybrid_context_parallel is not supported with MLA yet and is planned for future. Please disable hybrid_context_parallel."

        inference_context = deprecate_inference_params(inference_context, inference_params)

        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            inference_context, None, hidden_states, self.config, packed_seq_params
        )

        mscale = 1.0
        rotary_pos_cos = None
        rotary_pos_sin = None
        packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
        if self.config.rope_type == "rope":
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
        else:
            if self.config.apply_rope_fusion:
                rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb.get_cached_cos_sin(
                    rotary_seq_len, dtype=hidden_states.dtype, packed_seq=packed_seq
                )
                rotary_pos_emb = None
                assert inference_context is None, "Inference with MLA RoPE fusion is not supported"
                assert (
                    fused_apply_mla_rope_for_q is not None
                    and fused_apply_mla_rope_for_kv is not None
                ), "Fused MLA RoPE apply is not imported successfully"
            else:
                rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)

        if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
            if packed_seq_params.cu_seqlens_q_padded is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
            else:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
            if packed_seq_params.cu_seqlens_kv_padded is not None:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
            else:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
        else:
            cu_seqlens_q = cu_seqlens_kv = None

        if self.config.q_lora_rank is not None:
            q_compressed, _ = self.linear_q_down_proj(hidden_states)
            if q_compressed.size(-1) != self.config.q_lora_rank:
                q_compressed = gather_from_tensor_model_parallel_region(q_compressed)
                if self.config.sequence_parallel:
                    q_compressed = scatter_to_sequence_parallel_region(q_compressed)
        else:
            q_compressed = hidden_states

        kv_combined, _ = self.linear_kv_down_proj(hidden_states)
        if kv_combined.size(-1) != self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim:
            kv_combined = gather_from_tensor_model_parallel_region(kv_combined)
            kv_compressed, k_pos_emb = torch.split(
                kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
            )
            if self.config.sequence_parallel:
                kv_compressed = scatter_to_sequence_parallel_region(kv_compressed)
        else:
            kv_compressed, k_pos_emb = torch.split(
                kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
            )
            if parallel_state.get_tensor_model_parallel_world_size() > 1 and self.config.sequence_parallel:
                k_pos_emb = gather_from_sequence_parallel_region(k_pos_emb, group=self.tp_group)

        if packed_seq_params is not None:
            q_compressed = q_compressed.squeeze(1)
            kv_compressed = kv_compressed.squeeze(1)
            k_pos_emb = k_pos_emb.squeeze(1)

        if self.config.q_lora_rank is not None:
            q_compressed = self.q_layernorm(q_compressed)
        kv_compressed = self.kv_layernorm(kv_compressed)

        def qkv_up_proj_and_rope_apply(q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb):
            if self.config.q_lora_rank is not None:
                q, _ = self.linear_q_up_proj(q_compressed)
            else:
                q, _ = self.linear_q_proj(q_compressed)
            q = q.view(*q.size()[:-1], self.num_attention_heads_per_partition, self.q_head_dim)

            kv, _ = self.linear_kv_up_proj(kv_compressed)
            kv = kv.view(
                *kv.size()[:-1],
                self.num_attention_heads_per_partition,
                self.config.qk_head_dim + self.config.v_head_dim,
            )

            k_pos_emb = torch.unsqueeze(k_pos_emb, -2)

            if self.config.apply_rope_fusion:
                cp_rank = self.pg_collection.cp.rank()
                cp_size = self.pg_collection.cp.size()
                query = fused_apply_mla_rope_for_q(
                    q,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    self.config.qk_head_dim,
                    self.config.qk_pos_emb_head_dim,
                    cu_seqlens_q,
                    cp_rank,
                    cp_size,
                )
                key, value = fused_apply_mla_rope_for_kv(
                    kv,
                    k_pos_emb,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    self.config.qk_pos_emb_head_dim,
                    self.config.qk_head_dim,
                    self.config.v_head_dim,
                    cu_seqlens_kv,
                    cp_rank,
                    cp_size,
                )
            else:
                q_len = q.size()[0]
                if inference_context is not None:
                    sequence_start = inference_context.sequence_len_offset
                    sequence_end = sequence_start + q_len
                    rotary_pos_emb = rotary_pos_emb[sequence_start:sequence_end]
                elif packed_seq_params is None or parallel_state.get_context_parallel_world_size() == 1:
                    rotary_pos_emb = rotary_pos_emb[0:q_len]

                q_no_pe, q_pos_emb = torch.split(
                    q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1
                )
                k_no_pe, value = torch.split(
                    kv, [self.config.qk_head_dim, self.config.v_head_dim], dim=-1
                )

                if packed_seq_params is not None:
                    q_pos_emb = q_pos_emb.squeeze(1)
                    q_no_pe = q_no_pe.squeeze(1)
                    k_no_pe = k_no_pe.squeeze(1)
                    value = value.squeeze(1)

                q_pos_emb = apply_rotary_pos_emb(
                    q_pos_emb,
                    rotary_pos_emb,
                    config=self.config,
                    cu_seqlens=cu_seqlens_q,
                    mscale=mscale,
                    cp_group=self.pg_collection.cp,
                )
                k_pos_emb = apply_rotary_pos_emb(
                    k_pos_emb,
                    rotary_pos_emb,
                    config=self.config,
                    cu_seqlens=cu_seqlens_kv,
                    mscale=mscale,
                    cp_group=self.pg_collection.cp,
                )
                query = torch.cat([q_no_pe, q_pos_emb], dim=-1)
                if packed_seq_params is not None:
                    k_pos_emb = k_pos_emb.expand(-1, self.num_attention_heads_per_partition, -1)
                    key = torch.cat([k_no_pe, k_pos_emb], dim=-1)
                else:
                    k_pos_emb = k_pos_emb.expand(-1, -1, self.num_attention_heads_per_partition, -1)
                    key = torch.cat([k_no_pe, k_pos_emb], dim=-1)

            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()
            return query, key, value

        if self.recompute_up_proj:
            quantization = self.config.fp8 or self.config.fp4
            self.qkv_up_checkpoint = tensor_parallel.CheckpointWithoutOutput(fp8=quantization)
            query, key, value = self.qkv_up_checkpoint.checkpoint(
                qkv_up_proj_and_rope_apply, q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb
            )
        else:
            query, key, value = qkv_up_proj_and_rope_apply(
                q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb
            )

        return query, key, value, q_compressed, kv_compressed

    patch_get_query_key_value_tensors._tinker_mla_qkv_patch = True  # type: ignore[attr-defined]
    MLASelfAttention.get_query_key_value_tensors = patch_get_query_key_value_tensors
    logger.info("Applied MLA get_query_key_value_tensors patch aligned with current Megatron runtime")


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

        original_get_qkv = self.get_query_key_value_tensors
        had_instance_get_qkv = "get_query_key_value_tensors" in getattr(self, "__dict__", {})

        def compat_get_query_key_value_tensors(*a, **kw):
            result = original_get_qkv(*a, **kw)
            if not isinstance(result, tuple):
                raise RuntimeError(
                    "MLA get_query_key_value_tensors returned a non-tuple result: "
                    f"type={type(result).__name__}"
                )
            if len(result) == 5:
                return result
            if len(result) == 3:
                if getattr(self.config, "experimental_attention_variant", None) == "dsa":
                    raise RuntimeError(
                        "MLA get_query_key_value_tensors returned 3 values but DSA requires "
                        "q_compressed and kv_compressed"
                    )
                query, key, value = result
                return query, key, value, None, None
            raise RuntimeError(
                "MLA get_query_key_value_tensors returned an unexpected tuple length: "
                f"len={len(result)}"
            )

        # Temporarily replace core_attention.forward, expected head_dim, and QKV return arity
        core_attn_module.forward = padded_core_attention_forward
        self.get_query_key_value_tensors = compat_get_query_key_value_tensors
        if original_v_head_dim is not None:
            core_attn_module.hidden_size_per_attention_head_v = q_head_dim  # Change 128 -> 192
        try:
            result = original_forward(self, *args, **kwargs)
        finally:
            # Restore original forward method, head_dim, and QKV getter
            core_attn_module.forward = original_core_attn_forward
            if original_v_head_dim is not None:
                core_attn_module.hidden_size_per_attention_head_v = original_v_head_dim
            if had_instance_get_qkv:
                self.get_query_key_value_tensors = original_get_qkv
            else:
                delattr(self, "get_query_key_value_tensors")

        return result

    # Apply the patch
    MultiLatentAttention.forward = patched_forward
    print("[VERL_PATCH] Applied MLA value padding patch for FlashAttention 2 compatibility")
    logger.info("Applied MLA value padding patch (head_dim_v 128→192 padding)")
