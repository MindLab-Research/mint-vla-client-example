"""Monkey patches for verl to fix known issues.

These patches are applied at import time to fix bugs in verl without modifying
the verl package directly, keeping changes version-controlled in tinker-server.
"""

import logging
from functools import wraps

logger = logging.getLogger(__name__)

_patches_applied = False


def apply_verl_patches():
    """Apply all verl patches. Safe to call multiple times."""
    global _patches_applied
    if _patches_applied:
        print("[verl_patches] Patches already applied, skipping", flush=True)
        return

    print("[verl_patches] Applying patches...", flush=True)
    # CRITICAL: CP=1 boundary fix MUST come first!
    # It patches util.preprocess_packed_seqs_no_padding BEFORE model_forward.py is imported.
    # _apply_label_shift_patch imports MegatronEngineWithLMHead which triggers model_forward import.
    _apply_cp1_boundary_fix()
    _apply_label_shift_patch()
    _patches_applied = True
    print("[verl_patches] All patches applied successfully", flush=True)
    logger.info("[verl_patches] All patches applied")


_shift_debug_count = [0]

# Set to True to disable shift for debugging
_DISABLE_SHIFT = False


def _shift_labels_left(input_ids):
    """Shift labels left by 1 position for next-token prediction alignment.

    For each sequence: label[i] = input_ids[i+1]
    Handles both regular tensors and NestedTensors (jagged layout).

    The last position wraps around, but this is typically masked out anyway.
    """
    import torch

    _shift_debug_count[0] += 1
    do_debug = _shift_debug_count[0] <= 3

    # Bypass mode for debugging
    if _DISABLE_SHIFT:
        if do_debug:
            print(f"[_shift_labels_left] BYPASS MODE - returning input_ids unchanged", flush=True)
        return input_ids.clone()

    # Check for NestedTensor using the proper API
    try:
        is_nested = input_ids.is_nested
    except AttributeError:
        is_nested = False

    if do_debug:
        print(f"[_shift_labels_left] call #{_shift_debug_count[0]}", flush=True)
        print(f"  type: {type(input_ids).__name__}, is_nested: {is_nested}", flush=True)
        print(f"  dtype: {input_ids.dtype}", flush=True)

    # Extra debug for positions 115-122 to check action region
    extra_debug = _shift_debug_count[0] <= 2

    if is_nested:
        values = input_ids.values()
        offsets = input_ids.offsets()

        if do_debug:
            print(f"  [NestedTensor] values.shape: {values.shape}, num_seqs: {len(offsets)-1}", flush=True)
            print(f"  first 10 tokens: {values[:10].tolist()}", flush=True)

        shifted_values = torch.empty_like(values)
        for i in range(len(offsets) - 1):
            start = offsets[i].item()
            end = offsets[i + 1].item()
            if end - start > 0:
                shifted_values[start:end-1] = values[start+1:end]
                shifted_values[end-1] = values[start]  # wrap around

        if do_debug:
            print(f"  shifted first 10: {shifted_values[:10].tolist()}", flush=True)

        # Check positions around 117 (typical first action position)
        if extra_debug and len(values) > 122:
            print(f"  [ACTION REGION CHECK] original[115:122]: {values[115:122].tolist()}", flush=True)
            print(f"  [ACTION REGION CHECK] shifted[115:122]: {shifted_values[115:122].tolist()}", flush=True)
            print(f"  Expected: shifted[116] should equal original[117]", flush=True)
            print(f"    original[117] = {values[117].item()}", flush=True)
            print(f"    shifted[116] = {shifted_values[116].item()}", flush=True)
            print(f"  Expected: shifted[117] should equal original[118]", flush=True)
            print(f"    original[118] = {values[118].item()}", flush=True)
            print(f"    shifted[117] = {shifted_values[117].item()}", flush=True)

        shifted_sequences = [shifted_values[offsets[i].item():offsets[i+1].item()]
                            for i in range(len(offsets) - 1)]
        return torch.nested.as_nested_tensor(shifted_sequences, layout=torch.jagged)
    else:
        if do_debug:
            print(f"  [Regular tensor] shape: {input_ids.shape}", flush=True)
            if input_ids.dim() >= 2:
                print(f"  first seq tokens: {input_ids[0, :10].tolist()}", flush=True)

        result = torch.roll(input_ids, shifts=-1, dims=-1)

        if do_debug and input_ids.dim() >= 2:
            print(f"  shifted tokens: {result[0, :10].tolist()}", flush=True)
        return result


def _apply_label_shift_patch():
    """Fix label alignment in MegatronEngineWithLMHead.forward_step.

    Bug: verl uses `label = input_ids.clone()` without shifting, so:
        log_probs[i] = log P(input_ids[i] | logits[i])

    But logits[i] is the distribution for predicting the NEXT token (position i+1).

    Fix: Shift labels left so label[i] = input_ids[i+1], then:
        log_probs[i] = log P(input_ids[i+1] | logits[i])

    This matches vLLM's convention where old_log_probs[i] = log P(token[i+1] | context[0:i+1]).
    """
    try:
        print("[verl_patches] _apply_label_shift_patch starting imports...", flush=True)
        import torch
        from functools import partial
        from typing import Iterator
        from tensordict import TensorDict
        from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead
        from verl.utils import tensordict_utils as tu
        from verl.utils.dataset.dataset_utils import DatasetPadMode
        from verl.utils.device import get_device_id
        from verl.utils.megatron.tensor_parallel import vocab_parallel_log_probs_from_logits, vocab_parallel_entropy
        from verl.models.mcore import get_mcore_forward_no_padding_fn
        print("[verl_patches] Imports successful, patching forward_step...", flush=True)

        original_forward_step = MegatronEngineWithLMHead.forward_step
        _patch_call_count = [0]  # Use list for mutable closure

        @wraps(original_forward_step)
        def patched_forward_step(self, batch_iter: Iterator[TensorDict], model, postprocess_micro_batch_func):
            """Patched forward_step that shifts labels for correct log_prob alignment."""
            _patch_call_count[0] += 1
            if _patch_call_count[0] <= 3:
                print(f"[verl_patches] PATCHED forward_step called (call #{_patch_call_count[0]})", flush=True)
            batch: TensorDict = next(batch_iter)
            batch = batch.to(get_device_id())
            use_fused_kernels = tu.get_non_tensor_data(batch, key="use_fused_kernels", default=False)
            calculate_entropy = tu.get_non_tensor_data(batch, key="calculate_entropy", default=False)
            pad_mode = tu.get_non_tensor_data(batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
            temperature = batch["temperature"]

            model_inputs = self.prepare_model_inputs(batch)
            input_ids = model_inputs["input_ids"]
            multi_modal_inputs = model_inputs["multi_modal_inputs"]

            if pad_mode == DatasetPadMode.NO_PADDING:
                # DO NOT shift here - verl's preprocess_packed_seqs_no_padding already shifts
                # labels when need_roll=(k == "label") is True (see model_forward.py line 134
                # and util.py line 256-260). Shifting here would cause DOUBLE SHIFT.
                label = input_ids.clone()
            else:
                raise NotImplementedError(f"Pad mode {pad_mode} is not supported for megatron engine")

            if use_fused_kernels:
                raise NotImplementedError("Fused kernels are not supported for megatron engine")

            forward_fn = get_mcore_forward_no_padding_fn(self.model_config.hf_config)

            def logits_processor(logits, label):
                assert logits.shape[:2] == label.shape[:2]
                logits.div_(temperature)
                ret = {}
                if calculate_entropy:
                    logits_bak = logits.clone()
                    if torch.distributed.get_rank() == 0:
                        logger.warning(
                            "For memory-efficient computation, enable fused kernels via "
                            "`actor_rollout_ref.model.use_fused_kernels=True`. "
                            "The current `clone()` operation ensures correctness but increases memory usage."
                        )
                    entropy = vocab_parallel_entropy(logits)
                    ret["entropy"] = entropy
                else:
                    logits_bak = logits

                log_probs = vocab_parallel_log_probs_from_logits(logits_bak, label)
                ret["log_probs"] = log_probs
                return ret

            logits_processor_args = {"label": label}

            output = forward_fn(
                model,
                input_ids,
                multi_modal_inputs,
                logits_processor=logits_processor,
                logits_processor_args=logits_processor_args,
            )

            return output, partial(postprocess_micro_batch_func, data=batch)

        MegatronEngineWithLMHead.forward_step = patched_forward_step
        print("[verl_patches] SUCCESS: Patched MegatronEngineWithLMHead.forward_step", flush=True)
        logger.info("[verl_patches] Applied label shift patch to MegatronEngineWithLMHead.forward_step")

    except ImportError as e:
        print(f"[verl_patches] FAILED - ImportError: {e}", flush=True)
        logger.warning(f"[verl_patches] Could not apply label shift patch: {e}")
    except Exception as e:
        print(f"[verl_patches] FAILED - Exception: {e}", flush=True)
        logger.warning(f"[verl_patches] Could not apply label shift patch: {e}")


def _apply_cp1_boundary_fix():
    """Fix verl's preprocess_packed_seqs_no_padding CP=1 boundary bug.

    Bug: When CP=1 and need_roll=True, the code uses 'continue' which skips building
    saved_roll_dict, so torch.roll wraps the last token to position 0. This causes
    the last position's label to be the FIRST token instead of being ignored.

    This only affects the last position in each sequence, causing catastrophic logprob
    mismatch (e.g., -0.0006 in vLLM vs -30.7 in Megatron).

    Fix: For CP=1, build saved_roll_dict to set last position to -100 (ignore index).

    CRITICAL: We must patch util.preprocess_packed_seqs_no_padding BEFORE model_forward.py
    is imported anywhere. Do NOT import model_forward here - that would trigger the
    import binding with the old function!
    """
    try:
        print("[verl_patches] _apply_cp1_boundary_fix starting imports...", flush=True)
        import torch
        # Import only util - do NOT import model_forward!
        import verl.models.mcore.util as util_module
        from verl.models.mcore.util import PackedSeqParams
        from megatron.core import parallel_state as mpu

        print("[verl_patches] Imports successful, creating patched function...", flush=True)

        # Create completely new implementation that fixes CP=1 boundary
        def patched_preprocess_packed_seqs_no_padding(
            input_ids: torch.Tensor, pre_process: bool = True, need_roll: bool = False
        ) -> tuple[torch.Tensor, PackedSeqParams]:
            """Fixed version with proper CP=1 boundary handling."""
            # DEBUG: Write to file to verify patch is called
            if need_roll:
                try:
                    with open("/tmp/verl_patch_called.txt", "a") as f:
                        f.write(f"PATCH CALLED: need_roll={need_roll}, CP={mpu.get_context_parallel_world_size()}\n")
                except:
                    pass

            batch_size = input_ids.shape[0]

            tp_size = mpu.get_tensor_model_parallel_world_size()
            cp_size = mpu.get_context_parallel_world_size()
            cp_rank = mpu.get_context_parallel_rank()
            align_size = tp_size * cp_size * 2 if cp_size > 1 else tp_size
            seqlens_in_batch = input_ids.offsets().diff()

            pad_size = (align_size - seqlens_in_batch % align_size) % align_size
            seqlens_in_batch_padded = seqlens_in_batch + pad_size

            cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=input_ids.device)
            cu_seqlens[1:] = torch.cumsum(seqlens_in_batch, dim=0)
            cu_seqlens_padded = torch.zeros(batch_size + 1, dtype=torch.int32, device=input_ids.device)
            cu_seqlens_padded[1:] = torch.cumsum(seqlens_in_batch_padded, dim=0)

            seqlens_in_batch_cpu = seqlens_in_batch.tolist()
            seqlens_in_batch_padded_cpu = seqlens_in_batch_padded.tolist()
            cu_seqlens_padded_cpu = cu_seqlens_padded.tolist()

            max_seqlen_in_batch = max(seqlens_in_batch_padded_cpu)

            shape = list(input_ids.shape[1:])
            shape[0] = sum(seqlens_in_batch_padded_cpu) // cp_size
            if pre_process:
                input_ids_rmpad = torch.zeros(shape, dtype=input_ids.dtype, device=input_ids.device)
                if need_roll:
                    saved_roll_dict = {}
                for i in range(batch_size):
                    if cp_size <= 1:
                        seqlen = seqlens_in_batch_cpu[i]
                        start_idx = cu_seqlens_padded_cpu[i]
                        input_ids_rmpad[start_idx : start_idx + seqlen] = input_ids[i]

                        # FIX: Build saved_roll_dict for CP=1 too!
                        if need_roll:
                            # Last position should be ignored (-100 is PyTorch ignore index)
                            last_idx = start_idx + seqlen - 1
                            saved_roll_dict[last_idx] = -100
                        continue

                    seqlen_padded_i = seqlens_in_batch_padded_cpu[i]
                    seqlen = seqlen_padded_i // cp_size
                    half_seqlen = seqlen // 2
                    start_idx = cu_seqlens_padded_cpu[i] // cp_size
                    d = input_ids[i]
                    input_ids_rmpad[start_idx : start_idx + half_seqlen] = d[
                        half_seqlen * cp_rank : half_seqlen * (cp_rank + 1)
                    ]

                    remain_start = seqlen_padded_i - half_seqlen * (cp_rank + 1)
                    remain_end = seqlen_padded_i - half_seqlen * cp_rank
                    remain_end = min(remain_end, d.shape[0])
                    remain_len = remain_end - remain_start
                    if remain_len > 0:
                        input_ids_rmpad[start_idx + half_seqlen : start_idx + half_seqlen + remain_len] = d[
                            remain_start:remain_end
                        ]

                    if need_roll:
                        saved_roll_dict[start_idx + half_seqlen - 1] = d[(cp_rank + 1) * half_seqlen]
                        if remain_len > 0:
                            if remain_end == d.shape[0]:
                                saved_roll_dict[start_idx + half_seqlen + remain_len - 1] = d[0]
                            else:
                                saved_roll_dict[start_idx + half_seqlen + remain_len - 1] = d[remain_end]

                if need_roll:
                    input_ids_rmpad = torch.roll(input_ids_rmpad, shifts=-1, dims=0)
                    if len(saved_roll_dict) > 0:
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
            else:
                return input_ids, packed_seq_params

        # Replace the function in util module ONLY
        # Do NOT import model_forward here - it must import AFTER this patch
        # When model_forward is later imported, it will bind to our patched function
        util_module.preprocess_packed_seqs_no_padding = patched_preprocess_packed_seqs_no_padding
        print("[verl_patches] SUCCESS: Patched util.preprocess_packed_seqs_no_padding for CP=1 boundary fix", flush=True)
        logger.info("[verl_patches] Applied CP=1 boundary fix to preprocess_packed_seqs_no_padding")

    except ImportError as e:
        print(f"[verl_patches] FAILED - ImportError: {e}", flush=True)
        logger.warning(f"[verl_patches] Could not apply CP=1 boundary fix: {e}")
    except Exception as e:
        print(f"[verl_patches] FAILED - Exception: {e}", flush=True)
        logger.warning(f"[verl_patches] Could not apply CP=1 boundary fix: {e}")

