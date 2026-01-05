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
        return

    _apply_label_shift_patch()
    _patches_applied = True
    logger.info("[verl_patches] All patches applied")


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
        import torch
        from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead

        original_forward_step = MegatronEngineWithLMHead.forward_step

        @wraps(original_forward_step)
        def patched_forward_step(self, model, batch_iter, loss_function, forward_only):
            """Patched forward_step that shifts labels for correct log_prob alignment."""
            import verl.workers.engine.megatron.tensordict_utils as tu
            from verl.workers.engine.megatron.transformer_impl import DatasetPadMode, get_device_id
            from verl.utils.megatron.tensor_parallel import vocab_parallel_log_probs_from_logits, vocab_parallel_entropy
            from functools import partial

            # Get batch and model inputs (same as original)
            batch = next(batch_iter)
            batch = batch.to(get_device_id())
            use_fused_kernels = tu.get_non_tensor_data(batch, key="use_fused_kernels", default=False)
            calculate_entropy = tu.get_non_tensor_data(batch, key="calculate_entropy", default=False)
            pad_mode = tu.get_non_tensor_data(batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
            temperature = batch["temperature"]

            model_inputs = self.prepare_model_inputs(batch)
            input_ids = model_inputs["input_ids"]
            multi_modal_inputs = model_inputs["multi_modal_inputs"]

            if pad_mode == DatasetPadMode.NO_PADDING:
                # PATCH: Shift labels left to align with next-token prediction
                # logits[i] predicts token at position i+1, so label[i] should be input_ids[i+1]
                label = torch.roll(input_ids, shifts=-1, dims=-1)
            else:
                raise NotImplementedError(f"Pad mode {pad_mode} is not supported for megatron engine")

            from verl.models.mcore import get_mcore_forward_no_padding_fn

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
                        logger.warning_once(
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

            return output, partial(self.postprocess_micro_batch_func, data=batch)

        MegatronEngineWithLMHead.forward_step = patched_forward_step
        logger.info("[verl_patches] Applied label shift patch to MegatronEngineWithLMHead.forward_step")

    except ImportError as e:
        logger.warning(f"[verl_patches] Could not apply label shift patch: {e}")
