"""Shared Megatron training utilities used by the distributed Megatron backend.

This module provides:
- Tinker Datum -> verl TensorDict conversion
- Loss functions (SFT/PPO/logprobs/reverse-KL)
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.distributed
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData

from verl.utils import tensordict_utils as tu

if TYPE_CHECKING:
    pass

from mint_server.config import config as server_config
from mint_server.model_input_utils import flatten_encoded_text_chunks

logger = logging.getLogger(__name__)


def tinker_to_tensordict(
    data_items: list[dict],
    max_token_len_per_gpu: int = 10240,  # Single sample max: prompt (~2K) + max_tokens (8K)
    device: str | torch.device | None = None,
    dp_size: int | None = None,
) -> TensorDict:
    """Convert Tinker Datum format to verl TensorDict.

    Tinker format:
        {
            "model_input": {"chunks": [{"tokens": [...]}]},
            "loss_fn_inputs": {
                "target_tokens": {"data": [...]},
                "weights": {"data": [...]},       # or "loss_mask"/"mask" used in some test cases examples
                "logprobs": {"data": [...]},      # for RL
                "advantages": {"data": [...]}     # for RL
            }
        }

    verl TensorDict format:
        TensorDict({
            "input_ids": tensor [batch, seq_len],
            "attention_mask": tensor [batch, seq_len],  # 1 for real tokens, 0 for padding
            "loss_mask": tensor [batch, seq_len],
            "attention_mask": tensor [batch, seq_len],
            "position_ids": tensor [batch, seq_len],  # 0 to seq_len-1
            "temperature": tensor [batch, 1],         # logits scaling (1.0 = no scaling)
            "old_log_probs": tensor [batch, response_len],  # for RL
            "advantages": tensor [batch, response_len],     # for RL
            "prompts": tensor [batch, prompt_len],  # derived from loss_mask
            "responses": tensor [batch, response_len],  # derived from loss_mask
            "response_mask": tensor [batch, response_len],
            # Non-tensor metadata for verl's prepare_micro_batches:
            "use_dynamic_bsz": NonTensorData(False),  # Disabled for micro_batch control
            "micro_batch_size_per_gpu": NonTensorData(1),  # Gradient accumulation
            "max_token_len_per_gpu": NonTensorData(int),
        })

    Args:
        data_items: List of Tinker Datum dicts.
        max_token_len_per_gpu: Max tokens per GPU for dynamic batch sizing.
            Used by verl's prepare_micro_batches when use_dynamic_bsz=True.
        device: Target device for tensors. If None, uses CPU.
            Creating tensors directly on GPU avoids CPU-to-GPU copy issues with nested tensors.
        dp_size: Optional data-parallel size override for loss normalization.
    """
    def _extract_list(field_name: str, field: dict | list | None, item_index: int) -> list | None:
        if field is None:
            return None
        if isinstance(field, list):
            return field
        if not isinstance(field, dict):
            raise ValueError(f"Item {item_index}: {field_name} must be a dict with 'data'")
        if "data" not in field:
            return None
        data = field["data"]
        if data is None:
            raise ValueError(f"Item {item_index}: {field_name}.data is None")
        if not isinstance(data, list):
            raise ValueError(f"Item {item_index}: {field_name}.data must be a list")
        return data

    def _ensure_len(field_name: str, data: list, tokens_len: int, item_index: int) -> None:
        if len(data) != tokens_len:
            raise ValueError(
                f"Item {item_index}: {field_name} length {len(data)} != tokens length {tokens_len}"
            )

    input_ids_list = []
    loss_mask_list = []
    prompt_ids_list = []
    response_ids_list = []
    response_mask_list = []
    response_log_probs_list = []
    response_advantages_list = []
    response_lens = []
    target_tokens_list = []  # External labels (correctly shifted, with true last token)
    routed_experts_list = []
    has_routed_experts = False

    max_len = 0
    has_external_labels = False
    has_full_external_labels = True

    # First pass: collect data and find max length
    for item_index, item in enumerate(data_items):
        model_input = item.get("model_input", {})
        loss_fn_inputs = item.get("loss_fn_inputs", {})

        # Extract input tokens
        tokens = flatten_encoded_text_chunks(model_input)
        if not tokens:
            raise ValueError(f"Item {item_index}: model_input has no tokens")

        tokens_len = len(tokens)
        max_len = max(max_len, tokens_len)
        input_ids_list.append(tokens)

        # RL inputs (optional)
        logprobs = _extract_list("logprobs", loss_fn_inputs.get("logprobs"), item_index)
        if logprobs is not None:
            _ensure_len("logprobs", logprobs, tokens_len, item_index)

        advantages = _extract_list("advantages", loss_fn_inputs.get("advantages"), item_index)
        if advantages is not None:
            _ensure_len("advantages", advantages, tokens_len, item_index)

        # Extract weights (loss mask). Require explicit weights to avoid silent semantics changes.
        weights_data = None
        weights_key = None
        for key in ("loss_mask", "mask", "weights"):
            candidate = _extract_list(key, loss_fn_inputs.get(key), item_index)
            if candidate is not None:
                weights_data = candidate
                weights_key = key
                break
        if weights_data is None:
            raise ValueError(f"Item {item_index}: missing loss_mask/mask/weights")
        _ensure_len(weights_key, weights_data, tokens_len, item_index)
        weights = weights_data
        loss_mask_list.append(weights)

        # Derive prompt/response split from loss_mask (response assumed to be suffix).
        # loss_mask aligns to target tokens (shifted); use non-zero span to handle gaps/negative weights.
        nonzero_indices = [i for i, w in enumerate(weights) if w != 0]
        if not nonzero_indices:
            prompt_len = tokens_len
            response_len = 0
            first_response_idx = None
        else:
            first_response_idx = nonzero_indices[0]
            last_response_idx = nonzero_indices[-1]
            prompt_len = min(first_response_idx + 1, tokens_len)
            max_response_len = max(tokens_len - prompt_len, 0)
            response_len = min(last_response_idx - first_response_idx + 1, max_response_len)
        response_lens.append(response_len)
        prompt_ids_list.append(tokens[:prompt_len])
        response_ids_list.append(tokens[prompt_len : prompt_len + response_len])
        if response_len > 0:
            slice_start = max(prompt_len - 1, 0)
            slice_end = min(slice_start + response_len, tokens_len)
            response_mask_list.append([1.0 if w != 0 else 0.0 for w in weights[slice_start:slice_end]])
            if logprobs is not None:
                response_log_probs_list.append(logprobs[slice_start:slice_end])
            else:
                response_log_probs_list.append(None)
            if advantages is not None:
                response_advantages_list.append(advantages[slice_start:slice_end])
            else:
                response_advantages_list.append(None)
        else:
            response_mask_list.append([])
            response_log_probs_list.append([] if logprobs is not None else None)
            response_advantages_list.append([] if advantages is not None else None)

        # External labels (target_tokens) - correctly shifted with true last token
        # This solves the verl roll bug where last position gets wrapped first token
        target_tokens = _extract_list("target_tokens", loss_fn_inputs.get("target_tokens"), item_index)
        if target_tokens is not None:
            _ensure_len("target_tokens", target_tokens, tokens_len, item_index)
            has_external_labels = True
            target_tokens_list.append(target_tokens)
            logger.debug(
                "[tinker_to_tensordict] Extracted target_tokens, len=%d",
                len(target_tokens),
            )
        else:
            has_full_external_labels = False
            target_tokens_list.append(None)

        # Router replay (R3): per-token routed expert indices
        re_field = loss_fn_inputs.get("routed_experts")
        if re_field is not None:
            # re_field may be either:
            #   - nested list [seq_len][layer_num][topk] (from SampledSequence.routed_experts directly)
            #   - Tinker wire format dict {"data": flat_list, "shape": [S, L, K], "dtype": "int64"}
            # vLLM only returns routed_experts for decode tokens (not prefill/prompt).
            # Prepend zeros for the prompt tokens so shape[0] == tokens_len.
            if isinstance(re_field, dict):
                flat = re_field["data"]
                shape = re_field["shape"]
                if len(shape) != 3:
                    raise ValueError(f"Item {item_index}: routed_experts shape must be [S, L, K], got {shape}")
                re_tensor = torch.tensor(flat, dtype=torch.int64, device=device).reshape(shape)
            else:
                re_tensor = torch.tensor(re_field, dtype=torch.int64, device=device)
            if re_tensor.ndim != 3:
                raise ValueError(f"Item {item_index}: routed_experts must be shape [seq_len, layer_num, topk], got {list(re_tensor.shape)}")
            if re_tensor.shape[0] < tokens_len:
                pad_len = tokens_len - re_tensor.shape[0]
                padding = torch.zeros((pad_len, re_tensor.shape[1], re_tensor.shape[2]), dtype=torch.int64, device=device)
                re_tensor = torch.cat([padding, re_tensor], dim=0)
            if re_tensor.shape[0] != tokens_len:
                raise ValueError(f"Item {item_index}: routed_experts seq_len {re_tensor.shape[0]} != tokens length {tokens_len}")
            routed_experts_list.append(re_tensor)
            has_routed_experts = True
        else:
            routed_experts_list.append(None)

    if not input_ids_list:
        raise ValueError("No valid data items found")

    batch_size = len(input_ids_list)

    # verl expects nested/jagged tensors for variable-length sequences
    # This is required for gptmodel_forward_no_padding which calls input_ids.offsets()
    input_ids_tensors = [torch.tensor(seq, dtype=torch.long, device=device) for seq in input_ids_list]
    loss_mask_tensors = [torch.tensor(seq, dtype=torch.float, device=device) for seq in loss_mask_list]
    position_ids_tensors = [torch.arange(len(seq), dtype=torch.long, device=device) for seq in input_ids_list]

    # Create NestedTensors with jagged layout (variable-length sequences)
    input_ids = torch.nested.as_nested_tensor(input_ids_tensors, layout=torch.jagged)
    loss_mask = torch.nested.as_nested_tensor(loss_mask_tensors, layout=torch.jagged)
    position_ids = torch.nested.as_nested_tensor(position_ids_tensors, layout=torch.jagged)

    # TensorDict with NestedTensors
    td = TensorDict({
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
    }, batch_size=[batch_size])

    # Temperature for logits scaling (1.0 = no scaling during training)
    # verl's forward_step does logits.div_(batch["temperature"])
    # Must be a scalar float to broadcast correctly with logits shape [?, ?, vocab_size]
    # Using set_non_tensor with float prevents batching/slicing by prepare_micro_batches
    td.set_non_tensor("temperature", 1.0)

    # Add prompt/response splits for verl's PPO loss (slicing log_probs by response length).
    max_prompt_len = max((len(seq) for seq in prompt_ids_list), default=0)
    max_response_len = max(response_lens) if response_lens else 0
    prompt_tensor = torch.zeros((batch_size, max_prompt_len), dtype=torch.long, device=device)
    response_tensor = torch.zeros((batch_size, max_response_len), dtype=torch.long, device=device)
    for idx, (prompt_seq, response_seq) in enumerate(
        zip(prompt_ids_list, response_ids_list, strict=True)
    ):
        if prompt_seq:
            prompt_tensor[idx, :len(prompt_seq)] = torch.tensor(prompt_seq, dtype=torch.long, device=device)
        if response_seq:
            response_tensor[idx, :len(response_seq)] = torch.tensor(response_seq, dtype=torch.long, device=device)
    td["prompts"] = prompt_tensor
    td["responses"] = response_tensor

    response_mask = torch.zeros((batch_size, max_response_len), dtype=torch.float, device=device)
    for idx, mask_seq in enumerate(response_mask_list):
        if mask_seq:
            response_mask[idx, :len(mask_seq)] = torch.tensor(mask_seq, dtype=torch.float, device=device)
    td["response_mask"] = response_mask

    # Dense attention_mask for verl's response slicing (prompt padded to max_prompt_len).
    attention_mask = torch.zeros(
        (batch_size, max_prompt_len + max_response_len), dtype=torch.long, device=device
    )
    for idx, (prompt_len, response_len) in enumerate(zip(
        [len(seq) for seq in prompt_ids_list], response_lens, strict=True
    )):
        if prompt_len:
            attention_mask[idx, :prompt_len] = 1
        if response_len:
            attention_mask[idx, max_prompt_len : max_prompt_len + response_len] = 1
    td["attention_mask"] = attention_mask

    # Add RL inputs if present (response-aligned dense tensors).
    has_full_rl_inputs = all(lp is not None for lp in response_log_probs_list) and all(
        adv is not None for adv in response_advantages_list
    )
    if has_full_rl_inputs:
        old_log_probs = torch.zeros((batch_size, max_response_len), dtype=torch.float, device=device)
        advantages = torch.zeros((batch_size, max_response_len), dtype=torch.float, device=device)
        for idx, (lp_seq, adv_seq) in enumerate(
            zip(response_log_probs_list, response_advantages_list, strict=True)
        ):
            if lp_seq:
                old_log_probs[idx, :len(lp_seq)] = torch.tensor(lp_seq, dtype=torch.float, device=device)
            if adv_seq:
                advantages[idx, :len(adv_seq)] = torch.tensor(adv_seq, dtype=torch.float, device=device)
        td["old_log_probs"] = old_log_probs
        td["advantages"] = advantages

    # Add external labels if present (target_tokens with correct last token)
    # Key MUST NOT be "label" - verl applies torch.roll when key == "label"
    # Using "target" bypasses roll since need_roll=(k == "label") in model_forward.py
    disable_external_label = os.environ.get("MINT_DISABLE_EXTERNAL_LABEL", "0") == "1"
    if has_external_labels and target_tokens_list and not disable_external_label:
        if has_full_external_labels and all(seq is not None for seq in target_tokens_list):
            target_tokens_tensors = [torch.tensor(seq, dtype=torch.long, device=device) for seq in target_tokens_list]
            td["target"] = torch.nested.as_nested_tensor(target_tokens_tensors, layout=torch.jagged)
            td.set_non_tensor("use_external_label", True)
            logger.debug(
                "[tinker_to_tensordict] Added external labels (key='target', no roll), batch_size=%d",
                len(target_tokens_list),
            )
        else:
            logger.warning(
                "[tinker_to_tensordict] Mixed target_tokens presence in batch; "
                "skipping external labels to avoid TensorDict shape mismatch."
            )
    elif has_external_labels and disable_external_label:
        logger.warning("[tinker_to_tensordict] MINT_DISABLE_EXTERNAL_LABEL=1; forcing original rolled labels")

    require_r3 = (server_config.router_replay_mode == "R3")
    if require_r3:
        missing = [i for i, re in enumerate(routed_experts_list) if re is None]
        if missing:
            raise ValueError(
                "router_replay_mode=R3 requires routed_experts for every datum "
                f"(missing item indexes: {missing})"
            )

    if has_routed_experts or require_r3:
        if all(re is not None for re in routed_experts_list):
            re_nested = torch.nested.as_nested_tensor(routed_experts_list, layout=torch.jagged)
            td["routed_experts"] = re_nested
            td.set_non_tensor("enable_routing_replay", True)
        elif has_routed_experts:
            logger.warning(
                "[tinker_to_tensordict] Mixed routed_experts presence in batch; skipping R3"
            )

    # Add non-tensor metadata for verl's prepare_micro_batches
    # use_dynamic_bsz=True is REQUIRED for NestedTensor compatibility with verl's forward
    # verl's rearrange_micro_batches handles dynamic batching based on token count
    td["use_dynamic_bsz"] = NonTensorData(True)
    td["max_token_len_per_gpu"] = NonTensorData(max_token_len_per_gpu)
    td.set_non_tensor("sp_size", 1)  # Sequence parallel size (default 1)

    # Add fields required by verl's sft_loss
    # Compute total tokens in batch for loss normalization
    # Use set_non_tensor to avoid batch dimension validation (these are scalar values)
    if has_full_rl_inputs and response_mask_list:
        local_total_tokens = sum(float(sum(seq)) for seq in response_mask_list)
    else:
        local_total_tokens = sum(float(sum(seq)) for seq in loss_mask_list)

    dp_group = None
    if dp_size is None:
        if torch.distributed.is_initialized():
            try:
                from megatron.core import mpu

                dp_size = int(mpu.get_data_parallel_world_size())
                dp_group = mpu.get_data_parallel_group()
            except Exception:
                dp_size = int(torch.distributed.get_world_size())
        else:
            dp_size = 1

    global_total_tokens = local_total_tokens
    global_batch_size = batch_size
    if dp_size > 1 and torch.distributed.is_initialized():
        reduce_device = device
        if reduce_device is None:
            try:
                backend = torch.distributed.get_backend(dp_group)
            except Exception:
                backend = None
            if backend == "nccl" and torch.cuda.is_available():
                reduce_device = torch.device("cuda")
            else:
                reduce_device = torch.device("cpu")
        reduce_tensor = torch.tensor(
            [local_total_tokens, float(batch_size)],
            dtype=torch.float64,
            device=reduce_device,
        )
        torch.distributed.all_reduce(reduce_tensor, op=torch.distributed.ReduceOp.SUM, group=dp_group)
        global_total_tokens = float(reduce_tensor[0].item())
        global_batch_size = int(reduce_tensor[1].item())
    elif dp_size > 1:
        global_total_tokens = local_total_tokens * dp_size
        global_batch_size = batch_size * dp_size

    td.set_non_tensor("dp_size", int(dp_size))
    td.set_non_tensor("batch_num_tokens", global_total_tokens)
    td.set_non_tensor("global_batch_size", int(global_batch_size))

    return td


def create_sft_loss_fn(return_logprobs: bool = True) -> Callable:
    """Create SFT loss function that also extracts per-token log_probs.

    verl's postprocess_micro_batch_func calls:
        loss, metrics = loss_function(model_output=..., data=..., dp_group=...)

    Args:
        return_logprobs: If True, include log_probs in metrics (for cookbook train_nll)

    Returns:
        Loss function compatible with verl's forward_backward_batch.
    """
    def _flatten_rows(tensor: torch.Tensor) -> torch.Tensor:
        if getattr(tensor, "is_nested", False):
            rows = [row for row in tensor.unbind()]
            if not rows:
                return torch.empty((0,), dtype=tensor.dtype, device=tensor.device)
            return torch.cat(rows, dim=0)
        return tensor

    def sft_loss_with_logprobs(model_output: dict, data: TensorDict, dp_group=None) -> tuple:
        """SFT cross-entropy loss that also returns log_probs for metrics.

        Args:
            model_output: dict with "log_probs" key containing log probabilities
            data: TensorDict with loss_mask, dp_size, batch_num_tokens
            dp_group: data parallel group (unused)

        Returns:
            Tuple of (loss_tensor, metrics_dict_with_logprobs)
        """
        packed_present = "packed_log_probs" in model_output
        log_probs = model_output.get("packed_log_probs")
        if log_probs is None:
            log_probs = model_output.get("log_probs")

        if log_probs is None:
            raise ValueError("model_output missing required log_probs")

        # Prefer the packed non-nested tensor emitted by the patched Megatron
        # forward step. Falling back to nested .values() keeps older paths alive,
        # but the packed tensor avoids nested autograd edges on the SFT path.
        log_probs_flat = _flatten_rows(log_probs)

        # Get loss_mask to identify which tokens contribute to loss
        loss_mask = data.get("loss_mask")
        if loss_mask is not None:
            loss_mask_flat = _flatten_rows(loss_mask)
        else:
            raise ValueError("data missing required loss_mask")

        logger.info(
            "[sft_loss_with_logprobs] packed_present=%s log_probs_type=%s log_probs_flat_type=%s "
            "loss_mask_type=%s loss_mask_flat_type=%s",
            packed_present,
            type(log_probs).__name__,
            type(log_probs_flat).__name__,
            type(loss_mask).__name__ if loss_mask is not None else None,
            type(loss_mask_flat).__name__,
        )
        if os.environ.get("MINT_SFT_DIAG_FAIL", "0") == "1":
            raise RuntimeError(
                "SFT_DIAG "
                f"packed_present={packed_present} "
                f"log_probs_type={type(log_probs).__name__} "
                f"log_probs_is_nested={getattr(log_probs, 'is_nested', False)} "
                f"log_probs_flat_type={type(log_probs_flat).__name__} "
                f"log_probs_flat_is_nested={getattr(log_probs_flat, 'is_nested', False)} "
                f"loss_mask_type={type(loss_mask).__name__ if loss_mask is not None else None} "
                f"loss_mask_is_nested={getattr(loss_mask, 'is_nested', False) if loss_mask is not None else None} "
                f"loss_mask_flat_type={type(loss_mask_flat).__name__} "
                f"loss_mask_flat_is_nested={getattr(loss_mask_flat, 'is_nested', False)}"
            )

        use_external_label = tu.get_non_tensor_data(data, key="use_external_label", default=False)
        if not use_external_label:
            # Align mask with rolled labels when external labels are not used.
            loss_mask_flat = torch.roll(loss_mask_flat, shifts=-1, dims=0)

        # Ensure types are compatible
        loss_mask_float = loss_mask_flat.float()

        # Compute SFT loss: -sum(log_probs * mask) / sum(mask)
        # log_probs are already the target token log probs from verl's forward
        weighted_log_probs = log_probs_flat * loss_mask_float
        num_tokens = loss_mask_float.sum()

        dp_size = tu.get_non_tensor_data(data, key="dp_size", default=1)
        batch_num_tokens = tu.get_non_tensor_data(data, key="batch_num_tokens", default=None)
        if batch_num_tokens is None:
            batch_num_tokens = num_tokens

        if hasattr(batch_num_tokens, "item"):
            batch_num_tokens_value = batch_num_tokens.item()
        else:
            batch_num_tokens_value = float(batch_num_tokens)

        loss_sum = -weighted_log_probs.sum()
        if batch_num_tokens_value > 0:
            nll = loss_sum / batch_num_tokens_value * dp_size
        else:
            nll = loss_sum

        # Clone log_probs for metrics (detach to avoid affecting gradients)
        log_probs_cpu = log_probs_flat.detach().cpu()

        metrics = {
            "loss": nll.detach().item() if hasattr(nll, "item") else float(nll),
            "loss_sum": loss_sum.detach().item() if hasattr(loss_sum, "item") else float(loss_sum),
            "num_tokens": int(num_tokens.item()) if hasattr(num_tokens, 'item') else int(num_tokens),
        }

        if return_logprobs:
            metrics["log_probs"] = log_probs_cpu

        return nll, metrics

    return sft_loss_with_logprobs


def create_loss_fn() -> Callable:
    """Create default loss function for warmup/initialization.

    Returns SFT loss function as the default for forward-backward warmup.
    """
    return create_sft_loss_fn(return_logprobs=False)


def create_ppo_loss_fn(
    epsilon: float = 0.2,
    rollout_correction_config: dict[str, Any] | None = None,
) -> Callable:
    """Create PPO loss function by reusing verl's ppo_loss/agg_loss path."""
    import math
    import torch.nn.functional as F
    from verl.trainer.config import RolloutCorrectionConfig
    from verl.trainer.ppo.core_algos import agg_loss
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_rejection_mask
    from verl.workers.config import ActorConfig, PolicyLossConfig

    clip_ratio = float(epsilon)
    clip_ratio_c = 1e6 if math.isinf(clip_ratio) else 3.0
    actor_config_kwargs: dict[str, Any] = dict(
        strategy="tinker",
        rollout_n=1,
        use_dynamic_bsz=True,
        clip_ratio=clip_ratio,
        clip_ratio_low=clip_ratio,
        clip_ratio_high=clip_ratio,
        clip_ratio_c=clip_ratio_c,
    )

    if isinstance(rollout_correction_config, dict):
        # Keep only explicitly configured values so verl defaults can apply.
        rollout_corr_clean = {k: v for k, v in rollout_correction_config.items() if v is not None}
        bypass_mode = bool(rollout_corr_clean.get("bypass_mode", False))
        actor_config_kwargs["policy_loss"] = PolicyLossConfig(
            loss_mode="bypass_mode" if bypass_mode else "vanilla",
            rollout_correction=RolloutCorrectionConfig(**rollout_corr_clean),
        )

    actor_config = ActorConfig(**actor_config_kwargs)

    def _compute_vanilla_pg_losses(
        *,
        old_log_prob: torch.Tensor,
        log_prob: torch.Tensor,
        advantages: torch.Tensor,
        rollout_is_weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        negative_approx_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
        ratio = torch.exp(negative_approx_kl)

        clip_ratio_low = actor_config.clip_ratio_low if actor_config.clip_ratio_low is not None else actor_config.clip_ratio
        clip_ratio_high = actor_config.clip_ratio_high if actor_config.clip_ratio_high is not None else actor_config.clip_ratio
        clip_ratio_c_local = actor_config.get("clip_ratio_c", 3.0)

        pg_losses1 = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
        clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
        pg_losses3 = -advantages * clip_ratio_c_local
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
        if rollout_is_weights is not None:
            pg_losses = pg_losses * rollout_is_weights
        return pg_losses, ratio

    def _compute_bypass_pg_losses(
        *,
        rollout_log_prob: torch.Tensor,
        log_prob: torch.Tensor,
        advantages: torch.Tensor,
        response_mask_bool: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rollout_corr_config = (
            actor_config.policy_loss.get("rollout_correction", None) if hasattr(actor_config, "policy_loss") else None
        )
        if rollout_corr_config is None:
            raise ValueError(
                "rollout_correction config not found in policy_loss. "
                "When using bypass_mode, the rollout_correction config must be present."
            )

        with torch.no_grad():
            rollout_is_weights_proto, modified_response_mask, _ = compute_rollout_correction_and_rejection_mask(
                old_log_prob=log_prob,
                rollout_log_prob=rollout_log_prob,
                response_mask=response_mask_bool,
                rollout_is=rollout_corr_config.get("rollout_is", None),
                rollout_is_threshold=rollout_corr_config.get("rollout_is_threshold", 2.0),
                rollout_rs=rollout_corr_config.get("rollout_rs", None),
                rollout_rs_threshold=rollout_corr_config.get("rollout_rs_threshold", None),
                rollout_is_batch_normalize=rollout_corr_config.get("rollout_is_batch_normalize", False),
            )

        effective_mask = modified_response_mask.to(bool)
        ratio = torch.exp(torch.clamp(log_prob - rollout_log_prob, min=-20.0, max=20.0))
        loss_type = rollout_corr_config.get("loss_type", "ppo_clip")

        if loss_type == "reinforce":
            rollout_is_weights = rollout_is_weights_proto.batch["rollout_is_weights"] if rollout_is_weights_proto else None
            pg_losses = -advantages * log_prob
            if rollout_is_weights is not None:
                pg_losses = pg_losses * rollout_is_weights
            return pg_losses, effective_mask, ratio

        if loss_type == "ppo_clip":
            pg_losses, _ = _compute_vanilla_pg_losses(
                old_log_prob=rollout_log_prob,
                log_prob=log_prob,
                advantages=advantages,
                rollout_is_weights=None,
            )
            return pg_losses, effective_mask, ratio

        raise ValueError(f"Invalid bypass_mode loss_type: {loss_type!r}")

    def _slice_response_log_probs(tensor: torch.Tensor, data: TensorDict) -> torch.Tensor:
        prompt_ids = data["prompts"]
        response_ids = data["responses"]
        attention_mask = data["attention_mask"]

        if prompt_ids.is_nested:
            prompt_lens = prompt_ids.offsets().diff()
            response_lens = response_ids.offsets().diff()
            max_response_len = int(response_lens.max().item()) if response_lens.numel() else 0
        else:
            assert not attention_mask.is_nested
            prompt_lens = attention_mask[:, : prompt_ids.shape[1]].sum(dim=1)
            response_lens = attention_mask[:, prompt_ids.shape[1] :].sum(dim=1)
            max_response_len = int(response_ids.shape[1])

        rows = []
        if tensor.is_nested:
            for row_idx, (prompt_len, response_len) in enumerate(zip(prompt_lens, response_lens, strict=True)):
                prompt_len_i = int(prompt_len.item()) if hasattr(prompt_len, "item") else int(prompt_len)
                response_len_i = int(response_len.item()) if hasattr(response_len, "item") else int(response_len)
                start = max(prompt_len_i - 1, 0)
                end = start + response_len_i
                row = tensor[row_idx][start:end]
                pad_size = max_response_len - response_len_i
                if pad_size > 0:
                    row = F.pad(row, (0, pad_size))
                rows.append(row)
        else:
            values = tensor
            sequence_lens = prompt_lens + response_lens
            sequence_offsets = sequence_lens.cumsum(dim=0)
            assert sequence_offsets[-1].item() == values.shape[0]
            for response_len, seq_offset in zip(response_lens, sequence_offsets, strict=True):
                response_len_i = int(response_len.item()) if hasattr(response_len, "item") else int(response_len)
                seq_offset_i = int(seq_offset.item()) if hasattr(seq_offset, "item") else int(seq_offset)
                pad_size = max_response_len - response_len_i
                row = values[seq_offset_i - response_len_i - 1 : seq_offset_i - 1]
                if pad_size > 0:
                    row = F.pad(row, (0, pad_size))
                rows.append(row)

        if not rows:
            return torch.empty((0, 0), dtype=tensor.dtype, device=tensor.device)
        return torch.stack(rows, dim=0)

    def ppo_loss_fn(model_output: dict, data: TensorDict, dp_group=None) -> tuple:
        """PPO clipped objective loss via verl's implementation.

        Returns:
            Tuple of (loss_tensor, metrics_dict)
        """
        response_log_probs = model_output.get("response_log_probs")
        if response_log_probs is None:
            response_log_probs = _slice_response_log_probs(model_output["log_probs"], data)
        target_len = response_log_probs.shape[1] if response_log_probs.dim() > 1 else response_log_probs.numel()

        old_log_probs = data["old_log_probs"]
        advantages = data["advantages"]
        response_mask = data["response_mask"].float()

        def _pad_or_trunc(tensor: torch.Tensor, length: int, pad_value: float = 0.0) -> torch.Tensor:
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            current = tensor.shape[1]
            if current == length:
                return tensor
            if current > length:
                return tensor[:, :length]
            pad_size = length - current
            return F.pad(tensor, (0, pad_size), value=pad_value)

        old_log_probs = _pad_or_trunc(old_log_probs, target_len, pad_value=0.0)
        advantages = _pad_or_trunc(advantages, target_len, pad_value=0.0)
        response_mask = _pad_or_trunc(response_mask, target_len, pad_value=0.0)

        response_mask_bool = response_mask.to(bool)
        loss_mode = actor_config.policy_loss.get("loss_mode", "vanilla") if hasattr(actor_config, "policy_loss") else "vanilla"
        if loss_mode == "bypass_mode":
            pg_losses, effective_mask_bool, ratio = _compute_bypass_pg_losses(
                rollout_log_prob=old_log_probs,
                log_prob=response_log_probs,
                advantages=advantages,
                response_mask_bool=response_mask_bool,
            )
        else:
            rollout_is_weights = data.get("rollout_is_weights", None)
            pg_losses, ratio = _compute_vanilla_pg_losses(
                old_log_prob=old_log_probs,
                log_prob=response_log_probs,
                advantages=advantages,
                rollout_is_weights=rollout_is_weights,
            )
            effective_mask_bool = response_mask_bool

        effective_mask_float = effective_mask_bool.float()
        loss = agg_loss(
            loss_mat=pg_losses,
            loss_mask=effective_mask_float,
            loss_agg_mode=actor_config.loss_agg_mode,
            dp_size=data["dp_size"],
            batch_num_tokens=data["batch_num_tokens"],
            global_batch_size=data["global_batch_size"],
            loss_scale_factor=actor_config.loss_scale_factor,
        )
        num_tokens = effective_mask_float.sum()
        denom = num_tokens.clamp(min=1) if hasattr(num_tokens, "clamp") else max(num_tokens, 1)
        clipped = ((ratio < 1 - clip_ratio) | (ratio > 1 + clip_ratio)).float()
        clip_frac = (clipped * effective_mask_float).sum() / denom
        ratio_mean = (ratio * effective_mask_float).sum() / denom
        loss_sum = (pg_losses * effective_mask_float).sum()

        metrics = {
            "loss": loss.detach().item() if hasattr(loss, "item") else float(loss),
            "loss_sum": loss_sum.detach().item() if hasattr(loss_sum, "item") else float(loss_sum),
            "num_tokens": int(num_tokens.item()) if hasattr(num_tokens, "item") else int(num_tokens),
            "clip_frac": clip_frac.detach().item() if hasattr(clip_frac, "item") else float(clip_frac),
            "ratio_mean": ratio_mean.detach().item() if hasattr(ratio_mean, "item") else float(ratio_mean),
            "log_probs": response_log_probs.detach().cpu(),
        }

        # Calculate precision difference metrics if we have rollout log_probs
        from mint_server.backend.debug_metrics import calculate_debug_metrics

        # Build a batch dict compatible with calculate_debug_metrics
        # IMPORTANT: Naming convention:
        #   - data["old_log_probs"] = rollout log probs from vLLM (generated during rollout phase)
        #   - response_log_probs = actor log probs from current forward pass (Megatron)
        # calculate_debug_metrics expects:
        #   - batch["log_probs"] = rollout log probs (from vLLM generation)
        #   - batch["old_log_probs"] = actor log probs (from current training forward)
        batch_for_metrics = {
            "log_probs": old_log_probs,  # Rollout log probs (from vLLM, stored in data["old_log_probs"])
            "old_log_probs": response_log_probs,  # Actor log probs (current forward pass)
            "response_mask": response_mask.bool(),  # Mask for valid positions
            "responses": data.get("responses"),  # For computing response_length
        }
        import os
        debug_enabled = os.environ.get("MINT_PPO_LOSS_DEBUG", "0") == "1"
        if debug_enabled:
            print(f"[PPO_LOSS DEBUG] Shapes: rollout={batch_for_metrics['log_probs'].shape}, "
                  f"actor={batch_for_metrics['old_log_probs'].shape}, "
                  f"mask={batch_for_metrics['response_mask'].shape}", flush=True)
        debug_metrics = calculate_debug_metrics(batch_for_metrics)
        if debug_enabled:
            print(f"[PPO_LOSS DEBUG] Got debug_metrics: {debug_metrics}", flush=True)
        if debug_metrics:
            metrics.update(debug_metrics)
            if debug_enabled:
                print(f"[PPO_LOSS DEBUG] Updated metrics, keys now: {list(metrics.keys())}", flush=True)

        # DEBUG: Print tokens with very negative logprobs
        # Note: response_log_probs and old_log_probs may be 2D [batch, seq] or 1D [total_tokens]
        diff = (response_log_probs - old_log_probs).abs() * response_mask.float()
        if debug_enabled and diff.numel() > 0 and diff.max() > 10.0:  # Large difference threshold
            import torch.distributed as dist
            if not dist.is_initialized() or dist.get_rank() == 0:
                max_idx = diff.argmax()

                # Handle both 2D [batch, seq] and 1D [total_tokens] tensors
                if diff.dim() == 2:
                    # 2D tensor: convert flat index to (batch_idx, seq_idx)
                    batch_idx = max_idx // diff.shape[1]
                    seq_idx = max_idx % diff.shape[1]
                    position = (batch_idx.item(), seq_idx.item())

                    print("[PPO_LOSS DEBUG] Large logprob diff detected:")
                    print(f"  Position (batch, seq): {position}")
                    print(f"  Rollout logprob: {old_log_probs[batch_idx, seq_idx].item():.4f}")
                    print(f"  Training logprob: {response_log_probs[batch_idx, seq_idx].item():.4f}")
                    print(f"  Diff: {diff[batch_idx, seq_idx].item():.4f}")
                else:
                    # 1D tensor: use position directly
                    position = max_idx.item() if hasattr(max_idx, 'item') else int(max_idx)

                    print("[PPO_LOSS DEBUG] Large logprob diff detected:")
                    print(f"  Position in sequence: {position}")
                    print(f"  Rollout logprob: {old_log_probs[position].item():.4f}")
                    print(f"  Training logprob: {response_log_probs[position].item():.4f}")
                    print(f"  Diff: {diff[position].item():.4f}")

                print(flush=True)

        return loss, metrics

    return ppo_loss_fn


def create_logprob_extractor_fn() -> Callable:
    """Create a function that extracts per-token log_probs from model forward.

    Used for Chat SL and DPO which need per-token log probabilities
    rather than aggregate loss. verl computes log_probs during forward
    and passes them to the loss function.

    Returns a function that returns:
        - Zero loss (no training signal)
        - Metrics dict containing per-token log_probs
    """
    def logprob_extractor(model_output: dict, data: TensorDict, dp_group=None) -> tuple:
        """Extract per-token log_probs from model_output.

        Args:
            model_output: dict with "log_probs" containing per-token log probabilities
            data: TensorDict with input_ids, loss_mask, etc.
            dp_group: data parallel group (unused)

        Returns:
            Tuple of (zero_loss, metrics_with_logprobs)
        """
        log_probs = model_output.get("log_probs")

        if log_probs is None:
            # Fallback: model didn't compute log_probs
            loss = torch.tensor(0.0)
            return loss, {"error": "no_log_probs", "num_tokens": 0}

        # Handle NestedTensor format from verl (NO_PADDING mode)
        if hasattr(log_probs, 'values'):
            log_probs = log_probs.values()

        # Get loss_mask to identify response tokens
        loss_mask = data.get("loss_mask")
        if loss_mask is not None and hasattr(loss_mask, 'values'):
            loss_mask = loss_mask.values()

        # Clone and detach log_probs
        log_probs_cpu = log_probs.detach().cpu()

        # Compute aggregate loss for compatibility (NLL)
        if loss_mask is not None:
            loss_mask_float = loss_mask.float()
            use_external_label = tu.get_non_tensor_data(data, key="use_external_label", default=False)
            if not use_external_label:
                loss_mask_float = torch.roll(loss_mask_float, shifts=-1, dims=0)
            loss_sum = -(log_probs * loss_mask_float).sum()
            num_tokens = loss_mask_float.sum()
            dp_size = tu.get_non_tensor_data(data, key="dp_size", default=1)
            batch_num_tokens = tu.get_non_tensor_data(data, key="batch_num_tokens", default=None)
            if batch_num_tokens is None:
                batch_num_tokens = num_tokens
            if hasattr(batch_num_tokens, "item"):
                batch_num_tokens_value = batch_num_tokens.item()
            else:
                batch_num_tokens_value = float(batch_num_tokens)
            if batch_num_tokens_value > 0:
                nll = loss_sum / batch_num_tokens_value * dp_size
            else:
                nll = loss_sum
        else:
            loss_sum = -log_probs.sum()
            nll = -log_probs.mean()
            num_tokens = log_probs.numel()

        # Return log_probs in metrics
        metrics = {
            "loss": nll.detach().item() if hasattr(nll, 'item') else float(nll),
            "loss_sum": loss_sum.detach().item() if hasattr(loss_sum, "item") else float(loss_sum),
            "num_tokens": int(num_tokens.item()) if hasattr(num_tokens, "item") else int(num_tokens),
            "log_probs": log_probs_cpu,  # Per-token log probabilities tensor
        }
        return nll, metrics

    return logprob_extractor


def create_vocab_parallel_logits_extractor_fn() -> Callable:
    """Create a forward-only extractor that preserves vocab-parallel logits."""

    def extractor(model_output: dict, data: TensorDict, dp_group=None) -> tuple:
        log_probs = model_output.get("log_probs")
        vocab_parallel_logits = model_output.get("vocab_parallel_logits")
        if vocab_parallel_logits is None:
            raise ValueError("model_output missing required vocab_parallel_logits")

        log_probs_flat = None
        if log_probs is not None:
            if getattr(log_probs, "is_nested", False):
                log_probs_flat = log_probs.values()
            else:
                log_probs_flat = log_probs

        if getattr(vocab_parallel_logits, "is_nested", False):
            local_logits = vocab_parallel_logits.values()
        else:
            local_logits = vocab_parallel_logits

        loss_mask = data.get("loss_mask")
        if loss_mask is not None and getattr(loss_mask, "is_nested", False):
            loss_mask_flat = loss_mask.values()
        elif loss_mask is not None:
            loss_mask_flat = loss_mask
        else:
            raise ValueError("data missing required loss_mask")

        metrics = {
            "loss": 0.0,
            "num_tokens": int(loss_mask_flat.float().sum().item()),
            "vocab_parallel_logits": local_logits.detach(),
        }
        if log_probs_flat is not None:
            metrics["log_probs"] = log_probs_flat.detach().cpu()
        return local_logits.new_zeros(()), metrics

    return extractor


def create_reverse_kl_loss_fn(
    temperature: float,
    *,
    reference_log_probs: torch.Tensor | None = None,
) -> Callable:
    """Create reverse-KL loss over vocab-parallel logits against fixed teacher log-probs."""

    import torch.distributed as dist
    from megatron.core import parallel_state as mpu
    from .mintx_ops import vocab_parallel_reverse_kl_against_log_q

    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature!r}")

    def _flatten_rows(tensor: torch.Tensor) -> torch.Tensor:
        if getattr(tensor, "is_nested", False):
            rows = [row for row in tensor.unbind()]
            if not rows:
                return torch.empty((0,), dtype=tensor.dtype, device=tensor.device)
            return torch.cat(rows, dim=0)
        return tensor

    def reverse_kl_loss_fn(model_output: dict, data: TensorDict, dp_group=None) -> tuple:
        packed_student_logits = model_output.get("packed_vocab_parallel_logits")
        raw_student_logits = model_output.get("vocab_parallel_logits")
        if packed_student_logits is None:
            raise ValueError(
                "reverse_kl requires packed_vocab_parallel_logits; "
                f"model_output keys={sorted(model_output.keys())}"
            )
        student_logits = packed_student_logits
        student_token_log_probs = model_output.get("log_probs")
        teacher_log_probs = reference_log_probs if reference_log_probs is not None else data.get("reference_log_probs")
        loss_mask = data.get("loss_mask")

        if teacher_log_probs is None or loss_mask is None:
            raise ValueError("reverse_kl requires vocab_parallel_logits, reference_log_probs, and loss_mask")

        logger.info(
            "reverse_kl loss inputs packed_logits=%s raw_logits_nested=%s raw_logits_type=%s packed_logits_shape=%s raw_logits_shape=%s",
            packed_student_logits is not None,
            getattr(raw_student_logits, "is_nested", False) if raw_student_logits is not None else None,
            type(raw_student_logits).__name__ if raw_student_logits is not None else None,
            list(packed_student_logits.shape) if hasattr(packed_student_logits, "shape") else None,
            list(raw_student_logits.shape) if hasattr(raw_student_logits, "shape") else None,
        )

        student_logits = _flatten_rows(student_logits)
        if student_token_log_probs is not None:
            student_token_log_probs = _flatten_rows(student_token_log_probs)
        teacher_log_probs = _flatten_rows(teacher_log_probs)
        loss_mask = _flatten_rows(loss_mask)
        if os.environ.get("MINT_REVERSE_KL_DIAG_FAIL", "0") == "1":
            raise RuntimeError(
                "REVERSE_KL_DIAG "
                f"student_logits_type={type(student_logits).__name__} "
                f"student_logits_is_nested={getattr(student_logits, 'is_nested', False)} "
                f"student_token_log_probs_type={type(student_token_log_probs).__name__ if student_token_log_probs is not None else None} "
                f"student_token_log_probs_is_nested={getattr(student_token_log_probs, 'is_nested', False) if student_token_log_probs is not None else None} "
                f"teacher_log_probs_type={type(teacher_log_probs).__name__} "
                f"teacher_log_probs_is_nested={getattr(teacher_log_probs, 'is_nested', False)} "
                f"loss_mask_type={type(loss_mask).__name__} "
                f"loss_mask_is_nested={getattr(loss_mask, 'is_nested', False)}"
            )
        if hasattr(teacher_log_probs, "device") and teacher_log_probs.device != student_logits.device:
            teacher_log_probs = teacher_log_probs.to(device=student_logits.device, dtype=torch.float32)

        selected_idx = (loss_mask != 0).nonzero(as_tuple=False).squeeze(-1)
        student_logits = student_logits.index_select(0, selected_idx)
        selected_weights = loss_mask.float().index_select(0, selected_idx)

        if student_token_log_probs is not None:
            student_token_log_probs = student_token_log_probs.index_select(0, selected_idx)

        if student_logits.shape != teacher_log_probs.shape:
            raise ValueError(
                f"student logits shape {tuple(student_logits.shape)} != teacher log-probs shape {tuple(teacher_log_probs.shape)}"
            )

        token_kl = vocab_parallel_reverse_kl_against_log_q(student_logits, teacher_log_probs)
        token_kl = token_kl * (float(temperature) * float(temperature))
        weighted_kl = token_kl * selected_weights
        num_tokens = selected_weights.sum()

        dp_size = tu.get_non_tensor_data(data, key="dp_size", default=1)
        batch_num_tokens = tu.get_non_tensor_data(data, key="batch_num_tokens", default=None)
        if batch_num_tokens is None:
            batch_num_tokens = num_tokens
        batch_num_tokens_value = batch_num_tokens.item() if hasattr(batch_num_tokens, "item") else float(batch_num_tokens)
        if batch_num_tokens_value > 0:
            loss = weighted_kl.sum() / batch_num_tokens_value * dp_size
        else:
            loss = weighted_kl.sum()

        metrics = {
            "loss": loss.detach(),
            "num_tokens": int(num_tokens.item()) if hasattr(num_tokens, "item") else int(num_tokens),
            "reverse_kl_tokens": token_kl.detach().cpu(),
        }
        if student_token_log_probs is not None:
            metrics["log_probs"] = student_token_log_probs.detach().cpu()
        return loss, metrics

    return reverse_kl_loss_fn
