"""Shared Megatron training utilities and deprecated MegatronTrainingWorker.

This module provides:
- Tinker Datum -> verl TensorDict conversion
- Loss functions (SFT/PPO/logprobs)
- is_moe_model() routing helper

MegatronTrainingWorker remains for legacy single-process training and is not used by
VerlTrainingEngine (which uses megatron_distributed.MegatronWorkerGroup).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import ray
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData

from verl.utils import tensordict_utils as tu

if TYPE_CHECKING:
    pass

from tinker_server.backend.model_registry import get_model_config
from tinker_server.model_input_utils import flatten_encoded_text_chunks

logger = logging.getLogger(__name__)


@dataclass
class MegatronTrainingConfig:
    """Configuration for MegatronTrainingWorker.

    Translates Tinker API parameters to verl/Megatron config.
    """
    model_path: str
    lora_rank: int = 16
    lora_alpha: int = 32
    learning_rate: float = 1e-4
    # Parallelism config - single process for now (TP=1 to avoid distributed)
    # TODO: Implement proper multi-process parallelism for 8 GPUs
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 1
    context_parallel_size: int = 1
    # Offloading - enable to fit large models
    param_offload: bool = True
    optimizer_offload: bool = True
    grad_offload: bool = True
    # Training
    dtype: str = "bfloat16"
    seed: int = 42


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
    def _extract_list(field_name: str, field: dict | None, item_index: int) -> list | None:
        if field is None:
            return None
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
    if has_external_labels and target_tokens_list:
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

    if has_routed_experts:
        if all(re is not None for re in routed_experts_list):
            re_nested = torch.nested.as_nested_tensor(routed_experts_list, layout=torch.jagged)
            td["routed_experts"] = re_nested
            td.set_non_tensor("enable_routing_replay", True)
        else:
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
    def sft_loss_with_logprobs(model_output: dict, data: TensorDict, dp_group=None) -> tuple:
        """SFT cross-entropy loss that also returns log_probs for metrics.

        Args:
            model_output: dict with "log_probs" key containing log probabilities
            data: TensorDict with loss_mask, dp_size, batch_num_tokens
            dp_group: data parallel group (unused)

        Returns:
            Tuple of (loss_tensor, metrics_dict_with_logprobs)
        """
        log_probs = model_output.get("log_probs")

        if log_probs is None:
            raise ValueError("model_output missing required log_probs")

        # Handle NestedTensor format from verl (NO_PADDING mode)
        if hasattr(log_probs, 'values'):
            log_probs_flat = log_probs.values()
        else:
            log_probs_flat = log_probs

        # Get loss_mask to identify which tokens contribute to loss
        loss_mask = data.get("loss_mask")
        if loss_mask is not None and hasattr(loss_mask, 'values'):
            loss_mask_flat = loss_mask.values()
        elif loss_mask is not None:
            loss_mask_flat = loss_mask
        else:
            raise ValueError("data missing required loss_mask")

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

        if batch_num_tokens_value > 0:
            nll = -weighted_log_probs.sum() / batch_num_tokens_value * dp_size
        else:
            nll = -weighted_log_probs.sum()

        # Clone log_probs for metrics (detach to avoid affecting gradients)
        log_probs_cpu = log_probs_flat.detach().cpu()

        metrics = {
            "loss": nll.detach(),
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


def create_ppo_loss_fn(epsilon: float = 0.2) -> Callable:
    """Create PPO loss function by reusing verl's ppo_loss/agg_loss path."""
    import math
    import torch.nn.functional as F
    from verl.workers.config import ActorConfig
    from verl.workers.utils.losses import ppo_loss as verl_ppo_loss, _slice_response_from_unpad_output

    clip_ratio = float(epsilon)
    clip_ratio_c = 1e6 if math.isinf(clip_ratio) else 3.0
    actor_config = ActorConfig(
        strategy="tinker",
        rollout_n=1,
        use_dynamic_bsz=True,
        clip_ratio=clip_ratio,
        clip_ratio_low=clip_ratio,
        clip_ratio_high=clip_ratio,
        clip_ratio_c=clip_ratio_c,
    )

    def ppo_loss_fn(model_output: dict, data: TensorDict, dp_group=None) -> tuple:
        """PPO clipped objective loss via verl's implementation.

        Returns:
            Tuple of (loss_tensor, metrics_dict)
        """
        response_log_probs = _slice_response_from_unpad_output(model_output["log_probs"], data)
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

        data["old_log_probs"] = old_log_probs
        data["advantages"] = advantages
        data["response_mask"] = response_mask

        loss, _ = verl_ppo_loss(actor_config, model_output, data, dp_group=dp_group)

        response_mask_bool = response_mask.to(bool)
        response_mask_float = response_mask_bool.float()

        log_ratio = response_log_probs - old_log_probs
        log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)

        num_tokens = response_mask_float.sum()
        denom = num_tokens.clamp(min=1) if hasattr(num_tokens, "clamp") else max(num_tokens, 1)
        clipped = ((ratio < 1 - clip_ratio) | (ratio > 1 + clip_ratio)).float()
        clip_frac = (clipped * response_mask_float).sum() / denom
        ratio_mean = (ratio * response_mask_float).sum() / denom

        metrics = {
            "loss": loss.detach().item() if hasattr(loss, "item") else float(loss),
            "num_tokens": int(num_tokens.item()) if hasattr(num_tokens, "item") else int(num_tokens),
            "clip_frac": clip_frac.detach().item() if hasattr(clip_frac, "item") else float(clip_frac),
            "ratio_mean": ratio_mean.detach().item() if hasattr(ratio_mean, "item") else float(ratio_mean),
            "log_probs": response_log_probs.detach().cpu(),
        }

        # Calculate precision difference metrics if we have rollout log_probs
        from tinker_server.backend.debug_metrics import calculate_debug_metrics

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

                    print(f"[PPO_LOSS DEBUG] Large logprob diff detected:")
                    print(f"  Position (batch, seq): {position}")
                    print(f"  Rollout logprob: {old_log_probs[batch_idx, seq_idx].item():.4f}")
                    print(f"  Training logprob: {response_log_probs[batch_idx, seq_idx].item():.4f}")
                    print(f"  Diff: {diff[batch_idx, seq_idx].item():.4f}")
                else:
                    # 1D tensor: use position directly
                    position = max_idx.item() if hasattr(max_idx, 'item') else int(max_idx)

                    print(f"[PPO_LOSS DEBUG] Large logprob diff detected:")
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
            nll = -(log_probs * loss_mask_float).sum()
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
                nll = nll / batch_num_tokens_value * dp_size
        else:
            nll = -log_probs.mean()
            num_tokens = log_probs.numel()

        # Return log_probs in metrics
        metrics = {
            "loss": nll.detach().item() if hasattr(nll, 'item') else float(nll),
            "num_tokens": int(num_tokens.item()) if hasattr(num_tokens, "item") else int(num_tokens),
            "log_probs": log_probs_cpu,  # Per-token log probabilities tensor
        }
        return nll, metrics

    return logprob_extractor


@ray.remote(num_gpus=1)  # TODO: Implement multi-GPU parallelism
class MegatronTrainingWorker:
    """Ray actor for MoE training via verl's Megatron backend.

    Provides Tinker API compatibility:
    - forward_backward(data_items, loss_fn) -> {loss_fn_outputs, metrics}
    - optim_step(learning_rate) -> {metrics}
    - get_lora_state_dict() -> {name: tensor}
    """

    def __init__(
        self,
        base_model: str,
        lora_rank: int,
        learning_rate: float,
        config: MegatronTrainingConfig | None = None,
    ):
        """Initialize Megatron training worker.

        Currently runs single-GPU without parallelism. Multi-GPU parallelism
        requires launching distributed processes (torchrun) which is not yet
        implemented for Ray actors.

        Args:
            base_model: HuggingFace model path or local path.
            lora_rank: LoRA adapter rank.
            learning_rate: Initial learning rate.
            config: Optional full MegatronTrainingConfig.
        """
        self.base_model = base_model
        self.lora_rank = lora_rank
        self.learning_rate = learning_rate

        if config is None:
            config = MegatronTrainingConfig(
                model_path=base_model,
                lora_rank=lora_rank,
                learning_rate=learning_rate,
            )
        self.config = config

        # Will be set during initialization
        self.engine = None
        self.bridge = None
        self._step_count = 0

        # Initialize the Megatron backend
        self._initialize_megatron()

        logger.info(f"[MegatronTrainingWorker] Ready with model={base_model}, lora_rank={lora_rank}")

    def _initialize_megatron(self):
        """Initialize verl's MegatronEngine.

        This sets up:
        1. Distributed process group
        2. Model parallel groups (TP, PP, EP, CP)
        3. Model with LoRA via Megatron-Bridge
        4. Optimizer with offloading
        """
        # Import verl components
        from verl.workers.config import HFModelConfig, McoreEngineConfig, McoreOptimizerConfig
        from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead
        from verl.trainer.config import CheckpointConfig
        from verl.utils.fs import copy_to_local
        from verl.utils.torch_dtypes import PrecisionType

        # Initialize distributed if not already done
        # For Ray actor running single-process multi-GPU, we set up minimal distributed env
        if not torch.distributed.is_initialized():
            # Set required env vars for torch.distributed if not present
            if "RANK" not in os.environ:
                os.environ["RANK"] = "0"
            if "WORLD_SIZE" not in os.environ:
                os.environ["WORLD_SIZE"] = "1"
            if "LOCAL_RANK" not in os.environ:
                os.environ["LOCAL_RANK"] = "0"
            if "MASTER_ADDR" not in os.environ:
                os.environ["MASTER_ADDR"] = "localhost"
            if "MASTER_PORT" not in os.environ:
                os.environ["MASTER_PORT"] = "29500"

            rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.distributed.init_process_group(backend="nccl")
            torch.cuda.set_device(rank)

        # Copy model to local if needed
        local_path = copy_to_local(self.base_model)

        # Build HFModelConfig
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(local_path, trust_remote_code=True, local_files_only=True)

        model_config = HFModelConfig(
            path=self.base_model,  # HuggingFace model name for tokenizer
            local_path=local_path,
            hf_config=hf_config,
            architectures=hf_config.architectures,
            lora_rank=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules="all-linear",
            trust_remote_code=True,
        )

        # Build McoreEngineConfig
        engine_config = McoreEngineConfig(
            tensor_model_parallel_size=self.config.tensor_parallel_size,
            pipeline_model_parallel_size=self.config.pipeline_parallel_size,
            expert_model_parallel_size=self.config.expert_parallel_size,
            context_parallel_size=self.config.context_parallel_size,
            param_offload=self.config.param_offload,
            optimizer_offload=self.config.optimizer_offload,
            grad_offload=self.config.grad_offload,
            dtype=self.config.dtype,
            seed=self.config.seed,
            use_mbridge=True,
            use_distributed_optimizer=True,
        )

        # Build McoreOptimizerConfig
        # Use constant LR decay style for online learning (no fixed schedule)
        optimizer_config = McoreOptimizerConfig(
            lr=self.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            clip_grad=1.0,
            lr_decay_steps=100000,  # Large value for online learning
            lr_decay_style="constant",  # Don't decay learning rate
            lr_warmup_steps=0,
        )

        # Build CheckpointConfig (minimal)
        checkpoint_config = CheckpointConfig()

        # Create and initialize the engine
        # Use MegatronEngineWithLMHead which implements forward_step for LM training
        self.engine = MegatronEngineWithLMHead(
            model_config=model_config,
            engine_config=engine_config,
            optimizer_config=optimizer_config,
            checkpoint_config=checkpoint_config,
        )
        self.engine.initialize()

        # Store bridge reference for weight export
        self.bridge = self.engine.bridge

        logger.info("[MegatronTrainingWorker] MegatronEngineWithLMHead initialized")

    def forward_backward(
        self,
        data_items: list[dict],
        loss_fn: str = "cross_entropy",
        loss_fn_config: dict | None = None,
    ) -> dict:
        """Forward + backward pass via MegatronEngine.

        Args:
            data_items: List of Tinker Datum dicts.
            loss_fn: Loss function type ("cross_entropy", "importance_sampling", "ppo").
            loss_fn_config: Optional config (e.g., {"epsilon": 0.2} for PPO).

        Returns:
            Dict with loss_fn_outputs and metrics.
        """
        loss_fn_config = loss_fn_config or {}

        # Filter valid items and collect sequence lengths.
        valid_items: list[dict] = []
        valid_indices: list[int] = []
        seq_lengths: list[int] = []
        for item_index, item in enumerate(data_items):
            model_input = item.get("model_input", {})
            tokens = flatten_encoded_text_chunks(model_input)
            if tokens:
                valid_items.append(item)
                valid_indices.append(item_index)
                seq_lengths.append(len(tokens))
            else:
                logger.warning(f"[MegatronTrainingWorker] Missing tokens in item {item_index}, skipping")

        if not valid_items:
            empty_outputs = [
                {
                    "loss": {"data": [0.0], "shape": [1], "dtype": "float32"},
                    "logprobs": {"data": [], "shape": [0], "dtype": "float32"},
                }
                for _ in data_items
            ]
            return {
                "loss_fn_output_type": f"{loss_fn}_loss",
                "loss_fn_outputs": empty_outputs,
                "metrics": {"loss:mean": 0.0, "num_samples:sum": 0.0, "num_tokens:sum": 0.0},
            }

        # Create TensorDict directly on device to avoid NestedTensor .to() issues.
        if torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"
        else:
            device = "cpu"
        data = tinker_to_tensordict(valid_items, device=device)

        # Select loss function
        if loss_fn == "cross_entropy":
            loss_function = create_sft_loss_fn()
        elif loss_fn == "ppo":
            epsilon = loss_fn_config.get("epsilon", 0.2)
            loss_function = create_ppo_loss_fn(epsilon)
        elif loss_fn == "importance_sampling":
            # Importance sampling is PPO without clipping
            loss_function = create_ppo_loss_fn(epsilon=float("inf"))
        else:
            raise ValueError(f"Unknown loss_fn: {loss_fn}")

        # Zero gradients
        self.engine.optimizer_zero_grad()

        # Forward + backward via engine
        result = self.engine.forward_backward_batch(
            data=data,
            loss_function=loss_function,
            forward_only=False,
        )

        # Extract metrics from result (verl returns a single dict).
        loss_value = 0.0
        num_tokens = 0
        clip_frac_sum = 0.0
        ratio_mean_sum = 0.0
        n_ppo_results = 0
        all_log_probs = []
        loss_fn_outputs = []
        per_sample_log_probs = None
        if result and isinstance(result, dict):
            result_metrics = result.get("metrics", {})
            losses = result.get("loss", [])

            for loss in losses:
                if hasattr(loss, "item"):
                    loss = loss.item()
                loss_value += float(loss)

            num_tokens_list = result_metrics.get("num_tokens", [])
            for tokens in num_tokens_list:
                if hasattr(tokens, "item"):
                    tokens = tokens.item()
                num_tokens += int(tokens)

            log_probs_list = result_metrics.get("log_probs", [])
            for log_probs in log_probs_list:
                if log_probs is not None:
                    if hasattr(log_probs, "cpu"):
                        log_probs = log_probs.cpu()
                    all_log_probs.append(log_probs)

            clip_frac_list = result_metrics.get("clip_frac", [])
            for clip_frac in clip_frac_list:
                if hasattr(clip_frac, "item"):
                    clip_frac = clip_frac.item()
                clip_frac_sum += float(clip_frac)
                n_ppo_results += 1

            ratio_mean_list = result_metrics.get("ratio_mean", [])
            for ratio_mean in ratio_mean_list:
                if hasattr(ratio_mean, "item"):
                    ratio_mean = ratio_mean.item()
                ratio_mean_sum += float(ratio_mean)

            model_output = result.get("model_output", {})
            model_log_probs = model_output.get("log_probs")
            if model_log_probs is not None:
                if hasattr(model_log_probs, "unbind"):
                    per_sample_log_probs = [lp.detach().cpu() for lp in model_log_probs.unbind()]
                elif seq_lengths and hasattr(model_log_probs, "dim") and model_log_probs.dim() >= 2:
                    per_sample_log_probs = []
                    for idx, row in enumerate(model_log_probs):
                        seq_len = seq_lengths[idx] if idx < len(seq_lengths) else row.shape[0]
                        per_sample_log_probs.append(row[:seq_len].detach().cpu())

        if per_sample_log_probs:
            avg_loss_per_sample = loss_value / max(len(per_sample_log_probs), 1)
            for sample_log_probs in per_sample_log_probs:
                logprobs_list = sample_log_probs.tolist()
                loss_fn_outputs.append({
                    "loss": {"data": [avg_loss_per_sample], "shape": [1], "dtype": "float32"},
                    "logprobs": {"data": logprobs_list, "shape": [len(logprobs_list)], "dtype": "float32"},
                })
        elif loss_fn == "cross_entropy" and all_log_probs and seq_lengths:
            combined_log_probs = torch.cat(all_log_probs, dim=0) if len(all_log_probs) > 1 else all_log_probs[0]
            offset = 0
            avg_loss_per_sample = loss_value / max(len(seq_lengths), 1)
            for seq_len in seq_lengths:
                sample_log_probs = combined_log_probs[offset:offset + seq_len]
                offset += seq_len
                logprobs_list = sample_log_probs.tolist()
                loss_fn_outputs.append({
                    "loss": {"data": [avg_loss_per_sample], "shape": [1], "dtype": "float32"},
                    "logprobs": {"data": logprobs_list, "shape": [len(logprobs_list)], "dtype": "float32"},
                })

        valid_count = len(seq_lengths)
        expected_outputs = valid_count
        if expected_outputs and len(loss_fn_outputs) < expected_outputs:
            avg_loss_per_sample = loss_value / max(expected_outputs, 1)
            for _ in range(expected_outputs - len(loss_fn_outputs)):
                loss_fn_outputs.append({
                    "loss": {"data": [avg_loss_per_sample], "shape": [1], "dtype": "float32"},
                    "logprobs": {"data": [], "shape": [0], "dtype": "float32"},
                })

        if valid_indices:
            avg_loss_per_sample = loss_value / max(valid_count, 1)
            full_outputs = [
                {"loss": {"data": [avg_loss_per_sample], "shape": [1], "dtype": "float32"},
                 "logprobs": {"data": [], "shape": [0], "dtype": "float32"}}
                for _ in data_items
            ]
            for output, item_index in zip(loss_fn_outputs, valid_indices):
                full_outputs[item_index] = output
            loss_fn_outputs = full_outputs

        metrics = {
            "loss:mean": float(loss_value),
            "num_samples:sum": float(valid_count),
            "num_tokens:sum": float(num_tokens),
        }

        if loss_fn in ("ppo", "importance_sampling") and n_ppo_results > 0:
            metrics["clipfrac:mean"] = float(clip_frac_sum / n_ppo_results)
            metrics["ratio:mean"] = float(ratio_mean_sum / n_ppo_results)

        debug_metric_keys = [
            "training/rollout_probs_diff_valid",
            "training/rollout_probs_diff_mean",
            "training/rollout_probs_diff_max",
            "training/rollout_probs_diff_std",
            "training/rollout_actor_probs_pearson_corr",
        ]
        for debug_key in debug_metric_keys:
            values = result_metrics.get(debug_key)
            if values is None:
                continue
            if isinstance(values, list):
                numeric_values = []
                for value in values:
                    if hasattr(value, "item"):
                        value = value.item()
                    if isinstance(value, (int, float)):
                        numeric_values.append(float(value))
                if numeric_values:
                    metrics[f"{debug_key}:mean"] = float(sum(numeric_values) / len(numeric_values))
            else:
                if hasattr(values, "item"):
                    values = values.item()
                if isinstance(values, (int, float)):
                    metrics[f"{debug_key}:mean"] = float(values)

        logger.info(f"[MegatronTrainingWorker] forward_backward ({loss_fn}): loss={loss_value:.4f}")

        return {
            "loss_fn_output_type": f"{loss_fn}_loss",
            "loss_fn_outputs": loss_fn_outputs,
            "metrics": metrics,
        }

    def forward(self, data_items: list[dict]) -> dict:
        """Forward pass only (no backward). Returns per-token logprobs.

        Args:
            data_items: List of Tinker Datum dicts.

        Returns:
            Dict with loss_fn_outputs (including per-token logprobs) and metrics.
        """
        valid_items: list[dict] = []
        valid_indices: list[int] = []
        seq_lengths: list[int] = []
        for item_index, item in enumerate(data_items):
            model_input = item.get("model_input", {})
            tokens = flatten_encoded_text_chunks(model_input)
            if tokens:
                valid_items.append(item)
                valid_indices.append(item_index)
                seq_lengths.append(len(tokens))
            else:
                logger.warning(f"[MegatronTrainingWorker] Missing tokens in item {item_index}, skipping")

        if not valid_items:
            empty_outputs = [
                {
                    "loss": {"data": [0.0], "shape": [1], "dtype": "float32"},
                    "logprobs": [],
                }
                for _ in data_items
            ]
            return {
                "loss_fn_output_type": "logprob_extractor",
                "loss_fn_outputs": empty_outputs,
                "metrics": {"loss:mean": 0.0, "num_samples:sum": 0.0, "num_tokens:sum": 0.0},
                "log_probs": None,
            }

        if torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"
        else:
            device = "cpu"
        data = tinker_to_tensordict(valid_items, device=device)

        # Use logprob extractor to get per-token log probabilities
        loss_function = create_logprob_extractor_fn()

        # Forward only via engine
        with torch.no_grad():
            result = self.engine.forward_backward_batch(
                data=data,
                loss_function=loss_function,
                forward_only=True,
            )

        # Extract per-token log_probs from result (verl returns a dict).
        loss_value = 0.0
        num_tokens = 0
        all_log_probs = []
        loss_fn_outputs = []
        per_sample_log_probs = None
        combined_log_probs = None
        result_metrics: dict[str, Any] = {}

        if result and isinstance(result, dict):
            result_metrics = result.get("metrics", {})
            losses = result.get("loss", [])

            for loss in losses:
                if hasattr(loss, "item"):
                    loss = loss.item()
                loss_value += float(loss)

            num_tokens_list = result_metrics.get("num_tokens", [])
            for tokens in num_tokens_list:
                if hasattr(tokens, "item"):
                    tokens = tokens.item()
                num_tokens += int(tokens)

            log_probs_list = result_metrics.get("log_probs", [])
            for log_probs in log_probs_list:
                if log_probs is not None:
                    if hasattr(log_probs, "cpu"):
                        log_probs = log_probs.cpu()
                    all_log_probs.append(log_probs)

            model_output = result.get("model_output", {})
            model_log_probs = model_output.get("log_probs")
            if model_log_probs is not None:
                if hasattr(model_log_probs, "unbind"):
                    per_sample_log_probs = [lp.detach().cpu() for lp in model_log_probs.unbind()]
                elif seq_lengths and hasattr(model_log_probs, "dim") and model_log_probs.dim() >= 2:
                    per_sample_log_probs = []
                    for idx, row in enumerate(model_log_probs):
                        seq_len = seq_lengths[idx] if idx < len(seq_lengths) else row.shape[0]
                        per_sample_log_probs.append(row[:seq_len].detach().cpu())

        if per_sample_log_probs:
            combined_log_probs = (
                torch.cat(per_sample_log_probs, dim=0)
                if len(per_sample_log_probs) > 1
                else per_sample_log_probs[0]
            )
            avg_loss_per_sample = loss_value / max(len(per_sample_log_probs), 1)
            for sample_log_probs in per_sample_log_probs:
                logprobs_list = sample_log_probs.tolist()
                loss_fn_outputs.append({
                    "loss": {"data": [avg_loss_per_sample], "shape": [1], "dtype": "float32"},
                    "logprobs": logprobs_list,
                })
        else:
            if all_log_probs:
                combined_log_probs = (
                    torch.cat(all_log_probs, dim=0) if len(all_log_probs) > 1 else all_log_probs[0]
                )
            if combined_log_probs is not None and seq_lengths:
                offset = 0
                avg_loss_per_sample = loss_value / max(len(seq_lengths), 1)
                for seq_len in seq_lengths:
                    sample_log_probs = combined_log_probs[offset:offset + seq_len]
                    offset += seq_len
                    logprobs_list = sample_log_probs.tolist()
                    loss_fn_outputs.append({
                        "loss": {"data": [avg_loss_per_sample], "shape": [1], "dtype": "float32"},
                        "logprobs": logprobs_list,
                    })
            elif combined_log_probs is not None:
                logprobs_list = combined_log_probs.tolist()
                loss_fn_outputs.append({
                    "loss": {"data": [loss_value], "shape": [1], "dtype": "float32"},
                    "logprobs": logprobs_list,
                })

        valid_count = len(seq_lengths)
        expected_outputs = valid_count
        if expected_outputs and len(loss_fn_outputs) < expected_outputs:
            avg_loss_per_sample = loss_value / max(expected_outputs, 1)
            for _ in range(expected_outputs - len(loss_fn_outputs)):
                loss_fn_outputs.append({
                    "loss": {"data": [avg_loss_per_sample], "shape": [1], "dtype": "float32"},
                    "logprobs": [],
                })

        if valid_indices:
            avg_loss_per_sample = loss_value / max(valid_count, 1)
            full_outputs = [
                {"loss": {"data": [avg_loss_per_sample], "shape": [1], "dtype": "float32"},
                 "logprobs": []}
                for _ in data_items
            ]
            for output, item_index in zip(loss_fn_outputs, valid_indices):
                full_outputs[item_index] = output
            loss_fn_outputs = full_outputs

        log_probs_data = None
        if combined_log_probs is not None:
            log_probs_data = {
                "data": combined_log_probs.tolist(),
                "shape": list(combined_log_probs.shape),
                "dtype": str(combined_log_probs.dtype),
            }

        metrics = {
            "loss:mean": float(loss_value),
            "num_samples:sum": float(valid_count),
            "num_tokens:sum": float(num_tokens),
        }

        logger.info(
            f"[MegatronTrainingWorker] forward: loss={loss_value:.4f}, "
            f"log_probs={'present' if log_probs_data else 'none'}"
        )

        return {
            "loss_fn_output_type": "logprob_extractor",
            "loss_fn_outputs": loss_fn_outputs,
            "metrics": metrics,
            "log_probs": log_probs_data,
        }

    def optim_step(self, learning_rate: float | None = None) -> dict:
        """Optimizer step.

        Args:
            learning_rate: Optional new learning rate.

        Returns:
            Dict with metrics.
        """
        # Update learning rate if provided
        # Note: verl's optimizer handles LR scheduling differently
        # For now, we skip dynamic LR updates

        # Optimizer step via engine
        grad_norm = self.engine.optimizer_step()

        # LR scheduler step
        current_lr = self.engine.lr_scheduler_step()

        self._step_count += 1

        logger.info(f"[MegatronTrainingWorker] optim_step: grad_norm={grad_norm:.4f}, step={self._step_count}")

        return {
            "metrics": {
                "grad_norm": float(grad_norm) if grad_norm is not None else 0.0,
                "step": self._step_count,
                "lr": float(current_lr[0]) if current_lr else self.learning_rate,
            },
            "type": "optim_step",
        }

    def get_lora_state_dict(self, use_per_expert_lora: bool = False) -> dict[str, torch.Tensor]:
        """Extract LoRA adapter weights in PEFT format.

        NOTE: This class is DEPRECATED. Use MegatronWorkerGroup from megatron_distributed.py instead.
        This method uses the legacy single-actor approach and may not work with modern Megatron-Bridge.

        Uses bridge.export_hf_weights() and filters for LoRA parameters.
        Converts mbridge HuggingFace names to PEFT format for vLLM compatibility.

        mbridge format: layers.0.self_attn.q_proj.lora_A.weight
        PEFT format:    base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight

        Args:
            use_per_expert_lora: Ignored for legacy actor. Only applies to
                MegatronWorkerGroup for MoE models.

        Returns:
            Dict mapping LoRA parameter names (PEFT format) to CPU tensors.
        """
        import warnings
        warnings.warn(
            "MegatronTrainingWorker is deprecated. Use MegatronWorkerGroup from megatron_distributed.py instead.",
            DeprecationWarning,
            stacklevel=2
        )

        if self.bridge is None:
            raise RuntimeError("Bridge not initialized - cannot export weights")

        # Export all weights via bridge - requires export_hf_weights() API
        # The old export_weights() API merges LoRA into base weights, which is unusable
        # for multi-LoRA inference. We must have separate lora_A/lora_B matrices.
        if not hasattr(self.bridge, 'export_hf_weights'):
            raise RuntimeError(
                "Bridge lacks export_hf_weights() method. "
                "The old export_weights() API merges LoRA into base weights, "
                "which cannot be used for vLLM multi-LoRA inference."
            )
        full_state_dict = dict(self.bridge.export_hf_weights(self.engine.module))

        # Filter for LoRA parameters and convert to PEFT format
        lora_state_dict = {}
        for name, tensor in full_state_dict.items():
            if "lora" in name.lower():
                # Convert mbridge HuggingFace format to PEFT format
                peft_name = f"base_model.model.model.{name}"
                # Move to CPU for Ray serialization
                lora_state_dict[peft_name] = tensor.cpu() if tensor.is_cuda else tensor

        logger.info(f"[MegatronTrainingWorker] Extracted {len(lora_state_dict)} LoRA parameters (PEFT format)")
        if lora_state_dict:
            sample_keys = list(lora_state_dict.keys())[:3]
            logger.debug(f"[MegatronTrainingWorker] Sample LoRA keys: {sample_keys}")

        return lora_state_dict

    def get_lora_config(self) -> dict:
        """Get LoRA configuration as dictionary.

        Returns:
            PEFT config dict compatible with vLLM's PEFTHelper.
        """
        return {
            "r": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": 0.0,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "peft_type": "LORA",
            "base_model_name_or_path": self.base_model,
        }

    def get_tokenizer_info(self) -> dict:
        """Return tokenizer configuration.

        Returns:
            Dict with tokenizer info.
        """
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True, local_files_only=True)

        return {
            "vocab_size": tokenizer.vocab_size,
            "model_max_length": tokenizer.model_max_length,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "bos_token_id": tokenizer.bos_token_id,
        }


    def save_checkpoint(self, save_path: str) -> dict:
        """Save checkpoint: LoRA weights + config + training metadata.

        For distributed Megatron training, optimizer state is sharded across ranks
        and not saved here. Only rank 0 saves the checkpoint.

        Args:
            save_path: Directory path to save checkpoint files.

        Returns:
            Dict with training metadata.
        """
        import json
        import os

        from safetensors.torch import save_file

        os.makedirs(save_path, exist_ok=True)

        # 1. LoRA weights
        state_dict = self.get_lora_state_dict()
        save_file(state_dict, os.path.join(save_path, "adapter_model.safetensors"))

        # 2. LoRA config
        config = self.get_lora_config()
        with open(os.path.join(save_path, "adapter_config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # 3. Training metadata
        meta = {
            "current_step": self._step_count,
            "learning_rate": self.learning_rate,
        }
        with open(os.path.join(save_path, "training_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        abs_path = os.path.abspath(save_path)
        logger.info(f"[MegatronTrainingWorker] Saved checkpoint to {abs_path} (step={self._step_count})")
        return meta

    def shutdown(self) -> None:
        """Release resources."""
        logger.info("[MegatronTrainingWorker] Shutting down")
        # MegatronEngine cleanup handled by garbage collection
        self.engine = None
        self.bridge = None
        torch.cuda.empty_cache()


def is_moe_model(model_name: str) -> bool:
    """Check if a model is an MoE model requiring Megatron training.

    Args:
        model_name: Model name (e.g., "Qwen/Qwen3-30B-A3B").

    Returns:
        True if model is MoE and should use MegatronTrainingWorker.

    Raises:
        ValueError: If model is not in the supported list.
    """
    from .model_registry import get_model_config
    return get_model_config(model_name).is_moe
