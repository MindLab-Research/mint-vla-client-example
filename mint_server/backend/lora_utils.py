"""Utilities for LoRA adapter manipulation.

Provides functions for padding/truncating LoRA weights to support unified rank
training where a trainer with max_rank can train adapters with any rank <= max_rank.
"""

import json
import os
import re
from functools import lru_cache

import torch
from safetensors import safe_open


_LORA_WEIGHT_RE = re.compile(r"^(?P<module>.+)\.(?P<kind>lora_A|lora_B)\.weight$")


def _validate_rank_pair(actual_rank: int, trainer_rank: int) -> tuple[int, int]:
    actual_rank = int(actual_rank)
    trainer_rank = int(trainer_rank)
    if actual_rank <= 0:
        raise ValueError(f"actual_rank must be positive, got {actual_rank}")
    if trainer_rank <= 0:
        raise ValueError(f"trainer_rank must be positive, got {trainer_rank}")
    if actual_rank > trainer_rank:
        raise ValueError(f"actual_rank {actual_rank} exceeds trainer_rank {trainer_rank}")
    return actual_rank, trainer_rank


def _is_lora_a_name(name: str) -> bool:
    lowered = name.lower()
    return "lora_a" in lowered or ("adapter" in lowered and "linear_in" in lowered)


def _is_lora_b_name(name: str) -> bool:
    lowered = name.lower()
    return "lora_b" in lowered or ("adapter" in lowered and "linear_out" in lowered)


def lora_rank_tail_slice(
    name: str,
    tensor: torch.Tensor,
    actual_rank: int,
    trainer_rank: int,
    *,
    rank_shard_index: int = 0,
    rank_shard_count: int = 1,
) -> tuple[slice, ...] | None:
    """Return the slice covering padded LoRA rank dimensions for a tensor."""
    actual_rank, trainer_rank = _validate_rank_pair(actual_rank, trainer_rank)
    if actual_rank == trainer_rank:
        return None
    if not (_is_lora_a_name(name) or _is_lora_b_name(name)):
        return None
    if tensor.ndim < 2:
        raise ValueError(f"{name}: expected rank-2+ LoRA tensor, got shape={tuple(tensor.shape)}")
    rank_shard_index = int(rank_shard_index)
    rank_shard_count = int(rank_shard_count)
    if rank_shard_index < 0:
        raise ValueError(f"rank_shard_index must be non-negative, got {rank_shard_index}")
    if rank_shard_count <= 0:
        raise ValueError(f"rank_shard_count must be positive, got {rank_shard_count}")
    if rank_shard_index >= rank_shard_count:
        raise ValueError(
            f"rank_shard_index {rank_shard_index} must be smaller than rank_shard_count {rank_shard_count}"
        )

    def _local_tail(local_rank_dim: int) -> slice | None:
        if local_rank_dim == trainer_rank:
            return slice(actual_rank, trainer_rank)
        if rank_shard_count > 1 and local_rank_dim * rank_shard_count == trainer_rank:
            shard_start = rank_shard_index * local_rank_dim
            active_local = max(0, min(local_rank_dim, actual_rank - shard_start))
            if active_local == local_rank_dim:
                return None
            return slice(active_local, local_rank_dim)
        raise ValueError(
            f"{name}: LoRA rank dim is {local_rank_dim}, expected trainer_rank {trainer_rank} "
            f"or a TP-sharded local dim for rank_shard_count {rank_shard_count}"
        )

    if _is_lora_a_name(name):
        tail = _local_tail(int(tensor.shape[0]))
        if tail is None:
            return None
        return (tail,) + (slice(None),) * (tensor.ndim - 1)
    tail = _local_tail(int(tensor.shape[-1]))
    if tail is None:
        return None
    return (slice(None),) * (tensor.ndim - 1) + (tail,)


def zero_lora_rank_tail_named_parameters(
    named_parameters,
    actual_rank: int,
    trainer_rank: int,
    *,
    zero_grads: bool = True,
    rank_shard_index: int = 0,
    rank_shard_count: int = 1,
) -> dict[str, int]:
    """Project padded LoRA rank tails and optionally their gradients to exact zero."""
    _validate_rank_pair(actual_rank, trainer_rank)
    stats = {"params": 0, "grads": 0}
    if actual_rank == trainer_rank:
        return stats
    for name, param in named_parameters:
        tail = lora_rank_tail_slice(
            name,
            param.data,
            actual_rank,
            trainer_rank,
            rank_shard_index=rank_shard_index,
            rank_shard_count=rank_shard_count,
        )
        if tail is None:
            continue
        with torch.no_grad():
            param.data[tail].zero_()
        stats["params"] += 1
        if zero_grads and param.grad is not None:
            param.grad[tail].zero_()
            stats["grads"] += 1
    return stats


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


def _normalize_lora_module_name(module_name: str) -> str:
    if module_name.startswith("base_model.model."):
        module_name = module_name[len("base_model.model.") :]
    if not module_name.startswith(("model.", "language_model.", "llava_model.")):
        module_name = f"model.{module_name}"
    if module_name.endswith(".experts.base_layer"):
        return module_name[: -len(".base_layer")] + ".gate_up_proj"
    if module_name.endswith(".shared_expert.base_layer"):
        return module_name[: -len(".base_layer")] + ".gate_up_proj"
    return module_name


@lru_cache(maxsize=128)
def _model_weight_map(base_model_path: str) -> dict[str, str] | None:
    index_path = os.path.join(base_model_path, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        return None
    with open(index_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"Invalid safetensors index: {index_path}")
    return {str(k): str(v) for k, v in weight_map.items()}


def _resolve_base_weight_file(base_model_path: str, tensor_name: str) -> str:
    weight_map = _model_weight_map(base_model_path)
    if weight_map is not None:
        shard_name = weight_map.get(tensor_name)
        if not shard_name:
            raise KeyError(f"Base weight not found in index: {tensor_name}")
        return os.path.join(base_model_path, shard_name)

    single_path = os.path.join(base_model_path, "model.safetensors")
    if os.path.isfile(single_path):
        return single_path

    shard_paths = [
        os.path.join(base_model_path, name)
        for name in sorted(os.listdir(base_model_path))
        if name.endswith(".safetensors")
    ]
    for path in shard_paths:
        with safe_open(path, framework="pt", device="cpu") as handle:
            if tensor_name in handle.keys():
                return path
    raise FileNotFoundError(
        f"Could not resolve base weight {tensor_name!r} under base model path {base_model_path!r}"
    )


@lru_cache(maxsize=4096)
def _base_weight_shape(base_model_path: str, tensor_name: str) -> tuple[int, ...]:
    path = _resolve_base_weight_file(base_model_path, tensor_name)
    with safe_open(path, framework="pt", device="cpu") as handle:
        if tensor_name not in handle.keys():
            raise KeyError(f"Base weight {tensor_name!r} not found in {path!r}")
        tensor = handle.get_tensor(tensor_name)
        return tuple(int(dim) for dim in tensor.shape)


def _is_tp_sharded_dim(base_dim: int, shard_dim: int, *, tp_size: int) -> bool:
    if base_dim <= 0 or shard_dim <= 0:
        return False
    if shard_dim > base_dim:
        return False
    if base_dim % shard_dim != 0:
        return False
    shard_factor = base_dim // shard_dim
    return 1 < shard_factor <= tp_size


def _is_tp_fused_qkv_dim(
    module_name: str,
    shard_dim: int,
    base_model_path: str,
    *,
    tp_size: int,
) -> bool:
    if not module_name.endswith(".self_attn.q_proj"):
        return False
    try:
        k_shape = _base_weight_shape(base_model_path, module_name[:-len("q_proj")] + "k_proj.weight")
        v_shape = _base_weight_shape(base_model_path, module_name[:-len("q_proj")] + "v_proj.weight")
        q_shape = _base_weight_shape(base_model_path, f"{module_name}.weight")
    except Exception:
        return False
    if len(q_shape) != 2 or len(k_shape) != 2 or len(v_shape) != 2:
        return False
    fused_out = int(q_shape[0]) + int(k_shape[0]) + int(v_shape[0])
    if fused_out <= 0 or fused_out % tp_size != 0:
        return False
    return int(shard_dim) == fused_out // tp_size


def validate_peft_adapter_checkpoint_shapes(
    adapter_dir: str,
    base_model_path: str,
    *,
    tensor_parallel_size: int | None = None,
    fully_sharded_loras: bool = False,
) -> None:
    """Fail fast for clearly invalid adapter checkpoints.

    Strict full-shape checks are valid for unsharded adapters.
    For fully-sharded LoRAs, vLLM accepts TP-local dimensions, so we permit
    dimensions and ranks that are exact divisors of the full checkpoint shape.
    """
    if not adapter_dir or not os.path.isdir(adapter_dir):
        raise ValueError(f"Adapter directory not found: {adapter_dir!r}")
    if not base_model_path or not os.path.isdir(base_model_path):
        raise ValueError(f"Base model path not found: {base_model_path!r}")

    weights_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.isfile(weights_path):
        raise ValueError(f"Adapter weights not found: {weights_path!r}")

    tp_size = max(1, int(tensor_parallel_size or 1))
    allow_tp_shards = bool(fully_sharded_loras and tp_size > 1)

    config_rank: int | None = None
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as handle:
            adapter_config = json.load(handle)
        if not isinstance(adapter_config, dict):
            raise ValueError(
                f"adapter_config.json must contain a JSON object, got {type(adapter_config).__name__}"
            )
        peft_type = adapter_config.get("peft_type")
        if "peft_type" in adapter_config and (peft_type is None or peft_type == ""):
            raise ValueError(
                "adapter_config.json missing PEFT adapter type: expected peft_type='LORA'"
            )
        if peft_type is not None and not (
            isinstance(peft_type, str) and peft_type.upper() == "LORA"
        ):
            raise ValueError(
                f"adapter_config.json has unsupported peft_type={peft_type!r}; expected 'LORA'"
            )
        raw_rank = adapter_config.get("r")
        if isinstance(raw_rank, int) and raw_rank > 0:
            config_rank = raw_rank

    module_shapes: dict[str, dict[str, tuple[int, ...]]] = {}
    errors: list[str] = []
    matched = 0

    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            match = _LORA_WEIGHT_RE.match(str(key))
            if match is None:
                continue
            matched += 1
            module_name = _normalize_lora_module_name(match.group("module"))
            kind = match.group("kind")
            tensor_shape = tuple(int(dim) for dim in handle.get_tensor(key).shape)
            if len(tensor_shape) != 2:
                errors.append(f"{key}: expected rank-2 tensor, got shape={tensor_shape}")
                continue

            base_weight_name = f"{module_name}.weight"
            try:
                base_shape = _base_weight_shape(base_model_path, base_weight_name)
            except Exception as exc:
                errors.append(f"{key}: could not resolve base weight {base_weight_name!r}: {exc}")
                continue

            if len(base_shape) != 2:
                errors.append(f"{key}: expected rank-2 base weight, got shape={base_shape}")
                continue

            expected_out, expected_in = int(base_shape[0]), int(base_shape[1])
            if kind == "lora_A" and tensor_shape[1] != expected_in:
                if not (
                    allow_tp_shards
                    and _is_tp_sharded_dim(expected_in, int(tensor_shape[1]), tp_size=tp_size)
                ):
                    errors.append(
                        f"{key}: lora_A input dim mismatch got={tensor_shape} expected=(*, {expected_in}) "
                        f"from {base_weight_name} shape={base_shape}"
                    )
            if kind == "lora_B" and tensor_shape[0] != expected_out:
                if not (
                    allow_tp_shards
                    and (
                        _is_tp_sharded_dim(expected_out, int(tensor_shape[0]), tp_size=tp_size)
                        or _is_tp_fused_qkv_dim(
                            module_name,
                            int(tensor_shape[0]),
                            base_model_path,
                            tp_size=tp_size,
                        )
                    )
                ):
                    errors.append(
                        f"{key}: lora_B output dim mismatch got={tensor_shape} expected=({expected_out}, *) "
                        f"from {base_weight_name} shape={base_shape}"
                    )
            module_shapes.setdefault(module_name, {})[kind] = tensor_shape

    if matched == 0:
        raise ValueError(f"No LoRA tensors found in adapter checkpoint: {weights_path!r}")

    for module_name, shapes in module_shapes.items():
        lora_a_shape = shapes.get("lora_A")
        lora_b_shape = shapes.get("lora_B")
        if lora_a_shape is not None and lora_b_shape is not None and lora_a_shape[0] != lora_b_shape[1]:
            if not (
                allow_tp_shards
                and _is_tp_sharded_dim(int(lora_b_shape[1]), int(lora_a_shape[0]), tp_size=tp_size)
            ):
                errors.append(
                    f"{module_name}: rank mismatch between lora_A={lora_a_shape} and lora_B={lora_b_shape}"
                )
        if config_rank is not None:
            if lora_a_shape is not None and lora_a_shape[0] != config_rank:
                if not (
                    allow_tp_shards
                    and _is_tp_sharded_dim(int(config_rank), int(lora_a_shape[0]), tp_size=tp_size)
                ):
                    errors.append(
                        f"{module_name}: adapter_config.r={config_rank} but lora_A shape={lora_a_shape}"
                    )
            if lora_b_shape is not None and lora_b_shape[1] != config_rank:
                if not (
                    allow_tp_shards
                    and _is_tp_sharded_dim(int(config_rank), int(lora_b_shape[1]), tp_size=tp_size)
                ):
                    errors.append(
                        f"{module_name}: adapter_config.r={config_rank} but lora_B shape={lora_b_shape}"
                    )

    if errors:
        preview = errors[:16]
        remaining = len(errors) - len(preview)
        if remaining > 0:
            preview.append(f"... {remaining} more shape errors")
        raise ValueError(
            "PEFT adapter shape validation failed against the base model:\n" + "\n".join(preview)
        )



def maybe_validate_peft_adapter_checkpoint_shapes(
    adapter_dir: str,
    base_model_path: str,
    *,
    tensor_parallel_size: int | None = None,
    fully_sharded_loras: bool = False,
) -> None:
    """Run shape validation unless the explicit runtime bypass flag is set."""
    if _env_flag("MINT_VLLM_SKIP_PEFT_SHAPE_VALIDATION", default=False):
        return
    validate_peft_adapter_checkpoint_shapes(
        adapter_dir,
        base_model_path,
        tensor_parallel_size=tensor_parallel_size,
        fully_sharded_loras=fully_sharded_loras,
    )



def pad_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
    actual_rank: int,
    trainer_rank: int,
) -> dict[str, torch.Tensor]:
    """Pad LoRA weights from actual_rank to trainer_rank.

    LoRA A matrices: (actual_rank, hidden) -> (trainer_rank, hidden)
    LoRA B matrices: (hidden, actual_rank) -> (hidden, trainer_rank)

    Zero-padding preserves mathematical equivalence: extra dimensions contribute
    nothing to the output since they're multiplied by zeros.

    Args:
        state_dict: Adapter state dict with lora_A and lora_B weights.
        actual_rank: The rank of the loaded checkpoint.
        trainer_rank: The rank the trainer was initialized with.

    Returns:
        New state dict with padded weights.
    """
    actual_rank, trainer_rank = _validate_rank_pair(actual_rank, trainer_rank)
    if actual_rank == trainer_rank:
        return state_dict  # No padding needed

    padded = {}
    for name, tensor in state_dict.items():
        if _is_lora_a_name(name):
            # lora_A: (actual_rank, hidden) -> (trainer_rank, hidden)
            # Pad rows
            rank_dim = int(tensor.shape[0])
            if rank_dim == trainer_rank:
                padded[name] = tensor
                continue
            if rank_dim != actual_rank:
                raise ValueError(
                    f"{name}: lora_A/linear_in rank dim is {rank_dim}, "
                    f"expected actual_rank {actual_rank} or trainer_rank {trainer_rank}"
                )
            padded_tensor = torch.zeros(
                trainer_rank, *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device
            )
            padded_tensor[:actual_rank] = tensor
            padded[name] = padded_tensor
        elif _is_lora_b_name(name):
            # lora_B: (hidden, actual_rank) -> (hidden, trainer_rank)
            # Pad columns
            rank_dim = int(tensor.shape[-1])
            if rank_dim == trainer_rank:
                padded[name] = tensor
                continue
            if rank_dim != actual_rank:
                raise ValueError(
                    f"{name}: lora_B/linear_out rank dim is {rank_dim}, "
                    f"expected actual_rank {actual_rank} or trainer_rank {trainer_rank}"
                )
            padded_tensor = torch.zeros(
                *tensor.shape[:-1], trainer_rank, dtype=tensor.dtype, device=tensor.device
            )
            padded_tensor[..., :actual_rank] = tensor
            padded[name] = padded_tensor
        else:
            # Non-LoRA parameters pass through unchanged
            padded[name] = tensor

    return padded


def truncate_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
    trainer_rank: int,
    actual_rank: int,
) -> dict[str, torch.Tensor]:
    """Truncate LoRA weights from trainer_rank to actual_rank.

    LoRA A matrices: (trainer_rank, hidden) -> (actual_rank, hidden)
    LoRA B matrices: (hidden, trainer_rank) -> (hidden, actual_rank)

    Truncation discards the extra rows/columns that were zero-padded during loading.

    Args:
        state_dict: Adapter state dict with padded lora_A and lora_B weights.
        trainer_rank: The rank the trainer was initialized with.
        actual_rank: The rank to save the checkpoint as.

    Returns:
        New state dict with truncated weights.
    """
    actual_rank, trainer_rank = _validate_rank_pair(actual_rank, trainer_rank)
    if actual_rank == trainer_rank:
        return state_dict  # No truncation needed

    truncated = {}
    for name, tensor in state_dict.items():
        if _is_lora_a_name(name):
            # lora_A: (trainer_rank, hidden) -> (actual_rank, hidden)
            # Truncate rows
            rank_dim = int(tensor.shape[0])
            if rank_dim == actual_rank:
                truncated[name] = tensor.clone()
                continue
            if rank_dim != trainer_rank:
                raise ValueError(
                    f"{name}: lora_A/linear_in rank dim is {rank_dim}, "
                    f"expected actual_rank {actual_rank} or trainer_rank {trainer_rank}"
                )
            truncated[name] = tensor[:actual_rank].clone()
        elif _is_lora_b_name(name):
            # lora_B: (hidden, trainer_rank) -> (hidden, actual_rank)
            # Truncate columns
            rank_dim = int(tensor.shape[-1])
            if rank_dim == actual_rank:
                truncated[name] = tensor.clone()
                continue
            if rank_dim != trainer_rank:
                raise ValueError(
                    f"{name}: lora_B/linear_out rank dim is {rank_dim}, "
                    f"expected actual_rank {actual_rank} or trainer_rank {trainer_rank}"
                )
            truncated[name] = tensor[..., :actual_rank].clone()
        else:
            # Non-LoRA parameters pass through unchanged
            truncated[name] = tensor

    return truncated


def fit_lora_state_dict_to_reference(
    state_dict: dict[str, torch.Tensor],
    reference_state_dict: dict[str, torch.Tensor],
    *,
    rank_shard_index: int = 0,
    rank_shard_count: int = 1,
) -> dict[str, torch.Tensor]:
    """Resize LoRA rank dimensions to match a reference state dict.

    This is for Megatron TP-local adapter shards, where a checkpoint may store
    rank-64 tensors while the local model rank dimension is rank-64 / TP.
    Non-rank dimensions must already match.
    """
    rank_shard_index = int(rank_shard_index)
    rank_shard_count = int(rank_shard_count)
    if rank_shard_index < 0:
        raise ValueError(f"rank_shard_index must be non-negative, got {rank_shard_index}")
    if rank_shard_count <= 0:
        raise ValueError(f"rank_shard_count must be positive, got {rank_shard_count}")
    if rank_shard_index >= rank_shard_count:
        raise ValueError(
            f"rank_shard_index {rank_shard_index} must be smaller than rank_shard_count {rank_shard_count}"
        )
    fitted = {}
    for name, tensor in state_dict.items():
        reference = reference_state_dict.get(name)
        if reference is None or not (_is_lora_a_name(name) or _is_lora_b_name(name)):
            fitted[name] = tensor
            continue
        target_shape = tuple(int(dim) for dim in reference.shape)
        source_shape = tuple(int(dim) for dim in tensor.shape)
        if source_shape == target_shape:
            fitted[name] = tensor
            continue
        if tensor.ndim != reference.ndim:
            raise ValueError(
                f"{name}: LoRA tensor rank mismatch, checkpoint shape={source_shape}, "
                f"expected shape={target_shape}"
            )
        rank_axis = 0 if _is_lora_a_name(name) else tensor.ndim - 1
        source_non_rank = source_shape[:rank_axis] + source_shape[rank_axis + 1 :]
        target_non_rank = target_shape[:rank_axis] + target_shape[rank_axis + 1 :]
        if source_non_rank != target_non_rank:
            raise ValueError(
                f"{name}: LoRA non-rank dimensions mismatch, checkpoint shape={source_shape}, "
                f"expected shape={target_shape}"
            )
        output = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
        local_rank_dim = target_shape[rank_axis]
        shard_start = rank_shard_index * local_rank_dim
        shard_stop = shard_start + local_rank_dim
        source_rank_dim = source_shape[rank_axis]
        represented_rank_dim = local_rank_dim * rank_shard_count
        if source_rank_dim > represented_rank_dim:
            raise ValueError(
                f"{name}: checkpoint LoRA rank dim {source_rank_dim} exceeds represented trainer rank "
                f"{represented_rank_dim} for rank_shard_count {rank_shard_count}"
            )
        if source_rank_dim < shard_start:
            overlap = 0
        else:
            overlap = max(0, min(shard_stop, source_rank_dim) - shard_start)
        source_slice = [slice(None)] * tensor.ndim
        target_slice = [slice(None)] * tensor.ndim
        source_slice[rank_axis] = slice(shard_start, shard_start + overlap)
        target_slice[rank_axis] = slice(0, overlap)
        if overlap:
            output[tuple(target_slice)] = tensor[tuple(source_slice)]
        fitted[name] = output
    return fitted


def compute_lora_scaling(
    trainer_rank: int,
    actual_rank: int,
    alpha: float | None = None,
) -> float:
    """Compute scaling factor for LoRA output adjustment.

    LoRA output: (lora_B @ lora_A @ x) * (alpha / rank)

    When using a padded trainer_rank > actual_rank:
    - If alpha scales with trainer_rank: output = lora_B @ lora_A @ x * (alpha / trainer_rank)
    - But we want: output = lora_B @ lora_A @ x * (alpha / actual_rank)
    - So multiply by: trainer_rank / actual_rank

    Args:
        trainer_rank: The rank the trainer was initialized with.
        actual_rank: The actual rank of the adapter being trained.
        alpha: Optional custom alpha. If None, assumes alpha = 2 * trainer_rank.

    Returns:
        Scaling factor to multiply LoRA output by.
    """
    actual_rank, trainer_rank = _validate_rank_pair(actual_rank, trainer_rank)
    if actual_rank == trainer_rank:
        return 1.0  # No scaling needed

    # Scaling correction: trainer_rank / actual_rank
    return trainer_rank / actual_rank


def get_lora_rank_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int | None:
    """Infer LoRA rank from state dict by validating all LoRA tensor shapes.

    Args:
        state_dict: Adapter state dict.

    Returns:
        Inferred rank, or None if no LoRA parameters found.
    """
    ranks: list[tuple[str, int]] = []
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if (_is_lora_a_name(name) or _is_lora_b_name(name)) and tensor.ndim < 2:
            raise ValueError(f"{name}: expected rank-2+ LoRA tensor, got shape={tuple(tensor.shape)}")
        if _is_lora_a_name(name):
            ranks.append((name, int(tensor.shape[0])))
        elif _is_lora_b_name(name):
            ranks.append((name, int(tensor.shape[-1])))
    if not ranks:
        return None
    rank_values = {rank for _, rank in ranks}
    if len(rank_values) != 1:
        raise ValueError(f"LoRA tensor rank mismatch: {ranks}")
    return ranks[0][1]


__all__ = [
    "pad_lora_state_dict",
    "truncate_lora_state_dict",
    "fit_lora_state_dict_to_reference",
    "compute_lora_scaling",
    "get_lora_rank_from_state_dict",
    "lora_rank_tail_slice",
    "maybe_validate_peft_adapter_checkpoint_shapes",
    "validate_peft_adapter_checkpoint_shapes",
    "zero_lora_rank_tail_named_parameters",
]
