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


def validate_peft_adapter_checkpoint_shapes(adapter_dir: str, base_model_path: str) -> None:
    """Fail fast when adapter tensors do not match the base model module shapes."""
    if not adapter_dir or not os.path.isdir(adapter_dir):
        raise ValueError(f"Adapter directory not found: {adapter_dir!r}")
    if not base_model_path or not os.path.isdir(base_model_path):
        raise ValueError(f"Base model path not found: {base_model_path!r}")

    weights_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.isfile(weights_path):
        raise ValueError(f"Adapter weights not found: {weights_path!r}")

    config_rank: int | None = None
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as handle:
            adapter_config = json.load(handle)
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
                errors.append(
                    f"{key}: lora_A input dim mismatch got={tensor_shape} expected=(*, {expected_in}) "
                    f"from {base_weight_name} shape={base_shape}"
                )
            if kind == "lora_B" and tensor_shape[0] != expected_out:
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
            errors.append(
                f"{module_name}: rank mismatch between lora_A={lora_a_shape} and lora_B={lora_b_shape}"
            )
        if config_rank is not None:
            if lora_a_shape is not None and lora_a_shape[0] != config_rank:
                errors.append(
                    f"{module_name}: adapter_config.r={config_rank} but lora_A shape={lora_a_shape}"
                )
            if lora_b_shape is not None and lora_b_shape[1] != config_rank:
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
    if actual_rank >= trainer_rank:
        return state_dict  # No padding needed

    padded = {}
    for name, tensor in state_dict.items():
        if "lora_a" in name.lower():
            # lora_A: (actual_rank, hidden) -> (trainer_rank, hidden)
            # Pad rows
            padded_tensor = torch.zeros(
                trainer_rank, tensor.shape[1], dtype=tensor.dtype, device=tensor.device
            )
            padded_tensor[:actual_rank] = tensor
            padded[name] = padded_tensor
        elif "lora_b" in name.lower():
            # lora_B: (hidden, actual_rank) -> (hidden, trainer_rank)
            # Pad columns
            padded_tensor = torch.zeros(
                tensor.shape[0], trainer_rank, dtype=tensor.dtype, device=tensor.device
            )
            padded_tensor[:, :actual_rank] = tensor
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
    if actual_rank >= trainer_rank:
        return state_dict  # No truncation needed

    truncated = {}
    for name, tensor in state_dict.items():
        if "lora_a" in name.lower():
            # lora_A: (trainer_rank, hidden) -> (actual_rank, hidden)
            # Truncate rows
            truncated[name] = tensor[:actual_rank].clone()
        elif "lora_b" in name.lower():
            # lora_B: (hidden, trainer_rank) -> (hidden, actual_rank)
            # Truncate columns
            truncated[name] = tensor[:, :actual_rank].clone()
        else:
            # Non-LoRA parameters pass through unchanged
            truncated[name] = tensor

    return truncated


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
    if actual_rank >= trainer_rank:
        return 1.0  # No scaling needed

    # Scaling correction: trainer_rank / actual_rank
    return trainer_rank / actual_rank


def get_lora_rank_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int | None:
    """Infer LoRA rank from state dict by examining tensor shapes.

    Args:
        state_dict: Adapter state dict.

    Returns:
        Inferred rank, or None if no LoRA parameters found.
    """
    for name, tensor in state_dict.items():
        if "lora_a" in name.lower():
            # lora_A has shape (rank, hidden)
            return tensor.shape[0]
        elif "lora_b" in name.lower():
            # lora_B has shape (hidden, rank)
            return tensor.shape[1]
    return None


__all__ = [
    "pad_lora_state_dict",
    "truncate_lora_state_dict",
    "compute_lora_scaling",
    "get_lora_rank_from_state_dict",
    "validate_peft_adapter_checkpoint_shapes",
]
