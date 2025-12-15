"""Utilities for LoRA adapter manipulation.

Provides functions for padding/truncating LoRA weights to support unified rank
training where a trainer with max_rank can train adapters with any rank <= max_rank.
"""

import torch


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
]
