"""Debug metrics for verifying precision differences between inference and training.

Based on verl's verl/utils/debug/metrics.py implementation.

These metrics help detect precision mismatches between:
- Rollout (inference engine, e.g., vLLM) log probabilities
- Actor (training forward pass, e.g., Megatron) log probabilities

Such mismatches can indicate:
- Precision issues from different attention implementations (FA2 vs xformers)
- Numerical instability in long sequences
- Implementation differences between vLLM and Megatron backends

Expected values under normal circumstances:
- training/rollout_probs_diff_mean < 0.005
- training/rollout_probs_diff_max < 0.01

If values exceed 0.01, check:
- GPU architecture (A100 vs H100)
- vLLM cascade attention settings
- Sequence length
"""

import structlog

import torch

logger = structlog.get_logger(__name__)


def calculate_token_list_diff(
    tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Calculate number of differing tokens between two sequences (with mask).

    Args:
        tensor1: First token tensor [batch, seq_len]
        tensor2: Second token tensor [batch, seq_len]
        mask: Boolean mask [batch, seq_len], True for valid positions

    Returns:
        Tensor of diff counts per sample [batch]
    """
    # Verify inputs
    if tensor1.numel() == 0 or tensor2.numel() == 0:
        return torch.zeros(tensor1.shape[0], dtype=torch.long, device=tensor1.device)

    if tensor1.shape != tensor2.shape or mask.shape != tensor1.shape:
        logger.warning(
            f"Tensor shape mismatch: tensor1={tensor1.shape}, "
            f"tensor2={tensor2.shape}, mask={mask.shape}"
        )
        return torch.ones_like(tensor1)

    # Transfer to same device
    if tensor2.device != tensor1.device:
        tensor2 = tensor2.to(tensor1.device)
    if mask.device != tensor1.device:
        mask = mask.to(tensor1.device)

    # Calculate diff
    diff_mask = tensor1 != tensor2
    valid_diff_mask = diff_mask & (mask == 1)
    diff_counts = valid_diff_mask.sum(dim=1)

    return diff_counts


def pearson_correlation_coefficient(
    tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor
) -> float:
    """Calculate Pearson correlation coefficient between two tensors.

    Reference: https://arxiv.org/pdf/2506.13585

    Args:
        tensor1: First tensor [batch, seq_len]
        tensor2: Second tensor [batch, seq_len]
        mask: Boolean mask [batch, seq_len]

    Returns:
        Pearson correlation coefficient (scalar)
    """
    if tensor1.shape != tensor2.shape or mask.shape != tensor1.shape:
        return 0.0

    # Mask and flatten
    mt1 = torch.masked_select(tensor1, mask)
    mt2 = torch.masked_select(tensor2, mask)

    # Calculate correlation
    # Handle edge case: if either tensor is constant, corrcoef returns NaN
    if mt1.numel() < 2 or mt2.numel() < 2:
        return 0.0

    result = torch.corrcoef(torch.stack([mt1, mt2], dim=0))
    corr_value = result[0][1].detach().item()

    # Handle NaN (constant tensors have variance 0)
    if corr_value != corr_value:  # NaN check
        return 0.0
    return corr_value


def calculate_log_prob_diff(
    log_probs1: torch.Tensor, log_probs2: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Calculate absolute log probability differences (masked).

    Args:
        log_probs1: First log probabilities [batch, seq_len]
        log_probs2: Second log probabilities [batch, seq_len]
        mask: Boolean mask [batch, seq_len]

    Returns:
        Flattened tensor of absolute differences (only valid positions)
    """
    full_diff = torch.abs(log_probs1 - log_probs2)
    return torch.masked_select(full_diff, mask)


def calculate_debug_metrics(batch: dict) -> dict:
    """Calculate rollout vs actor log probability differences for debugging.

    Compares:
    - log_probs: From inference engine (vLLM) during generation
    - old_log_probs: From training actor (Megatron) forward pass

    Args:
        batch: Dictionary with fields:
            - log_probs: Log probs from rollout [batch, seq_len]
            - old_log_probs: Log probs from actor [batch, seq_len]
            - response_mask or attention_mask: Mask for valid positions
            - responses: Response tokens for length calculation

    Returns:
        Dictionary with metrics:
            - training/rollout_probs_diff_valid: 1 if valid, 0 if invalid input
            - training/rollout_probs_diff_max: Max absolute difference
            - training/rollout_probs_diff_mean: Mean absolute difference
            - training/rollout_probs_diff_std: Standard deviation of differences
            - training/rollout_actor_probs_pearson_corr: Pearson correlation

        Empty dict if log_probs not present.
    """
    # Check if log_probs is present
    if "log_probs" not in batch:
        logger.debug("log_probs not in batch, skipping debug metrics")
        return {}

    rollout_old_log_probs = batch["log_probs"]
    actor_old_log_probs = batch["old_log_probs"]

    # Get mask for valid positions
    if "response_mask" in batch:
        logger.debug("Using response_mask for log prob comparison")
        log_prob_mask = batch["response_mask"]
    elif "attention_mask" in batch:
        logger.debug("Using attention_mask for log prob comparison")
        log_prob_mask = batch["attention_mask"]
    else:
        logger.warning(
            f"No mask found in batch (keys: {list(batch.keys())}), using all positions"
        )
        log_prob_mask = torch.ones_like(rollout_old_log_probs)

    # Get response tokens for length calculation
    responses = batch.get("responses")
    if responses is not None:
        response_length = responses.size(1)
        # Only compare on response tokens (not prompt)
        response_mask = log_prob_mask[:, -response_length:]
    else:
        # No response info, use full mask
        response_mask = log_prob_mask

    # Calculate pearson correlation
    actor_probs = torch.exp(actor_old_log_probs)
    rollout_probs = torch.exp(rollout_old_log_probs)
    response_mask_bool = response_mask.bool()

    pearson_corrcoef = pearson_correlation_coefficient(
        actor_probs, rollout_probs, response_mask_bool
    )
    rollout_probs_diff = calculate_log_prob_diff(
        actor_probs, rollout_probs, response_mask_bool
    )

    # Handle empty tensor case (no valid positions to compare)
    if rollout_probs_diff.numel() == 0:
        logger.warning("rollout_probs_diff is empty (no valid positions), returning zero metrics")
        return {
            "training/rollout_probs_diff_valid": 0,
            "training/rollout_probs_diff_max": 0.0,
            "training/rollout_probs_diff_mean": 0.0,
            "training/rollout_probs_diff_std": 0.0,
            "training/rollout_actor_probs_pearson_corr": 0.0,
        }

    # std requires at least 2 elements, otherwise returns nan
    std_value = torch.std(rollout_probs_diff).detach().item() if rollout_probs_diff.numel() > 1 else 0.0

    return {
        "training/rollout_probs_diff_valid": 1,
        "training/rollout_probs_diff_max": torch.max(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_mean": torch.mean(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_std": std_value,
        "training/rollout_actor_probs_pearson_corr": pearson_corrcoef,
    }
