from __future__ import annotations

import math

import numpy as np
import pytest

import tinker_server.backend.openpi_fast_worker as worker_module


def test_compute_importance_sampling_stats_uses_only_masked_tokens() -> None:
    stats = worker_module._compute_importance_sampling_stats(
        current_logprobs=np.asarray([-0.2, -0.1, -9.0], dtype=np.float32),
        old_logprobs=np.asarray([-0.3, -0.3, -9.0], dtype=np.float32),
        advantages=np.asarray([1.0, -2.0, 999.0], dtype=np.float32),
        loss_mask=np.asarray([True, True, False], dtype=np.bool_),
    )

    expected_ratio_0 = math.exp(0.1)
    expected_ratio_1 = math.exp(0.2)
    expected_loss = -((expected_ratio_0 * 1.0) + (expected_ratio_1 * -2.0))

    assert stats["loss"] == pytest.approx(expected_loss)
    assert stats["ratio_mean"] == pytest.approx((expected_ratio_0 + expected_ratio_1) / 2.0)
    assert stats["token_count"] == 2


def test_compute_importance_sampling_stats_rejects_fully_masked_inputs() -> None:
    with pytest.raises(ValueError, match="masked"):
        worker_module._compute_importance_sampling_stats(
            current_logprobs=np.asarray([-0.2, -0.1], dtype=np.float32),
            old_logprobs=np.asarray([-0.3, -0.3], dtype=np.float32),
            advantages=np.asarray([1.0, -2.0], dtype=np.float32),
            loss_mask=np.asarray([False, False], dtype=np.bool_),
        )


def test_compute_importance_sampling_stats_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        worker_module._compute_importance_sampling_stats(
            current_logprobs=np.asarray([-0.2], dtype=np.float32),
            old_logprobs=np.asarray([-0.3, -0.3], dtype=np.float32),
            advantages=np.asarray([1.0, -2.0], dtype=np.float32),
            loss_mask=np.asarray([True, True], dtype=np.bool_),
        )
