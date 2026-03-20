from __future__ import annotations

import math
from types import SimpleNamespace

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


def test_compute_ppo_stats_reports_clipped_loss_and_clipfrac() -> None:
    stats = worker_module._compute_ppo_stats(
        current_logprobs=np.asarray([-0.2, -0.1, -9.0], dtype=np.float32),
        old_logprobs=np.asarray([-0.3, -0.3, -9.0], dtype=np.float32),
        advantages=np.asarray([1.0, 1.0, 999.0], dtype=np.float32),
        loss_mask=np.asarray([True, True, False], dtype=np.bool_),
        clip_low=0.9,
        clip_high=1.15,
    )

    expected_ratio_0 = math.exp(0.1)
    expected_ratio_1 = math.exp(0.2)
    expected_loss = -expected_ratio_0 - 1.15

    assert stats["loss"] == pytest.approx(expected_loss)
    assert stats["ratio_mean"] == pytest.approx((expected_ratio_0 + expected_ratio_1) / 2.0)
    assert stats["clipfrac_mean"] == pytest.approx(0.5)
    assert stats["token_count"] == 2


def test_forward_backward_accepts_ppo_and_reports_clipfrac() -> None:
    class _FakeTree:
        @staticmethod
        def map(fn, a, b):
            return fn(a, b)

    fake_session = SimpleNamespace(
        _pending_grads=None,
        _jax=SimpleNamespace(tree=_FakeTree()),
        _observation_from_payload=lambda item: ("obs", "act"),
        _compute_grads=lambda observation, actions: (_ for _ in ()).throw(AssertionError("ce path should not run")),
        _compute_importance_sampling_grads=lambda observation, actions, item: (_ for _ in ()).throw(
            AssertionError("importance_sampling path should not run")
        ),
        _compute_ppo_grads=lambda observation, actions, item, loss_fn_config: (
            "grads",
            -1.25,
            0.3,
            0.4,
            1.1,
            0.25,
            2.0,
            [-0.1, -0.2],
        ),
    )

    result = worker_module.OpenPIFastWorkerSession.forward_backward(
        fake_session,
        {
            "loss_fn": "ppo",
            "loss_fn_config": {"epsilon": 0.2},
            "batch": [{"token_loss_mask": [False, False, False, True, True]}],
        },
    )

    assert result["loss_fn_output_type"] == "ppo_loss"
    assert result["metrics"]["ratio:mean"] == pytest.approx(1.1)
    assert result["metrics"]["clipfrac:mean"] == pytest.approx(0.25)
    assert result["loss_fn_outputs"][0]["logprobs"]["data"] == [-0.1, -0.2]
