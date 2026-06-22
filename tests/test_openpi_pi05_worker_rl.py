from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

import mint_server.backend.openpi.openpi_pi05_worker as worker_module


def test_compute_pi05_importance_sampling_stats() -> None:
    stats = worker_module._compute_importance_sampling_stats(
        current_logprobs=np.asarray([[-0.2, -0.1]], dtype=np.float32),
        old_logprobs=np.asarray([[-0.3, -0.3]], dtype=np.float32),
        advantages=np.asarray([[1.0, -2.0]], dtype=np.float32),
    )

    expected_ratio_0 = math.exp(0.1)
    expected_ratio_1 = math.exp(0.2)
    expected_loss = -((expected_ratio_0 * 1.0) + (expected_ratio_1 * -2.0))

    assert stats["loss"] == pytest.approx(expected_loss)
    assert stats["ratio_mean"] == pytest.approx((expected_ratio_0 + expected_ratio_1) / 2.0)
    assert stats["action_count"] == 2


def test_compute_pi05_ppo_stats_reports_clipfrac() -> None:
    stats = worker_module._compute_ppo_stats(
        current_logprobs=np.asarray([[-0.2, -0.1]], dtype=np.float32),
        old_logprobs=np.asarray([[-0.3, -0.3]], dtype=np.float32),
        advantages=np.asarray([[1.0, 1.0]], dtype=np.float32),
        clip_low=0.9,
        clip_high=1.15,
    )

    expected_ratio_0 = math.exp(0.1)
    expected_loss = -expected_ratio_0 - 1.15

    assert stats["loss"] == pytest.approx(expected_loss)
    assert stats["ratio_mean"] == pytest.approx((math.exp(0.1) + math.exp(0.2)) / 2.0)
    assert stats["clipfrac_mean"] == pytest.approx(0.5)
    assert stats["action_count"] == 2


def test_pi05_forward_backward_accepts_ppo_and_reports_logprobs() -> None:
    class _FakeTree:
        @staticmethod
        def map(fn, a, b):
            return fn(a, b)

    fake_session = SimpleNamespace(
        _pending_grads=None,
        _jax=SimpleNamespace(tree=_FakeTree()),
        _action_horizon=2,
        _rl_observation_from_payload=lambda item: (
            "obs",
            "chains",
            np.asarray(item["old_logprobs"], dtype=np.float32).reshape(2, 1),
            np.asarray(item["advantages"], dtype=np.float32).reshape(2, 1),
        ),
        _observation_from_payload=lambda item: (_ for _ in ()).throw(AssertionError("flow_matching path should not run")),
        _compute_grads=lambda observation, actions: (_ for _ in ()).throw(AssertionError("flow_matching path should not run")),
        _compute_importance_sampling_grads=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("importance_sampling path should not run")
        ),
        _compute_ppo_grads=lambda observation, chains, old_logprobs, advantages, item, loss_fn_config: (
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

    result = worker_module.OpenPIPi05WorkerSession.forward_backward(
        fake_session,
        {
            "loss_fn": "ppo",
            "loss_fn_config": {"epsilon": 0.2, "noise_method": "flow_sde"},
            "batch": [
                {
                    "old_logprobs": [-0.3, -0.3],
                    "advantages": [1.0, 1.0],
                    "source_action_dim": 1,
                    "denoise_inds": [0],
                }
            ],
        },
    )

    assert result["loss_fn_output_type"] == "ppo_loss"
    assert result["metrics"]["ratio:mean"] == pytest.approx(1.1)
    assert result["metrics"]["clipfrac:mean"] == pytest.approx(0.25)
    assert result["loss_fn_outputs"][0]["logprobs"]["data"] == [-0.1, -0.2]
    assert result["loss_fn_outputs"][0]["logprobs"]["shape"] == [2, 1]
