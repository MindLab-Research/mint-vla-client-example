from __future__ import annotations

import math
import dataclasses
from pathlib import Path
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


def test_compute_target_logprobs_uses_eval_mode_and_disables_train_augmentation() -> None:
    seen: dict[str, object] = {}

    def _log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
        shifted = logits - np.max(logits, axis=axis, keepdims=True)
        return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))

    def _preprocess_observation(rng, observation, *, train, image_keys):
        del rng
        seen["train"] = train
        seen["image_keys"] = list(image_keys)
        return observation

    class _FakeLLM:
        def __call__(self, *, embedded_prefix=None, mask=None, return_prelogits=False, pre_logits=None):
            del embedded_prefix, mask
            if return_prelogits:
                return np.zeros((1, 2, 3), dtype=np.float32), None, None
            del pre_logits
            logits = np.asarray(
                [[[0.0, 2.0, -1.0, -2.0], [0.0, -3.0, 1.0, -2.0]]],
                dtype=np.float32,
            )
            return logits, None

    class _FakeModel:
        def __init__(self) -> None:
            self.PaliGemma = SimpleNamespace(llm=_FakeLLM())

        def eval(self) -> None:
            seen["eval_called"] = True

        def embed_inputs(self, observation):
            token_count = observation.tokenized_prompt.shape[1]
            embeddings = np.zeros((1, token_count, 3), dtype=np.float32)
            mask = np.ones((1, token_count), dtype=np.bool_)
            ar_mask = np.zeros((1, token_count), dtype=np.int32)
            return embeddings, mask, ar_mask

    fake_session = SimpleNamespace(
        _openpi_model=SimpleNamespace(preprocess_observation=_preprocess_observation),
        _openpi_pi0_fast=SimpleNamespace(make_attn_mask=lambda input_mask, ar_mask: np.ones((1, 3, 3), dtype=np.bool_)),
        _jax=SimpleNamespace(nn=SimpleNamespace(log_softmax=_log_softmax)),
        _jnp=np,
    )
    observation = SimpleNamespace(
        images={"base_0_rgb": np.zeros((1, 2, 2, 3), dtype=np.uint8)},
        tokenized_prompt=np.asarray([[0, 1, 2]], dtype=np.int32),
    )

    result = worker_module.OpenPIFastWorkerSession._compute_target_logprobs(
        fake_session,
        _FakeModel(),
        rng=None,
        observation=observation,
    )

    assert seen["eval_called"] is True
    assert seen["train"] is False
    assert seen["image_keys"] == ["base_0_rgb"]
    assert np.asarray(result).shape == (1, 2)


def test_load_weights_accepts_sampler_checkpoint_when_optimizer_restore_is_disabled(monkeypatch, tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "sampler_ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "metadata.json").write_text('{"step": 12}', encoding="utf-8")

    seen: dict[str, object] = {}

    class _FakeParams:
        def __init__(self) -> None:
            self.replaced = None

        def replace_by_pure_dict(self, payload):
            self.replaced = payload

        def filter(self, _filter):
            return {"filtered": True}

    class _FakeTx:
        def init(self, payload):
            seen["tx_init_payload"] = payload
            return {"opt": "fresh"}

    fake_params = _FakeParams()
    fake_ema_params = _FakeParams()
    @dataclasses.dataclass
    class _FakeState:
        step: int
        params: object
        ema_params: object
        tx: object
        opt_state: object | None = None

    fake_state = _FakeState(step=3, params=fake_params, ema_params=fake_ema_params, tx=_FakeTx())
    fake_session = SimpleNamespace(
        _state=fake_state,
        _config=SimpleNamespace(trainable_filter="trainable"),
        _openpi_model=SimpleNamespace(restore_params=lambda path, dtype=None: {"restored_from": str(path), "dtype": str(dtype)}),
        _jnp=SimpleNamespace(bfloat16="bf16"),
        _learning_rate=1e-4,
        _load_train_state_checkpoint=lambda path: (_ for _ in ()).throw(AssertionError("training checkpoint path should not run")),
    )
    def _fake_init_train_state(*, partial_params, step):
        fake_params.replace_by_pure_dict(partial_params)
        fake_ema_params.replace_by_pure_dict(partial_params)
        tx = _FakeTx()
        opt_state = tx.init({"filtered": True})
        return _FakeState(
            step=step,
            params=fake_params,
            ema_params=fake_ema_params,
            tx=tx,
            opt_state=opt_state,
        )

    fake_session._init_train_state = _fake_init_train_state
    fake_session._load_policy_weights_checkpoint = lambda path: worker_module.OpenPIFastWorkerSession._load_policy_weights_checkpoint(fake_session, path)

    monkeypatch.setattr(worker_module, "checkpoint_has_openpi_training_state", lambda path: False)
    monkeypatch.setattr(worker_module, "find_openpi_policy_checkpoint_dir", lambda path: Path(path))
    monkeypatch.setattr(worker_module, "checkpoint_has_openpi_policy_weights", lambda path: True)

    result = worker_module.OpenPIFastWorkerSession.load_weights(
        fake_session,
        {"load_path": str(ckpt_dir), "load_optimizer": False},
    )

    assert result == {"current_step": 12, "learning_rate": 1e-4}
    assert fake_params.replaced == {"restored_from": str(ckpt_dir / "params"), "dtype": "bf16"}
    assert fake_ema_params.replaced == {"restored_from": str(ckpt_dir / "params"), "dtype": "bf16"}
    assert seen["tx_init_payload"] == {"filtered": True}
    assert fake_session._state.step == 12
    assert fake_session._state.opt_state == {"opt": "fresh"}


def test_load_weights_keeps_training_checkpoint_path_for_optimizer_restore(monkeypatch, tmp_path: Path) -> None:
    fake_session = SimpleNamespace(
        _state=SimpleNamespace(step=0),
        _learning_rate=2e-4,
        _load_train_state_checkpoint=lambda path: SimpleNamespace(step=7, marker=str(path)),
    )

    monkeypatch.setattr(worker_module, "checkpoint_has_openpi_training_state", lambda path: False)

    result = worker_module.OpenPIFastWorkerSession.load_weights(
        fake_session,
        {"load_path": str(tmp_path / "sampler_ckpt"), "load_optimizer": True},
    )

    assert result == {"current_step": 7, "learning_rate": 2e-4}
    assert fake_session._state.marker == str(tmp_path / "sampler_ckpt")
