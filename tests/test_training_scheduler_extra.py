from __future__ import annotations

from types import SimpleNamespace

from tinker_server.routes.training import _build_training_scheduler_extra


def test_openpi_train_step_scheduler_extra_forces_session_rotation(monkeypatch):
    monkeypatch.delenv("MINT_SCHEDULER_ENABLE", raising=False)
    session = SimpleNamespace(
        backend="openpi_fast",
        base_model="openpi/pi0-fast-libero-low-mem-finetune",
    )

    extra = _build_training_scheduler_extra(
        session=session,
        model_id="model-1",
        training_op="train_step",
        seq_id=7,
    )

    assert extra["scheduler_enabled"] is True
    assert extra["scheduler_domain"] == "training:openpi/pi0-fast-libero-low-mem-finetune"
    assert extra["scheduler_session_key"] == "model-1"
    assert extra["scheduler_fairness"] == "rr"
    assert extra["scheduler_max_consecutive"] == 1
    assert extra["seq_id"] == 7


def test_non_openpi_train_step_scheduler_extra_follows_global_toggle(monkeypatch):
    monkeypatch.setenv("MINT_SCHEDULER_ENABLE", "0")
    session = SimpleNamespace(
        backend="peft",
        base_model="meta-llama/Llama-3.1-8B-Instruct",
    )

    extra = _build_training_scheduler_extra(
        session=session,
        model_id="model-2",
        training_op="train_step",
    )

    assert extra["scheduler_enabled"] is False
    assert "scheduler_fairness" not in extra
    assert "scheduler_max_consecutive" not in extra
