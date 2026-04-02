from __future__ import annotations

from types import SimpleNamespace


def test_issue_439_dense_trainer_does_not_pass_removed_session_state_root(monkeypatch) -> None:
    from tinker_server.backend import dense_trainer as dt
    from tinker_server import config as cfg

    monkeypatch.setattr(cfg, "PFS_RUNTIME_ENV_ROOT", "/tmp/runtime-root")
    monkeypatch.setattr(cfg, "PFS_TINKER_PATH", "/tmp/tinker-root")
    monkeypatch.setattr(cfg, "PFS_HF_MODULES_PATH", "/tmp/hf-modules")
    monkeypatch.setattr(cfg, "require_ray_address", lambda: "192.168.38.184:6379")

    remote_kwargs: dict[str, object] = {}

    class _ReadyRemote:
        def remote(self):
            return "ready-ref"

    class _FakeActor:
        def __init__(self) -> None:
            self.__ray_ready__ = _ReadyRemote()

    class _FakeRemoteBuilder:
        def remote(self, **kwargs):
            remote_kwargs.update(kwargs)
            return _FakeActor()

    class _FakeTrainingWorker:
        @staticmethod
        def options(**_kwargs):
            return _FakeRemoteBuilder()

    class _FakePool:
        def ensure_gpus_available(self, _num_gpus: int) -> None:
            return None

        def register(self, **kwargs):
            return SimpleNamespace(current_session=kwargs.get("session_id"))

        def mark_ready(self, _actor_name: str) -> None:
            return None

    monkeypatch.setattr(dt, "get_resource_pool", lambda: _FakePool())
    monkeypatch.setattr(dt, "_get_or_create_pg", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(dt.ray, "get_actor", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("missing")))
    monkeypatch.setattr(dt.ray, "get", lambda value, timeout=None: value)
    monkeypatch.setattr(
        dt.ray.util,
        "scheduling_strategies",
        SimpleNamespace(PlacementGroupSchedulingStrategy=lambda **kwargs: kwargs),
    )

    dt.get_or_create_dense_trainer(
        training_worker_cls=_FakeTrainingWorker,
        base_model="Qwen/Qwen3-0.6B",
        lora_rank=8,
        learning_rate=1e-4,
        session_id="model-439",
    )

    assert remote_kwargs == {
        "base_model": "Qwen/Qwen3-0.6B",
        "lora_rank": 64,
        "learning_rate": 1e-4,
    }
