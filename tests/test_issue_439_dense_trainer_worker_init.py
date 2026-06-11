from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

import mint_server.backend.runtime_observability as runtime_obs_module


def test_issue_439_dense_trainer_does_not_pass_removed_session_state_root(monkeypatch) -> None:
    from mint_server.backend import dense_trainer as dt
    from mint_server import config as cfg

    monkeypatch.setattr(cfg, "PFS_RUNTIME_ENV_ROOT", "/tmp/runtime-root")
    monkeypatch.setattr(cfg, "MINT_CODE_ROOT", "/tmp/mint-root")
    monkeypatch.setattr(cfg, "PFS_HF_MODULES_PATH", "/tmp/hf-modules")
    monkeypatch.setenv("RAY_ADDRESS", "192.168.38.184:6379")

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
        def get(self, _actor_name: str):
            return None

        def register(self, **kwargs):
            return SimpleNamespace(current_session=kwargs.get("session_id"))

        def mark_ready(self, _actor_name: str) -> None:
            return None

    monkeypatch.setattr(dt, "get_model_actor_supervisor", lambda: _FakePool())
    monkeypatch.setattr(dt, "_get_or_create_pg", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(dt.ray, "get_actor", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("missing")))
    monkeypatch.setattr(dt.ray, "get", lambda value, timeout=None: value)
    monkeypatch.setattr(dt, "PlacementGroupSchedulingStrategy", lambda **kwargs: kwargs)

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


@pytest.mark.parametrize(
    ("base_model", "actor_name"),
    [
        ("Qwen/Qwen3-0.6B", "mint_dense_qwen__qwen3_0_6b"),
        ("Qwen/Qwen3-4B-Instruct-2507", "mint_dense_qwen__qwen3_4b_instruct_2507"),
    ],
)
def test_issue_561_poisoned_dense_trainer_is_not_reused(monkeypatch, base_model: str, actor_name: str) -> None:
    from mint_server.backend import dense_trainer as dt
    from mint_server import config as cfg

    obs = runtime_obs_module.RuntimeObservability()
    monkeypatch.setattr(runtime_obs_module, "runtime_observability", obs)

    monkeypatch.setattr(cfg, "PFS_RUNTIME_ENV_ROOT", "/tmp/runtime-root")
    monkeypatch.setattr(cfg, "MINT_CODE_ROOT", "/tmp/mint-root")
    monkeypatch.setattr(cfg, "PFS_HF_MODULES_PATH", "/tmp/hf-modules")
    monkeypatch.setenv("RAY_ADDRESS", "192.168.38.184:6379")

    remote_kwargs: dict[str, object] = {}
    retire_calls: list[dict[str, object]] = []

    class _ExistingActor:
        pass

    class _ReadyRemote:
        def remote(self):
            return "ready-ref"

    class _NewActor:
        def __init__(self) -> None:
            self.__ray_ready__ = _ReadyRemote()

    class _FakeRemoteBuilder:
        def remote(self, **kwargs):
            remote_kwargs.update(kwargs)
            return _NewActor()

    class _FakeTrainingWorker:
        @staticmethod
        def options(**_kwargs):
            return _FakeRemoteBuilder()

    poisoned_entry = SimpleNamespace(
        metadata={"poisoned": True, "poison_reason": "forward_backward:CUDA error: device-side assert triggered"},
        current_session="stale-session",
    )

    class _FakePool:
        def get(self, queried_actor_name: str):
            assert queried_actor_name == actor_name
            return poisoned_entry

        def register(self, **kwargs):
            return SimpleNamespace(current_session=kwargs.get("session_id"))

        def mark_ready(self, _actor_name: str) -> None:
            return None

    def _fake_retire_dense_trainer(**kwargs) -> str:
        retire_calls.append(dict(kwargs))
        return "ok"

    monkeypatch.setattr(dt, "get_model_actor_supervisor", lambda: _FakePool())
    monkeypatch.setattr(dt, "retire_dense_trainer", _fake_retire_dense_trainer)
    monkeypatch.setattr(dt, "_get_or_create_pg", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(dt.ray, "get_actor", lambda *args, **kwargs: _ExistingActor())
    monkeypatch.setattr(dt.ray, "get", lambda value, timeout=None: value)
    monkeypatch.setattr(dt, "PlacementGroupSchedulingStrategy", lambda **kwargs: kwargs)

    dt.get_or_create_dense_trainer(
        training_worker_cls=_FakeTrainingWorker,
        base_model=base_model,
        lora_rank=8,
        learning_rate=1e-4,
        session_id="model-561",
    )

    assert len(retire_calls) == 1
    assert retire_calls[0]["actor_name"] == actor_name
    assert "reuse_blocked" in str(retire_calls[0]["reason"])
    assert obs.snapshot()["dense_actor_bind_decision"] == [
        {
            "base_model": base_model,
            "decision": "recreate_poisoned",
            "count": 1,
        }
    ]
    assert remote_kwargs == {
        "base_model": base_model,
        "lora_rank": 64,
        "learning_rate": 1e-4,
    }


def test_issue_561_poisoned_dense_trainer_recreate_aborts_when_retire_fails(monkeypatch) -> None:
    from mint_server.backend import dense_trainer as dt

    actor_name = "mint_dense_qwen__qwen3_0_6b"
    poisoned_entry = SimpleNamespace(
        metadata={"poisoned": True, "poison_reason": "forward_backward:CUDA error"},
        current_session="stale-session",
    )

    class _FakePool:
        def get(self, queried_actor_name: str):
            assert queried_actor_name == actor_name
            return poisoned_entry

    monkeypatch.setattr(dt, "get_model_actor_supervisor", lambda: _FakePool())
    monkeypatch.setattr(dt.ray, "get_actor", lambda *args, **kwargs: object())
    monkeypatch.setattr(dt, "retire_dense_trainer", lambda **kwargs: "kill_failed")

    with pytest.raises(RuntimeError, match="outcome=kill_failed"):
        dt.get_or_create_dense_trainer(
            training_worker_cls=object,
            base_model="Qwen/Qwen3-0.6B",
            lora_rank=8,
            learning_rate=1e-4,
            session_id="model-561",
        )


def test_issue_561_dead_dense_actor_absent_name_recreates(monkeypatch) -> None:
    from mint_server.backend import dense_trainer as dt
    from mint_server import config as cfg

    monkeypatch.setattr(cfg, "PFS_RUNTIME_ENV_ROOT", "/tmp/runtime-root")
    monkeypatch.setattr(cfg, "MINT_CODE_ROOT", "/tmp/mint-root")
    monkeypatch.setattr(cfg, "PFS_HF_MODULES_PATH", "/tmp/hf-modules")
    monkeypatch.setenv("RAY_ADDRESS", "192.168.38.184:6379")

    class _DeadActorError(RuntimeError):
        pass

    class _Heartbeat:
        def remote(self):
            raise _DeadActorError("dead")

    class _ExistingActor:
        heartbeat = _Heartbeat()

    class _ReadyRemote:
        def remote(self):
            return "ready-ref"

    class _NewActor:
        def __init__(self) -> None:
            self.__ray_ready__ = _ReadyRemote()

    class _FakeRemoteBuilder:
        def remote(self, **_kwargs):
            return _NewActor()

    class _FakeTrainingWorker:
        @staticmethod
        def options(**_kwargs):
            return _FakeRemoteBuilder()

    class _FakePool:
        def get(self, _actor_name: str):
            return None
        def register(self, **kwargs):
            return SimpleNamespace(current_session=kwargs.get("session_id"))

        def mark_ready(self, _actor_name: str) -> None:
            return None

    actors = [_ExistingActor(), ValueError("missing")]

    def _fake_get_actor(*_args, **_kwargs):
        out = actors.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr(dt, "RayActorError", _DeadActorError)
    monkeypatch.setattr(dt, "get_model_actor_supervisor", lambda: _FakePool())
    monkeypatch.setattr(dt.ray, "get_actor", _fake_get_actor)
    monkeypatch.setattr(dt.ray, "get", lambda value, timeout=None: value)
    monkeypatch.setattr(dt, "_get_or_create_pg", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(dt, "_remove_pg", lambda actor_name: None)

    handle = dt.get_or_create_dense_trainer(
        training_worker_cls=_FakeTrainingWorker,
        base_model="Qwen/Qwen3-0.6B",
        lora_rank=8,
        learning_rate=1e-4,
        session_id="model-561",
    )

    assert handle.actor_name == "mint_dense_qwen__qwen3_0_6b"


def test_issue_561_dense_trainer_recreates_on_max_rank_mismatch(monkeypatch) -> None:
    from mint_server.backend import dense_trainer as dt
    from mint_server import config as cfg

    monkeypatch.setattr(cfg, "PFS_RUNTIME_ENV_ROOT", "/tmp/runtime-root")
    monkeypatch.setattr(cfg, "MINT_CODE_ROOT", "/tmp/mint-root")
    monkeypatch.setattr(cfg, "PFS_HF_MODULES_PATH", "/tmp/hf-modules")
    monkeypatch.setenv("RAY_ADDRESS", "192.168.38.184:6379")

    actor_name = "mint_dense_qwen__qwen3_0_6b"
    retire_calls: list[dict[str, object]] = []

    class _Heartbeat:
        def remote(self):
            return {"max_lora_rank": 64}

    class _ExistingActor:
        heartbeat = _Heartbeat()

    class _ReadyRemote:
        def remote(self):
            return "ready-ref"

    class _NewActor:
        def __init__(self) -> None:
            self.__ray_ready__ = _ReadyRemote()

    class _FakeRemoteBuilder:
        def remote(self, **_kwargs):
            return _NewActor()

    class _FakeTrainingWorker:
        @staticmethod
        def options(**_kwargs):
            return _FakeRemoteBuilder()

    class _FakePool:
        def get(self, queried_actor_name: str):
            assert queried_actor_name == actor_name
            return SimpleNamespace(metadata={"max_lora_rank": 64})

        def register(self, **kwargs):
            return SimpleNamespace(current_session=kwargs.get("session_id"))

        def mark_ready(self, _actor_name: str) -> None:
            return None

    def _fake_retire_dense_trainer(**kwargs) -> str:
        retire_calls.append(dict(kwargs))
        return "ok"

    monkeypatch.setattr(dt, "get_model_actor_supervisor", lambda: _FakePool())
    monkeypatch.setattr(dt, "retire_dense_trainer", _fake_retire_dense_trainer)
    monkeypatch.setattr(dt.ray, "get_actor", lambda *args, **kwargs: _ExistingActor())
    monkeypatch.setattr(dt.ray, "get", lambda value, timeout=None: value)
    monkeypatch.setattr(dt, "_get_or_create_pg", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(dt, "PlacementGroupSchedulingStrategy", lambda **kwargs: kwargs)

    handle = dt.get_or_create_dense_trainer(
        training_worker_cls=_FakeTrainingWorker,
        base_model="Qwen/Qwen3-0.6B",
        lora_rank=8,
        max_lora_rank=128,
        learning_rate=1e-4,
        session_id="model-561",
    )

    assert handle.actor_name == actor_name
    assert handle.max_lora_rank == 128
    assert retire_calls[0]["actor_name"] == actor_name
    assert "max_lora_rank_mismatch" in str(retire_calls[0]["reason"])


def test_issue_561_dense_trainer_fails_closed_when_max_rank_unknown(monkeypatch) -> None:
    from mint_server.backend import dense_trainer as dt

    class _Heartbeat:
        def remote(self):
            return {"time_until_timeout": 60}

    class _ExistingActor:
        heartbeat = _Heartbeat()

    class _FakePool:
        def get(self, queried_actor_name: str):
            assert queried_actor_name == "mint_dense_qwen__qwen3_0_6b"
            return SimpleNamespace(metadata={})

    monkeypatch.setattr(dt, "get_model_actor_supervisor", lambda: _FakePool())
    monkeypatch.setattr(dt.ray, "get_actor", lambda *args, **kwargs: _ExistingActor())
    monkeypatch.setattr(dt.ray, "get", lambda value, timeout=None: value)

    with pytest.raises(RuntimeError, match="cannot verify max_lora_rank"):
        dt.get_or_create_dense_trainer(
            training_worker_cls=object,
            base_model="Qwen/Qwen3-0.6B",
            lora_rank=8,
            learning_rate=1e-4,
            session_id="model-561",
        )


def test_issue_561_inflight_guard_uses_actor_identity(monkeypatch) -> None:
    from mint_server.backend import dense_trainer as dt

    actor_name = dt._make_actor_name(model_key="Qwen/Qwen3-0.6B", max_rank=dt.DEFAULT_MAX_LORA_RANK)
    monkeypatch.setenv("MINT_DENSE_INFLIGHT_WAIT_S", "0.001")
    monkeypatch.setattr(dt, "_inflight", {actor_name: threading.Event()})
    monkeypatch.setattr(dt, "_inflight_errors", {})
    monkeypatch.setattr(
        dt,
        "get_model_actor_supervisor",
        lambda: (_ for _ in ()).throw(AssertionError("creation path should be guarded by actor identity")),
    )

    with pytest.raises(TimeoutError, match=actor_name):
        dt.get_or_create_dense_trainer(
            training_worker_cls=object,
            base_model="/mnt/hf-snapshots/qwen3-0.6b-a",
            model_key="Qwen/Qwen3-0.6B",
            lora_rank=8,
            learning_rate=1e-4,
            session_id="model-561-a",
        )


def test_issue_561_poison_metadata_preserves_first_fault() -> None:
    from mint_server.backend import dense_trainer as dt

    metadata = dt._poison_metadata(
        {
            dt.DENSE_POISONED_KEY: True,
            dt.DENSE_POISON_REASON_KEY: "forward_backward:first fault",
            dt.DENSE_POISONED_AT_KEY: 123.0,
            dt.DENSE_POISONED_SESSION_KEY: "model-first",
            dt.DENSE_LAST_FATAL_OP_KEY: "forward_backward",
            dt.DENSE_LAST_FATAL_REQUEST_ID_KEY: "req-first",
        },
        reason="reuse_blocked:second retire",
        session_id="model-second",
        fatal_op="save_weights_for_sampler",
        request_id="req-second",
    )

    assert metadata[dt.DENSE_POISON_REASON_KEY] == "forward_backward:first fault"
    assert metadata[dt.DENSE_POISONED_AT_KEY] == 123.0
    assert metadata[dt.DENSE_POISONED_SESSION_KEY] == "model-first"
    assert metadata[dt.DENSE_LAST_FATAL_OP_KEY] == "forward_backward"
    assert metadata[dt.DENSE_LAST_FATAL_REQUEST_ID_KEY] == "req-first"


def test_issue_561_retire_dense_trainer_persists_fatal_metadata(monkeypatch) -> None:
    from mint_server.backend import dense_trainer as dt

    metadata_updates: list[dict[str, object]] = []
    clears: list[tuple[str, object]] = []
    set_sessions: list[tuple[str, object]] = []
    unregisters: list[str] = []
    killed: list[dict[str, object]] = []
    obs = runtime_obs_module.RuntimeObservability()

    entry = SimpleNamespace(metadata={"poisoned": False})

    class _FakePool:
        def get(self, actor_name: str):
            assert actor_name == "mint_dense_qwen__qwen3_0_6b"
            return entry

        def update_metadata(self, actor_name: str, *, metadata, sample_source=None):
            metadata_updates.append({"actor_name": actor_name, "metadata": dict(metadata), "sample_source": sample_source})

        def clear_session(self, session_id: str, actor_type=None):
            clears.append((session_id, actor_type))

        def set_session(self, actor_name: str, session_id):
            set_sessions.append((actor_name, session_id))

        def unregister(self, actor_name: str):
            unregisters.append(actor_name)

    monkeypatch.setattr(dt, "get_model_actor_supervisor", lambda: _FakePool())
    monkeypatch.setattr(runtime_obs_module, "runtime_observability", obs)
    monkeypatch.setattr(dt.ray_kill, "kill", lambda *args, **kwargs: killed.append(dict(kwargs)))
    monkeypatch.setattr(dt, "_remove_pg", lambda actor_name: None)

    dt.retire_dense_trainer(
        actor_name="mint_dense_qwen__qwen3_0_6b",
        reason="forward_backward:CUDA error: device-side assert triggered",
        base_model="Qwen/Qwen3-0.6B",
        session_id="model-561",
        fatal_op="forward_backward",
        request_id="req-561",
        actor=object(),
    )

    assert len(metadata_updates) == 1
    update = metadata_updates[0]
    assert update["actor_name"] == "mint_dense_qwen__qwen3_0_6b"
    assert update["sample_source"] == "dense_retire"
    metadata = update["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["poisoned"] is True
    assert metadata["poison_reason"] == "forward_backward:CUDA error: device-side assert triggered"
    assert metadata["poisoned_session_id"] == "model-561"
    assert metadata["last_fatal_op"] == "forward_backward"
    assert metadata["last_fatal_request_id"] == "req-561"
    assert isinstance(metadata["poisoned_at"], float)
    assert clears == [("model-561", dt.ActorType.DENSE)]
    assert set_sessions == [("mint_dense_qwen__qwen3_0_6b", None)]
    assert unregisters == ["mint_dense_qwen__qwen3_0_6b"]
    assert killed[0]["verify_absent"] is True
    assert obs.snapshot()["dense_actor_retire"] == [
        {
            "base_model": "Qwen/Qwen3-0.6B",
            "outcome": "ok",
            "count": 1,
        }
    ]


def test_issue_561_retire_metadata_failure_does_not_block_recreate(monkeypatch) -> None:
    from mint_server.backend import dense_trainer as dt

    unregisters: list[str] = []
    obs = runtime_obs_module.RuntimeObservability()

    class _FakePool:
        def get(self, actor_name: str):
            assert actor_name == "mint_dense_qwen__qwen3_0_6b"
            return SimpleNamespace(metadata={"poisoned": True})

        def update_metadata(self, actor_name: str, *, metadata, sample_source=None):
            raise RuntimeError("metadata store unavailable")

        def clear_session(self, session_id: str, actor_type=None):
            return None

        def set_session(self, actor_name: str, session_id):
            return None

        def unregister(self, actor_name: str):
            unregisters.append(actor_name)

    monkeypatch.setattr(dt, "get_model_actor_supervisor", lambda: _FakePool())
    monkeypatch.setattr(runtime_obs_module, "runtime_observability", obs)
    monkeypatch.setattr(dt.ray_kill, "kill", lambda *args, **kwargs: None)
    monkeypatch.setattr(dt, "_remove_pg", lambda actor_name: None)

    outcome = dt.retire_dense_trainer(
        actor_name="mint_dense_qwen__qwen3_0_6b",
        reason="reuse_blocked:first fault",
        base_model="Qwen/Qwen3-0.6B",
        session_id="model-561",
        fatal_op="retire",
        actor=object(),
    )

    snap = obs.snapshot()
    assert outcome == "ok"
    assert unregisters == ["mint_dense_qwen__qwen3_0_6b"]
    assert snap["dense_actor_retire"] == [
        {
            "base_model": "Qwen/Qwen3-0.6B",
            "outcome": "ok",
            "count": 1,
        }
    ]
    assert [row["kind"] for row in snap["recent_training_incidents"]] == [
        "dense_actor_retire",
        "dense_actor_retire_auxiliary_failure",
    ]
    assert snap["recent_training_incidents"][0]["context"] == {
        "outcome": "ok",
        "auxiliary_failures": ["metadata_update_failed"],
    }
    assert snap["recent_training_incidents"][1]["failure_class"] == "metadata_update_failed"


def test_dense_trainer_registers_creating_before_ready_wait(monkeypatch) -> None:
    """The undesired-GPU-actor reaper kills any mint_dense_* actor not present in
    the supervisor inventory. The dense trainer must therefore register the actor
    (creating=True) BEFORE the up-to-600s __ray_ready__ wait, otherwise the actor
    is unprotected during init and gets reaped mid forward_backward.
    """
    from mint_server.backend import dense_trainer as dt
    from mint_server import config as cfg

    monkeypatch.setattr(cfg, "PFS_RUNTIME_ENV_ROOT", "/tmp/runtime-root")
    monkeypatch.setattr(cfg, "MINT_CODE_ROOT", "/tmp/mint-root")
    monkeypatch.setattr(cfg, "PFS_HF_MODULES_PATH", "/tmp/hf-modules")
    monkeypatch.setenv("RAY_ADDRESS", "192.168.38.184:6379")

    events: list[tuple[str, object]] = []

    class _ReadyRemote:
        def remote(self):
            events.append(("ready_wait", None))
            return "ready-ref"

    class _FakeActor:
        def __init__(self) -> None:
            self.__ray_ready__ = _ReadyRemote()

    class _FakeRemoteBuilder:
        def remote(self, **_kwargs):
            return _FakeActor()

    class _FakeTrainingWorker:
        @staticmethod
        def options(**_kwargs):
            return _FakeRemoteBuilder()

    class _FakePool:
        def get(self, _actor_name: str):
            return None

        def register(self, **kwargs):
            return SimpleNamespace(current_session=kwargs.get("session_id"))

        def mark_ready(self, _actor_name: str) -> None:
            return None

    def _fake_publish(launch, *, ready=True, **_kwargs):
        events.append(("publish", bool(ready)))
        return SimpleNamespace(current_session=launch.session_id)

    monkeypatch.setattr(dt, "get_model_actor_supervisor", lambda: _FakePool())
    monkeypatch.setattr(dt, "publish_backend_model_actor", _fake_publish)
    monkeypatch.setattr(dt, "_get_or_create_pg", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        dt.ray, "get_actor", lambda *a, **k: (_ for _ in ()).throw(ValueError("missing"))
    )
    monkeypatch.setattr(dt.ray, "get", lambda value, timeout=None: value)
    monkeypatch.setattr(dt, "PlacementGroupSchedulingStrategy", lambda **kwargs: kwargs)

    dt.get_or_create_dense_trainer(
        training_worker_cls=_FakeTrainingWorker,
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        lora_rank=8,
        learning_rate=1e-4,
        session_id="model-protect",
    )

    # First inventory publish (creating=True) must happen before the ready wait.
    assert events[0] == ("publish", False), events
    assert ("ready_wait", None) in events
    assert events.index(("publish", False)) < events.index(("ready_wait", None))
    # And a ready=True publish must follow once init completes.
    assert ("publish", True) in events
