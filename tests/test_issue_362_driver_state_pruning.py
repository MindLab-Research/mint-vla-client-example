import copy
import sys
import types

import pytest

import tinker_server.backend.session_heartbeat_store as session_heartbeat_store_module
import tinker_server.ray_utils as ray_utils
from tinker_server.backend.session_heartbeat_store import SessionHeartbeatStore
from tinker_server.backend.session_manager import SessionManager
from tinker_server.backend.training_session_manager import TrainingSessionManager


class _RemoteMethod:
    def __init__(self, fn):
        self.remote = fn


class _FakeHeartbeatActor:
    def __init__(self) -> None:
        self._last_seen: dict[str, float] = {}
        self._max_age_s = 7 * 86400.0
        self._prune_every = 256
        self._updates_since_prune = 0
        self.update = _RemoteMethod(self._update)
        self.last_seen = _RemoteMethod(self._last_seen_value)
        self.delete = _RemoteMethod(self._delete)
        self.size = _RemoteMethod(self._size)
        self.is_stale = _RemoteMethod(self._is_stale)
        self.prune = _RemoteMethod(self._prune)

    def _update(self, *, session_id: str, now: float | None = None) -> None:
        ts = 0.0 if now is None else float(now)
        self._last_seen[str(session_id)] = ts
        self._updates_since_prune += 1
        if self._updates_since_prune >= self._prune_every:
            self._prune_locked(now=ts, max_age_s=self._max_age_s)
            self._updates_since_prune = 0

    def _last_seen_value(self, *, session_id: str) -> float | None:
        return self._last_seen.get(str(session_id))

    def _delete(self, *, session_id: str) -> bool:
        return self._last_seen.pop(str(session_id), None) is not None

    def _size(self) -> int:
        return len(self._last_seen)

    def _is_stale(self, *, session_id: str, ttl_s: float) -> bool:
        last = self._last_seen.get(str(session_id))
        return last is not None and (200.0 - last) > float(ttl_s)

    def _prune(self, *, max_age_s: float) -> int:
        return self._prune_locked(now=120.0, max_age_s=float(max_age_s))

    def _prune_locked(self, *, now: float, max_age_s: float) -> int:
        to_delete = [sid for sid, ts in self._last_seen.items() if (now - ts) > max_age_s]
        for sid in to_delete:
            del self._last_seen[sid]
        return len(to_delete)


def _ensure_fake_ray_module(monkeypatch: pytest.MonkeyPatch):
    try:
        import ray

        return ray
    except ModuleNotFoundError:
        ray_module = types.ModuleType("ray")
        ray_module.get = lambda ref, *args, **kwargs: ref
        monkeypatch.setitem(sys.modules, "ray", ray_module)
        return ray_module


def _install_fake_heartbeat_store(monkeypatch: pytest.MonkeyPatch) -> SessionHeartbeatStore:
    store = SessionHeartbeatStore()
    actor = _FakeHeartbeatActor()
    monkeypatch.setattr(store, "_get_actor", lambda: actor)
    ray = _ensure_fake_ray_module(monkeypatch)

    monkeypatch.setattr(ray, "get", lambda ref, *args, **kwargs: ref, raising=False)
    return store


def test_issue_362_session_heartbeat_store_prunes_old_entries_on_update(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _install_fake_heartbeat_store(monkeypatch)
    actor = store._get_actor()
    actor._max_age_s = 50.0
    actor._prune_every = 2

    store.update("old", now=100.0)
    store.update("fresh", now=200.0)

    assert store.last_seen("old") is None
    assert store.last_seen("fresh") == 200.0


def test_issue_362_session_heartbeat_store_manual_prune_removes_stale_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _install_fake_heartbeat_store(monkeypatch)
    actor = store._get_actor()
    store.update("keep", now=100.0)
    store.update("drop", now=10.0)

    removed = actor._prune_locked(now=120.0, max_age_s=50.0)

    assert removed == 1
    assert store.last_seen("keep") == 100.0
    assert store.last_seen("drop") is None


def test_issue_362_session_heartbeat_store_recreates_actor_after_wrong_cluster_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[_FakeHeartbeatActor] = []
    reconnects: list[str | None] = []
    get_actor_calls = 0

    def _get_actor(_name: str, namespace: str | None = None):
        nonlocal get_actor_calls
        get_actor_calls += 1
        if get_actor_calls == 1:
            raise RuntimeError("WrongClusterID: stale driver connection")
        if created:
            return created[-1]
        raise ValueError(f"actor missing in namespace {namespace}")

    class _RemoteFactory:
        def __call__(self, *decorator_args, **_decorator_kwargs):
            def _decorate(_cls):
                class _RemoteClass:
                    @staticmethod
                    def options(**_options):
                        class _Options:
                            @staticmethod
                            def remote():
                                actor = _FakeHeartbeatActor()
                                created.append(actor)
                                return actor

                        return _Options()

                return _RemoteClass

            if decorator_args and isinstance(decorator_args[0], type):
                return _decorate(decorator_args[0])
            return _decorate

    fake_ray = _ensure_fake_ray_module(monkeypatch)
    monkeypatch.setattr(fake_ray, "get", lambda ref, *args, **kwargs: ref, raising=False)
    monkeypatch.setattr(fake_ray, "get_actor", _get_actor, raising=False)
    monkeypatch.setattr(fake_ray, "remote", _RemoteFactory(), raising=False)
    monkeypatch.setattr(session_heartbeat_store_module, "apply_detached_actor_resources", lambda options, ray: None)
    monkeypatch.setattr(session_heartbeat_store_module, "actor_runtime_env", lambda *, pythonpath, extra=None: {})
    monkeypatch.setattr(session_heartbeat_store_module, "_ACTOR_HANDLE", None)
    monkeypatch.setattr(
        ray_utils,
        "force_reconnect_ray",
        lambda *, namespace=None: reconnects.append(namespace),
    )
    monkeypatch.setattr(
        ray_utils,
        "is_wrong_cluster_error",
        lambda exc: "WrongClusterID" in str(exc),
    )

    actor = session_heartbeat_store_module._get_or_create_actor()

    assert isinstance(actor, _FakeHeartbeatActor)
    assert session_heartbeat_store_module._ACTOR_HANDLE is actor
    assert reconnects == [session_heartbeat_store_module._ray_namespace()]
    assert get_actor_calls == 2
    assert len(created) == 1


def test_issue_362_observability_snapshot_excludes_base_model_sessions_from_lora_loaded() -> None:
    manager = SessionManager()
    manager.register_base_model_session("base", "Qwen/Qwen3-4B-Instruct-2507")
    manager.register_multi_lora_session(
        session_id="lora",
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        lora_rank=32,
        adapter_path=None,
        lora_loaded=True,
    )

    snapshot = manager.observability_snapshot()

    assert snapshot["sampling_sessions_total"] == 2
    assert snapshot["sampling_sessions_base_model"] == 1
    assert snapshot["sampling_sessions_multi_lora"] == 2
    assert snapshot["sampling_sessions_lora_loaded"] == 1
    assert sorted(snapshot["sampling_sessions_by_model"], key=lambda row: row["base_model"]) == [
        {
            "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "total": 1,
            "inflight": 0,
            "lora_loaded": 1,
        },
        {
            "base_model": "Qwen/Qwen3-4B-Instruct-2507",
            "total": 1,
            "inflight": 0,
            "lora_loaded": 0,
        },
    ]


def test_issue_362_training_observability_snapshot_groups_by_model() -> None:
    manager = TrainingSessionManager()
    a = manager.create_session(
        model_id="m1",
        session_id="s1",
        model_seq_id=1,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    )
    b = manager.create_session(
        model_id="m2",
        session_id="s2",
        model_seq_id=1,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    )
    before = copy.copy(a)
    a.backend = "megatron"
    manager.refresh_observability_session("m1", before=before)
    before = copy.copy(b)
    b.backend = "megatron"
    manager.refresh_observability_session("m2", before=before)
    before = copy.copy(a)
    a.is_active = True
    manager.refresh_observability_session("m1", before=before)
    before = copy.copy(b)
    b.inflight_ops = 1
    manager.refresh_observability_session("m2", before=before)

    snapshot = manager.observability_snapshot()

    assert snapshot["training_sessions_total"] == 2
    assert snapshot["training_sessions_active"] == 1
    assert snapshot["training_sessions_inflight"] == 1
    assert snapshot["training_sessions_by_model"] == [
        {
            "base_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "backend": "megatron",
            "total": 2,
            "active": 1,
            "inflight": 1,
        }
    ]


def test_issue_362_training_observability_delete_uses_published_state() -> None:
    manager = TrainingSessionManager()
    session = manager.create_session(
        model_id="m1",
        session_id="s1",
        model_seq_id=1,
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    )
    before = copy.copy(session)
    session.backend = "megatron"
    session.is_active = True
    manager.refresh_observability_session("m1", before=before)

    session.is_active = False
    assert manager.delete_session("m1") is True

    snapshot = manager.observability_snapshot()
    assert snapshot["training_sessions_total"] == 0
    assert snapshot["training_sessions_active"] == 0
    assert snapshot["training_sessions_inflight"] == 0
    assert snapshot["training_sessions_by_model"] == []
