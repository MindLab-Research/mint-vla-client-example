import sys
from types import SimpleNamespace

import anyio
import pytest

from tinker_server.backend import capacity_manager as cm


class _AwaitableRef:
    def __init__(self, *, result=None, error: Exception | None = None):
        self._result = result
        self._error = error

    def __await__(self):
        async def _run():
            if self._error is not None:
                raise self._error
            return self._result

        return _run().__await__()


class _RemoteMethod:
    def __init__(self, *, result=None, error: Exception | None = None):
        self._result = result
        self._error = error

    def remote(self, *args, **kwargs):
        return _AwaitableRef(result=self._result, error=self._error)


class _StubActor:
    def __init__(self, *, result=None, error: Exception | None = None):
        self.try_reserve = _RemoteMethod(result=result, error=error)


def test_issue_432_capacity_manager_async_try_reserve_actor_died_clears_cache(monkeypatch):
    class _RayExceptions:
        class ActorDiedError(Exception):
            pass

    ray_stub = SimpleNamespace(exceptions=_RayExceptions, is_initialized=lambda: True)
    monkeypatch.setitem(sys.modules, "ray", ray_stub)

    mgr = cm.CapacityManager()
    mgr._ray_actor = _StubActor(error=_RayExceptions.ActorDiedError("dead actor"))

    async def _run_async_try_reserve():
        return await mgr.async_try_reserve("rid", queue_bytes=7, object_store_bytes=11)

    with pytest.raises(cm.CapacityManagerUnavailableError, match="died"):
        anyio.run(_run_async_try_reserve)

    assert mgr._ray_actor is None


def test_issue_432_capacity_manager_async_try_reserve_recovers_on_next_request(monkeypatch):
    class _RayExceptions:
        class ActorDiedError(Exception):
            pass

    ray_stub = SimpleNamespace(exceptions=_RayExceptions, is_initialized=lambda: True)
    monkeypatch.setitem(sys.modules, "ray", ray_stub)

    mgr = cm.CapacityManager()
    mgr._ray_actor = _StubActor(error=_RayExceptions.ActorDiedError("dead actor"))
    recovered = _StubActor(result={"ok": True})
    recreated = []

    def _get_or_create_ray_actor():
        recreated.append(True)
        return recovered

    monkeypatch.setattr(cm, "_get_or_create_ray_actor", _get_or_create_ray_actor)

    async def _run_async_try_reserve():
        return await mgr.async_try_reserve("rid", queue_bytes=7, object_store_bytes=11)

    with pytest.raises(cm.CapacityManagerUnavailableError):
        anyio.run(_run_async_try_reserve)

    out = anyio.run(_run_async_try_reserve)

    assert out == {"ok": True}
    assert recreated == [True]


def test_issue_432_capacity_manager_probe_failure_falls_back_to_named_actor(monkeypatch):
    existing_actor = object()
    recorder = {"ray_get_calls": []}

    import tinker_server.config as config_mod

    monkeypatch.setenv("RAY_ADDRESS", "ray://test")
    monkeypatch.setattr(config_mod, "PFS_RUNTIME_ENV_ROOT", "/tmp/runtime", raising=False)
    monkeypatch.setattr(config_mod, "PFS_TINKER_PATH", "/tmp/tinker", raising=False)
    monkeypatch.setattr(config_mod, "PFS_HF_MODULES_PATH", "/tmp/hfmods", raising=False)
    monkeypatch.setattr(config_mod, "PFS_PYTHONPATH", "/tmp/runtime:/tmp/tinker:/tmp/hfmods", raising=False)
    monkeypatch.setattr(
        config_mod,
        "actor_runtime_env",
        lambda pythonpath, extra: {"py_modules": [], "env_vars": {"PYTHONPATH": pythonpath, **extra}},
        raising=False,
    )

    def _get_actor(name, namespace):
        recorder.setdefault("get_actor_calls", []).append((name, namespace))
        if len(recorder["get_actor_calls"]) == 1:
            raise ValueError("missing")
        return existing_actor

    class _SnapshotMethod:
        def remote(self):
            return "snapshot_ref"

    class _CreatedActor:
        snapshot = _SnapshotMethod()

    class _RemoteActorFactory:
        def options(self, **options):
            recorder["options"] = options
            return self

        def remote(self, **kwargs):
            recorder["remote_kwargs"] = kwargs
            return _CreatedActor()

    def _remote(**remote_kwargs):
        recorder["remote_decorator_kwargs"] = remote_kwargs

        def _wrap(cls):
            recorder["actor_cls_name"] = cls.__name__
            return _RemoteActorFactory()

        return _wrap

    def _ray_get(ref, timeout=None):
        recorder["ray_get_calls"].append((ref, timeout))
        if ref == "snapshot_ref":
            raise RuntimeError("probe failed")
        return ref

    ray_stub = SimpleNamespace(
        get_actor=_get_actor,
        get=_ray_get,
        remote=_remote,
    )
    monkeypatch.setitem(sys.modules, "ray", ray_stub)

    actor = cm._get_or_create_ray_actor()

    assert actor is existing_actor
    assert recorder["remote_decorator_kwargs"] == {"num_cpus": 0}
    assert recorder["options"]["get_if_exists"] is True
    assert recorder["options"]["lifetime"] == "detached"
    assert recorder["get_actor_calls"] == [
        (cm._ray_capacity_manager_actor_name(), cm._ray_namespace()),
        (cm._ray_capacity_manager_actor_name(), cm._ray_namespace()),
    ]
    assert recorder["ray_get_calls"] == [("snapshot_ref", 1.0)]
