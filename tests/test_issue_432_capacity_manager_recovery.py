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


@pytest.mark.parametrize("exc_name", ["ActorDiedError", "RayActorError"])
def test_issue_432_capacity_manager_async_try_reserve_retries_after_actor_error(monkeypatch, exc_name):
    class _RayExceptions:
        class ActorDiedError(Exception):
            pass

        class RayActorError(Exception):
            pass

    dead_exc = getattr(_RayExceptions, exc_name)("dead actor")

    ray_stub = SimpleNamespace(exceptions=_RayExceptions)
    monkeypatch.setitem(sys.modules, "ray", ray_stub)

    mgr = cm.CapacityManager()
    first = _StubActor(error=dead_exc)
    second = _StubActor(result={"ok": True})
    resets = []

    async def _get_ray_actor_async():
        return second

    monkeypatch.setattr(mgr, "_get_cached_ray_actor_for_async_request_path", lambda: first)
    monkeypatch.setattr(mgr, "_get_ray_actor_async", _get_ray_actor_async)
    monkeypatch.setattr(mgr, "_reset_ray_actor", lambda actor=None: resets.append(actor))

    async def _run_async_try_reserve():
        return await mgr.async_try_reserve("rid", queue_bytes=7, object_store_bytes=11)

    out = anyio.run(_run_async_try_reserve)

    assert out == {"ok": True}
    assert resets == [first]


def test_issue_432_capacity_manager_create_race_falls_back_to_named_actor(monkeypatch):
    existing_actor = object()
    recorder = {}

    import tinker_server.config as config_mod

    monkeypatch.setenv("RAY_ADDRESS", "ray://test")
    monkeypatch.setattr(config_mod, "PFS_RUNTIME_ENV_ROOT", "/tmp/runtime", raising=False)
    monkeypatch.setattr(config_mod, "PFS_TINKER_PATH", "/tmp/tinker", raising=False)
    monkeypatch.setattr(config_mod, "PFS_HF_MODULES_PATH", "/tmp/hfmods", raising=False)
    monkeypatch.setattr(config_mod, "PFS_PYTHONPATH", "/tmp/runtime:/tmp/tinker:/tmp/hfmods", raising=False)

    def _get_actor(name, namespace):
        recorder.setdefault("get_actor_calls", []).append((name, namespace))
        if len(recorder["get_actor_calls"]) == 1:
            raise ValueError("missing")
        return existing_actor

    class _RemoteActorFactory:
        def options(self, **options):
            recorder["options"] = options
            return self

        def remote(self, **kwargs):
            recorder["remote_kwargs"] = kwargs
            raise RuntimeError("actor create raced")

    def _remote(**remote_kwargs):
        recorder["remote_decorator_kwargs"] = remote_kwargs

        def _wrap(cls):
            recorder["actor_cls_name"] = cls.__name__
            return _RemoteActorFactory()

        return _wrap

    ray_stub = SimpleNamespace(
        get_actor=_get_actor,
        cluster_resources=lambda: {"node:__internal_head__": 1.0},
        remote=_remote,
    )
    monkeypatch.setitem(sys.modules, "ray", ray_stub)

    actor = cm._get_or_create_ray_actor()

    assert actor is existing_actor
    assert recorder["remote_decorator_kwargs"] == {"num_cpus": 0}
    assert recorder["options"]["get_if_exists"] is True
    assert recorder["options"]["max_restarts"] == -1
    assert recorder["options"]["max_task_retries"] == -1
    assert recorder["options"]["resources"] == {"node:__internal_head__": 0.001}
    assert recorder["get_actor_calls"] == [
        (cm._ray_capacity_manager_actor_name(), cm._ray_namespace()),
        (cm._ray_capacity_manager_actor_name(), cm._ray_namespace()),
    ]


def test_issue_432_capacity_manager_prefers_configured_detached_actor_node(monkeypatch):
    existing_actor = object()
    recorder = {}

    import tinker_server.config as config_mod

    monkeypatch.setenv("RAY_ADDRESS", "ray://test")
    monkeypatch.setenv("MINT_DETACHED_ACTOR_NODE_IP", "192.168.38.175")
    monkeypatch.setattr(config_mod, "PFS_RUNTIME_ENV_ROOT", "/tmp/runtime", raising=False)
    monkeypatch.setattr(config_mod, "PFS_TINKER_PATH", "/tmp/tinker", raising=False)
    monkeypatch.setattr(config_mod, "PFS_HF_MODULES_PATH", "/tmp/hfmods", raising=False)
    monkeypatch.setattr(config_mod, "PFS_PYTHONPATH", "/tmp/runtime:/tmp/tinker:/tmp/hfmods", raising=False)

    def _get_actor(name, namespace):
        recorder.setdefault("get_actor_calls", []).append((name, namespace))
        if len(recorder["get_actor_calls"]) == 1:
            raise ValueError("missing")
        return existing_actor

    class _RemoteActorFactory:
        def options(self, **options):
            recorder["options"] = options
            return self

        def remote(self, **kwargs):
            recorder["remote_kwargs"] = kwargs
            raise RuntimeError("actor create raced")

    def _remote(**remote_kwargs):
        recorder["remote_decorator_kwargs"] = remote_kwargs

        def _wrap(cls):
            recorder["actor_cls_name"] = cls.__name__
            return _RemoteActorFactory()

        return _wrap

    ray_stub = SimpleNamespace(
        get_actor=_get_actor,
        cluster_resources=lambda: {"node:__internal_head__": 1.0, "node:192.168.38.175": 1.0},
        remote=_remote,
    )
    monkeypatch.setitem(sys.modules, "ray", ray_stub)

    actor = cm._get_or_create_ray_actor()

    assert actor is existing_actor
    assert recorder["options"]["resources"] == {"node:192.168.38.175": 0.001}
