import sys
from types import SimpleNamespace

import pytest

from tinker_server.backend import capacity_manager as cm


class _RemoteMethod:
    def __init__(self, ref):
        self._ref = ref

    def remote(self, *args, **kwargs):
        return self._ref


class _StubActor:
    def __init__(self, ref):
        self.try_reserve = _RemoteMethod(ref)


@pytest.mark.parametrize("exc_name", ["ActorDiedError", "RayActorError"])
def test_issue_432_capacity_manager_try_reserve_retries_after_actor_error(monkeypatch, exc_name):
    class _RayExceptions:
        class ActorDiedError(Exception):
            pass

        class RayActorError(Exception):
            pass

    dead_exc = getattr(_RayExceptions, exc_name)("dead actor")

    def _ray_get(ref, timeout=None):
        if ref == "dead":
            raise dead_exc
        if ref == "ok":
            return {"ok": True}
        raise AssertionError(f"unexpected ref: {ref!r}")

    ray_stub = SimpleNamespace(exceptions=_RayExceptions, get=_ray_get)
    monkeypatch.setitem(sys.modules, "ray", ray_stub)

    mgr = cm.CapacityManager()
    first = _StubActor("dead")
    second = _StubActor("ok")
    actors = iter([first, second])
    resets = []
    monkeypatch.setattr(mgr, "_get_ray_actor", lambda: next(actors))
    monkeypatch.setattr(mgr, "_reset_ray_actor", lambda actor=None: resets.append(actor))

    out = mgr.try_reserve("rid", queue_bytes=7, object_store_bytes=11)

    assert out == {"ok": True}
    assert resets == [first]


def test_issue_432_capacity_manager_create_race_falls_back_to_named_actor(monkeypatch):
    existing_actor = object()
    recorder = {}

    import tinker_server.config as config_mod

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
