import importlib
import importlib.machinery
import sys
import types

import pytest


def _install_stub(name: str, module: types.ModuleType) -> None:
    if name not in sys.modules:
        sys.modules[name] = module


def _ensure_ray_stubbed() -> None:
    try:
        import ray  # type: ignore
        if all(
            hasattr(ray, attr)
            for attr in ("remote", "kill", "get", "get_actor", "is_initialized", "actor")
        ) and hasattr(ray, "util"):
            return
    except ModuleNotFoundError:
        ray = None

    if ray is None:
        ray = types.ModuleType("ray")
        ray.__spec__ = importlib.machinery.ModuleSpec("ray", loader=None)

    def remote(**_kwargs):
        def deco(obj):
            return obj

        return deco

    ray.remote = getattr(ray, "remote", remote)  # type: ignore[attr-defined]
    ray.kill = getattr(ray, "kill", lambda *_a, **_k: None)  # type: ignore[attr-defined]
    ray.get = getattr(ray, "get", lambda *_a, **_k: None)  # type: ignore[attr-defined]
    ray.get_actor = getattr(ray, "get_actor", lambda *_a, **_k: None)  # type: ignore[attr-defined]
    ray.is_initialized = getattr(ray, "is_initialized", lambda: True)  # type: ignore[attr-defined]
    ray.actor = getattr(ray, "actor", types.SimpleNamespace(ActorHandle=object))  # type: ignore[attr-defined]

    ray_util = getattr(ray, "util", None)
    if ray_util is None:
        ray_util = types.ModuleType("ray.util")
        ray_util.__spec__ = importlib.machinery.ModuleSpec("ray.util", loader=None)
        ray.util = ray_util  # type: ignore[attr-defined]

    _install_stub("ray", ray)
    _install_stub("ray.util", ray_util)


def _ensure_peft_stubbed() -> None:
    try:
        import peft  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    peft = types.ModuleType("peft")
    peft.__spec__ = importlib.machinery.ModuleSpec("peft", loader=None)

    class LoraConfig:
        def __init__(self, *args, **kwargs):
            _ = args, kwargs

    class TaskType:
        CAUSAL_LM = "CAUSAL_LM"

    def get_peft_model(model, _config):
        return model

    peft.LoraConfig = LoraConfig  # type: ignore[attr-defined]
    peft.TaskType = TaskType  # type: ignore[attr-defined]
    peft.get_peft_model = get_peft_model  # type: ignore[attr-defined]

    _install_stub("peft", peft)


def _import_megatron_modules():
    _ensure_ray_stubbed()
    _ensure_peft_stubbed()
    dist = importlib.import_module("tinker_server.backend.megatron_distributed")
    ray_kill = importlib.import_module("tinker_server.backend.ray_kill")
    resource_pool = importlib.import_module("tinker_server.backend.resource_pool")
    return dist, ray_kill, resource_pool


class _ShutdownHandle:
    def remote(self):
        return "shutdown-ref"


class _ActorHandle:
    shutdown = _ShutdownHandle()


class _Pool:
    def __init__(self):
        self.unregistered: list[str] = []

    def unregister(self, actor_name: str) -> bool:
        self.unregistered.append(actor_name)
        return True

    def iter_entries(self):
        return []


def test_ray_kill_verify_absent_polls_until_lookup_fails(monkeypatch):
    _, ray_kill, _ = _import_megatron_modules()

    lookups = {"count": 0}
    kill_calls: list[tuple[object, bool]] = []

    def fake_get_actor(actor_name: str, namespace: str):
        assert actor_name == "actor"
        assert namespace == "ns"
        lookups["count"] += 1
        if lookups["count"] < 3:
            return object()
        raise ValueError("missing")

    monkeypatch.setattr(ray_kill.ray, "kill", lambda actor, no_restart=True: kill_calls.append((actor, no_restart)))
    monkeypatch.setattr(ray_kill.ray, "get_actor", fake_get_actor)
    monkeypatch.setattr(ray_kill, "_remove_placement_group_for_actor_name", lambda _actor_name: None)
    monkeypatch.setattr(ray_kill.time, "sleep", lambda _s: None)

    actor = object()
    ray_kill.kill(
        actor,
        reason="test",
        actor_name="actor",
        namespace="ns",
        no_restart=True,
        verify_absent=True,
        verify_timeout_s=1.0,
        verify_poll_interval_s=0.01,
    )

    assert kill_calls == [(actor, True)]
    assert lookups["count"] == 3


def test_kill_megatron_actor_unregisters_only_after_verified_disappearance(monkeypatch):
    dist, _, resource_pool = _import_megatron_modules()
    pool = _Pool()
    kill_kwargs: dict[str, object] = {}

    monkeypatch.setattr(dist.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(dist.ray, "get_actor", lambda actor_name, namespace: _ActorHandle())
    monkeypatch.setattr(dist.ray, "get", lambda _ref, timeout=None: None)
    monkeypatch.setattr(resource_pool, "get_resource_pool", lambda: pool)
    monkeypatch.setattr(
        dist.ray_kill,
        "kill",
        lambda actor, **kwargs: kill_kwargs.update(kwargs),
    )
    monkeypatch.setattr(dist.ray.util, "get_placement_group", lambda _name: (_ for _ in ()).throw(ValueError("missing")), raising=False)
    monkeypatch.setattr(dist.ray.util, "remove_placement_group", lambda _pg: None, raising=False)

    assert dist.kill_megatron_actor("Qwen/Qwen3-30B-A3B-Instruct-2507") is True
    assert kill_kwargs["verify_absent"] is True
    assert pool.unregistered == [dist._make_megatron_actor_name("Qwen/Qwen3-30B-A3B-Instruct-2507")]


def test_kill_megatron_actor_does_not_unregister_when_actor_stays_resolvable(monkeypatch):
    dist, ray_kill, resource_pool = _import_megatron_modules()
    pool = _Pool()

    monkeypatch.setattr(dist.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(dist.ray, "get_actor", lambda actor_name, namespace: _ActorHandle())
    monkeypatch.setattr(dist.ray, "get", lambda _ref, timeout=None: None)
    monkeypatch.setattr(resource_pool, "get_resource_pool", lambda: pool)
    monkeypatch.setattr(
        dist.ray_kill,
        "kill",
        lambda actor, **kwargs: (_ for _ in ()).throw(
            ray_kill.ActorStillAliveError("actor still exists after kill")
        ),
    )

    with pytest.raises(ray_kill.ActorStillAliveError, match="still exists"):
        dist.kill_megatron_actor("Qwen/Qwen3-30B-A3B-Instruct-2507")

    assert pool.unregistered == []
