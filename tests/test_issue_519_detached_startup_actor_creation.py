from __future__ import annotations

import importlib
import sys
import types

import pytest


class _FakeActorHandle:
    def __init__(self, label: str) -> None:
        self.label = str(label)

    def __getattr__(self, name: str):
        return types.SimpleNamespace(remote=lambda *args, **kwargs: (name, args, kwargs))


class _FakeRemoteFactory:
    def __init__(self, cls, created_actors: list[_FakeActorHandle]) -> None:
        self._cls = cls
        self._created_actors = created_actors

    def options(self, **_kwargs):
        return self

    def remote(self, *_args, **_kwargs):
        actor = _FakeActorHandle(self._cls.__name__)
        self._created_actors.append(actor)
        return actor


def _install_fake_ray(monkeypatch):
    created_actors: list[_FakeActorHandle] = []

    def _remote(*decorator_args, **_decorator_kwargs):
        def _wrap(cls):
            return _FakeRemoteFactory(cls, created_actors)

        if decorator_args and callable(decorator_args[0]) and len(decorator_args) == 1 and not _decorator_kwargs:
            return _wrap(decorator_args[0])
        return _wrap

    fake_ray = types.SimpleNamespace(
        remote=_remote,
        get_actor=lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("actor not found")),
        get=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("blocking ray.get probe should not run")),
        created_actors=created_actors,
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    return fake_ray


@pytest.mark.parametrize(
    ("module_name", "factory_name"),
    [
        ("mint_server.backend.maintenance_cron_actor", "_get_or_create_actor"),
        ("mint_server.backend.training_cleanup_executor", "_get_or_create_actor"),
        ("mint_server.backend.sampling_cleanup_executor", "_get_or_create_actor"),
    ],
)
def test_issue_519_detached_actor_creation_skips_blocking_probe(monkeypatch, module_name: str, factory_name: str) -> None:
    fake_ray = _install_fake_ray(monkeypatch)
    module = importlib.import_module(module_name)
    config_module = importlib.import_module("mint_server.config")
    monkeypatch.setattr(module, "_ACTOR_HANDLE", None, raising=False)
    monkeypatch.setattr(module, "apply_detached_actor_resources", lambda *_args, **_kwargs: None, raising=False)
    monkeypatch.setattr(module, "actor_runtime_env", lambda **_kwargs: {"env_vars": {}}, raising=False)
    monkeypatch.setattr(config_module, "apply_detached_actor_resources", lambda *_args, **_kwargs: None, raising=False)
    monkeypatch.setattr(config_module, "actor_runtime_env", lambda **_kwargs: {"env_vars": {}}, raising=False)
    monkeypatch.setattr(config_module, "actor_runtime_env_vars", lambda **_kwargs: {}, raising=False)

    actor = getattr(module, factory_name)()

    assert actor is fake_ray.created_actors[-1]
    assert len(fake_ray.created_actors) == 1
