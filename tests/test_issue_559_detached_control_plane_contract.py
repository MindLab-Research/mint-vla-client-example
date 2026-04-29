from __future__ import annotations

import importlib

import pytest


class _RemoteMethod:
    def __init__(self, value):
        self._value = value

    def remote(self, *args, **kwargs):
        _ = args, kwargs
        return self._value


class _SnapshotActor:
    def __init__(self, snapshot: dict):
        self.health_snapshot = _RemoteMethod(snapshot)
        self.snapshot = _RemoteMethod(snapshot)
        self.stats = _RemoteMethod(snapshot)


@pytest.mark.anyio
async def test_issue_559_queue_execution_runtime_recreates_on_contract_mismatch(monkeypatch) -> None:
    module = importlib.import_module("tinker_server.backend.queue_execution_runtime")

    stale = _SnapshotActor({"code_identity": "stale", "runtime_contract_digest": "stale"})
    fresh_snapshot = {
        "code_identity": module.CURRENT_CODE_IDENTITY,
        "runtime_contract_digest": module._runtime_contract_digest(),
    }
    fresh = _SnapshotActor(fresh_snapshot)
    runtime = module.QueueExecutionRuntime()
    runtime._ray_actor = stale

    killed: list[str] = []

    async def _await_ref(ref):
        return ref

    async def _to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(module, "_await_ray_ref", _await_ref)
    monkeypatch.setattr(module.asyncio, "to_thread", _to_thread)
    monkeypatch.setattr(module, "_kill_named_actor", lambda actor: killed.append("killed"))

    def _get_actor():
        if runtime._ray_actor is not None:
            return runtime._ray_actor
        runtime._ray_actor = fresh
        return fresh

    monkeypatch.setattr(runtime, "_get_ray_actor", _get_actor)

    await runtime._ensure_runtime_contract_async({"code_identity": "stale", "runtime_contract_digest": "stale"})

    assert killed == ["killed"]
    assert runtime._ray_actor is fresh


@pytest.mark.anyio
async def test_issue_559_queue_supervisor_recreates_on_contract_mismatch(monkeypatch) -> None:
    module = importlib.import_module("tinker_server.backend.queue_supervisor")

    stale = _SnapshotActor({"code_identity": "stale", "runtime_contract_digest": "stale"})
    fresh_snapshot = {
        "code_identity": module.CURRENT_CODE_IDENTITY,
        "runtime_contract_digest": module._runtime_contract_digest(),
    }
    fresh = _SnapshotActor(fresh_snapshot)
    supervisor = module.QueueSupervisor()
    supervisor._ray_actor = stale

    killed: list[str] = []

    async def _await_ref(ref):
        return ref

    async def _to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(module, "_await_ray_ref", _await_ref)
    monkeypatch.setattr(module.asyncio, "to_thread", _to_thread)
    monkeypatch.setattr(module, "_kill_named_actor", lambda actor: killed.append("killed"))

    def _get_actor():
        if supervisor._ray_actor is not None:
            return supervisor._ray_actor
        supervisor._ray_actor = fresh
        return fresh

    monkeypatch.setattr(supervisor, "_get_ray_actor", _get_actor)

    await supervisor._ensure_runtime_contract_async({"code_identity": "stale", "runtime_contract_digest": "stale"})

    assert killed == ["killed"]
    assert supervisor._ray_actor is fresh


@pytest.mark.anyio
async def test_issue_559_api_work_queue_recreates_on_contract_mismatch(monkeypatch) -> None:
    module = importlib.import_module("tinker_server.backend.api_work_queue")

    stale = _SnapshotActor({"code_identity": "stale", "runtime_contract_digest": "stale"})
    fresh_snapshot = {
        "code_identity": module.CURRENT_CODE_IDENTITY,
        "runtime_contract_digest": module._api_work_queue_runtime_contract_digest(),
    }
    fresh = _SnapshotActor(fresh_snapshot)
    client = module.ApiWorkQueueClient()
    client._ray_actor = stale

    killed: list[str] = []

    async def _await_ref(ref, *, timeout_s=None):
        _ = timeout_s
        return ref

    async def _to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(client, "_await_ray_ref", _await_ref)
    monkeypatch.setattr(module.asyncio, "to_thread", _to_thread)
    monkeypatch.setattr(module, "_kill_ray_actor", lambda actor, *, reason: killed.append(reason))
    monkeypatch.setattr(module, "_create_ray_actor", lambda require_ready=True: fresh)

    actor = await client._ensure_runtime_contract_async(stale, {"code_identity": "stale", "runtime_contract_digest": "stale"})

    assert killed == ["api_work_queue_runtime_contract_mismatch"]
    assert actor is fresh
    assert client._ray_actor is fresh


def test_queue_execution_runtime_contract_ignores_volatile_temp_env(monkeypatch) -> None:
    module = importlib.import_module("tinker_server.backend.queue_execution_runtime")

    monkeypatch.setenv("TMPDIR", "/tmp/driver-a")
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/cache-a")
    first = module._runtime_contract_digest()

    monkeypatch.setenv("TMPDIR", "/tmp/worker-b")
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/cache-b")
    second = module._runtime_contract_digest()

    assert first == second
    assert module._runtime_env_overrides()["TMPDIR"] == "/tmp/worker-b"
    assert module._runtime_env_overrides()["XDG_CACHE_HOME"] == "/tmp/cache-b"
    assert "TMPDIR" not in module._runtime_contract_payload()["runtime_env_overrides"]
    assert "XDG_CACHE_HOME" not in module._runtime_contract_payload()["runtime_env_overrides"]
