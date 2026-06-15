import asyncio

import pytest

from mint_server.backend.stores.task_state_store import TaskStateNotFoundError, TaskStateStoreClient


class _RemoteMethod:
    def __init__(self, exc):
        self._exc = exc

    def remote(self, **_kwargs):
        raise self._exc


class _Actor:
    def __init__(self, exc):
        self.future_get_task = _RemoteMethod(exc)


class _WrappedRayTaskError(RuntimeError):
    def __init__(self, cause):
        super().__init__("RayTaskError(TaskStateNotFoundError)")
        self.cause = cause

    def as_instanceof_cause(self):
        return self.cause


def test_async_call_unwraps_task_state_not_found_from_ray_task_error(monkeypatch):
    cause = TaskStateNotFoundError("missing-request")
    client = TaskStateStoreClient()
    actor = _Actor(_WrappedRayTaskError(cause))

    async def _get_actor(**_kwargs):
        return actor

    monkeypatch.setattr(client, "_get_ray_actor_async", _get_actor)

    with pytest.raises(TaskStateNotFoundError, match="missing-request"):
        asyncio.run(client._call("future_get_task", request_id="missing-request"))


def test_sync_call_unwraps_task_state_not_found_from_ray_task_error(monkeypatch):
    cause = TaskStateNotFoundError("missing-request")
    client = TaskStateStoreClient()
    actor = _Actor(_WrappedRayTaskError(cause))
    monkeypatch.setattr(client, "_get_ray_actor_sync", lambda **_kwargs: actor)

    with pytest.raises(TaskStateNotFoundError, match="missing-request"):
        client._call_sync("future_get_task", request_id="missing-request")
