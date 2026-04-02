import anyio
import importlib

from tinker_server.backend import api_work_queue as queue_mod
from tinker_server.backend import capacity_manager as capacity_manager_mod

future_store_mod = importlib.import_module("tinker_server.backend.future_store")


class _AwaitableRef:
    def __init__(self, result=None):
        self._result = result

    def __await__(self):
        async def _run():
            return self._result

        return _run().__await__()


class _StubActor:
    def __init__(self):
        self.released_consumer_ids: list[str] = []
        self.cleared_job_ids: list[str] = []

    class _ReleaseStaleRemote:
        def __init__(self, outer):
            self._outer = outer

        def remote(self, consumer_job_id: str):
            self._outer.released_consumer_ids.append(consumer_job_id)
            return _AwaitableRef([])

    class _ClearActiveRemote:
        def __init__(self, outer):
            self._outer = outer

        def remote(self, job_id: str):
            self._outer.cleared_job_ids.append(job_id)
            return _AwaitableRef(True)

    @property
    def release_stale_scheduler_leases(self):
        return self._ReleaseStaleRemote(self)

    @property
    def clear_active_job_id_if_matches(self):
        return self._ClearActiveRemote(self)


class _StubFutureStore:
    def __init__(self, stale_request_ids):
        self.calls: list[tuple[str, str]] = []
        self._stale_request_ids = list(stale_request_ids)

    async def async_fail_stale_running_requests(self, consumer_job_id: str, error: str):
        self.calls.append((consumer_job_id, error))
        return list(self._stale_request_ids)


class _StubCapacityManager:
    def __init__(self):
        self.released: list[str] = []

    async def async_release_all(self, request_id: str) -> None:
        self.released.append(request_id)


def test_reconcile_stale_running_requests(monkeypatch):
    client = queue_mod.ApiWorkQueueClient()
    actor = _StubActor()
    future_store = _StubFutureStore(["rid-a", "rid-b"])
    capacity_manager = _StubCapacityManager()

    async def _get_ray_actor_async():
        return actor

    async def _await_ray_ref(ref, *, timeout_s=None):
        return await ref

    monkeypatch.setattr(client, "_get_ray_actor_async", _get_ray_actor_async)
    monkeypatch.setattr(client, "_await_ray_ref", _await_ray_ref)
    monkeypatch.setattr(future_store_mod, "future_store", future_store)
    monkeypatch.setattr(capacity_manager_mod, "capacity_manager", capacity_manager)

    reconciled = anyio.run(client._reconcile_stale_running_requests, "job-new")

    assert reconciled == 2
    assert actor.released_consumer_ids == ["job-new"]
    assert future_store.calls == [("job-new", "api server restarted while request was running")]
    assert capacity_manager.released == ["rid-a", "rid-b"]
