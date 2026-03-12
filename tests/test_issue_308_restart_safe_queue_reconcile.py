import anyio
import importlib

from tinker_server.backend import api_work_queue as queue_mod
from tinker_server.backend import capacity_manager as capacity_manager_mod

future_store_mod = importlib.import_module("tinker_server.backend.future_store")


class _StubActor:
    def __init__(self):
        self.job_ids: list[str] = []

    class _SetActiveRemote:
        def __init__(self, outer):
            self._outer = outer

        def remote(self, job_id: str):
            self._outer.job_ids.append(job_id)
            return ("set_active_job_id", job_id)

    @property
    def set_active_job_id(self):
        return self._SetActiveRemote(self)


class _StubFutureStore:
    def __init__(self, stale_request_ids):
        self.calls: list[tuple[str, str]] = []
        self._stale_request_ids = list(stale_request_ids)

    def fail_stale_running_requests(self, consumer_job_id: str, error: str):
        self.calls.append((consumer_job_id, error))
        return list(self._stale_request_ids)


class _StubCapacityManager:
    def __init__(self):
        self.released: list[str] = []

    def release_all(self, request_id: str) -> None:
        self.released.append(request_id)


class _StubRuntimeContext:
    def get_job_id(self):
        return "job-new"


class _StubRay:
    class exceptions:
        class ActorDiedError(Exception):
            pass

        class RayActorError(Exception):
            pass

    @staticmethod
    def get_runtime_context():
        return _StubRuntimeContext()

    @staticmethod
    def get(ref, timeout=None):
        return None


async def _noop_worker_loop(self, worker_idx: int):
    return None


def test_start_workers_fails_stale_running_requests(monkeypatch):
    client = queue_mod.ApiWorkQueueClient()
    actor = _StubActor()
    future_store = _StubFutureStore(["rid-a", "rid-b"])
    capacity_manager = _StubCapacityManager()

    monkeypatch.setattr(client, "_get_ray_actor", lambda: actor)
    monkeypatch.setattr(queue_mod.ApiWorkQueueClient, "_worker_loop", _noop_worker_loop, raising=False)
    monkeypatch.setattr(future_store_mod, "future_store", future_store)
    monkeypatch.setattr(capacity_manager_mod, "capacity_manager", capacity_manager)
    monkeypatch.setitem(__import__("sys").modules, "ray", _StubRay)

    async def _run():
        await client.start_workers(num_workers=1)
        await client.shutdown()

    anyio.run(_run)

    assert actor.job_ids == ["job-new"]
    assert future_store.calls == [("job-new", "api server restarted while request was running")]
    assert capacity_manager.released == ["rid-a", "rid-b"]
