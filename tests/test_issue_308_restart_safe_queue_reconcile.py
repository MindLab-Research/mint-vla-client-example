import anyio
import importlib

from tinker_server.backend import api_work_queue as queue_mod
from tinker_server.backend import capacity_manager as capacity_manager_mod

task_state_store_mod = importlib.import_module("tinker_server.backend.task_state_store")


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

    class _MetricsSeedRemote:
        def __init__(self, payload):
            self._payload = payload

        def remote(self):
            return self._payload

    @property
    def release_stale_scheduler_leases(self):
        return self._ReleaseStaleRemote(self)

    @property
    def clear_active_job_id_if_matches(self):
        return self._ClearActiveRemote(self)

    @property
    def metrics_seed_snapshot(self):
        return self._MetricsSeedRemote(
            {
                "stats": {"enqueued": 0, "dequeued": 0},
                "queued_items": [],
            }
        )


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
        return ref() if callable(ref) else ref


async def _noop_worker_loop(self, worker_idx: int):
    return None


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
    monkeypatch.setattr(task_state_store_mod, "task_state_futures", future_store)
    monkeypatch.setattr(capacity_manager_mod, "capacity_manager", capacity_manager)

    reconciled = anyio.run(client._reconcile_stale_running_requests, "job-new")

    assert reconciled == 2
    assert actor.released_consumer_ids == ["job-new"]
    assert future_store.calls == [("job-new", "api server restarted while request was running")]
    assert capacity_manager.released == ["rid-a", "rid-b"]


def test_start_workers_continues_when_snapshot_hydration_baseline_missing(monkeypatch):
    client = queue_mod.ApiWorkQueueClient()
    actor = _StubActor()
    future_store = _StubFutureStore([])
    capacity_manager = _StubCapacityManager()

    monkeypatch.setattr(client, "_get_ray_actor", lambda: actor)
    monkeypatch.setattr(queue_mod.ApiWorkQueueClient, "_worker_loop", _noop_worker_loop, raising=False)
    monkeypatch.setattr(task_state_store_mod, "task_state_futures", future_store)
    monkeypatch.setattr(capacity_manager_mod, "capacity_manager", capacity_manager)
    monkeypatch.setitem(__import__("sys").modules, "ray", _StubRay)
    monkeypatch.setenv("MINT_API_WORK_QUEUE_METRICS_HYDRATE_STARTUP_RETRIES", "3")
    monkeypatch.setenv("MINT_API_WORK_QUEUE_METRICS_HYDRATE_RETRY_DELAY_S", "0")

    attempts = {"count": 0}

    def _always_fail_hydrate(*, timeout_s: float = 10.0, force: bool = False) -> bool:
        attempts["count"] += 1
        return False

    monkeypatch.setattr(client, "hydrate_metrics_snapshot", _always_fail_hydrate)

    async def _noop_queue_supervisor_loop():
        return None

    monkeypatch.setattr(client, "_queue_supervisor_loop", _noop_queue_supervisor_loop)

    async def _run():
        await client.start_workers(num_workers=1)
        await client.shutdown()

    anyio.run(_run)

    assert attempts["count"] == 3
    assert client._consumer_job_id is None
