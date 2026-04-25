from __future__ import annotations

import asyncio
import importlib
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_issue_364_queue_supervisor_claim_and_heartbeat(monkeypatch) -> None:
    import tinker_server.backend.queue_supervisor as qs

    class _FakeActor:
        class _ClaimRemote:
            def remote(self, **_kwargs):
                return {
                    "generation_id": 1,
                    "owner_id": qs.queue_supervisor.owner_id(),
                    "state": "starting",
                }

        class _HeartbeatRemote:
            def remote(self, **_kwargs):
                return True

        class _BeginRemote:
            def remote(self, **_kwargs):
                return True

        class _FinishRemote:
            def remote(self, **_kwargs):
                return True

        class _CurrentRemote:
            def remote(self, **_kwargs):
                return True

        class _FencedRemote:
            def remote(self, **_kwargs):
                return None

        class _SnapshotRemote:
            def remote(self):
                return {
                    "generation_id": 1,
                    "state": "active",
                    "code_identity": qs.CURRENT_CODE_IDENTITY,
                    "runtime_contract_digest": qs._runtime_contract_digest(),
                }

        @property
        def claim_generation(self):
            return self._ClaimRemote()

        @property
        def heartbeat(self):
            return self._HeartbeatRemote()

        @property
        def begin_reconcile(self):
            return self._BeginRemote()

        @property
        def finish_reconcile(self):
            return self._FinishRemote()

        @property
        def is_generation_current(self):
            return self._CurrentRemote()

        @property
        def record_fenced_worker(self):
            return self._FencedRemote()

        @property
        def snapshot(self):
            return self._SnapshotRemote()

    async def _identity(ref):
        return ref

    monkeypatch.setattr(qs.queue_supervisor, "_get_ray_actor", lambda: _FakeActor())
    monkeypatch.setattr(qs, "_await_ray_ref", _identity)

    claimed = await qs.queue_supervisor.async_claim_generation()
    assert claimed["generation_id"] == 1
    assert claimed["owner_id"] == qs.queue_supervisor.owner_id()
    assert await qs.queue_supervisor.async_heartbeat(generation_id=1) is True
    assert await qs.queue_supervisor.async_is_generation_current(generation_id=1) is True


def test_issue_364_queue_supervisor_same_owner_claim_keeps_active_state(monkeypatch) -> None:
    import tinker_server.backend.queue_supervisor as qs

    class _ActorState:
        def __init__(self) -> None:
            self.generation_id = 0
            self.owner_id = None
            self.expires_at = 0.0
            self.state = "inactive"
            self.now = 100.0

        def snapshot(self):
            return {
                "generation_id": self.generation_id,
                "owner_id": self.owner_id,
                "expires_at": self.expires_at,
                "state": self.state,
            }

        def claim_generation(self, *, owner_id: str, ttl_s: float):
            now = self.now
            requested_owner = str(owner_id)
            if self.expires_at > now and self.owner_id != requested_owner:
                return self.snapshot()
            if self.owner_id == requested_owner and int(self.generation_id) > 0:
                self.expires_at = now + float(ttl_s)
                return self.snapshot()
            if self.owner_id != requested_owner:
                self.generation_id += 1
            self.owner_id = requested_owner
            self.expires_at = now + float(ttl_s)
            self.state = "starting"
            return self.snapshot()

        def finish_reconcile(self):
            self.state = "active"
            return self.snapshot()

    s = _ActorState()
    first = s.claim_generation(owner_id="owner-a", ttl_s=30.0)
    assert first["generation_id"] == 1
    assert first["state"] == "starting"
    active = s.finish_reconcile()
    assert active["state"] == "active"
    claimed_again = s.claim_generation(owner_id="owner-a", ttl_s=30.0)
    assert claimed_again["generation_id"] == 1
    assert claimed_again["state"] == "active"
    s.now = 1000.0
    claimed_after_expiry = s.claim_generation(owner_id="owner-a", ttl_s=30.0)
    assert claimed_after_expiry["generation_id"] == 1
    assert claimed_after_expiry["state"] == "active"


def test_issue_364_future_store_rejects_stale_generation(monkeypatch) -> None:
    future_store_module = importlib.import_module("tinker_server.backend.future_store")

    calls: list[tuple[str, str]] = []

    class _FakeActor:
        class _FailRemote:
            def remote(self, *, request_id: str, error: str):
                calls.append((request_id, error))

        class _GetMetaRemote:
            def remote(self, *, request_id: str):
                return {"request_id": request_id}

        @property
        def fail(self):
            return self._FailRemote()

        @property
        def get_meta(self):
            return self._GetMetaRemote()

        class _ResolveRefRemote:
            def remote(self, **_kwargs):
                raise AssertionError("resolve_ref should not run for stale generation")

        @property
        def resolve_ref(self):
            return self._ResolveRefRemote()

    class _FakeQueueSupervisor:
        def is_generation_current(self, *, generation_id: int, timeout_s: float = 10.0) -> bool:
            assert generation_id == 7
            return False

        async def async_is_generation_current(self, *, generation_id: int, timeout_s: float = 10.0) -> bool:
            assert generation_id == 7
            return False

    monkeypatch.setattr(future_store_module.future_store, "_get_ray_actor", lambda: _FakeActor())
    queue_supervisor_module = importlib.import_module("tinker_server.backend.queue_supervisor")
    monkeypatch.setattr(queue_supervisor_module, "queue_supervisor", _FakeQueueSupervisor())
    monkeypatch.setattr(future_store_module, "get_current_queue_generation_id", lambda: 7)
    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(
            get=lambda value: value,
            put=lambda value: value,
            exceptions=SimpleNamespace(ActorDiedError=RuntimeError),
        ),
    )

    future_store_module.future_store.resolve("rid-1", {"ok": True})

    assert calls == [("rid-1", "stale generation finalize rejected (generation_id=7)")]


@pytest.mark.anyio
async def test_issue_364_api_work_queue_waits_for_first_generation_claim(monkeypatch) -> None:
    api_work_queue_module = importlib.import_module("tinker_server.backend.api_work_queue")

    client = api_work_queue_module.ApiWorkQueueClient()

    class _FakeQueueSupervisor:
        def owner_id(self) -> str:
            return "owner-queue"

        def poll_s(self) -> float:
            return 60.0

        async def async_claim_generation(self, *, timeout_s: float = 15.0):
            return {"generation_id": 3, "owner_id": "owner-queue", "state": "starting"}

        async def async_begin_reconcile(self, *, generation_id: int, timeout_s: float = 10.0) -> bool:
            return True

        async def async_finish_reconcile(self, *, generation_id: int, stale_reconciled: int, timeout_s: float = 10.0) -> bool:
            return True

        async def async_heartbeat(self, *, generation_id: int, timeout_s: float = 10.0) -> bool:
            return True

    class _FakeActor:
        class _SetActiveRemote:
            def remote(self, _consumer_job_id: str):
                return None

        @property
        def set_active_job_id(self):
            return self._SetActiveRemote()

    async def _get_actor_async(*, require_ready: bool = True):
        _ = require_ready
        return _FakeActor()

    async def _await_ref(ref, *, timeout_s: float | None = None):
        return ref

    worker_task = asyncio.create_task(asyncio.sleep(3600))

    async def _ensure_workers(_num_workers: int) -> None:
        client._worker_tasks = [worker_task]

    async def _reconcile(_consumer_job_id: str) -> int:
        return 0

    queue_supervisor_module = importlib.import_module("tinker_server.backend.queue_supervisor")
    monkeypatch.setattr(queue_supervisor_module, "queue_supervisor", _FakeQueueSupervisor())
    monkeypatch.setattr(client, "_get_ray_actor_async", _get_actor_async)
    monkeypatch.setattr(client, "_await_ray_ref", _await_ref)
    monkeypatch.setattr(client, "_ensure_local_workers_running", _ensure_workers)
    monkeypatch.setattr(client, "_reconcile_stale_running_requests", _reconcile)

    client._running = True
    client._desired_num_workers = 1
    loop_task = asyncio.create_task(client._queue_supervisor_loop())
    ready = await client.wait_until_execution_ready(timeout_s=1.0)

    assert ready["execution_ready"] is True
    assert ready["generation_id"] == 3
    assert client._consumer_generation_id == 3
    assert client._consumer_job_id == "owner-queue:3"
    assert client._execution_ready_event.is_set() is True

    client._running = False
    loop_task.cancel()
    await asyncio.gather(loop_task, return_exceptions=True)
    worker_task.cancel()
    await asyncio.gather(worker_task, return_exceptions=True)


@pytest.mark.anyio
async def test_issue_364_api_work_queue_restarts_workers_when_running_but_empty() -> None:
    api_work_queue_module = importlib.import_module("tinker_server.backend.api_work_queue")

    client = api_work_queue_module.ApiWorkQueueClient()
    client._running = True
    client._desired_num_workers = 1
    client._worker_tasks = []
    client._queue_supervisor_task = None
    client._execution_ready_event.set()

    scheduled = []
    original_create_task = asyncio.create_task

    def _record(coro):
        task = original_create_task(coro)
        scheduled.append(task)
        return task

    try:
        asyncio.create_task = _record  # type: ignore[assignment]
        await client.start_workers(num_workers=2)
    finally:
        asyncio.create_task = original_create_task  # type: ignore[assignment]
        for task in scheduled:
            task.cancel()
        if scheduled:
            await asyncio.gather(*scheduled, return_exceptions=True)

    assert client._desired_num_workers == 2
    assert client._execution_ready_event.is_set() is False
    assert client._queue_supervisor_task is not None
