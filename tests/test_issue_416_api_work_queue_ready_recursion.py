from __future__ import annotations

import asyncio


def test_queue_supervisor_loop_fetches_queue_actor_without_ready_recursion(monkeypatch) -> None:
    from tinker_server.backend import api_work_queue as mod
    from tinker_server.backend import queue_supervisor as queue_supervisor_module

    client = mod.ApiWorkQueueClient()
    client._running = True
    client._desired_num_workers = 1

    calls: list[bool] = []

    class _Actor:
        class _Remote:
            def remote(self, *_args, **_kwargs):
                return object()

        set_active_job_id = _Remote()

    async def _fake_get_ray_actor_async(*, require_ready: bool = True):
        calls.append(require_ready)
        return _Actor()

    async def _fake_await_ray_ref(_ref, *, timeout_s=None):
        return None

    async def _fake_reconcile(_consumer_job_id: str) -> int:
        return 0

    async def _fake_workers(_num_workers: int) -> None:
        return None

    monkeypatch.setattr(client, "_get_ray_actor_async", _fake_get_ray_actor_async)
    monkeypatch.setattr(client, "_await_ray_ref", _fake_await_ray_ref)
    monkeypatch.setattr(client, "_reconcile_stale_running_requests", _fake_reconcile)
    monkeypatch.setattr(client, "_ensure_local_workers_running", _fake_workers)

    class _QueueSupervisor:
        def owner_id(self) -> str:
            return "owner"

        def poll_s(self) -> float:
            return 0.0

        async def async_claim_generation(self, *, timeout_s: float):
            return {"generation_id": 1, "owner_id": "owner", "state": "starting"}

        async def async_begin_reconcile(self, *, generation_id: int):
            return True

        async def async_finish_reconcile(self, *, generation_id: int, stale_reconciled: int):
            return True

        async def async_heartbeat(self, *, generation_id: int):
            client._running = False
            return True

    monkeypatch.setattr(queue_supervisor_module, "queue_supervisor", _QueueSupervisor())

    asyncio.run(client._queue_supervisor_loop())

    assert calls == [False]
