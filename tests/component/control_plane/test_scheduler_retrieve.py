from __future__ import annotations

import asyncio
import time

import pytest

from mint_server.backend.control_plane_contracts import ExecutorOutcome
from mint_server.backend.task_state_store import FutureStatus

from .harness import SchedulerComponentWorld
from .invariants import assert_terminal_not_scheduled


pytestmark = pytest.mark.component


@pytest.mark.anyio
async def test_scheduler_component_retrieve_pending_survives_scheduler_restart(
    tmp_path,
    monkeypatch,
) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        request_id = "component-retrieve-orphan"
        await world.enqueue_sampling(request_id, assign=False)
        await world.acquire_owner(owner_id="component-scheduler-orphan-probe", now=time.time() + 31.0)
        world.replace_scheduler(owner_id="component-scheduler-orphan-probe")

        status_code, payload = await world.retrieve(request_id, monkeypatch)

        assert status_code == 408
        assert payload["request_id"] == request_id
        assert payload["type"] == "try_again"
        assert payload["status"] == "queued"
        assert payload["queue_kind"] == "model_work_scheduler"
        assert await world.observe_future_status(request_id) == FutureStatus.PENDING
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_retrieve_orphan_pending_waits_for_reaper(
    tmp_path,
    monkeypatch,
) -> None:
    class AbsentScheduler:
        async def contains_request(
            self,
            *,
            request_id: str,
            hydrate_task_state: bool = True,
            timeout_s: float | None = None,
        ) -> dict[str, object]:
            _ = request_id, hydrate_task_state, timeout_s
            return {"ok": True, "present": False}

        async def contains(self, **kwargs) -> dict[str, object]:
            return await self.contains_request(**kwargs)

    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        request_id = "component-retrieve-orphan-pending"
        await world.enqueue_sampling(request_id, assign=False)

        status_code, payload = await world.retrieve(
            request_id,
            monkeypatch,
            scheduler_override=AbsentScheduler(),
        )

        assert status_code == 408
        assert payload["request_id"] == request_id
        assert payload["type"] == "try_again"
        assert payload["queue_kind"] == "model_work_scheduler"
        assert await world.observe_future_status(request_id) == FutureStatus.PENDING
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_retrieve_wait_returns_terminal_result(
    tmp_path,
    monkeypatch,
) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        request_id = "component-retrieve-wait-terminal"
        await world.enqueue_sampling(request_id)

        async def _complete_later() -> None:
            await asyncio.sleep(0.01)
            await world.runtime_once()

        completion = asyncio.create_task(_complete_later())
        status_code, payload = await world.retrieve(
            request_id,
            monkeypatch,
            wait_timeout_s=0.2,
        )
        await completion

        assert status_code == 200
        assert payload == {"ok": True, "request_id": request_id}
        await assert_terminal_not_scheduled(world, request_id)
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_retrieve_masks_internal_error_for_non_admin(
    tmp_path,
    monkeypatch,
) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        request_id = "component-retrieve-mask-error"
        await world.enqueue_sampling(request_id)

        async def _failing_executor(_lease: dict) -> ExecutorOutcome:
            return ExecutorOutcome(kind="user_error", error="internal gpu traceback secret")

        await world.runtime_once(executor=_failing_executor)

        admin_status, admin_payload = await world.retrieve(request_id, monkeypatch, admin=True)
        user_status, user_payload = await world.retrieve(request_id, monkeypatch, admin=False)

        assert admin_status == 200
        assert "internal gpu traceback secret" in admin_payload["error"]
        assert user_status == 200
        assert user_payload["error"] == "Operation failed. Contact administrator if issue persists."
        assert user_payload["category"] == "system"
    finally:
        world.close()
