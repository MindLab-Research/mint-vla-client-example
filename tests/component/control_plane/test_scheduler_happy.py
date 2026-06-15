from __future__ import annotations

import time

import pytest
from typing import Any, cast

from mint_server.backend.contracts.control_plane_contracts import (
    AsyncSchedulerControlPlane,
    AsyncSchedulerQueue,
    AsyncTaskLedger,
    ModelWorkTaskGateway,
)
from mint_server.backend.contracts.engine_adapter import EngineHealth, EngineHealthStatus
from mint_server.backend.contracts.engine_liveness import EngineLivenessPush
from mint_server.backend.scheduling.model_work_task_gateway import SchedulerModelWorkTaskGateway
from mint_server.backend.stores.task_state_store import FutureStatus

from .harness import SchedulerComponentWorld

pytestmark = pytest.mark.component


def test_scheduler_component_world_exposes_typed_contracts(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        assert isinstance(world.task_gateway, ModelWorkTaskGateway)
        assert isinstance(world.task_ledger, AsyncTaskLedger)
        assert isinstance(world.runtime_queue, AsyncSchedulerQueue)
        assert isinstance(world.scheduler, AsyncSchedulerControlPlane)
        assert world.scheduler is world.runtime_queue
        assert cast(SchedulerModelWorkTaskGateway, world.task_gateway).scheduler is world.scheduler
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_happy_path_reaches_retrieve_future(tmp_path, monkeypatch) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-happy")

        await world.runtime_once()

        assert await world.observe_future_status("component-happy") == FutureStatus.DONE
        status_code, payload = await world.retrieve("component-happy", monkeypatch)
        assert status_code == 200
        assert payload == {"ok": True, "request_id": "component-happy"}
        await world.assert_consistent(terminal_request_ids=["component-happy"])
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_manual_assign_happy_path_reaches_retrieve_future(
    tmp_path,
    monkeypatch,
) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        request_id = "component-happy-manual-assign"
        submitted = cast(Any, await world.enqueue_sampling(request_id, assign=False))

        contains_before = cast(Any, await world.observe_scheduler(request_id))
        assigned = await world.scheduler.assign_pending(max_items=1)
        contains_assigned = cast(Any, await world.observe_scheduler(request_id))
        await world.assert_consistent()

        await world.runtime_once()

        assert submitted.scheduler_result.ok is True
        assert submitted.scheduler_result.assigned.get("assigned") == 0
        assert contains_before.present is True
        assert contains_before.location == "backlog"
        assert assigned.assigned == 1
        assert contains_assigned.present is True
        assert contains_assigned.location == "assigned"
        assert await world.observe_future_status(request_id) == FutureStatus.DONE
        status_code, payload = await world.retrieve(request_id, monkeypatch)
        assert status_code == 200
        assert payload == {"ok": True, "request_id": request_id}
        await world.assert_consistent(terminal_request_ids=[request_id])
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_retrieve_pending_uses_real_future_service(
    tmp_path,
    monkeypatch,
) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()
        await world.enqueue_sampling("component-pending")

        status_code, payload = await world.retrieve("component-pending", monkeypatch)

        assert status_code == 408
        assert payload["request_id"] == "component-pending"
        assert payload["type"] == "try_again"
        assert payload["status"] == "queued"
        assert payload["queue_kind"] == "model_work_scheduler"
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_supervisor_syncs_real_scheduler(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        supervisor = world.supervisor()

        out = await supervisor.reconcile_once()
        replica = out["snapshot"]["replicas"][f"{world.domain_key}::{world.replica_id}"]
        generation = int(replica["generation"])
        await supervisor.push_liveness(
            EngineLivenessPush(
                actor_name=str(replica["actor_name"]),
                domain_key=world.domain_key,
                replica_id=world.replica_id,
                consumer_id=str(replica["consumer_id"]),
                actor_generation=generation,
                running=True,
                engine_ready=True,
                engine_health=EngineHealth(status=EngineHealthStatus.READY),
                pushed_at=time.time(),
            )
        )
        await supervisor.reconcile_once()
        await world.enqueue_sampling("component-supervisor")
        claimed = await world.runtime_queue.claim(
            domain_key=world.domain_key,
            replica_id=world.replica_id,
            consumer_id=replica["consumer_id"],
            consumer_generation=generation,
            max_items=1,
            lease_ttl_s=30.0,
        )

        assert out["ok"] is True
        assert len(claimed.leases) == 1
        assert claimed.leases[0]["item"]["request_id"] == "component-supervisor"
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_duplicate_append_is_idempotent_while_pending(tmp_path) -> None:
    world = SchedulerComponentWorld(tmp_path)
    try:
        await world.start()

        first = cast(Any, await world.enqueue_sampling("component-duplicate"))
        second = cast(Any, await world.enqueue_sampling("component-duplicate"))

        assert first.scheduler_result.ok is True
        assert not first.scheduler_result.idempotent
        assert second.scheduler_result.ok is True
        assert second.scheduler_result.idempotent is True

        lease = await world.claim_one()
        assert lease["item"]["request_id"] == "component-duplicate"
        await world.claim_none()
    finally:
        world.close()
