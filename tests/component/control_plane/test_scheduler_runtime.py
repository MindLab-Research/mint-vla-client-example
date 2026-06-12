from __future__ import annotations

from typing import Any, cast

import pytest

from mint_server.backend.control_plane_contracts import ExecutorOutcome
from mint_server.backend.engine_liveness import EngineLivenessPush
from mint_server.backend.model_engine_host import ModelEngineHost
from mint_server.backend.task_state_store import FutureStatus

from .helpers import token
from .harness import SchedulerComponentWorld
from .invariants import assert_terminal_not_scheduled


pytestmark = pytest.mark.component


class _ComponentEngineLifecycle:
    def __init__(self) -> None:
        self.ready = True
        self.unhealthy_reasons: list[str] = []
        self.restart_calls = 0

    async def is_ready(self) -> bool:
        return self.ready

    async def mark_unhealthy(self, reason: str) -> None:
        self.ready = False
        self.unhealthy_reasons.append(str(reason))

    async def restart(self) -> None:
        self.restart_calls += 1
        self.ready = True


class _FlakyComponentLivenessPush:
    def __init__(self) -> None:
        self.payloads: list[EngineLivenessPush] = []
        self.failures = 0
        self.successes = 0

    async def __call__(self, payload: EngineLivenessPush) -> None:
        self.payloads.append(payload)
        if len(self.payloads) == 1:
            self.failures += 1
            raise RuntimeError("synthetic liveness push failure")
        self.successes += 1


@pytest.mark.anyio
async def test_scheduler_component_fail_cleans_scheduler_lease_for_missing_task(tmp_path) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
    try:
        await world.start()
        request_id = "component-runtime-missing-task"
        await world.enqueue_sampling(request_id)
        lease = await world.claim_one()
        await world.task_state.async_forget_task(request_id=request_id)

        failed = await world.scheduler.fail(
            lease=token(lease, consumer_id=world.consumer_id, consumer_generation=world.generation),
            requeue=False,
            reason="missing-task",
            abort_finalize=True,
        )

        assert failed.ok is True and failed.request_id == request_id
        assert (await world.observe_scheduler(request_id)).present is False
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_runtime_drops_assigned_missing_task_without_crashing(tmp_path) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
    try:
        await world.start()
        request_id = "component-runtime-assigned-missing-task"
        await world.enqueue_sampling(request_id)
        await world.task_state.async_forget_task(request_id=request_id)

        actor = await world.runtime_once()

        assert actor.health_snapshot()["processed_total"] == 0
        assert actor.health_snapshot()["last_error"] is None
        assert (await world.observe_scheduler(request_id)).present is False
        assert (await world.scheduler.stats())["counters"]["stale_dropped"] == 1
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_claim_skips_missing_stale_head_and_claims_next(tmp_path) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
    try:
        await world.start()
        stale_request_id = "component-missing-head"
        valid_request_id = "component-after-missing-head"
        await world.enqueue_sampling(stale_request_id)
        await world.enqueue_sampling(valid_request_id)
        await world.task_state.async_forget_task(request_id=stale_request_id)

        claimed = await world.scheduler.claim(
            domain_key=world.domain_key,
            replica_id=world.replica_id,
            consumer_id=world.consumer_id,
            consumer_generation=world.generation,
            max_items=1,
            lease_ttl_s=30.0,
        )

        assert [lease["item"]["request_id"] for lease in claimed.leases] == [valid_request_id]
        assert (await world.observe_scheduler(stale_request_id)).present is False
        assert (await world.observe_task(valid_request_id))["status"] == "leased"
        assert (await world.scheduler.stats())["counters"]["stale_dropped"] == 1
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_executor_failure_commits_failed_terminal(tmp_path, monkeypatch) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
    try:
        await world.start()
        await world.enqueue_sampling("component-exec-failed")

        async def _failing_executor(_lease: dict[str, Any]) -> ExecutorOutcome:
            return ExecutorOutcome(kind="user_error", error="synthetic executor failure")

        actor = await world.runtime_once(executor=_failing_executor)

        assert actor.health_snapshot()["failed_total"] == 1
        assert await world.observe_future_status("component-exec-failed") == FutureStatus.FAILED
        record = await world.observe_task("component-exec-failed")
        assert record["status"] == "failed"
        assert "synthetic executor failure" in str(record["error"])
        status_code, payload = await world.retrieve("component-exec-failed", monkeypatch)
        assert status_code == 200
        assert "synthetic executor failure" in payload["error"]
        assert payload["category"] == "system"
        await assert_terminal_not_scheduled(world, "component-exec-failed")
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_payload_write_failure_requeues_without_terminal_commit(tmp_path) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
    try:
        await world.start()
        await world.enqueue_sampling("component-payload-write-failed")
        world.inject_payload_write_failure(True)

        actor = await world.runtime_once()

        assert actor.health_snapshot()["requeued_total"] == 1
        record = await world.observe_task("component-payload-write-failed")
        assert record["status"] == "pending"
        assert record["result_path"] is None
        assert record["staged_payload_path"] is None
        assert (
            await world.scheduler.contains(request_id="component-payload-write-failed")
        ).present is True

        world.inject_payload_write_failure(False)
        assigned = await world.scheduler.assign_pending(max_items=1)
        await world.runtime_once()
        assert assigned.assigned == 1
        assert await world.observe_future_status("component-payload-write-failed") == FutureStatus.DONE
        await assert_terminal_not_scheduled(world, "component-payload-write-failed")
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_runtime_resumes_durable_finalize_after_engine_death(
    tmp_path,
) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
    try:
        await world.start()
        request_id = "component-engine-death-mid-finalize"
        staged_payload_path = str(world.tmp_path / "component-engine-death-mid-finalize.json")
        await world.enqueue_sampling(request_id)
        engine = _ComponentEngineLifecycle()

        async def _executor(lease: dict[str, Any]) -> ExecutorOutcome:
            begin = await world.scheduler.begin_finalize(
                lease=token(
                    lease,
                    consumer_id=world.consumer_id,
                    consumer_generation=world.generation,
                ),
                finalize_ttl_s=30.0,
                staged_payload_path=staged_payload_path,
            )
            assert begin.ok is True
            return ExecutorOutcome(
                kind="fatal_backend_death",
                error="synthetic engine died after durable finalize",
            )

        actor = ModelEngineHost(
            domain_key=world.domain_key,
            replica_id=world.replica_id,
            actor_name=f"component-runtime-{world.replica_id}",
            actor_generation=world.generation,
            poll_interval_s=0.01,
            lease_ttl_s=1.0,
            max_claim=1,
            scheduler_client=world.scheduler,
            task_futures_client=world.future_service,
            task_state_store_client=world.task_state,
            payload_store=world.payload_store,
            engine_lifecycle=engine,
            executor=_executor,
        )

        assert await actor.run_once() == {"claimed": 1, "executed": 1}

        snapshot = actor.health_snapshot()
        record = await world.observe_task(request_id)
        contains = await world.observe_scheduler(request_id)
        stats = await world.scheduler.stats()

        assert snapshot["completed_total"] == 1
        assert snapshot["failed_total"] == 0
        assert snapshot["active_lease_count"] == 0
        assert engine.unhealthy_reasons == ["synthetic engine died after durable finalize"]
        assert engine.restart_calls == 1
        assert record["status"] == "done"
        assert record["result_path"] == staged_payload_path
        assert contains.present is False
        assert stats["leases"] == []
        await assert_terminal_not_scheduled(world, request_id)
    finally:
        world.close()


@pytest.mark.anyio
async def test_scheduler_component_liveness_push_failure_does_not_interrupt_runtime(
    tmp_path,
) -> None:
    world = cast(Any, SchedulerComponentWorld(tmp_path))
    try:
        await world.start()
        request_id = "component-liveness-push-failure"
        await world.enqueue_sampling(request_id)
        engine = _ComponentEngineLifecycle()
        push = _FlakyComponentLivenessPush()

        async def _executor(lease: dict[str, Any]) -> ExecutorOutcome:
            lease_request_id = str(lease["item"]["request_id"])
            return ExecutorOutcome(
                kind="success",
                payload={"ok": True, "request_id": lease_request_id},
            )

        actor = ModelEngineHost(
            domain_key=world.domain_key,
            replica_id=world.replica_id,
            actor_name=f"component-runtime-{world.replica_id}",
            actor_generation=world.generation,
            poll_interval_s=0.01,
            lease_ttl_s=1.0,
            max_claim=1,
            scheduler_client=world.scheduler,
            task_futures_client=world.future_service,
            task_state_store_client=world.task_state,
            payload_store=world.payload_store,
            engine_lifecycle=engine,
            liveness_push=push,
            executor=_executor,
        )

        assert await actor.run_once() == {"claimed": 1, "executed": 1}
        assert await actor.run_once() == {"claimed": 0, "executed": 0}

        record = await world.observe_task(request_id)
        snapshot = actor.health_snapshot()

        assert push.failures == 1
        assert push.successes == 1
        assert len(push.payloads) == 2
        assert push.payloads[0].active_request_id is None
        assert push.payloads[1].last_error == "RuntimeError: synthetic liveness push failure"
        assert record["status"] == "done"
        assert snapshot["completed_total"] == 1
        assert snapshot["failed_total"] == 0
        assert snapshot["last_error"] == "RuntimeError: synthetic liveness push failure"
        await assert_terminal_not_scheduled(world, request_id)
    finally:
        world.close()
