from __future__ import annotations

import asyncio

import pytest

from tinker_server.backend.task_state_store import FutureStatus
from tinker_server.backend.model_runtime_actor import (
    ModelRuntimeActor,
    _default_executor,
    default_model_runtime_actor_name,
    get_or_create_model_runtime_actor,
)
from tinker_server.backend.model_work_execution_context import (
    ModelWorkFinalize,
    get_current_model_work_consumer_generation,
    get_current_model_work_consumer_id,
    get_current_model_work_finalize_buffer,
    get_current_model_work_lease_id,
)
from tinker_server.backend.task_payload_store import TaskPayloadStore


def _lease(request_id: str = "runtime-req-1", *, finalize: bool = True) -> dict:
    lease = {
        "lease_id": f"lease-{request_id}",
        "domain_key": "vllm:model-a",
        "replica_id": "replica-0",
        "queue_id": "vllm:model-a::replica-0",
        "consumer_id": "vllm:model-a::replica-0::generation::3",
        "consumer_generation": 3,
        "leased_at": 1.0,
        "lease_expires_at": 100.0,
        "claim_attempt": 1,
        "item": {
            "request_id": request_id,
            "op": "sampling.asample",
            "request_json": b"{}",
            "user_id": None,
            "apikey_id": None,
            "throttle_principal": None,
            "webhook_url": None,
            "extra": {"domain_key": "vllm:model-a"},
            "created_at": 1.0,
            "domain_key": "vllm:model-a",
            "affinity_group": "lora:session-a",
            "ordering_key": "session:session-a",
            "token_cost": 1,
        },
    }
    if finalize:
        lease["attempt_id"] = f"attempt-{request_id}"
        lease["scheduler_epoch"] = 7
        lease["item"]["extra"] = {
            **dict(lease["item"].get("extra") or {}),
            "model_work_attempt_id": lease["attempt_id"],
        }
    return lease


def _lease_with_attempt(request_id: str, attempt_id: str) -> dict:
    lease = _lease(request_id)
    lease["item"]["extra"] = {
        **dict(lease["item"].get("extra") or {}),
        "model_work_attempt_id": attempt_id,
    }
    return lease


class _FakeScheduler:
    def __init__(
        self,
        claims: list[list[dict]] | None = None,
        complete_ok: bool = True,
        begin_finalize_ok: bool = True,
    ) -> None:
        self.claims = claims or []
        self.complete_ok = bool(complete_ok)
        self.begin_finalize_ok = bool(begin_finalize_ok)
        self.claim_calls: list[dict] = []
        self.renewed: list[dict] = []
        self.begin_finalized: list[dict] = []
        self.completed: list[dict] = []
        self.failed: list[dict] = []
        self.assigned: list[dict] = []

    async def claim_from_replica_queue(self, **kwargs):
        self.claim_calls.append(kwargs)
        if not self.claims:
            return {"ok": True, "leases": []}
        return {"ok": True, "leases": self.claims.pop(0)}

    async def renew_lease(self, **kwargs):
        self.renewed.append(kwargs)
        return {"ok": True, **kwargs}

    async def begin_finalize_lease(self, **kwargs):
        self.begin_finalized.append(kwargs)
        if not self.begin_finalize_ok:
            return {"ok": False, "reason": "unknown_lease", **kwargs}
        return {"ok": True, **kwargs}

    async def complete_lease(self, **kwargs):
        self.completed.append(kwargs)
        if not self.complete_ok:
            return {"ok": False, "reason": "unknown_lease", **kwargs}
        return {"ok": True, **kwargs}

    async def fail_lease(self, **kwargs):
        self.failed.append(kwargs)
        return {"ok": True, **kwargs}

    async def assign_pending(self, **kwargs):
        self.assigned.append(kwargs)
        return {"ok": True, "assigned": 0, "expired": 0}


class _FakeTaskFutureService:
    def __init__(
        self,
        statuses: dict[str, FutureStatus] | None = None,
        fail_terminal_write: bool = False,
    ) -> None:
        self.statuses = statuses or {}
        self.fail_terminal_write = bool(fail_terminal_write)
        self.running: list[tuple[str, dict]] = []
        self.resolved: list[tuple[str, object]] = []
        self.failed: list[tuple[str, str]] = []

    async def async_get_status(self, request_id: str) -> FutureStatus:
        if request_id not in self.statuses:
            raise KeyError(request_id)
        return self.statuses[request_id]

    async def async_mark_running(self, request_id: str, meta: dict):
        self.running.append((request_id, dict(meta)))

    async def async_get_meta(self, request_id: str) -> dict | None:
        if request_id not in self.statuses:
            raise KeyError(request_id)
        return {"model_work_attempt_id": "current-attempt"}

    async def async_resolve(self, request_id: str, result):
        buffer = get_current_model_work_finalize_buffer()
        if buffer is not None:
            buffer.finalization = ModelWorkFinalize(
                kind="resolve",
                request_id=str(request_id),
                payload=result,
            )
            return
        if self.fail_terminal_write:
            raise RuntimeError("task state terminal write failed")
        self.resolved.append((request_id, result))

    async def async_fail(self, request_id: str, error: str):
        buffer = get_current_model_work_finalize_buffer()
        if buffer is not None:
            buffer.finalization = ModelWorkFinalize(
                kind="fail",
                request_id=str(request_id),
                payload=str(error),
            )
            return
        if self.fail_terminal_write:
            raise RuntimeError("task state terminal write failed")
        self.failed.append((request_id, str(error)))

    async def async_fail_if_pending_meta_matches(
        self,
        request_id: str,
        error: str,
        *,
        expected_meta: dict | None = None,
    ) -> dict:
        if request_id not in self.statuses or self.statuses[request_id] != FutureStatus.PENDING:
            return {"failed": False, "reason": "not_pending"}
        if expected_meta:
            meta = await self.async_get_meta(request_id)
            for key, value in expected_meta.items():
                if meta.get(str(key)) != value:
                    return {"failed": False, "reason": "meta_mismatch"}
        await self.async_fail(request_id, error)
        return {"failed": True, "reason": "failed"}


class _FakeTaskStateStore:
    def __init__(self) -> None:
        self.successes: list[dict] = []
        self.failures: list[dict] = []

    async def async_commit_finalize_success(self, **kwargs):
        self.successes.append(dict(kwargs))
        return {"ok": True, "record": dict(kwargs)}

    async def async_commit_finalize_failure(self, **kwargs):
        self.failures.append(dict(kwargs))
        return {"ok": True, "record": dict(kwargs)}


@pytest.mark.anyio
async def test_issue_593_model_runtime_claims_executes_renews_and_completes() -> None:
    lease = _lease()
    scheduler = _FakeScheduler(claims=[[lease]])
    task_futures = _FakeTaskFutureService(statuses={lease["item"]["request_id"]: FutureStatus.PENDING})
    task_state_store = _FakeTaskStateStore()
    seen_context: list[tuple[str | None, str | None, int | None]] = []

    async def _executor(_lease: dict) -> None:
        seen_context.append(
            (
                get_current_model_work_lease_id(),
                get_current_model_work_consumer_id(),
                get_current_model_work_consumer_generation(),
            )
        )
        await asyncio.sleep(0.16)
        await task_futures.async_resolve(_lease["item"]["request_id"], {"ok": True})

    actor = ModelRuntimeActor(
        domain_key="vllm:model-a",
        replica_id="replica-0",
        actor_name="runtime-a",
        actor_generation=3,
        lease_ttl_s=0.3,
        scheduler_client=scheduler,
        task_futures_client=task_futures,
        task_state_store_client=task_state_store,
        executor=_executor,
    )

    result = await actor.run_once()

    assert result == {"claimed": 1, "executed": 1}
    assert scheduler.claim_calls == [
        {
            "domain_key": "vllm:model-a",
            "replica_id": "replica-0",
            "consumer_id": "vllm:model-a::replica-0::generation::3",
            "consumer_generation": 3,
            "max_items": 1,
            "token_budget": None,
            "lease_ttl_s": 0.3,
        }
    ]
    assert scheduler.assigned == [{"max_items": 1}]
    assert seen_context == [(lease["lease_id"], "vllm:model-a::replica-0::generation::3", 3)]
    assert task_futures.running[0][0] == lease["item"]["request_id"]
    assert task_futures.running[0][1]["domain_key"] == "vllm:model-a"
    assert task_futures.running[0][1]["replica_id"] == "replica-0"
    assert task_futures.running[0][1]["lease_id"] == lease["lease_id"]
    assert scheduler.renewed and scheduler.renewed[0]["lease_id"] == lease["lease_id"]
    assert task_futures.resolved == []
    assert len(task_state_store.successes) == 1
    assert task_state_store.successes[0]["request_id"] == lease["item"]["request_id"]
    assert scheduler.begin_finalized == [
        {
            "lease_id": lease["lease_id"],
            "consumer_id": "vllm:model-a::replica-0::generation::3",
            "consumer_generation": 3,
            "finalize_ttl_s": 0.3,
        }
    ]
    assert scheduler.completed == [
        {
            "lease_id": lease["lease_id"],
            "consumer_id": "vllm:model-a::replica-0::generation::3",
            "consumer_generation": 3,
        }
    ]
    assert scheduler.failed == []
    snapshot = actor.health_snapshot()
    assert snapshot["completed_total"] == 1
    assert snapshot["failed_total"] == 0
    assert snapshot["active_request_id"] is None


@pytest.mark.anyio
async def test_issue_616_model_runtime_commits_success_to_task_state_store(tmp_path) -> None:
    lease = _lease("runtime-req-task-state-success", finalize=False)
    lease["attempt_id"] = "attempt-success"
    lease["scheduler_epoch"] = 7
    lease["item"]["extra"] = {
        **dict(lease["item"].get("extra") or {}),
        "model_work_attempt_id": "attempt-success",
    }
    scheduler = _FakeScheduler(claims=[[lease]])
    task_futures = _FakeTaskFutureService(statuses={lease["item"]["request_id"]: FutureStatus.PENDING})
    task_state_store = _FakeTaskStateStore()
    payload_store = TaskPayloadStore(tmp_path)

    async def _executor(_lease: dict) -> None:
        await task_futures.async_resolve(_lease["item"]["request_id"], {"ok": True})

    actor = ModelRuntimeActor(
        domain_key="vllm:model-a",
        replica_id="replica-0",
        actor_name="runtime-a",
        actor_generation=3,
        scheduler_client=scheduler,
        task_futures_client=task_futures,
        task_state_store_client=task_state_store,
        payload_store=payload_store,
        executor=_executor,
    )

    assert await actor.run_once() == {"claimed": 1, "executed": 1}
    assert task_futures.resolved == []
    assert task_state_store.failures == []
    assert len(task_state_store.successes) == 1
    success = task_state_store.successes[0]
    assert success["request_id"] == lease["item"]["request_id"]
    assert success["lease_id"] == lease["lease_id"]
    assert success["attempt_id"] == "attempt-success"
    assert success["scheduler_epoch"] == 7
    assert success["runtime_generation"] == 3
    assert success["result_checksum"].startswith("sha256:")
    assert success["result_size_bytes"] > 0
    assert payload_store.read_json_payload(
        path=success["result_path"],
        expected_checksum=success["result_checksum"],
    ) == {"ok": True}


@pytest.mark.anyio
async def test_issue_616_model_runtime_does_not_requeue_after_task_state_success_commit(
    tmp_path,
) -> None:
    lease = _lease("runtime-req-task-state-success-future-fails", finalize=False)
    lease["attempt_id"] = "attempt-success-future-fails"
    lease["scheduler_epoch"] = 9
    lease["item"]["extra"] = {
        **dict(lease["item"].get("extra") or {}),
        "model_work_attempt_id": "attempt-success-future-fails",
    }
    scheduler = _FakeScheduler(claims=[[lease]])
    task_futures = _FakeTaskFutureService(
        statuses={lease["item"]["request_id"]: FutureStatus.PENDING},
        fail_terminal_write=True,
    )
    task_state_store = _FakeTaskStateStore()

    async def _executor(_lease: dict) -> None:
        await task_futures.async_resolve(_lease["item"]["request_id"], {"ok": True})

    actor = ModelRuntimeActor(
        domain_key="vllm:model-a",
        replica_id="replica-0",
        actor_name="runtime-a",
        actor_generation=3,
        scheduler_client=scheduler,
        task_futures_client=task_futures,
        task_state_store_client=task_state_store,
        payload_store=TaskPayloadStore(tmp_path),
        executor=_executor,
    )

    assert await actor.run_once() == {"claimed": 1, "executed": 1}
    assert task_futures.resolved == []
    assert len(task_state_store.successes) == 1
    assert scheduler.completed == [
        {
            "lease_id": lease["lease_id"],
            "consumer_id": "vllm:model-a::replica-0::generation::3",
            "consumer_generation": 3,
        }
    ]
    assert scheduler.failed == []
    assert actor.health_snapshot()["completed_total"] == 1


@pytest.mark.anyio
async def test_issue_616_model_runtime_commits_executor_failure_to_task_state_store(tmp_path) -> None:
    lease = _lease("runtime-req-task-state-failure", finalize=False)
    lease["attempt_id"] = "attempt-failure"
    lease["scheduler_epoch"] = 8
    lease["item"]["extra"] = {
        **dict(lease["item"].get("extra") or {}),
        "model_work_attempt_id": "attempt-failure",
    }
    scheduler = _FakeScheduler(claims=[[lease]])
    task_futures = _FakeTaskFutureService(statuses={lease["item"]["request_id"]: FutureStatus.PENDING})
    task_state_store = _FakeTaskStateStore()

    async def _executor(_lease: dict) -> None:
        raise RuntimeError("boom")

    actor = ModelRuntimeActor(
        domain_key="vllm:model-a",
        replica_id="replica-0",
        actor_name="runtime-a",
        actor_generation=3,
        scheduler_client=scheduler,
        task_futures_client=task_futures,
        task_state_store_client=task_state_store,
        payload_store=TaskPayloadStore(tmp_path),
        executor=_executor,
    )

    assert await actor.run_once() == {"claimed": 1, "executed": 1}
    assert task_futures.failed == []
    assert task_state_store.successes == []
    assert task_state_store.failures == [
        {
            "request_id": lease["item"]["request_id"],
            "lease_id": lease["lease_id"],
            "attempt_id": "attempt-failure",
            "scheduler_epoch": 8,
            "runtime_generation": 3,
            "error": "executor failed: boom",
        }
    ]


@pytest.mark.anyio
async def test_issue_616_model_runtime_does_not_requeue_after_task_state_failure_commit(
    tmp_path,
) -> None:
    lease = _lease("runtime-req-task-state-failure-future-fails", finalize=False)
    lease["attempt_id"] = "attempt-failure-future-fails"
    lease["scheduler_epoch"] = 10
    lease["item"]["extra"] = {
        **dict(lease["item"].get("extra") or {}),
        "model_work_attempt_id": "attempt-failure-future-fails",
    }
    scheduler = _FakeScheduler(claims=[[lease]])
    task_futures = _FakeTaskFutureService(
        statuses={lease["item"]["request_id"]: FutureStatus.PENDING},
        fail_terminal_write=True,
    )
    task_state_store = _FakeTaskStateStore()

    async def _executor(_lease: dict) -> None:
        raise RuntimeError("boom")

    actor = ModelRuntimeActor(
        domain_key="vllm:model-a",
        replica_id="replica-0",
        actor_name="runtime-a",
        actor_generation=3,
        scheduler_client=scheduler,
        task_futures_client=task_futures,
        task_state_store_client=task_state_store,
        payload_store=TaskPayloadStore(tmp_path),
        executor=_executor,
    )

    assert await actor.run_once() == {"claimed": 1, "executed": 1}
    assert task_futures.failed == []
    assert len(task_state_store.failures) == 1
    assert scheduler.completed == []
    assert scheduler.failed == [
        {
            "lease_id": lease["lease_id"],
            "consumer_id": "vllm:model-a::replica-0::generation::3",
            "consumer_generation": 3,
            "reason": "executor_failed",
            "requeue": False,
        }
    ]
    assert actor.health_snapshot()["failed_total"] == 1


@pytest.mark.anyio
async def test_issue_593_model_runtime_executor_failure_fails_future_and_lease() -> None:
    lease = _lease("runtime-req-fail")
    scheduler = _FakeScheduler(claims=[[lease]])
    task_futures = _FakeTaskFutureService(statuses={lease["item"]["request_id"]: FutureStatus.PENDING})
    task_state_store = _FakeTaskStateStore()

    async def _executor(_lease: dict) -> None:
        raise RuntimeError("boom")

    actor = ModelRuntimeActor(
        domain_key="vllm:model-a",
        replica_id="replica-0",
        actor_name="runtime-a",
        actor_generation=3,
        scheduler_client=scheduler,
        task_futures_client=task_futures,
        task_state_store_client=task_state_store,
        executor=_executor,
    )

    result = await actor.run_once()

    assert result == {"claimed": 1, "executed": 1}
    assert task_futures.failed == []
    assert task_state_store.failures == [
        {
            "request_id": lease["item"]["request_id"],
            "lease_id": lease["lease_id"],
            "attempt_id": lease["attempt_id"],
            "scheduler_epoch": lease["scheduler_epoch"],
            "runtime_generation": 3,
            "error": "executor failed: boom",
        }
    ]
    assert scheduler.begin_finalized == [
        {
            "lease_id": lease["lease_id"],
            "consumer_id": "vllm:model-a::replica-0::generation::3",
            "consumer_generation": 3,
            "finalize_ttl_s": 30.0,
        }
    ]
    assert scheduler.completed == []
    assert scheduler.failed == [
        {
            "lease_id": lease["lease_id"],
            "consumer_id": "vllm:model-a::replica-0::generation::3",
            "consumer_generation": 3,
            "reason": "executor_failed",
            "requeue": False,
        }
    ]
    snapshot = actor.health_snapshot()
    assert snapshot["completed_total"] == 0
    assert snapshot["failed_total"] == 1
    assert "RuntimeError: boom" in snapshot["last_error"]


@pytest.mark.anyio
async def test_issue_593_model_runtime_future_fail_finalization_fails_lease() -> None:
    lease = _lease("runtime-req-finalized-fail")
    scheduler = _FakeScheduler(claims=[[lease]])
    task_futures = _FakeTaskFutureService(statuses={lease["item"]["request_id"]: FutureStatus.PENDING})
    task_state_store = _FakeTaskStateStore()

    async def _executor(_lease: dict) -> None:
        await task_futures.async_fail(_lease["item"]["request_id"], "engine startup failed")

    actor = ModelRuntimeActor(
        domain_key="vllm:model-a",
        replica_id="replica-0",
        actor_name="runtime-a",
        actor_generation=3,
        scheduler_client=scheduler,
        task_futures_client=task_futures,
        task_state_store_client=task_state_store,
        executor=_executor,
    )

    result = await actor.run_once()

    assert result == {"claimed": 1, "executed": 1}
    assert task_futures.failed == []
    assert task_state_store.failures == [
        {
            "request_id": lease["item"]["request_id"],
            "lease_id": lease["lease_id"],
            "attempt_id": lease["attempt_id"],
            "scheduler_epoch": lease["scheduler_epoch"],
            "runtime_generation": 3,
            "error": "engine startup failed",
        }
    ]
    assert scheduler.completed == []
    assert scheduler.failed == [
        {
            "lease_id": lease["lease_id"],
            "consumer_id": "vllm:model-a::replica-0::generation::3",
            "consumer_generation": 3,
            "reason": "future_failed",
            "requeue": False,
        }
    ]
    snapshot = actor.health_snapshot()
    assert snapshot["completed_total"] == 0
    assert snapshot["failed_total"] == 1
    assert snapshot["last_error"] == "future failed: engine startup failed"


@pytest.mark.anyio
async def test_issue_593_model_runtime_requeues_if_task_futures_finalize_fails() -> None:
    lease = _lease("runtime-req-finalize-fail")
    scheduler = _FakeScheduler(claims=[[lease]])
    task_futures = _FakeTaskFutureService(
        statuses={lease["item"]["request_id"]: FutureStatus.PENDING},
        fail_terminal_write=True,
    )

    async def _executor(_lease: dict) -> None:
        await task_futures.async_resolve(_lease["item"]["request_id"], {"ok": True})

    actor = ModelRuntimeActor(
        domain_key="vllm:model-a",
        replica_id="replica-0",
        actor_generation=3,
        scheduler_client=scheduler,
        task_futures_client=task_futures,
        executor=_executor,
    )

    result = await actor.run_once()

    assert result == {"claimed": 1, "executed": 1}
    assert task_futures.resolved == []
    assert scheduler.completed == []
    assert scheduler.failed == [
        {
            "lease_id": lease["lease_id"],
            "consumer_id": "vllm:model-a::replica-0::generation::3",
            "consumer_generation": 3,
            "reason": "task_state_finalize_failed",
            "requeue": True,
        }
    ]
    assert actor.health_snapshot()["requeued_total"] == 1


@pytest.mark.anyio
async def test_issue_593_model_runtime_fails_future_if_lease_missing_before_finalize() -> None:
    lease = _lease("runtime-req-missing-finalize-lease")
    scheduler = _FakeScheduler(claims=[[lease]], begin_finalize_ok=False)
    task_futures = _FakeTaskFutureService(statuses={lease["item"]["request_id"]: FutureStatus.PENDING})

    async def _executor(_lease: dict) -> None:
        await task_futures.async_resolve(_lease["item"]["request_id"], {"ok": True})

    actor = ModelRuntimeActor(
        domain_key="vllm:model-a",
        replica_id="replica-0",
        actor_generation=3,
        scheduler_client=scheduler,
        task_futures_client=task_futures,
        executor=_executor,
    )

    result = await actor.run_once()

    assert result == {"claimed": 1, "executed": 1}
    assert task_futures.resolved == []
    assert task_futures.failed == []
    assert scheduler.failed == []
    snapshot = actor.health_snapshot()
    assert snapshot["failed_total"] == 0
    assert snapshot["requeued_total"] == 1


@pytest.mark.anyio
async def test_issue_593_model_runtime_releases_capacity_if_lost_lease_fail_write_fails() -> None:
    lease = _lease("runtime-req-lost-lease-fail-write")
    scheduler = _FakeScheduler(claims=[[lease]], begin_finalize_ok=False)
    task_futures = _FakeTaskFutureService(
        statuses={lease["item"]["request_id"]: FutureStatus.PENDING},
        fail_terminal_write=True,
    )

    async def _executor(_lease: dict) -> None:
        await task_futures.async_resolve(_lease["item"]["request_id"], {"ok": True})

    actor = ModelRuntimeActor(
        domain_key="vllm:model-a",
        replica_id="replica-0",
        actor_generation=3,
        scheduler_client=scheduler,
        task_futures_client=task_futures,
        executor=_executor,
    )

    result = await actor.run_once()

    assert result == {"claimed": 1, "executed": 1}
    assert task_futures.failed == []
    assert actor.health_snapshot()["failed_total"] == 0


@pytest.mark.anyio
async def test_issue_593_model_runtime_does_not_recreate_forgotten_future_on_lost_lease() -> None:
    lease = _lease("runtime-req-forgotten-lost-lease")
    scheduler = _FakeScheduler(claims=[[lease]], begin_finalize_ok=False)
    task_futures = _FakeTaskFutureService(statuses={})

    async def _executor(_lease: dict) -> None:
        await task_futures.async_resolve(_lease["item"]["request_id"], {"ok": True})

    actor = ModelRuntimeActor(
        domain_key="vllm:model-a",
        replica_id="replica-0",
        actor_generation=3,
        scheduler_client=scheduler,
        task_futures_client=task_futures,
        executor=_executor,
    )

    result = await actor.run_once()

    assert result == {"claimed": 1, "executed": 1}
    assert task_futures.failed == []
    assert actor.health_snapshot()["failed_total"] == 0


@pytest.mark.anyio
async def test_issue_593_model_runtime_does_not_fail_new_retry_on_lost_old_lease() -> None:
    lease = _lease_with_attempt("runtime-req-retried-lost-lease", "old-attempt")
    scheduler = _FakeScheduler(claims=[[lease]], begin_finalize_ok=False)
    task_futures = _FakeTaskFutureService(statuses={lease["item"]["request_id"]: FutureStatus.PENDING})

    async def _executor(_lease: dict) -> None:
        await task_futures.async_resolve(_lease["item"]["request_id"], {"ok": True})

    actor = ModelRuntimeActor(
        domain_key="vllm:model-a",
        replica_id="replica-0",
        actor_generation=3,
        scheduler_client=scheduler,
        task_futures_client=task_futures,
        executor=_executor,
    )

    result = await actor.run_once()

    assert result == {"claimed": 1, "executed": 1}
    assert task_futures.failed == []
    assert actor.health_snapshot()["failed_total"] == 0


@pytest.mark.anyio
async def test_issue_593_model_runtime_fails_future_if_lease_missing_after_executor_failure() -> None:
    lease = _lease("runtime-req-missing-failure-lease")
    scheduler = _FakeScheduler(claims=[[lease]], begin_finalize_ok=False)
    task_futures = _FakeTaskFutureService(statuses={lease["item"]["request_id"]: FutureStatus.PENDING})

    async def _executor(_lease: dict) -> None:
        raise RuntimeError("boom")

    actor = ModelRuntimeActor(
        domain_key="vllm:model-a",
        replica_id="replica-0",
        actor_generation=3,
        scheduler_client=scheduler,
        task_futures_client=task_futures,
        executor=_executor,
    )

    result = await actor.run_once()

    assert result == {"claimed": 1, "executed": 1}
    assert task_futures.failed == []
    assert scheduler.failed == []
    snapshot = actor.health_snapshot()
    assert snapshot["failed_total"] == 0
    assert snapshot["requeued_total"] == 1


@pytest.mark.anyio
async def test_issue_593_model_runtime_releases_capacity_if_lost_failure_lease_fail_write_fails() -> None:
    lease = _lease("runtime-req-lost-failure-lease-fail-write")
    scheduler = _FakeScheduler(claims=[[lease]], begin_finalize_ok=False)
    task_futures = _FakeTaskFutureService(
        statuses={lease["item"]["request_id"]: FutureStatus.PENDING},
        fail_terminal_write=True,
    )

    async def _executor(_lease: dict) -> None:
        raise RuntimeError("boom")

    actor = ModelRuntimeActor(
        domain_key="vllm:model-a",
        replica_id="replica-0",
        actor_generation=3,
        scheduler_client=scheduler,
        task_futures_client=task_futures,
        executor=_executor,
    )

    result = await actor.run_once()

    assert result == {"claimed": 1, "executed": 1}
    assert task_futures.failed == []
    assert actor.health_snapshot()["failed_total"] == 0


@pytest.mark.anyio
async def test_issue_593_model_runtime_requeues_if_task_futures_fail_write_fails() -> None:
    lease = _lease("runtime-req-fail-write-fail")
    scheduler = _FakeScheduler(claims=[[lease]])
    task_futures = _FakeTaskFutureService(
        statuses={lease["item"]["request_id"]: FutureStatus.PENDING},
        fail_terminal_write=True,
    )

    async def _executor(_lease: dict) -> None:
        raise RuntimeError("boom")

    actor = ModelRuntimeActor(
        domain_key="vllm:model-a",
        replica_id="replica-0",
        actor_generation=3,
        scheduler_client=scheduler,
        task_futures_client=task_futures,
        executor=_executor,
    )

    result = await actor.run_once()

    assert result == {"claimed": 1, "executed": 1}
    assert task_futures.failed == []
    assert scheduler.completed == []
    assert scheduler.failed == [
        {
            "lease_id": lease["lease_id"],
            "consumer_id": "vllm:model-a::replica-0::generation::3",
            "consumer_generation": 3,
            "reason": "task_state_finalize_failed",
            "requeue": True,
        }
    ]
    assert actor.health_snapshot()["requeued_total"] == 1


@pytest.mark.anyio
async def test_issue_593_model_runtime_requeues_if_mark_running_fails() -> None:
    lease = _lease("runtime-req-mark-running-fail")
    scheduler = _FakeScheduler(claims=[[lease]])
    task_futures = _FakeTaskFutureService(statuses={lease["item"]["request_id"]: FutureStatus.PENDING})

    async def _executor(_lease: dict) -> None:
        await task_futures.async_resolve(_lease["item"]["request_id"], {"ok": True})

    actor = ModelRuntimeActor(
        domain_key="vllm:model-a",
        replica_id="replica-0",
        actor_generation=3,
        scheduler_client=scheduler,
        task_futures_client=task_futures,
        executor=_executor,
    )

    async def _mark_running(_lease: dict) -> None:
        raise RuntimeError("mark running boom")

    actor._mark_running = _mark_running
    result = await actor.run_once()

    assert result == {"claimed": 1, "executed": 1}
    assert task_futures.failed == []
    assert scheduler.failed == [
        {
            "lease_id": lease["lease_id"],
            "consumer_id": "vllm:model-a::replica-0::generation::3",
            "consumer_generation": 3,
            "reason": "mark_running_failed",
            "requeue": True,
        }
    ]
    assert actor.health_snapshot()["requeued_total"] == 1


@pytest.mark.anyio
async def test_issue_593_model_runtime_skips_non_pending_future_without_execution() -> None:
    lease = _lease("runtime-req-done")
    scheduler = _FakeScheduler(claims=[[lease]])
    task_futures = _FakeTaskFutureService(statuses={lease["item"]["request_id"]: FutureStatus.DONE})
    executed = False

    async def _executor(_lease: dict) -> None:
        nonlocal executed
        executed = True

    actor = ModelRuntimeActor(
        domain_key="vllm:model-a",
        replica_id="replica-0",
        actor_generation=3,
        scheduler_client=scheduler,
        task_futures_client=task_futures,
        executor=_executor,
    )

    result = await actor.run_once()

    assert result == {"claimed": 1, "executed": 1}
    assert executed is False
    assert scheduler.completed == [
        {
            "lease_id": lease["lease_id"],
            "consumer_id": "vllm:model-a::replica-0::generation::3",
            "consumer_generation": 3,
        }
    ]
    assert task_futures.running == []


@pytest.mark.anyio
async def test_issue_593_model_runtime_empty_poll_and_drain() -> None:
    scheduler = _FakeScheduler()
    actor = ModelRuntimeActor(
        domain_key="vllm:model-a",
        replica_id="replica-0",
        scheduler_client=scheduler,
        task_futures_client=_FakeTaskFutureService(),
        executor=lambda _lease: asyncio.sleep(0),
    )

    empty = await actor.run_once()
    assert empty == {"claimed": 0, "executed": 0}
    assert actor.health_snapshot()["empty_polls_total"] == 1

    drained = await actor.drain()
    assert drained["draining"] is True
    assert await actor.run_once() == {"claimed": 0, "executed": 0, "draining": True}
    assert len(scheduler.claim_calls) == 1


@pytest.mark.anyio
async def test_issue_593_model_runtime_empty_poll_preserves_last_error() -> None:
    lease = _lease("runtime-req-failed-then-idle")
    scheduler = _FakeScheduler(claims=[[lease]])
    task_futures = _FakeTaskFutureService(statuses={lease["item"]["request_id"]: FutureStatus.PENDING})
    task_state_store = _FakeTaskStateStore()

    async def _executor(_lease: dict) -> None:
        await task_futures.async_fail(_lease["item"]["request_id"], "engine startup failed")

    actor = ModelRuntimeActor(
        domain_key="vllm:model-a",
        replica_id="replica-0",
        scheduler_client=scheduler,
        task_futures_client=task_futures,
        task_state_store_client=task_state_store,
        executor=_executor,
    )

    assert await actor.run_once() == {"claimed": 1, "executed": 1}
    assert actor.health_snapshot()["last_error"] == "future failed: engine startup failed"
    assert await actor.run_once() == {"claimed": 0, "executed": 0}

    snapshot = actor.health_snapshot()
    assert snapshot["last_error"] == "future failed: engine startup failed"
    assert snapshot["last_error_traceback"] is None


@pytest.mark.anyio
async def test_issue_593_model_runtime_success_clears_previous_error() -> None:
    failed_lease = _lease("runtime-req-failed-then-success")
    ok_lease = _lease("runtime-req-success-after-error")
    scheduler = _FakeScheduler(claims=[[failed_lease], [ok_lease]])
    task_state_store = _FakeTaskStateStore()
    task_futures = _FakeTaskFutureService(
        statuses={
            failed_lease["item"]["request_id"]: FutureStatus.PENDING,
            ok_lease["item"]["request_id"]: FutureStatus.PENDING,
        }
    )

    async def _executor(lease: dict) -> None:
        request_id = lease["item"]["request_id"]
        if request_id == failed_lease["item"]["request_id"]:
            await task_futures.async_fail(request_id, "engine startup failed")
            return
        await task_futures.async_resolve(request_id, {"ok": True})

    actor = ModelRuntimeActor(
        domain_key="vllm:model-a",
        replica_id="replica-0",
        scheduler_client=scheduler,
        task_futures_client=task_futures,
        task_state_store_client=task_state_store,
        executor=_executor,
    )

    assert await actor.run_once() == {"claimed": 1, "executed": 1}
    assert actor.health_snapshot()["last_error"] == "future failed: engine startup failed"
    assert await actor.run_once() == {"claimed": 1, "executed": 1}

    snapshot = actor.health_snapshot()
    assert snapshot["last_error"] is None
    assert snapshot["last_error_traceback"] is None
    assert snapshot["completed_total"] == 1
    assert snapshot["failed_total"] == 1


def test_issue_593_model_runtime_default_actor_name_is_stable() -> None:
    assert (
        default_model_runtime_actor_name("vllm:Qwen/Qwen3-30B-A3B-Instruct-2507", "replica-0")
        == "mint_model_runtime_vllm-qwen-qwen3-30b-a3b-instruct-2507_replica-0"
    )


@pytest.mark.anyio
async def test_issue_593_default_executor_initializes_execution_bindings(monkeypatch) -> None:
    import tinker_server.backend.model_runtime_actor as runtime_module
    import tinker_server.backend.model_work_dispatch as dispatch_module

    calls: list[str] = []

    async def _initialize_execution_bindings():
        calls.append("init")
        return {"inference_manager": object()}

    async def _execute_work_item(item, **_kwargs):
        calls.append(f"execute:{item.op}")

    monkeypatch.setattr(runtime_module, "_EXECUTION_BINDINGS", None)
    monkeypatch.setattr(
        "tinker_server.backend.execution_bindings.initialize_execution_bindings",
        _initialize_execution_bindings,
    )
    monkeypatch.setattr(dispatch_module, "execute_model_work_item", _execute_work_item)

    await _default_executor(_lease())
    await _default_executor(_lease("runtime-req-2"))

    assert calls == ["init", "execute:sampling.asample", "execute:sampling.asample"]


@pytest.mark.anyio
async def test_issue_616_default_executor_accepts_non_sampling_ops(monkeypatch) -> None:
    import tinker_server.backend.model_runtime_actor as runtime_module
    import tinker_server.backend.model_work_dispatch as dispatch_module

    calls: list[str] = []
    lease = _lease("runtime-training-req")
    lease["item"]["op"] = "training.forward_backward"

    async def _initialize_execution_bindings():
        calls.append("init")
        return {"train_manager": object()}

    async def _execute_work_item(item, **_kwargs):
        calls.append(f"execute:{item.op}")

    monkeypatch.setattr(runtime_module, "_EXECUTION_BINDINGS", None)
    monkeypatch.setattr(
        "tinker_server.backend.execution_bindings.initialize_execution_bindings",
        _initialize_execution_bindings,
    )
    monkeypatch.setattr(dispatch_module, "execute_model_work_item", _execute_work_item)

    await _default_executor(lease)

    assert calls == ["init", "execute:training.forward_backward"]


def test_issue_593_get_or_create_recreates_stale_generation(monkeypatch) -> None:
    killed: list[object] = []
    created: list[dict] = []

    class _RemoteResult:
        def __init__(self, value):
            self.value = value

    class _ExistingActor:
        class _Health:
            @staticmethod
            def remote():
                return _RemoteResult(
                    {
                        "domain_key": "vllm:model-a",
                        "replica_id": "replica-0",
                        "actor_generation": 2,
                    }
                )

        health_snapshot = _Health()

    class _RemoteClass:
        def options(self, **options):
            created.append({"options": dict(options)})
            return self

        def remote(self, **kwargs):
            created[-1]["kwargs"] = dict(kwargs)
            return {"created": True}

    class _Ray:
        @staticmethod
        def get_actor(_name, namespace=None):
            _ = namespace
            return _ExistingActor()

        @staticmethod
        def kill(actor, no_restart=True):
            killed.append((actor, no_restart))

        @staticmethod
        def remote(**_kwargs):
            return lambda _cls: _RemoteClass()

    monkeypatch.setitem(__import__("sys").modules, "ray", _Ray)
    monkeypatch.setattr(
        "tinker_server.backend.model_runtime_actor.sync_get_ray_ref",
        lambda ref, timeout_s=None: ref.value,
    )
    monkeypatch.setattr(
        "tinker_server.backend.model_runtime_actor.apply_detached_actor_resources",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "tinker_server.backend.model_runtime_actor.actor_runtime_env_vars",
        lambda **_kwargs: {},
    )

    out = get_or_create_model_runtime_actor(
        domain_key="vllm:model-a",
        replica_id="replica-0",
        actor_name="actor-a",
        actor_generation=3,
    )

    assert out == {"created": True}
    assert killed and killed[0][1] is True
    assert created[-1]["kwargs"]["actor_generation"] == 3
