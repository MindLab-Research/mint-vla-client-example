from __future__ import annotations

import asyncio
import time

import pytest

from mint_server.backend.control_plane_contracts import ExecutorOutcome, LeaseToken
from mint_server.backend.task_state_store import FutureStatus
from mint_server.backend.model_engine_host import (
    ModelEngineHost,
    _default_executor,
    default_model_engine_host_name,
    get_or_create_model_engine_host,
)
from mint_server.backend.model_work_execution_context import (
    get_current_model_work_consumer_generation,
    get_current_model_work_consumer_id,
    get_current_model_work_lease_id,
)
from mint_server.backend.task_payload_store import TaskPayloadStore


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


def _finish_failure_kwargs(lease: dict, *, consumer_id: str, error: str) -> dict:
    return {
        "request_id": lease["item"]["request_id"],
        "lease_id": lease["lease_id"],
        "attempt_id": lease["attempt_id"],
        "scheduler_epoch": lease["scheduler_epoch"],
        "consumer_id": consumer_id,
        "consumer_generation": 3,
        "error": error,
    }


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
        finish_success_error: Exception | None = None,
        finish_failure_error: Exception | None = None,
    ) -> None:
        self.claims = claims or []
        self.complete_ok = bool(complete_ok)
        self.begin_finalize_ok = bool(begin_finalize_ok)
        self.finish_success_error = finish_success_error
        self.finish_failure_error = finish_failure_error
        self.claim_calls: list[dict] = []
        self.renewed: list[dict] = []
        self.begin_finalized: list[dict] = []
        self.completed: list[dict] = []
        self.finished_success: list[dict] = []
        self.finished_failure: list[dict] = []
        self.failed: list[dict] = []
        self.assigned: list[dict] = []
        self.request_id_by_lease_id: dict[str, str] = {}

    def _kwargs_with_lease_fields(self, kwargs: dict, *, include_full: bool = False) -> dict:
        lease = kwargs.pop("lease", None)
        if lease is None:
            return kwargs
        token = lease if isinstance(lease, LeaseToken) else LeaseToken.from_wire(dict(lease))
        token_kwargs = token.to_wire()
        if not include_full:
            token_kwargs.pop("request_id", None)
            token_kwargs.pop("attempt_id", None)
            token_kwargs.pop("scheduler_epoch", None)
        return {**token_kwargs, **kwargs}

    async def claim(self, **kwargs):
        self.claim_calls.append(kwargs)
        if not self.claims:
            return {"ok": True, "leases": []}
        leases = self.claims.pop(0)
        for lease in leases:
            self.request_id_by_lease_id[str(lease["lease_id"])] = str(lease["item"]["request_id"])
        return {"ok": True, "leases": leases}

    async def renew(self, **kwargs):
        kwargs = self._kwargs_with_lease_fields(kwargs)
        self.renewed.append(kwargs)
        return {"ok": True, **kwargs}

    async def begin_finalize(self, **kwargs):
        kwargs = self._kwargs_with_lease_fields(kwargs)
        self.begin_finalized.append(kwargs)
        if not self.begin_finalize_ok:
            return {"ok": False, "reason": "unknown_lease", **kwargs}
        return {"ok": True, **kwargs}

    async def complete(self, **kwargs):
        kwargs = self._kwargs_with_lease_fields(kwargs)
        self.completed.append(kwargs)
        if not self.complete_ok:
            return {"ok": False, "reason": "unknown_lease", **kwargs}
        return {"ok": True, **kwargs}

    async def finish_success(self, **kwargs):
        kwargs = self._kwargs_with_lease_fields(kwargs, include_full=True)
        self.finished_success.append(kwargs)
        if self.finish_success_error is not None:
            raise self.finish_success_error
        if not self.complete_ok:
            return {"ok": False, "reason": "unknown_lease", **kwargs}
        return {
            "ok": True,
            "request_id": self.request_id_by_lease_id.get(str(kwargs["lease_id"])),
            "status": "done",
        }

    async def finish_failure(self, **kwargs):
        kwargs = self._kwargs_with_lease_fields(kwargs, include_full=True)
        self.finished_failure.append(kwargs)
        if self.finish_failure_error is not None:
            raise self.finish_failure_error
        return {
            "ok": True,
            "request_id": self.request_id_by_lease_id.get(str(kwargs["lease_id"])),
            "status": "failed",
        }

    async def fail(self, **kwargs):
        kwargs = self._kwargs_with_lease_fields(kwargs)
        self.failed.append(kwargs)
        return {"ok": True, **kwargs}


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

    async def async_resolve(self, request_id: str, result, *, billing_observations=None):
        _ = billing_observations
        if self.fail_terminal_write:
            raise RuntimeError("task state terminal write failed")
        self.resolved.append((request_id, result))

    async def async_fail(self, request_id: str, error: str):
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

    async def async_ensure_ready(self, **kwargs):
        return {"ok": True}

    async def async_ping(self, **kwargs):
        return {"ok": True}

    async def async_acquire_owner(self, **kwargs):
        return {"ok": True}

    async def async_renew_owner(self, **kwargs):
        return {"ok": True}

    async def async_create_task(self, **kwargs):
        return {"ok": True, "created": True, "record": dict(kwargs)}

    async def async_assign_task(self, **kwargs):
        return {"ok": True, "record": dict(kwargs)}

    async def async_claim_task(self, **kwargs):
        return {"ok": True, "record": dict(kwargs)}

    async def async_renew_lease(self, **kwargs):
        return {"ok": True, "record": dict(kwargs)}

    async def async_begin_finalize(self, **kwargs):
        return {"ok": True, "record": dict(kwargs)}

    async def async_commit_finalize_success(self, **kwargs):
        self.successes.append(dict(kwargs))
        return {"ok": True, "record": dict(kwargs)}

    async def async_commit_finalize_failure(self, **kwargs):
        self.failures.append(dict(kwargs))
        return {"ok": True, "record": dict(kwargs)}

    async def async_complete_task_failure(self, **kwargs):
        self.failures.append(dict(kwargs))
        return {"ok": True, "record": dict(kwargs)}

    async def async_requeue_task(self, **kwargs):
        return {"ok": True, "record": dict(kwargs)}

    async def async_forget_task(self, **kwargs):
        return {"ok": True, "record": dict(kwargs)}

    async def async_get_task(self, **kwargs):
        return {"ok": True, **dict(kwargs)}

    async def async_list_active_tasks(self, **kwargs):
        return []

    async def async_wait_task_status_change(self, **kwargs):
        return {"changed": False, "timeout": True}

    async def async_update_task_metadata(self, **kwargs):
        return {"ok": True, "record": dict(kwargs)}


@pytest.mark.anyio
async def test_issue_593_model_runtime_claims_executes_renews_and_completes() -> None:
    lease = _lease()
    scheduler = _FakeScheduler(claims=[[lease]])
    task_futures = _FakeTaskFutureService(statuses={lease["item"]["request_id"]: FutureStatus.PENDING})
    task_state_store = _FakeTaskStateStore()
    seen_context: list[tuple[str | None, str | None, int | None]] = []

    async def _executor(_lease: dict) -> ExecutorOutcome:
        seen_context.append(
            (
                get_current_model_work_lease_id(),
                get_current_model_work_consumer_id(),
                get_current_model_work_consumer_generation(),
            )
        )
        await asyncio.sleep(0.16)
        return ExecutorOutcome(kind="success", payload={"ok": True})

    actor = ModelEngineHost(
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
    assert scheduler.assigned == []
    assert seen_context == [(lease["lease_id"], "vllm:model-a::replica-0::generation::3", 3)]
    assert task_futures.running[0][0] == lease["item"]["request_id"]
    assert task_futures.running[0][1]["domain_key"] == "vllm:model-a"
    assert task_futures.running[0][1]["replica_id"] == "replica-0"
    assert task_futures.running[0][1]["lease_id"] == lease["lease_id"]
    assert scheduler.renewed and scheduler.renewed[0]["lease_id"] == lease["lease_id"]
    assert task_futures.resolved == []
    assert task_state_store.successes == []
    assert len(scheduler.finished_success) == 1
    assert scheduler.finished_success[0]["result_path"] == scheduler.begin_finalized[0]["staged_payload_path"]
    assert scheduler.begin_finalized == [
        {
            "lease_id": lease["lease_id"],
            "consumer_id": "vllm:model-a::replica-0::generation::3",
            "consumer_generation": 3,
            "finalize_ttl_s": 0.3,
            "staged_payload_path": scheduler.begin_finalized[0]["staged_payload_path"],
        }
    ]
    assert scheduler.begin_finalized[0]["staged_payload_path"].endswith(
        "/ru/runtime-req-1/attempt-runtime-req-1__lease-runtime-req-1.json"
    )
    assert scheduler.completed == []
    assert scheduler.failed == []
    snapshot = actor.health_snapshot()
    assert snapshot["completed_total"] == 1
    assert snapshot["failed_total"] == 0
    assert snapshot["active_request_id"] is None


@pytest.mark.anyio
async def test_issue_616_model_runtime_finishes_success_via_scheduler(tmp_path) -> None:
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
    billing_observations = [
        {
            "account_id": "acct-1",
            "apikey_id": "key-1",
            "request_id": lease["item"]["request_id"],
            "charge_item": "sampling",
            "quantity": 3,
            "unit": "tokens",
            "route": "sampling.asample",
            "dimension": "sample",
            "model": "Qwen/Test",
            "metadata": {},
        }
    ]

    async def _executor(_lease: dict) -> ExecutorOutcome:
        return ExecutorOutcome(
            kind="success",
            payload={"ok": True},
            billing_observations=billing_observations,
        )

    actor = ModelEngineHost(
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
    assert task_state_store.successes == []
    assert len(scheduler.finished_success) == 1
    success = scheduler.finished_success[0]
    assert scheduler.begin_finalized[0]["staged_payload_path"] == success["result_path"]
    assert success["lease_id"] == lease["lease_id"]
    assert success["consumer_id"] == "vllm:model-a::replica-0::generation::3"
    assert success["consumer_generation"] == 3
    assert success["result_checksum"].startswith("sha256:")
    assert success["result_size_bytes"] > 0
    assert success["billing_observations"] == billing_observations
    assert payload_store.read_json_payload(
        path=success["result_path"],
        expected_checksum=success["result_checksum"],
    ) == {"ok": True}


@pytest.mark.anyio
async def test_issue_593_model_runtime_accepts_executor_outcome_success(tmp_path) -> None:
    lease = _lease("runtime-req-outcome-success")
    scheduler = _FakeScheduler(claims=[[lease]])
    task_futures = _FakeTaskFutureService(statuses={lease["item"]["request_id"]: FutureStatus.PENDING})
    payload_store = TaskPayloadStore(tmp_path)

    async def _executor(_lease: dict) -> ExecutorOutcome:
        return ExecutorOutcome(
            kind="success",
            payload={"ok": True, "source": "executor_outcome"},
            billing_observations=[{"tokens": 11}],
        )

    actor = ModelEngineHost(
        domain_key="vllm:model-a",
        replica_id="replica-0",
        actor_generation=3,
        scheduler_client=scheduler,
        task_futures_client=task_futures,
        payload_store=payload_store,
        executor=_executor,
    )

    assert await actor.run_once() == {"claimed": 1, "executed": 1}
    assert task_futures.resolved == []
    assert len(scheduler.finished_success) == 1
    success = scheduler.finished_success[0]
    assert success["billing_observations"] == [{"tokens": 11}]
    assert payload_store.read_json_payload(
        path=success["result_path"],
        expected_checksum=success["result_checksum"],
    ) == {"ok": True, "source": "executor_outcome"}


@pytest.mark.anyio
async def test_issue_593_model_runtime_offloads_sync_executor_and_renews(tmp_path) -> None:
    lease = _lease("runtime-req-sync-offload")
    scheduler = _FakeScheduler(claims=[[lease]])
    task_futures = _FakeTaskFutureService(statuses={lease["item"]["request_id"]: FutureStatus.PENDING})
    payload_store = TaskPayloadStore(tmp_path)

    def _executor(_lease: dict) -> ExecutorOutcome:
        time.sleep(0.35)
        return ExecutorOutcome(kind="success", payload={"ok": True, "source": "sync_executor"})

    actor = ModelEngineHost(
        domain_key="vllm:model-a",
        replica_id="replica-0",
        actor_generation=3,
        scheduler_client=scheduler,
        task_futures_client=task_futures,
        payload_store=payload_store,
        executor=_executor,
        lease_ttl_s=0.3,
    )

    assert await actor.run_once() == {"claimed": 1, "executed": 1}
    assert scheduler.renewed and scheduler.renewed[0]["lease_id"] == lease["lease_id"]
    snapshot = actor.health_snapshot()
    assert snapshot["renewed_total"] >= 1
    assert snapshot["max_renew_rpc_latency_s"] >= 0.0
    assert snapshot["consecutive_renew_failures"] == 0
    assert snapshot["last_renew_deadline_slack_s"] is not None
    success = scheduler.finished_success[0]
    assert payload_store.read_json_payload(
        path=success["result_path"],
        expected_checksum=success["result_checksum"],
    ) == {"ok": True, "source": "sync_executor"}


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

    async def _executor(_lease: dict) -> ExecutorOutcome:
        return ExecutorOutcome(kind="success", payload={"ok": True})

    actor = ModelEngineHost(
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
    assert task_state_store.successes == []
    assert len(scheduler.finished_success) == 1
    assert scheduler.completed == []
    assert scheduler.failed == []
    assert actor.health_snapshot()["completed_total"] == 1


@pytest.mark.anyio
async def test_issue_616_model_runtime_finishes_executor_user_error_via_scheduler(tmp_path) -> None:
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

    async def _executor(_lease: dict) -> ExecutorOutcome:
        return ExecutorOutcome(kind="user_error", error="boom")

    actor = ModelEngineHost(
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
    assert scheduler.finished_failure == [
        _finish_failure_kwargs(
            lease,
            consumer_id="vllm:model-a::replica-0::generation::3",
            error="boom",
        )
    ]


@pytest.mark.anyio
async def test_issue_616_model_runtime_does_not_requeue_after_task_state_user_error_commit(
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

    async def _executor(_lease: dict) -> ExecutorOutcome:
        return ExecutorOutcome(kind="user_error", error="boom")

    actor = ModelEngineHost(
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
    assert task_state_store.failures == []
    assert len(scheduler.finished_failure) == 1
    assert scheduler.completed == []
    assert scheduler.failed == []
    assert actor.health_snapshot()["failed_total"] == 1


@pytest.mark.anyio
async def test_issue_593_model_runtime_executor_exception_requeues_lease() -> None:
    lease = _lease("runtime-req-fail")
    scheduler = _FakeScheduler(claims=[[lease]])
    task_futures = _FakeTaskFutureService(statuses={lease["item"]["request_id"]: FutureStatus.PENDING})
    task_state_store = _FakeTaskStateStore()

    async def _executor(_lease: dict) -> None:
        raise RuntimeError("boom")

    actor = ModelEngineHost(
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
    assert task_state_store.failures == []
    assert scheduler.finished_failure == []
    assert scheduler.begin_finalized == []
    assert scheduler.failed == [
        {
            "lease_id": lease["lease_id"],
            "consumer_id": "vllm:model-a::replica-0::generation::3",
            "consumer_generation": 3,
            "reason": "executor_retryable_failure",
            "requeue": True,
            "abort_finalize": True,
        }
    ]
    assert scheduler.completed == []
    snapshot = actor.health_snapshot()
    assert snapshot["completed_total"] == 0
    assert snapshot["failed_total"] == 0
    assert snapshot["requeued_total"] == 1
    assert "RuntimeError: boom" in snapshot["last_error"]


@pytest.mark.anyio
async def test_issue_593_model_runtime_dispatch_dead_actor_outcome_requeues_gpu_actor_died() -> None:
    lease = _lease("runtime-req-gpu-actor-died")
    scheduler = _FakeScheduler(claims=[[lease]])
    task_futures = _FakeTaskFutureService(statuses={lease["item"]["request_id"]: FutureStatus.PENDING})

    async def _executor(_lease: dict) -> ExecutorOutcome:
        return ExecutorOutcome(kind="fatal_backend_death", error="backend actor died")

    actor = ModelEngineHost(
        domain_key="vllm:model-a",
        replica_id="replica-0",
        actor_name="runtime-a",
        actor_generation=3,
        scheduler_client=scheduler,
        task_futures_client=task_futures,
        executor=_executor,
    )

    result = await actor.run_once()

    assert result == {"claimed": 1, "executed": 1}
    assert task_futures.failed == []
    assert scheduler.finished_failure == []
    assert scheduler.failed == [
        {
            "lease_id": lease["lease_id"],
            "consumer_id": "vllm:model-a::replica-0::generation::3",
            "consumer_generation": 3,
            "reason": "gpu_actor_died",
            "requeue": True,
            "abort_finalize": True,
        }
    ]
    snapshot = actor.health_snapshot()
    assert snapshot["failed_total"] == 0
    assert snapshot["requeued_total"] == 1
    assert "backend actor died" in snapshot["last_error"]


@pytest.mark.anyio
async def test_issue_653_model_runtime_executor_timeout_fails_future_and_lease() -> None:
    lease = _lease("runtime-req-timeout")
    lease["item"]["op"] = "training.save_weights_for_sampler"
    lease["consumer_id"] = "training:Qwen/Qwen3-0.6B::replica-0::generation::3"
    scheduler = _FakeScheduler(claims=[[lease]])
    task_futures = _FakeTaskFutureService(statuses={lease["item"]["request_id"]: FutureStatus.PENDING})
    task_state_store = _FakeTaskStateStore()
    cancelled = False

    async def _executor(_lease: dict) -> None:
        nonlocal cancelled
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled = True
            raise

    actor = ModelEngineHost(
        domain_key="training:Qwen/Qwen3-0.6B",
        replica_id="replica-0",
        actor_name="runtime-a",
        actor_generation=3,
        scheduler_client=scheduler,
        task_futures_client=task_futures,
        task_state_store_client=task_state_store,
        executor=_executor,
        execution_timeout_s=0.05,
    )

    result = await actor.run_once()

    assert result == {"claimed": 1, "executed": 1}
    assert cancelled is True
    assert task_futures.failed == []
    assert task_state_store.failures == []
    assert scheduler.finished_failure == [
        _finish_failure_kwargs(
            lease,
            consumer_id="training:Qwen/Qwen3-0.6B::replica-0::generation::3",
            error=(
                "executor failed: model work executor timed out after 0.1s "
                "op=training.save_weights_for_sampler"
            ),
        )
    ]
    assert scheduler.completed == []
    assert scheduler.failed == []
    snapshot = actor.health_snapshot()
    assert snapshot["completed_total"] == 0
    assert snapshot["failed_total"] == 1
    assert snapshot["active_request_id"] is None
    assert snapshot["execution_timeout_s"] == 0.1
    assert "TimeoutError: model work executor timed out" in snapshot["last_error"]


def test_issue_656_save_weights_runtime_timeout_uses_model_cap(monkeypatch) -> None:
    lease = _lease("runtime-req-save-timeout")
    lease["item"]["op"] = "training.save_weights_for_sampler"
    monkeypatch.setenv("MINT_SAVE_LORA_TIMEOUT_S", "1800")
    monkeypatch.delenv("MINT_MODEL_RUNTIME_SAVE_WEIGHTS_TIMEOUT_S", raising=False)
    monkeypatch.delenv("MINT_MODEL_RUNTIME_EXECUTION_TIMEOUT_GRACE_S", raising=False)

    actor = ModelEngineHost(
        domain_key="training:Qwen/Qwen3-0.6B",
        replica_id="replica-0",
        actor_name="runtime-a",
        actor_generation=3,
        base_model="Qwen/Qwen3-0.6B",
    )

    assert actor._execution_timeout_s_for_lease(lease) == 360.0


def test_issue_656_save_weights_runtime_timeout_explicit_env_override(monkeypatch) -> None:
    lease = _lease("runtime-req-save-timeout-explicit")
    lease["item"]["op"] = "training.save_weights_for_sampler"
    monkeypatch.setenv("MINT_SAVE_LORA_TIMEOUT_S", "1800")
    monkeypatch.setenv("MINT_MODEL_RUNTIME_SAVE_WEIGHTS_TIMEOUT_S", "42")

    actor = ModelEngineHost(
        domain_key="training:Qwen/Qwen3-0.6B",
        replica_id="replica-0",
        actor_name="runtime-a",
        actor_generation=3,
        base_model="Qwen/Qwen3-0.6B",
    )

    assert actor._execution_timeout_s_for_lease(lease) == 42.0


@pytest.mark.anyio
async def test_issue_656_stale_recovered_generation_fails_without_executor() -> None:
    lease = _lease("runtime-req-stale-generation")
    lease["item"]["extra"] = {
        **dict(lease["item"].get("extra") or {}),
        "actor_generation": 2,
        "queue_state": "running",
    }
    scheduler = _FakeScheduler(claims=[[lease]])
    task_futures = _FakeTaskFutureService(statuses={lease["item"]["request_id"]: FutureStatus.PENDING})
    task_state_store = _FakeTaskStateStore()
    called = False

    async def _executor(_lease: dict) -> None:
        nonlocal called
        called = True

    actor = ModelEngineHost(
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
    assert called is False
    assert task_state_store.failures == []
    assert scheduler.finished_failure == [
        _finish_failure_kwargs(
            lease,
            consumer_id="vllm:model-a::replica-0::generation::3",
            error=(
                "executor failed: model work request recovered from stale runtime generation "
                "actor_generation=2 current_actor_generation=3; request must be retried"
            ),
        )
    ]
    assert scheduler.failed == []
    snapshot = actor.health_snapshot()
    assert snapshot["failed_total"] == 1
    assert snapshot["active_request_id"] is None
    assert "stale runtime generation" in snapshot["last_error"]


@pytest.mark.anyio
async def test_issue_648_vllm_runtime_uses_dynamic_token_budget_for_claim() -> None:
    lease = _lease("runtime-vllm-budget")
    scheduler = _FakeScheduler(claims=[[lease]])
    task_futures = _FakeTaskFutureService(statuses={lease["item"]["request_id"]: FutureStatus.PENDING})
    task_state_store = _FakeTaskStateStore()
    budget_calls = 0

    async def _token_budget_provider() -> int:
        nonlocal budget_calls
        budget_calls += 1
        return 604124

    async def _executor(_lease: dict) -> ExecutorOutcome:
        return ExecutorOutcome(kind="success", payload={"ok": True})

    actor = ModelEngineHost(
        domain_key="vllm:Qwen/Qwen3-30B-A3B-Instruct-2507",
        replica_id="replica-0",
        actor_generation=3,
        max_claim=64,
        scheduler_client=scheduler,
        task_futures_client=task_futures,
        task_state_store_client=task_state_store,
        executor=_executor,
        token_budget_provider=_token_budget_provider,
    )

    assert await actor.run_once() == {"claimed": 1, "executed": 1}
    assert budget_calls == 1
    assert scheduler.claim_calls[0]["max_items"] == 64
    assert scheduler.claim_calls[0]["token_budget"] == 604124
    assert actor.health_snapshot()["dynamic_token_budget"] == 604124


@pytest.mark.anyio
async def test_issue_648_vllm_runtime_falls_back_to_single_claim_without_budget() -> None:
    scheduler = _FakeScheduler()

    async def _token_budget_provider() -> None:
        return None

    actor = ModelEngineHost(
        domain_key="vllm:model-a",
        replica_id="replica-0",
        actor_generation=3,
        max_claim=64,
        scheduler_client=scheduler,
        task_futures_client=_FakeTaskFutureService(),
        executor=lambda _lease: asyncio.sleep(0),
        token_budget_provider=_token_budget_provider,
    )

    assert await actor.run_once() == {"claimed": 0, "executed": 0}
    assert scheduler.claim_calls[0]["max_items"] == 1
    assert scheduler.claim_calls[0]["token_budget"] is None


@pytest.mark.anyio
async def test_issue_648_default_token_budget_provider_uses_kv_debug_fallback(monkeypatch) -> None:
    class _RemoteResult:
        def __init__(self, value):
            self.value = value

    class _RemoteMethod:
        def __init__(self, value):
            self.value = value

        def __call__(self):
            return None

        def remote(self):
            return _RemoteResult(self.value)

    class _VllmActor:
        get_observability_binding = _RemoteMethod({})
        get_kv_debug_info = _RemoteMethod({"kv_cache_capacity_tokens": 1000})

    class _Ray:
        @staticmethod
        def get_actor(name, namespace=None):
            assert name == "mint_vllm_model-a"
            assert namespace is not None
            return _VllmActor()

    async def _async_get_ray_ref(ref, timeout_s=None):
        _ = timeout_s
        return ref.value

    monkeypatch.setitem(__import__("sys").modules, "ray", _Ray)
    monkeypatch.setattr(
        "mint_server.backend.model_engine_host.async_get_ray_ref",
        _async_get_ray_ref,
    )

    actor = ModelEngineHost(
        domain_key="vllm:model-a",
        replica_id="replica-0",
        actor_generation=3,
        scheduler_client=_FakeScheduler(),
        task_futures_client=_FakeTaskFutureService(),
        executor=lambda _lease: asyncio.sleep(0),
    )

    assert await actor._default_token_budget_provider() == 950
    snapshot = actor.health_snapshot()
    assert snapshot["dynamic_token_capacity_tokens"] == 1000
    assert snapshot["dynamic_token_budget_ratio"] == 0.95


@pytest.mark.anyio
async def test_issue_648_model_runtime_executes_claimed_vllm_leases_concurrently() -> None:
    lease_a = _lease("runtime-vllm-concurrent-a")
    lease_b = _lease("runtime-vllm-concurrent-b")
    scheduler = _FakeScheduler(claims=[[lease_a, lease_b]])
    task_futures = _FakeTaskFutureService(
        statuses={
            lease_a["item"]["request_id"]: FutureStatus.PENDING,
            lease_b["item"]["request_id"]: FutureStatus.PENDING,
        }
    )
    task_state_store = _FakeTaskStateStore()
    started: list[str] = []
    release = asyncio.Event()

    async def _executor(lease: dict) -> ExecutorOutcome:
        started.append(lease["item"]["request_id"])
        if len(started) == 2:
            release.set()
        await asyncio.wait_for(release.wait(), timeout=1)
        return ExecutorOutcome(kind="success", payload={"ok": True})

    actor = ModelEngineHost(
        domain_key="vllm:model-a",
        replica_id="replica-0",
        actor_generation=3,
        max_claim=64,
        scheduler_client=scheduler,
        task_futures_client=task_futures,
        task_state_store_client=task_state_store,
        executor=_executor,
        token_budget_provider=lambda: asyncio.sleep(0, result=1000),
    )

    assert await actor.run_once() == {"claimed": 2, "executed": 2}
    assert sorted(started) == [
        lease_a["item"]["request_id"],
        lease_b["item"]["request_id"],
    ]
    assert len(scheduler.finished_success) == 2
    assert actor.health_snapshot()["active_lease_count"] == 0


@pytest.mark.anyio
async def test_model_runtime_renews_pending_sequential_leases() -> None:
    lease_a = _lease("runtime-train-sequential-a")
    lease_b = _lease("runtime-train-sequential-b")
    scheduler = _FakeScheduler(claims=[[lease_a, lease_b]])
    task_futures = _FakeTaskFutureService(
        statuses={
            lease_a["item"]["request_id"]: FutureStatus.PENDING,
            lease_b["item"]["request_id"]: FutureStatus.PENDING,
        }
    )
    task_state_store = _FakeTaskStateStore()
    started: list[str] = []

    async def _executor(lease: dict) -> ExecutorOutcome:
        started.append(lease["item"]["request_id"])
        await asyncio.sleep(0.16)
        return ExecutorOutcome(kind="success", payload={"ok": True})

    actor = ModelEngineHost(
        domain_key="bumblebee:model-a",
        replica_id="replica-0",
        actor_generation=3,
        max_claim=16,
        lease_ttl_s=0.3,
        scheduler_client=scheduler,
        task_futures_client=task_futures,
        task_state_store_client=task_state_store,
        executor=_executor,
    )

    assert await actor.run_once() == {"claimed": 2, "executed": 2}
    assert started == [
        lease_a["item"]["request_id"],
        lease_b["item"]["request_id"],
    ]
    renewed_lease_ids = [call["lease_id"] for call in scheduler.renewed]
    assert lease_b["lease_id"] in renewed_lease_ids
    assert len(scheduler.finished_success) == 2


@pytest.mark.anyio
async def test_issue_593_model_runtime_future_fail_finalization_fails_lease() -> None:
    lease = _lease("runtime-req-finalized-fail")
    scheduler = _FakeScheduler(claims=[[lease]])
    task_futures = _FakeTaskFutureService(statuses={lease["item"]["request_id"]: FutureStatus.PENDING})
    task_state_store = _FakeTaskStateStore()

    async def _executor(_lease: dict) -> ExecutorOutcome:
        return ExecutorOutcome(kind="user_error", error="engine startup failed")

    actor = ModelEngineHost(
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
    assert task_state_store.failures == []
    assert scheduler.finished_failure == [
        _finish_failure_kwargs(
            lease,
            consumer_id="vllm:model-a::replica-0::generation::3",
            error="engine startup failed",
        )
    ]
    assert scheduler.completed == []
    assert scheduler.failed == []
    snapshot = actor.health_snapshot()
    assert snapshot["completed_total"] == 0
    assert snapshot["failed_total"] == 1
    assert snapshot["last_error"] == "future failed: engine startup failed"


@pytest.mark.anyio
async def test_issue_593_model_runtime_requeues_if_task_futures_finalize_fails() -> None:
    lease = _lease("runtime-req-finalize-fail")
    scheduler = _FakeScheduler(
        claims=[[lease]],
        finish_success_error=RuntimeError("task state terminal write failed"),
    )
    task_futures = _FakeTaskFutureService(statuses={lease["item"]["request_id"]: FutureStatus.PENDING})

    async def _executor(_lease: dict) -> ExecutorOutcome:
        return ExecutorOutcome(kind="success", payload={"ok": True})

    actor = ModelEngineHost(
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
    assert len(scheduler.finished_success) == 1
    assert scheduler.failed == [
        {
            "lease_id": lease["lease_id"],
            "consumer_id": "vllm:model-a::replica-0::generation::3",
            "consumer_generation": 3,
            "reason": "task_state_finalize_failed",
            "requeue": True,
            "abort_finalize": True,
        }
    ]
    assert actor.health_snapshot()["requeued_total"] == 1


@pytest.mark.anyio
async def test_issue_593_model_runtime_completes_without_task_state_finalize_metadata() -> None:
    lease = _lease("runtime-req-no-task-state-finalize", finalize=False)
    scheduler = _FakeScheduler(claims=[[lease]])
    task_futures = _FakeTaskFutureService(statuses={lease["item"]["request_id"]: FutureStatus.PENDING})

    async def _executor(_lease: dict) -> ExecutorOutcome:
        return ExecutorOutcome(kind="success", payload={"ok": True})

    actor = ModelEngineHost(
        domain_key="vllm:model-a",
        replica_id="replica-0",
        actor_generation=3,
        scheduler_client=scheduler,
        task_futures_client=task_futures,
        executor=_executor,
    )

    result = await actor.run_once()

    assert result == {"claimed": 1, "executed": 1}
    assert task_futures.resolved == [(lease["item"]["request_id"], {"ok": True})]
    assert scheduler.failed == []
    assert scheduler.completed == [
        {
            "lease_id": lease["lease_id"],
            "consumer_id": "vllm:model-a::replica-0::generation::3",
            "consumer_generation": 3,
        }
    ]
    snapshot = actor.health_snapshot()
    assert snapshot["completed_total"] == 1
    assert snapshot["requeued_total"] == 0


@pytest.mark.anyio
async def test_issue_593_model_runtime_fails_future_if_lease_missing_before_finalize() -> None:
    lease = _lease("runtime-req-missing-finalize-lease")
    scheduler = _FakeScheduler(claims=[[lease]], begin_finalize_ok=False)
    task_futures = _FakeTaskFutureService(statuses={lease["item"]["request_id"]: FutureStatus.PENDING})

    async def _executor(_lease: dict) -> ExecutorOutcome:
        return ExecutorOutcome(kind="success", payload={"ok": True})

    actor = ModelEngineHost(
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

    async def _executor(_lease: dict) -> ExecutorOutcome:
        return ExecutorOutcome(kind="success", payload={"ok": True})

    actor = ModelEngineHost(
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

    async def _executor(_lease: dict) -> ExecutorOutcome:
        return ExecutorOutcome(kind="success", payload={"ok": True})

    actor = ModelEngineHost(
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

    async def _executor(_lease: dict) -> ExecutorOutcome:
        return ExecutorOutcome(kind="success", payload={"ok": True})

    actor = ModelEngineHost(
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
async def test_issue_593_model_runtime_requeues_if_executor_failure_skips_finalize() -> None:
    lease = _lease("runtime-req-missing-failure-lease")
    scheduler = _FakeScheduler(claims=[[lease]], begin_finalize_ok=False)
    task_futures = _FakeTaskFutureService(statuses={lease["item"]["request_id"]: FutureStatus.PENDING})

    async def _executor(_lease: dict) -> None:
        raise RuntimeError("boom")

    actor = ModelEngineHost(
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
    assert scheduler.failed == [
        {
            "lease_id": lease["lease_id"],
            "consumer_id": "vllm:model-a::replica-0::generation::3",
            "consumer_generation": 3,
            "reason": "executor_retryable_failure",
            "requeue": True,
            "abort_finalize": True,
        }
    ]
    snapshot = actor.health_snapshot()
    assert snapshot["failed_total"] == 0
    assert snapshot["requeued_total"] == 1


@pytest.mark.anyio
async def test_issue_593_model_runtime_requeues_if_executor_failure_and_future_fail_write_fails() -> None:
    lease = _lease("runtime-req-lost-failure-lease-fail-write")
    scheduler = _FakeScheduler(claims=[[lease]], begin_finalize_ok=False)
    task_futures = _FakeTaskFutureService(
        statuses={lease["item"]["request_id"]: FutureStatus.PENDING},
        fail_terminal_write=True,
    )

    async def _executor(_lease: dict) -> None:
        raise RuntimeError("boom")

    actor = ModelEngineHost(
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
    assert scheduler.failed == [
        {
            "lease_id": lease["lease_id"],
            "consumer_id": "vllm:model-a::replica-0::generation::3",
            "consumer_generation": 3,
            "reason": "executor_retryable_failure",
            "requeue": True,
            "abort_finalize": True,
        }
    ]
    snapshot = actor.health_snapshot()
    assert snapshot["failed_total"] == 0
    assert snapshot["requeued_total"] == 1


@pytest.mark.anyio
async def test_issue_593_model_runtime_requeues_if_task_state_user_error_finish_fails() -> None:
    lease = _lease("runtime-req-fail-write-fail")
    scheduler = _FakeScheduler(
        claims=[[lease]],
        finish_failure_error=RuntimeError("task state terminal write failed"),
    )
    task_futures = _FakeTaskFutureService(statuses={lease["item"]["request_id"]: FutureStatus.PENDING})

    async def _executor(_lease: dict) -> ExecutorOutcome:
        return ExecutorOutcome(kind="user_error", error="boom")

    actor = ModelEngineHost(
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
    assert len(scheduler.finished_failure) == 1
    assert scheduler.failed == [
        {
            "lease_id": lease["lease_id"],
            "consumer_id": "vllm:model-a::replica-0::generation::3",
            "consumer_generation": 3,
            "reason": "task_state_finalize_failed",
            "requeue": True,
            "abort_finalize": True,
        }
    ]
    assert actor.health_snapshot()["requeued_total"] == 1


@pytest.mark.anyio
async def test_issue_593_model_runtime_requeues_if_mark_running_fails() -> None:
    lease = _lease("runtime-req-mark-running-fail")
    scheduler = _FakeScheduler(claims=[[lease]])
    task_futures = _FakeTaskFutureService(statuses={lease["item"]["request_id"]: FutureStatus.PENDING})

    async def _executor(_lease: dict) -> ExecutorOutcome:
        return ExecutorOutcome(kind="success", payload={"ok": True})

    actor = ModelEngineHost(
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

    actor = ModelEngineHost(
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
    actor = ModelEngineHost(
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

    async def _executor(_lease: dict) -> ExecutorOutcome:
        return ExecutorOutcome(kind="user_error", error="engine startup failed")

    actor = ModelEngineHost(
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
async def test_issue_593_model_runtime_empty_poll_clears_transient_scheduler_mismatch() -> None:
    scheduler = _FakeScheduler()
    actor = ModelEngineHost(
        domain_key="vllm:model-a",
        replica_id="replica-0",
        scheduler_client=scheduler,
        task_futures_client=_FakeTaskFutureService(),
        executor=lambda _lease: asyncio.sleep(0),
    )
    actor._last_error = (
        "RayTaskError(ModelWorkSchedulerConflictError): consumer_id mismatch for replica "
        "'replica-0': expected 'old', got 'new'"
    )
    actor._last_error_traceback = "traceback"

    assert await actor.run_once() == {"claimed": 0, "executed": 0}

    snapshot = actor.health_snapshot()
    assert snapshot["last_error"] is None
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

    async def _executor(lease: dict) -> ExecutorOutcome:
        request_id = lease["item"]["request_id"]
        if request_id == failed_lease["item"]["request_id"]:
            return ExecutorOutcome(kind="user_error", error="engine startup failed")
        return ExecutorOutcome(kind="success", payload={"ok": True})

    actor = ModelEngineHost(
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
        default_model_engine_host_name("vllm:Qwen/Qwen3-30B-A3B-Instruct-2507", "replica-0")
        == "mint_model_runtime_vllm-qwen-qwen3-30b-a3b-instruct-2507_replica-0"
    )


@pytest.mark.anyio
async def test_issue_593_default_executor_initializes_execution_bindings(monkeypatch) -> None:
    import mint_server.backend.model_engine_host as runtime_module
    import mint_server.backend.model_work_dispatch as dispatch_module
    from mint_server.backend.execution_context import current_execution_context
    from mint_server.routes import sampling

    calls: list[str] = []
    inference_manager = object()
    original_sampling_manager = sampling.session_manager

    async def _initialize_execution_bindings():
        calls.append("init")
        return {
            "inference_manager": inference_manager,
            "train_manager": object(),
            "train_engine": object(),
            "action_manager": object(),
        }

    async def _execute_work_item(item, **_kwargs):
        assert current_execution_context() is not None
        assert current_execution_context().inference_manager is inference_manager
        assert sampling.session_manager is original_sampling_manager
        calls.append(f"execute:{item.op}")
        return ExecutorOutcome(kind="success")

    monkeypatch.setattr(runtime_module, "_EXECUTION_BINDINGS", None)
    monkeypatch.setattr(
        "mint_server.backend.execution_bindings.initialize_execution_bindings",
        _initialize_execution_bindings,
    )
    monkeypatch.setattr(dispatch_module, "execute_model_work_item", _execute_work_item)

    await _default_executor(_lease())
    await _default_executor(_lease("runtime-req-2"))

    assert calls == ["init", "execute:sampling.asample", "execute:sampling.asample"]
    assert sampling.session_manager is original_sampling_manager


@pytest.mark.anyio
async def test_issue_616_default_executor_accepts_non_sampling_ops(monkeypatch) -> None:
    import mint_server.backend.model_engine_host as runtime_module
    import mint_server.backend.model_work_dispatch as dispatch_module
    from mint_server.backend.execution_context import current_execution_context
    from mint_server.routes import training

    calls: list[str] = []
    lease = _lease("runtime-training-req")
    lease["item"]["op"] = "training.forward_backward"
    train_manager = object()
    original_training_manager = training.training_manager

    async def _initialize_execution_bindings():
        calls.append("init")
        return {
            "inference_manager": object(),
            "train_manager": train_manager,
            "train_engine": object(),
            "action_manager": object(),
        }

    async def _execute_work_item(item, **_kwargs):
        assert current_execution_context() is not None
        assert current_execution_context().train_manager is train_manager
        assert training.training_manager is original_training_manager
        calls.append(f"execute:{item.op}")
        return ExecutorOutcome(kind="success")

    monkeypatch.setattr(runtime_module, "_EXECUTION_BINDINGS", None)
    monkeypatch.setattr(
        "mint_server.backend.execution_bindings.initialize_execution_bindings",
        _initialize_execution_bindings,
    )
    monkeypatch.setattr(dispatch_module, "execute_model_work_item", _execute_work_item)

    await _default_executor(lease)

    assert calls == ["init", "execute:training.forward_backward"]
    assert training.training_manager is original_training_manager


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
        "mint_server.backend.model_engine_host.sync_get_ray_ref",
        lambda ref, timeout_s=None: ref.value,
    )
    monkeypatch.setattr(
        "mint_server.backend.model_engine_host.apply_detached_actor_resources",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "mint_server.backend.model_engine_host.actor_runtime_env_vars",
        lambda **_kwargs: {},
    )
    monkeypatch.setenv("RAY_ADDRESS", "local")

    out = get_or_create_model_engine_host(
        domain_key="vllm:model-a",
        replica_id="replica-0",
        actor_name="actor-a",
        actor_generation=3,
    )

    assert out == {"created": True}
    assert killed and killed[0][1] is True
    assert created[-1]["kwargs"]["actor_generation"] == 3


def test_issue_648_get_or_create_recreates_stale_claim_config(monkeypatch) -> None:
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
                        "actor_generation": 3,
                        "max_claim": 1,
                        "token_budget": None,
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
        "mint_server.backend.model_engine_host.sync_get_ray_ref",
        lambda ref, timeout_s=None: ref.value,
    )
    monkeypatch.setattr(
        "mint_server.backend.model_engine_host.apply_detached_actor_resources",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "mint_server.backend.model_engine_host.actor_runtime_env_vars",
        lambda **_kwargs: {},
    )
    monkeypatch.setenv("RAY_ADDRESS", "local")

    out = get_or_create_model_engine_host(
        domain_key="vllm:model-a",
        replica_id="replica-0",
        actor_name="actor-a",
        actor_generation=3,
        max_claim=64,
    )

    assert out == {"created": True}
    assert killed and killed[0][1] is True
    assert created[-1]["kwargs"]["max_claim"] == 64


def test_issue_679_get_or_create_recreates_stale_runtime_env(monkeypatch) -> None:
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
                        "domain_key": "bumblebee:model-a",
                        "replica_id": "replica-0",
                        "actor_generation": 3,
                        "max_claim": 16,
                        "token_budget": 262144,
                        "runtime_env_fingerprint": "old-placement",
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
        "mint_server.backend.model_engine_host.sync_get_ray_ref",
        lambda ref, timeout_s=None: ref.value,
    )
    monkeypatch.setattr(
        "mint_server.backend.model_engine_host.apply_detached_actor_resources",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "mint_server.backend.model_engine_host.actor_runtime_env_vars",
        lambda **_kwargs: {},
    )
    monkeypatch.setenv("RAY_ADDRESS", "local")

    out = get_or_create_model_engine_host(
        domain_key="bumblebee:model-a",
        replica_id="replica-0",
        actor_name="actor-a",
        actor_generation=3,
        max_claim=16,
        token_budget=262144,
        runtime_env_extra={"MINT_MODEL_PLACEMENT_JSON": '{"model-a":{"node_ip":"10.0.0.7"}}'},
    )

    assert out == {"created": True}
    assert killed and killed[0][1] is True
    assert created[-1]["kwargs"]["runtime_env_fingerprint"]
