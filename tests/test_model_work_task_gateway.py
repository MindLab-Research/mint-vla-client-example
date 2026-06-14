from __future__ import annotations

import asyncio

from mint_server.backend.control_plane_contracts import (
    AppendWorkResult,
    CancelTaskResult,
    TaskRecord,
    TaskStatusChange,
)
from mint_server.backend.model_work_task_gateway import SchedulerModelWorkTaskGateway
from mint_server.backend.task_payload_presenter import present_terminal_retrieve_result
from mint_server.backend.task_payload_store import TaskPayloadStore
from mint_server.backend.task_state_store import TaskStateStore


def _submit_kwargs(request_id: str = "req-gateway") -> dict:
    return {
        "request_id": request_id,
        "op": "sampling.asample",
        "domain_key": "vllm:Qwen/Test",
        "request_json": b'{"model":"Qwen/Test"}',
        "metadata": {"queue_kind": "model_work_scheduler"},
        "user_id": "user-1",
        "apikey_id": "key-1",
        "assign": True,
        "assign_max_items": 1,
        "payload_hash": "hash-1",
    }


class _AppendWorkScheduler:
    def __init__(self, result):
        self.result = result
        self.calls: list[dict] = []

    async def append_work(self, **kwargs):
        self.calls.append(dict(kwargs))
        return self.result


class _AppendOnlyScheduler:
    def __init__(self, result):
        self.result = result
        self.calls: list[dict] = []

    async def append(self, **kwargs):
        self.calls.append(dict(kwargs))
        return self.result


class _CancelScheduler:
    def __init__(self, result):
        self.result = result
        self.calls: list[dict] = []

    async def cancel_request(self, **kwargs):
        self.calls.append(dict(kwargs))
        return self.result


class _FutureService:
    def __init__(self, *, created: bool = True):
        self.created = bool(created)
        self.created_calls: list[dict] = []
        self.cleaned: list[str] = []

    async def async_create_model_work_with_id(self, request_id: str, **kwargs):
        self.created_calls.append({"request_id": request_id, **dict(kwargs)})
        return {"request_id": request_id, "created": self.created}

    async def async_cleanup(self, request_id: str):
        self.cleaned.append(str(request_id))


def test_scheduler_gateway_submit_returns_typed_created_result() -> None:
    scheduler = _AppendWorkScheduler(
        AppendWorkResult(
            ok=True,
            request_id="req-gateway",
            assigned={"ok": True, "assigned": 1, "skipped_domains": []},
            idempotent=False,
            extra={"sampling_inflight_admission": {"ok": True}},
        )
    )
    futures = _FutureService()
    gateway = SchedulerModelWorkTaskGateway(
        scheduler_client=scheduler,
        future_service_client=futures,
    )

    result = asyncio.run(gateway.submit_task(**_submit_kwargs()))

    assert result.ok is True
    assert result.request_id == "req-gateway"
    assert result.created is True
    assert result.assigned is True
    assert result.extra["sampling_inflight_admission"] == {"ok": True}
    assert result.extra["scheduler_result"]["ok"] is True
    assert futures.created_calls == [
        {
            "request_id": "req-gateway",
            "op": "sampling.asample",
            "domain_key": "vllm:Qwen/Test",
            "request_json": b'{"model":"Qwen/Test"}',
            "meta": {"queue_kind": "model_work_scheduler", "payload_hash": "hash-1"},
            "payload_hash": "hash-1",
        }
    ]
    assert futures.cleaned == []
    assert scheduler.calls[0]["extra"]["payload_hash"] == "hash-1"
    assert scheduler.calls[0]["extra"]["queue_kind"] == "model_work_scheduler"


def test_scheduler_gateway_submit_preserves_idempotent_duplicate() -> None:
    scheduler = _AppendWorkScheduler(
        {
            "ok": True,
            "request_id": "req-gateway",
            "assigned": {"ok": True, "assigned": 0, "skipped_domains": []},
            "idempotent": True,
        }
    )
    futures = _FutureService(created=False)
    gateway = SchedulerModelWorkTaskGateway(
        scheduler_client=scheduler,
        future_service_client=futures,
    )

    result = asyncio.run(gateway.submit_task(**_submit_kwargs()))

    assert result.ok is True
    assert result.created is False
    assert result.assigned is False
    assert result.extra["scheduler_result"]["idempotent"] is True
    assert futures.created_calls[0]["request_id"] == "req-gateway"
    assert futures.cleaned == []


def test_scheduler_gateway_submit_admission_reject_is_not_created() -> None:
    scheduler = _AppendWorkScheduler(
        {
            "ok": False,
            "request_id": "req-gateway",
            "reason": "principal_domain_inflight_limit_exceeded",
            "assigned": {"ok": True, "assigned": 0, "skipped_domains": []},
            "sampling_inflight_admission": {
                "ok": False,
                "reason": "principal_domain_inflight_limit_exceeded",
            },
        }
    )
    futures = _FutureService()
    gateway = SchedulerModelWorkTaskGateway(
        scheduler_client=scheduler,
        future_service_client=futures,
    )

    result = asyncio.run(gateway.submit_task(**_submit_kwargs()))

    assert result.ok is False
    assert result.created is False
    assert result.assigned is False
    assert str(result.reason) == "principal_domain_inflight_limit_exceeded"
    assert result.extra["sampling_inflight_admission"]["ok"] is False
    assert futures.cleaned == ["req-gateway"]


def test_scheduler_gateway_submit_falls_back_to_append() -> None:
    scheduler = _AppendOnlyScheduler(
        AppendWorkResult(
            ok=True,
            request_id="req-fallback",
            assigned={"ok": True, "assigned": 0, "skipped_domains": []},
            idempotent=True,
        )
    )
    gateway = SchedulerModelWorkTaskGateway(
        scheduler_client=scheduler,
        future_service_client=_FutureService(created=False),
    )

    result = asyncio.run(gateway.submit_task(**_submit_kwargs("req-fallback")))

    assert result.ok is True
    assert result.request_id == "req-fallback"
    assert result.created is False
    assert len(scheduler.calls) == 1


def test_scheduler_gateway_submit_cleans_new_future_when_append_raises() -> None:
    class _RaisingScheduler:
        async def append_work(self, **_kwargs):
            raise RuntimeError("append failed")

    futures = _FutureService()
    gateway = SchedulerModelWorkTaskGateway(
        scheduler_client=_RaisingScheduler(),
        future_service_client=futures,
    )

    try:
        asyncio.run(gateway.submit_task(**_submit_kwargs()))
    except RuntimeError as exc:
        assert str(exc) == "append failed"
    else:
        raise AssertionError("expected append failure")

    assert futures.cleaned == ["req-gateway"]


def test_scheduler_gateway_cancel_returns_typed_result() -> None:
    scheduler = _CancelScheduler(
        {
            "ok": True,
            "request_id": "req-gateway",
            "was_terminal": False,
            "cancelled": True,
        }
    )
    gateway = SchedulerModelWorkTaskGateway(scheduler_client=scheduler)

    result = asyncio.run(
        gateway.cancel_task(
            request_id="req-gateway",
            reason="cancelled_by_user",
            timeout_s=3.0,
        )
    )

    assert isinstance(result, CancelTaskResult)
    assert result.ok is True
    assert result.request_id == "req-gateway"
    assert result.cancelled is True
    assert scheduler.calls == [
        {
            "request_id": "req-gateway",
            "reason": "cancelled_by_user",
            "timeout_s": 3.0,
        }
    ]


def test_scheduler_gateway_retrieve_ready_reads_typed_payload_ref(tmp_path) -> None:
    store = TaskStateStore.in_memory()
    payloads = TaskPayloadStore(root_dir=tmp_path / "payloads")
    try:
        store.create_task(
            request_id="req-ready",
            op="sampling.asample",
            domain_key="vllm:Qwen/Test",
            request_json=b"{}",
            metadata={"op": "sampling.asample"},
        )
        payload = payloads.write_json_payload(
            request_id="req-ready",
            attempt_id="attempt-1",
            payload={"ok": True, "value": 7},
        )
        store.complete_task_success(
            request_id="req-ready",
            result_path=str(payload["path"]),
            result_checksum=str(payload["checksum"]),
            result_size_bytes=int(payload["size_bytes"]),
        )
        gateway = SchedulerModelWorkTaskGateway(task_ledger_client=_LocalTaskLedger(store))

        result = asyncio.run(gateway.retrieve_task(request_id="req-ready"))
        presented = asyncio.run(
            present_terminal_retrieve_result(
                result,
                error_presenter=lambda error: {"error": error, "category": "system"},
                payload_store=payloads,
            )
        )

        assert result.status == "ready"
        assert result.result_path == str(payload["path"])
        assert result.result_checksum == payload["checksum"]
        assert result.result_size_bytes == payload["size_bytes"]
        assert presented == {"ok": True, "value": 7}
    finally:
        store.close()


def test_scheduler_gateway_retrieve_failed_returns_error_payload(tmp_path) -> None:
    store = TaskStateStore.in_memory()
    try:
        store.create_task(
            request_id="req-failed",
            op="sampling.asample",
            domain_key="vllm:Qwen/Test",
            request_json=b"{}",
            metadata={"op": "sampling.asample"},
        )
        store.complete_task_failure(request_id="req-failed", error="backend exploded")
        gateway = SchedulerModelWorkTaskGateway(task_ledger_client=_LocalTaskLedger(store))

        result = asyncio.run(gateway.retrieve_task(request_id="req-failed"))
        presented = asyncio.run(
            present_terminal_retrieve_result(
                result,
                error_presenter=lambda error: {"error": error, "category": "system"},
                payload_store=TaskPayloadStore(root_dir=tmp_path / "payloads"),
            )
        )

        assert result.status == "failed"
        assert result.error == {"message": "backend exploded"}
        assert presented == {"error": "backend exploded", "category": "system"}
    finally:
        store.close()


def test_scheduler_gateway_retrieve_pending_and_unknown() -> None:
    store = TaskStateStore.in_memory()
    try:
        store.create_task(
            request_id="req-pending",
            op="sampling.asample",
            domain_key="vllm:Qwen/Test",
            request_json=b"{}",
            metadata={"op": "sampling.asample"},
        )
        gateway = SchedulerModelWorkTaskGateway(task_ledger_client=_LocalTaskLedger(store))

        pending = asyncio.run(gateway.retrieve_task(request_id="req-pending"))
        unknown = asyncio.run(gateway.retrieve_task(request_id="req-missing"))

        assert pending.status == "pending"
        assert pending.extra["record"]["status"] == "pending"
        assert unknown.status == "unknown"
    finally:
        store.close()


class _LocalTaskLedger:
    def __init__(self, store: TaskStateStore) -> None:
        self.store = store

    async def get_task(self, *, request_id: str):
        record = self.store.get_task(request_id)
        return TaskRecord(request_id=str(record["request_id"]), status=str(record["status"]), data=record)

    async def wait_task_status_change(self, *, request_id: str, timeout_s: float, **_kwargs):
        _ = timeout_s
        try:
            record = self.store.get_task(request_id)
        except KeyError:
            return TaskStatusChange(changed=False, request_id=request_id, missing=True)
        return TaskStatusChange(
            changed=False,
            request_id=request_id,
            timeout=True,
            missing=False,
            record=record,
        )

    async def ensure_ready(self, **_kwargs):
        return {"ok": True}

    async def ping(self, **_kwargs):
        return {"ok": True}

    async def acquire_owner(self, **_kwargs):
        raise NotImplementedError

    async def renew_owner(self, **_kwargs):
        raise NotImplementedError

    async def create_task(self, **_kwargs):
        raise NotImplementedError

    async def assign_task(self, **_kwargs):
        raise NotImplementedError

    async def claim_task(self, **_kwargs):
        raise NotImplementedError

    async def renew_lease(self, **_kwargs):
        raise NotImplementedError

    async def begin_finalize(self, **_kwargs):
        raise NotImplementedError

    async def commit_finalize_success(self, **_kwargs):
        raise NotImplementedError

    async def commit_finalize_failure(self, **_kwargs):
        raise NotImplementedError

    async def complete_task_failure(self, **_kwargs):
        raise NotImplementedError

    async def requeue_task(self, **_kwargs):
        raise NotImplementedError

    async def forget_task(self, **_kwargs):
        raise NotImplementedError

    async def list_active_tasks(self, **_kwargs):
        raise NotImplementedError

    async def update_task_metadata(self, **_kwargs):
        raise NotImplementedError
