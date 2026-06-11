from __future__ import annotations

import asyncio

from mint_server.backend.control_plane_contracts import (
    AppendWorkResult,
    CancelTaskResult,
)
from mint_server.backend.model_work_task_gateway import SchedulerModelWorkTaskGateway


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
    gateway = SchedulerModelWorkTaskGateway(scheduler_client=scheduler)

    result = asyncio.run(gateway.submit_task(**_submit_kwargs()))

    assert result.ok is True
    assert result.request_id == "req-gateway"
    assert result.created is True
    assert result.assigned is True
    assert result.extra["sampling_inflight_admission"] == {"ok": True}
    assert result.extra["scheduler_result"]["ok"] is True
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
    gateway = SchedulerModelWorkTaskGateway(scheduler_client=scheduler)

    result = asyncio.run(gateway.submit_task(**_submit_kwargs()))

    assert result.ok is True
    assert result.created is False
    assert result.assigned is False
    assert result.extra["scheduler_result"]["idempotent"] is True


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
    gateway = SchedulerModelWorkTaskGateway(scheduler_client=scheduler)

    result = asyncio.run(gateway.submit_task(**_submit_kwargs()))

    assert result.ok is False
    assert result.created is False
    assert result.assigned is False
    assert str(result.reason) == "principal_domain_inflight_limit_exceeded"
    assert result.extra["sampling_inflight_admission"]["ok"] is False


def test_scheduler_gateway_submit_falls_back_to_append() -> None:
    scheduler = _AppendOnlyScheduler(
        AppendWorkResult(
            ok=True,
            request_id="req-fallback",
            assigned={"ok": True, "assigned": 0, "skipped_domains": []},
            idempotent=True,
        )
    )
    gateway = SchedulerModelWorkTaskGateway(scheduler_client=scheduler)

    result = asyncio.run(gateway.submit_task(**_submit_kwargs("req-fallback")))

    assert result.ok is True
    assert result.request_id == "req-fallback"
    assert result.created is False
    assert len(scheduler.calls) == 1


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
