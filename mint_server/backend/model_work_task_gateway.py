from __future__ import annotations

from typing import Any

from .control_plane_contracts import (
    AppendWorkResult,
    CancelTaskResult,
    RetrieveTaskResult,
    SubmitTaskResult,
    TaskRecord,
    TaskStatusChange,
    as_task_ledger,
)
from .task_state_store import TERMINAL_TASK_STATUSES, TaskStateNotFoundError, TaskStateStoreUnavailableError


class SchedulerModelWorkTaskGateway:
    """Typed API-side task lifecycle gateway backed by the scheduler control plane."""

    def __init__(self, *, scheduler_client: Any | None = None, task_ledger_client: Any | None = None) -> None:
        self._scheduler_client = scheduler_client
        self._task_ledger_client = task_ledger_client
        self._task_ledger = as_task_ledger(task_ledger_client) if task_ledger_client is not None else None

    @property
    def scheduler(self) -> Any:
        if self._scheduler_client is not None:
            return self._scheduler_client
        from .model_work_scheduler import model_work_scheduler

        return model_work_scheduler

    @property
    def task_ledger(self) -> Any:
        if self._task_ledger is not None:
            return self._task_ledger
        from .task_state_store import task_state_store

        return as_task_ledger(task_state_store)

    async def submit_task(
        self,
        *,
        request_id: str,
        op: str,
        domain_key: str,
        request_json: bytes,
        metadata: dict[str, Any],
        user_id: str | None = None,
        apikey_id: str | None = None,
        throttle_principal: str | None = None,
        webhook_url: str | None = None,
        affinity_group: str | None = None,
        ordering_key: str | None = None,
        token_cost: int = 1,
        assign: bool = True,
        assign_max_items: int | None = 1,
        payload_hash: str | None = None,
        timeout_s: float | None = None,
    ) -> SubmitTaskResult:
        scheduler_extra = dict(metadata)
        if payload_hash is not None:
            scheduler_extra["payload_hash"] = str(payload_hash)
        append = getattr(self.scheduler, "append_work", None)
        if not callable(append):
            append = getattr(self.scheduler, "append", None)
        if not callable(append):
            raise TypeError("scheduler client does not implement append_work or append")
        out = await append(
            request_id=request_id,
            op=op,
            request_json=request_json,
            user_id=user_id,
            apikey_id=apikey_id,
            throttle_principal=throttle_principal,
            webhook_url=webhook_url,
            domain_key=domain_key,
            affinity_group=affinity_group,
            ordering_key=ordering_key,
            token_cost=token_cost,
            assign=assign,
            assign_max_items=assign_max_items,
            extra=scheduler_extra,
            timeout_s=timeout_s,
        )
        if isinstance(out, dict):
            out = AppendWorkResult.from_wire(out)
        if not isinstance(out, AppendWorkResult):
            raise TypeError(f"scheduler.append_work returned non-AppendWorkResult: {type(out)}")
        assigned = bool((out.assigned or {}).get("assigned"))
        return SubmitTaskResult(
            ok=out.ok,
            request_id=str(out.request_id or request_id),
            created=bool(out.ok) and not bool(out.idempotent),
            assigned=assigned,
            reason=out.reason,
            record=None,
            extra={
                **dict(out.extra),
                "scheduler_result": out.to_wire(),
            },
        )

    async def cancel_task(
        self,
        *,
        request_id: str,
        reason: str,
        timeout_s: float | None = None,
    ) -> CancelTaskResult:
        out = await self.scheduler.cancel_request(
            request_id=request_id,
            reason=reason,
            timeout_s=timeout_s,
        )
        if isinstance(out, dict):
            return CancelTaskResult.from_wire(out)
        if not isinstance(out, CancelTaskResult):
            raise TypeError(f"scheduler.cancel_request returned non-CancelTaskResult: {type(out)}")
        return out

    async def retrieve_task(
        self,
        *,
        request_id: str,
        wait_timeout_s: float = 0.0,
        privileged: bool = False,
        timeout_s: float | None = None,
    ) -> RetrieveTaskResult:
        _ = privileged, timeout_s
        try:
            task_record = await self.task_ledger.get_task(request_id=str(request_id))
        except (KeyError, TaskStateNotFoundError):
            return RetrieveTaskResult(status="unknown", request_id=str(request_id))
        except TaskStateStoreUnavailableError as exc:
            return RetrieveTaskResult(
                status="unavailable",
                request_id=str(request_id),
                error={"message": str(exc), "type": type(exc).__name__},
            )
        if not isinstance(task_record, TaskRecord):
            raise TypeError(f"task ledger returned non-TaskRecord: {type(task_record)}")
        record = dict(task_record.data)
        if (
            str(record.get("status") or "") not in TERMINAL_TASK_STATUSES
            and float(wait_timeout_s) > 0.0
        ):
            try:
                changed = await self.task_ledger.wait_task_status_change(
                    request_id=str(request_id),
                    timeout_s=float(wait_timeout_s),
                    observed_status=str(record.get("status") or ""),
                    observed_updated_at=(
                        float(record["updated_at"]) if record.get("updated_at") is not None else None
                    ),
                    terminal_only=True,
                )
            except (KeyError, TaskStateNotFoundError):
                return RetrieveTaskResult(status="unknown", request_id=str(request_id))
            except TaskStateStoreUnavailableError as exc:
                return RetrieveTaskResult(
                    status="unavailable",
                    request_id=str(request_id),
                    error={"message": str(exc), "type": type(exc).__name__},
                )
            if not isinstance(changed, TaskStatusChange):
                raise TypeError(f"task ledger returned non-TaskStatusChange: {type(changed)}")
            if changed.record is not None:
                record = dict(changed.record)
        return self._retrieve_result_from_record(str(request_id), record)

    def _retrieve_result_from_record(self, request_id: str, record: dict[str, Any]) -> RetrieveTaskResult:
        status = str(record.get("status") or "")
        extra = {"record": dict(record)}
        if status in {"done", "retrieved"}:
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            terminal_status = str(metadata.get("terminal_status") or status)
            if terminal_status == "failed" or record.get("error") is not None:
                return RetrieveTaskResult(
                    status="failed",
                    request_id=request_id,
                    error={"message": str(record.get("error") or "Task failed")},
                    extra=extra,
                )
            return RetrieveTaskResult(
                status="ready",
                request_id=request_id,
                result_path=record.get("result_path") if isinstance(record.get("result_path"), str) else None,
                result_checksum=(
                    record.get("result_checksum") if isinstance(record.get("result_checksum"), str) else None
                ),
                result_size_bytes=(
                    int(record["result_size_bytes"])
                    if record.get("result_size_bytes") is not None
                    else None
                ),
                extra=extra,
            )
        if status == "failed":
            return RetrieveTaskResult(
                status="failed",
                request_id=request_id,
                error={"message": str(record.get("error") or "Task failed")},
                extra=extra,
            )
        return RetrieveTaskResult(
            status="pending",
            request_id=request_id,
            retry_after_s=1.0,
            extra=extra,
        )


model_work_task_gateway = SchedulerModelWorkTaskGateway()
