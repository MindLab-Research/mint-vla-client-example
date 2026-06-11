from __future__ import annotations

from typing import Any

from .control_plane_contracts import (
    AppendWorkResult,
    CancelTaskResult,
    RetrieveTaskResult,
    SubmitTaskResult,
)


class SchedulerModelWorkTaskGateway:
    """Typed API-side task lifecycle gateway backed by the scheduler control plane."""

    def __init__(self, *, scheduler_client: Any | None = None) -> None:
        self._scheduler_client = scheduler_client

    @property
    def scheduler(self) -> Any:
        if self._scheduler_client is not None:
            return self._scheduler_client
        from .model_work_scheduler import model_work_scheduler

        return model_work_scheduler

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
        raise NotImplementedError("retrieve_task will be implemented in the retrieve migration slice")


model_work_task_gateway = SchedulerModelWorkTaskGateway()
