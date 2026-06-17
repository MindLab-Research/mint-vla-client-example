from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import structlog
import time

from mint_server.logging_context import get_current_traceparent, record_span_event_otel, start_as_current_span

from mint_server.backend.contracts.control_plane_contracts import AppendWorkResult, SubmitTaskResult
from mint_server.backend.scheduling.model_work_task_gateway import SchedulerModelWorkTaskGateway

TraceEnqueue = Callable[..., Awaitable[Any]]

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ModelWorkAdmissionResult:
    request_id: str
    scheduler_result: AppendWorkResult
    submit_result: SubmitTaskResult | None = None


class ModelWorkAdmissionRejectedError(RuntimeError):
    def __init__(self, scheduler_result: AppendWorkResult | dict[str, Any]) -> None:
        if isinstance(scheduler_result, dict):
            scheduler_result = AppendWorkResult.from_wire(scheduler_result)
        self.scheduler_result = scheduler_result
        reason = str(scheduler_result.reason or "admission_rejected")
        self.reason = reason
        super().__init__(reason)


async def enqueue_model_work(
    *,
    request_id: str,
    op: str,
    request_json: bytes,
    domain_key: str,
    queued_meta: dict[str, Any],
    extra: dict[str, Any] | None = None,
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
    scheduler_client: Any | None = None,
    gateway_client: Any | None = None,
    future_service_client: Any | None = None,
    trace_enqueue: TraceEnqueue | None = None,
    trace_kwargs: dict[str, Any] | None = None,
) -> ModelWorkAdmissionResult:
    enqueue_extra = {
        **dict(extra or {}),
        "model_work_scheduler": True,
        "domain_key": str(domain_key),
    }
    if affinity_group is not None:
        enqueue_extra["affinity_group"] = str(affinity_group)
    if ordering_key is not None:
        enqueue_extra["ordering_key"] = str(ordering_key)

    scheduler_extra = {
        **enqueue_extra,
        **dict(queued_meta),
        "model_work_scheduler": True,
        "domain_key": str(domain_key),
        "request_json_bytes": len(request_json),
    }
    if payload_hash is not None:
        scheduler_extra["payload_hash"] = str(payload_hash)
    _admission_t0 = time.perf_counter()
    # Inject current traceparent so the full dispatch -> actor chain can
    # restore the original request trace.  model_engine_host reads
    # extra["_traceparent"] in _restore_item_context and _execute_lease.
    _tp = get_current_traceparent()
    if _tp:
        scheduler_extra["_traceparent"] = _tp
    if gateway_client is None:
        gateway = SchedulerModelWorkTaskGateway(
            scheduler_client=scheduler_client,
            future_service_client=future_service_client,
        )
    else:
        gateway = gateway_client
    try:
        submit_coro = gateway.submit_task(
            request_id=request_id,
            op=op,
            request_json=request_json,
            metadata=scheduler_extra,
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
            payload_hash=payload_hash,
        )
        if trace_enqueue is None:
            submit = await submit_coro
        else:
            submit = await trace_enqueue(
                **dict(trace_kwargs or {}),
                request_id=request_id,
                op=op,
                enqueue_coro=submit_coro,
            )
        if isinstance(submit, dict):
            submit = SubmitTaskResult.from_wire(submit)
        if not isinstance(submit, SubmitTaskResult):
            raise TypeError(f"model work gateway returned non-SubmitTaskResult: {type(submit)}")
        scheduler_result = submit.extra.get("scheduler_result")
        if isinstance(scheduler_result, dict):
            scheduler_result = AppendWorkResult.from_wire(scheduler_result)
        if not isinstance(scheduler_result, AppendWorkResult):
            scheduler_result = AppendWorkResult(
                ok=submit.ok,
                request_id=submit.request_id,
                assigned={"ok": True, "assigned": int(submit.assigned), "skipped_domains": []},
                idempotent=not submit.created,
                reason=submit.reason,
                extra=dict(submit.extra),
            )
        if not scheduler_result.ok:
            reason = str(scheduler_result.reason or "")
            if reason in {
                "principal_domain_inflight_limit_exceeded",
                "domain_inflight_limit_exceeded",
                "principal_domain_token_budget_exceeded",
                "domain_token_budget_exceeded",
            }:
                raise ModelWorkAdmissionRejectedError(scheduler_result)
        admission_t1 = time.perf_counter()
        _assigned = bool((scheduler_result.assigned or {}).get("assigned"))
        logger.info(
            "request_id=%s op=%s domain_key=%s status=admitted assigned=%s idempotent=%s backlog_depth=%s admission_ms=%.1f",
            request_id,
            op,
            domain_key,
            _assigned,
            bool(scheduler_result.idempotent),
            int(scheduler_result.backlog_depth or 0),
            (admission_t1 - _admission_t0) * 1000.0,
        )
        with start_as_current_span(
            "model_work.admission",
            component="scheduling.admission",
            op="admit",
            request_id=str(request_id),
            attributes={
                "admission_op": str(op),
                "domain_key": str(domain_key),
                "assigned": _assigned,
                "idempotent": bool(scheduler_result.idempotent),
                "backlog_depth": int(scheduler_result.backlog_depth or 0),
            },
        ):
            record_span_event_otel("admission.complete", attributes={
                "request_id": str(request_id),
                "op": str(op),
                "status": "admitted",
            })
        return ModelWorkAdmissionResult(
            request_id=request_id,
            scheduler_result=scheduler_result,
            submit_result=submit,
        )
    except ModelWorkAdmissionRejectedError:
        _admission_t1 = time.perf_counter()
        logger.warning(
            "request_id=%s op=%s domain_key=%s status=rejected reason=%s admission_ms=%.1f",
            request_id,
            op,
            domain_key,
            "admission_limit_exceeded",
            (_admission_t1 - _admission_t0) * 1000.0,
        )
        raise
    except Exception:
        _admission_t1 = time.perf_counter()
        logger.warning(
            "request_id=%s op=%s domain_key=%s status=error admission_ms=%.1f",
            request_id,
            op,
            domain_key,
            (_admission_t1 - _admission_t0) * 1000.0,
            exc_info=True,
        )
        raise
