from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

TraceEnqueue = Callable[..., Awaitable[Any]]


@dataclass(frozen=True)
class ModelWorkAdmissionResult:
    request_id: str
    scheduler_result: dict[str, Any]


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
    create_future: bool = True,
    payload_hash: str | None = None,
    task_futures_client: Any | None = None,
    scheduler_client: Any | None = None,
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

    _ = task_futures_client
    scheduler_confirmed = False
    if scheduler_client is None:
        from .model_work_scheduler import model_work_scheduler as scheduler
    else:
        scheduler = scheduler_client
    scheduler_extra = {
        **enqueue_extra,
        **dict(queued_meta),
        "model_work_scheduler": True,
        "domain_key": str(domain_key),
        "request_json_bytes": len(request_json),
    }
    if payload_hash is not None:
        scheduler_extra["payload_hash"] = str(payload_hash)
    try:
        if create_future and task_futures_client is not None:
            create_model_work = getattr(task_futures_client, "async_create_model_work_with_id", None)
            if callable(create_model_work):
                await create_model_work(
                    request_id,
                    op=op,
                    domain_key=domain_key,
                    request_json=request_json,
                    meta=scheduler_extra,
                    payload_hash=payload_hash,
                )
            else:
                create_with_id = getattr(task_futures_client, "async_create_with_id", None)
                if callable(create_with_id):
                    await create_with_id(request_id)
        append_coro = scheduler.append(
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
        )
        if trace_enqueue is None:
            out = await append_coro
        else:
            out = await trace_enqueue(
                **dict(trace_kwargs or {}),
                request_id=request_id,
                op=op,
                enqueue_coro=append_coro,
            )
        scheduler_confirmed = isinstance(out, dict) and bool(out.get("ok"))
        _ = create_future
        return ModelWorkAdmissionResult(
            request_id=request_id,
            scheduler_result=out if isinstance(out, dict) else {},
        )
    except Exception:
        if scheduler_confirmed:
            try:
                await scheduler.cancel_request(
                    request_id=request_id,
                    reason="admission_failed",
                )
            except Exception:
                pass
        raise
