from __future__ import annotations

import time
from typing import Any, Callable

from mint_server.backend.contracts.control_plane_contracts import RetrieveTaskResult
from mint_server.backend.stores.task_payload_store import TaskPayloadStore

ErrorPresenter = Callable[[str | None], dict[str, Any]]


def terminal_evicted_payload(record: dict[str, Any]) -> dict[str, Any]:
    meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    done_at = meta.get("done_at") or meta.get("failed_at") or record.get("updated_at")
    retrieved_at = record.get("updated_at") if str(record.get("status") or "") == "retrieved" else None
    try:
        done_at_s = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(done_at)))
    except Exception:
        done_at_s = None
    try:
        retrieved_at_s = None if retrieved_at is None else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(retrieved_at)))
    except Exception:
        retrieved_at_s = None
    return {
        "error": "Known terminal future evicted",
        "category": "system",
        "request_id": str(record.get("request_id") or ""),
        "op": str(meta.get("op") or record.get("op") or ""),
        "done_at": done_at_s,
        "retrieved_at": retrieved_at_s,
    }


async def present_terminal_retrieve_result(
    result: RetrieveTaskResult,
    *,
    error_presenter: ErrorPresenter,
    payload_store: TaskPayloadStore | None = None,
) -> Any:
    record = result.extra.get("record")
    record = dict(record) if isinstance(record, dict) else {"request_id": result.request_id}
    if result.status == "failed":
        error = result.error.get("message") if isinstance(result.error, dict) else None
        return error_presenter(None if error is None else str(error))
    if result.status != "ready":
        raise ValueError(f"cannot present non-terminal retrieve result: {result.status}")
    if result.result_path:
        store = payload_store or TaskPayloadStore()
        try:
            return await store.async_read_json_payload(
                path=result.result_path,
                expected_checksum=result.result_checksum,
            )
        except Exception:
            return terminal_evicted_payload(record)
    return terminal_evicted_payload(record)
