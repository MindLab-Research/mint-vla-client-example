from __future__ import annotations

import base64
import json
import os
import asyncio
import threading
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

from mint_server.config import PFS_PYTHONPATH, actor_runtime_env, apply_detached_actor_resources, config as server_config, otel_env_vars
from mint_server.runtime_env import env_nonempty
from mint_server.backend.ray_cluster.async_ray_control import async_get_ray_ref, sync_get_ray_ref
from mint_server.backend.contracts.control_plane_contracts import (
    BeginFinalizeResult,
    ClaimTaskResult,
    CommitFinalizeResult,
    ConflictReason,
    CreateTaskResult,
    OwnerLeaseResult,
    TaskMutationResult,
    WireCompatibleResult,
)
from mint_server.backend.core.model_work_execution_context import ModelWorkFinalize, get_current_model_work_finalize_buffer
from mint_server.backend.stores.billing_outbox import BillingOutbox
from mint_server.backend.stores.kv_backend import InMemoryKVBackend, KVBackend, RocksKVBackend
from mint_server.backend.stores.task_hot_kv_store import TaskHotKVStore


ACTIVE_TASK_STATUSES = frozenset({"pending", "queued", "running", "assigned", "leased", "finalizing"})
TERMINAL_TASK_STATUSES = frozenset({"done", "failed", "cancelled", "expired", "retrieved"})
_REAPER_METRICS: dict[str, float] = {
    "expire_pending": 0.0,
    "evict_payload": 0.0,
    "gc_staged_payload": 0.0,
    "delete_tombstone": 0.0,
}
_REAPER_PAYLOAD_EVICT_ERRORS_TOTAL = 0.0
_FUTURE_TIMEOUT_METRICS: dict[str, Any] = {
    "queue": 0.0,
    "execution": 0.0,
    "total": 0.0,
    "by_op": {},
}
_BILLING_EVENT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "mindlab.mint.billing.usage_event.v1")
BILLING_OUTBOX_STATUSES = frozenset({"pending", "flushing", "failed"})
_BILLING_FLUSH_METRICS: dict[str, float] = {
    "flush_success": 0.0,
    "flush_transient_error": 0.0,
    "flush_permanent_error": 0.0,
    "event_inserted": 0.0,
    "event_conflict": 0.0,
    "event_failed": 0.0,
    "write_error": 0.0,
    "outbox_conflict": 0.0,
    "skipped_missing_billing_context": 0.0,
}
_TASK_STATE_RPC_METRICS: dict[str, Any] = {
    "total": 0.0,
    "error": 0.0,
    "inflight": 0.0,
    "by_method": {},
}
_TASK_STATE_RPC_METRICS_LOCK = threading.Lock()
_TASK_STATE_STATS_METRICS: dict[str, float] = {
    "calls": 0.0,
    "cache_hits": 0.0,
    "total_duration_ms": 0.0,
    "last_duration_ms": 0.0,
    "max_duration_ms": 0.0,
}
_TASK_STATE_STATS_METRICS_LOCK = threading.Lock()


class TaskStateStoreError(RuntimeError):
    pass


class TaskStateConflictError(TaskStateStoreError):
    pass


class TaskStateNotFoundError(TaskStateStoreError, KeyError):
    pass


class TaskStateStoreUnavailableError(TaskStateStoreError):
    pass


def _wire_result(value: Any) -> Any:
    if isinstance(value, WireCompatibleResult):
        return value.to_wire()
    return value


def _task_state_cause_from_ray_error(exc: BaseException) -> TaskStateStoreError | None:
    as_instanceof_cause = getattr(exc, "as_instanceof_cause", None)
    if callable(as_instanceof_cause):
        try:
            cause = as_instanceof_cause()
        except Exception:
            cause = None
        if isinstance(cause, TaskStateStoreError):
            return cause
    cause = getattr(exc, "cause", None)
    if isinstance(cause, TaskStateStoreError):
        return cause
    return None


class FutureStatus(Enum):
    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"
    EXPIRED = "expired"
    RETRIEVED = "retrieved"


class BillingObservation(dict):
    """Dictionary-shaped business fact used to derive durable usage events."""


def _billing_event_id(*, account_id: str, apikey_id: str, request_id: str, charge_item: str, label: str) -> str:
    key = "|".join(
        [
            str(account_id).strip(),
            str(apikey_id).strip(),
            str(request_id).strip(),
            str(charge_item).strip(),
            str(label or "").strip(),
        ]
    )
    return uuid.uuid5(_BILLING_EVENT_NAMESPACE, key).hex


def _billing_label(*, model: str | None, route: str, dimension: str, unit: str) -> str:
    parts = []
    if model:
        parts.append(f"model={model}")
    parts.extend([f"route={route}", f"dimension={dimension}", f"unit={unit}"])
    return ",".join(parts)


def build_billing_observation(
    *,
    account_id: str,
    apikey_id: str,
    request_id: str,
    charge_item: str,
    quantity: int,
    unit: str,
    route: str,
    dimension: str,
    model: str | None = None,
    metadata: dict[str, Any] | None = None,
    observed_at: float | None = None,
) -> BillingObservation:
    return BillingObservation(
        account_id=str(account_id),
        apikey_id=str(apikey_id),
        request_id=str(request_id),
        charge_item=str(charge_item),
        quantity=int(quantity),
        unit=str(unit),
        model=None if model is None else str(model),
        route=str(route),
        dimension=str(dimension),
        metadata=dict(metadata or {}),
        observed_at=_now(observed_at),
    )


def billing_event_from_observation(observation: dict[str, Any]) -> dict[str, Any]:
    account_id = str(observation.get("account_id") or "").strip()
    apikey_id = str(observation.get("apikey_id") or "").strip()
    request_id = str(observation.get("request_id") or "").strip()
    charge_item = str(observation.get("charge_item") or "").strip()
    unit = str(observation.get("unit") or "").strip()
    route = str(observation.get("route") or "").strip()
    dimension = str(observation.get("dimension") or "").strip()
    model_raw = observation.get("model")
    model = None if model_raw is None else str(model_raw).strip()
    quantity = int(observation.get("quantity") or 0)
    if not account_id or not apikey_id or not request_id or not charge_item:
        raise ValueError("billing observation requires account_id, apikey_id, request_id, and charge_item")
    if not unit or not route or not dimension:
        raise ValueError("billing observation requires unit, route, and dimension")
    if quantity < 0:
        raise ValueError("billing observation quantity must be non-negative")
    metadata = observation.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    observed_at = float(observation.get("observed_at") or time.time())
    label = _billing_label(model=model, route=route, dimension=dimension, unit=unit)
    event_id = str(observation.get("event_id") or "").strip() or _billing_event_id(
        account_id=account_id,
        apikey_id=apikey_id,
        request_id=request_id,
        charge_item=charge_item,
        label=label,
    )
    return {
        "event_id": event_id,
        "event_time": observed_at,
        "account_id": account_id,
        "apikey_id": apikey_id,
        "charge_item": charge_item,
        "quantity": quantity,
        "request_id": request_id,
        "label": label,
        "metadata": dict(metadata),
    }


def billing_observations_from_auth(
    auth_ctx: Any | None,
    *,
    request_id: str,
    charge_item: str,
    quantity: int,
    unit: str,
    route: str,
    dimension: str,
    model: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if auth_ctx is None:
        _inc_billing_metric("skipped_missing_billing_context", 1)
        return []
    account_id = str(getattr(auth_ctx, "account_id", "") or "")
    apikey_id = str(getattr(auth_ctx, "apikey_id", "") or "")
    billing_request_id = str(getattr(auth_ctx, "request_id", "") or request_id)
    if not account_id or not apikey_id:
        _inc_billing_metric("skipped_missing_billing_context", 1)
        return []
    return [
        build_billing_observation(
            account_id=account_id,
            apikey_id=apikey_id,
            request_id=billing_request_id,
            charge_item=charge_item,
            quantity=quantity,
            unit=unit,
            route=route,
            dimension=dimension,
            model=model,
            metadata=metadata,
        )
    ]


def billing_observations_from_input(
    *,
    gateway_auth: dict | None,
    request_id: str,
    billing_input: dict | None,
) -> list[dict[str, Any]]:
    if billing_input is None:
        return []
    if not isinstance(billing_input, dict):
        raise ValueError("billing input must be a dict")
    from mint_server.gateway_auth import GatewayAuthContext

    required = ("charge_item", "quantity", "unit", "route", "dimension")
    missing = [key for key in required if key not in billing_input]
    if missing:
        raise ValueError(f"billing input missing required keys: {', '.join(missing)}")
    metadata = billing_input.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("billing input metadata must be a dict")
    auth_ctx = GatewayAuthContext(**gateway_auth) if gateway_auth else None
    return billing_observations_from_auth(
        auth_ctx=auth_ctx,
        request_id=request_id,
        charge_item=str(billing_input["charge_item"]),
        quantity=int(billing_input["quantity"]),
        unit=str(billing_input["unit"]),
        route=str(billing_input["route"]),
        dimension=str(billing_input["dimension"]),
        model=(
            None
            if billing_input.get("model") in (None, "")
            else str(billing_input.get("model"))
        ),
        metadata=metadata,
    )


def _now(now: float | None = None) -> float:
    return time.time() if now is None else float(now)


def _otel_metric_attrs() -> dict[str, str]:
    attrs = {
        "deployment.env": os.getenv("MINT_DEPLOYMENT_ENV", "").strip(),
        "mint.cluster_id": os.getenv("MINT_CLUSTER_ID", "").strip(),
        "ray_namespace": _ray_namespace(),
    }
    return {key: value for key, value in attrs.items() if value}


def _metric_number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _json_dumps(value: dict[str, Any] | None) -> str:
    return json.dumps({} if value is None else dict(value), ensure_ascii=True, sort_keys=True)


def _json_loads(value: str | bytes | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not value:
        return {}
    out = json.loads(value)
    if not isinstance(out, dict):
        raise TaskStateStoreError(f"expected JSON object, got {type(out).__name__}")
    return out


def _status_from_task_record(record: dict[str, Any]) -> FutureStatus:
    status = str(record.get("status") or "")
    if status in {"pending", "queued", "assigned", "leased", "running", "finalizing"}:
        return FutureStatus.PENDING
    if status == "done":
        return FutureStatus.DONE
    if status in {"failed", "cancelled"}:
        return FutureStatus.FAILED
    if status == "expired":
        return FutureStatus.EXPIRED
    if status == "retrieved":
        return FutureStatus.RETRIEVED
    raise KeyError(f"Unknown task status for request_id={record.get('request_id')!r}: {status!r}")


def _is_training_step_op(op: Any) -> bool:
    return str(op or "") in {"training.optim_step", "training.train_step"}


def _inc_reaper_rows(action: str, count: int) -> None:
    if int(count) <= 0:
        return
    key = str(action)
    _REAPER_METRICS[key] = float(_REAPER_METRICS.get(key, 0.0)) + float(count)
    try:
        from mint_server.logging_context import record_task_future_reaper_rows_metric

        record_task_future_reaper_rows_metric(action=key, count=int(count))
    except Exception:
        pass


def _inc_payload_evict_errors(count: int = 1) -> None:
    global _REAPER_PAYLOAD_EVICT_ERRORS_TOTAL
    if int(count) <= 0:
        return
    _REAPER_PAYLOAD_EVICT_ERRORS_TOTAL += float(count)
    try:
        from mint_server.logging_context import record_task_future_payload_evict_error_metric

        record_task_future_payload_evict_error_metric(count=int(count))
    except Exception:
        pass


def _inc_future_timeout(kind: str, *, op: str | None = None, count: int = 1) -> None:
    if int(count) <= 0:
        return
    normalized = str(kind or "execution")
    if normalized not in {"queue", "execution"}:
        normalized = "execution"
    amount = float(count)
    _FUTURE_TIMEOUT_METRICS[normalized] = float(_FUTURE_TIMEOUT_METRICS.get(normalized) or 0.0) + amount
    _FUTURE_TIMEOUT_METRICS["total"] = float(_FUTURE_TIMEOUT_METRICS.get("total") or 0.0) + amount
    if isinstance(op, str) and op.strip():
        by_op = _FUTURE_TIMEOUT_METRICS.setdefault("by_op", {})
        if isinstance(by_op, dict):
            op_key = op.strip()
            rec = by_op.setdefault(op_key, {"queue": 0.0, "execution": 0.0, "total": 0.0})
            if isinstance(rec, dict):
                rec[normalized] = float(rec.get(normalized) or 0.0) + amount
                rec["total"] = float(rec.get("total") or 0.0) + amount
    try:
        from mint_server.logging_context import record_task_futures_timeout_metric

        record_task_futures_timeout_metric(kind=normalized, op=op)
        record_task_futures_timeout_metric(kind="total", op=op)
    except Exception:
        pass


def _inc_billing_metric(key: str, count: int = 1) -> None:
    if int(count) <= 0:
        return
    name = str(key)
    _BILLING_FLUSH_METRICS[name] = float(_BILLING_FLUSH_METRICS.get(name, 0.0)) + float(count)


def _inc_billing_metrics(metrics: dict[str, Any]) -> None:
    for key, value in dict(metrics or {}).items():
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        _inc_billing_metric(str(key), count)


def billing_metrics_snapshot() -> dict[str, Any]:
    return dict(_BILLING_FLUSH_METRICS)


def _bounded_task_state_method(method: str | None) -> str:
    value = str(method or "unknown").strip() or "unknown"
    allowed = {
        "future_get_task",
        "future_wait_task_status_change",
        "future_ensure_task",
        "future_update_task_metadata",
        "future_complete_task_success",
        "future_complete_task_failure",
        "future_mark_task_retrieved",
        "future_expire_active_tasks",
        "future_list_terminal_payloads_for_eviction",
        "future_mark_payload_evicted",
        "future_list_staged_payloads_for_gc",
        "future_mark_staged_payload_gc_deleted",
        "future_delete_expired_tombstones",
        "stats",
        "future_stats",
        "ping",
        "future_ping",
    }
    if value in allowed:
        return value
    if value.startswith("future_"):
        return "future_other"
    return "other"


def _record_task_state_rpc_metric(method: str, *, duration_ms: float, ok: bool) -> None:
    bounded = _bounded_task_state_method(method)
    with _TASK_STATE_RPC_METRICS_LOCK:
        _TASK_STATE_RPC_METRICS["total"] = float(_TASK_STATE_RPC_METRICS.get("total", 0.0)) + 1.0
        if not ok:
            _TASK_STATE_RPC_METRICS["error"] = float(_TASK_STATE_RPC_METRICS.get("error", 0.0)) + 1.0
        by_method = _TASK_STATE_RPC_METRICS.setdefault("by_method", {})
        rec = by_method.setdefault(
            bounded,
            {
                "total": 0.0,
                "error": 0.0,
                "total_duration_ms": 0.0,
                "last_duration_ms": 0.0,
                "max_duration_ms": 0.0,
            },
        )
        rec["total"] = float(rec.get("total", 0.0)) + 1.0
        if not ok:
            rec["error"] = float(rec.get("error", 0.0)) + 1.0
        duration = max(0.0, float(duration_ms))
        rec["total_duration_ms"] = float(rec.get("total_duration_ms", 0.0)) + duration
        rec["last_duration_ms"] = duration
        rec["max_duration_ms"] = max(float(rec.get("max_duration_ms", 0.0)), duration)


def _inc_task_state_rpc_inflight(delta: float) -> None:
    with _TASK_STATE_RPC_METRICS_LOCK:
        current = float(_TASK_STATE_RPC_METRICS.get("inflight", 0.0))
        _TASK_STATE_RPC_METRICS["inflight"] = max(0.0, current + float(delta))


def task_state_rpc_metrics_snapshot() -> dict[str, Any]:
    with _TASK_STATE_RPC_METRICS_LOCK:
        return {
            "total": float(_TASK_STATE_RPC_METRICS.get("total", 0.0)),
            "error": float(_TASK_STATE_RPC_METRICS.get("error", 0.0)),
            "inflight": float(_TASK_STATE_RPC_METRICS.get("inflight", 0.0)),
            "by_method": {
                str(method): dict(rec)
                for method, rec in dict(_TASK_STATE_RPC_METRICS.get("by_method") or {}).items()
                if isinstance(rec, dict)
            },
        }


def _record_task_state_stats_metric(*, duration_ms: float, cache_hit: bool) -> None:
    duration = max(0.0, float(duration_ms))
    with _TASK_STATE_STATS_METRICS_LOCK:
        _TASK_STATE_STATS_METRICS["calls"] = float(_TASK_STATE_STATS_METRICS.get("calls", 0.0)) + 1.0
        if cache_hit:
            _TASK_STATE_STATS_METRICS["cache_hits"] = float(_TASK_STATE_STATS_METRICS.get("cache_hits", 0.0)) + 1.0
        _TASK_STATE_STATS_METRICS["total_duration_ms"] = (
            float(_TASK_STATE_STATS_METRICS.get("total_duration_ms", 0.0)) + duration
        )
        _TASK_STATE_STATS_METRICS["last_duration_ms"] = duration
        _TASK_STATE_STATS_METRICS["max_duration_ms"] = max(
            float(_TASK_STATE_STATS_METRICS.get("max_duration_ms", 0.0)),
            duration,
        )


def task_state_stats_metrics_snapshot() -> dict[str, float]:
    with _TASK_STATE_STATS_METRICS_LOCK:
        out = dict(_TASK_STATE_STATS_METRICS)
    calls = float(out.get("calls", 0.0))
    out["avg_duration_ms"] = float(out.get("total_duration_ms", 0.0)) / calls if calls > 0 else 0.0
    return out


def task_future_reaper_metrics_snapshot() -> dict[str, Any]:
    return {
        "rows_total": dict(_REAPER_METRICS),
        "payload_evict_errors_total": float(_REAPER_PAYLOAD_EVICT_ERRORS_TOTAL),
    }


def future_timeout_metrics_snapshot() -> dict[str, Any]:
    by_op: dict[str, dict[str, float]] = {}
    raw_by_op = _FUTURE_TIMEOUT_METRICS.get("by_op")
    if isinstance(raw_by_op, dict):
        for op, rec in raw_by_op.items():
            if not isinstance(rec, dict):
                continue
            by_op[str(op)] = {
                "queue": float(rec.get("queue") or 0.0),
                "execution": float(rec.get("execution") or 0.0),
                "total": float(rec.get("total") or 0.0),
            }
    return {
        "queue": float(_FUTURE_TIMEOUT_METRICS.get("queue") or 0.0),
        "execution": float(_FUTURE_TIMEOUT_METRICS.get("execution") or 0.0),
        "total": float(_FUTURE_TIMEOUT_METRICS.get("total") or 0.0),
        "by_op": by_op,
    }


def _extract_training_step(result: Any) -> int | None:
    if not isinstance(result, dict):
        return None
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        return None
    step = metrics.get("step")
    if isinstance(step, bool):
        return None
    if isinstance(step, int):
        return int(step)
    if isinstance(step, float) and step.is_integer():
        return int(step)
    return None


def _sync_training_session_step(meta: dict[str, Any] | None, result: Any) -> Any:
    if not isinstance(meta, dict) or not _is_training_step_op(meta.get("op")):
        return result
    model_id = meta.get("model_id")
    if not model_id:
        return result

    try:
        from mint_server.backend.stores.training_session_store import (
            bump_training_session_step_best_effort,
            set_training_session_step_best_effort,
        )

        step = _extract_training_step(result)
        if step is None:
            bump_training_session_step_best_effort(str(model_id))
            return result

        set_training_session_step_best_effort(str(model_id), int(step))
        if isinstance(result, dict):
            metrics = result.get("metrics")
            if isinstance(metrics, dict):
                metrics["step"] = int(step)
        return result
    except Exception:
        return result


def _meta_with_request_op(meta: dict[str, Any] | None, request_op: Any) -> dict[str, Any]:
    out = dict(meta or {})
    op = out.get("op")
    if isinstance(op, str) and op.strip():
        return out
    op = str(request_op or "").strip()
    if op:
        out["op"] = op
    return out


_SCHEMA_VERSION = 1
_OWNER_PREFIX = "owner:"
_TASK_PREFIX = "task:"
_INDEX_STATUS_PREFIX = "idx:task:status:"
_INDEX_DOMAIN_PREFIX = "idx:task:domain:"
_INDEX_META_PREFIX = "idx:task:meta:"
_INDEX_RESULT_PREFIX = "idx:task:result:"
_INDEX_STAGED_PREFIX = "idx:task:staged:"
_INDEX_LEASE_PREFIX = "idx:task:lease:"
_INDEX_CREATED_PREFIX = "idx:task:created:"
_INDEX_UPDATED_PREFIX = "idx:task:updated:"

_META_INDEX_KEYS = (
    "model_id",
    "sampling_session_id",
    "consumer_job_id",
    "op",
    "domain_key",
    "billing_status",
)


def _task_state_kv_db_path() -> str:
    task_state_db = Path(
        str(
            os.environ.get("MINT_TASK_STATE_STORE_DB_PATH")
            or getattr(
                server_config,
                "task_state_store_db_path",
                "/vePFS-Mindverse/share/mint/dev/data/task-state/task_state.sqlite3",
            )
        )
    )
    default_path = task_state_db.parent.parent / "future-state" / "futures.rocksdb"
    return str(
        os.environ.get("MINT_FUTURE_STATE_STORE_DB_PATH")
        or getattr(
            server_config,
            "future_state_store_db_path",
            str(default_path),
        )
    )


def _encode_record(record: dict[str, Any]) -> str:
    raw = dict(record)
    request_json = raw.get("request_json") or b""
    raw["request_json_b64"] = base64.b64encode(bytes(request_json)).decode("ascii")
    raw.pop("request_json", None)
    raw["schema_version"] = _SCHEMA_VERSION
    return json.dumps(raw, ensure_ascii=True, sort_keys=True)


def _decode_record(raw: str) -> dict[str, Any]:
    data = _json_loads(raw)
    request_json_b64 = str(data.pop("request_json_b64", "") or "")
    try:
        request_json = base64.b64decode(request_json_b64.encode("ascii")) if request_json_b64 else b""
    except Exception:
        request_json = b""
    data["request_json"] = request_json
    data.pop("schema_version", None)
    return data


def _task_key(request_id: str) -> str:
    return f"{_TASK_PREFIX}{str(request_id)}"


def _owner_key(name: str) -> str:
    return f"{_OWNER_PREFIX}{str(name)}"


def _index_value(value: Any) -> str:
    raw = str(value if value is not None else "")
    out = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw)
    return out or "_"


def _sortable_ts(value: Any) -> str:
    try:
        return f"{float(value):020.6f}"
    except Exception:
        return f"{0.0:020.6f}"


def _status_index_key(status: str, request_id: str) -> str:
    return f"{_INDEX_STATUS_PREFIX}{_index_value(status)}:{request_id}"


def _domain_index_key(domain_key: str, request_id: str) -> str:
    return f"{_INDEX_DOMAIN_PREFIX}{_index_value(domain_key)}:{request_id}"


def _meta_index_key(meta_key: str, meta_value: Any, request_id: str) -> str:
    return f"{_INDEX_META_PREFIX}{_index_value(meta_key)}:{_index_value(meta_value)}:{request_id}"


def _result_index_key(record: dict[str, Any], request_id: str) -> str:
    return f"{_INDEX_RESULT_PREFIX}{_sortable_ts(_terminal_completed_at(record))}:{request_id}"


def _staged_index_key(record: dict[str, Any], request_id: str) -> str:
    return f"{_INDEX_STAGED_PREFIX}{_sortable_ts(record.get('updated_at'))}:{request_id}"


def _lease_index_key(record: dict[str, Any], request_id: str) -> str:
    expiry = record.get("finalizing_until") or record.get("lease_expires_at") or 0.0
    return f"{_INDEX_LEASE_PREFIX}{_sortable_ts(expiry)}:{request_id}"


def _created_index_key(record: dict[str, Any], request_id: str) -> str:
    return f"{_INDEX_CREATED_PREFIX}{_sortable_ts(record.get('created_at'))}:{request_id}"


def _updated_index_key(record: dict[str, Any], request_id: str) -> str:
    return f"{_INDEX_UPDATED_PREFIX}{_sortable_ts(record.get('updated_at'))}:{request_id}"


def _merge_metadata(record: dict[str, Any], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    current = record.get("metadata")
    merged = dict(current) if isinstance(current, dict) else {}
    merged.update(dict(metadata or {}))
    return merged


def _merge_metadata_with_abandoned_staged_payload(
    record: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    *,
    new_staged_payload_path: str | None = None,
) -> dict[str, Any]:
    merged = _merge_metadata(record, metadata)
    old_path = record.get("staged_payload_path")
    if old_path is None:
        return merged
    old_path = str(old_path)
    if new_staged_payload_path is not None and old_path == str(new_staged_payload_path):
        return merged
    existing = merged.get("abandoned_staged_payload_paths")
    paths = [str(value) for value in existing] if isinstance(existing, list) else []
    if old_path not in paths:
        paths.append(old_path)
    merged["abandoned_staged_payload_paths"] = paths
    return merged


def _require_staged_success_path(record: dict[str, Any], result_path: str | None) -> bool:
    staged = record.get("staged_payload_path")
    return staged is None or str(staged) == str(result_path or "")


def _terminal_completed_at(record: dict[str, Any]) -> float:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    for key in ("done_at", "failed_at"):
        value = metadata.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        try:
            if value is not None:
                return float(value)
        except Exception:
            pass
    return float(record.get("updated_at") or 0.0)


class TaskStateStore:
    """SQLite-backed task state machine.

    The class is intentionally synchronous. V1 deployment should wrap it in a
    single-writer actor/service so API workers, schedulers, and runtimes do not
    open the SQLite file directly.
    """

    def __init__(self, db_path: str | os.PathLike[str]) -> None:
        self._db_path = str(db_path)
        self._lock = threading.RLock()
        if self._db_path == ":memory:":
            hot_kv_path = ":memory:"
            kv_path = ":memory:"
        else:
            # Derive sibling paths from db_path regardless of server_config,
            # so explicit paths (e.g. in tests) are not overridden by config.
            base = Path(self._db_path)
            hot_kv_path = str(base.parent.parent / "task-hot-kv" / "task_hot.rocksdb")
            kv_path = str(base.parent.parent / "future-state" / "futures.rocksdb")
        self._hot_kv = TaskHotKVStore(hot_kv_path)
        self._owner_lock = threading.RLock()
        self._index_lock = threading.RLock()
        self._locks = [threading.RLock() for _ in range(256)]
        self._kv: KVBackend = (
            InMemoryKVBackend()
            if kv_path == ":memory:"
            else RocksKVBackend(
                kv_path,
                unavailable_error=TaskStateStoreUnavailableError,
                memory_fallback_for_pytest=True,
            )
        )
        self._ensure_indexes()
        self._billing_outbox = BillingOutbox(
            self._hot_kv,
            inc_metric=_inc_billing_metric,
            metrics_snapshot=billing_metrics_snapshot,
        )
        self._session_heartbeat_max_age_s = float(
            os.environ.get("MINT_SESSION_HEARTBEAT_MAX_AGE_S", str(7 * 24 * 3600))
        )
        self._session_heartbeat_prune_every = max(
            1,
            int(os.environ.get("MINT_SESSION_HEARTBEAT_PRUNE_EVERY", "256")),
        )
        self._session_heartbeat_updates_since_prune = 0

    @classmethod
    def in_memory(cls) -> "TaskStateStore":
        return cls(":memory:")

    @property
    def db_path(self) -> str:
        return self._db_path

    def close(self) -> None:
        self._kv.close()
        self._hot_kv.close()

    def ping(self) -> dict[str, Any]:
        self._kv.put("__ping__", _json_dumps({"ok": True}))
        self._kv.delete("__ping__")
        return {"ok": True}

    def _lock_for_request(self, request_id: str) -> threading.RLock:
        return self._locks[hash(str(request_id)) % len(self._locks)]

    def _load(self, request_id: str) -> dict[str, Any]:
        raw = self._kv.get(_task_key(request_id))
        if raw is None:
            raise TaskStateNotFoundError(str(request_id))
        return _decode_record(raw)

    def _save(self, record: dict[str, Any]) -> dict[str, Any]:
        self._kv.put(_task_key(str(record["request_id"])), _encode_record(record))
        return dict(record)

    def _delete_index_for_record(self, record: dict[str, Any]) -> None:
        request_id = str(record.get("request_id") or "")
        if not request_id:
            return
        status = str(record.get("status") or "")
        self._kv.delete(_status_index_key(status, request_id))
        self._kv.delete(_domain_index_key(str(record.get("domain_key") or ""), request_id))
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        for key in _META_INDEX_KEYS:
            value = metadata.get(key, record.get(key))
            if value is not None:
                self._kv.delete(_meta_index_key(key, value, request_id))
        self._kv.delete(_created_index_key(record, request_id))
        self._kv.delete(_updated_index_key(record, request_id))
        if status in TERMINAL_TASK_STATUSES and record.get("result_path"):
            self._kv.delete(_result_index_key(record, request_id))
        if record.get("staged_payload_path"):
            self._kv.delete(_staged_index_key(record, request_id))
        if status in {"leased", "finalizing"}:
            self._kv.delete(_lease_index_key(record, request_id))

    def _write_index_for_record(self, record: dict[str, Any]) -> None:
        request_id = str(record.get("request_id") or "")
        if not request_id:
            return
        status = str(record.get("status") or "")
        self._kv.put(_status_index_key(status, request_id), request_id)
        self._kv.put(_domain_index_key(str(record.get("domain_key") or ""), request_id), request_id)
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        for key in _META_INDEX_KEYS:
            value = metadata.get(key, record.get(key))
            if value is not None:
                self._kv.put(_meta_index_key(key, value, request_id), request_id)
        self._kv.put(_created_index_key(record, request_id), request_id)
        self._kv.put(_updated_index_key(record, request_id), request_id)
        if status in TERMINAL_TASK_STATUSES and record.get("result_path"):
            self._kv.put(_result_index_key(record, request_id), request_id)
        if record.get("staged_payload_path"):
            self._kv.put(_staged_index_key(record, request_id), request_id)
        if status in {"leased", "finalizing"}:
            self._kv.put(_lease_index_key(record, request_id), request_id)

    def _save_indexed(self, record: dict[str, Any], *, old_record: dict[str, Any] | None = None) -> dict[str, Any]:
        if old_record is not None:
            self._delete_index_for_record(old_record)
        self._save(record)
        self._write_index_for_record(record)
        return dict(record)

    def _delete_task_indexed(self, request_id: str) -> bool:
        key = _task_key(str(request_id))
        old = None
        try:
            old = self._load(str(request_id))
        except TaskStateNotFoundError:
            pass
        if old is not None:
            self._delete_index_for_record(old)
        existed = self._kv.get(key) is not None
        self._kv.delete(key)
        return existed

    def _ensure_indexes(self) -> None:
        needs_rebuild = False
        if self._kv.get("__future_index_version__") != "2":
            needs_rebuild = True
        else:
            for key in self._kv.keys(prefix=_TASK_PREFIX, limit=1):
                request_id = key[len(_TASK_PREFIX):]
                if self._kv.get(_created_index_key(_decode_record(self._kv.get(key) or ""), request_id)) is None:
                    needs_rebuild = True
                break
        if not needs_rebuild:
            return
        with self._index_lock:
            for key in list(self._kv.keys(prefix="idx:")):
                self._kv.delete(key)
            for record in self._all_records_scan():
                self._write_index_for_record(record)
            self._kv.put("__future_index_version__", "2")

    def _all_records_scan(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for key in self._kv.keys(prefix=_TASK_PREFIX):
            raw = self._kv.get(key)
            if raw is not None:
                records.append(_decode_record(raw))
        records.sort(key=lambda r: (float(r.get("created_at") or 0.0), str(r.get("request_id") or "")))
        return records

    def _ids_from_index_prefix(self, prefix: str, *, limit: int | None = None) -> list[str]:
        out: list[str] = []
        max_items = None if limit is None else max(0, int(limit))
        if max_items == 0:
            return out
        keys = self._kv.keys(prefix=prefix, limit=max_items)
        for key in keys:
            value = self._kv.get(key)
            if value is None:
                continue
            out.append(str(value))
            if max_items is not None and len(out) >= max_items:
                break
        return out

    def _records_from_ids(self, request_ids: list[str], *, limit: int | None = None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for request_id in request_ids:
            if request_id in seen:
                continue
            seen.add(request_id)
            try:
                out.append(self.get_task(request_id))
            except TaskStateNotFoundError:
                continue
            if limit is not None and len(out) >= max(0, int(limit)):
                break
        return out

    def _owner(self, name: str) -> dict[str, Any] | None:
        raw = self._kv.get(_owner_key(str(name)))
        if raw is None:
            return None
        return _json_loads(raw)

    def upsert_sampling_session(self, *, session_id: str, info: dict[str, Any]) -> None:
        session_id = str(session_id)
        if not session_id:
            raise ValueError("session_id is required")
        incoming = dict(info)
        incoming["session_id"] = session_id
        incoming.setdefault("last_activity", time.time())
        incoming.setdefault("lora_loaded", False)
        incoming.setdefault("uses_base_model", False)
        incoming.setdefault("inflight_requests", 0)
        incoming_version = int(incoming.get("metadata_version") or 0)

        def _mutate(existing: dict[str, Any] | None) -> dict[str, Any]:
            if existing is None:
                incoming["metadata_version"] = max(incoming_version, 1)
                return incoming
            existing_version = int(existing.get("metadata_version") or 0)
            if incoming_version and incoming_version < existing_version:
                if "last_activity" in incoming:
                    existing["last_activity"] = max(
                        float(existing.get("last_activity") or 0.0),
                        float(incoming.get("last_activity") or 0.0),
                    )
                return existing
            merged = {**existing, **incoming}
            merged["metadata_version"] = max(existing_version + 1, incoming_version, 1)
            return merged

        self._hot_kv.mutate_record("sampling_session", session_id, _mutate)

    def delete_sampling_session(self, *, session_id: str) -> None:
        self._hot_kv.delete_record("sampling_session", str(session_id))

    def set_sampling_session_last_activity(self, *, session_id: str, last_activity: float) -> float | None:
        ts = float(last_activity)
        updated = self._hot_kv.mutate_record(
            "sampling_session",
            str(session_id),
            lambda existing: None
            if existing is None
            else {
                **existing,
                "last_activity": ts,
                "metadata_version": int(existing.get("metadata_version") or 0) + 1,
            },
        )
        if updated is None:
            return None
        return ts

    def get_sampling_session(self, *, session_id: str) -> dict[str, Any] | None:
        return self._hot_kv.get_record("sampling_session", str(session_id))

    def list_sampling_sessions(self) -> list[dict[str, Any]]:
        return self._hot_kv.list_records("sampling_session")

    def upsert_training_session(self, *, model_id: str, info: dict[str, Any]) -> None:
        model_id = str(model_id)
        if not model_id:
            raise ValueError("model_id is required")
        incoming = dict(info)
        incoming["model_id"] = model_id
        incoming.setdefault("current_step", 0)
        incoming.setdefault("last_activity", time.time())
        incoming_version = max(1, int(incoming.get("metadata_version") or 1))

        def _mutate(existing: dict[str, Any] | None) -> dict[str, Any]:
            current = dict(existing or {})
            current_version = max(1, int(current.get("metadata_version") or 1))
            if incoming_version < current_version:
                incoming_last_activity = float(incoming.get("last_activity", 0.0) or 0.0)
                current_last_activity = float(current.get("last_activity", 0.0) or 0.0)
                current["last_activity"] = max(current_last_activity, incoming_last_activity)
                try:
                    incoming_step = int(incoming.get("current_step", 0))
                    current_step = int(current.get("current_step", 0))
                    current["current_step"] = max(current_step, incoming_step)
                except Exception:
                    pass
                return current
            merged = {**current, **incoming}
            merged.setdefault("current_step", int(current.get("current_step", 0)))
            merged.setdefault("last_activity", time.time())
            merged["metadata_version"] = incoming_version
            return merged

        self._hot_kv.mutate_record("training_session", model_id, _mutate)

    def delete_training_session(self, *, model_id: str) -> None:
        self._hot_kv.delete_record("training_session", str(model_id))

    def set_training_session_last_activity(self, *, model_id: str, last_activity: float) -> float | None:
        updated = self._hot_kv.mutate_record(
            "training_session",
            str(model_id),
            lambda info: None if info is None else {**info, "last_activity": float(last_activity)},
        )
        if updated is None:
            return None
        return float(updated["last_activity"])

    def mark_training_session_inflight(self, *, model_id: str, delta: int) -> int | None:
        now = time.time()
        updated = self._hot_kv.mutate_record(
            "training_session",
            str(model_id),
            lambda info: None
            if info is None
            else {
                **info,
                "last_activity": now,
                "inflight_ops": max(0, int(info.get("inflight_ops") or 0) + int(delta)),
            },
        )
        if updated is None:
            return None
        return int(updated.get("inflight_ops") or 0)

    def get_training_session(self, *, model_id: str) -> dict[str, Any] | None:
        return self._hot_kv.get_record("training_session", str(model_id))

    def bump_training_session_step(self, *, model_id: str) -> int:
        updated = self._hot_kv.mutate_record(
            "training_session",
            str(model_id),
            lambda info: None
            if info is None
            else {**info, "current_step": int(info.get("current_step", 0)) + 1},
        )
        if updated is None:
            return 0
        return int(updated["current_step"])

    def set_training_session_step(self, *, model_id: str, step: int) -> int:
        updated = self._hot_kv.mutate_record(
            "training_session",
            str(model_id),
            lambda info: None
            if info is None
            else {**info, "current_step": max(int(info.get("current_step", 0)), int(step))},
        )
        if updated is None:
            return int(step)
        return int(updated["current_step"])

    def list_training_sessions(self) -> list[dict[str, Any]]:
        return self._hot_kv.list_records("training_session")

    def upsert_gateway_sampling_session(
        self,
        *,
        sampling_session_id: str,
        upstream_alias: str,
        base_model: str,
    ) -> None:
        self._hot_kv.replace_record(
            "gateway_sampling_session",
            str(sampling_session_id),
            {
                "sampling_session_id": str(sampling_session_id),
                "upstream_alias": str(upstream_alias),
                "base_model": str(base_model),
            },
        )

    def get_gateway_sampling_session(self, *, sampling_session_id: str) -> dict[str, str] | None:
        info = self._hot_kv.get_record("gateway_sampling_session", str(sampling_session_id))
        if info is None:
            return None
        return {
            "upstream_alias": str(info.get("upstream_alias") or ""),
            "base_model": str(info.get("base_model") or ""),
        }

    def delete_gateway_sampling_session(self, *, sampling_session_id: str) -> None:
        self._hot_kv.delete_record("gateway_sampling_session", str(sampling_session_id))

    def upsert_gateway_training_model(
        self,
        *,
        model_id: str,
        upstream_alias: str,
        base_model: str,
        owner_id: str | None = None,
    ) -> None:
        self._hot_kv.replace_record(
            "gateway_training_model",
            str(model_id),
            {
                "model_id": str(model_id),
                "upstream_alias": str(upstream_alias),
                "base_model": str(base_model),
                "owner_id": None if owner_id is None else str(owner_id),
            },
        )

    def get_gateway_training_model(self, *, model_id: str) -> dict[str, str | None] | None:
        info = self._hot_kv.get_record("gateway_training_model", str(model_id))
        if info is None:
            return None
        return {
            "upstream_alias": str(info.get("upstream_alias") or ""),
            "base_model": str(info.get("base_model") or ""),
            "owner_id": None if info.get("owner_id") is None else str(info.get("owner_id")),
        }

    def delete_gateway_training_model(self, *, model_id: str) -> None:
        self._hot_kv.delete_record("gateway_training_model", str(model_id))

    def list_gateway_routes(self) -> dict[str, Any]:
        return {
            "sampling_sessions": {
                str(item.get("sampling_session_id") or ""): {
                    key: value
                    for key, value in item.items()
                    if key != "sampling_session_id"
                }
                for item in self._hot_kv.list_records("gateway_sampling_session")
            },
            "training_models": {
                str(item.get("model_id") or ""): {key: value for key, value in item.items() if key != "model_id"}
                for item in self._hot_kv.list_records("gateway_training_model")
            },
        }

    def upsert_session_index(self, *, session_id: str, info: dict[str, Any]) -> None:
        session_id = str(session_id)
        if not session_id:
            raise ValueError("session_id is required")
        incoming = dict(info)
        incoming["session_id"] = session_id
        def _mutate(existing: dict[str, Any] | None) -> dict[str, Any]:
            existing = existing or {}
            merged = {**existing, **incoming}
            merged.setdefault("training_run_ids", list(existing.get("training_run_ids") or []))
            merged.setdefault("sampler_ids", list(existing.get("sampler_ids") or []))
            merged.setdefault("heartbeat_sampler_ids", list(existing.get("heartbeat_sampler_ids") or []))
            return merged

        self._hot_kv.mutate_record("session_index", session_id, _mutate)

    def add_training_run_to_session_index(
        self,
        *,
        session_id: str,
        training_run_id: str,
        user_id: str | None = None,
        created_at: Any | None = None,
    ) -> None:
        session_id = str(session_id)
        training_run_id = str(training_run_id)
        def _mutate(existing: dict[str, Any] | None) -> dict[str, Any]:
            item = dict(existing or {"session_id": session_id})
            runs = list(item.get("training_run_ids") or [])
            if training_run_id not in runs:
                runs.append(training_run_id)
            item["training_run_ids"] = runs
            item.setdefault("sampler_ids", [])
            item.setdefault("heartbeat_sampler_ids", [])
            if user_id is not None:
                item.setdefault("user_id", str(user_id))
            if created_at is not None:
                item.setdefault("created_at", created_at)
            return item

        self._hot_kv.mutate_record("session_index", session_id, _mutate)

    def add_sampler_to_session_index(
        self,
        *,
        session_id: str,
        sampler_id: str,
        user_id: str | None = None,
        created_at: Any | None = None,
    ) -> None:
        session_id = str(session_id)
        sampler_id = str(sampler_id)
        def _mutate(existing: dict[str, Any] | None) -> dict[str, Any]:
            item = dict(existing or {"session_id": session_id})
            samplers = list(item.get("sampler_ids") or [])
            if sampler_id not in samplers:
                samplers.append(sampler_id)
            item["sampler_ids"] = samplers
            item.setdefault("training_run_ids", [])
            item.setdefault("heartbeat_sampler_ids", list(item.get("heartbeat_sampler_ids") or []))
            if user_id is not None:
                item.setdefault("user_id", str(user_id))
            if created_at is not None:
                item.setdefault("created_at", created_at)
            return item

        self._hot_kv.mutate_record("session_index", session_id, _mutate)

    def add_heartbeat_sampler_to_session_index(
        self,
        *,
        session_id: str,
        sampler_id: str,
        user_id: str | None = None,
        created_at: Any | None = None,
    ) -> None:
        session_id = str(session_id)
        sampler_id = str(sampler_id)
        def _mutate(existing: dict[str, Any] | None) -> dict[str, Any]:
            item = dict(existing or {"session_id": session_id})
            samplers = list(item.get("sampler_ids") or [])
            if sampler_id not in samplers:
                samplers.append(sampler_id)
            heartbeat_samplers = list(item.get("heartbeat_sampler_ids") or [])
            if sampler_id not in heartbeat_samplers:
                heartbeat_samplers.append(sampler_id)
            item["sampler_ids"] = samplers
            item["heartbeat_sampler_ids"] = heartbeat_samplers
            item.setdefault("training_run_ids", [])
            if user_id is not None:
                item.setdefault("user_id", str(user_id))
            if created_at is not None:
                item.setdefault("created_at", created_at)
            return item

        self._hot_kv.mutate_record("session_index", session_id, _mutate)

    def remove_sampler_from_session_index(self, *, session_id: str, sampler_id: str) -> None:
        def _mutate(existing: dict[str, Any] | None) -> dict[str, Any] | None:
            item = existing
            if item is None:
                return None
            sid = str(sampler_id)
            item["sampler_ids"] = [x for x in list(item.get("sampler_ids") or []) if str(x) != sid]
            item["heartbeat_sampler_ids"] = [
                x for x in list(item.get("heartbeat_sampler_ids") or []) if str(x) != sid
            ]
            return item

        self._hot_kv.mutate_record("session_index", str(session_id), _mutate)

    def get_session_index(self, *, session_id: str) -> dict[str, Any] | None:
        return self._hot_kv.get_record("session_index", str(session_id))

    def list_session_index(self) -> list[dict[str, Any]]:
        return self._hot_kv.list_records("session_index")

    def upsert_sampler_index(self, *, sampler_id: str, info: dict[str, Any]) -> None:
        sampler_id = str(sampler_id)
        if not sampler_id:
            raise ValueError("sampler_id is required")
        incoming = dict(info)
        incoming["sampler_id"] = sampler_id
        self._hot_kv.upsert_record("sampler_index", sampler_id, incoming)

    def delete_sampler_index(self, *, sampler_id: str) -> None:
        self._hot_kv.delete_record("sampler_index", str(sampler_id))

    def get_sampler_index(self, *, sampler_id: str) -> dict[str, Any] | None:
        return self._hot_kv.get_record("sampler_index", str(sampler_id))

    def list_sampler_index(self) -> list[dict[str, Any]]:
        return self._hot_kv.list_records("sampler_index")

    def update_session_heartbeat(self, *, session_id: str, now: float | None = None) -> None:
        session_id = str(session_id)
        if not session_id:
            return
        ts = _now(now)
        self._hot_kv.update_heartbeat(session_id=session_id, now=ts)
        with self._lock:
            self._session_heartbeat_updates_since_prune += 1
            should_prune = self._session_heartbeat_updates_since_prune >= self._session_heartbeat_prune_every
            if should_prune:
                self._session_heartbeat_updates_since_prune = 0
        if should_prune:
            self._hot_kv.prune_heartbeats(now=ts, max_age_s=self._session_heartbeat_max_age_s)

    def get_session_heartbeat(self, *, session_id: str) -> float | None:
        return self._hot_kv.get_heartbeat(session_id=str(session_id))

    def delete_session_heartbeat(self, *, session_id: str) -> bool:
        return self._hot_kv.delete_record("heartbeat", str(session_id))

    def session_heartbeat_size(self) -> int:
        return self._hot_kv.heartbeat_size()

    def is_session_heartbeat_stale(self, *, session_id: str, ttl_s: float, now: float | None = None) -> bool:
        ttl = float(ttl_s)
        if ttl <= 0:
            return False
        session_id = str(session_id)
        if not session_id:
            return False
        ts = _now(now)
        last = self._hot_kv.get_heartbeat(session_id=session_id)
        if last is None:
            return False
        return (ts - float(last)) > ttl

    def prune_session_heartbeats(self, *, max_age_s: float, now: float | None = None) -> int:
        return self._hot_kv.prune_heartbeats(now=_now(now), max_age_s=float(max_age_s))

    def _prune_session_heartbeats_locked(self, *, now: float, max_age_s: float) -> int:
        return self._hot_kv.prune_heartbeats(now=now, max_age_s=max_age_s)

    def _configure_connection(self) -> None:
        pass

    def _apply_schema(self) -> None:
        pass

    def integrity_check(self) -> str:
        return "ok"

    def acquire_scheduler_owner(
        self,
        *,
        owner_id: str,
        ttl_s: float,
        name: str = "model_work_scheduler",
        now: float | None = None,
    ) -> OwnerLeaseResult:
        ts = _now(now)
        expires_at = ts + max(1.0, float(ttl_s))
        with self._owner_lock:
            row = self._owner(str(name))
            if row is not None and str(row.get("owner_id")) != owner_id and float(row.get("expires_at") or 0.0) > ts:
                return OwnerLeaseResult(
                    ok=False,
                    reason=ConflictReason.OWNER_ACTIVE,
                    owner_id=str(row.get("owner_id")),
                    epoch=int(row.get("epoch") or 0),
                    expires_at=float(row.get("expires_at") or 0.0),
                )
            epoch = 1 if row is None else int(row.get("epoch") or 0) + (0 if str(row.get("owner_id")) == owner_id else 1)
            fencing_token = f"{name}:{epoch}:{owner_id}"
            out = {
                "ok": True,
                "owner_id": owner_id,
                "epoch": epoch,
                "renewed_at": ts,
                "expires_at": expires_at,
                "fencing_token": fencing_token,
            }
            self._kv.put(_owner_key(name), _json_dumps(out))
            return OwnerLeaseResult(
                ok=True,
                owner_id=owner_id,
                epoch=epoch,
                expires_at=expires_at,
                fencing_token=fencing_token,
            )

    def renew_scheduler_owner(
        self,
        *,
        owner_id: str,
        epoch: int,
        ttl_s: float,
        name: str = "model_work_scheduler",
        now: float | None = None,
    ) -> OwnerLeaseResult:
        ts = _now(now)
        expires_at = ts + max(1.0, float(ttl_s))
        with self._owner_lock:
            row = self._owner(str(name))
            if row is None or str(row.get("owner_id")) != str(owner_id) or int(row.get("epoch") or 0) != int(epoch):
                return OwnerLeaseResult(ok=False, reason=ConflictReason.STALE_OWNER)
            row.update({"renewed_at": ts, "expires_at": expires_at})
            self._kv.put(_owner_key(str(name)), _json_dumps(row))
            return OwnerLeaseResult(ok=True, owner_id=str(owner_id), epoch=int(epoch), expires_at=expires_at)

    def assert_scheduler_owner(
        self,
        *,
        scheduler_epoch: int,
        name: str = "model_work_scheduler",
        now: float | None = None,
    ) -> None:
        self._assert_scheduler_owner(scheduler_epoch=scheduler_epoch, name=name, now=now)

    def _assert_scheduler_owner(
        self,
        *,
        scheduler_epoch: int,
        name: str = "model_work_scheduler",
        now: float | None = None,
    ) -> None:
        ts = _now(now)
        row = self._owner(name)
        if row is None or int(row.get("epoch") or 0) != int(scheduler_epoch) or float(row.get("expires_at") or 0.0) <= ts:
            raise TaskStateConflictError("stale scheduler owner epoch")

    def create_task(
        self,
        *,
        request_id: str,
        op: str,
        domain_key: str,
        request_json: bytes,
        payload_hash: str | None = None,
        metadata: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> CreateTaskResult:
        ts = _now(now)
        request_id = str(request_id)
        with self._lock_for_request(request_id):
            try:
                existing = self._load(request_id)
            except TaskStateNotFoundError:
                record = {
                    "request_id": request_id,
                    "op": str(op),
                    "status": "pending",
                    "domain_key": str(domain_key),
                    "subqueue_id": None,
                    "lease_id": None,
                    "attempt_id": None,
                    "scheduler_epoch": None,
                    "runtime_generation": None,
                    "consumer_id": None,
                    "request_json": bytes(request_json),
                    "payload_hash": payload_hash,
                    "result_path": None,
                    "result_checksum": None,
                    "result_size_bytes": None,
                    "staged_payload_path": None,
                    "staged_payload_checksum": None,
                    "staged_payload_size_bytes": None,
                    "error": None,
                    "metadata": dict(metadata or {}),
                    "created_at": ts,
                    "updated_at": ts,
                    "assigned_at": None,
                    "leased_at": None,
                    "lease_expires_at": None,
                    "finalizing_until": None,
                }
                self._save_indexed(record)
                return CreateTaskResult.from_wire({"ok": True, "created": True, "record": dict(record)})
            if payload_hash is not None and existing.get("payload_hash") not in (None, payload_hash):
                raise TaskStateConflictError("duplicate request_id with different payload hash")
            if str(existing.get("op")) != str(op) or str(existing.get("domain_key")) != str(domain_key):
                raise TaskStateConflictError("duplicate request_id with different task identity")
            if "model_work_scheduler_append_attempt_id" in (existing.get("metadata") or {}):
                return CreateTaskResult.from_wire({"ok": True, "created": False, "record": dict(existing)})
            old_record = dict(existing)
            existing["request_json"] = bytes(request_json)
            existing["metadata"] = _merge_metadata(existing, metadata)
            existing["updated_at"] = ts
            if existing.get("payload_hash") is None and payload_hash is not None:
                existing["payload_hash"] = payload_hash
            self._save_indexed(existing, old_record=old_record)
            return CreateTaskResult.from_wire({"ok": True, "created": False, "record": dict(existing)})

    def ensure_task(
        self,
        *,
        request_id: str,
        op: str = "unknown",
        domain_key: str = "future:default",
        request_json: bytes = b"{}",
        payload_hash: str | None = None,
        metadata: dict[str, Any] | None = None,
        status: str = "pending",
        now: float | None = None,
    ) -> dict[str, Any]:
        request_id = str(request_id)
        with self._lock_for_request(request_id):
            try:
                record = self._load(request_id)
            except TaskStateNotFoundError:
                ts = _now(now)
                record = {
                    "request_id": request_id,
                    "op": str(op),
                    "status": str(status),
                    "domain_key": str(domain_key),
                    "subqueue_id": None,
                    "lease_id": None,
                    "attempt_id": None,
                    "scheduler_epoch": None,
                    "runtime_generation": None,
                    "consumer_id": None,
                    "request_json": bytes(request_json),
                    "payload_hash": payload_hash,
                    "result_path": None,
                    "result_checksum": None,
                    "result_size_bytes": None,
                    "staged_payload_path": None,
                    "staged_payload_checksum": None,
                    "staged_payload_size_bytes": None,
                    "error": None,
                    "metadata": dict(metadata or {}),
                    "created_at": ts,
                    "updated_at": ts,
                    "assigned_at": None,
                    "leased_at": None,
                    "lease_expires_at": None,
                    "finalizing_until": None,
                }
                self._save_indexed(record)
                return {"ok": True, "created": True, "record": dict(record)}
            old_record = dict(record)
            record["metadata"] = _merge_metadata(record, metadata)
            record["updated_at"] = _now(now)
            self._save_indexed(record, old_record=old_record)
            return {"ok": True, "created": False, "record": dict(record)}

    def update_task_metadata(
        self,
        *,
        request_id: str,
        metadata: dict[str, Any] | None = None,
        status: str | None = None,
        now: float | None = None,
    ) -> TaskMutationResult:
        with self._lock_for_request(request_id):
            record = self._load(str(request_id))
            old_record = dict(record)
            record["metadata"] = _merge_metadata(record, metadata)
            if status is not None:
                record["status"] = str(status)
            record["updated_at"] = _now(now)
            self._save_indexed(record, old_record=old_record)
            return TaskMutationResult.from_wire({"ok": True, "record": dict(record)})

    def complete_task_success(
        self,
        *,
        request_id: str,
        result_path: str,
        result_checksum: str,
        result_size_bytes: int,
        metadata: dict[str, Any] | None = None,
        billing_observations: list[dict[str, Any]] | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        billing_meta: dict[str, Any] = {}
        if billing_observations:
            billing_meta = self._billing_outbox.append_after_terminal_success(
                observations=billing_observations,
                source=str(request_id),
                now=float(now) if now is not None else time.time(),
            )
        merged_meta = {**(metadata or {}), **billing_meta} if billing_meta else metadata
        result = self._complete_task_direct(
            status="done",
            error=None,
            request_id=request_id,
            result_path=result_path,
            result_checksum=result_checksum,
            result_size_bytes=result_size_bytes,
            metadata=merged_meta,
            now=now,
        )
        if billing_meta and result.get("record"):
            # On the idempotent path _complete_task_direct returns the old record;
            # persist the billing metadata so callers see billing_status immediately.
            if result.get("idempotent"):
                try:
                    updated = self.update_task_metadata(
                        request_id=request_id,
                        metadata=billing_meta,
                        now=now,
                    )
                    if isinstance(updated, TaskMutationResult) and isinstance(updated.record, dict):
                        result["record"] = updated.record
                    elif isinstance(updated, dict) and isinstance(updated.get("record"), dict):
                        result["record"] = updated["record"]
                    else:
                        result["record"].setdefault("metadata", {}).update(billing_meta)
                except Exception:
                    result["record"].setdefault("metadata", {}).update(billing_meta)
            else:
                result["record"].setdefault("metadata", {}).update(billing_meta)
        return result

    def stage_payload(
        self,
        *,
        request_id: str,
        staged_payload_path: str,
        metadata: dict[str, Any] | None = None,
        status: str | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        with self._lock_for_request(request_id):
            record = self._load(str(request_id))
            old_record = dict(record)
            if str(record.get("status")) in TERMINAL_TASK_STATUSES:
                raise TaskStateConflictError("cannot stage payload for terminal task")
            record["staged_payload_path"] = str(staged_payload_path)
            record["staged_payload_checksum"] = None
            record["staged_payload_size_bytes"] = None
            record["metadata"] = {
                **_merge_metadata_with_abandoned_staged_payload(
                    record,
                    metadata,
                    new_staged_payload_path=str(staged_payload_path),
                ),
                "payload_state": "staging",
            }
            if status is not None:
                record["status"] = str(status)
            record["updated_at"] = _now(now)
            self._save_indexed(record, old_record=old_record)
            return {"ok": True, "record": dict(record)}

    def complete_task_failure(
        self,
        *,
        request_id: str,
        error: str,
        result_path: str | None = None,
        result_checksum: str | None = None,
        result_size_bytes: int | None = None,
        metadata: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> TaskMutationResult:
        result = self._complete_task_direct(
            status="failed",
            error=error,
            request_id=request_id,
            result_path=result_path,
            result_checksum=result_checksum,
            result_size_bytes=result_size_bytes,
            metadata=metadata,
            now=now,
        )
        return TaskMutationResult.from_wire(result)

    def mark_task_retrieved(self, *, request_id: str, now: float | None = None) -> dict[str, Any]:
        with self._lock_for_request(request_id):
            record = self._load(str(request_id))
            old_record = dict(record)
            if str(record.get("status")) == "retrieved":
                return {"ok": True, "record": dict(record)}
            if str(record.get("status")) not in {"done", "failed", "expired", "cancelled"}:
                raise TaskStateConflictError(f"cannot mark retrieved; current status={record.get('status')!r}")
            metadata = dict(record.get("metadata") or {})
            metadata.setdefault("terminal_status", str(record.get("status")))
            record["metadata"] = metadata
            record["status"] = "retrieved"
            record["updated_at"] = _now(now)
            self._save_indexed(record, old_record=old_record)
            return {"ok": True, "record": dict(record)}

    def forget_task(self, *, request_id: str) -> TaskMutationResult:
        with self._lock_for_request(request_id):
            existed = self._delete_task_indexed(str(request_id))
            return TaskMutationResult.from_wire({"ok": True, "deleted": existed})

    def expire_active_tasks(
        self,
        *,
        older_than_s: float,
        now: float | None = None,
        limit: int = 1000,
    ) -> list[str]:
        ttl_s = float(older_than_s)
        if ttl_s <= 0:
            return []
        ts = _now(now)
        cutoff = ts - ttl_s
        expired: list[str] = []
        expired_by_op: dict[str, int] = {}
        ids: list[str] = []
        remaining_ids = max(0, int(limit))
        if remaining_ids == 0:
            return []
        for status in ("pending", "queued", "assigned"):
            ids.extend(
                self._ids_from_index_prefix(
                    f"{_INDEX_STATUS_PREFIX}{_index_value(status)}:",
                    limit=remaining_ids,
                )
            )
            remaining_ids = max(0, int(limit) - len(ids))
            if remaining_ids == 0:
                break
        for request_id in ids:
            if len(expired) >= max(0, int(limit)):
                break
            with self._lock_for_request(request_id):
                try:
                    record = self._load(request_id)
                except TaskStateNotFoundError:
                    continue
                if str(record.get("status")) not in {"pending", "queued", "assigned"}:
                    continue
                if float(record.get("created_at") or 0.0) > cutoff:
                    continue
                old_record = dict(record)
                metadata = dict(record.get("metadata") or {})
                metadata.setdefault("terminal_status", "expired")
                metadata.setdefault("expired_at", ts)
                metadata.setdefault("failed_at", ts)
                record["metadata"] = metadata
                record["status"] = "expired"
                record["error"] = record.get("error") or "Future expired"
                record["updated_at"] = ts
                self._save_indexed(record, old_record=old_record)
                request_id = str(record["request_id"])
                expired.append(request_id)
                op = str(record.get("op") or "unknown")
                expired_by_op[op] = expired_by_op.get(op, 0) + 1
        if expired:
            _inc_reaper_rows("expire_pending", len(expired))
            for op, count in expired_by_op.items():
                _inc_future_timeout("execution", op=op, count=count)
        return expired

    def list_terminal_payloads_for_eviction(
        self,
        *,
        older_than_s: float,
        now: float | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        ttl_s = float(older_than_s)
        if ttl_s <= 0:
            return []
        cutoff = _now(now) - ttl_s
        out: list[dict[str, Any]] = []
        for request_id in self._ids_from_index_prefix(_INDEX_RESULT_PREFIX, limit=max(0, int(limit))):
            if len(out) >= max(0, int(limit)):
                break
            try:
                record = self.get_task(request_id)
            except TaskStateNotFoundError:
                continue
            if str(record.get("status")) not in TERMINAL_TASK_STATUSES:
                continue
            if not record.get("result_path"):
                continue
            if _terminal_completed_at(record) <= cutoff:
                out.append(dict(record))
        return out

    def mark_payload_evicted(
        self,
        *,
        request_id: str,
        expected_result_path: str,
        now: float | None = None,
    ) -> dict[str, Any]:
        with self._lock_for_request(request_id):
            record = self._load(str(request_id))
            old_record = dict(record)
            if str(record.get("status")) not in TERMINAL_TASK_STATUSES:
                raise TaskStateConflictError(f"cannot evict payload; current status={record.get('status')!r}")
            if str(record.get("result_path") or "") != str(expected_result_path):
                return {"ok": False, "reason": "payload_changed", "record": dict(record)}
            metadata = dict(record.get("metadata") or {})
            metadata.setdefault("terminal_status", str(record.get("status")))
            metadata["payload_evicted_at"] = _now(now)
            metadata.setdefault("evicted_result_size_bytes", record.get("result_size_bytes"))
            record["metadata"] = metadata
            record["result_path"] = None
            record["result_checksum"] = None
            record["result_size_bytes"] = None
            record["updated_at"] = _now(now)
            self._save_indexed(record, old_record=old_record)
        _inc_reaper_rows("evict_payload", 1)
        return {"ok": True, "record": dict(record)}

    def list_staged_payloads_for_gc(
        self,
        *,
        older_than_s: float,
        now: float | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        ttl_s = float(older_than_s)
        if ttl_s <= 0:
            return []
        cutoff = _now(now) - ttl_s
        out: list[dict[str, Any]] = []
        remaining_ids = max(0, int(limit))
        if remaining_ids == 0:
            return []
        candidate_ids = self._ids_from_index_prefix(_INDEX_STAGED_PREFIX, limit=remaining_ids)
        remaining_ids = max(0, int(limit) - len(candidate_ids))
        if remaining_ids > 0:
            candidate_ids.extend(self._ids_from_index_prefix(_INDEX_UPDATED_PREFIX, limit=remaining_ids))
        for request_id in candidate_ids:
            if len(out) >= max(0, int(limit)):
                break
            try:
                record = self.get_task(request_id)
            except TaskStateNotFoundError:
                continue
            status = str(record.get("status") or "")
            active_path = record.get("staged_payload_path")
            if isinstance(active_path, str) and active_path:
                expiry_base = float(record.get("finalizing_until") or record.get("updated_at") or 0.0)
                if (status == "finalizing" and expiry_base <= cutoff) or (status != "finalizing" and float(record.get("updated_at") or 0.0) <= cutoff):
                    out.append({"request_id": str(record["request_id"]), "path": active_path, "kind": "active", "status": status})
                    if len(out) >= max(0, int(limit)):
                        break
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            abandoned = metadata.get("abandoned_staged_payload_paths")
            if isinstance(abandoned, list) and float(record.get("updated_at") or 0.0) <= cutoff:
                for path in abandoned:
                    if isinstance(path, str) and path:
                        out.append({"request_id": str(record["request_id"]), "path": path, "kind": "abandoned", "status": status})
                        if len(out) >= max(0, int(limit)):
                            break
        return out

    def mark_staged_payload_gc_deleted(
        self,
        *,
        request_id: str,
        expected_staged_payload_path: str,
        now: float | None = None,
    ) -> dict[str, Any]:
        expected = str(expected_staged_payload_path)
        with self._lock_for_request(request_id):
            record = self._load(str(request_id))
            old_record = dict(record)
            metadata = dict(record.get("metadata") or {})
            abandoned = metadata.get("abandoned_staged_payload_paths")
            changed = False
            if isinstance(abandoned, list) and expected in [str(value) for value in abandoned]:
                metadata["abandoned_staged_payload_paths"] = [
                    str(value) for value in abandoned if isinstance(value, str) and str(value) != expected
                ]
                changed = True
            active_matches = str(record.get("staged_payload_path") or "") == expected
            if not changed and not active_matches:
                return {"ok": False, "reason": "payload_changed", "record": dict(record)}
            if active_matches:
                record["staged_payload_path"] = None
                record["staged_payload_checksum"] = None
                record["staged_payload_size_bytes"] = None
            record["metadata"] = metadata
            record["updated_at"] = _now(now)
            self._save_indexed(record, old_record=old_record)
        _inc_reaper_rows("gc_staged_payload", 1)
        return {"ok": True, "record": dict(record)}

    def delete_expired_tombstones(
        self,
        *,
        older_than_s: float,
        now: float | None = None,
        limit: int = 1000,
    ) -> list[str]:
        ttl_s = float(older_than_s)
        if ttl_s <= 0:
            return []
        cutoff = _now(now) - ttl_s
        deleted: list[str] = []
        for request_id in self._ids_from_index_prefix(_INDEX_UPDATED_PREFIX, limit=max(0, int(limit))):
            if len(deleted) >= max(0, int(limit)):
                break
            with self._lock_for_request(request_id):
                try:
                    record = self._load(request_id)
                except TaskStateNotFoundError:
                    continue
                if str(record.get("status")) not in TERMINAL_TASK_STATUSES:
                    continue
                if record.get("result_path"):
                    continue
                if _terminal_completed_at(record) > cutoff:
                    continue
                if self._delete_task_indexed(request_id):
                    deleted.append(request_id)
        if deleted:
            _inc_reaper_rows("delete_tombstone", len(deleted))
        return deleted

    def record_payload_evict_error(self, *, count: int = 1) -> dict[str, Any]:
        _inc_payload_evict_errors(int(count))
        return {"ok": True, "metrics": task_future_reaper_metrics_snapshot()}

    def append_billing_outbox(
        self,
        *,
        observations: list[dict[str, Any]],
        source: str = "unknown",
        now: float | None = None,
    ) -> dict[str, Any]:
        return self._billing_outbox.append(observations=observations, source=source, now=now)

    def _append_billing_outbox_after_terminal_success(
        self,
        *,
        observations: list[dict[str, Any]] | None,
        source: str,
        now: float,
    ) -> dict[str, Any]:
        return self._billing_outbox.append_after_terminal_success(
            observations=observations,
            source=source,
            now=now,
        )

    def _best_effort_update_billing_metadata(
        self,
        *,
        request_id: str,
        metadata: dict[str, Any],
        now: float,
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        if not metadata:
            return fallback
        try:
            updated = self.update_task_metadata(request_id=request_id, metadata=metadata, now=now)
            if isinstance(updated, TaskMutationResult) and isinstance(updated.record, dict):
                return updated.record
            if isinstance(updated, dict) and isinstance(updated.get("record"), dict):
                return updated["record"]
        except Exception:
            _inc_billing_metric("write_error", 1)
        fallback_metadata = fallback.get("metadata")
        if isinstance(fallback_metadata, dict):
            fallback["metadata"] = {**fallback_metadata, **metadata}
        return fallback

    def claim_billing_outbox(
        self,
        *,
        claim_id: str,
        limit: int = 100,
        lease_ttl_s: float = 60.0,
        now: float | None = None,
    ) -> list[dict[str, Any]]:
        return self._billing_outbox.claim(
            claim_id=str(claim_id),
            limit=int(limit),
            lease_ttl_s=float(lease_ttl_s),
            now=now,
        )

    def delete_billing_outbox_claim(
        self,
        *,
        claim_id: str,
        outbox_ids: list[int],
    ) -> dict[str, Any]:
        return self._billing_outbox.delete_claim(
            claim_id=str(claim_id),
            outbox_ids=[int(value) for value in outbox_ids],
        )

    def mark_billing_outbox_claim_failed(
        self,
        *,
        claim_id: str,
        outbox_ids: list[int],
        permanent: bool,
        error: str,
        now: float | None = None,
    ) -> dict[str, Any]:
        return self._billing_outbox.mark_claim_failed(
            claim_id=str(claim_id),
            outbox_ids=[int(value) for value in outbox_ids],
            permanent=bool(permanent),
            error=str(error),
            now=now,
        )

    def billing_outbox_stats(self, *, now: float | None = None) -> dict[str, Any]:
        return self._billing_outbox.stats(now=now)

    def list_tasks_by_metadata(
        self,
        *,
        filters: dict[str, Any] | None = None,
        statuses: list[str] | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        status_values = set(str(value) for value in (statuses or []))
        normalized_filters = dict(filters or {})
        candidate_ids: set[str] | None = None
        if status_values:
            candidate_ids = set()
            for status in status_values:
                candidate_ids.update(self._ids_from_index_prefix(f"{_INDEX_STATUS_PREFIX}{_index_value(status)}:"))
        for key, value in normalized_filters.items():
            ids = set(self._ids_from_index_prefix(f"{_INDEX_META_PREFIX}{_index_value(key)}:{_index_value(value)}:"))
            candidate_ids = ids if candidate_ids is None else candidate_ids & ids
        if candidate_ids is None:
            candidate_ids = set(self._ids_from_index_prefix(_INDEX_CREATED_PREFIX))
        out: list[dict[str, Any]] = []
        for record in self._records_from_ids(sorted(candidate_ids), limit=int(limit)):
            if status_values and str(record.get("status")) not in status_values:
                continue
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            if all(metadata.get(key, record.get(key)) == value for key, value in normalized_filters.items()):
                out.append(dict(record))
                if len(out) >= int(limit):
                    break
        return out

    def _complete_task_direct(
        self,
        *,
        request_id: str,
        status: str,
        result_path: str | None,
        result_checksum: str | None,
        result_size_bytes: int | None,
        error: str | None,
        metadata: dict[str, Any] | None,
        billing_observations: list[dict[str, Any]] | None = None,
        now: float | None,
    ) -> dict[str, Any]:
        """KV implementation of terminal task completion (done or failed)."""
        ts = _now(now)
        with self._lock_for_request(request_id):
            record = self._load(str(request_id))
            old_record = dict(record)
            current_status = str(record.get("status"))
            if current_status in TERMINAL_TASK_STATUSES:
                if (
                    current_status == str(status)
                    and record.get("result_path") == result_path
                    and record.get("result_checksum") == result_checksum
                    and record.get("result_size_bytes") == result_size_bytes
                    and record.get("error") == error
                ):
                    return {"ok": True, "idempotent": True, "record": dict(record)}
                raise TaskStateConflictError("terminal task commit payload mismatch")
            if str(status) == "done" and not _require_staged_success_path(record, result_path):
                raise TaskStateConflictError(f"cannot complete task {status}; staged payload mismatch")
            record["metadata"] = (
                _merge_metadata(record, metadata)
                if str(status) == "done"
                else _merge_metadata_with_abandoned_staged_payload(record, metadata)
            )
            record.update({
                "status": str(status),
                "result_path": result_path,
                "result_checksum": result_checksum,
                "result_size_bytes": result_size_bytes,
                "staged_payload_path": None,
                "staged_payload_checksum": None,
                "staged_payload_size_bytes": None,
                "error": error,
                "lease_id": None,
                "attempt_id": None,
                "lease_expires_at": None,
                "finalizing_until": None,
                "updated_at": ts,
            })
            self._save_indexed(record, old_record=old_record)
            return {"ok": True, "idempotent": False, "record": dict(record)}

    def assign_task(
        self,
        *,
        request_id: str,
        subqueue_id: str,
        scheduler_epoch: int,
        now: float | None = None,
    ) -> TaskMutationResult:
        ts = _now(now)
        with self._lock_for_request(request_id):
            self._assert_scheduler_owner(scheduler_epoch=scheduler_epoch, now=ts)
            record = self._load(str(request_id))
            old_record = dict(record)
            if str(record.get("status")) != "pending":
                raise TaskStateConflictError(f"cannot assign from pending; current status={record.get('status')!r}")
            record.update({"status": "assigned", "subqueue_id": str(subqueue_id), "scheduler_epoch": int(scheduler_epoch), "assigned_at": ts, "updated_at": ts})
            self._save_indexed(record, old_record=old_record)
            return TaskMutationResult.from_wire({"ok": True, "record": dict(record)})

    def claim_task(
        self,
        *,
        request_id: str,
        subqueue_id: str,
        lease_id: str,
        attempt_id: str,
        consumer_id: str,
        scheduler_epoch: int,
        runtime_generation: int,
        lease_ttl_s: float,
        now: float | None = None,
    ) -> TaskMutationResult:
        ts = _now(now)
        expires_at = ts + max(1.0, float(lease_ttl_s))
        with self._lock_for_request(request_id):
            self._assert_scheduler_owner(scheduler_epoch=scheduler_epoch, now=ts)
            record = self._load(str(request_id))
            old_record = dict(record)
            if str(record.get("status")) != "assigned" or str(record.get("subqueue_id")) != str(subqueue_id) or int(record.get("scheduler_epoch") or 0) != int(scheduler_epoch):
                raise TaskStateConflictError(f"cannot claim assigned task; current status={record.get('status')!r}")
            record.update({
                "status": "leased",
                "lease_id": str(lease_id),
                "attempt_id": str(attempt_id),
                "consumer_id": str(consumer_id),
                "scheduler_epoch": int(scheduler_epoch),
                "runtime_generation": int(runtime_generation),
                "leased_at": ts,
                "lease_expires_at": expires_at,
                "updated_at": ts,
            })
            self._save_indexed(record, old_record=old_record)
            return TaskMutationResult.from_wire({"ok": True, "record": dict(record)})

    def renew_lease(
        self,
        *,
        request_id: str,
        lease_id: str,
        attempt_id: str,
        scheduler_epoch: int,
        runtime_generation: int,
        lease_ttl_s: float,
        now: float | None = None,
    ) -> TaskMutationResult:
        ts = _now(now)
        expires_at = ts + max(1.0, float(lease_ttl_s))
        with self._lock_for_request(request_id):
            self._assert_scheduler_owner(scheduler_epoch=scheduler_epoch, now=ts)
            record = self._load(str(request_id))
            old_record = dict(record)
            status = str(record.get("status"))
            if status in TERMINAL_TASK_STATUSES:
                return TaskMutationResult(ok=False, reason=ConflictReason.TERMINAL, record=dict(record))
            if (
                status not in {"leased", "running"}
                or str(record.get("lease_id")) != str(lease_id)
                or str(record.get("attempt_id")) != str(attempt_id)
                or int(record.get("scheduler_epoch") or 0) != int(scheduler_epoch)
                or int(record.get("runtime_generation") or 0) != int(runtime_generation)
            ):
                raise TaskStateConflictError(f"cannot renew lease; current status={record.get('status')!r}")
            record.update({"lease_expires_at": expires_at, "updated_at": ts})
            self._save_indexed(record, old_record=old_record)
            return TaskMutationResult.from_wire({"ok": True, "record": dict(record)})

    def batch_renew_leases(
        self,
        *,
        items: list[dict[str, object]],
    ) -> list[TaskMutationResult]:
        """Batch-renew multiple leases in a single call.

        Each item must contain: request_id, lease_id, attempt_id,
        scheduler_epoch, runtime_generation, lease_ttl_s.
        """
        results: list[TaskMutationResult] = []
        for item in items:
            try:
                result = self.renew_lease(
                    request_id=str(item["request_id"]),
                    lease_id=str(item["lease_id"]),
                    attempt_id=str(item["attempt_id"]),
                    scheduler_epoch=int(item["scheduler_epoch"]),
                    runtime_generation=int(item["runtime_generation"]),
                    lease_ttl_s=float(item["lease_ttl_s"]),
                )
            except Exception as exc:
                result = TaskMutationResult(
                    ok=False,
                    reason=ConflictReason.UNKNOWN,
                    record={"error": str(exc)},
                )
            results.append(result)
        return results

    def begin_finalize(
        self,
        *,
        request_id: str,
        lease_id: str,
        attempt_id: str,
        scheduler_epoch: int,
        runtime_generation: int,
        finalize_ttl_s: float,
        staged_payload_path: str | None = None,
        now: float | None = None,
    ) -> TaskMutationResult:
        ts = _now(now)
        finalizing_until = ts + max(1.0, float(finalize_ttl_s))
        with self._lock_for_request(request_id):
            self._assert_scheduler_owner(scheduler_epoch=scheduler_epoch, now=ts)
            record = self._load(str(request_id))
            old_record = dict(record)
            if (
                str(record.get("status")) not in {"leased", "running"}
                or str(record.get("lease_id")) != str(lease_id)
                or str(record.get("attempt_id")) != str(attempt_id)
                or int(record.get("scheduler_epoch") or 0) != int(scheduler_epoch)
                or int(record.get("runtime_generation") or 0) != int(runtime_generation)
            ):
                raise TaskStateConflictError(f"cannot begin finalize; current status={record.get('status')!r}")
            record["metadata"] = _merge_metadata_with_abandoned_staged_payload(record, new_staged_payload_path=staged_payload_path)
            record.update({
                "status": "finalizing",
                "finalizing_until": finalizing_until,
                "staged_payload_path": staged_payload_path,
                "staged_payload_checksum": None,
                "staged_payload_size_bytes": None,
                "lease_expires_at": max(float(record.get("lease_expires_at") or 0.0), finalizing_until),
                "updated_at": ts,
            })
            self._save_indexed(record, old_record=old_record)
            return TaskMutationResult.from_wire({"ok": True, "record": dict(record)})

    def commit_finalize_success(
        self,
        *,
        request_id: str,
        lease_id: str,
        attempt_id: str,
        scheduler_epoch: int,
        runtime_generation: int,
        result_path: str,
        result_checksum: str,
        result_size_bytes: int,
        billing_observations: list[dict[str, Any]] | None = None,
        now: float | None = None,
    ) -> TaskMutationResult:
        billing_meta: dict[str, Any] = {}
        if billing_observations:
            billing_meta = self._billing_outbox.append_after_terminal_success(
                observations=billing_observations,
                source=str(request_id),
                now=float(now) if now is not None else time.time(),
            )
        result = self._commit_finalize_kv(
            status="done",
            error=None,
            request_id=request_id,
            lease_id=lease_id,
            attempt_id=attempt_id,
            scheduler_epoch=scheduler_epoch,
            runtime_generation=runtime_generation,
            result_path=result_path,
            result_checksum=result_checksum,
            result_size_bytes=result_size_bytes,
            metadata=dict(billing_meta) if billing_meta else None,
            now=now,
        )
        if billing_meta and result.get("ok"):
            record = result.get("record") or {}
            record.setdefault("metadata", {}).update(billing_meta)
        return TaskMutationResult.from_wire(result)

    def commit_finalize_failure(
        self,
        *,
        request_id: str,
        lease_id: str,
        attempt_id: str,
        scheduler_epoch: int,
        runtime_generation: int,
        error: str,
        result_path: str | None = None,
        result_checksum: str | None = None,
        result_size_bytes: int | None = None,
        billing_observations: list[dict[str, Any]] | None = None,
        now: float | None = None,
    ) -> TaskMutationResult:
        result = self._commit_finalize_kv(
            status="failed",
            error=error,
            request_id=request_id,
            lease_id=lease_id,
            attempt_id=attempt_id,
            scheduler_epoch=scheduler_epoch,
            runtime_generation=runtime_generation,
            result_path=result_path,
            result_checksum=result_checksum,
            result_size_bytes=result_size_bytes,
            metadata=None,
            now=now,
        )
        return TaskMutationResult.from_wire(result)

    def requeue_task(
        self,
        *,
        request_id: str,
        scheduler_epoch: int,
        reason: str,
        now: float | None = None,
    ) -> TaskMutationResult:
        ts = _now(now)
        with self._lock_for_request(request_id):
            self._assert_scheduler_owner(scheduler_epoch=scheduler_epoch, now=ts)
            record = self._load(str(request_id))
            old_record = dict(record)
            if str(record.get("status")) in TERMINAL_TASK_STATUSES:
                return TaskMutationResult(ok=False, reason=ConflictReason.TERMINAL, record=dict(record))
            record["metadata"] = _merge_metadata_with_abandoned_staged_payload(record)
            record.update({
                "status": "pending",
                "subqueue_id": None,
                "lease_id": None,
                "attempt_id": None,
                "scheduler_epoch": None,
                "runtime_generation": None,
                "consumer_id": None,
                "assigned_at": None,
                "leased_at": None,
                "lease_expires_at": None,
                "finalizing_until": None,
                "staged_payload_path": None,
                "staged_payload_checksum": None,
                "staged_payload_size_bytes": None,
                "updated_at": ts,
            })
            self._save_indexed(record, old_record=old_record)
            return TaskMutationResult(ok=True, record=dict(record))

    def _commit_finalize_kv(
        self,
        *,
        request_id: str,
        lease_id: str,
        attempt_id: str,
        scheduler_epoch: int,
        runtime_generation: int,
        status: str,
        result_path: str | None = None,
        result_checksum: str | None = None,
        result_size_bytes: int | None = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        ts = _now(now)
        with self._lock_for_request(request_id):
            record = self._load(str(request_id))
            old_record = dict(record)
            if str(record.get("status")) in TERMINAL_TASK_STATUSES:
                if (
                    str(record.get("status")) == str(status)
                    and str(record.get("lease_id")) == str(lease_id)
                    and str(record.get("attempt_id")) == str(attempt_id)
                    and record.get("result_path") == result_path
                    and record.get("result_checksum") == result_checksum
                    and record.get("result_size_bytes") == result_size_bytes
                    and record.get("error") == error
                ):
                    return {"ok": True, "idempotent": True, "record": dict(record)}
                raise TaskStateConflictError("terminal task commit payload mismatch")
            if (
                str(record.get("status")) != "finalizing"
                or str(record.get("lease_id")) != str(lease_id)
                or str(record.get("attempt_id")) != str(attempt_id)
                or int(record.get("scheduler_epoch") or 0) != int(scheduler_epoch)
                or int(record.get("runtime_generation") or 0) != int(runtime_generation)
            ):
                raise TaskStateConflictError(f"cannot commit finalize {status}; current status={record.get('status')!r}")
            if str(status) == "done" and not _require_staged_success_path(record, result_path):
                raise TaskStateConflictError(f"cannot commit finalize {status}; staged payload mismatch")
            record["metadata"] = (
                _merge_metadata(record, metadata)
                if str(status) == "done"
                else _merge_metadata_with_abandoned_staged_payload(record, metadata)
            )
            record.update({
                "status": str(status),
                "result_path": result_path,
                "result_checksum": result_checksum,
                "result_size_bytes": result_size_bytes,
                "staged_payload_path": None,
                "staged_payload_checksum": None,
                "staged_payload_size_bytes": None,
                "error": error,
                "finalizing_until": None,
                "updated_at": ts,
            })
            self._save_indexed(record, old_record=old_record)
            return {"ok": True, "idempotent": False, "record": dict(record)}

    def list_active_tasks(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        ids: list[str] = []
        remaining = None if limit is None else max(0, int(limit))
        if remaining == 0:
            return []
        for status in ("pending", "queued", "assigned", "leased", "running", "finalizing"):
            status_limit = remaining
            ids.extend(
                self._ids_from_index_prefix(
                    f"{_INDEX_STATUS_PREFIX}{_index_value(status)}:",
                    limit=status_limit,
                )
            )
            if remaining is not None:
                remaining = max(0, int(limit) - len(ids))
                if remaining == 0:
                    break
        return self._records_from_ids(ids, limit=limit)

    def future_metrics_stats(self, *, now: float | None = None) -> dict[str, Any]:
        ts = _now(now)
        by_status: dict[str, int] = {}
        by_op: dict[str, dict[str, int]] = {}
        pending_ages: list[float] = []
        done_ages: list[float] = []
        refs = 0
        meta = 0
        records_scanned = 0

        def _consume(record: dict[str, Any]) -> None:
            nonlocal refs, meta, records_scanned
            records_scanned += 1
            status = str(record.get("status") or "unknown")
            op = str(record.get("op") or "unknown")
            by_status[status] = by_status.get(status, 0) + 1
            bucket = by_op.setdefault(op, {"pending": 0, "results": 0, "errors": 0})
            if status in ACTIVE_TASK_STATUSES:
                bucket["pending"] += 1
                pending_ages.append(max(0.0, ts - float(record.get("created_at") or ts)))
            elif status in {"done", "retrieved"}:
                bucket["results"] += 1
                done_ages.append(max(0.0, ts - float(record.get("updated_at") or ts)))
            elif status == "failed":
                bucket["errors"] += 1
                done_ages.append(max(0.0, ts - float(record.get("updated_at") or ts)))
            if record.get("result_path"):
                refs += 1
            if record.get("metadata"):
                meta += 1

        seen: set[str] = set()
        for status in ("pending", "queued", "assigned", "leased", "running", "finalizing", "done", "retrieved", "failed", "expired"):
            for request_id in self._ids_from_index_prefix(f"{_INDEX_STATUS_PREFIX}{_index_value(status)}:"):
                if request_id in seen:
                    continue
                seen.add(request_id)
                try:
                    _consume(self.get_task(request_id))
                except TaskStateNotFoundError:
                    continue
        for record in self._records_from_ids(self._ids_from_index_prefix(_INDEX_RESULT_PREFIX)):
            request_id = str(record.get("request_id") or "")
            if request_id in seen:
                continue
            seen.add(request_id)
            _consume(record)

        errors = int(by_status.get("failed", 0))
        return {
            "pending": sum(by_status.get(status, 0) for status in ACTIVE_TASK_STATUSES),
            "results": sum(by_status.get(status, 0) for status in {"done", "retrieved"}),
            "errors": errors,
            "refs": refs,
            "meta": meta,
            "expired": int(by_status.get("expired", 0)),
            "retrieved": int(by_status.get("retrieved", 0)),
            "execution_timeout_s": float(server_config.task_pending_ttl_s),
            "queue_timeout_s": float(getattr(server_config, "retrieve_future_wait_timeout_s", 20.0)),
            "result_ttl_s": float(server_config.task_result_ttl_s),
            "tombstone_ttl_s": float(server_config.task_tombstone_ttl_s),
            "by_op": by_op,
            "age_stats": {
                "oldest_pending_s": max(pending_ages) if pending_ages else 0.0,
                "oldest_done_s": max(done_ages) if done_ages else 0.0,
                "avg_pending_s": sum(pending_ages) / len(pending_ages) if pending_ages else 0.0,
                "avg_done_s": sum(done_ages) / len(done_ages) if done_ages else 0.0,
            },
            "payload_stats": {
                "result_refs_count": refs,
                "errors_count": errors,
                "refs_count": refs,
                "records_scanned": records_scanned,
            },
            "timeout_counts": future_timeout_metrics_snapshot(),
        }

    def list_expired_leases(self, *, now: float | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        ts = _now(now)
        out: list[dict[str, Any]] = []
        for request_id in self._ids_from_index_prefix(_INDEX_LEASE_PREFIX):
            if limit is not None and len(out) >= max(0, int(limit)):
                break
            try:
                record = self.get_task(request_id)
            except TaskStateNotFoundError:
                continue
            if str(record.get("status")) not in {"leased", "finalizing"}:
                continue
            if float(record.get("finalizing_until") or record.get("lease_expires_at") or 0.0) > ts:
                continue
            out.append(dict(record))
        return out

    def get_task(self, request_id: str) -> dict[str, Any]:
        return dict(self._load(str(request_id)))

    def _get_row(self, request_id: str) -> dict[str, Any]:
        return self.get_task(request_id)

    def _raise_task_transition_error(self, request_id: str, action: str) -> None:
        record = self.get_task(request_id)
        raise TaskStateConflictError(f"cannot {action}; current status={record.get('status')!r}")

    def _row_to_record(self, row: Any) -> dict[str, Any]:
        if isinstance(row, dict):
            return row
        return dict(row)

def _ray_namespace() -> str:
    v = env_nonempty(os.environ, "MINT_RAY_NAMESPACE")
    if v:
        return v
    try:
        from mint_server.config import RAY_NAMESPACE

        return RAY_NAMESPACE
    except Exception:
        return "mint"


def _ray_task_state_store_actor_name() -> str:
    return str(
        os.environ.get("MINT_TASK_STATE_STORE_ACTOR_NAME")
        or getattr(server_config, "task_state_store_actor_name", "mint_task_state_store")
    )


def _task_state_store_db_path() -> str:
    return str(
        os.environ.get("MINT_TASK_STATE_STORE_DB_PATH")
        or getattr(
            server_config,
            "task_state_store_db_path",
            "/vePFS-Mindverse/share/mint/dev/data/task-state/task_state.sqlite3",
        )
    )


def _task_hot_kv_store_db_path(task_state_db_path: str | None = None) -> str:
    configured = os.environ.get("MINT_TASK_HOT_KV_STORE_DB_PATH") or getattr(
        server_config,
        "task_hot_kv_store_db_path",
        None,
    )
    if configured:
        return str(configured)
    base = Path(str(task_state_db_path or _task_state_store_db_path()))
    return str(base.parent.parent / "task-hot-kv" / "task_hot.rocksdb")


def _future_state_store_db_path(task_state_db_path: str | None = None) -> str:
    configured = os.environ.get("MINT_FUTURE_STATE_STORE_DB_PATH") or getattr(
        server_config,
        "future_state_store_db_path",
        None,
    )
    if configured:
        return str(configured)
    base = Path(str(task_state_db_path or _task_state_store_db_path()))
    return str(base.parent.parent / "future-state" / "futures.rocksdb")


class _TaskStateStoreActor:
    def __init__(self, db_path: str | None = None) -> None:
        try:
            from mint_server.logging_context import init_actor_observability

            init_actor_observability()
        except Exception:
            pass
        self._started_at = time.time()
        self._lock = threading.RLock()
        self._store = TaskStateStore(db_path or _task_state_store_db_path())
        self._watchers: dict[str, list[threading.Event]] = {}
        self._watcher_count = 0
        self._watcher_limit = max(1, int(os.environ.get("MINT_TASK_STATE_STORE_WATCHER_MAX", "8192")))
        self._stats_cache: dict[str, Any] | None = None
        self._stats_cache_at = 0.0
        self._stats_cache_ttl_s = max(0.0, float(os.environ.get("MINT_TASK_STATE_STORE_STATS_CACHE_TTL_S", "5")))
        self._stats_lock = threading.Lock()
        self._otel_enabled = False
        self._otel_error: str | None = None
        self._init_otel_metrics()

    def close(self) -> None:
        self._store.close()

    def _read_task_or_none(self, request_id: str) -> dict[str, Any] | None:
        try:
            return self._store.get_task(str(request_id))
        except TaskStateNotFoundError:
            return None

    def _read_future_task_or_none(self, request_id: str) -> dict[str, Any] | None:
        return self._read_task_or_none(request_id)

    @staticmethod
    def _record_changed(
        record: dict[str, Any] | None,
        *,
        baseline_status: str,
        baseline_updated_at: float,
        terminal_only: bool = False,
    ) -> bool:
        if record is None:
            return True
        if str(record.get("status") or "") in TERMINAL_TASK_STATUSES:
            return True
        if terminal_only:
            return False
        if str(record.get("status") or "") != str(baseline_status):
            return True
        try:
            return float(record.get("updated_at") or 0.0) > float(baseline_updated_at)
        except Exception:
            return True

    def _add_watcher(
        self,
        request_id: str,
        event: threading.Event,
    ) -> bool:
        with self._lock:
            if self._watcher_count >= self._watcher_limit:
                return False
            self._watchers.setdefault(str(request_id), []).append(event)
            self._watcher_count += 1
            return True

    def _remove_watcher(self, request_id: str, event: threading.Event) -> None:
        with self._lock:
            waiters = self._watchers.get(str(request_id))
            if not waiters:
                return
            kept: list[threading.Event] = []
            removed = 0
            for waiter in waiters:
                if waiter is event:
                    removed += 1
                else:
                    kept.append(waiter)
            if kept:
                self._watchers[str(request_id)] = kept
            else:
                self._watchers.pop(str(request_id), None)
            self._watcher_count = max(0, self._watcher_count - removed)

    def _invalidate_stats_cache(self) -> None:
        self._stats_cache = None
        self._stats_cache_at = 0.0

    def _notify_task_changed(self, request_id: str | None) -> None:
        self._invalidate_stats_cache()
        if request_id is None:
            return
        with self._lock:
            waiters = self._watchers.pop(str(request_id), [])
            if not waiters:
                return
            self._watcher_count = max(0, self._watcher_count - len(waiters))
        for event in waiters:
            event.set()

    def wait_task_status_change(
        self,
        *,
        request_id: str,
        timeout_s: float,
        observed_status: str | None = None,
        observed_updated_at: float | None = None,
        terminal_only: bool = False,
    ) -> dict[str, Any]:
        request_id = str(request_id)
        timeout_s = max(0.0, float(timeout_s))
        record = self._read_task_or_none(request_id)
        if record is None:
            return {"changed": True, "missing": True, "request_id": request_id}

        baseline_status = str(observed_status or record.get("status") or "")
        try:
            baseline_updated_at = float(observed_updated_at if observed_updated_at is not None else record.get("updated_at") or 0.0)
        except Exception:
            baseline_updated_at = 0.0

        if self._record_changed(
            record,
            baseline_status=baseline_status,
            baseline_updated_at=baseline_updated_at,
            terminal_only=bool(terminal_only),
        ):
            return {"changed": True, "record": record}
        if timeout_s <= 0:
            return {"changed": False, "timeout": True, "record": record}

        deadline = time.monotonic() + timeout_s
        latest = record
        while True:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                return {"changed": False, "timeout": True, "record": latest}
            event = threading.Event()
            if not self._add_watcher(request_id, event):
                return {
                    "changed": False,
                    "watch_skipped": True,
                    "reason": "watcher_limit",
                    "record": latest,
                }
            try:
                latest = self._read_task_or_none(request_id)
                if self._record_changed(
                    latest,
                    baseline_status=baseline_status,
                    baseline_updated_at=baseline_updated_at,
                    terminal_only=bool(terminal_only),
                ):
                    if latest is None:
                        return {"changed": True, "missing": True, "request_id": request_id}
                    return {"changed": True, "record": latest}
                try:
                    signaled = event.wait(timeout=max(0.0, remaining_s))
                except Exception:
                    signaled = False
                if not signaled:
                    latest = self._read_task_or_none(request_id)
                    if self._record_changed(
                        latest,
                        baseline_status=baseline_status,
                        baseline_updated_at=baseline_updated_at,
                        terminal_only=bool(terminal_only),
                    ):
                        if latest is None:
                            return {"changed": True, "missing": True, "request_id": request_id}
                        return {"changed": True, "record": latest}
                    return {"changed": False, "timeout": True, "record": latest or record}

                latest = self._read_task_or_none(request_id)
                if latest is None:
                    return {"changed": True, "missing": True, "request_id": request_id}
                if self._record_changed(
                    latest,
                    baseline_status=baseline_status,
                    baseline_updated_at=baseline_updated_at,
                    terminal_only=bool(terminal_only),
                ):
                    return {"changed": True, "record": latest}
            finally:
                self._remove_watcher(request_id, event)

    def future_wait_task_status_change(
        self,
        *,
        request_id: str,
        timeout_s: float,
        observed_status: str | None = None,
        observed_updated_at: float | None = None,
        terminal_only: bool = False,
    ) -> dict[str, Any]:
        request_id = str(request_id)
        timeout_s = max(0.0, float(timeout_s))
        record = self._read_future_task_or_none(request_id)
        if record is None:
            return {"changed": True, "missing": True, "request_id": request_id}

        baseline_status = str(observed_status or record.get("status") or "")
        try:
            baseline_updated_at = float(observed_updated_at if observed_updated_at is not None else record.get("updated_at") or 0.0)
        except Exception:
            baseline_updated_at = 0.0

        if self._record_changed(
            record,
            baseline_status=baseline_status,
            baseline_updated_at=baseline_updated_at,
            terminal_only=bool(terminal_only),
        ):
            return {"changed": True, "record": record}
        if timeout_s <= 0:
            return {"changed": False, "timeout": True, "record": record}

        deadline = time.monotonic() + timeout_s
        latest = record
        while True:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                return {"changed": False, "timeout": True, "record": latest}
            event = threading.Event()
            if not self._add_watcher(request_id, event):
                return {
                    "changed": False,
                    "watch_skipped": True,
                    "reason": "watcher_limit",
                    "record": latest,
                }
            try:
                latest = self._read_future_task_or_none(request_id)
                if self._record_changed(
                    latest,
                    baseline_status=baseline_status,
                    baseline_updated_at=baseline_updated_at,
                    terminal_only=bool(terminal_only),
                ):
                    if latest is None:
                        return {"changed": True, "missing": True, "request_id": request_id}
                    return {"changed": True, "record": latest}
                signaled = event.wait(timeout=max(0.0, remaining_s))
                if not signaled:
                    latest = self._read_future_task_or_none(request_id)
                    if self._record_changed(
                        latest,
                        baseline_status=baseline_status,
                        baseline_updated_at=baseline_updated_at,
                        terminal_only=bool(terminal_only),
                    ):
                        if latest is None:
                            return {"changed": True, "missing": True, "request_id": request_id}
                        return {"changed": True, "record": latest}
                    return {"changed": False, "timeout": True, "record": latest or record}
            finally:
                self._remove_watcher(request_id, event)

    def future_ping(self) -> dict[str, Any]:
        out = self._store.ping()
        return {
            **out,
            "actor_name": _ray_task_state_store_actor_name(),
            "namespace": _ray_namespace(),
            "store": "future_state_store",
            "started_at": self._started_at,
        }

    def future_stats(self) -> dict[str, Any]:
        future_stats = self._store.future_metrics_stats()
        active = self._store.list_active_tasks()
        by_status: dict[str, int] = {}
        for record in active:
            status = str(record.get("status") or "unknown")
            by_status[status] = by_status.get(status, 0) + 1
        return {
            "actor_name": _ray_task_state_store_actor_name(),
            "namespace": _ray_namespace(),
            "store": "future_state_store",
            "db_path": self._store.db_path,
            "started_at": self._started_at,
            "active_tasks": len(active),
            "active_by_status": by_status,
            "watchers": self._watcher_count,
            **future_stats,
            "task_future_reaper": task_future_reaper_metrics_snapshot(),
        }

    def _future_call_and_notify(self, method: str, **kwargs: Any) -> Any:
        out = getattr(self._store, method)(**kwargs)
        self._notify_task_changed(kwargs.get("request_id"))
        return out

    def _future_append_billing_after_terminal_success(
        self,
        *,
        request_id: str,
        billing_observations: list[dict[str, Any]] | None,
        source: str,
        now: float | None = None,
        out: dict[str, Any],
    ) -> dict[str, Any]:
        if not billing_observations:
            return out
        ts = _now(now)
        billing_metadata = self._store._append_billing_outbox_after_terminal_success(
            observations=billing_observations,
            source=source,
            now=ts,
        )
        if billing_metadata:
            try:
                updated = self._store.update_task_metadata(
                    request_id=str(request_id),
                    metadata=billing_metadata,
                    now=ts,
                )
                if isinstance(updated, TaskMutationResult) and isinstance(updated.record, dict):
                    out = {**dict(out), "record": updated.record}
                elif isinstance(updated, dict) and isinstance(updated.get("record"), dict):
                    out = {**dict(out), "record": updated["record"]}
            except Exception:
                _inc_billing_metric("write_error", 1)
                record = out.get("record") if isinstance(out.get("record"), dict) else None
                if isinstance(record, dict):
                    out = {**dict(out), "record": {**record, "metadata": {**dict(record.get("metadata") or {}), **billing_metadata}}}
        self._notify_task_changed(request_id)
        return out

    def future_acquire_scheduler_owner(self, **kwargs: Any) -> dict[str, Any]:
        return self._store.acquire_scheduler_owner(**kwargs)

    def future_renew_scheduler_owner(self, **kwargs: Any) -> dict[str, Any]:
        return self._store.renew_scheduler_owner(**kwargs)

    def future_create_task(self, **kwargs: Any) -> dict[str, Any]:
        return self._future_call_and_notify("create_task", **kwargs)

    def future_ensure_task(self, **kwargs: Any) -> dict[str, Any]:
        return self._future_call_and_notify("ensure_task", **kwargs)

    def future_update_task_metadata(self, **kwargs: Any) -> dict[str, Any]:
        return self._future_call_and_notify("update_task_metadata", **kwargs)

    def future_stage_payload(self, **kwargs: Any) -> dict[str, Any]:
        return self._future_call_and_notify("stage_payload", **kwargs)

    def future_complete_task_success(self, **kwargs: Any) -> dict[str, Any]:
        billing_observations = kwargs.pop("billing_observations", None)
        out = self._future_call_and_notify("complete_task_success", **kwargs)
        return self._future_append_billing_after_terminal_success(
            request_id=str(kwargs.get("request_id")),
            billing_observations=billing_observations,
            source="task_terminal",
            now=kwargs.get("now"),
            out=out,
        )

    def future_complete_task_failure(self, **kwargs: Any) -> dict[str, Any]:
        return self._future_call_and_notify("complete_task_failure", **kwargs)

    def future_mark_task_retrieved(self, **kwargs: Any) -> dict[str, Any]:
        return self._future_call_and_notify("mark_task_retrieved", **kwargs)

    def future_forget_task(self, **kwargs: Any) -> dict[str, Any]:
        return self._future_call_and_notify("forget_task", **kwargs)

    def future_expire_active_tasks(self, **kwargs: Any) -> list[str]:
        out = self._store.expire_active_tasks(**kwargs)
        for request_id in out:
            self._notify_task_changed(str(request_id))
        return out

    def future_list_terminal_payloads_for_eviction(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self._store.list_terminal_payloads_for_eviction(**kwargs)

    def future_mark_payload_evicted(self, **kwargs: Any) -> dict[str, Any]:
        return self._future_call_and_notify("mark_payload_evicted", **kwargs)

    def future_delete_expired_tombstones(self, **kwargs: Any) -> list[str]:
        out = self._store.delete_expired_tombstones(**kwargs)
        for request_id in out:
            self._notify_task_changed(str(request_id))
        return out

    def future_record_payload_evict_error(self, **kwargs: Any) -> dict[str, Any]:
        out = self._store.record_payload_evict_error(**kwargs)
        self._invalidate_stats_cache()
        return out

    def future_list_staged_payloads_for_gc(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self._store.list_staged_payloads_for_gc(**kwargs)

    def future_mark_staged_payload_gc_deleted(self, **kwargs: Any) -> dict[str, Any]:
        return self._future_call_and_notify("mark_staged_payload_gc_deleted", **kwargs)

    def future_list_tasks_by_metadata(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self._store.list_tasks_by_metadata(**kwargs)

    def future_assign_task(self, **kwargs: Any) -> dict[str, Any]:
        return self._future_call_and_notify("assign_task", **kwargs)

    def future_claim_task(self, **kwargs: Any) -> dict[str, Any]:
        return self._future_call_and_notify("claim_task", **kwargs)

    def future_renew_lease(self, **kwargs: Any) -> dict[str, Any]:
        return self._future_call_and_notify("renew_lease", **kwargs)

    def future_begin_finalize(self, **kwargs: Any) -> dict[str, Any]:
        return self._future_call_and_notify("begin_finalize", **kwargs)

    def future_commit_finalize_success(self, **kwargs: Any) -> dict[str, Any]:
        billing_observations = kwargs.pop("billing_observations", None)
        out = self._future_call_and_notify("commit_finalize_success", **kwargs)
        return self._future_append_billing_after_terminal_success(
            request_id=str(kwargs.get("request_id")),
            billing_observations=billing_observations,
            source="model_work_terminal",
            now=kwargs.get("now"),
            out=out,
        )

    def future_commit_finalize_failure(self, **kwargs: Any) -> dict[str, Any]:
        return self._future_call_and_notify("commit_finalize_failure", **kwargs)

    def future_requeue_task(self, **kwargs: Any) -> dict[str, Any]:
        return self._future_call_and_notify("requeue_task", **kwargs)

    def future_list_active_tasks(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self._store.list_active_tasks(**kwargs)

    def future_list_expired_leases(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self._store.list_expired_leases(**kwargs)

    def future_get_task(self, request_id: str) -> dict[str, Any]:
        return self._store.get_task(request_id)

    def stats(self) -> dict[str, Any]:
        now = time.monotonic()
        cached = self._stats_cache
        if cached is not None and now - self._stats_cache_at <= self._stats_cache_ttl_s:
            _record_task_state_stats_metric(duration_ms=0.0, cache_hit=True)
            return dict(cached)
        with self._stats_lock:
            started = time.perf_counter()
            now = time.monotonic()
            cached = self._stats_cache
            if cached is not None and now - self._stats_cache_at <= self._stats_cache_ttl_s:
                _record_task_state_stats_metric(duration_ms=0.0, cache_hit=True)
                return dict(cached)
            future_stats = self._store.future_metrics_stats()
            by_status = {
                status: int(future_stats.get(status) or 0)
                for status in ACTIVE_TASK_STATUSES
                if int(future_stats.get(status) or 0) > 0
            }
            active_tasks = sum(by_status.values())
            out = {
                "actor_name": _ray_task_state_store_actor_name(),
                "namespace": _ray_namespace(),
                "db_path": self._store.db_path,
                "started_at": self._started_at,
                "active_tasks": int(active_tasks),
                "active_by_status": by_status,
                "watchers": self._watcher_count,
                **future_stats,
                "task_future_reaper": task_future_reaper_metrics_snapshot(),
                "billing_outbox": self._store.billing_outbox_stats(),
                "task_state_rpc": task_state_rpc_metrics_snapshot(),
            }
            stats_duration_ms = (time.perf_counter() - started) * 1000.0
            _record_task_state_stats_metric(duration_ms=stats_duration_ms, cache_hit=False)
            out["task_state_stats"] = task_state_stats_metrics_snapshot()
            self._stats_cache = dict(out)
            self._stats_cache_at = time.monotonic()
            return out

    def _init_otel_metrics(self) -> None:
        endpoint = (os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or "").strip()
        if not endpoint:
            return
        try:
            from opentelemetry import metrics
            from opentelemetry.metrics import Observation

            meter = metrics.get_meter("mint.task_state_store")

            def _attrs(**extra: object) -> dict[str, str]:
                attrs = _otel_metric_attrs()
                for key, value in extra.items():
                    text = str(value if value is not None else "").strip()
                    if text:
                        attrs[key] = text
                return attrs

            def _gauge(name: str, callback, *, unit: str | None = None) -> None:
                kwargs: dict[str, Any] = {"callbacks": [callback]}
                if unit:
                    kwargs["unit"] = unit
                meter.create_observable_gauge(name, **kwargs)

            def _scalar(field: str):
                def _callback(_options):
                    value = _metric_number(self.stats().get(field))
                    if value is None:
                        return []
                    return [Observation(value, _attrs())]

                return _callback

            for key in (
                "refs",
                "meta",
                "expired",
                "retrieved",
                "execution_timeout_s",
                "queue_timeout_s",
                "result_ttl_s",
                "tombstone_ttl_s",
            ):
                _gauge(f"mint_task_futures_{key}", _scalar(key), unit="s" if key.endswith("_s") else None)

            def _future_count(field: str):
                def _callback(_options):
                    stats = self.stats()
                    observations = []
                    value = _metric_number(stats.get(field))
                    if value is not None:
                        observations.append(Observation(value, _attrs()))
                    by_op = stats.get("by_op")
                    if not isinstance(by_op, dict):
                        return observations
                    for op, rec in sorted(by_op.items()):
                        if not isinstance(rec, dict):
                            continue
                        op_value = _metric_number(rec.get(field))
                        if op_value is None:
                            continue
                        observations.append(Observation(op_value, _attrs(op=op)))
                    return observations

                return _callback

            _gauge("mint_task_futures_pending", _future_count("pending"))
            _gauge("mint_task_futures_results", _future_count("results"))
            _gauge("mint_task_futures_errors", _future_count("errors"))

            def _nested_scalar(section: str, field: str):
                def _callback(_options):
                    section_value = self.stats().get(section)
                    if not isinstance(section_value, dict):
                        return []
                    value = _metric_number(section_value.get(field))
                    if value is None:
                        return []
                    return [Observation(value, _attrs())]

                return _callback

            for field in ("oldest_pending_s", "oldest_done_s", "avg_pending_s", "avg_done_s"):
                _gauge(f"mint_task_futures_{field}", _nested_scalar("age_stats", field), unit="s")
            _gauge("mint_task_futures_result_refs_count", _nested_scalar("payload_stats", "result_refs_count"))
            _gauge("mint_task_futures_errors_count", _nested_scalar("payload_stats", "errors_count"))
            _gauge("mint_task_futures_refs_count", _nested_scalar("payload_stats", "refs_count"))

            def _future_timeouts(_options):
                timeout_counts = self.stats().get("timeout_counts")
                if not isinstance(timeout_counts, dict):
                    return []
                observations = []
                for kind in ("queue", "execution", "total"):
                    value = _metric_number(timeout_counts.get(kind))
                    if value is not None:
                        observations.append(Observation(value, _attrs(kind=kind)))
                by_op = timeout_counts.get("by_op")
                if isinstance(by_op, dict):
                    for op, rec in sorted(by_op.items()):
                        if not isinstance(rec, dict):
                            continue
                        for kind in ("queue", "execution", "total"):
                            value = _metric_number(rec.get(kind))
                            if value is not None:
                                observations.append(Observation(value, _attrs(op=op, kind=kind)))
                return observations

            _gauge("mint_task_futures_timeouts_total", _future_timeouts)

            def _billing_status(field: str):
                def _callback(_options):
                    billing = self.stats().get("billing_outbox")
                    by_status = billing.get("by_status") if isinstance(billing, dict) else None
                    if not isinstance(by_status, dict):
                        return []
                    observations = []
                    for status, rec in sorted(by_status.items()):
                        if not isinstance(rec, dict):
                            continue
                        value = _metric_number(rec.get(field))
                        if value is None:
                            continue
                        observations.append(Observation(value, _attrs(status=status)))
                    return observations

                return _callback

            _gauge("mint_billing_outbox_rows", _billing_status("rows"))
            _gauge("mint_billing_outbox_oldest_age_s", _billing_status("oldest_age_s"), unit="s")

            def _billing_metric(field: str, labels: dict[str, str] | None = None):
                def _callback(_options):
                    billing = self.stats().get("billing_outbox")
                    metrics_map = billing.get("metrics") if isinstance(billing, dict) else None
                    if not isinstance(metrics_map, dict):
                        return []
                    value = _metric_number(metrics_map.get(field))
                    if value is None:
                        return []
                    return [Observation(value, _attrs(**dict(labels or {})))]

                return _callback

            def _billing_metric_by_result(mapping: tuple[tuple[str, str], ...]):
                def _callback(_options):
                    billing = self.stats().get("billing_outbox")
                    metrics_map = billing.get("metrics") if isinstance(billing, dict) else None
                    if not isinstance(metrics_map, dict):
                        return []
                    observations = []
                    for result, field in mapping:
                        value = _metric_number(metrics_map.get(field))
                        if value is None:
                            continue
                        observations.append(Observation(value, _attrs(result=result)))
                    return observations

                return _callback

            _gauge(
                "mint_billing_outbox_flush_attempts_total",
                _billing_metric_by_result(
                    (
                        ("success", "flush_success"),
                        ("transient_error", "flush_transient_error"),
                        ("permanent_error", "flush_permanent_error"),
                    )
                ),
            )
            _gauge(
                "mint_billing_outbox_events_total",
                _billing_metric_by_result(
                    (
                        ("inserted", "event_inserted"),
                        ("conflict", "event_conflict"),
                        ("failed", "event_failed"),
                    )
                ),
            )
            _gauge("mint_billing_outbox_write_errors_total", _billing_metric("write_error"))
            _gauge("mint_billing_outbox_conflict_total", _billing_metric("outbox_conflict"))
            _gauge(
                "mint_billing_observation_skipped_total",
                _billing_metric("skipped_missing_billing_context", {"reason": "missing_billing_context"}),
            )

            def _rpc_scalar(field: str):
                def _callback(_options):
                    rpc = self.stats().get("task_state_rpc")
                    if not isinstance(rpc, dict):
                        return []
                    value = _metric_number(rpc.get(field))
                    if value is None:
                        return []
                    return [Observation(value, _attrs())]

                return _callback

            def _rpc_method_field(field: str):
                def _callback(_options):
                    rpc = self.stats().get("task_state_rpc")
                    by_method = rpc.get("by_method") if isinstance(rpc, dict) else None
                    if not isinstance(by_method, dict):
                        return []
                    observations = []
                    for method, rec in sorted(by_method.items()):
                        if not isinstance(rec, dict):
                            continue
                        value = _metric_number(rec.get(field))
                        if value is None:
                            continue
                        observations.append(Observation(value, _attrs(method=method)))
                    return observations

                return _callback

            def _stats_metric(field: str):
                def _callback(_options):
                    metrics_map = self.stats().get("task_state_stats")
                    if not isinstance(metrics_map, dict):
                        return []
                    value = _metric_number(metrics_map.get(field))
                    if value is None:
                        return []
                    return [Observation(value, _attrs())]

                return _callback

            _gauge("mint_task_state_store_rpc_inflight", _rpc_scalar("inflight"))
            _gauge("mint_task_state_store_rpc_total", _rpc_method_field("total"))
            _gauge("mint_task_state_store_rpc_errors_total", _rpc_method_field("error"))
            _gauge("mint_task_state_store_rpc_last_duration_ms", _rpc_method_field("last_duration_ms"), unit="ms")
            _gauge("mint_task_state_store_rpc_max_duration_ms", _rpc_method_field("max_duration_ms"), unit="ms")
            _gauge("mint_task_state_store_stats_calls_total", _stats_metric("calls"))
            _gauge("mint_task_state_store_stats_cache_hits_total", _stats_metric("cache_hits"))
            _gauge("mint_task_state_store_stats_last_duration_ms", _stats_metric("last_duration_ms"), unit="ms")
            _gauge("mint_task_state_store_stats_max_duration_ms", _stats_metric("max_duration_ms"), unit="ms")
            self._otel_enabled = True
        except Exception as e:
            self._otel_error = f"{type(e).__name__}: {e}"

    def ping(self) -> dict[str, Any]:
        return {
            "ok": True,
            "actor_name": _ray_task_state_store_actor_name(),
            "namespace": _ray_namespace(),
            "started_at": self._started_at,
        }

    def integrity_check(self) -> str:
        return self._store.integrity_check()

    def acquire_scheduler_owner(self, **kwargs: Any) -> dict[str, Any]:
        return _wire_result(self._store.acquire_scheduler_owner(**kwargs))

    def renew_scheduler_owner(self, **kwargs: Any) -> dict[str, Any]:
        return _wire_result(self._store.renew_scheduler_owner(**kwargs))

    def create_task(self, **kwargs: Any) -> dict[str, Any]:
        out = self._store.create_task(**kwargs)
        self._notify_task_changed(kwargs.get("request_id"))
        return _wire_result(out)

    def ensure_task(self, **kwargs: Any) -> dict[str, Any]:
        out = self._store.ensure_task(**kwargs)
        self._notify_task_changed(kwargs.get("request_id"))
        return out

    def update_task_metadata(self, **kwargs: Any) -> dict[str, Any]:
        out = self._store.update_task_metadata(**kwargs)
        self._notify_task_changed(kwargs.get("request_id"))
        return _wire_result(out)

    def complete_task_success(self, **kwargs: Any) -> dict[str, Any]:
        out = self._store.complete_task_success(**kwargs)
        self._notify_task_changed(kwargs.get("request_id"))
        return out

    def complete_task_failure(self, **kwargs: Any) -> dict[str, Any]:
        out = self._store.complete_task_failure(**kwargs)
        self._notify_task_changed(kwargs.get("request_id"))
        return _wire_result(out)

    def mark_task_retrieved(self, **kwargs: Any) -> dict[str, Any]:
        out = self._store.mark_task_retrieved(**kwargs)
        self._notify_task_changed(kwargs.get("request_id"))
        return out

    def forget_task(self, **kwargs: Any) -> dict[str, Any]:
        out = self._store.forget_task(**kwargs)
        self._notify_task_changed(kwargs.get("request_id"))
        return _wire_result(out)

    def expire_active_tasks(self, **kwargs: Any) -> list[str]:
        out = self._store.expire_active_tasks(**kwargs)
        for request_id in out:
            self._notify_task_changed(str(request_id))
        return out

    def list_terminal_payloads_for_eviction(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self._store.list_terminal_payloads_for_eviction(**kwargs)

    def mark_payload_evicted(self, **kwargs: Any) -> dict[str, Any]:
        out = self._store.mark_payload_evicted(**kwargs)
        self._notify_task_changed(kwargs.get("request_id"))
        return out

    def delete_expired_tombstones(self, **kwargs: Any) -> list[str]:
        out = self._store.delete_expired_tombstones(**kwargs)
        for request_id in out:
            self._notify_task_changed(str(request_id))
        return out

    def record_payload_evict_error(self, **kwargs: Any) -> dict[str, Any]:
        out = self._store.record_payload_evict_error(**kwargs)
        self._invalidate_stats_cache()
        return out

    def record_billing_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        _inc_billing_metrics(dict(metrics or {}))
        self._invalidate_stats_cache()
        return {"ok": True, "metrics": billing_metrics_snapshot()}

    def append_billing_outbox(self, **kwargs: Any) -> dict[str, Any]:
        out = self._store.append_billing_outbox(**kwargs)
        self._invalidate_stats_cache()
        return out

    def claim_billing_outbox(self, **kwargs: Any) -> list[dict[str, Any]]:
        out = self._store.claim_billing_outbox(**kwargs)
        self._invalidate_stats_cache()
        return out

    def delete_billing_outbox_claim(self, **kwargs: Any) -> dict[str, Any]:
        out = self._store.delete_billing_outbox_claim(**kwargs)
        self._invalidate_stats_cache()
        return out

    def mark_billing_outbox_claim_failed(self, **kwargs: Any) -> dict[str, Any]:
        out = self._store.mark_billing_outbox_claim_failed(**kwargs)
        self._invalidate_stats_cache()
        return out

    def billing_outbox_stats(self, **kwargs: Any) -> dict[str, Any]:
        return self._store.billing_outbox_stats(**kwargs)

    def list_staged_payloads_for_gc(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self._store.list_staged_payloads_for_gc(**kwargs)

    def mark_staged_payload_gc_deleted(self, **kwargs: Any) -> dict[str, Any]:
        out = self._store.mark_staged_payload_gc_deleted(**kwargs)
        self._notify_task_changed(kwargs.get("request_id"))
        return out

    def list_tasks_by_metadata(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self._store.list_tasks_by_metadata(**kwargs)

    def assign_task(self, **kwargs: Any) -> dict[str, Any]:
        out = self._store.assign_task(**kwargs)
        self._notify_task_changed(kwargs.get("request_id"))
        return _wire_result(out)

    def claim_task(self, **kwargs: Any) -> dict[str, Any]:
        out = self._store.claim_task(**kwargs)
        self._notify_task_changed(kwargs.get("request_id"))
        return _wire_result(out)

    def renew_lease(self, **kwargs: Any) -> dict[str, Any]:
        out = self._store.renew_lease(**kwargs)
        self._notify_task_changed(kwargs.get("request_id"))
        return _wire_result(out)

    def batch_renew_leases(self, **kwargs: Any) -> list[dict[str, Any]]:
        results = self._store.batch_renew_leases(**kwargs)
        for item in kwargs.get("items", []):
            self._notify_task_changed(item.get("request_id"))
        return [_wire_result(r) for r in results]

    def begin_finalize(self, **kwargs: Any) -> dict[str, Any]:
        out = self._store.begin_finalize(**kwargs)
        self._notify_task_changed(kwargs.get("request_id"))
        return _wire_result(out)

    def stage_payload(self, **kwargs: Any) -> dict[str, Any]:
        out = self._store.stage_payload(**kwargs)
        self._notify_task_changed(kwargs.get("request_id"))
        return out

    def commit_finalize_success(self, **kwargs: Any) -> dict[str, Any]:
        out = self._store.commit_finalize_success(**kwargs)
        self._notify_task_changed(kwargs.get("request_id"))
        try:
            self._future_store_or_create().complete_task_success(
                request_id=str(kwargs["request_id"]),
                result_path=kwargs.get("result_path"),
                result_checksum=kwargs.get("result_checksum"),
                result_size_bytes=kwargs.get("result_size_bytes"),
                metadata={"done_at": _now(None), "final_status": "done", "payload_state": "committed"},
            )
        except Exception:
            pass
        return _wire_result(out)

    def commit_finalize_failure(self, **kwargs: Any) -> dict[str, Any]:
        out = self._store.commit_finalize_failure(**kwargs)
        self._notify_task_changed(kwargs.get("request_id"))
        try:
            self._future_store_or_create().complete_task_failure(
                request_id=str(kwargs["request_id"]),
                error=str(kwargs.get("error") or "task failed"),
                metadata={"failed_at": _now(None), "done_at": _now(None), "final_status": "failed"},
            )
        except Exception:
            pass
        return _wire_result(out)

    def requeue_task(self, **kwargs: Any) -> dict[str, Any]:
        out = self._store.requeue_task(**kwargs)
        self._notify_task_changed(kwargs.get("request_id"))
        return _wire_result(out)

    def list_active_tasks(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self._store.list_active_tasks(**kwargs)

    def list_expired_leases(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self._store.list_expired_leases(**kwargs)

    def get_task(self, request_id: str) -> dict[str, Any]:
        return self._store.get_task(request_id)

    def upsert_sampling_session(self, **kwargs: Any) -> None:
        return self._store.upsert_sampling_session(**kwargs)

    def delete_sampling_session(self, **kwargs: Any) -> None:
        return self._store.delete_sampling_session(**kwargs)

    def set_sampling_session_last_activity(self, **kwargs: Any) -> float | None:
        return self._store.set_sampling_session_last_activity(**kwargs)

    def get_sampling_session(self, **kwargs: Any) -> dict[str, Any] | None:
        return self._store.get_sampling_session(**kwargs)

    def list_sampling_sessions(self) -> list[dict[str, Any]]:
        return self._store.list_sampling_sessions()

    def upsert_training_session(self, **kwargs: Any) -> None:
        return self._store.upsert_training_session(**kwargs)

    def delete_training_session(self, **kwargs: Any) -> None:
        return self._store.delete_training_session(**kwargs)

    def set_training_session_last_activity(self, **kwargs: Any) -> float | None:
        return self._store.set_training_session_last_activity(**kwargs)

    def mark_training_session_inflight(self, **kwargs: Any) -> int | None:
        return self._store.mark_training_session_inflight(**kwargs)

    def get_training_session(self, **kwargs: Any) -> dict[str, Any] | None:
        return self._store.get_training_session(**kwargs)

    def bump_training_session_step(self, **kwargs: Any) -> int:
        return self._store.bump_training_session_step(**kwargs)

    def set_training_session_step(self, **kwargs: Any) -> int:
        return self._store.set_training_session_step(**kwargs)

    def list_training_sessions(self) -> list[dict[str, Any]]:
        return self._store.list_training_sessions()

    def upsert_gateway_sampling_session(self, **kwargs: Any) -> None:
        return self._store.upsert_gateway_sampling_session(**kwargs)

    def get_gateway_sampling_session(self, **kwargs: Any) -> dict[str, str] | None:
        return self._store.get_gateway_sampling_session(**kwargs)

    def delete_gateway_sampling_session(self, **kwargs: Any) -> None:
        return self._store.delete_gateway_sampling_session(**kwargs)

    def upsert_gateway_training_model(self, **kwargs: Any) -> None:
        return self._store.upsert_gateway_training_model(**kwargs)

    def get_gateway_training_model(self, **kwargs: Any) -> dict[str, str | None] | None:
        return self._store.get_gateway_training_model(**kwargs)

    def delete_gateway_training_model(self, **kwargs: Any) -> None:
        return self._store.delete_gateway_training_model(**kwargs)

    def list_gateway_routes(self) -> dict[str, Any]:
        return self._store.list_gateway_routes()

    def upsert_session_index(self, **kwargs: Any) -> None:
        return self._store.upsert_session_index(**kwargs)

    def add_training_run_to_session_index(self, **kwargs: Any) -> None:
        return self._store.add_training_run_to_session_index(**kwargs)

    def add_sampler_to_session_index(self, **kwargs: Any) -> None:
        return self._store.add_sampler_to_session_index(**kwargs)

    def add_heartbeat_sampler_to_session_index(self, **kwargs: Any) -> None:
        return self._store.add_heartbeat_sampler_to_session_index(**kwargs)

    def remove_sampler_from_session_index(self, **kwargs: Any) -> None:
        return self._store.remove_sampler_from_session_index(**kwargs)

    def get_session_index(self, **kwargs: Any) -> dict[str, Any] | None:
        return self._store.get_session_index(**kwargs)

    def list_session_index(self) -> list[dict[str, Any]]:
        return self._store.list_session_index()

    def upsert_sampler_index(self, **kwargs: Any) -> None:
        return self._store.upsert_sampler_index(**kwargs)

    def delete_sampler_index(self, **kwargs: Any) -> None:
        return self._store.delete_sampler_index(**kwargs)

    def get_sampler_index(self, **kwargs: Any) -> dict[str, Any] | None:
        return self._store.get_sampler_index(**kwargs)

    def list_sampler_index(self) -> list[dict[str, Any]]:
        return self._store.list_sampler_index()

    def update_session_heartbeat(self, **kwargs: Any) -> None:
        return self._store.update_session_heartbeat(**kwargs)

    def get_session_heartbeat(self, **kwargs: Any) -> float | None:
        return self._store.get_session_heartbeat(**kwargs)

    def delete_session_heartbeat(self, **kwargs: Any) -> bool:
        return self._store.delete_session_heartbeat(**kwargs)

    def session_heartbeat_size(self) -> int:
        return self._store.session_heartbeat_size()

    def is_session_heartbeat_stale(self, **kwargs: Any) -> bool:
        return self._store.is_session_heartbeat_stale(**kwargs)

    def prune_session_heartbeats(self, **kwargs: Any) -> int:
        return self._store.prune_session_heartbeats(**kwargs)


def _create_ray_actor_handle():
    try:
        import ray
    except Exception as e:
        raise TaskStateStoreUnavailableError("Ray import failed") from e

    actor_name = _ray_task_state_store_actor_name()
    namespace = _ray_namespace()
    db_path = _task_state_store_db_path()
    max_concurrency = int(os.environ.get("MINT_TASK_STATE_STORE_ACTOR_MAX_CONCURRENCY", "256"))

    @ray.remote(
        num_cpus=0,
        max_concurrency=max_concurrency,
        max_restarts=0,
        concurrency_groups={"health": 8},
    )
    class _RayTaskStateStoreActor(_TaskStateStoreActor):
        @ray.method(concurrency_group="health")
        def ping(self) -> dict[str, Any]:
            return super().ping()

    options: dict[str, Any] = {
        "name": actor_name,
        "namespace": namespace,
        "lifetime": "detached",
        "get_if_exists": True,
        "runtime_env": actor_runtime_env(
            pythonpath=PFS_PYTHONPATH,
            extra=otel_env_vars(),
            include_ray_attach_hints=False,
        ),
    }
    apply_detached_actor_resources(options, ray)
    actor = _RayTaskStateStoreActor.options(**options).remote(db_path)
    return actor


def _create_ray_actor(*, require_ready: bool = True):
    actor = _create_ray_actor_handle()
    if require_ready:
        out = sync_get_ray_ref(actor.ping.remote(), timeout_s=5.0)
        if not isinstance(out, dict):
            raise TypeError(f"TaskStateStore.ping returned non-dict: {type(out)}")
    return actor


async def _create_ray_actor_async(*, require_ready: bool = True):
    actor = await asyncio.to_thread(_create_ray_actor_handle)
    if require_ready:
        out = await async_get_ray_ref(actor.ping.remote(), timeout_s=5.0)
        if not isinstance(out, dict):
            raise TypeError(f"TaskStateStore.ping returned non-dict: {type(out)}")
    return actor


class TaskStateStoreClient:
    def __init__(self) -> None:
        self._ray_actor = None

    def _reset_ray_actor(self) -> None:
        self._ray_actor = None

    def _get_ray_actor_sync(self, *, require_ready: bool = True, create_if_missing: bool = False):
        try:
            import ray
        except Exception as e:
            raise TaskStateStoreUnavailableError("Ray import failed") from e
        if not ray.is_initialized():
            raise TaskStateStoreUnavailableError("Ray not initialized")
        actor = self._ray_actor
        if actor is not None:
            if not require_ready:
                return actor
            try:
                out = sync_get_ray_ref(actor.ping.remote(), timeout_s=1.0)
                if not isinstance(out, dict):
                    raise TypeError(f"TaskStateStore.ping returned non-dict: {type(out)}")
                return actor
            except Exception:
                self._reset_ray_actor()
        actor_name = _ray_task_state_store_actor_name()
        try:
            actor = ray.get_actor(actor_name, namespace=_ray_namespace())
        except Exception:
            if not create_if_missing:
                raise TaskStateStoreUnavailableError(
                    f"Detached Ray TaskStateStore actor unavailable actor_name={actor_name!r}"
                )
            try:
                actor = _create_ray_actor(require_ready=require_ready)
            except Exception as e:
                raise TaskStateStoreUnavailableError(
                    "Failed to get/create detached Ray TaskStateStore actor"
                ) from e
        self._ray_actor = actor
        return actor

    async def _get_ray_actor_async(self, *, require_ready: bool = True, create_if_missing: bool = False):
        try:
            import ray
        except Exception as e:
            raise TaskStateStoreUnavailableError("Ray import failed") from e
        if not ray.is_initialized():
            raise TaskStateStoreUnavailableError("Ray not initialized")
        actor = self._ray_actor
        if actor is not None:
            if not require_ready:
                return actor
            try:
                out = await async_get_ray_ref(actor.ping.remote(), timeout_s=1.0)
                if not isinstance(out, dict):
                    raise TypeError(f"TaskStateStore.ping returned non-dict: {type(out)}")
                return actor
            except Exception:
                self._reset_ray_actor()
        import ray

        actor_name = _ray_task_state_store_actor_name()
        try:
            actor = await asyncio.to_thread(
                ray.get_actor,
                actor_name,
                namespace=_ray_namespace(),
            )
        except Exception:
            if not create_if_missing:
                raise TaskStateStoreUnavailableError(
                    f"Detached Ray TaskStateStore actor unavailable actor_name={actor_name!r}"
                )
            try:
                actor = await _create_ray_actor_async(require_ready=require_ready)
            except Exception as e:
                raise TaskStateStoreUnavailableError(
                    "Failed to get/create detached Ray TaskStateStore actor"
                ) from e
        self._ray_actor = actor
        return actor

    async def _call(self, method: str, **kwargs: Any) -> Any:
        actor = await self._get_ray_actor_async()
        remote = getattr(actor, method).remote
        started = time.perf_counter()
        _inc_task_state_rpc_inflight(1.0)
        ok = False
        try:
            try:
                out = await async_get_ray_ref(remote(**kwargs))
            except Exception as exc:
                cause = _task_state_cause_from_ray_error(exc)
                if cause is not None:
                    raise cause from exc
                raise
            ok = True
            return out
        finally:
            _inc_task_state_rpc_inflight(-1.0)
            _record_task_state_rpc_metric(
                method,
                duration_ms=(time.perf_counter() - started) * 1000.0,
                ok=ok,
            )

    def _call_sync(self, method: str, **kwargs: Any) -> Any:
        actor = self._get_ray_actor_sync()
        remote = getattr(actor, method).remote
        started = time.perf_counter()
        _inc_task_state_rpc_inflight(1.0)
        ok = False
        try:
            try:
                out = sync_get_ray_ref(remote(**kwargs))
            except Exception as exc:
                cause = _task_state_cause_from_ray_error(exc)
                if cause is not None:
                    raise cause from exc
                raise
            ok = True
            return out
        finally:
            _inc_task_state_rpc_inflight(-1.0)
            _record_task_state_rpc_metric(
                method,
                duration_ms=(time.perf_counter() - started) * 1000.0,
                ok=ok,
            )

    def ensure_ready(
        self,
        *,
        timeout_s: float = 10.0,
        create_if_missing: bool = False,
    ) -> dict[str, Any]:
        actor = self._get_ray_actor_sync(require_ready=False, create_if_missing=create_if_missing)
        try:
            out = sync_get_ray_ref(actor.ping.remote(), timeout_s=timeout_s)
        except Exception:
            self._reset_ray_actor()
            if not create_if_missing:
                raise
            actor = self._get_ray_actor_sync(require_ready=False, create_if_missing=True)
            out = sync_get_ray_ref(actor.ping.remote(), timeout_s=timeout_s)
        if not isinstance(out, dict):
            raise TypeError(f"TaskStateStore.ping returned non-dict: {type(out)}")
        return out

    def ensure_started(self, *, timeout_s: float = 10.0) -> dict[str, Any]:
        actor = self._get_ray_actor_sync(require_ready=False, create_if_missing=True)
        out = sync_get_ray_ref(actor.ping.remote(), timeout_s=timeout_s)
        if not isinstance(out, dict):
            raise TypeError(f"TaskStateStore.ping returned non-dict: {type(out)}")
        return out

    def ping(self, *, timeout_s: float = 5.0) -> dict[str, Any]:
        actor = self._get_ray_actor_sync(require_ready=False, create_if_missing=False)
        try:
            out = sync_get_ray_ref(actor.ping.remote(), timeout_s=timeout_s)
        except Exception:
            self._reset_ray_actor()
            raise
        if not isinstance(out, dict):
            raise TypeError(f"TaskStateStore.ping returned non-dict: {type(out)}")
        if not bool(out.get("ok")):
            raise TaskStateStoreUnavailableError(f"TaskStateStore ping failed: {out!r}")
        return out

    async def async_ensure_started(self) -> None:
        await self._get_ray_actor_async(require_ready=False, create_if_missing=True)

    async def async_ensure_ready(
        self,
        *,
        timeout_s: float = 10.0,
        create_if_missing: bool = False,
    ) -> dict[str, Any]:
        actor = await self._get_ray_actor_async(require_ready=False, create_if_missing=create_if_missing)
        try:
            out = await async_get_ray_ref(actor.ping.remote(), timeout_s=timeout_s)
        except Exception:
            self._reset_ray_actor()
            if not create_if_missing:
                raise
            actor = await self._get_ray_actor_async(require_ready=False, create_if_missing=True)
            out = await async_get_ray_ref(actor.ping.remote(), timeout_s=timeout_s)
        if not isinstance(out, dict):
            raise TypeError(f"TaskStateStore.ping returned non-dict: {type(out)}")
        return out

    async def async_ping(self, *, timeout_s: float = 5.0) -> dict[str, Any]:
        actor = await self._get_ray_actor_async(require_ready=False, create_if_missing=False)
        try:
            out = await async_get_ray_ref(actor.ping.remote(), timeout_s=timeout_s)
        except Exception:
            self._reset_ray_actor()
            raise
        if not isinstance(out, dict):
            raise TypeError(f"TaskStateStore.ping returned non-dict: {type(out)}")
        if not bool(out.get("ok")):
            raise TaskStateStoreUnavailableError(f"TaskStateStore ping failed: {out!r}")
        return out

    async def async_stats(self) -> dict[str, Any]:
        out = await self._call("stats")
        if not isinstance(out, dict):
            raise TypeError(f"TaskStateStore.stats returned non-dict: {type(out)}")
        out = dict(out)
        out["task_state_rpc"] = task_state_rpc_metrics_snapshot()
        return out

    async def async_integrity_check(self) -> str:
        return str(await self._call("integrity_check"))

    async def async_acquire_scheduler_owner(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("acquire_scheduler_owner", **kwargs)

    async def async_acquire_owner(self, **kwargs: Any) -> dict[str, Any]:
        return await self.async_acquire_scheduler_owner(**kwargs)

    async def async_renew_scheduler_owner(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("renew_scheduler_owner", **kwargs)

    async def async_renew_owner(self, **kwargs: Any) -> dict[str, Any]:
        return await self.async_renew_scheduler_owner(**kwargs)

    async def async_create_task(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("create_task", **kwargs)

    async def async_ensure_task(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("ensure_task", **kwargs)

    async def async_update_task_metadata(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("update_task_metadata", **kwargs)

    async def async_stage_payload(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("stage_payload", **kwargs)

    async def async_complete_task_success(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("complete_task_success", **kwargs)

    async def async_complete_task_failure(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("complete_task_failure", **kwargs)

    async def async_mark_task_retrieved(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("mark_task_retrieved", **kwargs)

    async def async_forget_task(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("forget_task", **kwargs)

    async def async_expire_active_tasks(self, **kwargs: Any) -> list[str]:
        out = await self._call("expire_active_tasks", **kwargs)
        if not isinstance(out, list):
            raise TypeError(f"TaskStateStore.expire_active_tasks returned non-list: {type(out)}")
        return [str(x) for x in out]

    async def async_list_terminal_payloads_for_eviction(self, **kwargs: Any) -> list[dict[str, Any]]:
        out = await self._call("list_terminal_payloads_for_eviction", **kwargs)
        if not isinstance(out, list):
            raise TypeError(f"TaskStateStore.list_terminal_payloads_for_eviction returned non-list: {type(out)}")
        return out

    async def async_mark_payload_evicted(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("mark_payload_evicted", **kwargs)

    async def async_delete_expired_tombstones(self, **kwargs: Any) -> list[str]:
        out = await self._call("delete_expired_tombstones", **kwargs)
        if not isinstance(out, list):
            raise TypeError(f"TaskStateStore.delete_expired_tombstones returned non-list: {type(out)}")
        return [str(x) for x in out]

    async def async_record_payload_evict_error(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("record_payload_evict_error", **kwargs)

    async def async_record_billing_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        return await self._dict_call("record_billing_metrics", metrics=dict(metrics or {}))

    async def async_append_billing_outbox(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("append_billing_outbox", **kwargs)

    async def async_claim_billing_outbox(self, **kwargs: Any) -> list[dict[str, Any]]:
        out = await self._call("claim_billing_outbox", **kwargs)
        if not isinstance(out, list):
            raise TypeError(f"TaskStateStore.claim_billing_outbox returned non-list: {type(out)}")
        return out

    async def async_delete_billing_outbox_claim(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("delete_billing_outbox_claim", **kwargs)

    async def async_mark_billing_outbox_claim_failed(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("mark_billing_outbox_claim_failed", **kwargs)

    async def async_billing_outbox_stats(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("billing_outbox_stats", **kwargs)

    async def async_list_staged_payloads_for_gc(self, **kwargs: Any) -> list[dict[str, Any]]:
        out = await self._call("list_staged_payloads_for_gc", **kwargs)
        if not isinstance(out, list):
            raise TypeError(f"TaskStateStore.list_staged_payloads_for_gc returned non-list: {type(out)}")
        return out

    async def async_mark_staged_payload_gc_deleted(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("mark_staged_payload_gc_deleted", **kwargs)

    async def async_assign_task(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("assign_task", **kwargs)

    async def async_claim_task(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("claim_task", **kwargs)

    async def async_renew_lease(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("renew_lease", **kwargs)

    async def async_begin_finalize(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("begin_finalize", **kwargs)

    async def async_commit_finalize_success(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("commit_finalize_success", **kwargs)

    async def async_commit_finalize_failure(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("commit_finalize_failure", **kwargs)

    async def async_requeue_task(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("requeue_task", **kwargs)

    async def async_get_task(self, request_id: str) -> dict[str, Any]:
        return await self._dict_call("get_task", request_id=str(request_id))

    async def async_wait_task_status_change(
        self,
        *,
        request_id: str,
        timeout_s: float,
        observed_status: str | None = None,
        observed_updated_at: float | None = None,
        terminal_only: bool = False,
    ) -> dict[str, Any]:
        return await self._dict_call(
            "wait_task_status_change",
            request_id=str(request_id),
            timeout_s=float(timeout_s),
            observed_status=observed_status,
            observed_updated_at=observed_updated_at,
            terminal_only=bool(terminal_only),
        )

    async def async_future_ping(self, *, timeout_s: float = 5.0) -> dict[str, Any]:
        actor = await self._get_ray_actor_async(require_ready=False, create_if_missing=False)
        try:
            out = await async_get_ray_ref(actor.future_ping.remote(), timeout_s=timeout_s)
        except Exception:
            self._reset_ray_actor()
            raise
        if not isinstance(out, dict):
            raise TypeError(f"TaskStateStore.future_ping returned non-dict: {type(out)}")
        if not bool(out.get("ok")):
            raise TaskStateStoreUnavailableError(f"TaskStateStore future ping failed: {out!r}")
        return out

    async def async_future_stats(self) -> dict[str, Any]:
        return await self._dict_call("future_stats")

    async def async_future_acquire_scheduler_owner(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("future_acquire_scheduler_owner", **kwargs)

    async def async_future_renew_scheduler_owner(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("future_renew_scheduler_owner", **kwargs)

    async def async_future_create_task(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("future_create_task", **kwargs)

    async def async_future_ensure_task(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("future_ensure_task", **kwargs)

    async def async_future_update_task_metadata(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("future_update_task_metadata", **kwargs)

    async def async_future_stage_payload(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("future_stage_payload", **kwargs)

    async def async_future_complete_task_success(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("future_complete_task_success", **kwargs)

    async def async_future_complete_task_failure(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("future_complete_task_failure", **kwargs)

    async def async_future_mark_task_retrieved(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("future_mark_task_retrieved", **kwargs)

    async def async_future_forget_task(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("future_forget_task", **kwargs)

    async def async_future_expire_active_tasks(self, **kwargs: Any) -> list[str]:
        out = await self._call("future_expire_active_tasks", **kwargs)
        if not isinstance(out, list):
            raise TypeError(f"TaskStateStore.future_expire_active_tasks returned non-list: {type(out)}")
        return [str(x) for x in out]

    async def async_future_list_terminal_payloads_for_eviction(self, **kwargs: Any) -> list[dict[str, Any]]:
        out = await self._call("future_list_terminal_payloads_for_eviction", **kwargs)
        if not isinstance(out, list):
            raise TypeError(f"TaskStateStore.future_list_terminal_payloads_for_eviction returned non-list: {type(out)}")
        return out

    async def async_future_mark_payload_evicted(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("future_mark_payload_evicted", **kwargs)

    async def async_future_delete_expired_tombstones(self, **kwargs: Any) -> list[str]:
        out = await self._call("future_delete_expired_tombstones", **kwargs)
        if not isinstance(out, list):
            raise TypeError(f"TaskStateStore.future_delete_expired_tombstones returned non-list: {type(out)}")
        return [str(x) for x in out]

    async def async_future_record_payload_evict_error(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("future_record_payload_evict_error", **kwargs)

    async def async_future_list_staged_payloads_for_gc(self, **kwargs: Any) -> list[dict[str, Any]]:
        out = await self._call("future_list_staged_payloads_for_gc", **kwargs)
        if not isinstance(out, list):
            raise TypeError(f"TaskStateStore.future_list_staged_payloads_for_gc returned non-list: {type(out)}")
        return out

    async def async_future_mark_staged_payload_gc_deleted(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("future_mark_staged_payload_gc_deleted", **kwargs)

    async def async_future_assign_task(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("future_assign_task", **kwargs)

    async def async_future_claim_task(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("future_claim_task", **kwargs)

    async def async_future_renew_lease(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("future_renew_lease", **kwargs)

    async def async_future_begin_finalize(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("future_begin_finalize", **kwargs)

    async def async_future_commit_finalize_success(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("future_commit_finalize_success", **kwargs)

    async def async_future_commit_finalize_failure(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("future_commit_finalize_failure", **kwargs)

    async def async_future_requeue_task(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("future_requeue_task", **kwargs)

    async def async_future_get_task(self, request_id: str) -> dict[str, Any]:
        return await self._dict_call("future_get_task", request_id=str(request_id))

    async def async_future_wait_task_status_change(
        self,
        *,
        request_id: str,
        timeout_s: float,
        observed_status: str | None = None,
        observed_updated_at: float | None = None,
        terminal_only: bool = False,
    ) -> dict[str, Any]:
        return await self._dict_call(
            "future_wait_task_status_change",
            request_id=str(request_id),
            timeout_s=float(timeout_s),
            observed_status=observed_status,
            observed_updated_at=observed_updated_at,
            terminal_only=bool(terminal_only),
        )

    async def async_future_list_active_tasks(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        out = await self._call("future_list_active_tasks", limit=limit)
        if not isinstance(out, list):
            raise TypeError(f"TaskStateStore.future_list_active_tasks returned non-list: {type(out)}")
        return out

    async def async_future_list_expired_leases(
        self,
        *,
        now: float | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        out = await self._call("future_list_expired_leases", now=now, limit=limit)
        if not isinstance(out, list):
            raise TypeError(f"TaskStateStore.future_list_expired_leases returned non-list: {type(out)}")
        return out

    async def async_future_list_tasks_by_metadata(
        self,
        *,
        filters: dict[str, Any] | None = None,
        statuses: list[str] | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        out = await self._call("future_list_tasks_by_metadata", filters=filters, statuses=statuses, limit=limit)
        if not isinstance(out, list):
            raise TypeError(f"TaskStateStore.future_list_tasks_by_metadata returned non-list: {type(out)}")
        return out

    def upsert_sampling_session(self, *, session_id: str, info: dict[str, Any]) -> None:
        self._call_sync("upsert_sampling_session", session_id=str(session_id), info=dict(info))

    async def async_upsert_sampling_session(self, *, session_id: str, info: dict[str, Any]) -> None:
        await self._call("upsert_sampling_session", session_id=str(session_id), info=dict(info))

    def delete_sampling_session(self, *, session_id: str) -> None:
        self._call_sync("delete_sampling_session", session_id=str(session_id))

    async def async_delete_sampling_session(self, *, session_id: str) -> None:
        await self._call("delete_sampling_session", session_id=str(session_id))

    def set_sampling_session_last_activity(self, *, session_id: str, last_activity: float) -> float | None:
        out = self._call_sync(
            "set_sampling_session_last_activity",
            session_id=str(session_id),
            last_activity=float(last_activity),
        )
        return None if out is None else float(out)

    async def async_set_sampling_session_last_activity(self, *, session_id: str, last_activity: float) -> float | None:
        out = await self._call(
            "set_sampling_session_last_activity",
            session_id=str(session_id),
            last_activity=float(last_activity),
        )
        return None if out is None else float(out)

    def get_sampling_session(self, *, session_id: str) -> dict[str, Any] | None:
        out = self._call_sync("get_sampling_session", session_id=str(session_id))
        return dict(out) if isinstance(out, dict) else None

    async def async_get_sampling_session(self, *, session_id: str) -> dict[str, Any] | None:
        out = await self._call("get_sampling_session", session_id=str(session_id))
        return dict(out) if isinstance(out, dict) else None

    def list_sampling_sessions(self) -> list[dict[str, Any]]:
        out = self._call_sync("list_sampling_sessions")
        if not isinstance(out, list):
            raise TypeError(f"TaskStateStore.list_sampling_sessions returned non-list: {type(out)}")
        return out

    async def async_list_sampling_sessions(self) -> list[dict[str, Any]]:
        out = await self._call("list_sampling_sessions")
        if not isinstance(out, list):
            raise TypeError(f"TaskStateStore.list_sampling_sessions returned non-list: {type(out)}")
        return out

    def upsert_training_session(self, *, model_id: str, info: dict[str, Any]) -> None:
        self._call_sync("upsert_training_session", model_id=str(model_id), info=dict(info))

    async def async_upsert_training_session(self, *, model_id: str, info: dict[str, Any]) -> None:
        await self._call("upsert_training_session", model_id=str(model_id), info=dict(info))

    def delete_training_session(self, *, model_id: str) -> None:
        self._call_sync("delete_training_session", model_id=str(model_id))

    def set_training_session_last_activity(self, *, model_id: str, last_activity: float) -> float | None:
        out = self._call_sync("set_training_session_last_activity", model_id=str(model_id), last_activity=float(last_activity))
        return None if out is None else float(out)

    async def async_set_training_session_last_activity(self, *, model_id: str, last_activity: float) -> float | None:
        out = await self._call("set_training_session_last_activity", model_id=str(model_id), last_activity=float(last_activity))
        return None if out is None else float(out)

    def mark_training_session_inflight(self, *, model_id: str, delta: int) -> int | None:
        out = self._call_sync("mark_training_session_inflight", model_id=str(model_id), delta=int(delta))
        return None if out is None else int(out)

    async def async_mark_training_session_inflight(self, *, model_id: str, delta: int) -> int | None:
        out = await self._call("mark_training_session_inflight", model_id=str(model_id), delta=int(delta))
        return None if out is None else int(out)

    def get_training_session(self, *, model_id: str) -> dict[str, Any] | None:
        out = self._call_sync("get_training_session", model_id=str(model_id))
        return dict(out) if isinstance(out, dict) else None

    async def async_get_training_session(self, *, model_id: str) -> dict[str, Any] | None:
        out = await self._call("get_training_session", model_id=str(model_id))
        return dict(out) if isinstance(out, dict) else None

    def bump_training_session_step(self, *, model_id: str) -> int:
        return int(self._call_sync("bump_training_session_step", model_id=str(model_id)))

    def set_training_session_step(self, *, model_id: str, step: int) -> int:
        return int(self._call_sync("set_training_session_step", model_id=str(model_id), step=int(step)))

    def set_training_session_step_best_effort(self, *, model_id: str, step: int) -> None:
        self._call_sync("set_training_session_step", model_id=str(model_id), step=int(step))

    def bump_training_session_step_best_effort(self, *, model_id: str) -> None:
        self._call_sync("bump_training_session_step", model_id=str(model_id))

    def list_training_sessions(self) -> list[dict[str, Any]]:
        out = self._call_sync("list_training_sessions")
        if not isinstance(out, list):
            raise TypeError(f"TaskStateStore.list_training_sessions returned non-list: {type(out)}")
        return out

    async def async_list_training_sessions(self) -> list[dict[str, Any]]:
        out = await self._call("list_training_sessions")
        if not isinstance(out, list):
            raise TypeError(f"TaskStateStore.list_training_sessions returned non-list: {type(out)}")
        return out

    def upsert_gateway_sampling_session(self, *, sampling_session_id: str, upstream_alias: str, base_model: str) -> None:
        self._call_sync(
            "upsert_gateway_sampling_session",
            sampling_session_id=str(sampling_session_id),
            upstream_alias=str(upstream_alias),
            base_model=str(base_model),
        )

    async def async_upsert_gateway_sampling_session(self, *, sampling_session_id: str, upstream_alias: str, base_model: str) -> None:
        await self._call(
            "upsert_gateway_sampling_session",
            sampling_session_id=str(sampling_session_id),
            upstream_alias=str(upstream_alias),
            base_model=str(base_model),
        )

    def get_gateway_sampling_session(self, *, sampling_session_id: str) -> dict[str, str] | None:
        out = self._call_sync("get_gateway_sampling_session", sampling_session_id=str(sampling_session_id))
        return dict(out) if isinstance(out, dict) else None

    async def async_get_gateway_sampling_session(self, *, sampling_session_id: str) -> dict[str, str] | None:
        out = await self._call("get_gateway_sampling_session", sampling_session_id=str(sampling_session_id))
        return dict(out) if isinstance(out, dict) else None

    def delete_gateway_sampling_session(self, *, sampling_session_id: str) -> None:
        self._call_sync("delete_gateway_sampling_session", sampling_session_id=str(sampling_session_id))

    async def async_delete_gateway_sampling_session(self, *, sampling_session_id: str) -> None:
        await self._call("delete_gateway_sampling_session", sampling_session_id=str(sampling_session_id))

    def upsert_gateway_training_model(
        self,
        *,
        model_id: str,
        upstream_alias: str,
        base_model: str,
        owner_id: str | None = None,
    ) -> None:
        self._call_sync(
            "upsert_gateway_training_model",
            model_id=str(model_id),
            upstream_alias=str(upstream_alias),
            base_model=str(base_model),
            owner_id=owner_id,
        )

    async def async_upsert_gateway_training_model(
        self,
        *,
        model_id: str,
        upstream_alias: str,
        base_model: str,
        owner_id: str | None = None,
    ) -> None:
        await self._call(
            "upsert_gateway_training_model",
            model_id=str(model_id),
            upstream_alias=str(upstream_alias),
            base_model=str(base_model),
            owner_id=owner_id,
        )

    def get_gateway_training_model(self, *, model_id: str) -> dict[str, str | None] | None:
        out = self._call_sync("get_gateway_training_model", model_id=str(model_id))
        return dict(out) if isinstance(out, dict) else None

    async def async_get_gateway_training_model(self, *, model_id: str) -> dict[str, str | None] | None:
        out = await self._call("get_gateway_training_model", model_id=str(model_id))
        return dict(out) if isinstance(out, dict) else None

    def delete_gateway_training_model(self, *, model_id: str) -> None:
        self._call_sync("delete_gateway_training_model", model_id=str(model_id))

    async def async_delete_gateway_training_model(self, *, model_id: str) -> None:
        await self._call("delete_gateway_training_model", model_id=str(model_id))

    def list_gateway_routes(self) -> dict[str, Any]:
        out = self._call_sync("list_gateway_routes")
        if not isinstance(out, dict):
            raise TypeError(f"TaskStateStore.list_gateway_routes returned non-dict: {type(out)}")
        return out

    def upsert_session_index(self, *, session_id: str, info: dict[str, Any]) -> None:
        self._call_sync("upsert_session_index", session_id=str(session_id), info=dict(info))

    async def async_upsert_session_index(self, *, session_id: str, info: dict[str, Any]) -> None:
        await self._call("upsert_session_index", session_id=str(session_id), info=dict(info))

    def add_training_run_to_session_index(
        self,
        *,
        session_id: str,
        training_run_id: str,
        user_id: str | None = None,
        created_at: Any | None = None,
    ) -> None:
        self._call_sync(
            "add_training_run_to_session_index",
            session_id=str(session_id),
            training_run_id=str(training_run_id),
            user_id=user_id,
            created_at=created_at,
        )

    def add_sampler_to_session_index(
        self,
        *,
        session_id: str,
        sampler_id: str,
        user_id: str | None = None,
        created_at: Any | None = None,
    ) -> None:
        self._call_sync(
            "add_sampler_to_session_index",
            session_id=str(session_id),
            sampler_id=str(sampler_id),
            user_id=user_id,
            created_at=created_at,
        )

    def add_heartbeat_sampler_to_session_index(
        self,
        *,
        session_id: str,
        sampler_id: str,
        user_id: str | None = None,
        created_at: Any | None = None,
    ) -> None:
        self._call_sync(
            "add_heartbeat_sampler_to_session_index",
            session_id=str(session_id),
            sampler_id=str(sampler_id),
            user_id=user_id,
            created_at=created_at,
        )

    def remove_sampler_from_session_index(self, *, session_id: str, sampler_id: str) -> None:
        self._call_sync("remove_sampler_from_session_index", session_id=str(session_id), sampler_id=str(sampler_id))

    def get_session_index(self, *, session_id: str) -> dict[str, Any] | None:
        out = self._call_sync("get_session_index", session_id=str(session_id))
        return dict(out) if isinstance(out, dict) else None

    async def async_get_session_index(self, *, session_id: str) -> dict[str, Any] | None:
        out = await self._call("get_session_index", session_id=str(session_id))
        return dict(out) if isinstance(out, dict) else None

    def list_session_index(self) -> list[dict[str, Any]]:
        out = self._call_sync("list_session_index")
        if not isinstance(out, list):
            raise TypeError(f"TaskStateStore.list_session_index returned non-list: {type(out)}")
        return out

    async def async_list_session_index(self) -> list[dict[str, Any]]:
        out = await self._call("list_session_index")
        if not isinstance(out, list):
            raise TypeError(f"TaskStateStore.list_session_index returned non-list: {type(out)}")
        return out

    def upsert_sampler_index(self, *, sampler_id: str, info: dict[str, Any]) -> None:
        self._call_sync("upsert_sampler_index", sampler_id=str(sampler_id), info=dict(info))

    def delete_sampler_index(self, *, sampler_id: str) -> None:
        self._call_sync("delete_sampler_index", sampler_id=str(sampler_id))

    def get_sampler_index(self, *, sampler_id: str) -> dict[str, Any] | None:
        out = self._call_sync("get_sampler_index", sampler_id=str(sampler_id))
        return dict(out) if isinstance(out, dict) else None

    async def async_get_sampler_index(self, *, sampler_id: str) -> dict[str, Any] | None:
        out = await self._call("get_sampler_index", sampler_id=str(sampler_id))
        return dict(out) if isinstance(out, dict) else None

    def list_sampler_index(self) -> list[dict[str, Any]]:
        out = self._call_sync("list_sampler_index")
        if not isinstance(out, list):
            raise TypeError(f"TaskStateStore.list_sampler_index returned non-list: {type(out)}")
        return out

    async def async_list_sampler_index(self) -> list[dict[str, Any]]:
        out = await self._call("list_sampler_index")
        if not isinstance(out, list):
            raise TypeError(f"TaskStateStore.list_sampler_index returned non-list: {type(out)}")
        return out

    def update_session_heartbeat(self, *, session_id: str, now: float | None = None) -> None:
        self._call_sync("update_session_heartbeat", session_id=str(session_id), now=now)

    async def async_update_session_heartbeat(self, *, session_id: str, now: float | None = None) -> None:
        await self._call("update_session_heartbeat", session_id=str(session_id), now=now)

    def get_session_heartbeat(self, *, session_id: str) -> float | None:
        out = self._call_sync("get_session_heartbeat", session_id=str(session_id))
        return None if out is None else float(out)

    def delete_session_heartbeat(self, *, session_id: str) -> bool:
        return bool(self._call_sync("delete_session_heartbeat", session_id=str(session_id)))

    def session_heartbeat_size(self) -> int:
        return int(self._call_sync("session_heartbeat_size"))

    async def async_session_heartbeat_size(self, *, create_if_missing: bool = False) -> int:
        if create_if_missing:
            return int(await self._call("session_heartbeat_size"))
        actor = await self._get_ray_actor_async(require_ready=False, create_if_missing=False)
        out = await async_get_ray_ref(actor.session_heartbeat_size.remote())
        return int(out)

    def is_session_heartbeat_stale(self, *, session_id: str, ttl_s: float) -> bool:
        return bool(self._call_sync("is_session_heartbeat_stale", session_id=str(session_id), ttl_s=float(ttl_s)))

    async def async_is_session_heartbeat_stale(self, *, session_id: str, ttl_s: float) -> bool:
        return bool(await self._call("is_session_heartbeat_stale", session_id=str(session_id), ttl_s=float(ttl_s)))

    def prune_session_heartbeats(self, *, max_age_s: float) -> int:
        return int(self._call_sync("prune_session_heartbeats", max_age_s=float(max_age_s)))

    async def async_list_active_tasks(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        out = await self._call("list_active_tasks", limit=limit)
        if not isinstance(out, list):
            raise TypeError(f"TaskStateStore.list_active_tasks returned non-list: {type(out)}")
        return out

    async def async_list_expired_leases(
        self,
        *,
        now: float | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        out = await self._call("list_expired_leases", now=now, limit=limit)
        if not isinstance(out, list):
            raise TypeError(f"TaskStateStore.list_expired_leases returned non-list: {type(out)}")
        return out

    async def async_list_tasks_by_metadata(
        self,
        *,
        filters: dict[str, Any] | None = None,
        statuses: list[str] | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        out = await self._call("list_tasks_by_metadata", filters=filters, statuses=statuses, limit=limit)
        if not isinstance(out, list):
            raise TypeError(f"TaskStateStore.list_tasks_by_metadata returned non-list: {type(out)}")
        return out

    async def _dict_call(self, method: str, **kwargs: Any) -> dict[str, Any]:
        out = await self._call(method, **kwargs)
        out = _wire_result(out)
        if not isinstance(out, dict):
            raise TypeError(f"TaskStateStore.{method} returned non-dict: {type(out)}")
        return out


class _FutureStateAccess:
    """Map future-state contract calls onto TaskStateStore future methods."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def _call(self, future_method: str, **kwargs: Any) -> Any:
        method = getattr(self._client, f"async_future_{future_method}", None)
        if callable(method):
            return await method(**kwargs)
        raise AttributeError(f"future-state client missing async_future_{future_method}")

    async def _dict_call(self, future_method: str, **kwargs: Any) -> dict[str, Any]:
        out = await self._call(future_method, **kwargs)
        if not isinstance(out, dict):
            raise TypeError(f"FutureState.{future_method} returned non-dict: {type(out)}")
        return out

    async def _list_call(self, future_method: str, **kwargs: Any) -> list[Any]:
        out = await self._call(future_method, **kwargs)
        if not isinstance(out, list):
            raise TypeError(f"FutureState.{future_method} returned non-list: {type(out)}")
        return out

    async def async_ensure_started(self) -> None:
        ensure_started = getattr(self._client, "async_ensure_started", None)
        if callable(ensure_started):
            await ensure_started()
        await self.async_ping(timeout_s=5.0)

    async def async_ensure_ready(
        self,
        *,
        timeout_s: float = 10.0,
        create_if_missing: bool = False,
    ) -> dict[str, Any]:
        ensure_ready = getattr(self._client, "async_ensure_ready", None)
        if callable(ensure_ready):
            await ensure_ready(timeout_s=timeout_s, create_if_missing=create_if_missing)
        return await self.async_ping(timeout_s=timeout_s)

    async def async_ping(self, *, timeout_s: float = 5.0) -> dict[str, Any]:
        return await self._dict_call("ping", timeout_s=timeout_s)

    async def async_ensure_task(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("ensure_task", **kwargs)

    async def async_update_task_metadata(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("update_task_metadata", **kwargs)

    async def async_stage_payload(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("stage_payload", **kwargs)

    async def async_complete_task_success(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("complete_task_success", **kwargs)

    async def async_complete_task_failure(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("complete_task_failure", **kwargs)

    async def async_mark_task_retrieved(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("mark_task_retrieved", **kwargs)

    async def async_forget_task(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("forget_task", **kwargs)

    async def async_get_task(self, request_id: str) -> dict[str, Any]:
        return await self._dict_call("get_task", request_id=str(request_id))

    async def async_wait_task_status_change(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("wait_task_status_change", **kwargs)

    async def async_stats(self) -> dict[str, Any]:
        return await self._dict_call("stats")

    async def async_expire_active_tasks(self, **kwargs: Any) -> list[str]:
        return [str(x) for x in await self._list_call("expire_active_tasks", **kwargs)]

    async def async_list_terminal_payloads_for_eviction(self, **kwargs: Any) -> list[dict[str, Any]]:
        return await self._list_call("list_terminal_payloads_for_eviction", **kwargs)

    async def async_mark_payload_evicted(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("mark_payload_evicted", **kwargs)

    async def async_delete_expired_tombstones(self, **kwargs: Any) -> list[str]:
        return [str(x) for x in await self._list_call("delete_expired_tombstones", **kwargs)]

    async def async_record_payload_evict_error(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("record_payload_evict_error", **kwargs)

    async def async_list_staged_payloads_for_gc(self, **kwargs: Any) -> list[dict[str, Any]]:
        return await self._list_call("list_staged_payloads_for_gc", **kwargs)

    async def async_mark_staged_payload_gc_deleted(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("mark_staged_payload_gc_deleted", **kwargs)

    async def async_list_tasks_by_metadata(self, **kwargs: Any) -> list[dict[str, Any]]:
        return await self._list_call("list_tasks_by_metadata", **kwargs)


class TaskFutureService:
    """Future polling facade backed by TaskStateStore metadata and payload files.

    This owns the external Tinker future lifecycle while the durable state lives
    in the detached TaskStateStore actor. Result payload files are written by an
    in-process TaskPayloadStore helper; TaskPayloadStore is not a Ray actor.
    The facade intentionally mirrors the old async future methods so routes can
    migrate without changing public polling semantics.
    """

    def __init__(
        self,
        *,
        task_state_client: TaskStateStoreClient | None = None,
        future_state_client: Any | None = None,
        payload_store: Any | None = None,
    ) -> None:
        self._task_state = task_state_client if task_state_client is not None else task_state_store
        self._future_state_client = future_state_client
        self._payload_store = payload_store

    @property
    def _future_state(self) -> Any:
        if self._future_state_client is None:
            self._future_state_client = _FutureStateAccess(self._task_state)
        return self._future_state_client

    @property
    def _payloads(self) -> Any:
        if self._payload_store is None:
            from mint_server.backend.stores.task_payload_store import TaskPayloadStore

            self._payload_store = TaskPayloadStore()
        return self._payload_store

    async def async_create_with_id(self, request_id: str) -> str:
        await self._future_state.async_ensure_task(request_id=str(request_id), status="pending")
        return str(request_id)

    async def async_create_model_work_with_id(
        self,
        request_id: str,
        *,
        op: str,
        domain_key: str,
        request_json: bytes,
        meta: dict[str, Any] | None = None,
        payload_hash: str | None = None,
    ) -> dict[str, Any]:
        out = await self._future_state.async_ensure_task(
            request_id=str(request_id),
            op=str(op),
            domain_key=str(domain_key),
            request_json=bytes(request_json),
            payload_hash=payload_hash,
            metadata=dict(meta or {}),
            status="pending",
        )
        out = _wire_result(out)
        return {
            "request_id": str(request_id),
            "created": bool(out.get("created")),
            "record": out.get("record"),
        }

    async def async_ensure_pending(self, request_id: str, meta: dict[str, Any] | None = None) -> dict[str, Any]:
        out = await self._future_state.async_ensure_task(
            request_id=str(request_id),
            op=str((meta or {}).get("op") or "unknown"),
            domain_key=str((meta or {}).get("domain_key") or "future:default"),
            metadata=dict(meta or {}),
            status="pending",
        )
        return {"created": bool(out.get("created")), "meta": out.get("record", {}).get("metadata") or {}}

    async def async_mark_queued(self, request_id: str, meta: dict[str, Any] | None = None) -> None:
        await self._future_state.async_ensure_task(
            request_id=str(request_id),
            op=str((meta or {}).get("op") or "unknown"),
            domain_key=str((meta or {}).get("domain_key") or "future:default"),
            metadata={**dict(meta or {}), "queue_state": "queued"},
            status="queued",
        )

    async def async_mark_running(self, request_id: str, meta: dict[str, Any] | None = None) -> None:
        await self._future_state.async_update_task_metadata(
            request_id=str(request_id),
            metadata={**dict(meta or {}), "queue_state": "running"},
            status="running",
        )

    async def async_update_meta(self, request_id: str, meta: dict[str, Any] | None = None) -> None:
        await self._future_state.async_update_task_metadata(
            request_id=str(request_id),
            metadata=dict(meta or {}),
        )

    async def async_resolve(
        self,
        request_id: str,
        result: Any,
        *,
        billing_observations: list[dict[str, Any]] | None = None,
    ) -> None:
        if self._buffer_model_work_finalize(
            kind="resolve",
            request_id=request_id,
            payload=result,
            billing_observations=billing_observations,
        ):
            return
        meta = await self.async_get_meta(request_id)
        result = _sync_training_session_step(meta, result)
        attempt_id = f"future__{uuid.uuid4().hex}"
        staged_payload_path = str(
            self._payloads.payload_path(
                request_id=str(request_id),
                attempt_id=attempt_id,
            )
        )
        await self._future_state.async_stage_payload(
            request_id=str(request_id),
            staged_payload_path=staged_payload_path,
            metadata={
                "staged_payload_path": staged_payload_path,
            },
        )
        payload = await asyncio.to_thread(
            self._payloads.write_json_payload,
            request_id=str(request_id),
            attempt_id=attempt_id,
            payload=result,
        )
        await self._future_state.async_complete_task_success(
            request_id=str(request_id),
            result_path=str(payload["path"]),
            result_checksum=str(payload["checksum"]),
            result_size_bytes=int(payload["size_bytes"]),
            metadata={
                "done_at": time.time(),
                "final_status": FutureStatus.DONE.value,
                "payload_state": "committed",
                "staged_payload_path": None,
            },
            billing_observations=billing_observations,
        )

    async def async_fail(self, request_id: str, error: str) -> None:
        if self._buffer_model_work_finalize(kind="fail", request_id=request_id, payload=str(error)):
            return
        try:
            await self._future_state.async_complete_task_failure(
                request_id=str(request_id),
                error=str(error),
                metadata={"failed_at": time.time(), "done_at": time.time(), "final_status": FutureStatus.FAILED.value},
            )
        except (KeyError, TaskStateNotFoundError):
            await self._future_state.async_ensure_task(
                request_id=str(request_id),
                status="pending",
                metadata={"failed_at": time.time()},
            )
            await self._future_state.async_complete_task_failure(
                request_id=str(request_id),
                error=str(error),
                metadata={"failed_at": time.time(), "done_at": time.time(), "final_status": FutureStatus.FAILED.value},
            )

    async def async_fail_if_pending_meta_matches(
        self,
        request_id: str,
        error: str,
        *,
        expected_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            record = await self._future_state.async_get_task(str(request_id))
        except (KeyError, TaskStateNotFoundError):
            return {"failed": False, "reason": "unknown"}
        if _status_from_task_record(record) != FutureStatus.PENDING:
            return {"failed": False, "reason": "not_pending"}
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        for key, value in dict(expected_meta or {}).items():
            if metadata.get(key) != value:
                return {"failed": False, "reason": "meta_mismatch"}
        await self.async_fail(str(request_id), str(error))
        return {"failed": True}

    async def async_get_status(self, request_id: str) -> FutureStatus:
        try:
            record = await self._future_state.async_get_task(str(request_id))
        except (KeyError, TaskStateNotFoundError):
            try:
                record = await self._task_state.async_get_task(str(request_id))
            except (KeyError, TaskStateNotFoundError):
                raise KeyError(f"Unknown request_id: {request_id}") from None
        return _status_from_task_record(record)

    async def async_wait_status_change(
        self,
        request_id: str,
        *,
        timeout_s: float,
        terminal_only: bool = False,
    ) -> FutureStatus | None:
        wait = getattr(self._future_state, "async_wait_task_status_change", None)
        if wait is None:
            return None
        try:
            out = await wait(
                request_id=str(request_id),
                timeout_s=float(timeout_s),
                terminal_only=bool(terminal_only),
            )
        except (KeyError, TaskStateNotFoundError):
            out = {"missing": True}
        if bool(out.get("missing")):
            out = await self._task_state.async_wait_task_status_change(
                request_id=str(request_id),
                timeout_s=float(timeout_s),
                terminal_only=bool(terminal_only),
            )
            if bool(out.get("missing")):
                raise KeyError(f"Unknown request_id: {request_id}") from None
        record = out.get("record")
        if not isinstance(record, dict):
            return None
        if not bool(out.get("changed")) and bool(out.get("timeout")):
            return None
        return _status_from_task_record(record)

    async def async_get_result(self, request_id: str) -> Any:
        try:
            record = await self._future_state.async_get_task(str(request_id))
            state_client = self._future_state
        except (KeyError, TaskStateNotFoundError):
            record = await self._task_state.async_get_task(str(request_id))
            state_client = self._task_state
        status = str(record.get("status"))
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        if status == "retrieved" and str(metadata.get("terminal_status") or "done") != "done":
            raise KeyError(f"Future already retrieved without result: {request_id}")
        if status not in {"done", "retrieved"}:
            raise KeyError(f"Future is not done: {request_id}")
        result_path = record.get("result_path")
        if not isinstance(result_path, str) or not result_path:
            raise KeyError(f"Future result payload missing: {request_id}")
        payload = await asyncio.to_thread(
            self._payloads.read_json_payload,
            path=result_path,
            expected_checksum=record.get("result_checksum"),
        )
        if status != "retrieved":
            await state_client.async_mark_task_retrieved(request_id=str(request_id))
        return payload

    async def async_get_error(self, request_id: str) -> str | None:
        try:
            record = await self._future_state.async_get_task(str(request_id))
        except (KeyError, TaskStateNotFoundError):
            record = await self._task_state.async_get_task(str(request_id))
        return None if record.get("error") is None else str(record.get("error"))

    async def async_get_meta(self, request_id: str) -> dict[str, Any] | None:
        try:
            record = await self._future_state.async_get_task(str(request_id))
        except (KeyError, TaskStateNotFoundError):
            record = await self._task_state.async_get_task(str(request_id))
        metadata = record.get("metadata")
        return dict(metadata) if isinstance(metadata, dict) else None

    async def async_list_pending_by_meta(
        self,
        filters: dict[str, Any] | None = None,
        *,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        records = await self._future_state.async_list_tasks_by_metadata(
            filters=dict(filters or {}),
            statuses=["pending", "queued", "assigned", "leased", "running", "finalizing"],
            limit=int(limit),
        )
        return [
            {
                "request_id": record["request_id"],
                "status": record["status"],
                "meta": record.get("metadata") or {},
            }
            for record in records
        ]

    async def async_fail_training_requests_for_model(self, model_id: str, error: str) -> list[str]:
        return await self._fail_requests_by_metadata(
            filters={"model_id": str(model_id)},
            op_prefix="training.",
            error=error,
        )

    async def async_fail_sampling_requests_for_session(self, sampling_session_id: str, error: str) -> list[str]:
        return await self._fail_requests_by_metadata(
            filters={"sampling_session_id": str(sampling_session_id)},
            op_prefix="sampling.",
            error=error,
        )

    async def _fail_requests_by_metadata(
        self,
        *,
        filters: dict[str, Any],
        op_prefix: str,
        error: str,
    ) -> list[str]:
        records = await self._future_state.async_list_tasks_by_metadata(
            filters=dict(filters),
            statuses=["pending", "queued", "assigned", "leased", "running", "finalizing"],
            limit=10000,
        )
        failed: list[str] = []
        for record in records:
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            op = metadata.get("op") or record.get("op")
            if not str(op or "").startswith(op_prefix):
                continue
            await self.async_fail(str(record["request_id"]), str(error))
            failed.append(str(record["request_id"]))
        return failed

    async def async_forget(self, request_id: str) -> None:
        await self._future_state.async_forget_task(request_id=str(request_id))

    async def async_cleanup(self, request_id: str) -> None:
        try:
            await self._future_state.async_mark_task_retrieved(request_id=str(request_id))
        except Exception:
            await self._future_state.async_forget_task(request_id=str(request_id))

    async def async_reap(self) -> dict[str, Any]:
        from mint_server.config import config as cfg

        now = time.time()
        batch_size = max(1, int(os.environ.get("MINT_TASK_FUTURE_REAPER_BATCH_SIZE", "1000")))

        expired = await self._future_state.async_expire_active_tasks(
            older_than_s=float(getattr(cfg, "task_pending_ttl_s", 86400.0)),
            now=now,
            limit=batch_size,
        )

        evicted: list[str] = []
        payload_evict_errors: list[dict[str, str]] = []
        staged_payload_gc_deleted: list[str] = []
        staged_payload_gc_errors: list[dict[str, str]] = []
        payload_candidates = await self._future_state.async_list_terminal_payloads_for_eviction(
            older_than_s=float(getattr(cfg, "task_result_ttl_s", 86400.0)),
            now=now,
            limit=batch_size,
        )
        for record in payload_candidates:
            request_id = str(record.get("request_id") or "")
            result_path = record.get("result_path")
            if not request_id or not isinstance(result_path, str) or not result_path:
                continue
            try:
                await self._payloads.async_delete_json_payload(path=result_path)
            except Exception as e:
                await self._future_state.async_record_payload_evict_error(count=1)
                payload_evict_errors.append(
                    {
                        "request_id": request_id,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
                continue
            marked = await self._future_state.async_mark_payload_evicted(
                request_id=request_id,
                expected_result_path=result_path,
                now=now,
            )
            if bool(marked.get("ok")):
                evicted.append(request_id)

        staged_candidates = await self._future_state.async_list_staged_payloads_for_gc(
            older_than_s=float(getattr(cfg, "task_result_ttl_s", 86400.0)),
            now=now,
            limit=batch_size,
        )
        for record in staged_candidates:
            request_id = str(record.get("request_id") or "")
            path = record.get("path")
            if not request_id or not isinstance(path, str) or not path:
                continue
            try:
                await self._payloads.async_delete_json_payload(path=path)
            except FileNotFoundError:
                pass
            except Exception as e:
                await self._future_state.async_record_payload_evict_error(count=1)
                staged_payload_gc_errors.append(
                    {
                        "request_id": request_id,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
                continue
            marked = await self._future_state.async_mark_staged_payload_gc_deleted(
                request_id=request_id,
                expected_staged_payload_path=path,
                now=now,
            )
            if bool(marked.get("ok")):
                staged_payload_gc_deleted.append(request_id)

        deleted_tombstones = await self._future_state.async_delete_expired_tombstones(
            older_than_s=float(getattr(cfg, "task_tombstone_ttl_s", 604800.0)),
            now=now,
            limit=batch_size,
        )

        return {
            "expired": expired,
            "timed_out": [],
            "payload_evicted": evicted,
            "staged_payload_gc_deleted": staged_payload_gc_deleted,
            "tombstones_deleted": deleted_tombstones,
            "payload_evict_errors": payload_evict_errors,
            "staged_payload_gc_errors": staged_payload_gc_errors,
            "metrics": task_future_reaper_metrics_snapshot(),
        }

    async def async_fail_stale_running_requests(self, active_consumer_job_id: str, error: str) -> list[str]:
        records = await self._future_state.async_list_tasks_by_metadata(
            filters={"consumer_job_id": str(active_consumer_job_id)},
            statuses=["running"],
            limit=10000,
        )
        failed: list[str] = []
        for record in records:
            await self.async_fail(str(record["request_id"]), str(error))
            failed.append(str(record["request_id"]))
        return failed

    async def async_ensure_started(self) -> None:
        await self._future_state.async_ensure_started()

    async def async_ensure_ready(
        self,
        *,
        timeout_s: float = 10.0,
        create_if_missing: bool = False,
    ) -> dict[str, Any]:
        return await self._future_state.async_ensure_ready(
            timeout_s=timeout_s,
            create_if_missing=create_if_missing,
        )

    async def async_ping(self, *, timeout_s: float = 5.0) -> dict[str, Any]:
        return await self._future_state.async_ping(timeout_s=timeout_s)

    def metrics_snapshot(self) -> dict[str, Any]:
        return {
            "backend": "future_state_store",
            "task_future_reaper": task_future_reaper_metrics_snapshot(),
            "billing_outbox": billing_metrics_snapshot(),
        }

    async def async_append_billing_outbox(
        self,
        observations: list[dict[str, Any]] | None,
        *,
        source: str = "unknown",
    ) -> dict[str, Any]:
        return await self._task_state.async_append_billing_outbox(
            observations=observations or [],
            source=str(source),
        )

    async def async_billing_outbox_stats(self) -> dict[str, Any]:
        return await self._task_state.async_billing_outbox_stats()

    async def async_rss_bytes(self, *, timeout_s: float = 10.0) -> int:
        _ = timeout_s
        return 0

    async def async_debug_snapshot(self, *, timeout_s: float = 10.0) -> dict[str, Any]:
        _ = timeout_s
        try:
            return {
                "backend": "future_state_store",
                "future_state_store": await self._future_state.async_stats(),
                "task_state_store": await self._task_state.async_stats(),
            }
        except Exception as e:
            return {"backend": "future_state_store", "error": f"{type(e).__name__}: {e}"}

    async def async_flush_billing_outbox(
        self,
        *,
        limit: int = 100,
        lease_ttl_s: float = 60.0,
        claim_id: str | None = None,
    ) -> dict[str, Any]:
        claim = str(claim_id or f"billing_flush_{uuid.uuid4().hex}")
        rows = await self._task_state.async_claim_billing_outbox(
            claim_id=claim,
            limit=max(1, int(limit)),
            lease_ttl_s=max(1.0, float(lease_ttl_s)),
        )
        if not rows:
            return {"ok": True, "claimed": 0, "inserted": 0, "conflict": 0, "failed": 0}
        outbox_ids = [int(row["outbox_id"]) for row in rows]
        try:
            from datetime import datetime, timezone

            from mint_server.usage_store import UsageEvent, get_usage_store, is_permanent_usage_write_error

            events: list[UsageEvent] = []
            for row in rows:
                event = dict(row.get("event") or {})
                events.append(
                    UsageEvent(
                        account_id=str(event["account_id"]),
                        apikey_id=str(event["apikey_id"]),
                        charge_item=str(event["charge_item"]),
                        quantity=int(event["quantity"]),
                        request_id=str(event["request_id"]),
                        label=str(event.get("label") or ""),
                        event_id=str(event["event_id"]),
                        event_time=datetime.fromtimestamp(float(event.get("event_time") or time.time()), tz=timezone.utc),
                    )
                )
            usage_store = await get_usage_store()
            inserted_ids = set(await usage_store.write_events(events))
            conflict = max(0, len(events) - len(inserted_ids))
            await self._task_state.async_delete_billing_outbox_claim(claim_id=claim, outbox_ids=outbox_ids)
            billing_metrics = {
                "flush_success": 1,
                "event_inserted": len(inserted_ids),
                "event_conflict": conflict,
            }
            try:
                await self._task_state.async_record_billing_metrics(billing_metrics)
            except Exception:
                pass
            return {
                "ok": True,
                "claimed": len(rows),
                "inserted": len(inserted_ids),
                "conflict": conflict,
                "failed": 0,
            }
        except Exception as e:
            try:
                permanent = bool(is_permanent_usage_write_error(e))  # type: ignore[name-defined]
            except Exception:
                permanent = False
            await self._task_state.async_mark_billing_outbox_claim_failed(
                claim_id=claim,
                outbox_ids=outbox_ids,
                permanent=permanent,
                error=f"{type(e).__name__}: {e}",
            )
            billing_metrics = {
                "flush_permanent_error" if permanent else "flush_transient_error": 1,
                "event_failed": len(rows),
            }
            try:
                await self._task_state.async_record_billing_metrics(billing_metrics)
            except Exception:
                pass
            return {
                "ok": False,
                "claimed": len(rows),
                "inserted": 0,
                "conflict": 0,
                "failed": len(rows),
                "permanent": permanent,
                "error": f"{type(e).__name__}: {e}",
            }

    def _buffer_model_work_finalize(
        self,
        *,
        kind: str,
        request_id: str,
        payload: Any,
        billing_observations: list[dict[str, Any]] | None = None,
    ) -> bool:
        buffer = get_current_model_work_finalize_buffer()
        if buffer is None:
            return False
        buffer.finalization = ModelWorkFinalize(
            kind=kind,
            request_id=str(request_id),
            payload=payload,
            billing_observations=billing_observations,
        )
        return True


task_state_store = TaskStateStoreClient()
task_futures = TaskFutureService()
