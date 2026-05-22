from __future__ import annotations

import asyncio
import base64
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from ..config import PFS_PYTHONPATH, actor_runtime_env, apply_detached_actor_resources, config as server_config, otel_env_vars
from ..runtime_env import env_nonempty
from .async_ray_control import async_get_ray_ref, sync_get_ray_ref
from .task_state_store import (
    ACTIVE_TASK_STATUSES,
    TERMINAL_TASK_STATUSES,
    FutureStatus,
    TaskStateConflictError,
    TaskStateNotFoundError,
    TaskStateStoreError,
    TaskStateStoreUnavailableError,
    _metric_number,
    _otel_metric_attrs,
    _ray_namespace,
)


_SCHEMA_VERSION = 1
_OWNER_PREFIX = "owner:"
_TASK_PREFIX = "task:"


def _now(now: float | None = None) -> float:
    return time.time() if now is None else float(now)


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


def _ray_future_state_store_actor_name() -> str:
    return str(
        os.environ.get("MINT_FUTURE_STATE_STORE_ACTOR_NAME")
        or getattr(server_config, "future_state_store_actor_name", "mint_future_state_store")
    )


def _future_state_store_db_path() -> str:
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


class _DictKV:
    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self._data.get(str(key))

    def put(self, key: str, value: str) -> None:
        self._data[str(key)] = str(value)

    def delete(self, key: str) -> None:
        self._data.pop(str(key), None)

    def keys(self) -> list[str]:
        return list(self._data.keys())

    def close(self) -> None:
        self._data.clear()


class _RocksKV:
    def __init__(self, path: str) -> None:
        try:
            from rocksdict import Rdict
        except Exception as e:
            raise TaskStateStoreUnavailableError(
                "rocksdict is required for persistent FutureStateStore"
            ) from e
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = Rdict(str(path))

    def get(self, key: str) -> str | None:
        value = self._db.get(str(key))
        if value is None:
            return None
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    def put(self, key: str, value: str) -> None:
        self._db[str(key)] = str(value)

    def delete(self, key: str) -> None:
        try:
            del self._db[str(key)]
        except KeyError:
            pass

    def keys(self) -> list[str]:
        return [str(key.decode("utf-8") if isinstance(key, bytes) else key) for key in self._db.keys()]

    def close(self) -> None:
        close = getattr(self._db, "close", None)
        if callable(close):
            close()


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


class FutureStateStore:
    """Persistent future state machine backed by a single-writer KV store."""

    def __init__(self, db_path: str | os.PathLike[str]) -> None:
        self._db_path = str(db_path)
        self._lock = threading.RLock()
        self._kv = _DictKV() if self._db_path == ":memory:" else _RocksKV(self._db_path)

    @classmethod
    def in_memory(cls) -> "FutureStateStore":
        return cls(":memory:")

    @property
    def db_path(self) -> str:
        return self._db_path

    def close(self) -> None:
        with self._lock:
            self._kv.close()

    def ping(self) -> dict[str, Any]:
        with self._lock:
            self._kv.put("__ping__", _json_dumps({"ok": True}))
            self._kv.delete("__ping__")
        return {"ok": True}

    def _load(self, request_id: str) -> dict[str, Any]:
        raw = self._kv.get(_task_key(request_id))
        if raw is None:
            raise TaskStateNotFoundError(str(request_id))
        return _decode_record(raw)

    def _save(self, record: dict[str, Any]) -> dict[str, Any]:
        self._kv.put(_task_key(str(record["request_id"])), _encode_record(record))
        return dict(record)

    def _all_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for key in self._kv.keys():
            if key.startswith(_TASK_PREFIX):
                raw = self._kv.get(key)
                if raw is not None:
                    records.append(_decode_record(raw))
        records.sort(key=lambda r: (float(r.get("created_at") or 0.0), str(r.get("request_id") or "")))
        return records

    def _owner(self, name: str) -> dict[str, Any] | None:
        raw = self._kv.get(_owner_key(name))
        return None if raw is None else _json_loads(raw)

    def acquire_scheduler_owner(
        self,
        *,
        owner_id: str,
        ttl_s: float,
        name: str = "model_work_scheduler",
        now: float | None = None,
    ) -> dict[str, Any]:
        ts = _now(now)
        expires_at = ts + max(1.0, float(ttl_s))
        owner_id = str(owner_id)
        name = str(name)
        with self._lock:
            row = self._owner(name)
            if row is not None and str(row.get("owner_id")) != owner_id and float(row.get("expires_at") or 0.0) > ts:
                return {
                    "ok": False,
                    "reason": "owner_active",
                    "owner_id": str(row.get("owner_id")),
                    "epoch": int(row.get("epoch") or 0),
                    "expires_at": float(row.get("expires_at") or 0.0),
                }
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
            return dict(out)

    def renew_scheduler_owner(
        self,
        *,
        owner_id: str,
        epoch: int,
        ttl_s: float,
        name: str = "model_work_scheduler",
        now: float | None = None,
    ) -> dict[str, Any]:
        ts = _now(now)
        expires_at = ts + max(1.0, float(ttl_s))
        with self._lock:
            row = self._owner(str(name))
            if row is None or str(row.get("owner_id")) != str(owner_id) or int(row.get("epoch") or 0) != int(epoch):
                return {"ok": False, "reason": "stale_owner"}
            row.update({"renewed_at": ts, "expires_at": expires_at})
            self._kv.put(_owner_key(str(name)), _json_dumps(row))
            return {"ok": True, "owner_id": str(owner_id), "epoch": int(epoch), "expires_at": expires_at}

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
    ) -> dict[str, Any]:
        ts = _now(now)
        request_id = str(request_id)
        with self._lock:
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
                self._save(record)
                return {"ok": True, "created": True, "record": dict(record)}
            if payload_hash is not None and existing.get("payload_hash") not in (None, payload_hash):
                raise TaskStateConflictError("duplicate request_id with different payload hash")
            if str(existing.get("op")) != str(op) or str(existing.get("domain_key")) != str(domain_key):
                raise TaskStateConflictError("duplicate request_id with different task identity")
            existing["request_json"] = bytes(request_json)
            existing["metadata"] = _merge_metadata(existing, metadata)
            existing["updated_at"] = ts
            if existing.get("payload_hash") is None and payload_hash is not None:
                existing["payload_hash"] = payload_hash
            self._save(existing)
            return {"ok": True, "created": False, "record": dict(existing)}

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
        with self._lock:
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
                self._save(record)
                return {"ok": True, "created": True, "record": dict(record)}
            record["metadata"] = _merge_metadata(record, metadata)
            record["updated_at"] = _now(now)
            self._save(record)
            return {"ok": True, "created": False, "record": dict(record)}

    def update_task_metadata(
        self,
        *,
        request_id: str,
        metadata: dict[str, Any] | None = None,
        status: str | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._load(str(request_id))
            record["metadata"] = _merge_metadata(record, metadata)
            if status is not None:
                record["status"] = str(status)
            record["updated_at"] = _now(now)
            self._save(record)
            return {"ok": True, "record": dict(record)}

    def stage_payload(
        self,
        *,
        request_id: str,
        staged_payload_path: str,
        metadata: dict[str, Any] | None = None,
        status: str | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._load(str(request_id))
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
            self._save(record)
            return {"ok": True, "record": dict(record)}

    def complete_task_success(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.pop("billing_observations", None)
        return self._complete_task_direct(status="done", error=None, **kwargs)

    def complete_task_failure(self, **kwargs: Any) -> dict[str, Any]:
        return self._complete_task_direct(status="failed", **kwargs)

    def _complete_task_direct(
        self,
        *,
        request_id: str,
        status: str,
        result_path: str | None = None,
        result_checksum: str | None = None,
        result_size_bytes: int | None = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        ts = _now(now)
        with self._lock:
            record = self._load(str(request_id))
            if str(record.get("status")) in TERMINAL_TASK_STATUSES:
                if (
                    str(record.get("status")) == str(status)
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
            record.update(
                {
                    "status": str(status),
                    "result_path": result_path,
                    "result_checksum": result_checksum,
                    "result_size_bytes": result_size_bytes,
                    "staged_payload_path": None,
                    "staged_payload_checksum": None,
                    "staged_payload_size_bytes": None,
                    "error": error,
                    "updated_at": ts,
                    "finalizing_until": None,
                }
            )
            self._save(record)
            return {"ok": True, "idempotent": False, "record": dict(record)}

    def mark_task_retrieved(self, *, request_id: str, now: float | None = None) -> dict[str, Any]:
        with self._lock:
            record = self._load(str(request_id))
            if str(record.get("status")) == "retrieved":
                return {"ok": True, "record": dict(record)}
            if str(record.get("status")) not in {"done", "failed", "expired", "cancelled"}:
                raise TaskStateConflictError(f"cannot mark retrieved; current status={record.get('status')!r}")
            metadata = dict(record.get("metadata") or {})
            metadata.setdefault("terminal_status", str(record.get("status")))
            record["metadata"] = metadata
            record["status"] = "retrieved"
            record["updated_at"] = _now(now)
            self._save(record)
            return {"ok": True, "record": dict(record)}

    def forget_task(self, *, request_id: str) -> dict[str, Any]:
        with self._lock:
            existed = self._kv.get(_task_key(str(request_id))) is not None
            self._kv.delete(_task_key(str(request_id)))
            return {"ok": True, "deleted": existed}

    def expire_active_tasks(self, *, older_than_s: float, now: float | None = None, limit: int = 1000) -> list[str]:
        ttl_s = float(older_than_s)
        if ttl_s <= 0:
            return []
        ts = _now(now)
        cutoff = ts - ttl_s
        expired: list[str] = []
        expired_by_op: dict[str, int] = {}
        with self._lock:
            for record in self._all_records():
                if len(expired) >= max(0, int(limit)):
                    break
                if str(record.get("status")) not in {"pending", "queued", "assigned"}:
                    continue
                if float(record.get("created_at") or 0.0) > cutoff:
                    continue
                metadata = dict(record.get("metadata") or {})
                metadata.setdefault("terminal_status", "expired")
                metadata.setdefault("expired_at", ts)
                metadata.setdefault("failed_at", ts)
                record["metadata"] = metadata
                record["status"] = "expired"
                record["error"] = record.get("error") or "Future expired"
                record["updated_at"] = ts
                self._save(record)
                request_id = str(record["request_id"])
                expired.append(request_id)
                op = str(record.get("op") or "unknown")
                expired_by_op[op] = expired_by_op.get(op, 0) + 1
        if expired:
            from .task_state_store import _inc_future_timeout, _inc_reaper_rows

            _inc_reaper_rows("expire_pending", len(expired))
            for op, count in expired_by_op.items():
                _inc_future_timeout("execution", op=op, count=count)
        return expired

    def list_terminal_payloads_for_eviction(self, *, older_than_s: float, now: float | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        ttl_s = float(older_than_s)
        if ttl_s <= 0:
            return []
        cutoff = _now(now) - ttl_s
        out: list[dict[str, Any]] = []
        with self._lock:
            for record in self._all_records():
                if len(out) >= max(0, int(limit)):
                    break
                if str(record.get("status")) not in TERMINAL_TASK_STATUSES:
                    continue
                if not record.get("result_path"):
                    continue
                if self._terminal_completed_at(record) <= cutoff:
                    out.append(dict(record))
        return out

    def mark_payload_evicted(self, *, request_id: str, expected_result_path: str, now: float | None = None) -> dict[str, Any]:
        with self._lock:
            record = self._load(str(request_id))
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
            self._save(record)
        from .task_state_store import _inc_reaper_rows

        _inc_reaper_rows("evict_payload", 1)
        return {"ok": True, "record": dict(record)}

    def list_staged_payloads_for_gc(self, *, older_than_s: float, now: float | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        ttl_s = float(older_than_s)
        if ttl_s <= 0:
            return []
        cutoff = _now(now) - ttl_s
        out: list[dict[str, Any]] = []
        with self._lock:
            for record in self._all_records():
                if len(out) >= max(0, int(limit)):
                    break
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

    def mark_staged_payload_gc_deleted(self, *, request_id: str, expected_staged_payload_path: str, now: float | None = None) -> dict[str, Any]:
        expected = str(expected_staged_payload_path)
        with self._lock:
            record = self._load(str(request_id))
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
            self._save(record)
        from .task_state_store import _inc_reaper_rows

        _inc_reaper_rows("gc_staged_payload", 1)
        return {"ok": True, "record": dict(record)}

    def delete_expired_tombstones(self, *, older_than_s: float, now: float | None = None, limit: int = 1000) -> list[str]:
        ttl_s = float(older_than_s)
        if ttl_s <= 0:
            return []
        cutoff = _now(now) - ttl_s
        deleted: list[str] = []
        with self._lock:
            for record in self._all_records():
                if len(deleted) >= max(0, int(limit)):
                    break
                if str(record.get("status")) not in TERMINAL_TASK_STATUSES:
                    continue
                if record.get("result_path"):
                    continue
                if self._terminal_completed_at(record) > cutoff:
                    continue
                request_id = str(record["request_id"])
                self._kv.delete(_task_key(request_id))
                deleted.append(request_id)
        if deleted:
            from .task_state_store import _inc_reaper_rows

            _inc_reaper_rows("delete_tombstone", len(deleted))
        return deleted

    def record_payload_evict_error(self, *, count: int = 1) -> dict[str, Any]:
        from .task_state_store import _inc_payload_evict_errors

        _inc_payload_evict_errors(int(count))
        return {"ok": True}

    def list_tasks_by_metadata(self, *, filters: dict[str, Any] | None = None, statuses: list[str] | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        status_values = set(str(value) for value in (statuses or []))
        normalized_filters = dict(filters or {})
        out: list[dict[str, Any]] = []
        with self._lock:
            for record in self._all_records():
                if status_values and str(record.get("status")) not in status_values:
                    continue
                metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
                if all(metadata.get(key) == value for key, value in normalized_filters.items()):
                    out.append(dict(record))
                    if len(out) >= int(limit):
                        break
        return out

    def assign_task(self, *, request_id: str, subqueue_id: str, scheduler_epoch: int, now: float | None = None) -> dict[str, Any]:
        ts = _now(now)
        with self._lock:
            self._assert_scheduler_owner(scheduler_epoch=scheduler_epoch, now=ts)
            record = self._load(str(request_id))
            if str(record.get("status")) != "pending":
                raise TaskStateConflictError(f"cannot assign from pending; current status={record.get('status')!r}")
            record.update({"status": "assigned", "subqueue_id": str(subqueue_id), "scheduler_epoch": int(scheduler_epoch), "assigned_at": ts, "updated_at": ts})
            self._save(record)
            return {"ok": True, "record": dict(record)}

    def claim_task(self, *, request_id: str, subqueue_id: str, lease_id: str, attempt_id: str, consumer_id: str, scheduler_epoch: int, runtime_generation: int, lease_ttl_s: float, now: float | None = None) -> dict[str, Any]:
        ts = _now(now)
        expires_at = ts + max(1.0, float(lease_ttl_s))
        with self._lock:
            self._assert_scheduler_owner(scheduler_epoch=scheduler_epoch, now=ts)
            record = self._load(str(request_id))
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
            self._save(record)
            return {"ok": True, "record": dict(record)}

    def begin_finalize(self, *, request_id: str, lease_id: str, attempt_id: str, scheduler_epoch: int, runtime_generation: int, finalize_ttl_s: float, staged_payload_path: str | None = None, now: float | None = None) -> dict[str, Any]:
        ts = _now(now)
        finalizing_until = ts + max(1.0, float(finalize_ttl_s))
        with self._lock:
            self._assert_scheduler_owner(scheduler_epoch=scheduler_epoch, now=ts)
            record = self._load(str(request_id))
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
            self._save(record)
            return {"ok": True, "record": dict(record)}

    def commit_finalize_success(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.pop("billing_observations", None)
        return self._commit_finalize(status="done", error=None, **kwargs)

    def commit_finalize_failure(self, **kwargs: Any) -> dict[str, Any]:
        return self._commit_finalize(status="failed", **kwargs)

    def _commit_finalize(
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
        with self._lock:
            record = self._load(str(request_id))
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
            self._save(record)
            return {"ok": True, "idempotent": False, "record": dict(record)}

    def requeue_task(self, *, request_id: str, scheduler_epoch: int, reason: str, now: float | None = None) -> dict[str, Any]:
        ts = _now(now)
        with self._lock:
            self._assert_scheduler_owner(scheduler_epoch=scheduler_epoch, now=ts)
            record = self._load(str(request_id))
            if str(record.get("status")) in TERMINAL_TASK_STATUSES:
                return {"ok": False, "reason": "terminal", "record": dict(record)}
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
            self._save(record)
            return {"ok": True, "record": dict(record)}

    def list_active_tasks(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        with self._lock:
            for record in self._all_records():
                if str(record.get("status")) in {"pending", "assigned", "leased", "finalizing"}:
                    out.append(dict(record))
                    if limit is not None and len(out) >= max(0, int(limit)):
                        break
        return out

    def list_expired_leases(self, *, now: float | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        ts = _now(now)
        out: list[dict[str, Any]] = []
        with self._lock:
            for record in self._all_records():
                if str(record.get("status")) not in {"leased", "finalizing"}:
                    continue
                if float(record.get("finalizing_until") or record.get("lease_expires_at") or 0.0) > ts:
                    continue
                out.append(dict(record))
                if limit is not None and len(out) >= max(0, int(limit)):
                    break
        return out

    def get_task(self, request_id: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._load(str(request_id)))

    def _terminal_completed_at(self, record: dict[str, Any]) -> float:
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

    def future_metrics_stats(self, *, now: float | None = None) -> dict[str, Any]:
        ts = _now(now)
        records = self._all_records()
        by_status: dict[str, int] = {}
        by_op: dict[str, dict[str, int]] = {}
        pending_ages: list[float] = []
        done_ages: list[float] = []
        refs = 0
        meta = 0
        for record in records:
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
        from .task_state_store import future_timeout_metrics_snapshot

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
            },
            "timeout_counts": future_timeout_metrics_snapshot(),
        }


class _FutureStateStoreActor:
    def __init__(self, db_path: str | None = None) -> None:
        try:
            from ..logging_context import init_actor_observability

            init_actor_observability()
        except Exception:
            pass
        self._started_at = time.time()
        self._lock = threading.RLock()
        self._store = FutureStateStore(db_path or _future_state_store_db_path())
        self._watchers: dict[str, list[threading.Event]] = {}
        self._watcher_count = 0
        self._watcher_limit = max(1, int(os.environ.get("MINT_FUTURE_STATE_STORE_WATCHER_MAX", "8192")))
        self._stats_cache: dict[str, Any] | None = None
        self._stats_cache_at = 0.0
        self._stats_cache_ttl_s = max(0.0, float(os.environ.get("MINT_FUTURE_STATE_STORE_STATS_CACHE_TTL_S", "5")))
        self._stats_lock = threading.Lock()
        self._otel_enabled = False
        self._otel_error: str | None = None
        self._init_otel_metrics()

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

    def _read_task_or_none(self, request_id: str) -> dict[str, Any] | None:
        try:
            return self._store.get_task(str(request_id))
        except TaskStateNotFoundError:
            return None

    @staticmethod
    def _record_changed(record: dict[str, Any] | None, *, baseline_status: str, baseline_updated_at: float, terminal_only: bool = False) -> bool:
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

    def _add_watcher(self, request_id: str, event: threading.Event) -> bool:
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

    def wait_task_status_change(self, *, request_id: str, timeout_s: float, observed_status: str | None = None, observed_updated_at: float | None = None, terminal_only: bool = False) -> dict[str, Any]:
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
        if self._record_changed(record, baseline_status=baseline_status, baseline_updated_at=baseline_updated_at, terminal_only=bool(terminal_only)):
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
                return {"changed": False, "watch_skipped": True, "reason": "watcher_limit", "record": latest}
            try:
                latest = self._read_task_or_none(request_id)
                if self._record_changed(latest, baseline_status=baseline_status, baseline_updated_at=baseline_updated_at, terminal_only=bool(terminal_only)):
                    if latest is None:
                        return {"changed": True, "missing": True, "request_id": request_id}
                    return {"changed": True, "record": latest}
                if not event.wait(timeout=max(0.0, remaining_s)):
                    latest = self._read_task_or_none(request_id)
                    if self._record_changed(latest, baseline_status=baseline_status, baseline_updated_at=baseline_updated_at, terminal_only=bool(terminal_only)):
                        if latest is None:
                            return {"changed": True, "missing": True, "request_id": request_id}
                        return {"changed": True, "record": latest}
                    return {"changed": False, "timeout": True, "record": latest or record}
            finally:
                self._remove_watcher(request_id, event)

    def _call_and_notify(self, method: str, **kwargs: Any) -> Any:
        out = getattr(self._store, method)(**kwargs)
        self._notify_task_changed(kwargs.get("request_id"))
        return out

    def ping(self) -> dict[str, Any]:
        out = self._store.ping()
        return {**out, "actor_name": _ray_future_state_store_actor_name(), "namespace": _ray_namespace(), "started_at": self._started_at}

    def stats(self) -> dict[str, Any]:
        now = time.monotonic()
        cached = self._stats_cache
        if cached is not None and now - self._stats_cache_at <= self._stats_cache_ttl_s:
            return dict(cached)
        with self._stats_lock:
            future_stats = self._store.future_metrics_stats()
            active = self._store.list_active_tasks()
            by_status: dict[str, int] = {}
            for record in active:
                status = str(record.get("status") or "unknown")
                by_status[status] = by_status.get(status, 0) + 1
            from .task_state_store import task_future_reaper_metrics_snapshot

            out = {
                "actor_name": _ray_future_state_store_actor_name(),
                "namespace": _ray_namespace(),
                "db_path": self._store.db_path,
                "started_at": self._started_at,
                "active_tasks": len(active),
                "active_by_status": by_status,
                "watchers": self._watcher_count,
                **future_stats,
                "task_future_reaper": task_future_reaper_metrics_snapshot(),
            }
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

            meter = metrics.get_meter("mint.future_state_store")

            def _attrs(**extra: object) -> dict[str, str]:
                attrs = _otel_metric_attrs()
                for key, value in extra.items():
                    text = str(value if value is not None else "").strip()
                    if text:
                        attrs[key] = text
                return attrs

            def _scalar(field: str):
                def _callback(_options):
                    value = _metric_number(self.stats().get(field))
                    if value is None:
                        return []
                    return [Observation(value, _attrs())]

                return _callback

            for name in (
                "pending",
                "results",
                "errors",
                "refs",
                "meta",
                "expired",
                "retrieved",
            ):
                meter.create_observable_gauge(f"mint_task_futures_{name}", callbacks=[_scalar(name)])
            self._otel_enabled = True
        except Exception as e:
            self._otel_error = f"{type(e).__name__}: {e}"

    def close(self) -> None:
        self._store.close()

    def acquire_scheduler_owner(self, **kwargs: Any) -> dict[str, Any]:
        return self._store.acquire_scheduler_owner(**kwargs)

    def renew_scheduler_owner(self, **kwargs: Any) -> dict[str, Any]:
        return self._store.renew_scheduler_owner(**kwargs)

    def list_terminal_payloads_for_eviction(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self._store.list_terminal_payloads_for_eviction(**kwargs)

    def list_staged_payloads_for_gc(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self._store.list_staged_payloads_for_gc(**kwargs)

    def list_tasks_by_metadata(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self._store.list_tasks_by_metadata(**kwargs)

    def list_active_tasks(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self._store.list_active_tasks(**kwargs)

    def list_expired_leases(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self._store.list_expired_leases(**kwargs)

    def get_task(self, request_id: str) -> dict[str, Any]:
        return self._store.get_task(request_id)

    def create_task(self, **kwargs: Any) -> dict[str, Any]:
        return self._call_and_notify("create_task", **kwargs)

    def ensure_task(self, **kwargs: Any) -> dict[str, Any]:
        return self._call_and_notify("ensure_task", **kwargs)

    def update_task_metadata(self, **kwargs: Any) -> dict[str, Any]:
        return self._call_and_notify("update_task_metadata", **kwargs)

    def complete_task_success(self, **kwargs: Any) -> dict[str, Any]:
        return self._call_and_notify("complete_task_success", **kwargs)

    def complete_task_failure(self, **kwargs: Any) -> dict[str, Any]:
        return self._call_and_notify("complete_task_failure", **kwargs)

    def mark_task_retrieved(self, **kwargs: Any) -> dict[str, Any]:
        return self._call_and_notify("mark_task_retrieved", **kwargs)

    def forget_task(self, **kwargs: Any) -> dict[str, Any]:
        return self._call_and_notify("forget_task", **kwargs)

    def expire_active_tasks(self, **kwargs: Any) -> list[str]:
        out = self._store.expire_active_tasks(**kwargs)
        for request_id in out:
            self._notify_task_changed(str(request_id))
        return out

    def mark_payload_evicted(self, **kwargs: Any) -> dict[str, Any]:
        return self._call_and_notify("mark_payload_evicted", **kwargs)

    def delete_expired_tombstones(self, **kwargs: Any) -> list[str]:
        out = self._store.delete_expired_tombstones(**kwargs)
        for request_id in out:
            self._notify_task_changed(str(request_id))
        return out

    def record_payload_evict_error(self, **kwargs: Any) -> dict[str, Any]:
        out = self._store.record_payload_evict_error(**kwargs)
        self._invalidate_stats_cache()
        return out

    def mark_staged_payload_gc_deleted(self, **kwargs: Any) -> dict[str, Any]:
        return self._call_and_notify("mark_staged_payload_gc_deleted", **kwargs)

    def assign_task(self, **kwargs: Any) -> dict[str, Any]:
        return self._call_and_notify("assign_task", **kwargs)

    def claim_task(self, **kwargs: Any) -> dict[str, Any]:
        return self._call_and_notify("claim_task", **kwargs)

    def begin_finalize(self, **kwargs: Any) -> dict[str, Any]:
        return self._call_and_notify("begin_finalize", **kwargs)

    def stage_payload(self, **kwargs: Any) -> dict[str, Any]:
        return self._call_and_notify("stage_payload", **kwargs)

    def commit_finalize_success(self, **kwargs: Any) -> dict[str, Any]:
        return self._call_and_notify("commit_finalize_success", **kwargs)

    def commit_finalize_failure(self, **kwargs: Any) -> dict[str, Any]:
        return self._call_and_notify("commit_finalize_failure", **kwargs)

    def requeue_task(self, **kwargs: Any) -> dict[str, Any]:
        return self._call_and_notify("requeue_task", **kwargs)


def _create_ray_actor(*, require_ready: bool = True):
    try:
        import ray
    except Exception as e:
        raise TaskStateStoreUnavailableError("Ray import failed") from e

    actor_name = _ray_future_state_store_actor_name()
    namespace = _ray_namespace()
    db_path = _future_state_store_db_path()
    max_concurrency = int(os.environ.get("MINT_FUTURE_STATE_STORE_ACTOR_MAX_CONCURRENCY", "32"))

    @ray.remote(num_cpus=0, max_concurrency=max_concurrency, max_restarts=0)
    class _RayFutureStateStoreActor(_FutureStateStoreActor):
        pass

    options: dict[str, Any] = {
        "name": actor_name,
        "namespace": namespace,
        "lifetime": "detached",
        "get_if_exists": True,
        "runtime_env": actor_runtime_env(pythonpath=PFS_PYTHONPATH, extra=otel_env_vars()),
    }
    apply_detached_actor_resources(options, ray)
    actor = _RayFutureStateStoreActor.options(**options).remote(db_path)
    if require_ready:
        out = sync_get_ray_ref(actor.ping.remote(), timeout_s=5.0)
        if not isinstance(out, dict):
            raise TypeError(f"FutureStateStore.ping returned non-dict: {type(out)}")
    return actor


class FutureStateStoreClient:
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
        if self._ray_actor is not None:
            if not require_ready:
                return self._ray_actor
            try:
                out = sync_get_ray_ref(self._ray_actor.ping.remote(), timeout_s=1.0)
                if not isinstance(out, dict):
                    raise TypeError(f"FutureStateStore.ping returned non-dict: {type(out)}")
                return self._ray_actor
            except Exception:
                self._reset_ray_actor()
        actor_name = _ray_future_state_store_actor_name()
        try:
            self._ray_actor = ray.get_actor(actor_name, namespace=_ray_namespace())
        except Exception:
            if not create_if_missing:
                raise TaskStateStoreUnavailableError(
                    f"Detached Ray FutureStateStore actor unavailable actor_name={actor_name!r}"
                )
            try:
                self._ray_actor = _create_ray_actor(require_ready=require_ready)
            except Exception as e:
                raise TaskStateStoreUnavailableError(
                    "Failed to get/create detached Ray FutureStateStore actor"
                ) from e
        return self._ray_actor

    async def _get_ray_actor_async(self, *, require_ready: bool = True, create_if_missing: bool = False):
        try:
            import ray
        except Exception as e:
            raise TaskStateStoreUnavailableError("Ray import failed") from e
        if not ray.is_initialized():
            raise TaskStateStoreUnavailableError("Ray not initialized")
        if self._ray_actor is not None:
            if not require_ready:
                return self._ray_actor
            try:
                out = await async_get_ray_ref(self._ray_actor.ping.remote(), timeout_s=1.0)
                if not isinstance(out, dict):
                    raise TypeError(f"FutureStateStore.ping returned non-dict: {type(out)}")
                return self._ray_actor
            except Exception:
                self._reset_ray_actor()
        import ray

        actor_name = _ray_future_state_store_actor_name()
        try:
            self._ray_actor = await asyncio.to_thread(ray.get_actor, actor_name, namespace=_ray_namespace())
        except Exception:
            if not create_if_missing:
                raise TaskStateStoreUnavailableError(
                    f"Detached Ray FutureStateStore actor unavailable actor_name={actor_name!r}"
                )
            try:
                self._ray_actor = _create_ray_actor(require_ready=require_ready)
            except Exception as e:
                raise TaskStateStoreUnavailableError(
                    "Failed to get/create detached Ray FutureStateStore actor"
                ) from e
        return self._ray_actor

    async def _call(self, method: str, **kwargs: Any) -> Any:
        actor = await self._get_ray_actor_async()
        remote = getattr(actor, method).remote
        return await async_get_ray_ref(remote(**kwargs))

    def _call_sync(self, method: str, **kwargs: Any) -> Any:
        actor = self._get_ray_actor_sync()
        remote = getattr(actor, method).remote
        return sync_get_ray_ref(remote(**kwargs))

    async def _dict_call(self, method: str, **kwargs: Any) -> dict[str, Any]:
        out = await self._call(method, **kwargs)
        if not isinstance(out, dict):
            raise TypeError(f"FutureStateStore.{method} returned non-dict: {type(out)}")
        return out

    def ensure_ready(self, *, timeout_s: float = 10.0, create_if_missing: bool = False) -> dict[str, Any]:
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
            raise TypeError(f"FutureStateStore.ping returned non-dict: {type(out)}")
        return out

    def ensure_started(self, *, timeout_s: float = 10.0) -> dict[str, Any]:
        actor = self._get_ray_actor_sync(require_ready=False, create_if_missing=True)
        out = sync_get_ray_ref(actor.ping.remote(), timeout_s=timeout_s)
        if not isinstance(out, dict):
            raise TypeError(f"FutureStateStore.ping returned non-dict: {type(out)}")
        return out

    def ping(self, *, timeout_s: float = 5.0) -> dict[str, Any]:
        actor = self._get_ray_actor_sync(require_ready=False, create_if_missing=False)
        try:
            out = sync_get_ray_ref(actor.ping.remote(), timeout_s=timeout_s)
        except Exception:
            self._reset_ray_actor()
            raise
        if not isinstance(out, dict):
            raise TypeError(f"FutureStateStore.ping returned non-dict: {type(out)}")
        if not bool(out.get("ok")):
            raise TaskStateStoreUnavailableError(f"FutureStateStore ping failed: {out!r}")
        return out

    async def async_ensure_started(self) -> None:
        await self._get_ray_actor_async(require_ready=False, create_if_missing=True)

    async def async_ensure_ready(self, *, timeout_s: float = 10.0, create_if_missing: bool = False) -> dict[str, Any]:
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
            raise TypeError(f"FutureStateStore.ping returned non-dict: {type(out)}")
        return out

    async def async_ping(self, *, timeout_s: float = 5.0) -> dict[str, Any]:
        actor = await self._get_ray_actor_async(require_ready=False, create_if_missing=False)
        try:
            out = await async_get_ray_ref(actor.ping.remote(), timeout_s=timeout_s)
        except Exception:
            self._reset_ray_actor()
            raise
        if not isinstance(out, dict):
            raise TypeError(f"FutureStateStore.ping returned non-dict: {type(out)}")
        if not bool(out.get("ok")):
            raise TaskStateStoreUnavailableError(f"FutureStateStore ping failed: {out!r}")
        return out

    async def async_stats(self) -> dict[str, Any]:
        return await self._dict_call("stats")

    async def async_acquire_scheduler_owner(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("acquire_scheduler_owner", **kwargs)

    async def async_renew_scheduler_owner(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("renew_scheduler_owner", **kwargs)

    async def async_create_task(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("create_task", **kwargs)

    async def async_ensure_task(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("ensure_task", **kwargs)

    async def async_update_task_metadata(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("update_task_metadata", **kwargs)

    async def async_stage_payload(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("stage_payload", **kwargs)

    async def async_complete_task_success(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.pop("billing_observations", None)
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
            raise TypeError(f"FutureStateStore.expire_active_tasks returned non-list: {type(out)}")
        return [str(x) for x in out]

    async def async_list_terminal_payloads_for_eviction(self, **kwargs: Any) -> list[dict[str, Any]]:
        out = await self._call("list_terminal_payloads_for_eviction", **kwargs)
        if not isinstance(out, list):
            raise TypeError(f"FutureStateStore.list_terminal_payloads_for_eviction returned non-list: {type(out)}")
        return out

    async def async_mark_payload_evicted(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("mark_payload_evicted", **kwargs)

    async def async_delete_expired_tombstones(self, **kwargs: Any) -> list[str]:
        out = await self._call("delete_expired_tombstones", **kwargs)
        if not isinstance(out, list):
            raise TypeError(f"FutureStateStore.delete_expired_tombstones returned non-list: {type(out)}")
        return [str(x) for x in out]

    async def async_record_payload_evict_error(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("record_payload_evict_error", **kwargs)

    async def async_list_staged_payloads_for_gc(self, **kwargs: Any) -> list[dict[str, Any]]:
        out = await self._call("list_staged_payloads_for_gc", **kwargs)
        if not isinstance(out, list):
            raise TypeError(f"FutureStateStore.list_staged_payloads_for_gc returned non-list: {type(out)}")
        return out

    async def async_mark_staged_payload_gc_deleted(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("mark_staged_payload_gc_deleted", **kwargs)

    async def async_assign_task(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("assign_task", **kwargs)

    async def async_claim_task(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("claim_task", **kwargs)

    async def async_begin_finalize(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("begin_finalize", **kwargs)

    async def async_commit_finalize_success(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.pop("billing_observations", None)
        return await self._dict_call("commit_finalize_success", **kwargs)

    async def async_commit_finalize_failure(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("commit_finalize_failure", **kwargs)

    async def async_requeue_task(self, **kwargs: Any) -> dict[str, Any]:
        return await self._dict_call("requeue_task", **kwargs)

    async def async_get_task(self, request_id: str) -> dict[str, Any]:
        return await self._dict_call("get_task", request_id=str(request_id))

    async def async_wait_task_status_change(self, *, request_id: str, timeout_s: float, observed_status: str | None = None, observed_updated_at: float | None = None, terminal_only: bool = False) -> dict[str, Any]:
        return await self._dict_call(
            "wait_task_status_change",
            request_id=str(request_id),
            timeout_s=float(timeout_s),
            observed_status=observed_status,
            observed_updated_at=observed_updated_at,
            terminal_only=bool(terminal_only),
        )

    async def async_list_active_tasks(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        out = await self._call("list_active_tasks", limit=limit)
        if not isinstance(out, list):
            raise TypeError(f"FutureStateStore.list_active_tasks returned non-list: {type(out)}")
        return out

    async def async_list_expired_leases(self, *, now: float | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        out = await self._call("list_expired_leases", now=now, limit=limit)
        if not isinstance(out, list):
            raise TypeError(f"FutureStateStore.list_expired_leases returned non-list: {type(out)}")
        return out

    async def async_list_tasks_by_metadata(self, *, filters: dict[str, Any] | None = None, statuses: list[str] | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        out = await self._call("list_tasks_by_metadata", filters=filters, statuses=statuses, limit=limit)
        if not isinstance(out, list):
            raise TypeError(f"FutureStateStore.list_tasks_by_metadata returned non-list: {type(out)}")
        return out


future_state_store = FutureStateStoreClient()
