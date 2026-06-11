from __future__ import annotations

import base64
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from ..config import config as server_config
from .task_state_store import (
    ACTIVE_TASK_STATUSES,
    TERMINAL_TASK_STATUSES,
    TaskStateConflictError,
    TaskStateNotFoundError,
    TaskStateStoreError,
    TaskStateStoreUnavailableError,
)


_SCHEMA_VERSION = 1
_OWNER_PREFIX = "owner:"
_TASK_PREFIX = "task:"
_INDEX_STATUS_PREFIX = "idx:status:"
_INDEX_DOMAIN_PREFIX = "idx:domain:"
_INDEX_META_PREFIX = "idx:meta:"
_INDEX_RESULT_PREFIX = "idx:result:"
_INDEX_STAGED_PREFIX = "idx:staged:"
_INDEX_LEASE_PREFIX = "idx:lease:"
_INDEX_CREATED_PREFIX = "idx:created:"
_INDEX_UPDATED_PREFIX = "idx:updated:"

_META_INDEX_KEYS = (
    "model_id",
    "sampling_session_id",
    "consumer_job_id",
    "op",
    "domain_key",
)


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


class FutureStateStore:
    """Persistent future state machine backed by striped KV locks."""

    def __init__(self, db_path: str | os.PathLike[str]) -> None:
        self._db_path = str(db_path)
        self._owner_lock = threading.RLock()
        self._index_lock = threading.RLock()
        self._locks = [threading.RLock() for _ in range(256)]
        self._kv = _DictKV() if self._db_path == ":memory:" else _RocksKV(self._db_path)
        self._ensure_indexes()

    @classmethod
    def in_memory(cls) -> "FutureStateStore":
        return cls(":memory:")

    @property
    def db_path(self) -> str:
        return self._db_path

    def close(self) -> None:
        self._kv.close()

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
            for key in self._kv.keys():
                if key.startswith(_TASK_PREFIX):
                    request_id = key[len(_TASK_PREFIX):]
                    if self._kv.get(_created_index_key(_decode_record(self._kv.get(key) or ""), request_id)) is None:
                        needs_rebuild = True
                    break
        if not needs_rebuild:
            return
        with self._index_lock:
            for key in list(self._kv.keys()):
                if key.startswith("idx:"):
                    self._kv.delete(key)
            for record in self._all_records_scan():
                self._write_index_for_record(record)
            self._kv.put("__future_index_version__", "2")

    def _all_records_scan(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for key in self._kv.keys():
            if key.startswith(_TASK_PREFIX):
                raw = self._kv.get(key)
                if raw is not None:
                    records.append(_decode_record(raw))
        records.sort(key=lambda r: (float(r.get("created_at") or 0.0), str(r.get("request_id") or "")))
        return records

    def _ids_from_index_prefix(self, prefix: str, *, limit: int | None = None) -> list[str]:
        keys = sorted(key for key in self._kv.keys() if key.startswith(prefix))
        out: list[str] = []
        for key in keys:
            value = self._kv.get(key)
            if value is None:
                continue
            out.append(str(value))
            if limit is not None and len(out) >= max(0, int(limit)):
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
        with self._owner_lock:
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
        with self._owner_lock:
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
                return {"ok": True, "created": True, "record": dict(record)}
            if payload_hash is not None and existing.get("payload_hash") not in (None, payload_hash):
                raise TaskStateConflictError("duplicate request_id with different payload hash")
            if str(existing.get("op")) != str(op) or str(existing.get("domain_key")) != str(domain_key):
                raise TaskStateConflictError("duplicate request_id with different task identity")
            old_record = dict(existing)
            existing["request_json"] = bytes(request_json)
            existing["metadata"] = _merge_metadata(existing, metadata)
            existing["updated_at"] = ts
            if existing.get("payload_hash") is None and payload_hash is not None:
                existing["payload_hash"] = payload_hash
            self._save_indexed(existing, old_record=old_record)
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
    ) -> dict[str, Any]:
        with self._lock_for_request(request_id):
            record = self._load(str(request_id))
            old_record = dict(record)
            record["metadata"] = _merge_metadata(record, metadata)
            if status is not None:
                record["status"] = str(status)
            record["updated_at"] = _now(now)
            self._save_indexed(record, old_record=old_record)
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
        with self._lock_for_request(request_id):
            record = self._load(str(request_id))
            old_record = dict(record)
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
            self._save_indexed(record, old_record=old_record)
            return {"ok": True, "idempotent": False, "record": dict(record)}

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

    def forget_task(self, *, request_id: str) -> dict[str, Any]:
        with self._lock_for_request(request_id):
            existed = self._delete_task_indexed(str(request_id))
            return {"ok": True, "deleted": existed}

    def expire_active_tasks(self, *, older_than_s: float, now: float | None = None, limit: int = 1000) -> list[str]:
        ttl_s = float(older_than_s)
        if ttl_s <= 0:
            return []
        ts = _now(now)
        cutoff = ts - ttl_s
        expired: list[str] = []
        expired_by_op: dict[str, int] = {}
        ids: list[str] = []
        for status in ("pending", "queued", "assigned"):
            ids.extend(self._ids_from_index_prefix(f"{_INDEX_STATUS_PREFIX}{_index_value(status)}:"))
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
        for request_id in self._ids_from_index_prefix(_INDEX_RESULT_PREFIX):
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

    def mark_payload_evicted(self, *, request_id: str, expected_result_path: str, now: float | None = None) -> dict[str, Any]:
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
        from .task_state_store import _inc_reaper_rows

        _inc_reaper_rows("evict_payload", 1)
        return {"ok": True, "record": dict(record)}

    def list_staged_payloads_for_gc(self, *, older_than_s: float, now: float | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        ttl_s = float(older_than_s)
        if ttl_s <= 0:
            return []
        cutoff = _now(now) - ttl_s
        out: list[dict[str, Any]] = []
        candidate_ids = self._ids_from_index_prefix(_INDEX_STAGED_PREFIX)
        candidate_ids.extend(self._ids_from_index_prefix(_INDEX_UPDATED_PREFIX))
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

    def mark_staged_payload_gc_deleted(self, *, request_id: str, expected_staged_payload_path: str, now: float | None = None) -> dict[str, Any]:
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
        from .task_state_store import _inc_reaper_rows

        _inc_reaper_rows("gc_staged_payload", 1)
        return {"ok": True, "record": dict(record)}

    def delete_expired_tombstones(self, *, older_than_s: float, now: float | None = None, limit: int = 1000) -> list[str]:
        ttl_s = float(older_than_s)
        if ttl_s <= 0:
            return []
        cutoff = _now(now) - ttl_s
        deleted: list[str] = []
        for request_id in self._ids_from_index_prefix(_INDEX_UPDATED_PREFIX):
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

    def assign_task(self, *, request_id: str, subqueue_id: str, scheduler_epoch: int, now: float | None = None) -> dict[str, Any]:
        ts = _now(now)
        with self._lock_for_request(request_id):
            self._assert_scheduler_owner(scheduler_epoch=scheduler_epoch, now=ts)
            record = self._load(str(request_id))
            old_record = dict(record)
            if str(record.get("status")) != "pending":
                raise TaskStateConflictError(f"cannot assign from pending; current status={record.get('status')!r}")
            record.update({"status": "assigned", "subqueue_id": str(subqueue_id), "scheduler_epoch": int(scheduler_epoch), "assigned_at": ts, "updated_at": ts})
            self._save_indexed(record, old_record=old_record)
            return {"ok": True, "record": dict(record)}

    def claim_task(self, *, request_id: str, subqueue_id: str, lease_id: str, attempt_id: str, consumer_id: str, scheduler_epoch: int, runtime_generation: int, lease_ttl_s: float, now: float | None = None) -> dict[str, Any]:
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
            return {"ok": True, "record": dict(record)}

    def begin_finalize(self, *, request_id: str, lease_id: str, attempt_id: str, scheduler_epoch: int, runtime_generation: int, finalize_ttl_s: float, staged_payload_path: str | None = None, now: float | None = None) -> dict[str, Any]:
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

    def requeue_task(self, *, request_id: str, scheduler_epoch: int, reason: str, now: float | None = None) -> dict[str, Any]:
        ts = _now(now)
        with self._lock_for_request(request_id):
            self._assert_scheduler_owner(scheduler_epoch=scheduler_epoch, now=ts)
            record = self._load(str(request_id))
            old_record = dict(record)
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
            self._save_indexed(record, old_record=old_record)
            return {"ok": True, "record": dict(record)}

    def list_active_tasks(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        ids: list[str] = []
        remaining = None if limit is None else max(0, int(limit))
        if remaining == 0:
            return []
        for status in ("pending", "queued", "assigned", "leased", "running", "finalizing"):
            ids.extend(
                self._ids_from_index_prefix(
                    f"{_INDEX_STATUS_PREFIX}{_index_value(status)}:",
                    limit=remaining,
                )
            )
            if remaining is not None:
                remaining = max(0, int(limit) - len(ids))
                if remaining == 0:
                    break
        return self._records_from_ids(ids, limit=limit)

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
                "records_scanned": records_scanned,
            },
            "timeout_counts": future_timeout_metrics_snapshot(),
        }
