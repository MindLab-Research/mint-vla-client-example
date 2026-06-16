from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from typing import Any, Callable

from mint_server.backend.stores.kv_backend import InMemoryKVBackend, KVBackend, RocksKVBackend

class TaskHotKVStoreError(RuntimeError):
    pass


class TaskHotKVStoreUnavailableError(TaskHotKVStoreError):
    pass


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
        raise TaskHotKVStoreError(f"expected JSON object, got {type(out).__name__}")
    return out


class TaskHotKVStore:
    """RocksDB-backed hot metadata store owned by TaskStateStore.

    Reads and single-key put/delete operations rely on the KV backend's own
    concurrency. Per-key Python locks are only used for read-modify-write
    transitions that need a compare/update window.
    """

    _STRIPES = 256

    def __init__(self, db_path: str | os.PathLike[str]) -> None:
        self._db_path = str(db_path)
        self._kv: KVBackend = (
            InMemoryKVBackend()
            if self._db_path == ":memory:"
            else RocksKVBackend(
                self._db_path,
                unavailable_error=TaskHotKVStoreUnavailableError,
                memory_fallback_for_pytest=True,
            )
        )
        self._locks = [threading.RLock() for _ in range(self._STRIPES)]
        self._billing_id_lock = threading.RLock()
        self._billing_next_id = 1
        self._rebuild_indexes()

    @classmethod
    def in_memory(cls) -> "TaskHotKVStore":
        return cls(":memory:")

    @property
    def db_path(self) -> str:
        return self._db_path

    def close(self) -> None:
        self._kv.close()

    def _lock_for(self, key: str) -> threading.RLock:
        return self._locks[hash(str(key)) % len(self._locks)]

    def _record_key(self, namespace: str, item_id: str) -> str:
        return f"{namespace}:{str(item_id)}"

    def _get_json(self, key: str) -> dict[str, Any] | None:
        raw = self._kv.get(key)
        return None if raw is None else _json_loads(raw)

    def _put_json(self, key: str, value: dict[str, Any]) -> None:
        self._kv.put(key, _json_dumps(value))

    def _delete(self, key: str) -> bool:
        existed = self._kv.get(key) is not None
        self._kv.delete(key)
        return existed

    def _list_namespace(self, namespace: str) -> list[dict[str, Any]]:
        prefix = f"{namespace}:"
        out: list[dict[str, Any]] = []
        for key in self._kv.keys(prefix=prefix):
            record = self._get_json(key)
            if record is not None:
                out.append(record)
        return out

    def _rebuild_indexes(self) -> None:
        max_id = 0
        for key in list(self._kv.keys(prefix="idx:billing:")):
            self._kv.delete(key)
        for key in self._kv.keys(prefix="billing:"):
            record = self._get_json(key)
            if record is None:
                continue
            outbox_id = int(record.get("outbox_id") or 0)
            if outbox_id <= 0:
                continue
            max_id = max(max_id, outbox_id)
            self._write_billing_indexes(record)
        self._billing_next_id = max_id + 1

    def _next_billing_id(self) -> int:
        with self._billing_id_lock:
            out = self._billing_next_id
            self._billing_next_id += 1
            self._kv.put("__billing_next_id__", str(self._billing_next_id))
            return out

    def _billing_key(self, outbox_id: int) -> str:
        return f"billing:{int(outbox_id):020d}"

    def _billing_status_index_key(self, status: str, outbox_id: int) -> str:
        return f"idx:billing:status:{str(status)}:{int(outbox_id):020d}"

    def _billing_event_index_key(self, event_id: str) -> str:
        return f"idx:billing:event:{str(event_id)}"

    def _billing_status_ids(self, status: str) -> list[int]:
        prefix = f"idx:billing:status:{str(status)}:"
        out: list[int] = []
        for key in self._kv.keys(prefix=prefix):
            value = self._kv.get(key)
            if value is not None:
                out.append(int(value))
        return out

    def _write_billing_indexes(self, record: dict[str, Any]) -> None:
        outbox_id = int(record["outbox_id"])
        status = str(record.get("status") or "pending")
        self._kv.put(self._billing_status_index_key(status, outbox_id), str(outbox_id))
        event_id = str(record.get("event_id") or "")
        if event_id:
            self._kv.put(self._billing_event_index_key(event_id), str(outbox_id))

    def _delete_billing_indexes(self, record: dict[str, Any]) -> None:
        outbox_id = int(record["outbox_id"])
        status = str(record.get("status") or "pending")
        self._kv.delete(self._billing_status_index_key(status, outbox_id))
        event_id = str(record.get("event_id") or "")
        if event_id:
            self._kv.delete(self._billing_event_index_key(event_id))

    def _billing_get(self, outbox_id: int) -> dict[str, Any] | None:
        return self._get_json(self._billing_key(int(outbox_id)))

    def _billing_put(self, record: dict[str, Any], *, old_status: str | None = None) -> None:
        outbox_id = int(record["outbox_id"])
        if old_status and old_status != str(record.get("status") or "pending"):
            self._kv.delete(self._billing_status_index_key(old_status, outbox_id))
        self._put_json(self._billing_key(outbox_id), record)
        self._write_billing_indexes(record)

    def _billing_delete(self, outbox_id: int) -> bool:
        record = self._billing_get(outbox_id)
        existed = self._delete(self._billing_key(outbox_id))
        if record is not None:
            self._delete_billing_indexes(record)
        return existed

    def append_billing_outbox(
        self,
        *,
        observations: list[dict[str, Any]],
        source: str = "unknown",
        now: float | None = None,
    ) -> dict[str, Any]:
        ts = _now(now)
        normalized = [dict(item) for item in observations if isinstance(item, dict)]
        if not normalized:
            return {"ok": True, "source": str(source), "inserted": 0, "duplicate": 0, "conflicts": 0, "errors": []}
        inserted = 0
        duplicate = 0
        conflicts = 0
        errors: list[dict[str, str]] = []
        for observation in normalized:
            try:
                from mint_server.backend.stores.task_state_store import billing_event_from_observation

                event = billing_event_from_observation(observation)
                event_id = str(event["event_id"])
                event_hash = hashlib.sha256(_json_dumps(event).encode("utf-8")).hexdigest()
                with self._lock_for(f"billing_event:{event_id}"):
                    existing_raw = self._kv.get(self._billing_event_index_key(event_id))
                    existing_id = None if existing_raw is None else int(existing_raw)
                    if existing_id is not None:
                        existing = self._billing_get(existing_id)
                        if existing is not None and str(existing.get("event_hash")) == event_hash:
                            duplicate += 1
                        else:
                            conflicts += 1
                        continue
                    outbox_id = self._next_billing_id()
                    record = {
                        "outbox_id": outbox_id,
                        "event_id": event_id,
                        "event_hash": event_hash,
                        "event": event,
                        "status": "pending",
                        "claim_id": None,
                        "claimed_until": None,
                        "attempt_count": 0,
                        "last_error": None,
                        "created_at": ts,
                        "updated_at": ts,
                    }
                    self._billing_put(record)
                inserted += 1
            except Exception as e:
                errors.append({"error": f"{type(e).__name__}: {e}"})
        return {
            "ok": not errors and conflicts == 0,
            "source": str(source),
            "inserted": inserted,
            "duplicate": duplicate,
            "conflicts": conflicts,
            "errors": errors,
        }

    def claim_billing_outbox(
        self,
        *,
        claim_id: str,
        limit: int = 100,
        lease_ttl_s: float = 60.0,
        now: float | None = None,
    ) -> list[dict[str, Any]]:
        ts = _now(now)
        claimed_until = ts + max(1.0, float(lease_ttl_s))
        max_rows = max(1, int(limit))
        candidate_ids = sorted(set(self._billing_status_ids("pending")) | set(self._billing_status_ids("flushing")))
        out: list[dict[str, Any]] = []
        for outbox_id in candidate_ids:
            if len(out) >= max_rows:
                break
            key = self._billing_key(outbox_id)
            with self._lock_for(key):
                record = self._billing_get(outbox_id)
                if record is None:
                    continue
                status = str(record.get("status") or "pending")
                if status == "flushing" and float(record.get("claimed_until") or 0.0) > ts:
                    continue
                if status not in {"pending", "flushing"}:
                    continue
                old_status = status
                record["status"] = "flushing"
                record["claim_id"] = str(claim_id)
                record["claimed_until"] = claimed_until
                record["attempt_count"] = int(record.get("attempt_count") or 0) + 1
                record["updated_at"] = ts
                self._billing_put(record, old_status=old_status)
                out.append(dict(record))
        out.sort(key=lambda row: (float(row.get("created_at") or 0.0), int(row.get("outbox_id") or 0)))
        return out

    def delete_billing_outbox_claim(self, *, claim_id: str, outbox_ids: list[int]) -> dict[str, Any]:
        deleted = 0
        for value in [int(v) for v in outbox_ids]:
            with self._lock_for(self._billing_key(value)):
                record = self._billing_get(value)
                if record is None or str(record.get("claim_id") or "") != str(claim_id):
                    continue
                if self._billing_delete(value):
                    deleted += 1
        return {"ok": True, "deleted": deleted}

    def mark_billing_outbox_claim_failed(
        self,
        *,
        claim_id: str,
        outbox_ids: list[int],
        permanent: bool,
        error: str,
        now: float | None = None,
    ) -> dict[str, Any]:
        ts = _now(now)
        updated = 0
        next_status = "failed" if bool(permanent) else "pending"
        for value in [int(v) for v in outbox_ids]:
            key = self._billing_key(value)
            with self._lock_for(key):
                record = self._billing_get(value)
                if record is None or str(record.get("claim_id") or "") != str(claim_id):
                    continue
                old_status = str(record.get("status") or "pending")
                record["status"] = next_status
                record["claim_id"] = None
                record["claimed_until"] = None
                record["last_error"] = str(error)
                record["updated_at"] = ts
                self._billing_put(record, old_status=old_status)
                updated += 1
        return {"ok": True, "updated": updated}

    def billing_outbox_stats(self, *, now: float | None = None) -> dict[str, Any]:
        ts = _now(now)
        by_status: dict[str, dict[str, Any]] = {}
        for status in ("pending", "flushing", "failed"):
            oldest: float | None = None
            rows = 0
            for outbox_id in self._billing_status_ids(status):
                record = self._billing_get(outbox_id)
                if record is None or str(record.get("status") or "pending") != status:
                    continue
                rows += 1
                created_at = float(record.get("created_at") or ts)
                oldest = created_at if oldest is None else min(oldest, created_at)
            if rows:
                by_status[status] = {
                    "rows": rows,
                    "oldest_age_s": None if oldest is None else max(0.0, ts - oldest),
                }
        return {"by_status": by_status}

    def upsert_record(self, namespace: str, item_id: str, info: dict[str, Any]) -> None:
        key = self._record_key(namespace, item_id)
        incoming = dict(info)
        with self._lock_for(key):
            existing = self._get_json(key) or {}
            self._put_json(key, {**existing, **incoming})

    def mutate_record(
        self,
        namespace: str,
        item_id: str,
        mutator: Callable[[dict[str, Any] | None], dict[str, Any] | None],
    ) -> dict[str, Any] | None:
        key = self._record_key(namespace, item_id)
        with self._lock_for(key):
            existing = self._get_json(key)
            updated = mutator(None if existing is None else dict(existing))
            if updated is None:
                return None
            self._put_json(key, dict(updated))
            return dict(updated)

    def replace_record(self, namespace: str, item_id: str, info: dict[str, Any]) -> None:
        self._put_json(self._record_key(namespace, item_id), dict(info))

    def get_record(self, namespace: str, item_id: str) -> dict[str, Any] | None:
        record = self._get_json(self._record_key(namespace, item_id))
        return dict(record) if record is not None else None

    def delete_record(self, namespace: str, item_id: str) -> bool:
        return self._delete(self._record_key(namespace, item_id))

    def list_records(self, namespace: str) -> list[dict[str, Any]]:
        return [dict(record) for record in self._list_namespace(namespace)]

    def update_heartbeat(self, *, session_id: str, now: float | None = None) -> None:
        if not str(session_id):
            return
        self.replace_record("heartbeat", str(session_id), {"session_id": str(session_id), "last": _now(now)})

    def get_heartbeat(self, *, session_id: str) -> float | None:
        record = self.get_record("heartbeat", str(session_id))
        if record is None or record.get("last") is None:
            return None
        return float(record["last"])

    def heartbeat_size(self) -> int:
        return len(self.list_records("heartbeat"))

    def prune_heartbeats(self, *, max_age_s: float, now: float | None = None) -> int:
        if float(max_age_s) <= 0:
            return 0
        cutoff = _now(now) - float(max_age_s)
        deleted = 0
        for record in self.list_records("heartbeat"):
            if float(record.get("last") or 0.0) < cutoff:
                if self.delete_record("heartbeat", str(record.get("session_id") or "")):
                    deleted += 1
        return deleted
