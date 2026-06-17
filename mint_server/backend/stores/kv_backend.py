from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Protocol

from mint_server.logging_context import record_store_op_otel


class KVBackend(Protocol):
    """Small internal KV contract for TaskStateStore-owned state machines.

    `apply_batch` is a single-backend atomic mutation: callers may assume that
    all puts/deletes become visible together, or none become visible if the call
    fails before commit. Deletes are applied before puts within the batch.

    `keys(prefix, limit, after)` returns keys in lexicographic order. `prefix`
    restricts the returned range, `limit` is pushed into backend iteration, and
    `after` is an exclusive key cursor within that same lexicographic order.
    """

    def get(self, key: str) -> str | None: ...

    def put(self, key: str, value: str) -> None: ...

    def delete(self, key: str) -> None: ...

    def keys(self, prefix: str | None = None, *, limit: int | None = None, after: str | None = None) -> list[str]: ...

    def apply_batch(self, puts: dict[str, str] | None = None, deletes: list[str] | None = None) -> None: ...

    def close(self) -> None: ...


class InMemoryKVBackend:
    """Thread-safe in-process KV backend for single actor tests and local mode."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> str | None:
        with self._lock:
            return self._data.get(str(key))

    def put(self, key: str, value: str) -> None:
        with self._lock:
            self._data[str(key)] = str(value)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(str(key), None)

    def keys(self, prefix: str | None = None, *, limit: int | None = None, after: str | None = None) -> list[str]:
        max_rows = None if limit is None else max(0, int(limit))
        after_key = None if after is None else str(after)
        with self._lock:
            out: list[str] = []
            if prefix is None:
                keys = sorted(self._data.keys())
            else:
                prefix = str(prefix)
                keys = (key for key in sorted(self._data.keys()) if key.startswith(prefix))
            for key in keys:
                if after_key is not None and key <= after_key:
                    continue
                out.append(key)
                if max_rows is not None and len(out) >= max_rows:
                    break
            return out

    def apply_batch(self, puts: dict[str, str] | None = None, deletes: list[str] | None = None) -> None:
        with self._lock:
            for key in deletes or []:
                self._data.pop(str(key), None)
            for key, value in (puts or {}).items():
                self._data[str(key)] = str(value)

    def close(self) -> None:
        with self._lock:
            self._data.clear()


_PYTEST_ROCKS_FALLBACKS: dict[str, InMemoryKVBackend] = {}
_PYTEST_ROCKS_FALLBACKS_LOCK = threading.RLock()


class RocksKVBackend:
    """RocksDB-compatible backend for TaskStateStore-owned KV state."""

    def __init__(
        self,
        path: str,
        *,
        unavailable_error: type[Exception],
        memory_fallback_for_pytest: bool = False,
    ) -> None:
        try:
            from rocksdict import Rdict, WriteBatch
        except Exception as e:
            if memory_fallback_for_pytest and "pytest-" in str(path):
                with _PYTEST_ROCKS_FALLBACKS_LOCK:
                    self._fallback = _PYTEST_ROCKS_FALLBACKS.setdefault(str(path), InMemoryKVBackend())
                self._db = None
                self._write_batch_cls = None
                return
            raise unavailable_error("rocksdict is required for persistent KV store") from e
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._fallback = None
        self._db = Rdict(str(path))
        self._write_batch_cls = WriteBatch

    def get(self, key: str) -> str | None:
        _t0 = time.perf_counter()
        try:
            if self._fallback is not None:
                return self._fallback.get(key)
            value = self._db.get(str(key))
            if value is None:
                return None
            if isinstance(value, bytes):
                return value.decode("utf-8")
            return str(value)
        except Exception:
            record_store_op_otel(store="rocks", op="get", status="error", duration_s=time.perf_counter() - _t0)
            raise
        finally:
            record_store_op_otel(store="rocks", op="get", status="ok", duration_s=time.perf_counter() - _t0)

    def put(self, key: str, value: str) -> None:
        _t0 = time.perf_counter()
        try:
            if self._fallback is not None:
                self._fallback.put(key, value)
                return
            self._db[str(key)] = str(value)
        except Exception:
            record_store_op_otel(store="rocks", op="put", status="error", duration_s=time.perf_counter() - _t0)
            raise
        finally:
            record_store_op_otel(store="rocks", op="put", status="ok", duration_s=time.perf_counter() - _t0)

    def delete(self, key: str) -> None:
        if self._fallback is not None:
            self._fallback.delete(key)
            return
        try:
            del self._db[str(key)]
        except KeyError:
            pass

    def keys(self, prefix: str | None = None, *, limit: int | None = None, after: str | None = None) -> list[str]:
        if self._fallback is not None:
            return self._fallback.keys(prefix, limit=limit, after=after)
        max_rows = None if limit is None else max(0, int(limit))
        prefix_str = None if prefix is None else str(prefix)
        after_key = None if after is None else str(after)
        out: list[str] = []
        if max_rows == 0:
            return out
        if prefix_str is None:
            raw_keys = self._db.keys()
            for raw in raw_keys:
                key = str(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
                if after_key is not None and key <= after_key:
                    continue
                out.append(key)
                if max_rows is not None and len(out) >= max_rows:
                    break
            return out

        raw_keys = self._db.keys(from_key=after_key if after_key is not None else prefix_str)
        for raw in raw_keys:
            key = str(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
            if after_key is not None and key <= after_key:
                continue
            if not key.startswith(prefix_str):
                break
            out.append(key)
            if max_rows is not None and len(out) >= max_rows:
                break
        return out

    def apply_batch(self, puts: dict[str, str] | None = None, deletes: list[str] | None = None) -> None:
        _t0 = time.perf_counter()
        try:
            if self._fallback is not None:
                self._fallback.apply_batch(puts=puts, deletes=deletes)
                return
            batch = self._write_batch_cls()
            op_count = 0
            for key in deletes or []:
                batch.delete(str(key))
                op_count += 1
            for key, value in (puts or {}).items():
                batch.put(str(key), str(value))
                op_count += 1
            if op_count:
                self._db.write(batch)
        except Exception:
            record_store_op_otel(store="rocks", op="batch", status="error", duration_s=time.perf_counter() - _t0)
            raise
        finally:
            record_store_op_otel(store="rocks", op="batch", status="ok", duration_s=time.perf_counter() - _t0)

    def close(self) -> None:
        if self._fallback is not None:
            # Path-scoped pytest fallback simulates a persistent local KV. Do
            # not clear it on close, or reopen/recovery tests lose durability.
            return
        close = getattr(self._db, "close", None)
        if callable(close):
            close()
