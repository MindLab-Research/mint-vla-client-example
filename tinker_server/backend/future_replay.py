from __future__ import annotations

import json
import logging
import os
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import config

logger = logging.getLogger(__name__)

REPLAYABLE_TRAINING_OPS = frozenset(
    {
        "training.forward_backward",
        "training.optim_step",
        "training.train_step",
    }
)
_INDEX_DIRNAME = "index"
_INDEX_FILENAME = "future_replay.sqlite3"
_OBJECTS_DIRNAME = "objects"


@dataclass(frozen=True)
class ReplayEntry:
    request_id: str
    op: str
    model_id: str | None
    final_status: str
    done_at_ts: float
    retrieved_at_ts: float | None
    payload_expires_at_ts: float
    object_relpath: str
    codec: str
    size_bytes: int | None
    payload_deleted_at_ts: float | None


@dataclass(frozen=True)
class ReplayLookupResult:
    state: str
    entry: ReplayEntry | None = None
    envelope: dict[str, Any] | None = None


class SQLiteReplayIndexStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._ready = False

    def _connect(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS future_replay (
                    request_id TEXT PRIMARY KEY,
                    op TEXT NOT NULL,
                    model_id TEXT,
                    final_status TEXT NOT NULL,
                    done_at_ts REAL NOT NULL,
                    retrieved_at_ts REAL,
                    payload_expires_at_ts REAL NOT NULL,
                    object_relpath TEXT NOT NULL,
                    codec TEXT NOT NULL,
                    size_bytes INTEGER,
                    payload_deleted_at_ts REAL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_future_replay_expiry
                ON future_replay(payload_expires_at_ts)
                """
            )
            conn.commit()
            self._ready = True
        finally:
            conn.close()

    def upsert(self, entry: ReplayEntry) -> None:
        self._ensure_ready()
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO future_replay (
                    request_id,
                    op,
                    model_id,
                    final_status,
                    done_at_ts,
                    retrieved_at_ts,
                    payload_expires_at_ts,
                    object_relpath,
                    codec,
                    size_bytes,
                    payload_deleted_at_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(request_id) DO UPDATE SET
                    op=excluded.op,
                    model_id=excluded.model_id,
                    final_status=excluded.final_status,
                    done_at_ts=excluded.done_at_ts,
                    retrieved_at_ts=excluded.retrieved_at_ts,
                    payload_expires_at_ts=excluded.payload_expires_at_ts,
                    object_relpath=excluded.object_relpath,
                    codec=excluded.codec,
                    size_bytes=excluded.size_bytes,
                    payload_deleted_at_ts=excluded.payload_deleted_at_ts
                """,
                (
                    entry.request_id,
                    entry.op,
                    entry.model_id,
                    entry.final_status,
                    float(entry.done_at_ts),
                    None if entry.retrieved_at_ts is None else float(entry.retrieved_at_ts),
                    float(entry.payload_expires_at_ts),
                    entry.object_relpath,
                    entry.codec,
                    entry.size_bytes,
                    None if entry.payload_deleted_at_ts is None else float(entry.payload_deleted_at_ts),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get(self, request_id: str) -> ReplayEntry | None:
        self._ensure_ready()
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM future_replay WHERE request_id = ?",
                (str(request_id),),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return ReplayEntry(
            request_id=str(row["request_id"]),
            op=str(row["op"]),
            model_id=None if row["model_id"] is None else str(row["model_id"]),
            final_status=str(row["final_status"]),
            done_at_ts=float(row["done_at_ts"]),
            retrieved_at_ts=None if row["retrieved_at_ts"] is None else float(row["retrieved_at_ts"]),
            payload_expires_at_ts=float(row["payload_expires_at_ts"]),
            object_relpath=str(row["object_relpath"]),
            codec=str(row["codec"]),
            size_bytes=None if row["size_bytes"] is None else int(row["size_bytes"]),
            payload_deleted_at_ts=None if row["payload_deleted_at_ts"] is None else float(row["payload_deleted_at_ts"]),
        )

    def list_expired_payloads(self, cutoff_ts: float, limit: int) -> list[ReplayEntry]:
        self._ensure_ready()
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT * FROM future_replay
                WHERE payload_deleted_at_ts IS NULL AND payload_expires_at_ts <= ?
                ORDER BY payload_expires_at_ts ASC
                LIMIT ?
                """,
                (float(cutoff_ts), int(limit)),
            ).fetchall()
        finally:
            conn.close()
        return [
            ReplayEntry(
                request_id=str(row["request_id"]),
                op=str(row["op"]),
                model_id=None if row["model_id"] is None else str(row["model_id"]),
                final_status=str(row["final_status"]),
                done_at_ts=float(row["done_at_ts"]),
                retrieved_at_ts=None if row["retrieved_at_ts"] is None else float(row["retrieved_at_ts"]),
                payload_expires_at_ts=float(row["payload_expires_at_ts"]),
                object_relpath=str(row["object_relpath"]),
                codec=str(row["codec"]),
                size_bytes=None if row["size_bytes"] is None else int(row["size_bytes"]),
                payload_deleted_at_ts=None if row["payload_deleted_at_ts"] is None else float(row["payload_deleted_at_ts"]),
            )
            for row in rows
        ]

    def mark_payload_deleted(self, request_id: str, deleted_at_ts: float) -> None:
        self._ensure_ready()
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE future_replay SET payload_deleted_at_ts = ? WHERE request_id = ?",
                (float(deleted_at_ts), str(request_id)),
            )
            conn.commit()
        finally:
            conn.close()


def should_persist_training_future(op: str | None) -> bool:
    if not isinstance(op, str):
        return False
    return op.strip() in REPLAYABLE_TRAINING_OPS


class FutureReplayStore:
    def __init__(self, root_dir: Path) -> None:
        self._root_dir = Path(root_dir)
        self._index = SQLiteReplayIndexStore(self._root_dir / _INDEX_DIRNAME / _INDEX_FILENAME)

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    def index_get(self, request_id: str) -> ReplayEntry | None:
        return self._index.get(request_id)

    def persist_terminal_payload(
        self,
        *,
        request_id: str,
        op: str,
        model_id: str | None,
        final_status: str,
        payload: Any,
        done_at_ts: float,
        retrieved_at_ts: float,
    ) -> ReplayEntry:
        existing = self._index.get(request_id)
        if existing is not None:
            return existing

        object_relpath = _object_relpath(request_id)
        object_path = self._root_dir / object_relpath
        envelope = {
            "schema_version": 1,
            "request_id": str(request_id),
            "op": str(op),
            "model_id": None if model_id is None else str(model_id),
            "final_status": str(final_status),
            "done_at": _iso8601(done_at_ts),
            "retrieved_at": _iso8601(retrieved_at_ts),
            "payload_expires_at": _iso8601(retrieved_at_ts + float(config.future_replay_disk_ttl_s)),
            "payload": payload,
        }
        _write_json_atomic(object_path, envelope)
        size_bytes = object_path.stat().st_size if object_path.exists() else None
        entry = ReplayEntry(
            request_id=str(request_id),
            op=str(op),
            model_id=None if model_id is None else str(model_id),
            final_status=str(final_status),
            done_at_ts=float(done_at_ts),
            retrieved_at_ts=float(retrieved_at_ts),
            payload_expires_at_ts=float(retrieved_at_ts + float(config.future_replay_disk_ttl_s)),
            object_relpath=str(object_relpath),
            codec="json",
            size_bytes=None if size_bytes is None else int(size_bytes),
            payload_deleted_at_ts=None,
        )
        self._index.upsert(entry)
        return entry

    def lookup(self, request_id: str, *, now_ts: float | None = None) -> ReplayLookupResult:
        entry = self._index.get(request_id)
        if entry is None:
            return ReplayLookupResult(state="miss")
        now = time.time() if now_ts is None else float(now_ts)
        if entry.payload_deleted_at_ts is not None or entry.payload_expires_at_ts <= now:
            return ReplayLookupResult(state="evicted", entry=entry)

        path = self._root_dir / entry.object_relpath
        if not path.exists():
            deleted_at_ts = time.time()
            self._index.mark_payload_deleted(request_id, deleted_at_ts)
            entry = ReplayEntry(
                request_id=entry.request_id,
                op=entry.op,
                model_id=entry.model_id,
                final_status=entry.final_status,
                done_at_ts=entry.done_at_ts,
                retrieved_at_ts=entry.retrieved_at_ts,
                payload_expires_at_ts=entry.payload_expires_at_ts,
                object_relpath=entry.object_relpath,
                codec=entry.codec,
                size_bytes=entry.size_bytes,
                payload_deleted_at_ts=deleted_at_ts,
            )
            return ReplayLookupResult(state="evicted", entry=entry)

        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("future replay read failed: request_id=%s path=%s", request_id, path)
            return ReplayLookupResult(state="evicted", entry=entry)
        return ReplayLookupResult(state="ok", entry=entry, envelope=envelope)

    def sweep_expired_payloads(self, *, now_ts: float | None = None, limit: int = 256) -> dict[str, int]:
        now = time.time() if now_ts is None else float(now_ts)
        deleted = 0
        entries = self._index.list_expired_payloads(cutoff_ts=now, limit=limit)
        for entry in entries:
            path = self._root_dir / entry.object_relpath
            try:
                if path.exists():
                    path.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                logger.exception("future replay payload delete failed: request_id=%s path=%s", entry.request_id, path)
                continue
            self._index.mark_payload_deleted(entry.request_id, now)
            deleted += 1
        return {"deleted": deleted}


def future_replay_store() -> FutureReplayStore:
    return FutureReplayStore(Path(str(config.future_replay_root_dir or "")).expanduser())


def ensure_future_replay_sweeper(*, timeout_s: float = 10.0) -> dict[str, Any]:
    actor = _get_or_create_sweeper_actor()
    import ray

    return ray.get(actor.poke.remote(), timeout=float(timeout_s))


def _future_replay_sweeper_actor_name() -> str:
    return os.environ.get("MINT_FUTURE_REPLAY_SWEEPER_ACTOR_NAME", "mint_future_replay_sweeper")


def _get_or_create_sweeper_actor():
    from .future_store import _ray_namespace

    import ray

    actor_name = _future_replay_sweeper_actor_name()
    namespace = _ray_namespace()
    sweep_interval_s = max(1.0, float(config.future_replay_sweep_interval_s))

    try:
        return ray.get_actor(actor_name, namespace=namespace)
    except ValueError:
        pass

    @ray.remote(num_cpus=0)
    class _RayFutureReplaySweeperActor:
        def __init__(self, interval_s: float) -> None:
            import threading

            from ..logging_context import init_actor_observability

            init_actor_observability()
            self._interval_s = max(1.0, float(interval_s))
            self._last_sweep_ts = 0.0
            self._last_deleted = 0
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

        def _loop(self) -> None:
            while True:
                try:
                    deleted = future_replay_store().sweep_expired_payloads()["deleted"]
                    self._last_deleted = int(deleted)
                    self._last_sweep_ts = time.time()
                except Exception:
                    logger.exception("future replay sweeper loop failed")
                time.sleep(self._interval_s)

        def poke(self) -> dict[str, Any]:
            deleted = future_replay_store().sweep_expired_payloads()["deleted"]
            self._last_deleted = int(deleted)
            self._last_sweep_ts = time.time()
            return self.stats()

        def stats(self) -> dict[str, Any]:
            return {
                "interval_s": float(self._interval_s),
                "last_sweep_ts": float(self._last_sweep_ts),
                "last_deleted": int(self._last_deleted),
                "root_dir": str(future_replay_store().root_dir),
            }

    from ..config import PFS_PYTHONPATH, actor_runtime_env_vars, apply_detached_actor_resources, otel_env_vars

    options: dict[str, Any] = {
        "name": actor_name,
        "namespace": namespace,
        "lifetime": "detached",
        "runtime_env": {
            "env_vars": actor_runtime_env_vars(
                pythonpath=PFS_PYTHONPATH,
                extra=otel_env_vars(),
            )
        },
    }
    apply_detached_actor_resources(options, ray)

    try:
        return _RayFutureReplaySweeperActor.options(**options).remote(sweep_interval_s)
    except Exception:
        return ray.get_actor(actor_name, namespace=namespace)


def _object_relpath(request_id: str) -> Path:
    rid = str(request_id)
    prefix_a = rid[:2] if len(rid) >= 2 else "__"
    prefix_b = rid[2:4] if len(rid) >= 4 else "__"
    return Path(_OBJECTS_DIRNAME) / prefix_a / prefix_b / f"{rid}.json"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload_bytes = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload_bytes)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
        _fsync_dir(path.parent)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _fsync_dir(path: Path) -> None:
    flags = getattr(os, "O_DIRECTORY", 0)
    fd = os.open(str(path), os.O_RDONLY | flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _iso8601(ts: float) -> str:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat().replace("+00:00", "Z")
