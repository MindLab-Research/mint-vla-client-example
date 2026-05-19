from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from .runtime_env import env_get

_CHECKPOINT_STAGING_TABLE = "checkpoint_staging"
_CHECKPOINT_CATALOG_TABLE = "checkpoint_catalog"
_SCHEMA_READY_DSN: str | None = None


class CheckpointIndexError(RuntimeError):
    pass


class CheckpointAlreadyUploadingError(CheckpointIndexError):
    pass


class CheckpointAlreadyExistsError(CheckpointIndexError):
    pass


class CheckpointAlreadyFailedError(CheckpointIndexError):
    pass


class CheckpointNotFoundError(CheckpointIndexError):
    pass


def checkpoint_index_enabled() -> bool:
    return bool(_checkpoint_index_pg_dsn())


def _runtime_config():
    from . import config as config_module

    return getattr(config_module, "config", None)


def _checkpoint_index_pg_dsn() -> str:
    cfg = _runtime_config()
    if cfg is not None:
        return str(getattr(cfg, "checkpoint_index_pg_dsn", "") or "").strip()
    return str(env_get(os.environ, "MINT_CHECKPOINT_INDEX_PG_DSN", "") or "").strip()


def _checkpoint_index_timeout_s() -> float:
    cfg = _runtime_config()
    raw = getattr(cfg, "checkpoint_index_write_timeout_ms", 2000) if cfg is not None else env_get(
        os.environ, "MINT_CHECKPOINT_INDEX_WRITE_TIMEOUT_MS", "2000"
    )
    try:
        return max(0.1, float(raw) / 1000.0)
    except Exception:
        return 2.0


def _import_asyncpg():
    import asyncpg

    return asyncpg


def _uploading_stale_timeout_s() -> float:
    raw = env_get(os.environ, "MINT_CHECKPOINT_INDEX_UPLOADING_STALE_S", "1800")
    try:
        return max(0.0, float(raw))
    except Exception:
        return 1800.0


def _checkpoint_created_at(value: str | None) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _lock_key(owner_id: str | None, model_id: str, raw_checkpoint_id: str, checkpoint_type: str) -> str:
    owner = str(owner_id or "anonymous").strip() or "anonymous"
    return f"{owner}\x1f{model_id}\x1f{raw_checkpoint_id}\x1f{checkpoint_type}"


async def _connect():
    dsn = _checkpoint_index_pg_dsn()
    if not dsn:
        raise RuntimeError("checkpoint index PG DSN is not configured")
    asyncpg = _import_asyncpg()
    return await asyncpg.connect(
        dsn=dsn,
        command_timeout=max(5.0, _checkpoint_index_timeout_s()),
        statement_cache_size=0,
        server_settings={"application_name": "mint_server_checkpoint_index"},
    )


async def _ensure_schema(conn) -> None:
    global _SCHEMA_READY_DSN
    dsn = _checkpoint_index_pg_dsn()
    if _SCHEMA_READY_DSN == dsn and dsn:
        return

    await conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_CHECKPOINT_STAGING_TABLE} (
          ckpt_id UUID PRIMARY KEY,
          owner_id TEXT NOT NULL,
          model_id TEXT NOT NULL,
          raw_checkpoint_id TEXT NOT NULL,
          checkpoint_type TEXT NOT NULL,
          storage_root TEXT NOT NULL,
          storage_layout_version INTEGER NOT NULL DEFAULT 2,
          model_name TEXT,
          checkpoint_created_at TIMESTAMPTZ,
          size_bytes BIGINT,
          status TEXT NOT NULL,
          fail_reason TEXT,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          CHECK (checkpoint_type IN ('training', 'sampler')),
          CHECK (status IN ('uploading', 'failed'))
        )
        """
    )
    await conn.execute(
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_checkpoint_staging_uploading_key
          ON {_CHECKPOINT_STAGING_TABLE}(owner_id, model_id, raw_checkpoint_id, checkpoint_type)
          WHERE status = 'uploading'
        """
    )
    await conn.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_checkpoint_staging_key_status_updated_at
          ON {_CHECKPOINT_STAGING_TABLE}(owner_id, model_id, raw_checkpoint_id, checkpoint_type, status, updated_at)
        """
    )
    await conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_CHECKPOINT_CATALOG_TABLE} (
          ckpt_id UUID PRIMARY KEY,
          owner_id TEXT NOT NULL,
          model_id TEXT NOT NULL,
          raw_checkpoint_id TEXT NOT NULL,
          checkpoint_type TEXT NOT NULL,
          storage_root TEXT NOT NULL,
          storage_layout_version INTEGER NOT NULL DEFAULT 2,
          model_name TEXT,
          checkpoint_created_at TIMESTAMPTZ,
          size_bytes BIGINT,
          published_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          deleted_at TIMESTAMPTZ,
          CHECK (checkpoint_type IN ('training', 'sampler'))
        )
        """
    )
    await conn.execute(
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_checkpoint_catalog_key
          ON {_CHECKPOINT_CATALOG_TABLE}(owner_id, model_id, raw_checkpoint_id, checkpoint_type)
        """
    )
    await conn.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_checkpoint_catalog_owner_created
          ON {_CHECKPOINT_CATALOG_TABLE}(owner_id, checkpoint_created_at DESC, published_at DESC)
        """
    )
    await conn.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_checkpoint_catalog_model_created
          ON {_CHECKPOINT_CATALOG_TABLE}(model_id, checkpoint_created_at DESC, published_at DESC)
        """
    )

    if dsn:
        _SCHEMA_READY_DSN = dsn


async def _quartet_lock(conn, *, owner_id: str | None, model_id: str, raw_checkpoint_id: str, checkpoint_type: str) -> None:
    await conn.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
        _lock_key(owner_id, model_id, raw_checkpoint_id, checkpoint_type),
    )


async def claim_checkpoint_publication(
    *,
    owner_id: str | None,
    model_id: str,
    raw_checkpoint_id: str,
    checkpoint_type: str,
    storage_root: str,
    model_name: str | None,
    checkpoint_created_at: str | None,
    retry: bool = False,
) -> str | None:
    if not checkpoint_index_enabled():
        return None

    conn = await _connect()
    try:
        await _ensure_schema(conn)
        created_at = _checkpoint_created_at(checkpoint_created_at)
        async with conn.transaction():
            await _quartet_lock(
                conn,
                owner_id=owner_id,
                model_id=model_id,
                raw_checkpoint_id=raw_checkpoint_id,
                checkpoint_type=checkpoint_type,
            )
            existing = await conn.fetchrow(
                f"""
                SELECT ckpt_id
                FROM {_CHECKPOINT_CATALOG_TABLE}
                WHERE owner_id = $1
                  AND model_id = $2
                  AND raw_checkpoint_id = $3
                  AND checkpoint_type = $4
                  AND deleted_at IS NULL
                """,
                str(owner_id or "anonymous"),
                model_id,
                raw_checkpoint_id,
                checkpoint_type,
            )
            if existing is not None:
                raise CheckpointAlreadyExistsError(
                    f"checkpoint already exists: {model_id}/{raw_checkpoint_id}/{checkpoint_type}"
                )

            rows = await conn.fetch(
                f"""
                SELECT ckpt_id, status, updated_at
                FROM {_CHECKPOINT_STAGING_TABLE}
                WHERE owner_id = $1
                  AND model_id = $2
                  AND raw_checkpoint_id = $3
                  AND checkpoint_type = $4
                ORDER BY created_at DESC
                """,
                str(owner_id or "anonymous"),
                model_id,
                raw_checkpoint_id,
                checkpoint_type,
            )

            stale_cutoff = datetime.now(timezone.utc) - timedelta(seconds=_uploading_stale_timeout_s())
            stale_uploading_ids: list[str] = []
            for row in rows:
                if str(row["status"]) != "uploading":
                    continue
                updated_at = row.get("updated_at")
                if isinstance(updated_at, datetime) and updated_at.astimezone(timezone.utc) < stale_cutoff:
                    stale_uploading_ids.append(str(row["ckpt_id"]))

            stale_uploading_id_set = set(stale_uploading_ids)
            if stale_uploading_ids:
                await conn.execute(
                    f"""
                    UPDATE {_CHECKPOINT_STAGING_TABLE}
                    SET status = 'failed',
                        fail_reason = COALESCE(fail_reason, 'stale_uploading_timeout'),
                        updated_at = now()
                    WHERE ckpt_id = ANY($1::uuid[])
                      AND status = 'uploading'
                    """,
                    stale_uploading_ids,
                )

            active_rows = [row for row in rows if str(row["ckpt_id"]) not in stale_uploading_id_set]
            if any(str(row["status"]) == "uploading" for row in active_rows):
                raise CheckpointAlreadyUploadingError(
                    f"checkpoint already uploading: {model_id}/{raw_checkpoint_id}/{checkpoint_type}"
                )
            if active_rows and not retry:
                raise CheckpointAlreadyFailedError(
                    f"checkpoint previously failed: {model_id}/{raw_checkpoint_id}/{checkpoint_type}"
                )

            ckpt_id = str(uuid.uuid4())
            await conn.execute(
                f"""
                INSERT INTO {_CHECKPOINT_STAGING_TABLE} (
                  ckpt_id,
                  owner_id,
                  model_id,
                  raw_checkpoint_id,
                  checkpoint_type,
                  storage_root,
                  storage_layout_version,
                  model_name,
                  checkpoint_created_at,
                  status,
                  created_at,
                  updated_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, 2, $7, $8, 'uploading', now(), now())
                """,
                ckpt_id,
                str(owner_id or "anonymous"),
                model_id,
                raw_checkpoint_id,
                checkpoint_type,
                storage_root,
                model_name,
                created_at,
            )
            return ckpt_id
    finally:
        await conn.close()


async def mark_checkpoint_failed(ckpt_id: str | None, *, fail_reason: str) -> None:
    if not ckpt_id or not checkpoint_index_enabled():
        return
    conn = await _connect()
    try:
        await _ensure_schema(conn)
        await conn.execute(
            f"""
            UPDATE {_CHECKPOINT_STAGING_TABLE}
            SET status = 'failed',
                fail_reason = $2,
                updated_at = now()
            WHERE ckpt_id = $1
              AND status = 'uploading'
            """,
            ckpt_id,
            fail_reason,
        )
    finally:
        await conn.close()


async def publish_checkpoint_catalog(
    ckpt_id: str | None,
    *,
    storage_root: str,
    size_bytes: int,
) -> dict[str, Any] | None:
    if not ckpt_id or not checkpoint_index_enabled():
        return None

    conn = await _connect()
    try:
        await _ensure_schema(conn)
        async with conn.transaction():
            row = await conn.fetchrow(
                f"""
                SELECT owner_id, model_id, raw_checkpoint_id, checkpoint_type
                FROM {_CHECKPOINT_STAGING_TABLE}
                WHERE ckpt_id = $1
                """,
                ckpt_id,
            )
            if row is None:
                existing = await conn.fetchrow(
                    f"""
                    SELECT ckpt_id, owner_id, model_id, raw_checkpoint_id, checkpoint_type,
                           storage_root, storage_layout_version, model_name,
                           checkpoint_created_at, size_bytes, published_at, updated_at
                    FROM {_CHECKPOINT_CATALOG_TABLE}
                    WHERE ckpt_id = $1
                      AND deleted_at IS NULL
                    """,
                    ckpt_id,
                )
                if existing is None:
                    raise CheckpointNotFoundError(f"checkpoint staging row missing: {ckpt_id}")
                return dict(existing)

            await _quartet_lock(
                conn,
                owner_id=row["owner_id"],
                model_id=row["model_id"],
                raw_checkpoint_id=row["raw_checkpoint_id"],
                checkpoint_type=row["checkpoint_type"],
            )
            moved = await conn.fetchrow(
                f"""
                DELETE FROM {_CHECKPOINT_STAGING_TABLE}
                WHERE ckpt_id = $1
                  AND status = 'uploading'
                RETURNING ckpt_id, owner_id, model_id, raw_checkpoint_id, checkpoint_type,
                          storage_layout_version, model_name, checkpoint_created_at
                """,
                ckpt_id,
            )
            if moved is None:
                existing = await conn.fetchrow(
                    f"""
                    SELECT ckpt_id, owner_id, model_id, raw_checkpoint_id, checkpoint_type,
                           storage_root, storage_layout_version, model_name,
                           checkpoint_created_at, size_bytes, published_at, updated_at
                    FROM {_CHECKPOINT_CATALOG_TABLE}
                    WHERE ckpt_id = $1
                      AND deleted_at IS NULL
                    """,
                    ckpt_id,
                )
                if existing is None:
                    raise CheckpointNotFoundError(f"checkpoint staging row not publishable: {ckpt_id}")
                return dict(existing)

            await conn.execute(
                f"""
                INSERT INTO {_CHECKPOINT_CATALOG_TABLE} (
                  ckpt_id,
                  owner_id,
                  model_id,
                  raw_checkpoint_id,
                  checkpoint_type,
                  storage_root,
                  storage_layout_version,
                  model_name,
                  checkpoint_created_at,
                  size_bytes,
                  published_at,
                  updated_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, now(), now())
                ON CONFLICT (ckpt_id) DO NOTHING
                """,
                str(moved["ckpt_id"]),
                str(moved["owner_id"]),
                str(moved["model_id"]),
                str(moved["raw_checkpoint_id"]),
                str(moved["checkpoint_type"]),
                storage_root,
                int(moved["storage_layout_version"]),
                moved["model_name"],
                moved["checkpoint_created_at"],
                int(size_bytes),
            )
            existing = await conn.fetchrow(
                f"""
                SELECT ckpt_id, owner_id, model_id, raw_checkpoint_id, checkpoint_type,
                       storage_root, storage_layout_version, model_name,
                       checkpoint_created_at, size_bytes, published_at, updated_at
                FROM {_CHECKPOINT_CATALOG_TABLE}
                WHERE ckpt_id = $1
                  AND deleted_at IS NULL
                """,
                ckpt_id,
            )
            if existing is None:
                raise CheckpointNotFoundError(f"checkpoint catalog row missing after publish: {ckpt_id}")
            return dict(existing)
    finally:
        await conn.close()


def _catalog_order_key(row: dict[str, Any]) -> tuple[str, str]:
    created = row.get("checkpoint_created_at")
    if isinstance(created, datetime):
        created_text = created.astimezone(timezone.utc).isoformat()
    else:
        created_text = str(created or "")
    published = row.get("published_at")
    if isinstance(published, datetime):
        published_text = published.astimezone(timezone.utc).isoformat()
    else:
        published_text = str(published or "")
    return created_text, published_text


def _catalog_row_to_dict(row: Any) -> dict[str, Any]:
    record = dict(row)
    for key in (
        "ckpt_id",
        "owner_id",
        "model_id",
        "raw_checkpoint_id",
        "checkpoint_type",
        "storage_root",
    ):
        value = record.get(key)
        if value is not None and not isinstance(value, str):
            record[key] = str(value)
    return record


async def list_catalog_checkpoints(*, owner_id: str | None, is_admin: bool) -> list[dict[str, Any]]:
    if not checkpoint_index_enabled():
        return []
    conn = await _connect()
    try:
        await _ensure_schema(conn)
        if is_admin:
            rows = await conn.fetch(
                f"""
                SELECT ckpt_id, owner_id, model_id, raw_checkpoint_id, checkpoint_type,
                       storage_root, storage_layout_version, model_name,
                       checkpoint_created_at, size_bytes, published_at, updated_at
                FROM {_CHECKPOINT_CATALOG_TABLE}
                WHERE deleted_at IS NULL
                ORDER BY COALESCE(checkpoint_created_at, published_at) DESC, published_at DESC
                """
            )
        else:
            rows = await conn.fetch(
                f"""
                SELECT ckpt_id, owner_id, model_id, raw_checkpoint_id, checkpoint_type,
                       storage_root, storage_layout_version, model_name,
                       checkpoint_created_at, size_bytes, published_at, updated_at
                FROM {_CHECKPOINT_CATALOG_TABLE}
                WHERE deleted_at IS NULL
                  AND owner_id = $1
                ORDER BY COALESCE(checkpoint_created_at, published_at) DESC, published_at DESC
                """,
                str(owner_id or "anonymous"),
            )
        return [_catalog_row_to_dict(row) for row in rows]
    finally:
        await conn.close()


async def list_catalog_checkpoints_for_model(
    model_id: str,
    *,
    owner_id: str | None,
    is_admin: bool,
) -> list[dict[str, Any]]:
    if not checkpoint_index_enabled():
        return []
    conn = await _connect()
    try:
        await _ensure_schema(conn)
        if is_admin:
            rows = await conn.fetch(
                f"""
                SELECT ckpt_id, owner_id, model_id, raw_checkpoint_id, checkpoint_type,
                       storage_root, storage_layout_version, model_name,
                       checkpoint_created_at, size_bytes, published_at, updated_at
                FROM {_CHECKPOINT_CATALOG_TABLE}
                WHERE deleted_at IS NULL
                  AND model_id = $1
                ORDER BY COALESCE(checkpoint_created_at, published_at) DESC, published_at DESC
                """,
                str(model_id),
            )
        else:
            rows = await conn.fetch(
                f"""
                SELECT ckpt_id, owner_id, model_id, raw_checkpoint_id, checkpoint_type,
                       storage_root, storage_layout_version, model_name,
                       checkpoint_created_at, size_bytes, published_at, updated_at
                FROM {_CHECKPOINT_CATALOG_TABLE}
                WHERE deleted_at IS NULL
                  AND owner_id = $1
                  AND model_id = $2
                ORDER BY COALESCE(checkpoint_created_at, published_at) DESC, published_at DESC
                """,
                str(owner_id or "anonymous"),
                str(model_id),
            )
        return [_catalog_row_to_dict(row) for row in rows]
    finally:
        await conn.close()


async def get_catalog_checkpoint(
    checkpoint_id: str,
    *,
    owner_id: str | None,
    is_admin: bool,
) -> dict[str, Any] | None:
    if not checkpoint_index_enabled():
        return None
    conn = await _connect()
    try:
        await _ensure_schema(conn)
        if is_admin:
            row = await conn.fetchrow(
                f"""
                SELECT ckpt_id, owner_id, model_id, raw_checkpoint_id, checkpoint_type,
                       storage_root, storage_layout_version, model_name,
                       checkpoint_created_at, size_bytes, published_at, updated_at
                FROM {_CHECKPOINT_CATALOG_TABLE}
                WHERE ckpt_id = $1
                  AND deleted_at IS NULL
                """,
                checkpoint_id,
            )
        else:
            row = await conn.fetchrow(
                f"""
                SELECT ckpt_id, owner_id, model_id, raw_checkpoint_id, checkpoint_type,
                       storage_root, storage_layout_version, model_name,
                       checkpoint_created_at, size_bytes, published_at, updated_at
                FROM {_CHECKPOINT_CATALOG_TABLE}
                WHERE ckpt_id = $1
                  AND deleted_at IS NULL
                  AND owner_id = $2
                """,
                checkpoint_id,
                str(owner_id or "anonymous"),
            )
        return _catalog_row_to_dict(row) if row is not None else None
    finally:
        await conn.close()


async def get_catalog_checkpoint_by_key(
    *,
    owner_id: str | None,
    model_id: str,
    raw_checkpoint_id: str,
    checkpoint_type: str,
) -> dict[str, Any] | None:
    if not checkpoint_index_enabled():
        return None
    conn = await _connect()
    try:
        await _ensure_schema(conn)
        row = await conn.fetchrow(
            f"""
            SELECT ckpt_id, owner_id, model_id, raw_checkpoint_id, checkpoint_type,
                   storage_root, storage_layout_version, model_name,
                   checkpoint_created_at, size_bytes, published_at, updated_at
            FROM {_CHECKPOINT_CATALOG_TABLE}
            WHERE deleted_at IS NULL
              AND owner_id = $1
              AND model_id = $2
              AND raw_checkpoint_id = $3
              AND checkpoint_type = $4
            """,
            str(owner_id or "anonymous"),
            str(model_id),
            str(raw_checkpoint_id),
            str(checkpoint_type),
        )
        return _catalog_row_to_dict(row) if row is not None else None
    finally:
        await conn.close()


async def mark_catalog_checkpoint_deleted(
    ckpt_id: str,
    *,
    owner_id: str | None,
    is_admin: bool,
) -> bool:
    if not checkpoint_index_enabled():
        return False
    conn = await _connect()
    try:
        await _ensure_schema(conn)
        if is_admin:
            status = await conn.execute(
                f"""
                UPDATE {_CHECKPOINT_CATALOG_TABLE}
                SET deleted_at = now(),
                    updated_at = now()
                WHERE ckpt_id = $1
                  AND deleted_at IS NULL
                """,
                str(ckpt_id),
            )
        else:
            status = await conn.execute(
                f"""
                UPDATE {_CHECKPOINT_CATALOG_TABLE}
                SET deleted_at = now(),
                    updated_at = now()
                WHERE ckpt_id = $1
                  AND owner_id = $2
                  AND deleted_at IS NULL
                """,
                str(ckpt_id),
                str(owner_id or "anonymous"),
            )
        return status.endswith(" 1")
    finally:
        await conn.close()
