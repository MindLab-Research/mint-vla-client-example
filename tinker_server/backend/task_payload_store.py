from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from ..config import config as server_config


class TaskPayloadStoreError(RuntimeError):
    pass


def _task_payload_root_dir() -> str:
    env_value = os.environ.get("MINT_TASK_PAYLOAD_ROOT_DIR")
    if env_value:
        return str(env_value)
    db_path = Path(
        str(
            getattr(
                server_config,
                "task_state_store_db_path",
                "/vePFS-Mindverse/share/mint-prod-dev/task-state/task_state.sqlite3",
            )
        )
    )
    return str(db_path.parent / "payloads")


def _safe_component(value: str) -> str:
    value = str(value)
    if not value:
        raise TaskPayloadStoreError("empty path component")
    if "/" in value or "\\" in value or value in {".", ".."}:
        raise TaskPayloadStoreError(f"unsafe path component: {value!r}")
    return value


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class TaskPayloadStore:
    """Filesystem payload store with temp-file + atomic rename publish."""

    def __init__(self, root_dir: str | os.PathLike[str] | None = None) -> None:
        self._root_dir = Path(root_dir or _task_payload_root_dir())

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    def _final_path(self, *, request_id: str, attempt_id: str) -> Path:
        request_id = _safe_component(request_id)
        attempt_id = _safe_component(attempt_id)
        shard = request_id[:2] if len(request_id) >= 2 else "__"
        return self._root_dir / shard / request_id / f"{attempt_id}.json"

    def write_json_payload(
        self,
        *,
        request_id: str,
        attempt_id: str,
        payload: Any,
    ) -> dict[str, Any]:
        final_path = self._final_path(request_id=request_id, attempt_id=attempt_id)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = final_path.with_name(f".{final_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        encoded = json.dumps(
            _jsonable(payload),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        checksum = "sha256:" + hashlib.sha256(encoded).hexdigest()
        try:
            with open(tmp_path, "wb") as fh:
                fh.write(encoded)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, final_path)
            _fsync_directory(final_path.parent)
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
        return {
            "path": str(final_path),
            "checksum": checksum,
            "size_bytes": len(encoded),
        }

    def read_json_payload(
        self,
        *,
        path: str | os.PathLike[str],
        expected_checksum: str | None = None,
    ) -> Any:
        p = Path(path)
        try:
            resolved = p.resolve()
            root = self._root_dir.resolve()
        except OSError as e:
            raise TaskPayloadStoreError(f"failed to resolve payload path: {p}") from e
        if root not in resolved.parents and resolved != root:
            raise TaskPayloadStoreError(f"payload path outside root: {p}")
        data = p.read_bytes()
        checksum = "sha256:" + hashlib.sha256(data).hexdigest()
        if expected_checksum is not None and checksum != str(expected_checksum):
            raise TaskPayloadStoreError(
                f"payload checksum mismatch: expected={expected_checksum!r} actual={checksum!r}"
            )
        return json.loads(data.decode("utf-8"))
