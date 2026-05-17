"""Dense PEFT session-state storage helpers."""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path

from ..checkpoints import DEFAULT_RUNTIME_CHECKPOINTS_DIR
from ..config import config as server_config

logger = logging.getLogger(__name__)

DENSE_SESSION_STATE_DIRNAME = "dense_session_state"
_SESSION_SUFFIX = "_checkpoint"


def _dedupe_paths(paths: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in paths:
        path = os.path.abspath(str(raw).strip())
        if not path:
            continue
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def get_dense_session_state_root() -> str:
    configured = str(server_config.training_dense_session_state_root or "").strip()
    if configured:
        return os.path.abspath(configured)
    runtime_root = os.environ.get("TINKER_RUNTIME_CHECKPOINT_DIR", DEFAULT_RUNTIME_CHECKPOINTS_DIR)
    return os.path.abspath(os.path.join(runtime_root, DENSE_SESSION_STATE_DIRNAME))


def get_dense_session_path(session_id: str, *, root: str | None = None) -> str:
    base = root or get_dense_session_state_root()
    return os.path.join(base, f"{session_id}{_SESSION_SUFFIX}")


def _iter_dense_session_dirs(root: str) -> list[Path]:
    base = Path(root)
    if not base.exists() or not base.is_dir():
        return []
    out: list[Path] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        if not child.name.endswith(_SESSION_SUFFIX):
            continue
        out.append(child)
    return out


def _remove_tree(path: str) -> None:
    if not os.path.lexists(path):
        return
    if os.path.islink(path) or os.path.isfile(path):
        os.unlink(path)
        return
    shutil.rmtree(path)


def delete_dense_session_state(session_id: str, *, root: str | None = None) -> bool:
    deleted = False
    paths = [get_dense_session_path(session_id, root=root or get_dense_session_state_root())]
    for path in _dedupe_paths(paths):
        if not os.path.lexists(path):
            continue
        _remove_tree(path)
        deleted = True
    if deleted:
        logger.info("[dense_session_state] deleted session state session_id=%s", session_id)
    return deleted


def _tree_size_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for filename in files:
            full_path = os.path.join(root, filename)
            try:
                total += os.path.getsize(full_path)
            except OSError:
                continue
    return total


def collect_dense_session_state_stats(*, root: str | None = None, now: float | None = None) -> dict[str, object]:
    base = root or get_dense_session_state_root()
    current = time.time() if now is None else float(now)
    dir_count = 0
    total_bytes = 0
    oldest_age_s = 0.0

    for session_dir in _iter_dense_session_dirs(base):
        dir_count += 1
        total_bytes += _tree_size_bytes(session_dir)
        try:
            age_s = max(0.0, current - session_dir.stat().st_mtime)
        except OSError:
            age_s = 0.0
        oldest_age_s = max(oldest_age_s, age_s)

    return {
        "dense_session_state_root": os.path.abspath(base),
        "dense_session_state_bytes": int(total_bytes),
        "dense_session_state_dirs": int(dir_count),
        "dense_session_state_oldest_age_s": float(oldest_age_s),
    }
