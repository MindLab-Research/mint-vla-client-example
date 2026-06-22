"""Checkpoint path, archive, and runtime-cache helpers shared across routes."""

from __future__ import annotations

import asyncio
import fcntl
import glob
import json
import os
import shutil
import tarfile
import threading
import time
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Literal

import structlog

from mint_server.observability.logging_context import start_as_current_span, record_span_event_otel
from mint_server.ray.runtime_env import env_get
from .checkpoint_index import (
    CheckpointNotFoundError,
    checkpoint_index_enabled,
    mark_checkpoint_failed,
    publish_checkpoint_catalog,
)

logger = structlog.get_logger(__name__)

DEFAULT_PERSISTENT_CHECKPOINTS_DIR = "/tos-mindverse/mint_checkpoints"
DEFAULT_RUNTIME_CHECKPOINTS_DIR = "/vePFS-Mindverse/share/mint/prod/data/runtime-checkpoints"
DEFAULT_EPHEMERAL_TTL_S = 24 * 60 * 60
DEFAULT_PERSISTENT_CACHE_TTL_S = 24 * 60 * 60
DEFAULT_REAP_INTERVAL_S = 10 * 60
DEFAULT_MIRROR_POLL_S = 5
DEFAULT_PUBLISH_RETRY_BACKOFF_S = 60

# Backward-compatible module globals. Existing tests patch CHECKPOINTS_DIR directly.
CHECKPOINTS_DIR = env_get(os.environ, "MINT_CHECKPOINT_DIR", DEFAULT_PERSISTENT_CHECKPOINTS_DIR)
PERSISTENT_CHECKPOINTS_DIR = env_get(os.environ, "MINT_PERSISTENT_CHECKPOINT_DIR", CHECKPOINTS_DIR)
RUNTIME_CHECKPOINTS_DIR = env_get(os.environ, "MINT_RUNTIME_CHECKPOINT_DIR", DEFAULT_RUNTIME_CHECKPOINTS_DIR)

CheckpointType = Literal["training", "sampler"]
_CHECKPOINT_TYPES: tuple[CheckpointType, ...] = ("training", "sampler")
MIRROR_STATUS_PENDING = "pending"
MIRROR_STATUS_IN_PROGRESS = "in_progress"
MIRROR_STATUS_COMPLETE = "complete"
MIRROR_STATUS_FAILED = "failed"

_mirror_process_lock = threading.Lock()
_mirror_thread_lock = threading.Lock()
_mirror_thread: threading.Thread | None = None


def _dedupe_paths(paths: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for path in paths:
        real = os.path.realpath(path)
        if real in seen:
            continue
        seen.add(real)
        out.append(path)
    return out


def checkpoint_owner_dir(user_id: str | None) -> str:
    return user_id or "anonymous"


def _is_valid_checkpoint_segment(value: str) -> bool:
    if not value or value in (".", ".."):
        return False
    if "/" in value or "\\" in value:
        return False
    return True


def checkpoint_logical_name(path: str) -> str:
    base = os.path.basename(os.path.normpath(path))
    if base in _CHECKPOINT_TYPES:
        return os.path.basename(os.path.dirname(os.path.normpath(path)))
    return base


def is_ephemeral_checkpoint_name(name: str) -> bool:
    return checkpoint_logical_name(name).startswith("_ephemeral_")


def get_checkpoints_dir() -> str:
    return CHECKPOINTS_DIR


def get_persistent_checkpoints_dir() -> str:
    return PERSISTENT_CHECKPOINTS_DIR


def get_runtime_checkpoints_dir() -> str:
    return RUNTIME_CHECKPOINTS_DIR


def get_ephemeral_checkpoints_dir() -> str:
    return os.path.join(get_runtime_checkpoints_dir(), "ephemeral")


def get_persistent_cache_dir() -> str:
    return os.path.join(get_runtime_checkpoints_dir(), "persistent_cache")


def get_persistent_cache_ttl_s() -> int:
    return int(os.environ.get("MINT_PERSISTENT_CHECKPOINT_CACHE_TTL_S", str(DEFAULT_PERSISTENT_CACHE_TTL_S)))


def get_ephemeral_checkpoint_ttl_s() -> int:
    return int(os.environ.get("MINT_EPHEMERAL_CHECKPOINT_TTL_S", str(DEFAULT_EPHEMERAL_TTL_S)))


def get_checkpoint_reap_interval_s() -> int:
    return int(os.environ.get("MINT_CHECKPOINT_REAP_INTERVAL_S", str(DEFAULT_REAP_INTERVAL_S)))


def get_checkpoint_mirror_poll_s() -> float:
    return float(os.environ.get("MINT_CHECKPOINT_MIRROR_POLL_S", str(DEFAULT_MIRROR_POLL_S)))


def get_checkpoint_publish_retry_backoff_s() -> float:
    raw = os.environ.get("MINT_CHECKPOINT_INDEX_PUBLISH_RETRY_S", str(DEFAULT_PUBLISH_RETRY_BACKOFF_S))
    try:
        return max(0.0, float(raw))
    except Exception:
        return float(DEFAULT_PUBLISH_RETRY_BACKOFF_S)


def get_persistent_search_roots(primary_root: str | None = None) -> list[str]:
    paths = [primary_root] if primary_root else []
    paths.append(get_persistent_checkpoints_dir())
    return _dedupe_paths([p for p in paths if p])


def get_resolution_roots(*, primary_root: str | None = None, include_ephemeral: bool = True) -> list[str]:
    paths = get_persistent_search_roots(primary_root)
    if include_ephemeral:
        paths.append(get_ephemeral_checkpoints_dir())
        paths.append(get_persistent_cache_dir())
    return _dedupe_paths(paths)


def checkpoint_namespace_dir(checkpoint_type: CheckpointType | None) -> str | None:
    if checkpoint_type is None:
        return None
    if checkpoint_type not in _CHECKPOINT_TYPES:
        raise ValueError(f"Unsupported checkpoint_type: {checkpoint_type!r}")
    return checkpoint_type


def build_checkpoint_dir(
    root: str,
    *,
    user_id: str | None,
    model_id: str,
    checkpoint_name: str,
    checkpoint_type: CheckpointType | None = None,
) -> str:
    owner_dir = checkpoint_owner_dir(user_id)
    path = os.path.join(root, owner_dir, model_id, checkpoint_name)
    type_dir = checkpoint_namespace_dir(checkpoint_type)
    return os.path.join(path, type_dir) if type_dir else path


def build_ephemeral_checkpoint_dir(
    *,
    user_id: str | None,
    model_id: str,
    checkpoint_name: str,
    checkpoint_type: CheckpointType | None = None,
) -> str:
    return build_checkpoint_dir(
        get_ephemeral_checkpoints_dir(),
        user_id=user_id,
        model_id=model_id,
        checkpoint_name=checkpoint_name,
        checkpoint_type=checkpoint_type,
    )


def build_persistent_checkpoint_dir(
    *,
    user_id: str | None,
    model_id: str,
    checkpoint_name: str,
    checkpoint_type: CheckpointType | None = None,
) -> str:
    return build_checkpoint_dir(
        get_persistent_checkpoints_dir(),
        user_id=user_id,
        model_id=model_id,
        checkpoint_name=checkpoint_name,
        checkpoint_type=checkpoint_type,
    )


def build_persistent_cache_dir(
    *,
    user_id: str | None,
    model_id: str,
    checkpoint_name: str,
    checkpoint_type: CheckpointType | None = None,
) -> str:
    return build_checkpoint_dir(
        get_persistent_cache_dir(),
        user_id=user_id,
        model_id=model_id,
        checkpoint_name=checkpoint_name,
        checkpoint_type=checkpoint_type,
    )


def checkpoint_has_lora_weights(path: str) -> bool:
    return os.path.exists(os.path.join(path, "adapter_model.safetensors")) or bool(
        glob.glob(os.path.join(path, "mp_rank_*_adapter.pt"))
    ) or os.path.exists(
        os.path.join(path, "bumblebee_rank_sharded_adapter.json")
    )


def checkpoint_has_sampling_adapter_weights(path: str) -> bool:
    return os.path.exists(os.path.join(path, "adapter_model.safetensors")) or os.path.exists(
        os.path.join(path, "bumblebee_rank_sharded_adapter.json")
    )


def _checkpoint_has_openpi_norm_stats(root: Path) -> bool:
    assets_root = root / "assets"
    if not assets_root.is_dir():
        return False
    return any(path.is_file() for path in assets_root.glob("**/norm_stats.json"))


def checkpoint_has_openpi_policy_weights(path: str) -> bool:
    root = Path(path)
    return (root / "params" / "_METADATA").exists() and _checkpoint_has_openpi_norm_stats(root)


def _latest_openpi_step_dir(path: str) -> Path | None:
    root = Path(path)
    candidates = [child for child in root.iterdir() if child.is_dir() and child.name.isdigit()]
    if not candidates:
        return None
    return max(candidates, key=lambda child: int(child.name))


def checkpoint_has_openpi_training_state(path: str) -> bool:
    latest = _latest_openpi_step_dir(path)
    if latest is None:
        return False
    return (latest / "params" / "_METADATA").exists() and (latest / "train_state" / "_METADATA").exists()


def checkpoint_has_optimizer_state(path: str) -> bool:
    bumblebee_meta_path = os.path.join(path, "training_meta.json")
    bumblebee_rank_states = glob.glob(os.path.join(path, "rank_*", "adapter_train_state.pt"))
    bumblebee_has_optimizer = False
    if bumblebee_rank_states and os.path.exists(bumblebee_meta_path):
        try:
            with open(bumblebee_meta_path, encoding="utf-8") as f:
                bumblebee_has_optimizer = bool(json.load(f).get("has_optimizer"))
        except Exception:
            bumblebee_has_optimizer = False
    return (
        os.path.exists(os.path.join(path, "optimizer.pt"))
        or bool(glob.glob(os.path.join(path, "*_optimizer.pt")))
        or bool(glob.glob(os.path.join(path, "*", "train_state", "_METADATA")))
        or bumblebee_has_optimizer
    )


def read_checkpoint_metadata(path: str) -> dict:
    meta_path = os.path.join(path, "metadata.json")
    with open(meta_path) as f:
        return json.load(f)


def _checkpoint_metadata_lock_path(path: str) -> str:
    return os.path.join(path, ".metadata.lock")


def _locked_checkpoint_metadata(path: str):
    os.makedirs(path, exist_ok=True)
    lock_path = _checkpoint_metadata_lock_path(path)
    lock_file = open(lock_path, "a+", encoding="utf-8")
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    return lock_file


def _load_checkpoint_metadata_locked(path: str) -> dict:
    meta_path = os.path.join(path, "metadata.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path) as f:
        return json.load(f)


def _isoformat_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def metadata_expires_at(metadata: dict) -> datetime | None:
    raw = metadata.get("expires_at")
    if isinstance(raw, str) and raw:
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    ttl = metadata.get("ttl_seconds")
    if ttl is None:
        return None
    try:
        ttl_int = int(ttl)
    except (TypeError, ValueError):
        return None
    created = metadata.get("created_at")
    if not isinstance(created, str):
        return None
    try:
        created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
    except ValueError:
        return None
    return datetime.fromtimestamp(created_dt.timestamp() + ttl_int, tz=timezone.utc)


def write_checkpoint_metadata(path: str, metadata: dict) -> None:
    data = dict(metadata)
    created_at = data.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        created_at = _isoformat_utc(time.time())
        data["created_at"] = created_at

    ttl_raw = data.get("ttl_seconds")
    if ttl_raw is not None:
        try:
            ttl_int = int(ttl_raw)
        except (TypeError, ValueError):
            data.pop("ttl_seconds", None)
        else:
            data["ttl_seconds"] = ttl_int
            if ttl_int > 0 and not data.get("expires_at"):
                created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                data["expires_at"] = _isoformat_utc(created_dt.timestamp() + ttl_int)

    os.makedirs(path, exist_ok=True)
    meta_path = os.path.join(path, "metadata.json")
    temp_path = f"{meta_path}.tmp-{uuid.uuid4().hex[:8]}"
    lock_file = _locked_checkpoint_metadata(path)
    try:
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, meta_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


def update_checkpoint_metadata(path: str, updates: dict) -> dict:
    os.makedirs(path, exist_ok=True)
    meta_path = os.path.join(path, "metadata.json")
    temp_path = f"{meta_path}.tmp-{uuid.uuid4().hex[:8]}"
    lock_file = _locked_checkpoint_metadata(path)
    try:
        try:
            current = _load_checkpoint_metadata_locked(path)
        except (json.JSONDecodeError, OSError) as e:
            if os.path.exists(meta_path):
                raise RuntimeError(
                    f"Refusing to overwrite invalid checkpoint metadata at {meta_path}: {type(e).__name__}: {e}"
                ) from e
            current = {}
        current.update(dict(updates))

        created_at = current.get("created_at")
        if not isinstance(created_at, str) or not created_at:
            created_at = _isoformat_utc(time.time())
            current["created_at"] = created_at

        ttl_raw = current.get("ttl_seconds")
        if ttl_raw is not None:
            try:
                ttl_int = int(ttl_raw)
            except (TypeError, ValueError):
                current.pop("ttl_seconds", None)
            else:
                current["ttl_seconds"] = ttl_int
                if ttl_int > 0 and not current.get("expires_at"):
                    created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    current["expires_at"] = _isoformat_utc(created_dt.timestamp() + ttl_int)

        with open(temp_path, "w") as f:
            json.dump(current, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, meta_path)
        return current
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


def validate_checkpoint_load_contract(
    path: str, *, load_optimizer: bool
) -> tuple[CheckpointType, bool]:
    metadata = read_checkpoint_metadata(path)

    checkpoint_type = metadata.get("checkpoint_type") or metadata.get("type")
    if checkpoint_type not in ("training", "sampler"):
        raise ValueError(f"Invalid checkpoint_type in metadata.json: {checkpoint_type!r}")

    optimizer_present = metadata.get("optimizer_present")
    if not isinstance(optimizer_present, bool):
        optimizer_present = bool(checkpoint_has_optimizer_state(path))

    if load_optimizer:
        if checkpoint_type != "training":
            raise ValueError(
                "Optimizer restore requested, but checkpoint_type is not 'training'"
            )
        if not optimizer_present:
            raise ValueError(
                "Optimizer restore requested, but optimizer artifacts are missing"
            )

    return checkpoint_type, optimizer_present


def safe_extract_checkpoint_archive(archive_path: str, dest_dir: str) -> None:
    os.makedirs(dest_dir, exist_ok=True)

    try:
        tf = tarfile.open(archive_path, "r:gz")
    except tarfile.TarError as e:
        raise ValueError("Invalid tar.gz archive") from e

    with tf:
        members = [m for m in tf.getmembers() if m.name not in ("", ".")]
        if not members:
            raise ValueError("Empty archive")

        def _norm(name: str) -> str:
            return name[2:] if name.startswith("./") else name

        roots = set()
        for m in members:
            name = _norm(m.name)
            p = Path(name)
            if p.is_absolute() or ".." in p.parts:
                raise ValueError(f"Unsafe path in archive: {m.name}")
            if m.islnk() or m.issym():
                raise ValueError(f"Links are not allowed in archive: {m.name}")
            if p.parts:
                roots.add(p.parts[0])

        if len(roots) != 1:
            raise ValueError(f"Expected single top-level directory, found: {sorted(roots)}")

        root = next(iter(roots))
        for m in members:
            name = _norm(m.name)
            p = Path(name)
            if not p.parts or p.parts[0] != root:
                raise ValueError(f"Unexpected archive member: {m.name}")

            rel_parts = p.parts[1:]
            if not rel_parts:
                continue

            rel = Path(*rel_parts)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"Unsafe path in archive: {m.name}")

            out_path = Path(dest_dir) / rel
            if m.isdir():
                out_path.mkdir(parents=True, exist_ok=True)
                continue

            out_path.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(m)
            if src is None:
                continue
            with src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)


def validate_checkpoint_dir(path: str, *, checkpoint_type: CheckpointType | None = None) -> None:
    has_lora = checkpoint_has_lora_weights(path)
    has_optimizer = checkpoint_has_optimizer_state(path)
    has_openpi_sampler = checkpoint_has_openpi_policy_weights(path)
    has_openpi_training = checkpoint_has_openpi_training_state(path)

    if checkpoint_type == "sampler":
        if not has_lora and not has_openpi_sampler:
            raise ValueError("Missing sampler weights in extracted checkpoint")
        if has_optimizer:
            raise ValueError("Sampler checkpoint must not include optimizer state")
        return

    if checkpoint_type == "training":
        if not has_lora and not has_openpi_training:
            raise ValueError("Missing LoRA weights in extracted checkpoint")
        if not has_optimizer:
            raise ValueError("Missing optimizer state in extracted checkpoint")
        return

    if not has_lora and not has_openpi_training:
        raise ValueError("Missing LoRA weights in extracted checkpoint")
    if not has_optimizer:
        raise ValueError("Missing optimizer state in extracted checkpoint")


def validate_sampler_checkpoint_for_sampling(path: str) -> None:
    if not checkpoint_has_sampling_adapter_weights(path) and not checkpoint_has_openpi_policy_weights(path):
        raise ValueError(
            "Missing sampling weights: expected adapter_model.safetensors, "
            "bumblebee_rank_sharded_adapter.json, or OpenPI params/_METADATA with assets/**/norm_stats.json"
        )
    if checkpoint_has_optimizer_state(path):
        raise ValueError("Sampler checkpoint must not include optimizer state")
    if checkpoint_has_openpi_policy_weights(path):
        return
    if os.path.exists(os.path.join(path, "bumblebee_rank_sharded_adapter.json")):
        try:
            from mint_server.backend.training.bumblebee.bumblebee_lora import prepare_lora_adapter_for_vllm

            prepare_lora_adapter_for_vllm(path)
            return
        except Exception as e:
            raise ValueError(f"Unreadable Bumblebee rank-sharded adapter for sampling: {e}") from e
    try:
        from safetensors import safe_open

        with safe_open(
            os.path.join(path, "adapter_model.safetensors"),
            framework="np",
            device="cpu",
        ) as f:
            list(f.keys())
    except Exception as e:
        raise ValueError(f"Unreadable adapter_model.safetensors for sampling: {e}") from e


def _validate_checkpoint_for_publication(path: str, checkpoint_type: CheckpointType) -> None:
    if checkpoint_type == "sampler":
        validate_sampler_checkpoint_for_sampling(path)
    else:
        validate_checkpoint_dir(path, checkpoint_type=checkpoint_type)


def create_checkpoint_archive(checkpoint_dir: str, archive_path: str) -> None:
    if not os.path.isdir(checkpoint_dir):
        raise ValueError(f"Checkpoint dir is not a directory: {checkpoint_dir}")

    root = os.path.basename(os.path.normpath(checkpoint_dir))
    if not root:
        raise ValueError(f"Invalid checkpoint dir: {checkpoint_dir}")

    with tarfile.open(archive_path, "w:gz") as tf:
        tf.add(checkpoint_dir, arcname=root, recursive=True)


def build_gateway_proxy_archive_path() -> str:
    archive_dir = os.path.join(get_runtime_checkpoints_dir(), "gateway_proxy_archives")
    os.makedirs(archive_dir, exist_ok=True)
    return os.path.join(archive_dir, f"gateway_ckpt_proxy_{uuid.uuid4().hex}.tar.gz")


@lru_cache(maxsize=1)
def _create_checkpoint_archive_remote():
    import ray

    @ray.remote(num_cpus=1)
    def _task(checkpoint_dir: str, archive_path: str) -> str:
        create_checkpoint_archive(checkpoint_dir, archive_path)
        return archive_path

    return _task


async def async_create_checkpoint_archive(
    checkpoint_dir: str,
    archive_path: str,
    *,
    timeout_s: float = 600.0,
) -> None:
    from mint_server.backend.ray_cluster.async_ray_control import _await_ray_ref, _ensure_ray_initialized, control_plane_task_runtime_env

    import ray

    if not os.path.isdir(checkpoint_dir):
        raise ValueError(f"Checkpoint dir is not a directory: {checkpoint_dir}")

    archive_dir = os.path.dirname(os.path.abspath(archive_path))
    if archive_dir:
        os.makedirs(archive_dir, exist_ok=True)

    _ensure_ray_initialized()
    ref = _create_checkpoint_archive_remote().options(runtime_env=control_plane_task_runtime_env()).remote(
        str(checkpoint_dir),
        str(archive_path),
    )
    try:
        await asyncio.wait_for(_await_ray_ref(ref), timeout=float(timeout_s))
    except asyncio.TimeoutError:
        ray.cancel(ref, force=True)
        try:
            if os.path.exists(archive_path):
                os.remove(archive_path)
        except Exception:
            pass
        raise


def _scoped_checkpoint_owner_dir(user_id: str | None, *, is_admin: bool) -> str:
    raw = str(user_id or "").strip()
    if is_admin and not raw:
        raise ValueError("owner_id is required for admin checkpoint references")
    owner_dir = checkpoint_owner_dir(raw or None)
    if not _is_valid_checkpoint_segment(owner_dir):
        raise ValueError("Invalid owner_id")
    return owner_dir


def _deterministic_checkpoint_candidates(
    roots: list[str],
    *,
    owner_dir: str,
    path_part: str,
) -> list[str]:
    candidates: list[str] = []
    # Always include "anonymous" as a fallback owner_dir, because dense
    # trainer actors save checkpoints without a user_id (owner_dir="anonymous"),
    # while API requests resolve with the HTTP user_id. This ensures the
    # resolver can find checkpoints saved by actors regardless of the
    # request's user context.
    owner_dirs = [owner_dir, "anonymous"] if owner_dir != "anonymous" else ["anonymous"]
    for root in roots:
        for od in owner_dirs:
            candidates.append(os.path.join(root, od, path_part))
        candidates.append(os.path.join(root, path_part))
    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        real = os.path.realpath(candidate)
        if real in seen:
            continue
        seen.add(real)
        out.append(candidate)
    return out


def resolve_checkpoint_id(
    checkpoint_id: str,
    checkpoints_dir: str,
    *,
    user_id: str | None = None,
    is_admin: bool = False,
) -> str | None:
    roots = get_resolution_roots(primary_root=checkpoints_dir)
    owner_dir = _scoped_checkpoint_owner_dir(user_id, is_admin=is_admin)
    for candidate in _deterministic_checkpoint_candidates(roots, owner_dir=owner_dir, path_part=checkpoint_id):
        resolved = _existing_checkpoint_view(candidate, checkpoint_type=None)
        if resolved is None:
            continue
        if is_admin and not _checkpoint_view_matches_owner(resolved, owner_dir=owner_dir):
            continue
        return resolved
    return None


def _checkpoint_type_from_uri_path(path_part: str) -> CheckpointType | None:
    parts = path_part.split("/")
    for i, part in enumerate(parts):
        if part == "weights" and 0 < i < len(parts) - 1:
            return "training"
        if part == "sampler_weights" and 0 < i < len(parts) - 1:
            return "sampler"
    return None


def _strip_checkpoint_kind(path_part: str) -> str:
    parts = path_part.split("/")
    for i, part in enumerate(parts):
        if part in ("weights", "sampler_weights") and 0 < i < len(parts) - 1:
            return "/".join(parts[:i] + parts[i + 1 :])
    return path_part


def _existing_checkpoint_view(
    path: str,
    *,
    checkpoint_type: CheckpointType | None,
) -> str | None:
    candidates: list[str] = []
    if checkpoint_type is not None:
        candidates.append(os.path.join(path, checkpoint_type))
    candidates.append(path)
    for candidate in candidates:
        if not os.path.isdir(candidate):
            continue
        if os.path.exists(os.path.join(candidate, "metadata.json")):
            return candidate
        if checkpoint_has_lora_weights(candidate) or checkpoint_has_optimizer_state(candidate):
            return candidate
    return None


def _checkpoint_view_matches_owner(path: str, *, owner_dir: str) -> bool:
    metadata_path = os.path.join(path, "metadata.json")
    if not os.path.exists(metadata_path):
        return False
    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    actual = checkpoint_owner_dir(metadata.get("owner_id"))
    return actual == owner_dir


def _prefer_cached_checkpoint_view(
    path_part: str,
    *,
    checkpoint_type: CheckpointType,
    owner_dir: str,
    is_admin: bool,
) -> str | None:
    """Return persistent-cache checkpoint view when available.

    Newly saved checkpoints are immediately present under persistent_cache and
    may take time to mirror into persistent storage. Prefer cache first so
    save_state -> load_state can run deterministically in one session.
    """
    cache_root = get_persistent_cache_dir()
    anonymous_dir = checkpoint_owner_dir(None)
    if is_admin:
        candidates = [
            os.path.join(cache_root, path_part),
            os.path.join(cache_root, owner_dir, path_part),
            os.path.join(cache_root, anonymous_dir, path_part),
        ]
    else:
        candidates = [os.path.join(cache_root, owner_dir, path_part)]
        if owner_dir != anonymous_dir:
            candidates.append(os.path.join(cache_root, anonymous_dir, path_part))

    for candidate in _dedupe_paths(candidates):
        resolved = _existing_checkpoint_view(candidate, checkpoint_type=checkpoint_type)
        if resolved is None:
            continue
        if is_admin and not (
            _checkpoint_view_matches_owner(resolved, owner_dir=owner_dir)
            or _checkpoint_view_matches_owner(resolved, owner_dir=anonymous_dir)
        ):
            continue
        return resolved
    return None


def resolve_checkpoint_uri(
    uri: str,
    checkpoints_dir: str,
    *,
    user_id: str | None = None,
    is_admin: bool = False,
) -> str:
    if uri.startswith("file://"):
        return uri[7:]

    roots = get_resolution_roots(primary_root=checkpoints_dir)

    if uri.startswith("ckpt_"):
        resolved = resolve_checkpoint_id(uri, checkpoints_dir, user_id=user_id, is_admin=is_admin)
        return resolved or uri

    if uri.startswith("mint://"):
        raw_path_part = uri[len("mint://") :]
    else:
        return uri

    checkpoint_type = _checkpoint_type_from_uri_path(raw_path_part)
    if checkpoint_type is None:
        raise ValueError(
            "Checkpoint URIs must include an explicit checkpoint type "
            "('/weights/' or '/sampler_weights/')."
        )
    owner_dir = _scoped_checkpoint_owner_dir(user_id, is_admin=is_admin)
    path_part = _strip_checkpoint_kind(raw_path_part)

    cached = _prefer_cached_checkpoint_view(
        path_part,
        checkpoint_type=checkpoint_type,
        owner_dir=owner_dir,
        is_admin=is_admin,
    )
    if cached is not None:
        return cached

    for candidate in _deterministic_checkpoint_candidates(roots, owner_dir=owner_dir, path_part=path_part):
        resolved = _existing_checkpoint_view(candidate, checkpoint_type=checkpoint_type)
        if resolved is None:
            continue
        if is_admin and not _checkpoint_view_matches_owner(resolved, owner_dir=owner_dir):
            continue
        return resolved

    base = os.path.join(get_persistent_checkpoints_dir(), owner_dir, path_part)
    fallback_type = checkpoint_namespace_dir(checkpoint_type)
    return os.path.join(base, fallback_type) if fallback_type else base


def resolve_checkpoint_path(state_uri: str, *, user_id: str | None = None, is_admin: bool = False) -> str:
    return resolve_checkpoint_uri(state_uri, CHECKPOINTS_DIR, user_id=user_id, is_admin=is_admin)


def checkpoint_access_roots(
    *,
    user_id: str | None,
    is_admin: bool = False,
    include_ephemeral: bool = True,
    include_cache: bool = True,
) -> list[str]:
    roots = get_persistent_search_roots()
    if include_ephemeral:
        roots.append(get_ephemeral_checkpoints_dir())
    if include_cache:
        roots.append(get_persistent_cache_dir())

    if is_admin:
        return _dedupe_paths(roots)

    owner_dir = checkpoint_owner_dir(user_id)
    return _dedupe_paths([os.path.join(root, owner_dir) for root in roots])


def checkpoint_path_is_allowed(
    path: str,
    *,
    user_id: str | None,
    is_admin: bool = False,
    include_ephemeral: bool = True,
    include_cache: bool = True,
) -> bool:
    real = os.path.realpath(path)
    for allowed_root in checkpoint_access_roots(
        user_id=user_id,
        is_admin=is_admin,
        include_ephemeral=include_ephemeral,
        include_cache=include_cache,
    ):
        root_real = os.path.realpath(allowed_root)
        if real == root_real or real.startswith(root_real + os.sep):
            return True
    return False


def ensure_checkpoint_path_allowed(
    path: str,
    *,
    user_id: str | None,
    is_admin: bool = False,
    include_ephemeral: bool = True,
    include_cache: bool = True,
) -> None:
    if is_admin:
        return
    if not checkpoint_path_is_allowed(
        path,
        user_id=user_id,
        is_admin=is_admin,
        include_ephemeral=include_ephemeral,
        include_cache=include_cache,
    ):
        raise PermissionError("Access denied")


def _dir_size_bytes(path: str) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, filename))
            except OSError:
                pass
    return total


def sync_checkpoint_tree(src_dir: str, dst_dir: str) -> str:
    if os.path.realpath(src_dir) == os.path.realpath(dst_dir):
        return dst_dir
    if not os.path.isdir(src_dir):
        raise ValueError(f"Checkpoint dir is not a directory: {src_dir}")
    parent_dir = os.path.dirname(dst_dir)
    os.makedirs(parent_dir, exist_ok=True)
    lock_path = os.path.join(parent_dir, f".{os.path.basename(dst_dir)}.lock")
    temp_dir = f"{dst_dir}.tmp-{uuid.uuid4().hex[:8]}"
    old_dir = f"{dst_dir}.old-{uuid.uuid4().hex[:8]}"

    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if os.path.isdir(dst_dir):
                src_meta = os.path.join(src_dir, "metadata.json")
                dst_meta = os.path.join(dst_dir, "metadata.json")
                if os.path.exists(src_meta) and os.path.exists(dst_meta):
                    try:
                        if read_checkpoint_metadata(src_dir) == read_checkpoint_metadata(dst_dir):
                            os.utime(dst_dir, None)
                            return dst_dir
                    except Exception:
                        pass

            shutil.copytree(src_dir, temp_dir)
            if os.path.isdir(dst_dir):
                os.rename(dst_dir, old_dir)
            os.rename(temp_dir, dst_dir)
            os.utime(dst_dir, None)
            if os.path.isdir(old_dir):
                shutil.rmtree(old_dir, ignore_errors=True)
            return dst_dir
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            if os.path.exists(old_dir):
                shutil.rmtree(old_dir, ignore_errors=True)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def mirror_checkpoint_to_persistent_store(
    src_dir: str,
    *,
    user_id: str | None,
    model_id: str,
    checkpoint_name: str,
    checkpoint_type: CheckpointType | None = None,
) -> str:
    _t0 = time.perf_counter()
    with start_as_current_span(
        "checkpoint.mirror",
        component="checkpoints",
        op="mirror",
        attributes={"model_id": str(model_id), "checkpoint_name": str(checkpoint_name)},
    ):
        dst_dir = build_persistent_checkpoint_dir(
            user_id=user_id,
            model_id=model_id,
            checkpoint_name=checkpoint_name,
            checkpoint_type=checkpoint_type,
        )
        result = sync_checkpoint_tree(src_dir, dst_dir)
        record_span_event_otel("checkpoint.mirror.complete", attributes={
            "model_id": str(model_id),
            "checkpoint_name": str(checkpoint_name),
            "duration_s": time.perf_counter() - _t0,
        })
        return result


def _mirror_target_path(metadata: dict) -> str:
    configured = metadata.get("persistent_mirror_path")
    if isinstance(configured, str) and configured:
        return configured
    checkpoint_name = metadata.get("checkpoint_id")
    model_id = metadata.get("model_id")
    checkpoint_type = metadata.get("checkpoint_type") or metadata.get("type")
    if not isinstance(checkpoint_name, str) or not checkpoint_name:
        raise ValueError("mirror metadata missing checkpoint_id")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("mirror metadata missing model_id")
    return build_persistent_checkpoint_dir(
        user_id=metadata.get("owner_id"),
        model_id=model_id,
        checkpoint_name=checkpoint_name,
        checkpoint_type=checkpoint_type if checkpoint_type in _CHECKPOINT_TYPES else None,
    )


def _kickoff_pending_checkpoint_mirrors() -> None:
    global _mirror_thread

    def _runner() -> None:
        global _mirror_thread
        try:
            process_pending_checkpoint_mirrors()
        finally:
            with _mirror_thread_lock:
                _mirror_thread = None

    with _mirror_thread_lock:
        if _mirror_thread is not None and _mirror_thread.is_alive():
            return
        _mirror_thread = threading.Thread(
            target=_runner,
            name="checkpoint-mirror-worker",
            daemon=True,
        )
        _mirror_thread.start()


def begin_async_checkpoint_mirror(
    src_dir: str,
    *,
    user_id: str | None,
    model_id: str,
    checkpoint_name: str,
    checkpoint_type: CheckpointType | None = None,
) -> str:
    dst_dir = build_persistent_checkpoint_dir(
        user_id=user_id,
        model_id=model_id,
        checkpoint_name=checkpoint_name,
        checkpoint_type=checkpoint_type,
    )
    update_checkpoint_metadata(
        src_dir,
        {
            "mirror_status": MIRROR_STATUS_PENDING,
            "mirror_error": None,
            "persistent_mirror_path": dst_dir,
            "last_mirror_enqueued_at": _isoformat_utc(time.time()),
        },
    )
    _kickoff_pending_checkpoint_mirrors()
    return dst_dir


def _process_pending_checkpoint_mirror(checkpoint_path: str) -> tuple[str, str]:
    _t0 = time.perf_counter()
    with start_as_current_span(
        "checkpoint.mirror_process",
        component="checkpoints",
        op="mirror_process",
        attributes={"checkpoint_path": str(checkpoint_path)},
    ):
        metadata = read_checkpoint_metadata(checkpoint_path)
    checkpoint_type = metadata.get("checkpoint_type") or metadata.get("type")
    if checkpoint_type not in ("training", "sampler"):
        raise ValueError(f"invalid checkpoint_type for mirror: {checkpoint_type!r}")
    dst_dir = _mirror_target_path(metadata)
    attempt = int(metadata.get("mirror_attempts") or 0) + 1
    update_checkpoint_metadata(
        checkpoint_path,
        {
            "mirror_status": MIRROR_STATUS_IN_PROGRESS,
            "mirror_error": None,
            "mirror_attempts": attempt,
            "last_mirror_attempt_at": _isoformat_utc(time.time()),
            "persistent_mirror_path": dst_dir,
        },
    )
    delay_s = float(os.environ.get("MINT_CHECKPOINT_MIRROR_DELAY_S", "0") or 0.0)
    if delay_s > 0:
        time.sleep(delay_s)
    mirrored_path = sync_checkpoint_tree(checkpoint_path, dst_dir)
    _validate_checkpoint_for_publication(mirrored_path, checkpoint_type)
    mirrored_at = _isoformat_utc(time.time())
    cache_meta = read_checkpoint_metadata(checkpoint_path)
    persistent_meta = dict(cache_meta)
    persistent_meta.update(
        {
            "storage_tier": "persistent_tos",
            "mirror_status": MIRROR_STATUS_COMPLETE,
            "mirror_error": None,
            "next_publish_retry_at": None,
            "persistent_mirror_path": mirrored_path,
            "mirror_completed_at": mirrored_at,
        }
    )
    write_checkpoint_metadata(mirrored_path, persistent_meta)
    update_checkpoint_metadata(
        checkpoint_path,
        {
            "mirror_status": MIRROR_STATUS_COMPLETE,
            "mirror_error": None,
            "next_publish_retry_at": None,
            "persistent_mirror_path": mirrored_path,
            "mirror_completed_at": mirrored_at,
        },
    )
    ckpt_id = persistent_meta.get("ckpt_id")
    if isinstance(ckpt_id, str) and ckpt_id and checkpoint_index_enabled():
        try:
            asyncio.run(
                publish_checkpoint_catalog(
                    ckpt_id,
                    storage_root=get_persistent_checkpoints_dir(),
                    size_bytes=_dir_size_bytes(mirrored_path),
                )
            )
        except CheckpointNotFoundError:
            update_checkpoint_metadata(
                checkpoint_path,
                {
                    "mirror_status": MIRROR_STATUS_FAILED,
                    "mirror_error": "checkpoint_index_publish_not_found",
                },
            )
            raise
        except Exception:
            update_checkpoint_metadata(
                checkpoint_path,
                {
                    "mirror_status": MIRROR_STATUS_FAILED,
                    "mirror_error": "checkpoint_index_publish_failed",
                },
            )
            raise
    return checkpoint_path, mirrored_path


def process_pending_checkpoint_mirrors(*, max_items: int | None = None) -> dict[str, list[str]]:
    if not _mirror_process_lock.acquire(blocking=False):
        return {"mirrored": [], "failed": []}
    try:
        results = {"mirrored": [], "failed": []}
        cache_dirs = sorted(_iter_runtime_checkpoint_dirs(get_persistent_cache_dir()))
        for checkpoint_path in cache_dirs:
            if max_items is not None and len(results["mirrored"]) + len(results["failed"]) >= max_items:
                break
            try:
                metadata = read_checkpoint_metadata(checkpoint_path)
            except Exception:
                continue
            if metadata.get("storage_tier") != "persistent_cache":
                continue
            status = metadata.get("mirror_status")
            if status == MIRROR_STATUS_COMPLETE:
                continue
            if status == MIRROR_STATUS_FAILED and metadata.get("mirror_error") != "checkpoint_index_publish_failed":
                continue

            if status == MIRROR_STATUS_PENDING and metadata.get("mirror_error") == "checkpoint_index_publish_failed":
                retry_at = metadata.get("next_publish_retry_at")
                if isinstance(retry_at, str) and retry_at:
                    try:
                        retry_at_ts = datetime.fromisoformat(retry_at.replace("Z", "+00:00")).timestamp()
                        if time.time() < retry_at_ts:
                            continue
                    except ValueError:
                        pass

            try:
                _, mirrored_path = _process_pending_checkpoint_mirror(checkpoint_path)
                results["mirrored"].append(mirrored_path)
            except Exception as e:
                current_error = None
                try:
                    current_error = read_checkpoint_metadata(checkpoint_path).get("mirror_error")
                except Exception:
                    current_error = None

                if current_error == "checkpoint_index_publish_failed":
                    update_checkpoint_metadata(
                        checkpoint_path,
                        {
                            "mirror_status": MIRROR_STATUS_PENDING,
                            "last_mirror_failure_at": _isoformat_utc(time.time()),
                            "next_publish_retry_at": _isoformat_utc(
                                time.time() + get_checkpoint_publish_retry_backoff_s()
                            ),
                        },
                    )
                else:
                    update_checkpoint_metadata(
                        checkpoint_path,
                        {
                            "mirror_status": MIRROR_STATUS_FAILED,
                            "mirror_error": f"{type(e).__name__}: {e}",
                            "last_mirror_failure_at": _isoformat_utc(time.time()),
                            "next_publish_retry_at": None,
                        },
                    )
                    try:
                        metadata = read_checkpoint_metadata(checkpoint_path)
                        ckpt_id = metadata.get("ckpt_id")
                        if isinstance(ckpt_id, str) and ckpt_id and checkpoint_index_enabled():
                            asyncio.run(mark_checkpoint_failed(ckpt_id, fail_reason="mirror_failed"))
                    except Exception:
                        pass
                results["failed"].append(checkpoint_path)
        return results
    finally:
        _mirror_process_lock.release()


def materialize_persistent_checkpoint(path: str) -> str:
    if not os.path.isdir(path):
        return path
    path_real = os.path.realpath(path)
    runtime_root = os.path.realpath(get_runtime_checkpoints_dir())
    if path_real == runtime_root or path_real.startswith(runtime_root + os.sep):
        return path
    persistent_root = os.path.realpath(get_persistent_checkpoints_dir())
    if not (path_real == persistent_root or path_real.startswith(persistent_root + os.sep)):
        return path

    rel = os.path.relpath(path_real, persistent_root)
    cache_dir = os.path.join(get_persistent_cache_dir(), rel)
    cache_meta = os.path.join(cache_dir, "metadata.json")
    src_meta = os.path.join(path_real, "metadata.json")
    if os.path.isdir(cache_dir) and os.path.exists(src_meta) and os.path.exists(cache_meta):
        try:
            if read_checkpoint_metadata(cache_dir) == read_checkpoint_metadata(path_real):
                os.utime(cache_dir, None)
                return cache_dir
        except Exception:
            pass
    return sync_checkpoint_tree(path_real, cache_dir)


def _iter_runtime_checkpoint_dirs(root: str) -> list[str]:
    if not os.path.isdir(root):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for current_root, dirnames, filenames in os.walk(root):
        has_metadata = "metadata.json" in filenames
        has_lora = "adapter_model.safetensors" in filenames or "bumblebee_rank_sharded_adapter.json" in filenames or bool(
            glob.glob(os.path.join(current_root, "mp_rank_*_adapter.pt"))
        )
        has_optimizer = "optimizer.pt" in filenames or bool(
            glob.glob(os.path.join(current_root, "*_optimizer.pt"))
        )
        if not (has_metadata or has_lora or has_optimizer):
            continue
        real = os.path.realpath(current_root)
        if real in seen:
            continue
        seen.add(real)
        out.append(current_root)
        dirnames[:] = []
    return out


def reap_runtime_checkpoints(*, now: float | None = None) -> dict[str, list[str]]:
    current = now or time.time()
    reaped = {"ephemeral": [], "persistent_cache": [], "persistent": []}

    ephemeral_root = get_ephemeral_checkpoints_dir()
    for checkpoint_path in _iter_runtime_checkpoint_dirs(ephemeral_root):
        checkpoint_name = checkpoint_logical_name(checkpoint_path)
        meta = {}
        try:
            meta = read_checkpoint_metadata(checkpoint_path)
        except Exception:
            meta = {}
        expires_at = metadata_expires_at(meta)
        if expires_at is None:
            age_s = current - os.path.getmtime(checkpoint_path)
            expires_s = get_ephemeral_checkpoint_ttl_s()
            should_delete = is_ephemeral_checkpoint_name(checkpoint_name) and age_s >= expires_s
        else:
            should_delete = current >= expires_at.timestamp()
        if should_delete:
            shutil.rmtree(checkpoint_path, ignore_errors=True)
            reaped["ephemeral"].append(checkpoint_path)

    cache_root = get_persistent_cache_dir()
    cache_ttl_s = get_persistent_cache_ttl_s()
    for checkpoint_path in _iter_runtime_checkpoint_dirs(cache_root):
        meta = {}
        try:
            meta = read_checkpoint_metadata(checkpoint_path)
        except Exception:
            meta = {}
        if meta.get("storage_tier") == "persistent_cache" and meta.get("mirror_status") != MIRROR_STATUS_COMPLETE:
            continue
        age_s = current - os.path.getmtime(checkpoint_path)
        if age_s < cache_ttl_s:
            continue
        shutil.rmtree(checkpoint_path, ignore_errors=True)
        reaped["persistent_cache"].append(checkpoint_path)

    persistent_root = get_persistent_checkpoints_dir()
    for checkpoint_path in _iter_runtime_checkpoint_dirs(persistent_root):
        try:
            meta = read_checkpoint_metadata(checkpoint_path)
        except Exception:
            continue
        expires_at = metadata_expires_at(meta)
        if expires_at is None or current < expires_at.timestamp():
            continue
        shutil.rmtree(checkpoint_path, ignore_errors=True)
        reaped["persistent"].append(checkpoint_path)

    return reaped
