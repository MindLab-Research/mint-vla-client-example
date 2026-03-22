"""Checkpoint path, archive, and runtime-cache helpers shared across routes."""

from __future__ import annotations

import asyncio
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
import fcntl

DEFAULT_PERSISTENT_CHECKPOINTS_DIR = "/tos-mindverse/tinker_checkpoints"
DEFAULT_RUNTIME_CHECKPOINTS_DIR = "/vePFS-Mindverse/share/tinker_runtime_checkpoints"
DEFAULT_LEGACY_PFS_CHECKPOINTS_DIR = "/vePFS-Mindverse/share/tinker_checkpoints"
DEFAULT_LEGACY_CODE_CHECKPOINTS_DIR = "/vePFS-Mindverse/share/code/tinker-server/checkpoints"
DEFAULT_EPHEMERAL_TTL_S = 24 * 60 * 60
DEFAULT_PERSISTENT_CACHE_TTL_S = 24 * 60 * 60
DEFAULT_REAP_INTERVAL_S = 10 * 60
DEFAULT_MIRROR_POLL_S = 5

# Backward-compatible module globals. Existing tests patch CHECKPOINTS_DIR directly.
CHECKPOINTS_DIR = os.environ.get("TINKER_CHECKPOINT_DIR", DEFAULT_PERSISTENT_CHECKPOINTS_DIR)
PERSISTENT_CHECKPOINTS_DIR = os.environ.get("TINKER_PERSISTENT_CHECKPOINT_DIR", CHECKPOINTS_DIR)
RUNTIME_CHECKPOINTS_DIR = os.environ.get("TINKER_RUNTIME_CHECKPOINT_DIR", DEFAULT_RUNTIME_CHECKPOINTS_DIR)

CheckpointType = Literal["training", "sampler"]
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


def is_ephemeral_checkpoint_name(name: str) -> bool:
    return os.path.basename(name).startswith("_ephemeral_")


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


def get_legacy_checkpoint_dirs() -> list[str]:
    extra = os.environ.get("MINT_LEGACY_CHECKPOINT_DIRS", "")
    paths = [DEFAULT_LEGACY_PFS_CHECKPOINTS_DIR, DEFAULT_LEGACY_CODE_CHECKPOINTS_DIR]
    if os.path.realpath(CHECKPOINTS_DIR) != os.path.realpath(PERSISTENT_CHECKPOINTS_DIR):
        paths.insert(0, CHECKPOINTS_DIR)
    paths.extend([p for p in extra.split(":") if p])
    return _dedupe_paths(paths)


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


def build_checkpoint_dir(root: str, *, user_id: str | None, model_id: str, checkpoint_name: str) -> str:
    owner_dir = checkpoint_owner_dir(user_id)
    return os.path.join(root, owner_dir, model_id, checkpoint_name)


def build_ephemeral_checkpoint_dir(*, user_id: str | None, model_id: str, checkpoint_name: str) -> str:
    return build_checkpoint_dir(
        get_ephemeral_checkpoints_dir(), user_id=user_id, model_id=model_id, checkpoint_name=checkpoint_name
    )


def build_persistent_checkpoint_dir(*, user_id: str | None, model_id: str, checkpoint_name: str) -> str:
    return build_checkpoint_dir(
        get_persistent_checkpoints_dir(), user_id=user_id, model_id=model_id, checkpoint_name=checkpoint_name
    )


def build_persistent_cache_dir(*, user_id: str | None, model_id: str, checkpoint_name: str) -> str:
    return build_checkpoint_dir(
        get_persistent_cache_dir(), user_id=user_id, model_id=model_id, checkpoint_name=checkpoint_name
    )


def checkpoint_has_lora_weights(path: str) -> bool:
    return os.path.exists(os.path.join(path, "adapter_model.safetensors")) or bool(
        glob.glob(os.path.join(path, "mp_rank_*_adapter.pt"))
    )


def checkpoint_has_sampling_adapter_weights(path: str) -> bool:
    return os.path.exists(os.path.join(path, "adapter_model.safetensors"))


def checkpoint_has_optimizer_state(path: str) -> bool:
    return os.path.exists(os.path.join(path, "optimizer.pt")) or bool(
        glob.glob(os.path.join(path, "*_optimizer.pt"))
    )


def read_checkpoint_metadata(path: str) -> dict:
    meta_path = os.path.join(path, "metadata.json")
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
    with open(meta_path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def update_checkpoint_metadata(path: str, updates: dict) -> dict:
    current = {}
    try:
        current = read_checkpoint_metadata(path)
    except Exception:
        current = {}
    current.update(dict(updates))
    write_checkpoint_metadata(path, current)
    return current


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

    if not has_lora:
        raise ValueError("Missing LoRA weights in extracted checkpoint")
    if checkpoint_type == "training":
        if not has_optimizer:
            raise ValueError("Missing optimizer state in extracted checkpoint")
    elif checkpoint_type == "sampler":
        if has_optimizer:
            raise ValueError("Sampler checkpoint must not include optimizer state")
    else:
        if not has_optimizer:
            raise ValueError("Missing optimizer state in extracted checkpoint")


def validate_sampler_checkpoint_for_sampling(path: str) -> None:
    if not checkpoint_has_sampling_adapter_weights(path):
        raise ValueError("Missing adapter_model.safetensors for sampling")
    if checkpoint_has_optimizer_state(path):
        raise ValueError("Sampler checkpoint must not include optimizer state")
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
    from .backend.async_ray_control import _await_ray_ref, _ensure_ray_initialized

    import ray

    if not os.path.isdir(checkpoint_dir):
        raise ValueError(f"Checkpoint dir is not a directory: {checkpoint_dir}")

    archive_dir = os.path.dirname(os.path.abspath(archive_path))
    if archive_dir:
        os.makedirs(archive_dir, exist_ok=True)

    _ensure_ray_initialized()
    ref = _create_checkpoint_archive_remote().remote(str(checkpoint_dir), str(archive_path))
    try:
        await asyncio.wait_for(_await_ray_ref(ref), timeout=float(timeout_s))
    except asyncio.TimeoutError:
        ray.cancel(ref, force=True)
        raise


def _iter_metadata_paths(
    roots: list[str],
    *,
    user_id: str | None = None,
    is_admin: bool = False,
) -> list[str]:
    patterns: list[str] = []
    if user_id and not is_admin:
        base = os.path.join("", checkpoint_owner_dir(user_id))
        prefixes = [base]
    else:
        prefixes = ["*", ""]

    for root in roots:
        if not os.path.exists(root):
            continue
        for prefix in prefixes:
            if prefix:
                patterns.extend(
                    [
                        os.path.join(root, prefix, "*", "metadata.json"),
                        os.path.join(root, prefix, "*", "*", "metadata.json"),
                    ]
                )
            else:
                patterns.extend(
                    [
                        os.path.join(root, "*", "metadata.json"),
                        os.path.join(root, "*", "*", "metadata.json"),
                    ]
                )

    out: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for metadata_path in glob.glob(pattern):
            real = os.path.realpath(metadata_path)
            if real in seen:
                continue
            seen.add(real)
            out.append(metadata_path)
    return out


def resolve_checkpoint_id(
    checkpoint_id: str,
    checkpoints_dir: str,
    *,
    user_id: str | None = None,
    is_admin: bool = False,
) -> str | None:
    for metadata_path in _iter_metadata_paths(
        get_resolution_roots(primary_root=checkpoints_dir),
        user_id=user_id,
        is_admin=is_admin,
    ):
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if metadata.get("checkpoint_id") == checkpoint_id:
            return os.path.dirname(metadata_path)
    return None


def _strip_tinker_checkpoint_kind(path_part: str) -> str:
    parts = path_part.split("/")
    if len(parts) >= 3 and parts[1] in ("weights", "sampler_weights"):
        return "/".join([parts[0], *parts[2:]])
    return path_part


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
    owner_dir = checkpoint_owner_dir(user_id)

    if uri.startswith("ckpt_"):
        resolved = resolve_checkpoint_id(uri, checkpoints_dir, user_id=owner_dir, is_admin=is_admin)
        return resolved or uri

    if uri.startswith("tinker://"):
        path_part = _strip_tinker_checkpoint_kind(uri[len("tinker://") :])
    elif uri.startswith("mint://"):
        path_part = _strip_tinker_checkpoint_kind(uri[len("mint://") :])
    else:
        return uri

    if is_admin:
        for root in roots:
            legacy = os.path.join(root, path_part)
            if os.path.exists(legacy):
                return legacy
            matches = glob.glob(os.path.join(root, "*", path_part))
            if len(matches) == 1 and os.path.exists(matches[0]):
                return matches[0]
        return os.path.join(get_persistent_checkpoints_dir(), path_part)

    for root in roots:
        candidate = os.path.join(root, owner_dir, path_part)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(get_persistent_checkpoints_dir(), owner_dir, path_part)


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
) -> str:
    dst_dir = build_persistent_checkpoint_dir(user_id=user_id, model_id=model_id, checkpoint_name=checkpoint_name)
    return sync_checkpoint_tree(src_dir, dst_dir)


def _mirror_target_path(metadata: dict) -> str:
    configured = metadata.get("persistent_mirror_path")
    if isinstance(configured, str) and configured:
        return configured
    checkpoint_name = metadata.get("checkpoint_id")
    model_id = metadata.get("model_id")
    if not isinstance(checkpoint_name, str) or not checkpoint_name:
        raise ValueError("mirror metadata missing checkpoint_id")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("mirror metadata missing model_id")
    return build_persistent_checkpoint_dir(
        user_id=metadata.get("owner_id"),
        model_id=model_id,
        checkpoint_name=checkpoint_name,
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
) -> str:
    dst_dir = build_persistent_checkpoint_dir(
        user_id=user_id,
        model_id=model_id,
        checkpoint_name=checkpoint_name,
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
            "persistent_mirror_path": mirrored_path,
            "mirror_completed_at": mirrored_at,
        },
    )
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
            if status in (MIRROR_STATUS_COMPLETE, MIRROR_STATUS_FAILED):
                continue
            try:
                _, mirrored_path = _process_pending_checkpoint_mirror(checkpoint_path)
                results["mirrored"].append(mirrored_path)
            except Exception as e:
                update_checkpoint_metadata(
                    checkpoint_path,
                    {
                        "mirror_status": MIRROR_STATUS_FAILED,
                        "mirror_error": f"{type(e).__name__}: {e}",
                        "last_mirror_failure_at": _isoformat_utc(time.time()),
                    },
                )
                results["failed"].append(checkpoint_path)
        return results
    finally:
        _mirror_process_lock.release()


def materialize_persistent_checkpoint(path: str) -> str:
    if not os.path.isdir(path):
        return path
    persistent_root = os.path.realpath(get_persistent_checkpoints_dir())
    path_real = os.path.realpath(path)
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
    for owner in os.listdir(root):
        owner_path = os.path.join(root, owner)
        if not os.path.isdir(owner_path):
            continue
        for model_id in os.listdir(owner_path):
            model_path = os.path.join(owner_path, model_id)
            if not os.path.isdir(model_path):
                continue
            for checkpoint_name in os.listdir(model_path):
                checkpoint_path = os.path.join(model_path, checkpoint_name)
                if os.path.isdir(checkpoint_path):
                    out.append(checkpoint_path)
    return out


def reap_runtime_checkpoints(*, now: float | None = None) -> dict[str, list[str]]:
    current = now or time.time()
    reaped = {"ephemeral": [], "persistent_cache": [], "persistent": []}

    ephemeral_root = get_ephemeral_checkpoints_dir()
    for checkpoint_path in _iter_runtime_checkpoint_dirs(ephemeral_root):
        checkpoint_name = os.path.basename(checkpoint_path)
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
