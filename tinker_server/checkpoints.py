"""Checkpoint path and archive helpers shared across routes.

Issue #86: training resume failed due to inconsistent checkpoint directory defaults and
checkpoint path resolvers between routes.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import tarfile
from pathlib import Path

DEFAULT_CHECKPOINTS_DIR = "/vePFS-Mindverse/share/tinker_checkpoints"
CHECKPOINTS_DIR = os.environ.get("TINKER_CHECKPOINT_DIR", DEFAULT_CHECKPOINTS_DIR)


def get_checkpoints_dir() -> str:
    return CHECKPOINTS_DIR


def safe_extract_checkpoint_archive(archive_path: str, dest_dir: str) -> None:
    """Safely extract a tar.gz checkpoint archive into dest_dir.

    Requirements:
    - Must contain exactly one top-level directory (the checkpoint folder).
    - Reject absolute paths, parent traversal (..), and symlinks/hardlinks.
    - Extract contents of the top-level directory into dest_dir (strip the root folder).
    """
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
                continue  # top-level directory entry itself

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


def validate_checkpoint_dir(path: str) -> None:
    has_lora = os.path.exists(os.path.join(path, "adapter_model.safetensors")) or bool(
        glob.glob(os.path.join(path, "mp_rank_*_adapter.pt"))
    )
    has_optimizer = os.path.exists(os.path.join(path, "optimizer.pt")) or bool(
        glob.glob(os.path.join(path, "*_optimizer.pt"))
    )

    if not has_lora:
        raise ValueError("Missing LoRA weights in extracted checkpoint")
    if not has_optimizer:
        raise ValueError("Missing optimizer state in extracted checkpoint")


def resolve_checkpoint_id(
    checkpoint_id: str, checkpoints_dir: str, *, user_id: str | None = None
) -> str | None:
    if not os.path.exists(checkpoints_dir):
        return None

    if user_id and user_id != "admin":
        base = os.path.join(checkpoints_dir, user_id)
        patterns = [
            os.path.join(base, "*", "metadata.json"),  # /checkpoints/<user>/<ckpt_id>/
            os.path.join(base, "*", "*", "metadata.json"),  # /checkpoints/<user>/<model_id>/<ckpt_name>/
        ]
    else:
        patterns = [
            os.path.join(checkpoints_dir, "*", "*", "metadata.json"),  # /checkpoints/<model_id>/<ckpt_name>/
            os.path.join(checkpoints_dir, "*", "metadata.json"),  # /checkpoints/<ckpt_id>/ (rare)
            os.path.join(checkpoints_dir, "*", "*", "*", "metadata.json"),  # /checkpoints/<user>/<model_id>/<ckpt_name>/
        ]

    for pattern in patterns:
        for metadata_path in glob.glob(pattern):
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
                if metadata.get("checkpoint_id") == checkpoint_id:
                    return os.path.dirname(metadata_path)
            except (json.JSONDecodeError, OSError):
                pass

    return None


def resolve_checkpoint_uri(uri: str, checkpoints_dir: str, *, user_id: str | None = None) -> str:
    if uri.startswith("file://"):
        return uri[7:]

    if uri.startswith("ckpt_"):
        owner_dir = user_id or "anonymous"
        resolved = resolve_checkpoint_id(uri, checkpoints_dir, user_id=owner_dir)
        if resolved is None and user_id == "admin":
            resolved = resolve_checkpoint_id(uri, checkpoints_dir, user_id=None)
        return resolved or uri

    if uri.startswith("tinker://"):
        path_part = uri[len("tinker://") :]
        if user_id == "admin":
            legacy = os.path.join(checkpoints_dir, path_part)
            if os.path.exists(legacy):
                return legacy
            matches = glob.glob(os.path.join(checkpoints_dir, "*", path_part))
            if len(matches) == 1 and os.path.exists(matches[0]):
                return matches[0]
            return legacy

        owner_dir = user_id or "anonymous"
        return os.path.join(checkpoints_dir, owner_dir, path_part)
    if uri.startswith("mint://"):
        path_part = uri[len("mint://") :]
        if user_id == "admin":
            legacy = os.path.join(checkpoints_dir, path_part)
            if os.path.exists(legacy):
                return legacy
            matches = glob.glob(os.path.join(checkpoints_dir, "*", path_part))
            if len(matches) == 1 and os.path.exists(matches[0]):
                return matches[0]
            return legacy

        owner_dir = user_id or "anonymous"
        return os.path.join(checkpoints_dir, owner_dir, path_part)
    return uri


def resolve_checkpoint_path(state_uri: str, *, user_id: str | None = None) -> str:
    return resolve_checkpoint_uri(state_uri, CHECKPOINTS_DIR, user_id=user_id)
