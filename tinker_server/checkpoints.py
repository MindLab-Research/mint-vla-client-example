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
from typing import Literal

DEFAULT_CHECKPOINTS_DIR = "/vePFS-Mindverse/share/tinker_checkpoints"
CHECKPOINTS_DIR = os.environ.get("TINKER_CHECKPOINT_DIR", DEFAULT_CHECKPOINTS_DIR)

CheckpointType = Literal["training", "sampler"]


def checkpoint_has_lora_weights(path: str) -> bool:
    return os.path.exists(os.path.join(path, "adapter_model.safetensors")) or bool(
        glob.glob(os.path.join(path, "mp_rank_*_adapter.pt"))
    )


def checkpoint_has_optimizer_state(path: str) -> bool:
    return os.path.exists(os.path.join(path, "optimizer.pt")) or bool(
        glob.glob(os.path.join(path, "*_optimizer.pt"))
    )


def read_checkpoint_metadata(path: str) -> dict:
    meta_path = os.path.join(path, "metadata.json")
    with open(meta_path) as f:
        return json.load(f)


def write_checkpoint_metadata(path: str, metadata: dict) -> None:
    os.makedirs(path, exist_ok=True)
    meta_path = os.path.join(path, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)


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
        # Backward-compatible default: uploads are full-state unless explicitly labeled.
        if not has_optimizer:
            raise ValueError("Missing optimizer state in extracted checkpoint")


def create_checkpoint_archive(checkpoint_dir: str, archive_path: str) -> None:
    """Create a tar.gz containing exactly one top-level checkpoint directory.

    This produces an archive compatible with safe_extract_checkpoint_archive().
    """
    if not os.path.isdir(checkpoint_dir):
        raise ValueError(f"Checkpoint dir is not a directory: {checkpoint_dir}")

    root = os.path.basename(os.path.normpath(checkpoint_dir))
    if not root:
        raise ValueError(f"Invalid checkpoint dir: {checkpoint_dir}")

    with tarfile.open(archive_path, "w:gz") as tf:
        tf.add(checkpoint_dir, arcname=root, recursive=True)


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
        # If a non-admin tries to use a checkpoint_id they don't own, resolve the real
        # path so callers can raise PermissionError("Access denied") instead of a
        # misleading FileNotFoundError.
        if resolved is None and user_id and user_id != "admin":
            resolved = resolve_checkpoint_id(uri, checkpoints_dir, user_id=None)
        return resolved or uri

    def _strip_tinker_checkpoint_kind(path_part: str) -> str:
        # Canonical Tinker checkpoint paths include an explicit kind segment:
        #   {run_id}/weights/{checkpoint_name}
        #   {run_id}/sampler_weights/{checkpoint_name}
        #
        # MinT's on-disk layout is:
        #   {owner}/{run_id}/{checkpoint_name}
        #
        # So for canonical paths, we strip the kind segment when resolving to disk.
        parts = path_part.split("/")
        if len(parts) >= 3 and parts[1] in ("weights", "sampler_weights"):
            return "/".join([parts[0], *parts[2:]])
        return path_part

    if uri.startswith("tinker://"):
        path_part = _strip_tinker_checkpoint_kind(uri[len("tinker://") :])
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
        path_part = _strip_tinker_checkpoint_kind(uri[len("mint://") :])
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
