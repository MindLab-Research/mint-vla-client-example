from __future__ import annotations

import glob
import json
import os
import shutil
import tarfile
from pathlib import Path


def safe_extract_checkpoint_archive(archive_path: str, dest_dir: str) -> None:
    """Safely extract a tar.gz checkpoint archive into dest_dir.

    Requirements:
    - Must contain exactly one top-level directory (the checkpoint folder).
    - Reject absolute paths, parent traversal (..), and symlinks/hardlinks.
    - Extract contents of the top-level directory into dest_dir (strip the root folder).
    """
    os.makedirs(dest_dir, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as tf:
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


def resolve_checkpoint_id(checkpoint_id: str, checkpoints_dir: str) -> str | None:
    if not os.path.exists(checkpoints_dir):
        return None

    for top_level in os.listdir(checkpoints_dir):
        top_path = os.path.join(checkpoints_dir, top_level)
        if not os.path.isdir(top_path):
            continue
        for sub_dir in os.listdir(top_path):
            sub_path = os.path.join(top_path, sub_dir)
            if not os.path.isdir(sub_path):
                continue
            metadata_path = os.path.join(sub_path, "metadata.json")
            if not os.path.exists(metadata_path):
                continue
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
                if metadata.get("checkpoint_id") == checkpoint_id:
                    return sub_path
            except (json.JSONDecodeError, OSError):
                pass

    return None


def resolve_checkpoint_uri(uri: str, checkpoints_dir: str) -> str:
    if uri.startswith("ckpt_"):
        resolved = resolve_checkpoint_id(uri, checkpoints_dir)
        return resolved or uri

    if uri.startswith("tinker://"):
        path_part = uri[len("tinker://"):]
        return os.path.join(checkpoints_dir, path_part)
    if uri.startswith("mint://"):
        path_part = uri[len("mint://"):]
        return os.path.join(checkpoints_dir, path_part)
    if uri.startswith("file://"):
        return uri[7:]
    return uri

