"""Checkpoint path helpers shared across routes.

Issue #86: training resume failed due to inconsistent CHECKPOINTS_DIR defaults and
checkpoint path resolvers between routes/weights.py and routes/training.py.
"""

from __future__ import annotations

import json
import os

# Shared filesystem required for distributed deployments.
CHECKPOINTS_DIR = os.environ.get(
    "TINKER_CHECKPOINT_DIR",
    "/vePFS-Mindverse/share/code/tinker-server/checkpoints",
)


def _find_checkpoint_dir_by_id(checkpoint_id: str) -> str | None:
    """Find checkpoint directory by scanning metadata.json files."""
    try:
        owners = os.listdir(CHECKPOINTS_DIR)
    except OSError:
        return None

    for owner in owners:
        owner_path = os.path.join(CHECKPOINTS_DIR, owner)
        if not os.path.isdir(owner_path):
            continue
        try:
            sub_dirs = os.listdir(owner_path)
        except OSError:
            continue

        for sub_dir in sub_dirs:
            ckpt_path = os.path.join(owner_path, sub_dir)
            if not os.path.isdir(ckpt_path):
                continue
            metadata_path = os.path.join(ckpt_path, "metadata.json")
            if not os.path.exists(metadata_path):
                continue
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if metadata.get("checkpoint_id") == checkpoint_id:
                return ckpt_path

    return None


def resolve_checkpoint_path(state_uri: str) -> str:
    """Resolve a checkpoint identifier/URI to a filesystem path.

    Supported inputs:
    - checkpoint_id ("ckpt_xxx"): scans CHECKPOINTS_DIR/*/*/metadata.json
    - tinker://{path}: mapped under CHECKPOINTS_DIR
    - mint://{path}: mapped under CHECKPOINTS_DIR
    - file:///abs/path: strips prefix
    - /abs/path: returned as-is
    """
    if state_uri.startswith("ckpt_"):
        found = _find_checkpoint_dir_by_id(state_uri)
        return found if found is not None else state_uri

    if state_uri.startswith("tinker://"):
        return os.path.join(CHECKPOINTS_DIR, state_uri[len("tinker://") :])
    if state_uri.startswith("mint://"):
        return os.path.join(CHECKPOINTS_DIR, state_uri[len("mint://") :])
    if state_uri.startswith("file://"):
        return state_uri[7:]

    return state_uri

