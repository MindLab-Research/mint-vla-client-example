"""Canonical local output paths for generated-data inference artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import re


_MODE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def client_results_root() -> Path:
    """Return the client-local generated-artifact root, honoring an explicit override."""
    configured = os.environ.get("VLA_CLIENT_RESULTS_ROOT")
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parents[2] / "results"


def client_inference_root() -> Path:
    """Return the canonical root for Mode3/Mode4 generated-data inference."""
    configured = os.environ.get("VLA_CLIENT_INFERENCE_ROOT")
    if configured:
        return Path(configured).expanduser()
    return client_results_root() / "inference"


def default_inference_output_dir(mode: str) -> Path:
    """Create a collision-resistant, inspectable default run path without touching disk."""
    if not _MODE_RE.fullmatch(mode):
        raise ValueError(f"invalid inference mode name: {mode!r}")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return client_inference_root() / f"{mode}_{timestamp}_{os.getpid()}"
