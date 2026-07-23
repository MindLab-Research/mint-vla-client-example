"""Immutable, hash-addressed pi0.5 contracts shared by training and action inference."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import uuid


PROFILE_MANIFEST_FILENAME = "mint_pi05_profile.json"


@dataclasses.dataclass(frozen=True)
class Pi05Profile:
    profile_id: str
    paligemma_variant: str
    action_expert_variant: str
    action_dim: int
    action_horizon: int
    max_token_len: int
    discrete_state_input: bool
    expected_trainable_count: int

    def manifest(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "paligemma_variant": self.paligemma_variant,
            "action_expert_variant": self.action_expert_variant,
            "action_dim": self.action_dim,
            "action_horizon": self.action_horizon,
            "max_token_len": self.max_token_len,
            "discrete_state_input": self.discrete_state_input,
            "expected_trainable_count": self.expected_trainable_count,
        }

    @property
    def manifest_hash(self) -> str:
        encoded = json.dumps(self.manifest(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def checkpoint_manifest(self) -> dict[str, Any]:
        return {**self.manifest(), "manifest_hash": self.manifest_hash}

    def pi0_config_kwargs(self) -> dict[str, Any]:
        return {
            "pi05": True,
            "action_dim": self.action_dim,
            "action_horizon": self.action_horizon,
            "max_token_len": self.max_token_len,
            "discrete_state_input": self.discrete_state_input,
            "paligemma_variant": self.paligemma_variant,
            "action_expert_variant": self.action_expert_variant,
        }


PI05_ACTION_LORA_R16_V1 = Pi05Profile(
    profile_id="pi05_action_lora_r16_v1",
    paligemma_variant="gemma_2b",
    action_expert_variant="gemma_300m_lora_r16",
    action_dim=32,
    action_horizon=10,
    max_token_len=200,
    discrete_state_input=True,
    expected_trainable_count=13_224_992,
)

_PROFILES = {PI05_ACTION_LORA_R16_V1.profile_id: PI05_ACTION_LORA_R16_V1}


def get_pi05_profile(profile_id: str) -> Pi05Profile:
    try:
        return _PROFILES[profile_id]
    except KeyError as exc:
        raise ValueError(f"Unknown pi0.5 profile {profile_id!r}") from exc


def profile_from_model_config(model_config: Any) -> Pi05Profile:
    profile_id = str(getattr(model_config, "profile", "") or "")
    if not profile_id:
        raise ValueError("OpenPI pi0.5 model config must declare a profile")
    return get_pi05_profile(profile_id)


def profile_manifest_path(root: str | Path) -> Path:
    return Path(root) / PROFILE_MANIFEST_FILENAME


def write_profile_manifest(root: str | Path, profile: Pi05Profile) -> Path:
    """Atomically writes the exact profile contract before a checkpoint is published."""
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    destination = profile_manifest_path(root_path)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    payload = json.dumps(profile.checkpoint_manifest(), sort_keys=True, separators=(",", ":"))
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def validate_profile_manifest(root: str | Path, profile: Pi05Profile) -> Path:
    """Requires an on-disk manifest exactly matching *profile*, including its hash."""
    path = profile_manifest_path(root)
    if not path.is_file():
        raise FileNotFoundError(f"OpenPI pi0.5 profile manifest is required: {path}")
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"OpenPI pi0.5 profile manifest is invalid JSON: {path}") from exc
    expected = profile.checkpoint_manifest()
    if actual != expected:
        raise ValueError(
            "OpenPI pi0.5 profile manifest mismatch: "
            f"expected profile_id={expected['profile_id']!r} hash={expected['manifest_hash']}, "
            f"got profile_id={actual.get('profile_id')!r} hash={actual.get('manifest_hash')!r}"
        )
    return path
