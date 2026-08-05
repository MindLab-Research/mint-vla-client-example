"""Immutable, hash-addressed pi0.5 contracts shared by training and action inference."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping
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
    state_dim: int | None = None
    fail_on_token_truncation: bool = False
    state_schema: str | None = None
    action_physical_dim: int | None = None
    delta_mask_segments: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.action_dim <= 0 or self.action_horizon <= 0 or self.max_token_len <= 0:
            raise ValueError("pi0.5 profile dimensions must be positive")
        if self.state_dim is not None and self.state_dim <= 0:
            raise ValueError("pi0.5 profile state_dim must be positive")
        if self.state_schema is not None and self.state_dim is None:
            raise ValueError("state_schema requires an explicit state_dim")
        if self.action_physical_dim is not None and not (0 < self.action_physical_dim <= self.action_dim):
            raise ValueError("action_physical_dim must be within action_dim")
        if self.delta_mask_segments is not None:
            if sum(abs(value) for value in self.delta_mask_segments) != self.action_dim:
                raise ValueError("delta_mask_segments must cover action_dim exactly")
            if self.action_physical_dim is None:
                raise ValueError("delta_mask_segments requires action_physical_dim")

    def manifest(self) -> dict[str, Any]:
        manifest = {
            "profile_id": self.profile_id,
            "paligemma_variant": self.paligemma_variant,
            "action_expert_variant": self.action_expert_variant,
            "action_dim": self.action_dim,
            "action_horizon": self.action_horizon,
            "max_token_len": self.max_token_len,
            "discrete_state_input": self.discrete_state_input,
            "expected_trainable_count": self.expected_trainable_count,
        }
        # Preserve the published v1 manifest byte contract.  A distinct state
        # dimension is explicit and hash-authenticated for new profiles.
        if self.resolved_state_dim != self.action_dim:
            manifest["state_dim"] = self.resolved_state_dim
        if self.fail_on_token_truncation:
            manifest["fail_on_token_truncation"] = True
        if self.state_schema is not None:
            manifest["state_schema"] = self.state_schema
        if self.action_physical_dim is not None:
            manifest["action_physical_dim"] = self.action_physical_dim
        if self.delta_mask_segments is not None:
            manifest["delta_mask_segments"] = list(self.delta_mask_segments)
        return manifest

    @property
    def resolved_state_dim(self) -> int:
        return self.action_dim if self.state_dim is None else self.state_dim

    @property
    def manifest_hash(self) -> str:
        encoded = json.dumps(self.manifest(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def checkpoint_manifest(self) -> dict[str, Any]:
        return {**self.manifest(), "manifest_hash": self.manifest_hash}

    def pi0_config_kwargs(self) -> dict[str, Any]:
        kwargs = {
            "pi05": True,
            "action_dim": self.action_dim,
            "action_horizon": self.action_horizon,
            "max_token_len": self.max_token_len,
            "discrete_state_input": self.discrete_state_input,
            "paligemma_variant": self.paligemma_variant,
            "action_expert_variant": self.action_expert_variant,
        }
        if self.resolved_state_dim != self.action_dim:
            kwargs["state_dim"] = self.resolved_state_dim
        if self.fail_on_token_truncation:
            kwargs["fail_on_token_truncation"] = True
        return kwargs


PI05_ACTION_LORA_R16_V1 = Pi05Profile(
    profile_id="pi05_action_lora_r16_v1",
    paligemma_variant="gemma_2b",
    action_expert_variant="gemma_300m_lora_r16",
    action_dim=32,
    state_dim=None,
    action_horizon=10,
    max_token_len=200,
    discrete_state_input=True,
    expected_trainable_count=13_224_992,
)

PI05_ACTION_LORA_R16_STATE54_V1 = Pi05Profile(
    profile_id="pi05_action_lora_r16_state54_v1",
    paligemma_variant="gemma_2b",
    action_expert_variant="gemma_300m_lora_r16",
    action_dim=32,
    state_dim=54,
    action_horizon=10,
    max_token_len=256,
    discrete_state_input=True,
    expected_trainable_count=13_224_992,
    fail_on_token_truncation=True,
)

PI05_ACTION_LORA_R16_STATE56_28DOF_V1 = Pi05Profile(
    profile_id="pi05_action_lora_r16_state56_28dof_v1",
    paligemma_variant="gemma_2b",
    action_expert_variant="gemma_300m_lora_r16",
    action_dim=32,
    state_dim=56,
    action_horizon=10,
    max_token_len=256,
    discrete_state_input=True,
    expected_trainable_count=13_224_992,
    fail_on_token_truncation=True,
    state_schema="mano_object_dynamics_state56_native28_v1",
    action_physical_dim=28,
    delta_mask_segments=(3, -3, 22, -4),
)

_PROFILES = {
    profile.profile_id: profile
    for profile in (
        PI05_ACTION_LORA_R16_V1,
        PI05_ACTION_LORA_R16_STATE54_V1,
        PI05_ACTION_LORA_R16_STATE56_28DOF_V1,
    )
}


# This is intentionally outside Pi05Profile.manifest(): it is a runtime model-layout
# contract, while the published checkpoint manifest remains byte-for-byte stable.
_PI05_ACTION_LORA_R16_V1_TRAINABLE_LEAF_SHAPES: dict[str, tuple[int, ...]] = {
    "PaliGemma/llm/layers/attn/attn_vec_einsum_1/lora_a": (18, 8, 256, 16),
    "PaliGemma/llm/layers/attn/attn_vec_einsum_1/lora_b": (18, 8, 16, 1024),
    "PaliGemma/llm/layers/attn/kv_einsum_1/lora_a": (18, 2, 1, 1024, 16),
    "PaliGemma/llm/layers/attn/kv_einsum_1/lora_b": (18, 2, 1, 16, 256),
    "PaliGemma/llm/layers/attn/q_einsum_1/lora_a": (18, 8, 1024, 16),
    "PaliGemma/llm/layers/attn/q_einsum_1/lora_b": (18, 8, 16, 256),
    "PaliGemma/llm/layers/mlp_1/gating_einsum_lora_a": (18, 2, 1024, 16),
    "PaliGemma/llm/layers/mlp_1/gating_einsum_lora_b": (18, 2, 16, 4096),
    "PaliGemma/llm/layers/mlp_1/linear_lora_a": (18, 4096, 16),
    "PaliGemma/llm/layers/mlp_1/linear_lora_b": (18, 16, 1024),
    "action_in_proj/bias": (1024,),
    "action_in_proj/kernel": (32, 1024),
    "action_out_proj/bias": (32,),
    "action_out_proj/kernel": (1024, 32),
    "time_mlp_in/bias": (1024,),
    "time_mlp_in/kernel": (1024, 1024),
    "time_mlp_out/bias": (1024,),
    "time_mlp_out/kernel": (1024, 1024),
}


def validate_profile_trainable_leaves(
    profile: Pi05Profile, actual: Mapping[str, tuple[int, ...]]
) -> None:
    """Fail closed when a profiled model's trainable leaves differ from its layout contract."""
    if profile.profile_id not in {
        PI05_ACTION_LORA_R16_V1.profile_id,
        PI05_ACTION_LORA_R16_STATE54_V1.profile_id,
        PI05_ACTION_LORA_R16_STATE56_28DOF_V1.profile_id,
    }:
        return

    expected = _PI05_ACTION_LORA_R16_V1_TRAINABLE_LEAF_SHAPES
    normalized_actual = {path: tuple(int(dim) for dim in shape) for path, shape in actual.items()}
    missing = sorted(set(expected) - set(normalized_actual))
    unexpected = sorted(set(normalized_actual) - set(expected))
    wrong_shape = [
        f"{path}: expected={expected[path]}, actual={normalized_actual[path]}"
        for path in sorted(set(expected) & set(normalized_actual))
        if expected[path] != normalized_actual[path]
    ]
    if missing or unexpected or wrong_shape:
        raise ValueError(
            "OpenPI pi0.5 profile trainable-leaf mismatch: "
            f"profile={profile.profile_id!r}; missing={missing}; "
            f"unexpected={unexpected}; wrong_shape={wrong_shape}"
        )


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
