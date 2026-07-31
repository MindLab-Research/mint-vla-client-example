"""Dependency-light OpenPI client model identities and invariants.

These profiles select the server contract and the client-side tokenization mode.
They do not select or place adapters: that remains entirely server-owned.
"""
from __future__ import annotations

from dataclasses import dataclass


LEGACY_L_LORA_MODEL = "openpi/pi05-libero-low-mem-finetune"
ACTION_LORA_R16_MODEL = "openpi/pi05-action-lora-r16-finetune"
ACTION_LORA_R16_STATE54_MODEL = "openpi/pi05-action-lora-r16-state54-finetune"


@dataclass(frozen=True)
class OpenPIClientProfile:
    profile_id: str
    base_model: str
    discrete_state_input: bool
    paligemma_variant: str
    action_expert_variant: str
    action_dim: int = 32
    state_dim: int = 32
    action_horizon: int = 10
    max_tokens: int = 200
    fail_on_token_truncation: bool = False


LEGACY_L_LORA_PROFILE = OpenPIClientProfile(
    profile_id="pi05_libero_low_mem_lora_v1",
    base_model=LEGACY_L_LORA_MODEL,
    discrete_state_input=False,
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m",
)
ACTION_LORA_R16_PROFILE = OpenPIClientProfile(
    profile_id="pi05_action_lora_r16_v1",
    base_model=ACTION_LORA_R16_MODEL,
    discrete_state_input=True,
    paligemma_variant="gemma_2b",
    action_expert_variant="gemma_300m_lora_r16",
)
ACTION_LORA_R16_STATE54_PROFILE = OpenPIClientProfile(
    profile_id="pi05_action_lora_r16_state54_v1",
    base_model=ACTION_LORA_R16_STATE54_MODEL,
    discrete_state_input=True,
    paligemma_variant="gemma_2b",
    action_expert_variant="gemma_300m_lora_r16",
    state_dim=54,
    max_tokens=256,
    fail_on_token_truncation=True,
)

MODEL_PROFILES = {
    profile.base_model: profile
    for profile in (
        LEGACY_L_LORA_PROFILE,
        ACTION_LORA_R16_PROFILE,
        ACTION_LORA_R16_STATE54_PROFILE,
    )
}
PROFILE_IDS = {profile.profile_id: profile for profile in MODEL_PROFILES.values()}
MODEL_CHOICES = tuple(MODEL_PROFILES)


def resolve_profile(
    base_model: str | None = None, *, profile: OpenPIClientProfile | str | None = None
) -> OpenPIClientProfile:
    """Resolve a supported model identity, rejecting contradictory selection."""
    selected = (
        MODEL_PROFILES.get(base_model, LEGACY_L_LORA_PROFILE)
        if profile is None
        else PROFILE_IDS[profile] if isinstance(profile, str) else profile
    )
    if base_model is not None and base_model != selected.base_model:
        raise ValueError(
            f"base_model {base_model!r} does not match profile {selected.profile_id!r} "
            f"({selected.base_model!r})"
        )
    if selected.base_model not in MODEL_PROFILES:
        raise ValueError(f"unsupported OpenPI client profile: {selected!r}")
    return selected
