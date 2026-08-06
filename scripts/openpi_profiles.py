"""Dependency-light OpenPI client model identities and invariants.

These profiles select the server contract and the client-side tokenization mode.
They do not select or place adapters: that remains entirely server-owned.
"""
from __future__ import annotations

from dataclasses import dataclass


LEGACY_L_LORA_MODEL = "openpi/pi05-libero-low-mem-finetune"
ACTION_LORA_R16_MODEL = "openpi/pi05-action-lora-r16-finetune"
ACTION_LORA_R16_STATE44_MODEL = "openpi/pi05-action-lora-r16-state44-finetune"
ACTION_LORA_R16_STATE41_MODEL = (
    "openpi/pi05-action-lora-r16-state41-28dof-finetune"
)
ACTION_LORA_R16_STATE45_MODEL = (
    "openpi/pi05-action-lora-r16-state45-phase-28dof-finetune"
)


@dataclass(frozen=True)
class OpenPIClientProfile:
    profile_id: str
    base_model: str
    discrete_state_input: bool
    paligemma_variant: str
    action_expert_variant: str
    state_dim: int = 32
    action_dim: int = 32
    action_horizon: int = 10
    max_tokens: int = 200
    # Camera layout (mirrors mint_server model_registry.py:172) -- used to order
    # image chunks in the train_step payload. Kept here so the client can build
    # batches without importing mint_server.
    camera_layout: tuple[str, ...] = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    fail_on_token_truncation: bool = False
    state_contract_id: str | None = None
    delta_mask_segments: tuple[int, ...] = (3, -3, 20, -6)


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
    state_contract_id="mano_five_finger_contact_lift_v1",
)
ACTION_LORA_R16_STATE44_PROFILE = OpenPIClientProfile(
    profile_id="pi05_action_lora_r16_state44_v1",
    base_model=ACTION_LORA_R16_STATE44_MODEL,
    discrete_state_input=True,
    paligemma_variant="gemma_2b",
    action_expert_variant="gemma_300m_lora_r16",
    state_dim=44,
    fail_on_token_truncation=True,
    state_contract_id="mano_five_finger_contact_geom_rate_v2",
)
# Frozen compatibility descriptor for historical State41 artifacts. It is
# intentionally excluded from MODEL_PROFILES/MODEL_CHOICES: State45 is the only
# maintained MANO 28DoF model identity.
ACTION_LORA_R16_STATE41_PROFILE = OpenPIClientProfile(
    profile_id="pi05_action_lora_r16_state41_28dof_v1",
    base_model=ACTION_LORA_R16_STATE41_MODEL,
    discrete_state_input=True,
    paligemma_variant="gemma_2b",
    action_expert_variant="gemma_300m_lora_r16",
    state_dim=41,
    fail_on_token_truncation=True,
    state_contract_id="mano_state41_native_sim_28d_v1",
    delta_mask_segments=(3, -3, 22, -4),
)
ACTION_LORA_R16_STATE45_PROFILE = OpenPIClientProfile(
    profile_id="pi05_action_lora_r16_state45_phase_28dof_v1",
    base_model=ACTION_LORA_R16_STATE45_MODEL,
    discrete_state_input=True,
    paligemma_variant="gemma_2b",
    action_expert_variant="gemma_300m_lora_r16",
    state_dim=45,
    max_tokens=224,
    fail_on_token_truncation=True,
    state_contract_id="mano_state45_phase_native_sim_28d_v1",
    delta_mask_segments=(3, -3, 22, -4),
)

MODEL_PROFILES = {
    profile.base_model: profile
    for profile in (
        LEGACY_L_LORA_PROFILE,
        ACTION_LORA_R16_PROFILE,
        ACTION_LORA_R16_STATE44_PROFILE,
        ACTION_LORA_R16_STATE45_PROFILE,
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
