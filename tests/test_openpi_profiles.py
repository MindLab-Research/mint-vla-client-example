from __future__ import annotations

import unittest

from scripts.openpi_profiles import (
    ACTION_LORA_R16_MODEL,
    ACTION_LORA_R16_PROFILE,
    ACTION_LORA_R16_STATE54_MODEL,
    ACTION_LORA_R16_STATE54_PROFILE,
    ACTION_LORA_R16_STATE56_28DOF_MODEL,
    ACTION_LORA_R16_STATE56_28DOF_PROFILE,
    LEGACY_L_LORA_MODEL,
    LEGACY_L_LORA_PROFILE,
    resolve_profile,
)
from scripts.eval.mode4_support import action_session_payload


class OpenPIProfileTests(unittest.TestCase):
    def test_profiles_preserve_legacy_and_define_discrete_action_lora_contract(self) -> None:
        self.assertFalse(resolve_profile(LEGACY_L_LORA_MODEL).discrete_state_input)
        profile = resolve_profile(ACTION_LORA_R16_MODEL)
        self.assertEqual(profile, ACTION_LORA_R16_PROFILE)
        self.assertTrue(profile.discrete_state_input)
        self.assertEqual((profile.action_dim, profile.action_horizon, profile.max_tokens), (32, 10, 200))
        self.assertEqual(
            (profile.paligemma_variant, profile.action_expert_variant),
            ("gemma_2b", "gemma_300m_lora_r16"),
        )
        self.assertEqual(
            (LEGACY_L_LORA_PROFILE.paligemma_variant, LEGACY_L_LORA_PROFILE.action_expert_variant),
            ("gemma_2b_lora", "gemma_300m"),
        )
        self.assertEqual(LEGACY_L_LORA_PROFILE.base_model, LEGACY_L_LORA_MODEL)

    def test_state54_profile_separates_observation_and_action_widths(self) -> None:
        profile = resolve_profile(ACTION_LORA_R16_STATE54_MODEL)
        self.assertEqual(profile, ACTION_LORA_R16_STATE54_PROFILE)
        self.assertEqual(profile.state_dim, 54)
        self.assertEqual(profile.action_dim, 32)
        self.assertEqual(profile.action_horizon, 10)
        self.assertEqual(profile.max_tokens, 256)
        self.assertTrue(profile.fail_on_token_truncation)

    def test_state56_native28_profile_is_additive_and_explicit(self) -> None:
        profile = resolve_profile(ACTION_LORA_R16_STATE56_28DOF_MODEL)
        self.assertEqual(profile, ACTION_LORA_R16_STATE56_28DOF_PROFILE)
        self.assertEqual((profile.state_dim, profile.action_dim, profile.action_horizon), (56, 32, 10))
        self.assertEqual(profile.max_tokens, 256)
        self.assertTrue(profile.fail_on_token_truncation)
        self.assertEqual(profile.state_contract_id, "mano_object_dynamics_state56_native28_v1")
        self.assertEqual(profile.physical_action_dim, 28)
        self.assertEqual(profile.delta_mask_segments, (3, -3, 22, -4))
        self.assertEqual(ACTION_LORA_R16_STATE54_PROFILE.delta_mask_segments, (3, -3, 20, -6))

    def test_rejects_conflicting_model_and_profile(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match profile"):
            resolve_profile(ACTION_LORA_R16_MODEL, profile=LEGACY_L_LORA_PROFILE)

    def test_action_session_payload_forwards_selected_model(self) -> None:
        self.assertEqual(
            action_session_payload(
                session_id="session", base_model=ACTION_LORA_R16_MODEL,
                model_path="mint://checkpoint", owner_id="owner",
            )["base_model"],
            ACTION_LORA_R16_MODEL,
        )


if __name__ == "__main__":
    unittest.main()
