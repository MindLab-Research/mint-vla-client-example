from __future__ import annotations

import unittest

from scripts.openpi_profiles import (
    ACTION_LORA_R16_MODEL,
    ACTION_LORA_R16_PROFILE,
    ACTION_LORA_R16_STATE44_MODEL,
    ACTION_LORA_R16_STATE44_PROFILE,
    ACTION_LORA_R16_STATE41_MODEL,
    ACTION_LORA_R16_STATE41_PROFILE,
    ACTION_LORA_R16_STATE45_MODEL,
    ACTION_LORA_R16_STATE45_PROFILE,
    LEGACY_L_LORA_MODEL,
    LEGACY_L_LORA_PROFILE,
    MODEL_PROFILES,
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

    def test_state44_profile_separates_state_and_action_widths(self) -> None:
        profile = resolve_profile(ACTION_LORA_R16_STATE44_MODEL)
        self.assertEqual(profile, ACTION_LORA_R16_STATE44_PROFILE)
        self.assertEqual((profile.state_dim, profile.action_dim), (44, 32))
        self.assertEqual(profile.action_horizon, 10)
        self.assertEqual(profile.max_tokens, 200)
        self.assertTrue(profile.fail_on_token_truncation)
        self.assertEqual(profile.state_contract_id, "mano_five_finger_contact_geom_rate_v2")

    def test_state45_is_the_only_maintained_mano_28dof_profile(self) -> None:
        profile = resolve_profile(ACTION_LORA_R16_STATE45_MODEL)
        self.assertEqual(profile, ACTION_LORA_R16_STATE45_PROFILE)
        self.assertEqual((profile.state_dim, profile.action_dim), (45, 32))
        self.assertEqual(profile.max_tokens, 224)
        self.assertNotIn(ACTION_LORA_R16_STATE41_MODEL, MODEL_PROFILES)
        with self.assertRaisesRegex(ValueError, "unsupported OpenPI client profile"):
            resolve_profile(ACTION_LORA_R16_STATE41_MODEL, profile=ACTION_LORA_R16_STATE41_PROFILE)

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
