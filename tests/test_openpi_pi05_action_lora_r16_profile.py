from mint_server.backend.openpi.pi05_profiles import (
    PI05_ACTION_LORA_R16_V1,
    get_pi05_profile,
)


def test_action_lora_r16_profile_is_hash_stable_and_complete() -> None:
    profile = PI05_ACTION_LORA_R16_V1

    assert get_pi05_profile(profile.profile_id) is profile
    assert profile.manifest_hash == profile.checkpoint_manifest()["manifest_hash"]
    assert profile.pi0_config_kwargs() == {
        "pi05": True,
        "action_dim": 32,
        "action_horizon": 10,
        "max_token_len": 200,
        "discrete_state_input": True,
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m_lora_r16",
    }
    assert profile.expected_trainable_count == 13_224_992
