import pytest

from mint_server.backend.openpi.pi05_profiles import (
    PI05_ACTION_LORA_R16_V1,
    get_pi05_profile,
    validate_profile_trainable_leaves,
)


EXPECTED_LEAVES = {
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


def test_action_lora_r16_trainable_leaf_contract_accepts_exact_layout() -> None:
    validate_profile_trainable_leaves(PI05_ACTION_LORA_R16_V1, EXPECTED_LEAVES)


def test_action_lora_r16_trainable_leaf_contract_rejects_missing_projection() -> None:
    actual = {key: value for key, value in EXPECTED_LEAVES.items() if key != "action_out_proj/kernel"}

    with pytest.raises(ValueError, match="missing=.*action_out_proj/kernel"):
        validate_profile_trainable_leaves(PI05_ACTION_LORA_R16_V1, actual)


def test_action_lora_r16_trainable_leaf_contract_rejects_unexpected_context_lora() -> None:
    actual = {**EXPECTED_LEAVES, "PaliGemma/llm/layers/attn/q_einsum_0/lora_a": (18, 8, 1024, 16)}

    with pytest.raises(ValueError, match="unexpected=.*q_einsum_0/lora_a"):
        validate_profile_trainable_leaves(PI05_ACTION_LORA_R16_V1, actual)


def test_action_lora_r16_trainable_leaf_contract_rejects_wrong_rank_or_shape() -> None:
    actual = {**EXPECTED_LEAVES, "time_mlp_out/kernel": (1024, 1024, 1)}

    with pytest.raises(ValueError, match="wrong_shape=.*time_mlp_out/kernel"):
        validate_profile_trainable_leaves(PI05_ACTION_LORA_R16_V1, actual)
