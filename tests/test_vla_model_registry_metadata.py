from tinker_server.backend.model_registry import MODEL_CONFIGS, ModelConfig


def test_text_models_default_to_text_family_metadata() -> None:
    config = MODEL_CONFIGS["Qwen/Qwen3-0.6B"]

    assert config.policy_family == "text_lm"
    assert config.inference_modality == "tokens"
    assert config.training_backend == "mint_text"
    assert config.action_dim is None
    assert config.action_horizon is None
    assert config.camera_layout == ()


def test_model_config_can_describe_autoregressive_action_models() -> None:
    config = ModelConfig(
        num_parameters=3.0,
        is_moe=False,
        inference_tp=1,
        inference_dp=1,
        max_model_len=1024,
        policy_family="ar_action_tokens",
        inference_modality="actions",
        training_backend="openpi",
        action_dim=8,
        action_horizon=10,
        camera_layout=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
    )

    assert config.policy_family == "ar_action_tokens"
    assert config.inference_modality == "actions"
    assert config.training_backend == "openpi"
    assert config.action_dim == 8
    assert config.action_horizon == 10
    assert config.camera_layout == ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def test_model_config_can_describe_flow_action_models() -> None:
    config = ModelConfig(
        num_parameters=3.0,
        is_moe=False,
        inference_tp=1,
        inference_dp=1,
        max_model_len=1024,
        policy_family="flow_action",
        inference_modality="actions",
    )

    assert config.policy_family == "flow_action"
    assert config.inference_modality == "actions"
