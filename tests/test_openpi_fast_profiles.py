from mint_server.backend.model_registry import MODEL_CONFIGS, list_supported_models


OPENPI_FAST_MODEL = "openpi/pi0-fast-libero-low-mem-finetune"


def test_openpi_fast_profile_is_registered_with_vla_metadata() -> None:
    config = MODEL_CONFIGS[OPENPI_FAST_MODEL]

    assert config.policy_family == "ar_action_tokens"
    assert config.inference_modality == "actions"
    assert config.training_backend == "openpi_fast"
    assert config.camera_layout == ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    assert config.action_dim == 7
    assert config.action_horizon == 10
    assert config.max_model_len == 180


def test_openpi_fast_profile_is_in_default_supported_models(monkeypatch) -> None:
    monkeypatch.delenv("MINT_SUPPORTED_MODELS", raising=False)
    monkeypatch.delenv("MINT_SUPPORTED_MODELS", raising=False)

    assert OPENPI_FAST_MODEL in list_supported_models()
