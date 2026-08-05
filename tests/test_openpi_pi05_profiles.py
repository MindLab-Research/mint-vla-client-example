from mint_server.backend.core.model_registry import MODEL_CONFIGS, list_supported_models
from mint_server.backend.openpi.openpi_pi05_training import get_openpi_pi05_config_name


OPENPI_PI05_MODEL = "openpi/pi05-libero-low-mem-finetune"
OPENPI_PI05_ACTION_LORA_R16_MODEL = "openpi/pi05-action-lora-r16-finetune"
OPENPI_PI05_ACTION_LORA_R16_STATE54_MODEL = "openpi/pi05-action-lora-r16-state54-finetune"
OPENPI_PI05_ACTION_LORA_R16_STATE56_MODEL = "openpi/pi05-action-lora-r16-state56-28dof-finetune"


def test_openpi_pi05_profile_is_registered_with_vla_metadata() -> None:
    config = MODEL_CONFIGS[OPENPI_PI05_MODEL]

    assert config.policy_family == "flow_action"
    assert config.inference_modality == "actions"
    assert config.training_backend == "openpi_pi05"
    assert config.camera_layout == ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    assert config.action_dim == 32
    assert config.action_horizon == 10
    assert config.max_model_len == 200


def test_openpi_pi05_action_lora_r16_registry_identity_resolves_profile() -> None:
    config = MODEL_CONFIGS[OPENPI_PI05_ACTION_LORA_R16_MODEL]

    assert config.profile == "pi05_action_lora_r16_v1"
    assert config.action_dim == 32
    assert config.action_horizon == 10


def test_openpi_pi05_state54_registry_separates_state_and_action_widths() -> None:
    config = MODEL_CONFIGS[OPENPI_PI05_ACTION_LORA_R16_STATE54_MODEL]

    assert config.profile == "pi05_action_lora_r16_state54_v1"
    assert config.state_dim == 54
    assert config.action_dim == 32
    assert config.action_horizon == 10
    assert config.max_model_len == 256


def test_openpi_pi05_state56_registry_separates_state_and_native_action_widths() -> None:
    config = MODEL_CONFIGS[OPENPI_PI05_ACTION_LORA_R16_STATE56_MODEL]
    assert config.profile == "pi05_action_lora_r16_state56_28dof_v1"
    assert config.state_dim == 56
    assert config.action_dim == 32
    assert config.action_horizon == 10
    assert config.max_model_len == 256
    assert get_openpi_pi05_config_name(OPENPI_PI05_ACTION_LORA_R16_STATE56_MODEL) == "pi05_libero_state56_native28"


def test_openpi_pi05_profile_is_not_in_default_supported_models_without_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("MINT_SUPPORTED_MODELS", raising=False)
    monkeypatch.delenv("MINT_SUPPORTED_MODELS", raising=False)

    assert OPENPI_PI05_MODEL not in list_supported_models()
