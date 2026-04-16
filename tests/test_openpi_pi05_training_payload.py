import pytest

import tinker_server.backend.openpi_pi05_training as openpi_pi05_training
from tinker_server.backend.model_registry import ModelConfig
from tinker_server.models.types import Datum, EncodedTextChunk, ImageChunk, ModelInput, TensorData


def _pi05_config() -> ModelConfig:
    return ModelConfig(
        num_parameters=3.0,
        is_moe=False,
        inference_tp=1,
        inference_dp=1,
        train_tp=1,
        train_ep=1,
        max_model_len=200,
        policy_family="flow_action",
        inference_modality="actions",
        training_backend="openpi_pi05",
        camera_layout=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
        action_dim=32,
        action_horizon=10,
    )


def _make_datum(*, num_images: int = 3, action_horizon: int = 10) -> Datum:
    image_chunks = [
        ImageChunk(data=f"img-{i}".encode("utf-8"), format="png", expected_tokens=256)
        for i in range(num_images)
    ]
    actions = [float(i) for i in range(action_horizon * 7)]
    return Datum(
        model_input=ModelInput(
            chunks=[
                *image_chunks,
                EncodedTextChunk(tokens=[2, 314, 271, 99]),
            ]
        ),
        loss_fn_inputs={
            "state": TensorData(data=[0.5] * 8, shape=[8], dtype="float32"),
            "actions": TensorData(data=actions, shape=[action_horizon, 7], dtype="float32"),
        },
    )


def test_build_openpi_pi05_sft_runtime_payload_pads_state_and_actions() -> None:
    payload = openpi_pi05_training.build_openpi_pi05_sft_runtime_payload(
        datum=_make_datum(),
        model_config=_pi05_config(),
    )

    assert tuple(payload["image_bytes"].keys()) == (
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    )
    assert payload["image_mask"] == {
        "base_0_rgb": True,
        "left_wrist_0_rgb": True,
        "right_wrist_0_rgb": True,
    }
    assert payload["tokenized_prompt"] == [2, 314, 271, 99]
    assert payload["tokenized_prompt_mask"] == [True, True, True, True]
    assert len(payload["state"]) == 32
    assert payload["state"][:8] == [0.5] * 8
    assert payload["state"][8:] == [0.0] * 24
    assert len(payload["actions"]) == 10
    assert all(len(step) == 32 for step in payload["actions"])
    assert payload["actions"][0][:7] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert payload["actions"][0][7:] == [0.0] * 25


def test_build_openpi_pi05_sft_runtime_payload_rejects_action_horizon_mismatch() -> None:
    with pytest.raises(ValueError, match="action_horizon"):
        openpi_pi05_training.build_openpi_pi05_sft_runtime_payload(
            datum=_make_datum(action_horizon=9),
            model_config=_pi05_config(),
        )


def test_build_openpi_pi05_sft_runtime_payload_rejects_camera_count_mismatch() -> None:
    with pytest.raises(ValueError, match="image chunks"):
        openpi_pi05_training.build_openpi_pi05_sft_runtime_payload(
            datum=_make_datum(num_images=2),
            model_config=_pi05_config(),
        )


def test_build_openpi_pi05_action_observation_payload_pads_state_and_keeps_prompt_tokens() -> None:
    datum = _make_datum()
    payload = openpi_pi05_training.build_openpi_pi05_action_observation_payload(
        observation=datum.model_input,
        extra_inputs={"state": TensorData(data=[0.25] * 8, shape=[8], dtype="float32")},
        model_config=_pi05_config(),
    )

    assert tuple(payload["image_bytes"].keys()) == (
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    )
    assert payload["tokenized_prompt"] == [2, 314, 271, 99]
    assert payload["tokenized_prompt_mask"] == [True, True, True, True]
    assert len(payload["state"]) == 32
    assert payload["state"][:8] == [0.25] * 8
    assert payload["state"][8:] == [0.0] * 24
