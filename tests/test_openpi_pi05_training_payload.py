import pytest

import mint_server.backend.openpi.openpi_pi05_training as openpi_pi05_training
from mint_server.backend.core.model_registry import ModelConfig
from mint_server.models.types import Datum, EncodedTextChunk, ImageChunk, ModelInput, TensorData


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


def _make_rl_datum(*, chain_action_dim: int = 7) -> Datum:
    datum = _make_datum()
    chains = [float(i) / 100.0 for i in range(2 * 10 * chain_action_dim)]
    datum.loss_fn_inputs.update(
        {
            "chains": TensorData(data=chains, shape=[2, 10, chain_action_dim], dtype="float32"),
            "denoise_inds": TensorData(data=[0.0], shape=[1], dtype="int64"),
            "logprobs": TensorData(data=[-0.1] * (10 * 7), shape=[10, 7], dtype="float32"),
            "advantages": TensorData(data=[1.0] * (10 * 7), shape=[10, 7], dtype="float32"),
        }
    )
    return datum


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


def test_build_openpi_pi05_rl_runtime_payload_pads_chains_and_keeps_advantages() -> None:
    payload = openpi_pi05_training.build_openpi_pi05_rl_runtime_payload(
        datum=_make_rl_datum(),
        model_config=_pi05_config(),
    )

    assert len(payload["chains"]) == 2
    assert len(payload["chains"][0]) == 10
    assert all(len(row) == 32 for row in payload["chains"][0])
    assert payload["chains"][0][0][:7] == pytest.approx([i / 100.0 for i in range(7)])
    assert payload["chains"][0][0][7:] == [0.0] * 25
    assert payload["source_action_dim"] == 7
    assert payload["denoise_inds"] == [0]
    assert payload["old_logprobs"] == [-0.1] * 70
    assert payload["advantages"] == [1.0] * 70


def test_build_openpi_pi05_rl_runtime_payload_accepts_full_dim_chains_with_env_dim_logprobs() -> None:
    payload = openpi_pi05_training.build_openpi_pi05_rl_runtime_payload(
        datum=_make_rl_datum(chain_action_dim=32),
        model_config=_pi05_config(),
    )

    assert len(payload["chains"][0][0]) == 32
    assert payload["source_action_dim"] == 7
    assert payload["old_logprobs"] == [-0.1] * 70


def test_build_openpi_pi05_rl_runtime_payload_rejects_logprob_shape_mismatch() -> None:
    datum = _make_rl_datum()
    datum.loss_fn_inputs["logprobs"] = TensorData(data=[-0.1] * 10, shape=[10], dtype="float32")

    with pytest.raises(ValueError, match="logprobs shape"):
        openpi_pi05_training.build_openpi_pi05_rl_runtime_payload(
            datum=datum,
            model_config=_pi05_config(),
        )
