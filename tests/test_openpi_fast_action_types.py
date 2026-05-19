from __future__ import annotations

from mint_server.models.types import (
    ActRequest,
    ActResponse,
    CreateActionSessionRequest,
    CreateActionSessionResponse,
    EncodedTextChunk,
    ImageChunk,
    ModelInput,
    TensorData,
)


def test_action_session_types_roundtrip_minimal_payload() -> None:
    request = CreateActionSessionRequest(
        session_id="session-1",
        base_model="openpi/pi0-fast-libero-low-mem-finetune",
        model_path="mint://model-1/weights/export-1",
    )

    assert request.session_id == "session-1"
    assert request.base_model == "openpi/pi0-fast-libero-low-mem-finetune"
    assert request.model_path == "mint://model-1/weights/export-1"

    response = CreateActionSessionResponse(action_session_id="action-session-1")
    assert response.action_session_id == "action-session-1"


def test_act_request_and_response_use_observation_and_actions() -> None:
    request = ActRequest(
        action_session_id="action-session-1",
        observation=ModelInput(
            chunks=[
                ImageChunk(data=b"img", format="png", expected_tokens=256),
                EncodedTextChunk(tokens=[1, 2, 3]),
            ]
        ),
        extra_inputs={
            "state": TensorData(data=[0.1] * 8, shape=[8], dtype="float32"),
        },
    )

    assert request.action_session_id == "action-session-1"
    assert request.observation.length == 259
    assert request.extra_inputs["state"].shape == [8]

    response = ActResponse(
        actions=TensorData(
            data=[0.0] * (4 * 7),
            shape=[4, 7],
            dtype="float32",
        ),
        policy_timing={"infer_ms": 12.5},
    )

    assert response.actions.shape == [4, 7]
    assert response.policy_timing == {"infer_ms": 12.5}
