from __future__ import annotations

import base64
from typing import Any

from .model_registry import ModelConfig


OPENPI_PI05_TRAINING_BACKEND = "openpi_pi05"


def _tensor_payload(value: Any, key: str) -> tuple[list[Any], list[int]]:
    if value is None:
        raise ValueError(f"Missing loss_fn_inputs[{key!r}]")

    data = getattr(value, "data", None)
    shape = getattr(value, "shape", None)
    if data is None and isinstance(value, dict):
        data = value.get("data")
        shape = value.get("shape")

    if data is None or shape is None:
        raise ValueError(f"loss_fn_inputs[{key!r}] must provide tensor-style data and shape")
    if not isinstance(shape, list):
        raise ValueError(f"loss_fn_inputs[{key!r}] shape must be a list")
    if isinstance(data, list):
        return data, shape
    return [data], shape


def _pad(values: list[float], target_dim: int, *, key: str) -> list[float]:
    if len(values) > target_dim:
        raise ValueError(f"{key} length {len(values)} exceeds action_dim {target_dim}")
    return values + ([0.0] * (target_dim - len(values)))


def _common_input_payload(
    *,
    model_input: Any,
    state_input: Any,
    model_config: ModelConfig,
) -> dict[str, Any]:
    if model_config.training_backend != OPENPI_PI05_TRAINING_BACKEND:
        raise ValueError("OpenPI pi0.5 helpers only support OpenPI pi0.5 model configs")

    image_chunks = [chunk for chunk in model_input.chunks if chunk.type == "image"]
    text_chunks = [chunk for chunk in model_input.chunks if chunk.type == "encoded_text"]
    other_chunks = [
        chunk.type
        for chunk in model_input.chunks
        if chunk.type not in {"image", "encoded_text"}
    ]
    if other_chunks:
        raise ValueError(f"Unsupported OpenPI pi0.5 chunk types: {other_chunks}")
    if len(text_chunks) != 1:
        raise ValueError("OpenPI pi0.5 expects exactly one encoded_text prompt chunk")

    camera_layout = tuple(model_config.camera_layout)
    if len(image_chunks) != len(camera_layout):
        raise ValueError(
            f"OpenPI pi0.5 expects {len(camera_layout)} image chunks, got {len(image_chunks)}"
        )

    action_dim = int(model_config.action_dim or 0)
    if action_dim <= 0:
        raise ValueError("OpenPI pi0.5 model config must define a positive action_dim")

    state_data, state_shape = _tensor_payload(state_input, "state")
    if len(state_shape) != 1:
        raise ValueError("OpenPI pi0.5 state must be rank-1")
    state = _pad([float(value) for value in state_data], action_dim, key="state")

    image_bytes = {
        name: {
            "data": base64.b64encode(chunk.data).decode("utf-8"),
            "format": chunk.format,
        }
        for name, chunk in zip(camera_layout, image_chunks, strict=True)
    }
    prompt_tokens = [int(token) for token in text_chunks[0].tokens]
    return {
        "image_bytes": image_bytes,
        "image_mask": {name: True for name in camera_layout},
        "state": state,
        "tokenized_prompt": prompt_tokens,
        "tokenized_prompt_mask": [True] * len(prompt_tokens),
    }


def build_openpi_pi05_sft_runtime_payload(
    *,
    datum: Any,
    model_config: ModelConfig,
) -> dict[str, Any]:
    payload = _common_input_payload(
        model_input=datum.model_input,
        state_input=datum.loss_fn_inputs.get("state"),
        model_config=model_config,
    )

    action_dim = int(model_config.action_dim or 0)
    action_horizon = int(model_config.action_horizon or 0)
    if action_horizon <= 0:
        raise ValueError("OpenPI pi0.5 model config must define positive action_dim and action_horizon")

    actions_data, actions_shape = _tensor_payload(datum.loss_fn_inputs.get("actions"), "actions")
    if len(actions_shape) != 2:
        raise ValueError("OpenPI pi0.5 actions must be rank-2")
    if int(actions_shape[0]) != action_horizon:
        raise ValueError(
            f"OpenPI pi0.5 actions action_horizon mismatch: expected {action_horizon}, got {actions_shape[0]}"
        )
    source_action_dim = int(actions_shape[1])
    if source_action_dim <= 0:
        raise ValueError("OpenPI pi0.5 actions must have a positive trailing dimension")
    if len(actions_data) != action_horizon * source_action_dim:
        raise ValueError("OpenPI pi0.5 actions data length does not match shape")

    actions: list[list[float]] = []
    for row_idx in range(action_horizon):
        start = row_idx * source_action_dim
        row = [float(value) for value in actions_data[start : start + source_action_dim]]
        actions.append(_pad(row, action_dim, key="actions"))

    return {**payload, "actions": actions}


def build_openpi_pi05_action_observation_payload(
    *,
    observation: Any,
    extra_inputs: dict[str, Any],
    model_config: ModelConfig,
) -> dict[str, Any]:
    return _common_input_payload(
        model_input=observation,
        state_input=extra_inputs.get("state"),
        model_config=model_config,
    )
