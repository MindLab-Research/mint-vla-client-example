import pytest

import mint_server.backend.openpi.openpi_fast_training as openpi_fast_training
from mint_server.backend.core.model_registry import MODEL_CONFIGS
from mint_server.models.types import Datum, EncodedTextChunk, ImageChunk, ModelInput


OPENPI_FAST_MODEL = "openpi/pi0-fast-libero-low-mem-finetune"


def _make_datum(
    *,
    weights: list[float],
    target_tokens: list[int],
    token_ar_mask: list[int],
    logprobs: list[float] | None = None,
    advantages: list[float] | None = None,
    num_images: int = 3,
) -> Datum:
    loss_fn_inputs: dict[str, object] = {
        "state": {"data": [0.1] * 7, "shape": [7], "dtype": "float32"},
        "target_tokens": {"data": target_tokens, "shape": [len(target_tokens)], "dtype": "int64"},
        "weights": {"data": weights, "shape": [len(weights)], "dtype": "float32"},
        "token_ar_mask": {"data": token_ar_mask, "shape": [len(token_ar_mask)], "dtype": "int32"},
    }
    if logprobs is not None:
        loss_fn_inputs["logprobs"] = {
            "data": logprobs,
            "shape": [len(logprobs)],
            "dtype": "float32",
        }
    if advantages is not None:
        loss_fn_inputs["advantages"] = {
            "data": advantages,
            "shape": [len(advantages)],
            "dtype": "float32",
        }
    image_chunks = [
        ImageChunk(data=f"img-{i}".encode("utf-8"), format="png", expected_tokens=256)
        for i in range(num_images)
    ]
    return Datum(
        model_input=ModelInput(
            chunks=[
                *image_chunks,
                EncodedTextChunk(tokens=[11, 12, 13]),
            ]
        ),
        loss_fn_inputs=loss_fn_inputs,
    )


def test_build_openpi_fast_runtime_payload_concatenates_prefix_and_suffix() -> None:
    config = MODEL_CONFIGS[OPENPI_FAST_MODEL]

    payload = openpi_fast_training.build_openpi_fast_sft_runtime_payload(
        datum=_make_datum(weights=[1.0, 1.0], target_tokens=[21, 22], token_ar_mask=[1, 1]),
        model_config=config,
    )

    assert tuple(payload["image_bytes"].keys()) == config.camera_layout
    assert payload["image_mask"] == {name: True for name in config.camera_layout}
    assert payload["state"] == [0.1] * 7
    assert payload["tokenized_prompt"] == [11, 12, 13, 21, 22]
    assert payload["tokenized_prompt_mask"] == [True, True, True, True, True]
    assert payload["token_ar_mask"] == [0, 0, 0, 1, 1]
    assert payload["token_loss_mask"] == [False, False, False, True, True]
    assert payload["image_bytes"]["base_0_rgb"]["format"] == "png"


def test_build_openpi_fast_runtime_payload_rejects_non_binary_weights() -> None:
    config = MODEL_CONFIGS[OPENPI_FAST_MODEL]

    with pytest.raises(ValueError, match="binary"):
        openpi_fast_training.build_openpi_fast_sft_runtime_payload(
            datum=_make_datum(weights=[1.0, 0.5], target_tokens=[21, 22], token_ar_mask=[1, 1]),
            model_config=config,
        )


def test_build_openpi_fast_runtime_payload_rejects_camera_count_mismatch() -> None:
    config = MODEL_CONFIGS[OPENPI_FAST_MODEL]

    with pytest.raises(ValueError, match="image chunks"):
        openpi_fast_training.build_openpi_fast_sft_runtime_payload(
            datum=_make_datum(weights=[1.0], target_tokens=[21], token_ar_mask=[1], num_images=2),
            model_config=config,
        )


def test_build_openpi_fast_rl_runtime_payload_keeps_prefix_and_exposes_suffix_rl_fields() -> None:
    config = MODEL_CONFIGS[OPENPI_FAST_MODEL]

    payload = openpi_fast_training.build_openpi_fast_rl_runtime_payload(
        datum=_make_datum(
            weights=[1.0, 1.0],
            target_tokens=[21, 22],
            token_ar_mask=[1, 1],
            logprobs=[-0.1, -0.2],
            advantages=[1.5, -0.5],
        ),
        model_config=config,
    )

    assert payload["tokenized_prompt"] == [11, 12, 13, 21, 22]
    assert payload["tokenized_prompt_mask"] == [True, True, True, True, True]
    assert payload["token_ar_mask"] == [0, 0, 0, 1, 1]
    assert payload["token_loss_mask"] == [False, False, False, True, True]
    assert payload["old_logprobs"] == [-0.1, -0.2]
    assert payload["advantages"] == [1.5, -0.5]


@pytest.mark.parametrize(
    ("logprobs", "advantages", "missing_key"),
    [
        (None, [1.0, -1.0], "logprobs"),
        ([-0.3, -0.4], None, "advantages"),
    ],
)
def test_build_openpi_fast_rl_runtime_payload_requires_rl_inputs(
    logprobs: list[float] | None,
    advantages: list[float] | None,
    missing_key: str,
) -> None:
    config = MODEL_CONFIGS[OPENPI_FAST_MODEL]

    with pytest.raises(ValueError, match=missing_key):
        openpi_fast_training.build_openpi_fast_rl_runtime_payload(
            datum=_make_datum(
                weights=[1.0, 1.0],
                target_tokens=[21, 22],
                token_ar_mask=[1, 1],
                logprobs=logprobs,
                advantages=advantages,
            ),
            model_config=config,
        )
