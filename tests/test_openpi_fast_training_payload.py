import pytest

from tinker_server.backend.model_registry import MODEL_CONFIGS
from tinker_server.backend.openpi_fast_training import build_openpi_fast_sft_runtime_payload
from tinker_server.models.types import Datum, EncodedTextChunk, ImageChunk, ModelInput


OPENPI_FAST_MODEL = "openpi/pi0-fast-libero-low-mem-finetune"


def _make_datum(
    *,
    weights: list[float],
    target_tokens: list[int],
    token_ar_mask: list[int],
    num_images: int = 3,
) -> Datum:
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
        loss_fn_inputs={
            "state": {"data": [0.1] * 7, "shape": [7], "dtype": "float32"},
            "target_tokens": {"data": target_tokens, "shape": [len(target_tokens)], "dtype": "int64"},
            "weights": {"data": weights, "shape": [len(weights)], "dtype": "float32"},
            "token_ar_mask": {"data": token_ar_mask, "shape": [len(token_ar_mask)], "dtype": "int32"},
        },
    )


def test_build_openpi_fast_runtime_payload_concatenates_prefix_and_suffix() -> None:
    config = MODEL_CONFIGS[OPENPI_FAST_MODEL]

    payload = build_openpi_fast_sft_runtime_payload(
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
        build_openpi_fast_sft_runtime_payload(
            datum=_make_datum(weights=[1.0, 0.5], target_tokens=[21, 22], token_ar_mask=[1, 1]),
            model_config=config,
        )


def test_build_openpi_fast_runtime_payload_rejects_camera_count_mismatch() -> None:
    config = MODEL_CONFIGS[OPENPI_FAST_MODEL]

    with pytest.raises(ValueError, match="image chunks"):
        build_openpi_fast_sft_runtime_payload(
            datum=_make_datum(weights=[1.0], target_tokens=[21], token_ar_mask=[1], num_images=2),
            model_config=config,
        )
