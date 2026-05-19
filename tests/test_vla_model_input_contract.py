import base64
import json

import pytest

import mint_server.models.types as types_module


def test_model_input_supports_multimodal_chunks_and_length() -> None:
    assert hasattr(types_module, "ImageChunk")
    assert hasattr(types_module, "ImageAssetPointerChunk")

    model_input = types_module.ModelInput(
        chunks=[
            types_module.EncodedTextChunk(tokens=[1, 2]),
            types_module.ImageChunk(data=b"img-bytes", format="png", expected_tokens=256),
            types_module.ImageAssetPointerChunk(
                format="jpeg",
                location="file:///tmp/example.jpg",
                expected_tokens=128,
            ),
        ]
    )

    assert model_input.length == 386


def test_model_input_to_ints_rejects_non_text_chunks() -> None:
    assert hasattr(types_module, "ImageChunk")

    model_input = types_module.ModelInput(
        chunks=[
            types_module.EncodedTextChunk(tokens=[1, 2]),
            types_module.ImageChunk(data=b"img-bytes", format="png", expected_tokens=32),
        ]
    )

    with pytest.raises(ValueError, match="EncodedTextChunk"):
        model_input.to_ints()


def test_image_chunk_json_round_trip_uses_base64() -> None:
    assert hasattr(types_module, "ImageChunk")

    chunk = types_module.ImageChunk(data=b"raw-image", format="png", expected_tokens=64)
    payload = json.loads(chunk.model_dump_json())

    assert payload["data"] == base64.b64encode(b"raw-image").decode("utf-8")

    restored = types_module.ImageChunk.model_validate_json(chunk.model_dump_json())
    assert restored.data == b"raw-image"
    assert restored.length == 64
