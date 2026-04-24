import pytest

from tinker_server.models.types import CreateModelRequest, LoRAConfig

from tinker_server.backend.openpi_fast_training import (
    OPENPI_FAST_LORA_RANK,
    validate_openpi_fast_create_request,
)


OPENPI_FAST_MODEL = "openpi/pi0-fast-libero-low-mem-finetune"


def _make_request(lora_config: LoRAConfig | None) -> CreateModelRequest:
    return CreateModelRequest(
        session_id="session-1",
        model_seq_id=1,
        base_model=OPENPI_FAST_MODEL,
        lora_config=lora_config,
    )


def test_openpi_fast_requires_lora_config() -> None:
    with pytest.raises(ValueError, match="lora_config"):
        validate_openpi_fast_create_request(_make_request(None))


def test_openpi_fast_rejects_unsupported_lora_rank() -> None:
    with pytest.raises(ValueError, match=str(OPENPI_FAST_LORA_RANK)):
        validate_openpi_fast_create_request(_make_request(LoRAConfig(rank=8)))


def test_openpi_fast_rejects_partial_lora_toggle_contract() -> None:
    with pytest.raises(ValueError, match="train_mlp"):
        validate_openpi_fast_create_request(
            _make_request(
                LoRAConfig(
                    rank=OPENPI_FAST_LORA_RANK,
                    train_mlp=False,
                )
            )
        )


def test_openpi_fast_accepts_the_upstream_lora_contract() -> None:
    request = _make_request(LoRAConfig(rank=OPENPI_FAST_LORA_RANK))

    validate_openpi_fast_create_request(request)


def test_non_openpi_fast_models_are_left_untouched() -> None:
    request = CreateModelRequest(
        session_id="session-1",
        model_seq_id=1,
        base_model="Qwen/Qwen3-0.6B",
        lora_config=LoRAConfig(rank=8, train_mlp=False),
    )

    validate_openpi_fast_create_request(request)
