import pytest

from mint_server.models.types import CreateModelRequest, LoRAConfig


OPENPI_PI05_MODEL = "openpi/pi05-libero-low-mem-finetune"


def _make_request(lora_config: LoRAConfig | None) -> CreateModelRequest:
    return CreateModelRequest(
        session_id="session-1",
        model_seq_id=1,
        base_model=OPENPI_PI05_MODEL,
        lora_config=lora_config,
    )


def _fake_get_model_config(base_model: str):
    if base_model == OPENPI_PI05_MODEL:
        return type("_Cfg", (), {"training_backend": "openpi_pi05"})()
    raise KeyError(base_model)


def test_openpi_pi05_requires_lora_config(monkeypatch) -> None:
    from mint_server.backend.openpi import openpi_pi05_training

    monkeypatch.setattr(openpi_pi05_training, "get_model_config", _fake_get_model_config)

    with pytest.raises(ValueError, match="lora_config"):
        openpi_pi05_training.validate_openpi_pi05_create_request(_make_request(None))


def test_openpi_pi05_rejects_unsupported_lora_rank(monkeypatch) -> None:
    from mint_server.backend.openpi import openpi_pi05_training

    monkeypatch.setattr(openpi_pi05_training, "get_model_config", _fake_get_model_config)

    with pytest.raises(ValueError, match=str(openpi_pi05_training.OPENPI_PI05_LORA_RANK)):
        openpi_pi05_training.validate_openpi_pi05_create_request(_make_request(LoRAConfig(rank=8)))


def test_openpi_pi05_rejects_partial_lora_toggle_contract(monkeypatch) -> None:
    from mint_server.backend.openpi import openpi_pi05_training

    monkeypatch.setattr(openpi_pi05_training, "get_model_config", _fake_get_model_config)

    with pytest.raises(ValueError, match="train_mlp"):
        openpi_pi05_training.validate_openpi_pi05_create_request(
            _make_request(
                LoRAConfig(
                    rank=openpi_pi05_training.OPENPI_PI05_LORA_RANK,
                    train_mlp=False,
                )
            )
        )


def test_openpi_pi05_accepts_the_upstream_lora_contract(monkeypatch) -> None:
    from mint_server.backend.openpi import openpi_pi05_training

    monkeypatch.setattr(openpi_pi05_training, "get_model_config", _fake_get_model_config)

    request = _make_request(LoRAConfig(rank=openpi_pi05_training.OPENPI_PI05_LORA_RANK))
    openpi_pi05_training.validate_openpi_pi05_create_request(request)


def test_non_openpi_pi05_models_are_left_untouched(monkeypatch) -> None:
    from mint_server.backend.openpi import openpi_pi05_training

    monkeypatch.setattr(openpi_pi05_training, "get_model_config", _fake_get_model_config)

    request = CreateModelRequest(
        session_id="session-1",
        model_seq_id=1,
        base_model="Qwen/Qwen3-0.6B",
        lora_config=LoRAConfig(rank=8, train_mlp=False),
    )

    openpi_pi05_training.validate_openpi_pi05_create_request(request)

