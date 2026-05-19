from __future__ import annotations

import pytest
from fastapi import HTTPException

from mint_server.models.types import LoRAConfig
from mint_server.routes import training as training_routes


def test_issue_416_lora_rank_guard_rejects_rank_above_server_max(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training_routes.server_config, "max_lora_rank", 64, raising=False)

    with pytest.raises(HTTPException) as exc_info:
        training_routes._validate_lora_rank_or_400(LoRAConfig(rank=128))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Requested LoRA rank 128 exceeds server max_lora_rank=64"


def test_issue_416_lora_rank_guard_allows_rank_at_server_max(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training_routes.server_config, "max_lora_rank", 64, raising=False)
    training_routes._validate_lora_rank_or_400(LoRAConfig(rank=64))
