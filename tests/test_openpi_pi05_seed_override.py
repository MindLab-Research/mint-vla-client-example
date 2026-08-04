from __future__ import annotations

import pytest

from mint_server.backend.openpi.openpi_pi05_worker import OpenPIPi05RuntimeInitOverrides


def test_runtime_override_parses_training_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINT_OPENPI_PI05_SEED", "44")
    assert OpenPIPi05RuntimeInitOverrides.from_env().seed == 44
    monkeypatch.setenv("MINT_OPENPI_PI05_SEED", "-1")
    with pytest.raises(ValueError, match="non-negative integer"):
        OpenPIPi05RuntimeInitOverrides.from_env()
    monkeypatch.setenv("MINT_OPENPI_PI05_SEED", "invalid")
    with pytest.raises(ValueError):
        OpenPIPi05RuntimeInitOverrides.from_env()
