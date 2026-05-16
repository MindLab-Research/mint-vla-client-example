from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SECRETS_ENV = REPO_ROOT / ".secrets.env"
PROD_ENV = REPO_ROOT / "configs" / "prod_volcano.env.sh"
DEV_ENV = REPO_ROOT / "configs" / "dev_volcano.env.sh"


def _exported_names(path: Path) -> list[str]:
    names: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("export ") or "=" not in line:
            continue
        lhs = line[len("export ") :].split("=", 1)[0].strip()
        if lhs:
            names.append(lhs)
    return names


def test_secrets_env_contains_only_secret_exports() -> None:
    if not SECRETS_ENV.exists():
        pytest.skip(".secrets.env is intentionally absent in this worktree")
    names = _exported_names(SECRETS_ENV)
    assert names == [
        "TINKER_API_KEY",
        "TINKER_TOKEN_SECRET_KEY",
        "MINT_API_KEY",
        "CRS_OAI_KEY",
        "MINT_APMPLUS_APP_KEY",
    ]


def test_repo_prod_env_is_external_config_wrapper() -> None:
    assert PROD_ENV.exists()
    text = PROD_ENV.read_text(encoding="utf-8")
    assert "/share/mint/prod/config/prod.env" in text
    assert "MINT_PROD_CONFIG_ENV" in text
    assert "MINT_VLLM_MODEL_PLACEMENT_JSON=" not in text
    assert "node_ip" not in text


def test_repo_dev_env_is_external_config_wrapper() -> None:
    assert DEV_ENV.exists()
    text = DEV_ENV.read_text(encoding="utf-8")
    assert "/share/mint/dev/config/common.env" in text
    assert "MINT_DEV_CONFIG_ENV" in text
    assert "MINT_VLLM_MODEL_PLACEMENT_JSON=" not in text
    assert "node_ip" not in text
