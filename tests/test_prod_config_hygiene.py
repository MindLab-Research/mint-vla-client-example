from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SECRETS_ENV = REPO_ROOT / ".secrets.env"
PROD_ENV = REPO_ROOT / "configs" / "prod_volcano.env.sh"


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
    assert SECRETS_ENV.exists()
    names = _exported_names(SECRETS_ENV)
    assert names == [
        "TINKER_API_KEY",
        "TINKER_TOKEN_SECRET_KEY",
        "MINT_API_KEY",
        "CRS_OAI_KEY",
        "MINT_APMPLUS_APP_KEY",
    ]


def test_prod_runtime_env_contains_non_secret_authoritative_knobs() -> None:
    assert PROD_ENV.exists()
    text = PROD_ENV.read_text(encoding="utf-8")
    assert "export MINT_MODEL_CONFIG_OVERRIDES_JSON=" in text
    assert "export MINT_MOE_LORA_SPARSE_EXPERT_EXPORT=1" in text
    assert "export MINT_MODEL_NODE_IPS_JSON=" in text
