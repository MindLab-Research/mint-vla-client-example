from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
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


def test_repo_prod_env_is_external_config_wrapper() -> None:
    assert PROD_ENV.exists()
    text = PROD_ENV.read_text(encoding="utf-8")
    assert "/vePFS-Mindverse/share/mint/prod/config/prod.env" in text
    assert "MINT_PROD_CONFIG_ENV" in text
    assert "MINT_VLLM_MODEL_PLACEMENT_JSON=" not in text
    assert "node_ip" not in text
