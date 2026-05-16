from __future__ import annotations

from pathlib import Path
import json
import shlex

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


def _exported_value(path: Path, name: str) -> str:
    prefix = f"export {name}="
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith(prefix):
            return shlex.split(line[len("export ") :], posix=True)[0].split("=", 1)[1]
    raise AssertionError(f"{name} is not exported by {path}")


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


def test_prod_runtime_env_contains_non_secret_authoritative_knobs() -> None:
    assert PROD_ENV.exists()
    text = PROD_ENV.read_text(encoding="utf-8")
    assert "export MINT_MODEL_CONFIG_OVERRIDES_JSON=" in text
    assert "export MINT_MOE_LORA_SPARSE_EXPERT_EXPORT=1" in text
    assert "export MINT_MODEL_PLACEMENT_JSON=" in text
    assert "export MINT_VLLM_MODEL_PLACEMENT_JSON=" in text
    assert "export MINT_MEGATRON_MODEL_PLACEMENT_JSON=" in text
    assert 'export MINT_TMP_ROOT="${MINT_TMP_ROOT:-/vePFS-Mindverse/share/mint-data/prod}"' in text
    assert "export TINKER_RUNTIME_CHECKPOINT_DIR=" in text


def test_prod_openpi_models_have_model_placement() -> None:
    supported = set(_exported_value(PROD_ENV, "MINT_SUPPORTED_MODELS").split(","))
    placement = json.loads(_exported_value(PROD_ENV, "MINT_MODEL_PLACEMENT_JSON"))

    openpi_models = {model for model in supported if model.startswith("openpi/")}
    assert openpi_models
    assert openpi_models <= set(placement)
    for model in openpi_models:
        assert placement[model]["replica"] == 0
        assert placement[model]["node_ip"]
        assert placement[model]["gpu_count"] == 1


def test_dev_supported_models_have_authoritative_placement() -> None:
    supported = set(_exported_value(DEV_ENV, "MINT_SUPPORTED_MODELS").split(","))
    vllm_placement = json.loads(_exported_value(DEV_ENV, "MINT_VLLM_MODEL_PLACEMENT_JSON"))
    megatron_placement = json.loads(_exported_value(DEV_ENV, "MINT_MEGATRON_MODEL_PLACEMENT_JSON"))
    dense_placement = json.loads(_exported_value(DEV_ENV, "MINT_DENSE_MODEL_PLACEMENT_JSON"))

    assert supported <= set(vllm_placement)
    assert "Qwen/Qwen3-30B-A3B-Instruct-2507" in megatron_placement
    assert {
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen3-4B-Thinking-2507",
    } <= set(dense_placement)
