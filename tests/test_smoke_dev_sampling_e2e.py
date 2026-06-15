from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "tools" / "smoke_dev_sampling_e2e.py"


@pytest.fixture(scope="module")
def smoke_module():
    spec = importlib.util.spec_from_file_location("smoke_dev_sampling_e2e", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_first_sequence_tokens_accepts_success_payload(smoke_module):
    payload = {"sequences": [{"tokens": [151644, 872, 198]}]}

    assert smoke_module.extract_first_sequence_tokens(payload) == [151644, 872, 198]


def test_extract_first_sequence_tokens_rejects_empty_or_error_payload(smoke_module):
    with pytest.raises(RuntimeError, match="returned error"):
        smoke_module.extract_first_sequence_tokens({"error": "engine failed"})

    with pytest.raises(RuntimeError, match="missing non-empty sequences"):
        smoke_module.extract_first_sequence_tokens({"sequences": []})

    with pytest.raises(RuntimeError, match="missing non-empty tokens"):
        smoke_module.extract_first_sequence_tokens({"sequences": [{"tokens": []}]})


def test_smoke_script_does_not_read_dotenv_files():
    text = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "dotenv" not in text.lower()
    assert "load_local_env" not in text
    assert "read_text" not in text
    assert "os.environ.update" not in text
    assert "MINT_DEV_SECRETS_ENV" not in text
    assert "secrets.env" not in text


def test_smoke_script_ignores_proxy_environment_for_local_dev_server():
    text = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "trust_env=False" in text
