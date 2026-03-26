from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dev_volcano_env_does_not_pin_tinker_checkout_paths():
    text = (REPO_ROOT / "configs" / "dev_volcano.env.sh").read_text()

    assert "export PFS_TINKER_PATH=" not in text
    assert "export PYTHONPATH=" not in text


def test_start_mint_dev_server_uses_current_repo_for_api_and_worker_paths():
    text = (REPO_ROOT / "scripts" / "start_mint_dev_server.sh").read_text()

    assert 'export PFS_TINKER_PATH="$repo_root"' in text
    assert '$repo_root${PFS_HF_MODULES_PATH:+:$PFS_HF_MODULES_PATH}' in text
