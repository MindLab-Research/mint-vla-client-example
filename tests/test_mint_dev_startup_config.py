from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dev_volcano_env_pins_runtime_paths_for_dev_host() -> None:
    text = (REPO_ROOT / "configs" / "dev_volcano.env.sh").read_text()

    assert "export PFS_RUNTIME_ENV_ROOT=" in text
    assert "export PFS_TINKER_PATH=" in text
    assert "export MINT_VLLM_CHILD_PYTHON_EXECUTABLE=" in text


def test_start_dev_server_script_sources_env_and_uses_runtime_python() -> None:
    text = (REPO_ROOT / "scripts" / "start_dev_server.sh").read_text()

    assert ". ./configs/dev_volcano.env.sh" in text
    assert 'exec "${PFS_RUNTIME_ENV_ROOT}/host-venv/bin/python" scripts/run_server.py' in text
