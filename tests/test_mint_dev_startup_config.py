from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dev_volcano_env_pins_runtime_paths_for_dev_host() -> None:
    text = (REPO_ROOT / "configs" / "dev_volcano.env.sh").read_text()

    assert "export PFS_RUNTIME_ENV_ROOT=" in text
    assert "export PFS_TINKER_PATH=" in text
    assert "export MINT_VLLM_CHILD_PYTHON_EXECUTABLE=" in text
    assert 'export MINT_TMP_ROOT="${MINT_TMP_ROOT:-/vePFS-Mindverse/share/mint-data/dev}"' in text
    assert "export TINKER_RUNTIME_CHECKPOINT_DIR=" in text


def test_start_dev_server_script_sources_env_and_uses_runtime_python() -> None:
    text = (REPO_ROOT / "scripts" / "start_dev_server.sh").read_text()

    assert ". ./configs/dev_volcano.env.sh" in text
    assert 'api_tmp_root="${MINT_TMP_ROOT}/api/${USER:-unknown}"' in text
    assert 'api_tmp_link="/tmp/mda"' in text
    assert 'export TMPDIR="${api_tmp_link}/t"' in text
    assert 'export XDG_CACHE_HOME="${api_tmp_link}/c"' in text
    assert 'mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${TINKER_RUNTIME_CHECKPOINT_DIR}"' in text
    assert 'ray_head_ip_path="${MINT_RAY_HEAD_ADDRESS_PATH:-/vePFS-Mindverse/share/code/tinker-server/ray_head_ip.txt}"' in text
    assert 'export RAY_ADDRESS="ray://${ray_head_ip}:10001"' in text
    assert 'export MINT_RAY_CLIENT_ADDRESS="${RAY_ADDRESS}"' in text
    assert 'exec "${PFS_RUNTIME_ENV_ROOT}/host-venv/bin/python" scripts/run_server.py' in text


def test_prod_volcano_env_pins_runtime_paths_for_prod_host() -> None:
    text = (REPO_ROOT / "configs" / "prod_volcano.env.sh").read_text()

    assert 'export MINT_TMP_ROOT="${MINT_TMP_ROOT:-/vePFS-Mindverse/share/mint-data/prod}"' in text
    assert "export TINKER_RUNTIME_CHECKPOINT_DIR=" in text


def test_start_prod_server_script_uses_tmp_root_shortlink() -> None:
    text = (REPO_ROOT / "scripts" / "start_prod_server.sh").read_text()

    assert '. ./configs/prod_volcano.env.sh' in text
    assert 'api_tmp_root="${MINT_TMP_ROOT}/api/${USER:-unknown}"' in text
    assert 'api_tmp_link="/tmp/mpa"' in text
    assert 'export TMPDIR="${api_tmp_link}/t"' in text
    assert 'export XDG_CACHE_HOME="${api_tmp_link}/c"' in text
    assert 'mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${TINKER_RUNTIME_CHECKPOINT_DIR}"' in text
    assert 'exec "${PFS_RUNTIME_ENV_ROOT}/host-venv/bin/python" scripts/run_server.py' in text
