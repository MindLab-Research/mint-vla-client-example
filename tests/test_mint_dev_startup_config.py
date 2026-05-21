from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dev_volcano_env_sources_external_config() -> None:
    text = (REPO_ROOT / "configs" / "dev_volcano.env.sh").read_text()

    assert 'mint_dev_config_env="${MINT_DEV_CONFIG_ENV:-/vePFS-Mindverse/share/mint/dev/config/common.env}"' in text
    assert '. "${mint_dev_config_env}"' in text


def test_start_dev_server_script_sources_env_and_uses_runtime_python() -> None:
    text = (REPO_ROOT / "scripts" / "start_dev_server.sh").read_text()

    assert 'dev_config_env="${MINT_DEV_CONFIG_ENV:-/vePFS-Mindverse/share/mint/dev/config/common.env}"' in text
    assert '. "${dev_config_env}"' in text
    assert 'dev_secrets_env="${MINT_DEV_SECRETS_ENV:-/vePFS-Mindverse/share/mint/dev/config/secrets.env}"' in text
    assert 'export MINT_RUNTIME_CHECKPOINT_DIR="${TINKER_RUNTIME_CHECKPOINT_DIR:-/vePFS-Mindverse/share/mint/dev/data/runtime-checkpoints}"' in text
    assert 'api_tmp_root="${MINT_TMP_ROOT}/api/${USER:-unknown}"' in text
    assert 'api_tmp_link="/tmp/mda"' in text
    assert 'export TMPDIR="${api_tmp_link}/t"' in text
    assert 'export XDG_CACHE_HOME="${api_tmp_link}/c"' in text
    assert 'mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${MINT_RUNTIME_CHECKPOINT_DIR}"' in text
    assert 'ray_head_ip_path="${MINT_RAY_HEAD_ADDRESS_PATH:-/vePFS-Mindverse/share/mint/dev/mint-server/ray_head_ip.txt}"' in text
    assert 'export RAY_ADDRESS="ray://${ray_head_ip}:10001"' in text
    assert 'export MINT_RAY_CLIENT_ADDRESS="${RAY_ADDRESS}"' in text
    assert 'exec "${PFS_RUNTIME_ENV_ROOT}/host-venv/bin/python" scripts/run_server.py' in text


def test_prod_volcano_env_sources_external_config() -> None:
    text = (REPO_ROOT / "configs" / "prod_volcano.env.sh").read_text()

    assert 'mint_prod_config_env="${MINT_PROD_CONFIG_ENV:-/vePFS-Mindverse/share/mint/prod/config/prod.env}"' in text
    assert '. "${mint_prod_config_env}"' in text


def test_start_prod_server_script_uses_tmp_root_shortlink() -> None:
    text = (REPO_ROOT / "scripts" / "start_prod_server.sh").read_text()

    assert 'prod_config_env="${MINT_PROD_CONFIG_ENV:-/vePFS-Mindverse/share/mint/prod/config/prod.env}"' in text
    assert '. "${prod_config_env}"' in text
    assert 'prod_secrets_env="${MINT_PROD_SECRETS_ENV:-/vePFS-Mindverse/share/mint/prod/config/secrets.env}"' in text
    assert '. "${prod_secrets_env}"' in text
    assert 'export MINT_RUNTIME_CHECKPOINT_DIR="${TINKER_RUNTIME_CHECKPOINT_DIR:-/vePFS-Mindverse/share/mint/prod/data/runtime-checkpoints}"' in text
    assert 'if [ -n "${MINT_GATEWAY_GLM51_BASE_URL:-}" ]; then' in text
    assert 'missing MINT_API_KEY for GLM5.1 static gateway auth' in text
    assert 'model_to_upstream[model] = alias' in text
    assert 'upstreams[alias] = {' in text
    assert 'api_tmp_root="${MINT_TMP_ROOT}/api/${USER:-unknown}"' in text
    assert 'api_tmp_link="/tmp/mpa"' in text
    assert 'export TMPDIR="${api_tmp_link}/t"' in text
    assert 'export XDG_CACHE_HOME="${api_tmp_link}/c"' in text
    assert 'mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${MINT_RUNTIME_CHECKPOINT_DIR}"' in text
    assert 'exec "${PFS_RUNTIME_ENV_ROOT}/host-venv/bin/python" scripts/run_server.py' in text
