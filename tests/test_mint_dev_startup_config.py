from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_DEV_SECRETS_ENV = "MINT_DEV_" + "SECRETS_ENV"
LEGACY_DEV_SECRETS_PATH = "/vePFS-Mindverse/share/mint/dev/config/" + "secrets.env"


def test_dev_volcano_legacy_config_wrapper_is_removed() -> None:
    assert not (REPO_ROOT / "configs" / "dev_volcano.env.sh").exists()


def test_start_dev_server_script_uses_minimal_launch_contract() -> None:
    text = (REPO_ROOT / "scripts" / "start_dev_server.sh").read_text()

    # MINT_CODE_ROOT is a required, explicit input: no default, refuse if unset.
    assert "if [ -z \"${MINT_CODE_ROOT:-}\" ]; then" in text
    assert "error: MINT_CODE_ROOT is required" in text
    # Namespace is user-scoped and never defaults to a shared/root name.
    assert 'export MINT_RAY_NAMESPACE="${MINT_RAY_NAMESPACE:-mint_${mint_user}}"' in text
    assert '""|mint|root|mint_root)' in text
    # Driver attaches as a Ray client; the file-path env is not exported to it.
    assert "unset MINT_RAY_HEAD_ADDRESS_PATH" in text
    assert "unset RAY_ADDRESS" in text
    assert 'export MINT_RAY_CLIENT_ADDRESS="ray://${ray_head_ip}:10001"' in text
    assert 'export MINT_RAY_GCS_ADDRESS="${ray_head_ip}:6379"' in text
    assert 'export RAY_ADDRESS="${ray_head_ip}:6379"' not in text
    # vLLM worker bootstrap must always use the wrapper from this checkout.
    assert 'vllm_worker_python="${MINT_CODE_ROOT}/scripts/vllm_worker_python.py"' in text
    assert 'export MINT_VLLM_CHILD_PYTHON_EXECUTABLE="${vllm_worker_python}"' in text
    assert "MINT_VLLM_CHILD_PYTHON_EXECUTABLE" in text
    # Optional deployment policy must not carry code root or namespace.
    assert "MINT_DEV_DEPLOYMENT_ENV" in text
    assert (
        "MINT_CODE_ROOT|MINT_RAY_NAMESPACE|TINKER_RAY_NAMESPACE|"
        "MINT_RAY_HEAD_ADDRESS_PATH|MINT_VLLM_CHILD_PYTHON_EXECUTABLE"
    ) in text
    assert LEGACY_DEV_SECRETS_ENV not in text
    assert LEGACY_DEV_SECRETS_PATH not in text
    # Runtime root and HF modules default to dev infra (not business code).
    assert 'export PFS_RUNTIME_ENV_ROOT="${PFS_RUNTIME_ENV_ROOT:-/vePFS-Mindverse/share/mint/dev/runtime}"' in text
    assert 'exec "${py}" scripts/run_server.py' in text


def test_runtime_config_has_no_dev_secrets_env_shim() -> None:
    text = (REPO_ROOT / "mint_server" / "runtime_config.py").read_text()

    assert LEGACY_DEV_SECRETS_ENV not in text
    assert "MINT_DEV_CONFIG_ENV" not in text


def test_agent_dev_skills_do_not_revive_legacy_dev_secrets() -> None:
    for relpath in (
        ".claude/skills/mint-dev/SKILL.md",
        ".claude/skills/auto-bugfix/SKILL.md",
    ):
        text = (REPO_ROOT / relpath).read_text()

        assert LEGACY_DEV_SECRETS_ENV not in text
        assert LEGACY_DEV_SECRETS_PATH not in text


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
    assert "missing MINT_RAY_GCS_ADDRESS in prod config" in text
    assert '--address="${MINT_RAY_GCS_ADDRESS}"' in text
    assert '--address="${RAY_ADDRESS}"' not in text
    assert 'export MINT_RUNTIME_CHECKPOINT_DIR="/vePFS-Mindverse/share/mint/prod/data/runtime-checkpoints"' in text
    assert 'export MINT_CODE_ROOT="$repo_root"' in text
    legacy_gateway_prefix = "MINT_GATEWAY_" + "GLM" + "51"
    legacy_model_label = "GLM" + "5.1"
    assert legacy_gateway_prefix not in text
    assert legacy_model_label not in text
    assert 'api_tmp_root="${MINT_TMP_ROOT}/api/${USER:-unknown}"' in text
    assert 'api_tmp_link="/tmp/mpa"' in text
    assert 'export TMPDIR="${api_tmp_link}/t"' in text
    assert 'export XDG_CACHE_HOME="${api_tmp_link}/c"' in text
    assert 'mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${MINT_RUNTIME_CHECKPOINT_DIR}"' in text
    assert 'exec "${PFS_RUNTIME_ENV_ROOT}/host-venv/bin/python" scripts/run_server.py' in text
