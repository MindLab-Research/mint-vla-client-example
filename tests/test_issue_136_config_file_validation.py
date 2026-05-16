import pytest

from tinker_server.config import ServerConfig
from tinker_server.config_file import load_tinker_config_file


def test_config_file_load_ok(tmp_path):
    p = tmp_path / "ok.toml"
    p.write_text(
        "\n".join(
            [
                "[server]",
                "max_loras = 13",
                "max_cpu_loras = 19",
                "max_lora_rank = 23",
                "vllm_attention_backend = 'FLASH_ATTN'",
                "",
                "[sampling]",
                "max_inflight_sample_tasks = 7",
                "max_concurrent_samples_per_request = 3",
                "",
                "[paths]",
                'pfs_runtime_env_root = "/vePFS/runtime/tinker-py31213"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_tinker_config_file(p)
    assert cfg.server.max_loras == 13
    assert cfg.server.vllm_attention_backend == "FLASH_ATTN"
    assert cfg.sampling.max_inflight_sample_tasks == 7
    assert cfg.paths.pfs_runtime_env_root == "/vePFS/runtime/tinker-py31213"


def test_server_config_vllm_attention_backend_prefers_env_over_file(tmp_path):
    p = tmp_path / "ok.toml"
    p.write_text("[server]\nvllm_attention_backend = 'TRITON_ATTN'\n", encoding="utf-8")
    file_cfg = load_tinker_config_file(p)

    cfg = ServerConfig.from_sources(
        environ={"TINKER_VLLM_ATTENTION_BACKEND": "FLASH_ATTN"},
        config_path=None,
        config_file=file_cfg,
    )

    assert cfg.vllm_attention_backend == "FLASH_ATTN"


def test_config_file_unknown_key_fails_fast(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text("unknown = 1\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_tinker_config_file(p)
    assert "Config validation failed" in str(exc.value)


def test_config_file_type_mismatch_fails_fast(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text("[server]\nmax_loras = 'x'\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_tinker_config_file(p)
    assert "Config validation failed" in str(exc.value)


def test_config_file_legacy_runtime_path_keys_fail_fast(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text(
        "\n".join(
            [
                "[paths]",
                'pfs_runtime_env_root = "/vePFS/runtime/tinker-py31213"',
                'pfs_verl_path = "/vePFS/verl"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as exc:
        load_tinker_config_file(p)
    assert "Config validation failed" in str(exc.value)


def test_config_file_sampling_window_loads(tmp_path):
    p = tmp_path / "ok.toml"
    p.write_text(
        "\n".join(
            [
                "[sampling]",
                "sample_coalesce = true",
                "sample_coalesce_window_ms = 12.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_tinker_config_file(p)
    assert cfg.sampling.sample_coalesce is True
    assert cfg.sampling.sample_coalesce_window_ms == 12.5


def test_config_file_retrieve_future_settings_load(tmp_path):
    p = tmp_path / "ok.toml"
    p.write_text(
        "\n".join(
            [
                "[future]",
                "retrieve_future_hot_ttl_s = 30",
                "retrieve_future_grace_s = 45",
                "retrieve_future_min_poll_s = 2.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_tinker_config_file(p)
    assert cfg.future.retrieve_future_hot_ttl_s == 30
    assert cfg.future.retrieve_future_grace_s == 45
    assert cfg.future.retrieve_future_min_poll_s == 2.5


def test_server_config_retrieve_future_settings_read_from_file(tmp_path):
    p = tmp_path / "ok.toml"
    p.write_text(
        "\n".join(
            [
                "[future]",
                "retrieve_future_hot_ttl_s = 30",
                "retrieve_future_grace_s = 45",
                "retrieve_future_min_poll_s = 2.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    file_cfg = load_tinker_config_file(p)

    cfg = ServerConfig.from_sources(
        environ={},
        config_path=str(p),
        config_file=file_cfg,
    )

    assert cfg.retrieve_future_hot_ttl_s == 30.0
    assert cfg.retrieve_future_grace_s == 45.0
    assert cfg.retrieve_future_min_poll_s == 2.5


def test_config_file_task_state_store_settings_load(tmp_path):
    p = tmp_path / "ok.toml"
    p.write_text(
        "\n".join(
            [
                "[task_state_store]",
                "actor_name = 'mint_task_state_store_test'",
                "db_path = '/tmp/task-state.sqlite3'",
                "owner_ttl_s = 45",
                "owner_renew_s = 15",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    file_cfg = load_tinker_config_file(p)

    cfg = ServerConfig.from_sources(
        environ={},
        config_path=str(p),
        config_file=file_cfg,
    )

    assert cfg.task_state_store_actor_name == "mint_task_state_store_test"
    assert cfg.task_state_store_db_path == "/tmp/task-state.sqlite3"
    assert cfg.task_state_store_owner_ttl_s == 45.0
    assert cfg.task_state_store_owner_renew_s == 15.0


def test_server_config_task_state_store_defaults_follow_auth_mode():
    dev = ServerConfig.from_sources(environ={}, config_path=None, config_file=None)
    assert dev.task_state_store_db_path == "/vePFS-Mindverse/share/mint-prod-dev/task-state/task_state.sqlite3"

    prod = ServerConfig.from_sources(
        environ={"TINKER_API_KEY": "secret"},
        config_path=None,
        config_file=None,
    )
    assert prod.task_state_store_db_path == "/vePFS-Mindverse/share/mint-prod-data/task-state/task_state.sqlite3"


def test_server_config_retrieve_future_env_overrides_file_independently(tmp_path):
    p = tmp_path / "ok.toml"
    p.write_text(
        "\n".join(
            [
                "[future]",
                "retrieve_future_hot_ttl_s = 30",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    file_cfg = load_tinker_config_file(p)

    cfg = ServerConfig.from_sources(
        environ={
            "MINT_RETRIEVE_FUTURE_HOT_TTL_S": "31",
        },
        config_path=None,
        config_file=file_cfg,
    )

    assert cfg.retrieve_future_hot_ttl_s == 31.0


def test_server_config_reads_usage_log_dir_from_env():
    cfg = ServerConfig.from_sources(
        environ={"TINKER_USAGE_LOG_DIR": "/vePFS/shared/billing"},
        config_path=None,
        config_file=None,
    )

    assert cfg.usage_log_dir == "/vePFS/shared/billing"


def test_server_config_reads_usage_log_dir_from_file(tmp_path):
    p = tmp_path / "usage.toml"
    p.write_text("[server]\nusage_log_dir = '/vePFS/shared/from-file'\n", encoding="utf-8")
    file_cfg = load_tinker_config_file(p)

    cfg = ServerConfig.from_sources(
        environ={},
        config_path=str(p),
        config_file=file_cfg,
    )

    assert cfg.usage_log_dir == "/vePFS/shared/from-file"


def test_server_config_fails_fast_for_non_postgres_usage_backend():
    with pytest.raises(ValueError, match="Unsupported usage backend 'sqlite'"):
        ServerConfig.from_sources(
            environ={"TINKER_USAGE_BACKEND": "sqlite"},
            config_path=None,
            config_file=None,
        )


def test_server_config_fails_fast_for_non_postgres_usage_backend_from_file(tmp_path):
    p = tmp_path / "bad-usage-backend.toml"
    p.write_text("[server]\nusage_backend = 'sqlite'\n", encoding="utf-8")
    file_cfg = load_tinker_config_file(p)

    with pytest.raises(ValueError, match="Unsupported usage backend 'sqlite'"):
        ServerConfig.from_sources(
            environ={},
            config_path=str(p),
            config_file=file_cfg,
        )


def test_server_config_accepts_disabled_usage_backend():
    cfg = ServerConfig.from_sources(
        environ={"TINKER_USAGE_BACKEND": "disabled"},
        config_path=None,
        config_file=None,
    )

    assert cfg.usage_backend == "disabled"
