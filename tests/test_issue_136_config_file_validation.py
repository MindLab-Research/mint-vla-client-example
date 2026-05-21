import pytest

from mint_server.config import ServerConfig
from mint_server.config_file import load_mint_config_file


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
                'pfs_runtime_env_root = "/vePFS/runtime/mint-py31213"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_mint_config_file(p)
    assert cfg.server.max_loras == 13
    assert cfg.server.vllm_attention_backend == "FLASH_ATTN"
    assert cfg.sampling.max_inflight_sample_tasks == 7
    assert cfg.paths.pfs_runtime_env_root == "/vePFS/runtime/mint-py31213"


def test_server_config_vllm_attention_backend_prefers_env_over_file(tmp_path):
    p = tmp_path / "ok.toml"
    p.write_text("[server]\nvllm_attention_backend = 'TRITON_ATTN'\n", encoding="utf-8")
    file_cfg = load_mint_config_file(p)

    cfg = ServerConfig.from_sources(
        environ={"MINT_VLLM_ATTENTION_BACKEND": "FLASH_ATTN"},
        config_path=None,
        config_file=file_cfg,
    )

    assert cfg.vllm_attention_backend == "FLASH_ATTN"


def test_config_file_unknown_key_fails_fast(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text("unknown = 1\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_mint_config_file(p)
    assert "Config validation failed" in str(exc.value)


def test_config_file_type_mismatch_fails_fast(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text("[server]\nmax_loras = 'x'\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_mint_config_file(p)
    assert "Config validation failed" in str(exc.value)


def test_config_file_legacy_runtime_path_keys_fail_fast(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text(
        "\n".join(
            [
                "[paths]",
                'pfs_runtime_env_root = "/vePFS/runtime/mint-py31213"',
                'pfs_verl_path = "/vePFS/verl"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as exc:
        load_mint_config_file(p)
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
    cfg = load_mint_config_file(p)
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
                "retrieve_future_wait_timeout_s = 20",
                "task_pending_ttl_s = 100",
                "task_result_ttl_s = 200",
                "task_tombstone_ttl_s = 300",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_mint_config_file(p)
    assert cfg.future.retrieve_future_hot_ttl_s == 30
    assert cfg.future.retrieve_future_grace_s == 45
    assert cfg.future.retrieve_future_min_poll_s == 2.5
    assert cfg.future.retrieve_future_wait_timeout_s == 20
    assert cfg.future.task_pending_ttl_s == 100
    assert cfg.future.task_result_ttl_s == 200
    assert cfg.future.task_tombstone_ttl_s == 300


def test_server_config_retrieve_future_settings_read_from_file(tmp_path):
    p = tmp_path / "ok.toml"
    p.write_text(
        "\n".join(
            [
                "[future]",
                "retrieve_future_hot_ttl_s = 30",
                "retrieve_future_grace_s = 45",
                "retrieve_future_min_poll_s = 2.5",
                "retrieve_future_wait_timeout_s = 20",
                "task_pending_ttl_s = 100",
                "task_result_ttl_s = 200",
                "task_tombstone_ttl_s = 300",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    file_cfg = load_mint_config_file(p)

    cfg = ServerConfig.from_sources(
        environ={},
        config_path=str(p),
        config_file=file_cfg,
    )

    assert cfg.retrieve_future_hot_ttl_s == 30.0
    assert cfg.retrieve_future_grace_s == 45.0
    assert cfg.retrieve_future_min_poll_s == 2.5
    assert cfg.retrieve_future_wait_timeout_s == 20.0
    assert cfg.task_pending_ttl_s == 100.0
    assert cfg.task_result_ttl_s == 200.0
    assert cfg.task_tombstone_ttl_s == 300.0


def test_server_config_task_future_ttl_defaults():
    cfg = ServerConfig.from_sources(environ={}, config_path=None, config_file=None)

    assert cfg.retrieve_future_hot_ttl_s == 300.0
    assert cfg.retrieve_future_grace_s == 600.0
    assert cfg.retrieve_future_min_poll_s == 1.0
    assert cfg.retrieve_future_wait_timeout_s == 20.0
    assert cfg.task_pending_ttl_s == 86400.0
    assert cfg.task_result_ttl_s == 86400.0
    assert cfg.task_tombstone_ttl_s == 604800.0


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
    file_cfg = load_mint_config_file(p)

    cfg = ServerConfig.from_sources(
        environ={},
        config_path=str(p),
        config_file=file_cfg,
    )

    assert cfg.task_state_store_actor_name == "mint_task_state_store_test"
    assert cfg.task_state_store_db_path == "/tmp/task-state.sqlite3"
    assert cfg.task_state_store_owner_ttl_s == 45.0
    assert cfg.task_state_store_owner_renew_s == 15.0


def test_config_file_supervisor_state_settings_load(tmp_path):
    p = tmp_path / "ok.toml"
    p.write_text(
        "\n".join(
            [
                "[supervisor_state]",
                "backend = 'sqlite'",
                "db_path = '/tmp/supervisor-state.sqlite3'",
                "owner_ttl_s = 45",
                "event_limit = 17",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    file_cfg = load_mint_config_file(p)

    cfg = ServerConfig.from_sources(
        environ={},
        config_path=str(p),
        config_file=file_cfg,
    )

    assert cfg.supervisor_state_backend == "sqlite"
    assert cfg.supervisor_state_db_path == "/tmp/supervisor-state.sqlite3"
    assert cfg.supervisor_state_owner_ttl_s == 45.0
    assert cfg.supervisor_state_event_limit == 17


def test_server_config_supervisor_state_defaults_and_env_override():
    dev = ServerConfig.from_sources(environ={}, config_path=None, config_file=None)
    assert dev.supervisor_state_backend == "memory"
    assert dev.supervisor_state_db_path == "/vePFS-Mindverse/share/mint/dev/runtime/supervisor_state.sqlite3"

    prod = ServerConfig.from_sources(
        environ={"MINT_DEPLOYMENT_ENV": "prod"},
        config_path=None,
        config_file=None,
    )
    assert prod.supervisor_state_db_path == "/vePFS-Mindverse/share/mint/prod/runtime/supervisor_state.sqlite3"

    override = ServerConfig.from_sources(
        environ={
            "MINT_SUPERVISOR_STATE_BACKEND": "sqlite",
            "MINT_SUPERVISOR_STATE_DB_PATH": "/tmp/override.sqlite3",
            "MINT_SUPERVISOR_STATE_OWNER_TTL_S": "9",
            "MINT_SUPERVISOR_STATE_EVENT_LIMIT": "11",
        },
        config_path=None,
        config_file=None,
    )
    assert override.supervisor_state_backend == "sqlite"
    assert override.supervisor_state_db_path == "/tmp/override.sqlite3"
    assert override.supervisor_state_owner_ttl_s == 9.0
    assert override.supervisor_state_event_limit == 11


def test_server_config_task_state_store_defaults_follow_auth_mode():
    dev = ServerConfig.from_sources(environ={}, config_path=None, config_file=None)
    assert dev.task_state_store_db_path == "/vePFS-Mindverse/share/mint/dev/data/task-state/task_state.sqlite3"

    prod = ServerConfig.from_sources(
        environ={"MINT_INTERNAL_API_TOKEN": "secret"},
        config_path=None,
        config_file=None,
    )
    assert prod.task_state_store_db_path == "/vePFS-Mindverse/share/mint/prod/data/task-state/task_state.sqlite3"


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
    file_cfg = load_mint_config_file(p)

    cfg = ServerConfig.from_sources(
        environ={
            "MINT_RETRIEVE_FUTURE_HOT_TTL_S": "31",
            "MINT_RETRIEVE_FUTURE_WAIT_TIMEOUT_S": "21",
            "MINT_TASK_PENDING_TTL_S": "101",
            "MINT_TASK_RESULT_TTL_S": "201",
            "MINT_TASK_TOMBSTONE_TTL_S": "301",
        },
        config_path=None,
        config_file=file_cfg,
    )

    assert cfg.retrieve_future_hot_ttl_s == 31.0
    assert cfg.retrieve_future_wait_timeout_s == 21.0
    assert cfg.task_pending_ttl_s == 101.0
    assert cfg.task_result_ttl_s == 201.0
    assert cfg.task_tombstone_ttl_s == 301.0


def test_server_config_fails_fast_for_non_postgres_usage_backend():
    with pytest.raises(ValueError, match="Unsupported usage backend 'sqlite'"):
        ServerConfig.from_sources(
            environ={"MINT_USAGE_BACKEND": "sqlite"},
            config_path=None,
            config_file=None,
        )


def test_server_config_fails_fast_for_non_postgres_usage_backend_from_file(tmp_path):
    p = tmp_path / "bad-usage-backend.toml"
    p.write_text("[server]\nusage_backend = 'sqlite'\n", encoding="utf-8")
    file_cfg = load_mint_config_file(p)

    with pytest.raises(ValueError, match="Unsupported usage backend 'sqlite'"):
        ServerConfig.from_sources(
            environ={},
            config_path=str(p),
            config_file=file_cfg,
        )


def test_server_config_accepts_disabled_usage_backend():
    cfg = ServerConfig.from_sources(
        environ={"MINT_USAGE_BACKEND": "disabled"},
        config_path=None,
        config_file=None,
    )

    assert cfg.usage_backend == "disabled"
